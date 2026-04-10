"""Smoke E2E para validar flujo real feed -> engine -> executor -> DB.

Uso:
    cd /root/Kalshi-crypto-agents
    source .venv/bin/activate
    set -a && source .env && set +a
    PYTHONPATH=. python scripts/e2e_smoke.py

Garantías:
- Las señales LIVE se persisten en la DB de producción (comportamiento normal).
- Los trades SHADOW se ejecutan en una DB temporal aislada que se destruye al
  terminar el script → no contaminan métricas reales.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

from core.config import load_config
from core.database import Database
from core.interfaces import EventBus
from core.models import Decision
from engine.ev_calculator import EVCalculator
from engine.probability import ProbabilityEngine, classify_time_zone
from engine.signal_router import SignalRouter
from engine.timing import TimingFilter
from execution.order_executor import PaperOrderExecutor
from feeds.binance_feed import BinancePriceFeed
from feeds.kalshi_feed import KalshiFeed


async def run_e2e() -> None:
    cfg = load_config("config.json")

    # ── DB de producción (señales live) ───────────────────────────────────────
    prod_db = Database(
        path=cfg.database.path,
        wal_mode=cfg.database.wal_mode,
        busy_timeout_ms=cfg.database.busy_timeout_ms,
    )
    prod_db.initialize()

    # ── DB temporal para shadow (completamente aislada) ───────────────────────
    shadow_fd, shadow_db_path_str = tempfile.mkstemp(suffix="_shadow_smoke.db")
    os.close(shadow_fd)
    shadow_db_path = Path(shadow_db_path_str)
    shadow_db = Database(path=shadow_db_path)
    shadow_db.initialize()

    def _make_router(db: Database, blocked: set[str] | None = None) -> SignalRouter:
        return SignalRouter(
            prob_engine=ProbabilityEngine(),
            ev_calc=EVCalculator(),
            timing_filter=TimingFilter(),
            config=cfg.engine,
            db=db,
            blocked_categories=blocked or set(),
        )

    live_router = _make_router(prod_db, blocked_categories=set(prod_db.get_blocked_categories()))
    shadow_router = _make_router(shadow_db)   # sin blocked, DB vacía → siempre fresco

    live_executor = PaperOrderExecutor(db=prod_db, mode="demo")
    shadow_executor = PaperOrderExecutor(db=shadow_db, mode="demo")

    kfeed = KalshiFeed(cfg, EventBus())
    bfeed = BinancePriceFeed(cfg.binance, EventBus())

    try:
        await kfeed.connect()

        url = f"{cfg.kalshi.base_url}/markets"
        headers = kfeed._auth_headers(method="GET", path="/markets")
        cursor: str | None = None
        crypto = []
        for _ in range(12):
            params = {"status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            async with kfeed._session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()
            parsed = [kfeed._parse_market(raw) for raw in data.get("markets", [])]
            for market in parsed:
                if market is not None and market.category in {"BTC", "ETH"} and market.strike is not None:
                    crypto.append(market)
            cursor = data.get("cursor")
            if not cursor:
                break

        if not crypto:
            raise RuntimeError("No hay mercados crypto abiertos para E2E")

        await bfeed.connect()
        price = await asyncio.wait_for(anext(bfeed.stream()), timeout=20)

        live_market = sorted(
            [m for m in crypto if m.category == price.symbol] or crypto,
            key=lambda m: m.time_to_expiry_s,
        )[0]

        # ── Señal LIVE (persiste en prod_db) ──────────────────────────────────
        live_signal = live_router.evaluate(market=live_market, price=price, bankroll=1000.0)
        live_zone = classify_time_zone(live_market.time_to_expiry_s)

        # ── Señal SHADOW (persiste en shadow_db solamente) ────────────────────
        # Se construye un mercado near-ATM para que el pipeline ejercite el
        # camino YES/NO con delta razonable, sin sesgar a OTM.
        from dataclasses import replace
        shadow_market = replace(
            live_market,
            time_to_expiry_s=600,
            implied_prob=0.52,
            yes_ask=0.52,
            no_ask=0.49,
            timestamp=time.time(),
        )
        shadow_signal = shadow_router.evaluate(market=shadow_market, price=price, bankroll=1000.0)
        shadow_zone = classify_time_zone(shadow_market.time_to_expiry_s)

        shadow_trade = None
        if shadow_signal.decision in (Decision.YES, Decision.NO):
            opened = await shadow_executor.submit(shadow_signal)
            move = 0.03 if opened.side == "YES" else -0.03
            exit_price = min(0.99, max(0.01, opened.entry_price + move))
            shadow_trade = await shadow_executor.close_with_price(opened, exit_price)

        now = time.time()
        prod_signals = prod_db.get_signals(now - 3600, now, limit=1000)

        print("=" * 60)
        print("E2E_RESULT")
        print("=" * 60)
        print(f"binance_price    : {price.symbol}@{price.price:.2f}")
        print()
        print(f"[LIVE]  market   : {live_market.ticker}  tte={live_market.time_to_expiry_s}s  zone={live_zone}")
        print(f"[LIVE]  signal   : {live_signal.decision.value}  delta={live_signal.delta:.4f}  ev={live_signal.ev_net_fees:.4f}  conf={live_signal.confidence.value}")
        print(f"[LIVE]  reason   : {live_signal.reasoning}")
        print()
        print(f"[SHADOW] market  : {shadow_market.ticker}  tte={shadow_market.time_to_expiry_s}s  zone={shadow_zone}")
        print(f"[SHADOW] implied : {shadow_market.implied_prob:.2f}  yes_ask={shadow_market.yes_ask:.2f}  no_ask={shadow_market.no_ask:.2f}")
        print(f"[SHADOW] signal  : {shadow_signal.decision.value}  delta={shadow_signal.delta:.4f}  ev={shadow_signal.ev_net_fees:.4f}")
        if shadow_trade:
            print(f"[SHADOW] trade   : id={shadow_trade.id}  side={shadow_trade.side}  pnl={shadow_trade.pnl:.4f}  status={shadow_trade.status.value}")
        else:
            print("[SHADOW] trade   : not executed (signal was SKIP/WAIT/ERROR)")
        print()
        print(f"prod_signals_1h  : {len(prod_signals)}")
        print(f"prod_open_trades : {len(prod_db.get_open_trades())}")
        print(f"shadow_db        : ISOLATED — destroyed after script")
        print("=" * 60)

    finally:
        for feed in (bfeed, kfeed):
            try:
                await feed.disconnect()
            except Exception:
                pass
        prod_db.close()
        shadow_db.close()
        try:
            shadow_db_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    asyncio.run(run_e2e())


if __name__ == "__main__":
    main()
