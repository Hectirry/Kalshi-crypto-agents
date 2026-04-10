"""
scripts/trading_loop.py

Bucle de trading demo en tiempo real.

Flujo por ciclo de mercado:
  1. BinancePriceFeed mantiene caché del último precio BTC/ETH.
  2. KalshiFeed streama actualizaciones de mercados crypto de 15 min.
  3. Para cada MarketSnapshot recibido:
       a. Cerrar posición abierta si time_to_expiry_s <= EXPIRY_CLOSE_S
       b. Evaluar SL/TP en posiciones abiertas del mismo ticker
       c. Generar señal si el ticker no tiene posición abierta
       d. Abrir trade si la señal es actionable y open_positions < max_positions
  4. Task periódico: sync desde DB, log de estado y go/no-go.

Uso:
    PYTHONPATH=. python scripts/trading_loop.py --bankroll 1000 --max-positions 3
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from core.config import load_config
from core.database import Database
from core.interfaces import EventBus
from core.models import PriceSnapshot
from engine.ev_calculator import EVCalculator
from engine.probability import ProbabilityEngine, classify_time_zone
from engine.signal_router import SignalRouter
from engine.timing import TimingFilter
from execution.order_executor import PaperOrderExecutor
from execution.position_manager import PositionManager
from feeds.binance_feed import BinancePriceFeed
from feeds.kalshi_feed import KalshiFeed

logger = logging.getLogger(__name__)

# Umbral de expiración: cerrar posición cuando queden <= N segundos.
EXPIRY_CLOSE_S = 120
# Pausa del loop de supervisión periódica.
SUPERVISOR_INTERVAL_S = 60
# Intervalo de polling REST de mercados Kalshi (complementa el WS).
REST_POLL_INTERVAL_S = 90


async def run_trading_demo(
    config_path: str = "config.json",
    bankroll: float = 1000.0,
    max_positions: int = 3,
) -> None:
    """
    Corre el loop de trading demo indefinidamente.

    Args:
        config_path: ruta a config.json.
        bankroll: capital virtual en USD para cálculo de Kelly.
        max_positions: máximo de posiciones demo abiertas simultáneamente.
    """
    cfg = load_config(config_path)

    # ── Persistencia ──────────────────────────────────────────────────────────
    db = Database(
        path=cfg.database.path,
        wal_mode=cfg.database.wal_mode,
        busy_timeout_ms=cfg.database.busy_timeout_ms,
    )
    db.initialize()

    # ── Engine ────────────────────────────────────────────────────────────────
    router = SignalRouter(
        prob_engine=ProbabilityEngine(),
        ev_calc=EVCalculator(),
        timing_filter=TimingFilter(),
        config=cfg.engine,
        db=db,
        blocked_categories=set(db.get_blocked_categories()),
    )
    executor = PaperOrderExecutor(db=db, mode="demo")
    pos_mgr = PositionManager(db=db, executor=executor)
    pos_mgr.sync_open_trades()

    # ── Feeds ─────────────────────────────────────────────────────────────────
    bus = EventBus()
    bfeed = BinancePriceFeed(cfg.binance, bus)
    kfeed = KalshiFeed(cfg, bus)

    # ── Estado compartido (asyncio es single-thread → acceso seguro) ──────────
    latest_prices: dict[str, PriceSnapshot] = {}   # "BTC" / "ETH" → última snapshot

    logger.info(
        "trading_loop_start mode=demo bankroll=%.0f max_positions=%d",
        bankroll, max_positions,
    )
    print(f"[TRADING] Iniciando demo loop  bankroll={bankroll:.0f}  max_pos={max_positions}")

    await bfeed.connect()
    await kfeed.connect()

    try:
        await asyncio.gather(
            _price_task(bfeed, latest_prices),
            _market_task(kfeed, router, pos_mgr, latest_prices, bankroll, max_positions),
            _rest_poll_task(kfeed, router, pos_mgr, latest_prices, bankroll, max_positions),
            _supervisor_task(pos_mgr, db),
        )
    finally:
        for feed in (bfeed, kfeed):
            try:
                await feed.disconnect()
            except Exception:
                pass
        db.close()
        logger.info("trading_loop_stopped")


# ══════════════════════════════════════════════════════════════════════════════
# Tasks internas
# ══════════════════════════════════════════════════════════════════════════════

async def _price_task(
    bfeed: BinancePriceFeed,
    latest_prices: dict[str, PriceSnapshot],
) -> None:
    """Actualiza la caché de precios spot continuamente."""
    async for snap in bfeed.stream():
        latest_prices[snap.symbol] = snap


async def _market_task(
    kfeed: KalshiFeed,
    router: SignalRouter,
    pos_mgr: PositionManager,
    latest_prices: dict[str, PriceSnapshot],
    bankroll: float,
    max_positions: int,
) -> None:
    """Procesa cada MarketSnapshot del stream WS de Kalshi."""
    async for market in kfeed.stream_markets():
        await _process_market(market, router, pos_mgr, latest_prices, bankroll, max_positions)


async def _process_market(
    market,
    router: SignalRouter,
    pos_mgr: PositionManager,
    latest_prices: dict,
    bankroll: float,
    max_positions: int,
) -> None:
    """Lógica compartida de evaluación para un MarketSnapshot (WS o REST)."""
    ticker = market.ticker

    if market.time_to_expiry_s <= EXPIRY_CLOSE_S:
        for trade_id, trade in list(pos_mgr.open_positions.items()):
            if trade.ticker == ticker:
                exit_price = (
                    market.implied_prob
                    if trade.side == "YES"
                    else max(0.01, min(0.99, 1.0 - market.implied_prob))
                )
                closed = await pos_mgr.executor.close_with_price(trade, exit_price)
                del pos_mgr.open_positions[trade_id]
                zone = classify_time_zone(market.time_to_expiry_s)
                _print_trade_close(closed, reason="expiry", zone=zone)
        return

    closes = await pos_mgr.evaluate_price(
        ticker=ticker,
        current_yes_price=market.implied_prob,
    )
    for mc in closes:
        zone = classify_time_zone(market.time_to_expiry_s)
        _print_trade_close(mc.trade, reason=mc.reason, zone=zone)

    already_open = any(t.ticker == ticker for t in pos_mgr.open_positions.values())
    if already_open:
        return

    if len(pos_mgr.open_positions) >= max_positions:
        return

    price = latest_prices.get(market.category)
    if price is None:
        return

    signal = router.evaluate(market=market, price=price, bankroll=bankroll)
    zone = classify_time_zone(market.time_to_expiry_s)

    if signal.is_actionable:
        trade = await pos_mgr.open_from_signal(signal)
        _print_trade_open(trade, signal, zone=zone)
    else:
        logger.debug(
            "signal_skip ticker=%s decision=%s zone=%s reason=%s",
            ticker, signal.decision.value, zone, signal.reasoning,
        )


async def _rest_poll_task(
    kfeed: KalshiFeed,
    router: SignalRouter,
    pos_mgr: PositionManager,
    latest_prices: dict,
    bankroll: float,
    max_positions: int,
) -> None:
    """Polling REST periódico para no depender exclusivamente del WS."""
    # Espera inicial para que los precios spot estén disponibles
    await asyncio.sleep(5)
    while True:
        try:
            markets = await kfeed.get_active_markets()
            logger.info("rest_poll markets_found=%d", len(markets))
            for market in markets:
                await _process_market(market, router, pos_mgr, latest_prices, bankroll, max_positions)
        except Exception as exc:
            logger.warning("rest_poll_error: %s", exc)
        await asyncio.sleep(REST_POLL_INTERVAL_S)


async def _supervisor_task(pos_mgr: PositionManager, db: Database) -> None:
    """Log periódico de estado y sincronización de trades desde DB."""
    while True:
        await asyncio.sleep(SUPERVISOR_INTERVAL_S)
        pos_mgr.sync_open_trades()
        snap = pos_mgr.observability_snapshot()
        gng = pos_mgr.go_no_go_status()
        print(
            f"[SUPERVISOR] "
            f"open={snap['open_positions']}  "
            f"closed={snap['closed_positions']}  "
            f"win_rate={snap['win_rate']:.1%}  "
            f"pnl={snap['total_pnl']:.4f}  "
            f"fees={snap['total_fees']:.4f}  "
            f"go_no_go={'GO' if gng.allowed else 'NO-GO: ' + gng.reason}"
        )
        logger.info(
            "supervisor open=%d closed=%d win_rate=%.3f pnl=%.4f go=%s",
            snap["open_positions"],
            snap["closed_positions"],
            snap["win_rate"],
            snap["total_pnl"],
            gng.allowed,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de salida
# ══════════════════════════════════════════════════════════════════════════════

def _print_trade_open(trade, signal, zone: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(
        f"[{ts}] OPEN  {trade.ticker:40s}  "
        f"side={trade.side}  "
        f"contracts={trade.contracts}  "
        f"entry={trade.entry_price:.4f}  "
        f"delta={signal.delta:+.4f}  "
        f"ev={signal.ev_net_fees:.4f}  "
        f"conf={signal.confidence.value}  "
        f"zone={zone}"
    )
    logger.info(
        "trade_opened ticker=%s side=%s contracts=%d entry=%.4f delta=%.4f ev=%.4f zone=%s",
        trade.ticker, trade.side, trade.contracts, trade.entry_price,
        signal.delta, signal.ev_net_fees, zone,
    )


def _print_trade_close(trade, reason: str, zone: str) -> None:
    ts = time.strftime("%H:%M:%S")
    pnl = trade.pnl or 0.0
    sign = "+" if pnl >= 0 else ""
    print(
        f"[{ts}] CLOSE {trade.ticker:40s}  "
        f"side={trade.side}  "
        f"exit={trade.exit_price:.4f}  "
        f"pnl={sign}{pnl:.4f}  "
        f"reason={reason}  "
        f"zone={zone}"
    )
    logger.info(
        "trade_closed ticker=%s side=%s exit=%.4f pnl=%.4f reason=%s zone=%s",
        trade.ticker, trade.side, trade.exit_price or 0, pnl, reason, zone,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint standalone
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Demo trading loop")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--max-positions", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run_trading_demo(
        config_path=args.config,
        bankroll=args.bankroll,
        max_positions=args.max_positions,
    ))


if __name__ == "__main__":
    main()
