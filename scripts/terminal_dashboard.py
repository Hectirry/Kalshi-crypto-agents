"""Dashboard terminal estilo trading desk + matrix para Kalshi crypto agents."""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import sqlite3
import sys
import time
from dataclasses import dataclass

import aiohttp

from core.config import AppConfig, load_config
from core.database import Database
from core.interfaces import EventBus
from core.models import Trade
from engine.ev_calculator import EVCalculator
from feeds.kalshi_feed import KalshiFeed

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_BRIGHT_GREEN = "\033[1;92m"
ANSI_DIM_GREEN = "\033[2;92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"


@dataclass(slots=True)
class TradeView:
    """Representa un trade con estimación de P&L no realizado."""

    trade: Trade
    mark_price: float | None
    unrealized_pnl: float | None


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _coerce_price(*values: object, default: float | None = None) -> float | None:
    """Convierte precios de payload Kalshi a rango [0.01, 0.99]."""

    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
            if parsed > 1.0:
                parsed = parsed / 100.0
            return max(0.01, min(0.99, parsed))
        except (TypeError, ValueError):
            continue
    return default


def _style_pnl(value: float | None) -> str:
    if value is None:
        return f"{ANSI_DIM_GREEN}n/a{ANSI_RESET}"
    color = ANSI_GREEN if value >= 0 else ANSI_RED
    return f"{color}{_fmt_money(value)}{ANSI_RESET}"


def _clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")


def _matrix_banner(width: int) -> str:
    charset = "01ABCDEF$%#@"
    line = "".join(random.choice(charset) for _ in range(max(20, width - 2)))
    title = " KALSHI CRYPTO TERMINAL "
    start = max(0, (len(line) - len(title)) // 2)
    merged = line[:start] + title + line[start + len(title):]
    return f"{ANSI_BRIGHT_GREEN}{merged[:width]}{ANSI_RESET}"


def _read_closed_stats(db_path: str) -> tuple[int, int, float]:
    """Devuelve (total_closed, wins, total_realized_pnl)."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT pnl
            FROM trades
            WHERE status = 'closed'
            """
        ).fetchall()
    finally:
        conn.close()
    total = len(rows)
    wins = sum(1 for row in rows if (row["pnl"] or 0.0) > 0.0)
    pnl = sum((row["pnl"] or 0.0) for row in rows)
    return total, wins, pnl


def _read_recent_signals(db_path: str, limit: int = 8) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT ticker, decision, delta, ev_net_fees, reasoning, created_at
            FROM signals
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return rows


async def _fetch_market_quote(
    session: aiohttp.ClientSession,
    scanner: KalshiFeed,
    base_url: str,
    ticker: str,
) -> float | None:
    """Consulta el mercado y devuelve precio mark para YES side."""

    path = f"/markets/{ticker}"
    url = f"{base_url}{path}"
    headers = scanner._auth_headers(method="GET", path=path)
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status != 200:
                return None
            payload = await response.json()
    except (aiohttp.ClientError, TimeoutError):
        return None

    market = payload.get("market", payload)
    yes_ask = _coerce_price(market.get("yes_ask"), market.get("yes_ask_dollars"), default=None)
    yes_bid = _coerce_price(market.get("yes_bid"), market.get("yes_bid_dollars"), default=None)
    last = _coerce_price(
        market.get("last_price"),
        market.get("last_price_dollars"),
        market.get("previous_price_dollars"),
        default=None,
    )

    if yes_ask is not None and yes_bid is not None:
        return (yes_ask + yes_bid) / 2.0
    if yes_ask is not None:
        return yes_ask
    if yes_bid is not None:
        return yes_bid
    return last


async def _build_trade_views(
    config: AppConfig,
    open_trades: list[Trade],
) -> list[TradeView]:
    """Construye estimaciones MTM de trades abiertos."""

    if not open_trades:
        return []

    scanner = KalshiFeed(config, EventBus())
    fee_calc = EVCalculator()
    async with aiohttp.ClientSession() as session:
        quotes = await asyncio.gather(
            *[
                _fetch_market_quote(
                    session=session,
                    scanner=scanner,
                    base_url=config.kalshi.base_url,
                    ticker=trade.ticker,
                )
                for trade in open_trades
            ]
        )

    views: list[TradeView] = []
    for trade, yes_mark in zip(open_trades, quotes):
        if yes_mark is None:
            views.append(TradeView(trade=trade, mark_price=None, unrealized_pnl=None))
            continue

        mark_side = yes_mark if trade.side == "YES" else max(0.01, min(0.99, 1.0 - yes_mark))
        entry_fee = fee_calc.fee_per_contract(trade.entry_price) * trade.contracts
        exit_fee = fee_calc.fee_per_contract(mark_side) * trade.contracts
        unrealized = ((mark_side - trade.entry_price) * trade.contracts) - entry_fee - exit_fee
        views.append(TradeView(trade=trade, mark_price=mark_side, unrealized_pnl=unrealized))
    return views


def _render_table_rows(rows: list[str]) -> str:
    return "\n".join(rows) if rows else f"{ANSI_DIM_GREEN}(sin datos){ANSI_RESET}"


async def run_dashboard(config_path: str, refresh_s: int) -> None:
    """Loop principal del dashboard terminal."""

    config = load_config(config_path)
    db = Database(
        path=config.database.path,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    )
    db.initialize()

    try:
        while True:
            start = time.time()
            width = 120

            open_trades = db.get_open_trades()
            trade_views = await _build_trade_views(config=config, open_trades=open_trades)
            closed_total, closed_wins, realized_pnl = _read_closed_stats(str(config.database.path))
            recent_signals = _read_recent_signals(str(config.database.path), limit=8)

            win_rate = (closed_wins / closed_total) if closed_total else 0.0
            unrealized_total = sum(view.unrealized_pnl or 0.0 for view in trade_views)
            open_notional = sum(trade.entry_price * trade.contracts for trade in open_trades)

            _clear_screen()
            print(_matrix_banner(width))
            print(
                f"{ANSI_DIM_GREEN}UTC {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}"
                f" | refresh {refresh_s}s | db {config.database.path}{ANSI_RESET}"
            )
            print()
            print(
                f"{ANSI_BRIGHT_GREEN}OPEN:{ANSI_RESET} {len(open_trades):>3}  "
                f"{ANSI_BRIGHT_GREEN}NOTIONAL:{ANSI_RESET} {_fmt_money(open_notional)}  "
                f"{ANSI_BRIGHT_GREEN}UNREALIZED:{ANSI_RESET} {_style_pnl(unrealized_total)}  "
                f"{ANSI_BRIGHT_GREEN}REALIZED:{ANSI_RESET} {_style_pnl(realized_pnl)}  "
                f"{ANSI_BRIGHT_GREEN}WIN RATE:{ANSI_RESET} {_fmt_pct(win_rate)}"
            )
            print()
            print(f"{ANSI_BRIGHT_GREEN}OPEN POSITIONS{ANSI_RESET}")
            print(
                f"{ANSI_DIM_GREEN}{'TICKER':<34} {'SIDE':<4} {'CTS':>4} {'ENTRY':>7} "
                f"{'MARK':>7} {'uPnL':>12}{ANSI_RESET}"
            )

            trade_rows: list[str] = []
            for view in trade_views[:12]:
                trade_rows.append(
                    f"{view.trade.ticker:<34} {view.trade.side:<4} {view.trade.contracts:>4} "
                    f"{view.trade.entry_price:>7.3f} "
                    f"{(f'{view.mark_price:0.3f}' if view.mark_price is not None else 'n/a'):>7} "
                    f"{(_fmt_money(view.unrealized_pnl) if view.unrealized_pnl is not None else 'n/a'):>12}"
                )
            print(_render_table_rows(trade_rows))
            print()
            print(f"{ANSI_BRIGHT_GREEN}LATEST SIGNALS{ANSI_RESET}")
            print(
                f"{ANSI_DIM_GREEN}{'DEC':<5} {'DELTA':>8} {'EV':>9} {'TICKER':<34} {'REASON':<24}{ANSI_RESET}"
            )

            signal_rows: list[str] = []
            for row in recent_signals:
                decision = row["decision"]
                color = ANSI_GREEN if decision in ("YES", "NO") else ANSI_YELLOW
                reason = (row["reasoning"] or "")[:24]
                signal_rows.append(
                    f"{color}{decision:<5}{ANSI_RESET} {row['delta']:>8.4f} {row['ev_net_fees']:>9.4f} "
                    f"{row['ticker']:<34} {reason:<24}"
                )
            print(_render_table_rows(signal_rows))
            print()
            latency_ms = math.floor((time.time() - start) * 1000)
            print(f"{ANSI_DIM_GREEN}heartbeat: ok | render_latency={latency_ms}ms{ANSI_RESET}")

            await asyncio.sleep(refresh_s)
    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal dashboard para Kalshi crypto agents.")
    parser.add_argument("--config", default="config.json", help="Ruta a config.json")
    parser.add_argument("--refresh", type=int, default=5, help="Segundos entre refrescos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_dashboard(config_path=args.config, refresh_s=max(2, args.refresh)))


if __name__ == "__main__":
    main()
