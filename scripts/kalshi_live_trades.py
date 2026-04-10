"""
scripts/kalshi_live_trades.py

Tail publico de trades de Kalshi usando /markets/trades.
No requiere credenciales y sirve para comparar contra https://kalshi.com/calendar.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
CRYPTO_MARKERS = ("BTC", "ETH", "SOL")


@dataclass(frozen=True, slots=True)
class PublicTrade:
    """Trade publico devuelto por Kalshi."""

    trade_id: str
    ticker: str
    created_time: str
    taker_side: str
    yes_price: str
    no_price: str
    count: str


def parse_args() -> argparse.Namespace:
    """Parsea argumentos del watcher."""

    parser = argparse.ArgumentParser(description="Tail de trades publicos de Kalshi")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ticker", default=None, help="Filtra por ticker exacto")
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Muestra solo tickers que contengan BTC, ETH o SOL",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--poll-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Ejecuta polling continuo e imprime nuevos trades."""

    args = parse_args()
    seen: set[str] = set()
    min_ts = int(time.time()) - 60

    while True:
        trades = fetch_trades(
            base_url=args.base_url,
            limit=args.limit,
            ticker=args.ticker,
            min_ts=min_ts,
        )
        fresh = [trade for trade in reversed(trades) if trade.trade_id not in seen]
        for trade in fresh:
            seen.add(trade.trade_id)
            if args.crypto_only and not _is_crypto_ticker(trade.ticker):
                continue
            print(_format_trade(trade), flush=True)

        if trades:
            latest = max(_parse_time(trade.created_time) for trade in trades)
            min_ts = max(min_ts, int(latest.timestamp()) - 2)
        time.sleep(max(0.5, args.poll_s))


def fetch_trades(
    *,
    base_url: str,
    limit: int,
    ticker: str | None,
    min_ts: int,
) -> list[PublicTrade]:
    """Carga trades publicos recientes desde Kalshi."""

    params: dict[str, Any] = {
        "limit": max(1, min(limit, 1000)),
        "min_ts": min_ts,
    }
    if ticker:
        params["ticker"] = ticker

    url = f"{base_url.rstrip('/')}/markets/trades?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)

    return [_parse_trade(raw) for raw in payload.get("trades", [])]


def _parse_trade(raw: dict[str, Any]) -> PublicTrade:
    return PublicTrade(
        trade_id=str(raw.get("trade_id", "")),
        ticker=str(raw.get("ticker", "")),
        created_time=str(raw.get("created_time", "")),
        taker_side=str(raw.get("taker_side", "")),
        yes_price=str(raw.get("yes_price_dollars", "")),
        no_price=str(raw.get("no_price_dollars", "")),
        count=str(raw.get("count_fp", "")),
    )


def _format_trade(trade: PublicTrade) -> str:
    created = _parse_time(trade.created_time).strftime("%H:%M:%S")
    return (
        f"{created} {trade.ticker} taker={trade.taker_side.upper():<3} "
        f"yes={trade.yes_price:<6} no={trade.no_price:<6} count={trade.count}"
    )


def _parse_time(value: str) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_crypto_ticker(ticker: str) -> bool:
    upper = ticker.upper()
    return any(marker in upper for marker in CRYPTO_MARKERS)


if __name__ == "__main__":
    main()
