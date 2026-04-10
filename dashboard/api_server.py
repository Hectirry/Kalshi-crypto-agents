"""
dashboard/api_server.py

API minima de observabilidad para exponer estado del sistema al dashboard.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.config import AppConfig, load_config
from core.database import Database
from core.interfaces import EventBus
from core.models import MarketSnapshot, Signal, Trade
from feeds.kalshi_feed import KalshiFeed

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    db: Database = app.state.db
    if not app.state.external_db:
        db.initialize()
    try:
        yield
    finally:
        if not app.state.external_db:
            db.close()


def _signal_payload(signal: Signal) -> dict[str, Any]:
    """Normaliza una señal para respuesta JSON."""

    return {
        "ticker": signal.market_ticker,
        "decision": signal.decision.value,
        "my_probability": signal.my_probability,
        "market_probability": signal.market_probability,
        "delta": signal.delta,
        "ev_net_fees": signal.ev_net_fees,
        "kelly_size": signal.kelly_size,
        "confidence": signal.confidence.value,
        "time_remaining_s": signal.time_remaining_s,
        "timestamp": signal.timestamp,
        "reasoning": signal.reasoning,
        "outcome": signal.outcome.value if signal.outcome else None,
    }


def _trade_payload(trade: Trade) -> dict[str, Any]:
    """Normaliza un trade para respuesta JSON."""

    return {
        "id": trade.id,
        "signal_id": trade.signal_id,
        "ticker": trade.ticker,
        "side": trade.side,
        "contracts": trade.contracts,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "fee_paid": trade.fee_paid,
        "pnl": trade.pnl,
        "mode": trade.mode.value,
        "status": trade.status.value,
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
    }


def _market_payload(market: MarketSnapshot) -> dict[str, Any]:
    """Normaliza un mercado vivo para respuesta JSON."""

    return {
        "ticker": market.ticker,
        "category": market.category,
        "strike": market.strike,
        "implied_prob": market.implied_prob,
        "yes_ask": market.yes_ask,
        "no_ask": market.no_ask,
        "volume_24h": market.volume_24h,
        "time_to_expiry_s": market.time_to_expiry_s,
        "timestamp": market.timestamp,
    }


def create_app(db: Database | None = None, config: AppConfig | None = None) -> FastAPI:
    """
    Crea la aplicacion FastAPI del dashboard.

    Si no se pasa DB, se construye desde el config activo.
    """

    external_db = db is not None
    if config is None:
        config = load_config()
    if db is None:
        db = Database(
            path=config.database.path,
            wal_mode=config.database.wal_mode,
            busy_timeout_ms=config.database.busy_timeout_ms,
        )

    app = FastAPI(title="Kalshi Crypto Agents API", version="0.1.0", lifespan=_lifespan)
    app.state.db = db
    app.state.config = config
    app.state.external_db = external_db
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/state")
    async def state(limit: int = 20) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 200))
        now = time.time()
        recent_signals = app.state.db.get_signals(
            from_ts=0,
            to_ts=now,
            limit=safe_limit,
        )
        open_trades = app.state.db.get_open_trades()
        blocked_categories = sorted(app.state.db.get_blocked_categories())
        current_params = app.state.db.get_current_params()
        return {
            "timestamp": now,
            "open_trades_count": len(open_trades),
            "open_trades": [_trade_payload(trade) for trade in open_trades],
            "blocked_categories": blocked_categories,
            "current_params": current_params,
            "recent_signals": [_signal_payload(signal) for signal in recent_signals[-safe_limit:]],
        }

    @app.get("/live-markets")
    async def live_markets(limit: int = 40) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 200))
        started = time.time()
        feed = KalshiFeed(app.state.config, EventBus())
        markets = await feed.get_active_markets()
        markets = sorted(
            markets,
            key=lambda market: (market.category, market.time_to_expiry_s, market.ticker),
        )
        return {
            "timestamp": time.time(),
            "latency_ms": round((time.time() - started) * 1000),
            "count": len(markets[:safe_limit]),
            "markets": [_market_payload(market) for market in markets[:safe_limit]],
        }

    return app
