"""
dashboard/api_server.py

API minima de observabilidad para exponer estado del sistema al dashboard.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from core.config import load_config
from core.database import Database


@asynccontextmanager
async def _lifespan(app: FastAPI):
    db: Database = app.state.db
    db.initialize()
    try:
        yield
    finally:
        db.close()


def _signal_payload(signal: Any) -> dict[str, Any]:
    """Normaliza una señal para respuesta JSON."""

    return {
        "ticker": signal.market_ticker,
        "decision": signal.decision.value,
        "delta": signal.delta,
        "ev_net_fees": signal.ev_net_fees,
        "confidence": signal.confidence.value,
        "time_remaining_s": signal.time_remaining_s,
        "timestamp": signal.timestamp,
        "reasoning": signal.reasoning,
        "outcome": signal.outcome.value if signal.outcome else None,
    }


def create_app(db: Database | None = None) -> FastAPI:
    """
    Crea la aplicacion FastAPI del dashboard.

    Si no se pasa DB, se construye desde el config activo.
    """

    if db is None:
        config = load_config()
        db = Database(
            path=config.database.path,
            wal_mode=config.database.wal_mode,
            busy_timeout_ms=config.database.busy_timeout_ms,
        )

    app = FastAPI(title="Kalshi Crypto Agents API", version="0.1.0", lifespan=_lifespan)
    app.state.db = db

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
            "blocked_categories": blocked_categories,
            "current_params": current_params,
            "recent_signals": [_signal_payload(signal) for signal in recent_signals[-safe_limit:]],
        }

    return app
