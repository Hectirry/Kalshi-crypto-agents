"""
dashboard/api_server.py

API minima de observabilidad para exponer estado del sistema al dashboard.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analytics.execution_quality import BucketStats, ExecutionQualityAnalyzer
from core.config import AppConfig, load_config
from core.database import Database
from core.interfaces import EventBus
from core.models import Confidence, Decision, MarketSnapshot, Signal, Trade, TradeMode, TradeStatus
from execution.order_executor import PaperOrderExecutor
from feeds.kalshi_feed import KalshiFeed

STATIC_DIR = Path(__file__).parent / "static"


class ManualPaperOrderRequest(BaseModel):
    ticker: str
    side: str
    contracts: int = Field(default=1, ge=1, le=100)
    note: str | None = None


class ManualPaperCloseRequest(BaseModel):
    trade_id: int
    note: str | None = None


def _period_bounds(now_ts: float) -> dict[str, tuple[float, float, str]]:
    """Calcula ventanas temporales UTC para métricas del dashboard."""

    now = datetime.fromtimestamp(now_ts, tz=UTC)
    today_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
    week_start = today_start - timedelta(days=today_start.weekday())
    previous_week_start = week_start - timedelta(days=7)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    if now.month == 12:
        next_month_start = datetime(now.year + 1, 1, 1, tzinfo=UTC)
    else:
        next_month_start = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
    return {
        "today": (today_start.timestamp(), now_ts, today_start.date().isoformat()),
        "current_week": (
            week_start.timestamp(),
            now_ts,
            f"{week_start.date().isoformat()} to {now.date().isoformat()}",
        ),
        "previous_week": (
            previous_week_start.timestamp(),
            week_start.timestamp(),
            f"{previous_week_start.date().isoformat()} to {(week_start - timedelta(days=1)).date().isoformat()}",
        ),
        "current_month": (
            month_start.timestamp(),
            min(now_ts, next_month_start.timestamp()),
            month_start.strftime("%Y-%m"),
        ),
    }


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
        "label": _human_market_label(
            signal.market_ticker,
            _infer_category(signal.market_ticker),
            _extract_strike_from_ticker(signal.market_ticker),
        ),
        "market_url": _market_url(
            signal.market_ticker,
            category=_infer_category(signal.market_ticker),
        ),
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
        "label": _human_market_label(
            trade.ticker,
            _infer_category(trade.ticker),
            _extract_strike_from_ticker(trade.ticker),
        ),
        "market_url": _market_url(
            trade.ticker,
            category=_infer_category(trade.ticker),
        ),
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
        "label": _human_market_label(market.ticker, market.category, market.strike),
        "subtitle": _human_market_subtitle(market),
        "market_url": _market_url(
            market.ticker,
            category=market.category,
            title=market.title,
            event_ticker=market.event_ticker,
        ),
        "implied_prob": market.implied_prob,
        "yes_ask": market.yes_ask,
        "no_ask": market.no_ask,
        "volume_24h": market.volume_24h,
        "time_to_expiry_s": market.time_to_expiry_s,
        "timestamp": market.timestamp,
        "event_ticker": market.event_ticker,
        "title": market.title,
    }


def _market_url(
    ticker: str,
    *,
    category: str | None = None,
    title: str | None = None,
    event_ticker: str | None = None,
) -> str:
    """Construye un link probable al mercado web de Kalshi usando su path real."""

    safe_ticker = (ticker or "").strip()
    if not safe_ticker:
        return "https://kalshi.com/markets"

    event = (event_ticker or _event_ticker_from_ticker(safe_ticker)).strip().lower()
    slug = _slugify(title or _default_market_title(category=category, ticker=safe_ticker))
    return (
        f"https://kalshi.com/markets/"
        f"{quote(event, safe='')}/"
        f"{quote(slug, safe='')}/"
        f"{quote(safe_ticker.lower(), safe='')}"
    )


def _event_ticker_from_ticker(ticker: str) -> str:
    """Deriva el event_ticker eliminando el último segmento del ticker de mercado."""

    parts = [part for part in ticker.split("-") if part]
    if len(parts) <= 1:
        return ticker
    return "-".join(parts[:-1])


def _default_market_title(category: str | None, ticker: str) -> str:
    """Genera un slug humano por defecto si no viene title desde la API."""

    normalized = (category or _infer_category(ticker)).lower()
    if "15M" in ticker.upper():
        return f"{normalized} price up in next 15 mins"
    return f"{normalized} market"


def _slugify(value: str) -> str:
    """Convierte texto libre a slug web simple."""

    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "market"


def _infer_category(ticker: str) -> str:
    upper = ticker.upper()
    for category in ("BTC", "ETH", "SOL"):
        if category in upper:
            return category
    return "UNKNOWN"


def _extract_strike_from_ticker(ticker: str) -> float | None:
    """Intenta inferir el strike desde el sufijo del ticker."""

    match = re.search(r"-(?:B|T)(\d+(?:\.\d+)?)$", ticker.upper())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _human_market_label(ticker: str, category: str, strike: float | None) -> str:
    """Genera una etiqueta legible para humanos."""

    if strike is not None:
        return f"{category} · strike {strike:,.2f}".replace(".00", "")
    return f"{category} · contrato de 15m"


def _human_market_subtitle(market: MarketSnapshot) -> str:
    """Genera un resumen compacto del contrato."""

    if market.strike is not None:
        return (
            f"Mercado de 15m para {market.category}. "
            f"YES y NO se resuelven contra el strike {market.strike:,.2f}."
        ).replace(".00", "")
    return f"Mercado crypto de 15m para {market.category}."


def _effective_execution_mode(config: AppConfig, runtime_mode: str | None = None) -> str:
    """Resuelve el modo efectivo del dashboard respecto a paper trading."""

    if runtime_mode in {"demo", "production"}:
        return runtime_mode
    if config.is_demo:
        return "demo"
    if str(os.getenv("PAPER_TRADE", "")).lower() in {"1", "true", "yes"}:
        return "demo"
    return "production"


def _manual_actions_enabled(config: AppConfig, runtime_mode: str | None = None) -> bool:
    """Indica si el dashboard debe permitir acciones manuales."""

    return _effective_execution_mode(config, runtime_mode=runtime_mode) == "demo"


def _effective_thresholds(config: AppConfig, db: Database, category: str) -> dict[str, float | int]:
    """Compone umbrales efectivos usando config + overrides en DB."""

    base = {
        "min_delta": config.engine.min_delta,
        "min_ev_threshold": config.engine.min_ev_threshold,
        "min_time_remaining_s": config.engine.min_time_remaining_s,
        "min_volume_24h": config.engine.min_volume_24h,
        "min_contract_price": config.engine.min_contract_price,
        "max_contract_price": config.engine.max_contract_price,
    }
    override = config.engine.category_overrides.get(category.upper())
    if override is not None:
        if override.min_delta is not None:
            base["min_delta"] = override.min_delta
        if override.min_ev_threshold is not None:
            base["min_ev_threshold"] = override.min_ev_threshold
        if override.min_time_remaining_s is not None:
            base["min_time_remaining_s"] = override.min_time_remaining_s
        if override.min_contract_price is not None:
            base["min_contract_price"] = override.min_contract_price
        if override.max_contract_price is not None:
            base["max_contract_price"] = override.max_contract_price

    current_params = db.get_current_params(category)
    for key in ("min_delta", "min_ev_threshold"):
        if key in current_params:
            base[key] = float(current_params[key])
    return base


def _latest_signals_by_ticker(signals: list[Signal]) -> dict[str, Signal]:
    """Indexa la última señal conocida por ticker."""

    indexed: dict[str, Signal] = {}
    for signal in signals:
        indexed[signal.market_ticker] = signal
    return indexed


def _market_flow_payload(
    market: MarketSnapshot,
    *,
    config: AppConfig,
    db: Database,
    blocked_categories: set[str],
    latest_signal: Signal | None,
    open_tickers: set[str],
    traded_tickers: set[str],
) -> dict[str, Any]:
    """Explica por qué un mercado está listo o no para operar."""

    thresholds = _effective_thresholds(config, db, market.category)
    yes_price_ok = thresholds["min_contract_price"] <= market.yes_ask <= thresholds["max_contract_price"]
    no_price_ok = thresholds["min_contract_price"] <= market.no_ask <= thresholds["max_contract_price"]
    checks = [
        {
            "key": "category",
            "label": "Categoria habilitada",
            "status": "fail" if market.category in blocked_categories else "pass",
            "detail": (
                f"{market.category} bloqueada por performance historica"
                if market.category in blocked_categories
                else f"{market.category} habilitada"
            ),
        },
        {
            "key": "time",
            "label": "Ventana de tiempo",
            "status": "pass" if market.time_to_expiry_s >= thresholds["min_time_remaining_s"] else "fail",
            "detail": (
                f"Quedan {market.time_to_expiry_s}s, minimo {thresholds['min_time_remaining_s']}s"
            ),
        },
        {
            "key": "liquidity",
            "label": "Liquidez minima",
            "status": "pass" if market.volume_24h >= thresholds["min_volume_24h"] else "fail",
            "detail": (
                f"Volumen 24h {market.volume_24h}, minimo {thresholds['min_volume_24h']}"
            ),
        },
        {
            "key": "price_band_yes",
            "label": "Precio YES en rango",
            "status": "pass" if yes_price_ok else "fail",
            "detail": (
                f"YES ask {market.yes_ask:.2f}, rango {thresholds['min_contract_price']:.2f}-{thresholds['max_contract_price']:.2f}"
            ),
        },
        {
            "key": "price_band_no",
            "label": "Precio NO en rango",
            "status": "pass" if no_price_ok else "fail",
            "detail": (
                f"NO ask {market.no_ask:.2f}, rango {thresholds['min_contract_price']:.2f}-{thresholds['max_contract_price']:.2f}"
            ),
        },
        {
            "key": "position",
            "label": "Estado del ticker",
            "status": "warn" if market.ticker in traded_tickers else "pass",
            "detail": (
                "Ya existe posicion abierta en este ticker"
                if market.ticker in open_tickers
                else "Este ticker ya fue operado en el historial"
                if market.ticker in traded_tickers
                else "Sin operaciones previas en este ticker"
            ),
        },
    ]

    latest_signal_payload = None
    if latest_signal is not None:
        latest_signal_payload = {
            "decision": latest_signal.decision.value,
            "reasoning": latest_signal.reasoning,
            "delta": latest_signal.delta,
            "ev_net_fees": latest_signal.ev_net_fees,
            "timestamp": latest_signal.timestamp,
        }

    status = "ready"
    summary = "Listo para evaluacion del motor"
    if market.category in blocked_categories:
        status = "blocked"
        summary = "Categoria bloqueada"
    elif market.ticker in open_tickers:
        status = "active"
        summary = "Ya existe una posicion abierta"
    elif market.ticker in traded_tickers:
        status = "spent"
        summary = "Ticker ya operado; el bot evita reentrada"
    elif latest_signal is not None and latest_signal.decision in (Decision.SKIP, Decision.WAIT, Decision.ERROR):
        status = latest_signal.decision.value.lower()
        summary = latest_signal.reasoning or latest_signal.decision.value.lower()
    elif market.time_to_expiry_s < thresholds["min_time_remaining_s"]:
        status = "timing"
        summary = "Queda poco tiempo para entrar"
    elif market.volume_24h < thresholds["min_volume_24h"]:
        status = "liquidity"
        summary = "Liquidez insuficiente"
    elif not (yes_price_ok or no_price_ok):
        status = "pricing"
        summary = "Ningun lado del contrato esta en el rango de precios permitido"

    return {
        "status": status,
        "summary": summary,
        "thresholds": thresholds,
        "latest_signal": latest_signal_payload,
        "checks": checks,
    }


async def _fetch_market(config: AppConfig, ticker: str) -> MarketSnapshot:
    """Busca un mercado activo por ticker."""

    feed = KalshiFeed(config, EventBus())
    markets = await feed.get_active_markets()
    for market in markets:
        if market.ticker == ticker:
            return market
    raise HTTPException(status_code=404, detail=f"Mercado no encontrado: {ticker}")


def _skip_reason_summary(signals: list[Signal]) -> list[dict[str, Any]]:
    """Agrega razones frecuentes de no entrada para el dashboard."""

    counts: dict[str, int] = {}
    for signal in signals:
        if signal.decision in (Decision.YES, Decision.NO):
            continue
        counts[signal.reasoning] = counts.get(signal.reasoning, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"reason": reason, "count": count} for reason, count in ranked[:6]]


def create_app(
    db: Database | None = None,
    config: AppConfig | None = None,
    runtime_mode: str | None = None,
) -> FastAPI:
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
    app.state.runtime_mode = runtime_mode
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
        period_bounds = _period_bounds(now)
        recent_signals = app.state.db.get_signals(
            from_ts=max(0.0, now - 12 * 3600),
            to_ts=now,
            limit=max(safe_limit, 100),
        )
        open_trades = app.state.db.get_open_trades()
        closed_trades = app.state.db.get_closed_trades(limit=safe_limit)
        pnl_summary = app.state.db.get_trade_pnl_summary(mode=TradeMode.DEMO)
        period_pnl = {
            name: {
                "start": start_ts,
                "end": end_ts,
                "label": label,
                "realized_pnl": app.state.db.get_realized_pnl_between(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    mode=TradeMode.DEMO,
                ),
            }
            for name, (start_ts, end_ts, label) in period_bounds.items()
        }
        blocked_categories = sorted(app.state.db.get_blocked_categories())
        current_params = app.state.db.get_current_params()
        signal_decisions: dict[str, int] = {}
        for signal in recent_signals:
            signal_decisions[signal.decision.value] = signal_decisions.get(signal.decision.value, 0) + 1
        return {
            "timestamp": now,
            "effective_mode": _effective_execution_mode(app.state.config, app.state.runtime_mode),
            "manual_actions_enabled": _manual_actions_enabled(app.state.config, app.state.runtime_mode),
            "open_trades_count": len(open_trades),
            "closed_trades_count": int(pnl_summary["closed_trades"]),
            "realized_pnl": float(pnl_summary["realized_pnl"]),
            "unrealized_pnl": 0.0,
            "total_pnl": float(pnl_summary["realized_pnl"]),
            "paper_account_negative": float(pnl_summary["realized_pnl"]) < 0.0,
            "open_trades": [_trade_payload(trade) for trade in open_trades],
            "recent_closed_trades": [_trade_payload(trade) for trade in reversed(closed_trades)],
            "period_pnl": period_pnl,
            "blocked_categories": blocked_categories,
            "current_params": current_params,
            "signal_decisions": signal_decisions,
            "skip_reason_summary": _skip_reason_summary(recent_signals),
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
        recent_signals = app.state.db.get_signals(
            from_ts=max(0.0, time.time() - 12 * 3600),
            to_ts=time.time(),
            limit=2_000,
        )
        latest_by_ticker = _latest_signals_by_ticker(recent_signals)
        open_tickers = {trade.ticker for trade in app.state.db.get_open_trades()}
        traded_tickers = {trade.ticker for trade in app.state.db.get_closed_trades(limit=10_000)} | open_tickers
        blocked_categories = app.state.db.get_blocked_categories()
        return {
            "timestamp": time.time(),
            "latency_ms": round((time.time() - started) * 1000),
            "count": len(markets[:safe_limit]),
            "markets": [
                {
                    **_market_payload(market),
                    "flow": _market_flow_payload(
                        market,
                        config=app.state.config,
                        db=app.state.db,
                        blocked_categories=blocked_categories,
                        latest_signal=latest_by_ticker.get(market.ticker),
                        open_tickers=open_tickers,
                        traded_tickers=traded_tickers,
                    ),
                }
                for market in markets[:safe_limit]
            ],
        }

    @app.post("/manual/paper-order")
    async def manual_paper_order(payload: ManualPaperOrderRequest) -> dict[str, Any]:
        if _effective_execution_mode(app.state.config, app.state.runtime_mode) != "demo":
            raise HTTPException(
                status_code=409,
                detail="Las ordenes manuales desde dashboard solo estan habilitadas en paper/demo mode.",
            )
        side = payload.side.upper()
        if side not in {"YES", "NO"}:
            raise HTTPException(status_code=400, detail="side debe ser YES o NO")
        if app.state.db.has_trade_for_ticker(payload.ticker, mode=TradeMode.DEMO):
            raise HTTPException(status_code=409, detail="Ese ticker ya fue operado en paper mode.")

        market = await _fetch_market(app.state.config, payload.ticker)
        market_probability = market.yes_ask if side == "YES" else (1.0 - market.no_ask)
        signal = Signal(
            market_ticker=market.ticker,
            decision=Decision(side),
            my_probability=market_probability,
            market_probability=market_probability,
            delta=0.0,
            ev_net_fees=0.0,
            kelly_size=min(1.0, payload.contracts / 100.0),
            confidence=Confidence.LOW,
            time_remaining_s=market.time_to_expiry_s,
            reasoning=f"manual_dashboard:{payload.note or 'paper_entry'}",
            timestamp=time.time(),
        )
        executor = PaperOrderExecutor(app.state.db, mode=TradeMode.DEMO)
        trade = await executor.submit(signal)
        return {
            "status": "ok",
            "message": f"Orden paper creada para {trade.ticker} {trade.side}",
            "trade": _trade_payload(trade),
        }

    @app.get("/analytics/execution-quality")
    async def execution_quality(limit: int = 500) -> dict[str, Any]:
        """
        Resumen de calidad de ejecución por categoría, overround y delta.

        Basado en señales resueltas con trades cerrados. Solo datos locales.
        Incluye sugerencia de calibración para max_market_overround_bps.

        Args:
            limit: número máximo de registros a analizar (más recientes).
        """
        safe_limit = max(1, min(limit, 5_000))
        analyzer = ExecutionQualityAnalyzer(db=app.state.db)
        report = analyzer.analyze(limit=safe_limit)

        def _bucket_dict(stats: BucketStats) -> dict[str, Any]:
            return {
                "label": stats.label,
                "sample_size": stats.sample_size,
                "win_rate": round(stats.win_rate, 4),
                "total_pnl": round(stats.total_pnl, 4),
                "avg_pnl": round(stats.avg_pnl, 4),
                "avg_contract_price": round(stats.avg_contract_price, 4),
                "avg_overround_bps": round(stats.avg_overround_bps, 2),
                "avg_delta": round(stats.avg_delta, 4),
                "avg_entry_edge_bps": round(stats.avg_entry_edge_bps, 2),
            }

        return {
            "total_resolved": report.total_resolved,
            "overall_win_rate": round(report.overall_win_rate, 4),
            "overall_pnl": round(report.overall_pnl, 4),
            "suggested_max_overround_bps": report.suggested_max_overround_bps,
            "by_category": {k: _bucket_dict(v) for k, v in report.by_category.items()},
            "by_overround_bucket": {
                k: _bucket_dict(v) for k, v in report.by_overround_bucket.items()
            },
            "by_delta_bucket": {
                k: _bucket_dict(v) for k, v in report.by_delta_bucket.items()
            },
        }

    @app.post("/manual/paper-close")
    async def manual_paper_close(payload: ManualPaperCloseRequest) -> dict[str, Any]:
        if _effective_execution_mode(app.state.config, app.state.runtime_mode) != "demo":
            raise HTTPException(
                status_code=409,
                detail="Los cierres manuales desde dashboard solo estan habilitados en paper/demo mode.",
            )
        trade = app.state.db.get_trade_by_id(payload.trade_id)
        if trade is None:
            raise HTTPException(status_code=404, detail=f"Trade no encontrado: {payload.trade_id}")
        if trade.status != TradeStatus.OPEN:
            raise HTTPException(status_code=409, detail="El trade ya no esta abierto.")

        market = await _fetch_market(app.state.config, trade.ticker)
        exit_price = market.implied_prob if trade.side == "YES" else max(0.01, min(0.99, 1.0 - market.implied_prob))
        executor = PaperOrderExecutor(app.state.db, mode=TradeMode.DEMO)
        closed = await executor.close_with_price(trade=trade, exit_price=exit_price)
        return {
            "status": "ok",
            "message": f"Trade cerrado en paper mode para {closed.ticker}",
            "trade": _trade_payload(closed),
            "note": payload.note,
        }

    return app
