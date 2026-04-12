"""
main.py — Orquestador principal de Kalshi crypto agents.

Modos:
  python main.py                    # loop de trading usando ENV del .env
  python main.py --dry-run          # fuerza demo aunque ENV=production
  python main.py --backtest-only    # solo recalibra y sale
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

import uvicorn

from backtesting.backtest_runner import BacktestRunner
from backtesting.category_blocker import CategoryBlocker
from backtesting.outcome_resolver import OutcomeResolver
from backtesting.param_injector import ParamInjector
from core.config import AppConfig, load_config
from core.database import Database
from core.interfaces import EventBus
from core.models import PriceSnapshot
from engine.ev_calculator import EVCalculator
from engine.price_resolver import resolve_reference_price
from engine.probability import ProbabilityEngine, classify_time_zone
from engine.signal_router import SignalRouter
from engine.timing import TimingFilter
from execution.order_executor import PaperOrderExecutor
from execution.position_manager import PositionManager
from feeds.binance_feed import BinancePriceFeed
from feeds.hyperliquid_feed import HyperliquidFeed
from feeds.kalshi_feed import KalshiFeed
from intelligence.reddit_provider import RedditSocialSentimentProvider
from intelligence.social_sentiment import SocialSentimentService
from memory.openclaw_adapter import OpenClawMemoryAdapter

logger = logging.getLogger(__name__)

# ── Constantes operativas ──────────────────────────────────────────────────────
BANKROLL_DEFAULT: float = 1000.0
SIGNALS_LOOKBACK_S: float = 48 * 3600   # 48h para backtest pre-arranque
SUPERVISOR_INTERVAL_S: int = 60          # cada 60s evalúa SL/TP
RECALIBRATE_INTERVAL_S: int = 86400      # cada 24h recalibra
EXPIRY_CLOSE_S: int = 120                # cierra posición si quedan ≤ 120s
REST_POLL_INTERVAL_S: int = 90           # polling REST de mercados Kalshi
OUTCOME_RESOLVE_INTERVAL_S: int = 300    # resolución de outcomes cada 5 min
OUTCOME_CALIBRATE_THRESHOLD: int = 20    # recalibrar si ≥ 20 outcomes nuevos
DASHBOARD_PORT: int = 8090


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parsea argumentos CLI."""
    parser = argparse.ArgumentParser(description="Kalshi crypto agents — orquestador principal")

    # Nuevo estilo
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Modo demo completo: fuerza ENV=demo (API demo + paper trading). Útil para pruebas.",
    )
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        default=False,
        help=(
            "Paper trading con API de producción: usa datos reales de Kalshi "
            "pero no envía órdenes reales. ENV no cambia."
        ),
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        default=False,
        help="Solo recalibra parámetros y sale",
    )
    parser.add_argument("--config", default="config.json", help="Ruta a config.json")
    parser.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Capital USD (default: env BANKROLL_USD o 1000.0)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Máximo de posiciones abiertas simultáneamente",
    )

    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _resolve_bankroll(args_bankroll: float | None) -> float:
    """Lee BANKROLL_USD del env o usa el default."""
    if args_bankroll is not None:
        return args_bankroll
    env_val = os.getenv("BANKROLL_USD")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            logger.warning("BANKROLL_USD inválido ('%s'), usando %.0f", env_val, BANKROLL_DEFAULT)
    return BANKROLL_DEFAULT


def _maybe_recalibrate(cfg: AppConfig, db: Database, bankroll: float) -> None:
    """
    Recalibra parámetros y categorías si hay señales en las últimas 48h.

    Corre BacktestRunner → ParamInjector → CategoryBlocker en secuencia.
    No propaga excepciones — el bot arranca igual si falla.
    """
    now = time.time()
    try:
        signals = db.get_signals(from_ts=now - SIGNALS_LOOKBACK_S, to_ts=now, limit=1)
        if not signals:
            logger.info("recalibrate_skip reason=no_recent_signals")
            return

        logger.info("recalibrate_start signals_in_48h=True")
        from_ts = now - SIGNALS_LOOKBACK_S

        runner = BacktestRunner(db=db, initial_bankroll=bankroll, config=cfg.engine)
        result = runner.run(from_ts=from_ts, to_ts=now)
        logger.info(
            "backtest_complete win_rate=%.3f signals=%d pnl=%.4f",
            result.win_rate,
            result.actionable_signals,
            result.total_pnl,
        )

        injector = ParamInjector(db=db)
        calibrations = injector.calibrate(from_ts=from_ts, to_ts=now)
        logger.info("param_injector calibrations=%d", len(calibrations))

        blocker = CategoryBlocker(db=db, runner=runner)
        decisions = blocker.evaluate_and_apply(from_ts=from_ts, to_ts=now)
        blocked = [d.category for d in decisions if d.blocked]
        logger.info("category_blocker decisions=%d blocked=%s", len(decisions), blocked)

    except Exception as exc:  # noqa: BLE001
        logger.warning("recalibrate_error exc=%s — continuando arranque", exc)


def _build_social_sentiment_service(cfg: AppConfig) -> SocialSentimentService | None:
    """Construye el servicio opcional de sentimiento social cacheado."""
    if not cfg.social_sentiment.enabled:
        logger.info("social_sentiment disabled")
        return None

    provider_name = cfg.social_sentiment.provider
    if provider_name != "reddit":
        logger.warning("social_sentiment provider=%s not supported", provider_name)
        return None

    service = SocialSentimentService(
        config=cfg.social_sentiment,
        provider=RedditSocialSentimentProvider(config=cfg.social_sentiment),
    )
    logger.info("social_sentiment enabled provider=%s", provider_name)
    return service


def _build_router(
    cfg: AppConfig,
    db: Database,
    social_sentiment_service: SocialSentimentService | None = None,
) -> SignalRouter:
    """Construye el SignalRouter, con OpenRouter si la key está disponible."""
    openrouter_agent = None
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        from engine.openrouter_agent import OpenRouterAgent
        openrouter_agent = OpenRouterAgent(api_key=openrouter_key)
        logger.info("openrouter_agent enabled")
    else:
        logger.info("openrouter_agent disabled (OPENROUTER_API_KEY not set)")

    return SignalRouter(
        prob_engine=ProbabilityEngine(),
        ev_calc=EVCalculator(),
        timing_filter=TimingFilter(),
        config=cfg.engine,
        db=db,
        blocked_categories=set(db.get_blocked_categories()),
        openrouter_agent=openrouter_agent,
        social_sentiment_service=social_sentiment_service,
    )


def _build_memory_adapter() -> OpenClawMemoryAdapter | None:
    """Construye un adapter opcional hacia el workspace de OpenClaw."""

    workspace = os.getenv("OPENCLAW_WORKSPACE", "").strip()
    if not workspace:
        logger.info("openclaw_memory disabled (OPENCLAW_WORKSPACE not set)")
        return None
    adapter = OpenClawMemoryAdapter(workspace=Path(workspace).expanduser())
    adapter.initialize()
    logger.info("openclaw_memory enabled workspace=%s", adapter.workspace)
    return adapter


def _resolve_execution_mode(cfg: AppConfig, paper_trade: bool) -> str:
    """Resuelve el modo efectivo del executor."""

    return "demo" if (paper_trade or cfg.is_demo) else "production"


# ── Dashboard en background ────────────────────────────────────────────────────

async def _serve_dashboard(
    db: Database,
    cfg: AppConfig,
    runtime_mode: str,
    port: int = DASHBOARD_PORT,
) -> None:
    """Arranca el servidor FastAPI en background como tarea asyncio."""
    from dashboard.api_server import create_app
    app = create_app(db=db, config=cfg, runtime_mode=runtime_mode)
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        loop="none",         # usamos el loop de asyncio existente
    )
    server = uvicorn.Server(config)
    logger.info("dashboard_starting port=%d", port)
    await server.serve()


# ── Loop principal ─────────────────────────────────────────────────────────────

async def _run_orchestrator(
    cfg: AppConfig,
    db: Database,
    bankroll: float,
    max_positions: int = 3,
    paper_trade: bool = True,
) -> None:
    """
    Loop principal async del bot de trading.

    Tareas concurrentes:
      - _price_task:      actualiza caché de precios BTC/ETH desde Binance
      - _market_task:     procesa stream WS de Kalshi → señales → trades
      - _rest_poll_task:  polling REST cada 90s como respaldo del WS
      - _supervisor_task: cada 60s → SL/TP + log de estado
      - _recal_task:      cada 24h → BacktestRunner + ParamInjector + CategoryBlocker
      - _serve_dashboard: FastAPI en background en :8090

    Args:
        paper_trade: si True, los trades se registran en DB pero no se envían
                     órdenes reales a Kalshi (paper trading). Si False y
                     cfg.is_production, se requiere un cliente real de Kalshi.
    """
    social_sentiment_service = _build_social_sentiment_service(cfg)
    router = _build_router(cfg, db, social_sentiment_service=social_sentiment_service)

    # Determinar modo de ejecución:
    # - paper_trade=True  → siempre demo executor (sin órdenes reales)
    # - paper_trade=False y production → production executor (requiere client real)
    # - demo env → siempre paper
    exec_mode = _resolve_execution_mode(cfg=cfg, paper_trade=paper_trade)
    executor = PaperOrderExecutor(db=db, mode=exec_mode)
    logger.info("executor_mode=%s api_env=%s", exec_mode, cfg.env)
    pos_mgr = PositionManager(
        db=db,
        executor=executor,
        initial_bankroll=bankroll,
        min_closed_trades=5,
        min_win_rate=0.35,
        min_total_pnl=-(bankroll * 0.15),
        max_drawdown_pct=0.50,
    )
    await pos_mgr.hydrate_from_db()

    bus = EventBus()
    bfeed = BinancePriceFeed(cfg.binance, bus)
    hfeed = HyperliquidFeed(cfg.hyperliquid, bus)
    kfeed = KalshiFeed(cfg, bus)
    memory_adapter = _build_memory_adapter()

    latest_prices: dict[str, dict[str, PriceSnapshot]] = {}

    shutdown_event = asyncio.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("shutdown_requested signum=%d", signum)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "orchestrator_start env=%s bankroll=%.0f max_positions=%d",
        cfg.env,
        bankroll,
        max_positions,
    )

    await bfeed.connect()
    await hfeed.connect()
    await kfeed.connect()

    resolver = OutcomeResolver(db=db, kalshi_client=kfeed)
    if social_sentiment_service is not None:
        await social_sentiment_service.start()
    if memory_adapter is not None:
        snap = pos_mgr.observability_snapshot()
        memory_adapter.record_session_start(
            mode=exec_mode,
            bankroll=bankroll,
            total_pnl=float(snap["total_pnl"]),
            go_allowed=pos_mgr.go_no_go_status().allowed,
        )

    tasks: list[asyncio.Task] = []
    try:
        tasks = [
            asyncio.create_task(_price_task(bfeed, latest_prices), name="price_binance"),
            asyncio.create_task(_price_task(hfeed, latest_prices), name="price_hyperliquid"),
            asyncio.create_task(
                _market_task(kfeed, router, pos_mgr, latest_prices, bankroll, max_positions, memory_adapter),
                name="market",
            ),
            asyncio.create_task(
                _rest_poll_task(kfeed, router, pos_mgr, latest_prices, bankroll, max_positions, memory_adapter),
                name="rest_poll",
            ),
            asyncio.create_task(
                _supervisor_task(pos_mgr, db),
                name="supervisor",
            ),
            asyncio.create_task(
                _recal_task(cfg, db, router, bankroll),
                name="recal",
            ),
            asyncio.create_task(
                _outcome_task(cfg, resolver, db, router, bankroll),
                name="outcome_resolver",
            ),
            asyncio.create_task(
                _serve_dashboard(db, cfg=cfg, runtime_mode=exec_mode, port=DASHBOARD_PORT),
                name="dashboard",
            ),
        ]

        await shutdown_event.wait()
        logger.info("shutdown_graceful cancelling %d tasks", len(tasks))

    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        for feed in (bfeed, hfeed, kfeed):
            try:
                await feed.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug("feed_disconnect_error: %s", exc)
        if social_sentiment_service is not None:
            await social_sentiment_service.stop()

        snap = pos_mgr.observability_snapshot()
        logger.info(
            "shutdown_summary open=%d closed=%d win_rate=%.3f pnl=%.4f fees=%.4f",
            snap["open_positions"],
            snap["closed_positions"],
            snap["win_rate"],
            snap["total_pnl"],
            snap["total_fees"],
        )
        db.close()
        logger.info("orchestrator_stopped")


# ── Tasks internas ─────────────────────────────────────────────────────────────

async def _price_task(
    price_feed,
    latest_prices: dict[str, dict[str, PriceSnapshot]],
) -> None:
    """Actualiza la caché de precios spot continuamente."""
    async for snap in price_feed.stream():
        latest_prices.setdefault(snap.symbol, {})[snap.source] = snap


async def _market_task(
    kfeed: KalshiFeed,
    router: SignalRouter,
    pos_mgr: PositionManager,
    latest_prices: dict[str, dict[str, PriceSnapshot]],
    bankroll: float,
    max_positions: int,
    memory_adapter: OpenClawMemoryAdapter | None = None,
) -> None:
    """Procesa cada MarketSnapshot del stream WS de Kalshi."""
    async for market in kfeed.stream_markets():
        await _process_market(
            market,
            router,
            pos_mgr,
            latest_prices,
            bankroll,
            max_positions,
            memory_adapter,
        )


def _log_decision(**kwargs: object) -> None:
    """Emite un registro de decisión estructurado en JSON para auditoría."""
    logger.info("decision_record %s", json.dumps(kwargs, default=str))


async def _process_market(
    market,
    router: SignalRouter,
    pos_mgr: PositionManager,
    latest_prices: dict[str, dict[str, PriceSnapshot]],
    bankroll: float,
    max_positions: int,
    memory_adapter: OpenClawMemoryAdapter | None = None,
) -> None:
    """Lógica compartida de evaluación para un MarketSnapshot (WS o REST)."""
    ticker = market.ticker

    if market.time_to_expiry_s <= EXPIRY_CLOSE_S:
        for _trade_id, trade in list(pos_mgr.open_positions.items()):
            if trade.ticker == ticker:
                exit_price = (
                    market.implied_prob
                    if trade.side == "YES"
                    else max(0.01, min(0.99, 1.0 - market.implied_prob))
                )
                managed_close = await pos_mgr.close_trade(trade, exit_price, "expiry")
                logger.info(
                    "expiry_close ticker=%s side=%s exit=%.4f",
                    ticker, trade.side, exit_price,
                )
                if memory_adapter is not None:
                    memory_adapter.record_trade_close(trade=managed_close.trade, reason="expiry")
        return

    closes = await pos_mgr.evaluate_price(
        ticker=ticker,
        current_yes_price=market.implied_prob,
        time_remaining_s=market.time_to_expiry_s,
    )
    for mc in closes:
        logger.info(
            "position_closed ticker=%s pnl=%.4f reason=%s",
            ticker, mc.trade.pnl or 0.0, mc.reason,
        )
        if memory_adapter is not None:
            memory_adapter.record_trade_close(trade=mc.trade, reason=mc.reason)

    if pos_mgr.has_traded_ticker(ticker) or len(pos_mgr.open_positions) >= max_positions:
        return

    resolution = resolve_reference_price(
        symbol=market.category,
        latest_prices=latest_prices,
        now_ts=market.timestamp,
    )
    if resolution.snapshot is None:
        if resolution.blocked_reason is not None:
            logger.info(
                "price_resolution_blocked ticker=%s reason=%s spread_pct=%s",
                ticker,
                resolution.blocked_reason,
                f"{resolution.spread_pct:.6f}" if resolution.spread_pct is not None else "n/a",
            )
        return
    price = resolution.snapshot
    if resolution.blocked_reason is not None:
        return

    signal = await router.evaluate_async(market=market, price=price, bankroll=bankroll)

    if signal.is_actionable:
        status = pos_mgr.go_no_go_status(max_open_positions=max_positions, category=market.category)
        if not status.allowed:
            _log_decision(
                event="trade_blocked",
                ticker=ticker,
                decision=signal.decision.value,
                reason=status.reason,
                go=False,
                delta=signal.delta,
                ev_net=signal.ev_net_fees,
            )
            if memory_adapter is not None:
                memory_adapter.record_trade_blocked(ticker=ticker, reason=status.reason)
            return
        trade = await pos_mgr.try_open_from_signal(signal, max_positions=max_positions)
        if trade is None:
            return
        zone = classify_time_zone(market.time_to_expiry_s)
        _log_decision(
            event="trade_opened",
            ticker=trade.ticker,
            decision=trade.side,
            reason="signal_actionable",
            go=True,
            delta=signal.delta,
            ev_net=signal.ev_net_fees,
            contracts=trade.contracts,
            entry_price=trade.entry_price,
            zone=zone,
        )
        if memory_adapter is not None:
            memory_adapter.record_trade_open(trade, signal, price.source)
    else:
        logger.debug(
            "signal_skip ticker=%s decision=%s reason=%s",
            ticker, signal.decision.value, signal.reasoning,
        )


async def _rest_poll_task(
    kfeed: KalshiFeed,
    router: SignalRouter,
    pos_mgr: PositionManager,
    latest_prices: dict[str, dict[str, PriceSnapshot]],
    bankroll: float,
    max_positions: int,
    memory_adapter: OpenClawMemoryAdapter | None = None,
) -> None:
    """Polling REST periódico para no depender exclusivamente del WS."""
    await asyncio.sleep(5)   # espera inicial para que los precios estén disponibles
    while True:
        try:
            markets = await kfeed.get_active_markets()
            logger.info("rest_poll markets=%d", len(markets))
            for market in markets:
                await _process_market(
                    market,
                    router,
                    pos_mgr,
                    latest_prices,
                    bankroll,
                    max_positions,
                    memory_adapter,
                )
        except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning("rest_poll_error: %s", exc)
        await asyncio.sleep(REST_POLL_INTERVAL_S)


async def _supervisor_task(pos_mgr: PositionManager, db: Database) -> None:
    """Log periódico de estado y sincronización desde DB."""
    while True:
        await asyncio.sleep(SUPERVISOR_INTERVAL_S)
        pos_mgr.sync_open_trades()
        snap = pos_mgr.observability_snapshot()
        gng = pos_mgr.go_no_go_status()
        logger.info(
            "supervisor open=%d closed=%d win_rate=%.3f pnl=%.4f go=%s",
            snap["open_positions"],
            snap["closed_positions"],
            snap["win_rate"],
            snap["total_pnl"],
            gng.allowed,
        )


async def _recal_task(cfg: AppConfig, db: Database, router: SignalRouter, bankroll: float) -> None:
    """Recalibra parámetros y bloqueos cada 24h."""
    while True:
        await asyncio.sleep(RECALIBRATE_INTERVAL_S)
        logger.info("recal_task_start")
        _maybe_recalibrate(cfg, db, bankroll)
        # Recargar categorías bloqueadas en el router
        router.blocked_categories = set(db.get_blocked_categories())
        logger.info("recal_task_done blocked=%s", router.blocked_categories)


async def _outcome_task(
    cfg: AppConfig,
    resolver: OutcomeResolver,
    db: Database,
    router: SignalRouter,
    bankroll: float,
) -> None:
    """Resuelve outcomes de señales expiradas cada 5 minutos."""
    accumulated_new = 0
    logger.info("outcome_resolver initialized")
    while True:
        await asyncio.sleep(OUTCOME_RESOLVE_INTERVAL_S)
        try:
            n = await resolver.resolve_expired()
            logger.info("outcome_resolver: resolved %d outcomes this cycle", n)
            if n > 0:
                accumulated_new += n
                if accumulated_new >= OUTCOME_CALIBRATE_THRESHOLD:
                    accumulated_new = 0
                    logger.info(
                        "outcome_resolver: triggering recalibration (%d new outcomes)",
                        n,
                    )
                    _maybe_recalibrate(cfg, db, bankroll)
                    router.blocked_categories = set(db.get_blocked_categories())
        except (ConnectionError, RuntimeError) as exc:
            logger.warning("outcome_resolver: error en ciclo: %s", exc)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()

    # ── Orquestador principal ─────────────────────────────────────────────────
    # --dry-run = demo completo (API demo + paper).
    # --paper-trade = API producción + paper (no cambia ENV).
    if args.dry_run:
        os.environ["ENV"] = "demo"

    paper_trade = args.dry_run or args.paper_trade or (
        os.getenv("PAPER_TRADE", "").lower() in ("1", "true", "yes")
    )

    cfg = load_config(args.config)
    _setup_logging(cfg.log_level)
    bankroll = _resolve_bankroll(args.bankroll)

    db = Database(
        path=cfg.database.path,
        wal_mode=cfg.database.wal_mode,
        busy_timeout_ms=cfg.database.busy_timeout_ms,
    )
    db.initialize()

    # Cargar blocked_categories y current_params desde DB
    blocked = db.get_blocked_categories()
    params = db.get_current_params()
    logger.info(
        "state_loaded blocked_categories=%s current_params=%s",
        sorted(blocked),
        list(params.keys()),
    )

    # Pre-arranque: recalibrar si hay señales recientes
    _maybe_recalibrate(cfg, db, bankroll)

    if args.backtest_only:
        logger.info("backtest_only mode — exiting after recalibration")
        db.close()
        return

    logger.info(
        "orchestrator_init env=%s bankroll=%.0f dry_run=%s paper_trade=%s",
        cfg.env,
        bankroll,
        args.dry_run,
        paper_trade,
    )
    asyncio.run(_run_orchestrator(
        cfg=cfg,
        db=db,
        bankroll=bankroll,
        max_positions=args.max_positions,
        paper_trade=paper_trade,
    ))


if __name__ == "__main__":
    main()
