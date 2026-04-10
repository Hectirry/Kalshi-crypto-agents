"""
main.py — Orquestador principal de Kalshi crypto agents.

Modos:
  python main.py                    # loop de trading usando ENV del .env
  python main.py --dry-run          # fuerza demo aunque ENV=production
  python main.py --backtest-only    # solo recalibra y sale

Retrocompatibilidad con scripts:
  python main.py --mode smoke       # E2E smoke test
  python main.py --mode loop        # loop de smoke tests
  python main.py --mode trading     # trading demo vía script
  python main.py --mode dashboard   # dashboard terminal
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time

import uvicorn

from backtesting.backtest_runner import BacktestRunner
from backtesting.category_blocker import CategoryBlocker
from backtesting.param_injector import ParamInjector
from core.config import AppConfig, load_config
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

# ── Constantes operativas ──────────────────────────────────────────────────────
BANKROLL_DEFAULT: float = 100.0
SIGNALS_LOOKBACK_S: float = 48 * 3600   # 48h para backtest pre-arranque
SUPERVISOR_INTERVAL_S: int = 60          # cada 60s evalúa SL/TP
RECALIBRATE_INTERVAL_S: int = 86400      # cada 24h recalibra
EXPIRY_CLOSE_S: int = 120                # cierra posición si quedan ≤ 120s
REST_POLL_INTERVAL_S: int = 90           # polling REST de mercados Kalshi
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
        help="Capital USD (default: env BANKROLL_USD o 100.0)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=3,
        help="Máximo de posiciones abiertas simultáneamente",
    )

    # Retrocompatibilidad
    parser.add_argument(
        "--mode",
        choices=("smoke", "loop", "dashboard", "trading"),
        default=None,
        help="Modo legacy (retrocompatibilidad con scripts/)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Intervalo en segundos para --mode loop",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh en segundos para --mode dashboard",
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


def _maybe_recalibrate(db: Database, bankroll: float) -> None:
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

        runner = BacktestRunner(db=db, initial_bankroll=bankroll)
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


def _build_router(cfg: AppConfig, db: Database) -> SignalRouter:
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
    )


# ── Dashboard en background ────────────────────────────────────────────────────

async def _serve_dashboard(db: Database, port: int = DASHBOARD_PORT) -> None:
    """Arranca el servidor FastAPI en background como tarea asyncio."""
    from dashboard.api_server import create_app
    app = create_app(db=db)
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
    router = _build_router(cfg, db)

    # Determinar modo de ejecución:
    # - paper_trade=True  → siempre demo executor (sin órdenes reales)
    # - paper_trade=False y production → production executor (requiere client real)
    # - demo env → siempre paper
    exec_mode = "demo" if (paper_trade or cfg.is_demo) else "production"
    executor = PaperOrderExecutor(db=db, mode=exec_mode)
    logger.info("executor_mode=%s api_env=%s", exec_mode, cfg.env)
    pos_mgr = PositionManager(db=db, executor=executor)
    pos_mgr.sync_open_trades()

    bus = EventBus()
    bfeed = BinancePriceFeed(cfg.binance, bus)
    kfeed = KalshiFeed(cfg, bus)

    latest_prices: dict[str, PriceSnapshot] = {}

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
    await kfeed.connect()

    tasks: list[asyncio.Task] = []
    try:
        tasks = [
            asyncio.create_task(_price_task(bfeed, latest_prices), name="price"),
            asyncio.create_task(
                _market_task(kfeed, router, pos_mgr, latest_prices, bankroll, max_positions),
                name="market",
            ),
            asyncio.create_task(
                _rest_poll_task(kfeed, router, pos_mgr, latest_prices, bankroll, max_positions),
                name="rest_poll",
            ),
            asyncio.create_task(
                _supervisor_task(pos_mgr, db),
                name="supervisor",
            ),
            asyncio.create_task(
                _recal_task(db, router, bankroll),
                name="recal",
            ),
            asyncio.create_task(
                _serve_dashboard(db, port=DASHBOARD_PORT),
                name="dashboard",
            ),
        ]

        await shutdown_event.wait()
        logger.info("shutdown_graceful cancelling %d tasks", len(tasks))

    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        for feed in (bfeed, kfeed):
            try:
                await feed.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug("feed_disconnect_error: %s", exc)

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
                await pos_mgr.executor.close_with_price(trade, exit_price)
                del pos_mgr.open_positions[trade_id]
                logger.info(
                    "expiry_close ticker=%s side=%s exit=%.4f",
                    ticker, trade.side, exit_price,
                )
        return

    closes = await pos_mgr.evaluate_price(
        ticker=ticker,
        current_yes_price=market.implied_prob,
    )
    for mc in closes:
        logger.info(
            "position_closed ticker=%s pnl=%.4f reason=%s",
            ticker, mc.trade.pnl or 0.0, mc.reason,
        )

    if pos_mgr.has_open_ticker(ticker) or len(pos_mgr.open_positions) >= max_positions:
        return

    price = latest_prices.get(market.category)
    if price is None:
        return

    signal = await router.evaluate_async(market=market, price=price, bankroll=bankroll)

    if signal.is_actionable:
        trade = await pos_mgr.try_open_from_signal(signal, max_positions=max_positions)
        if trade is None:
            return
        zone = classify_time_zone(market.time_to_expiry_s)
        logger.info(
            "trade_opened ticker=%s side=%s contracts=%d entry=%.4f delta=%.4f zone=%s",
            trade.ticker, trade.side, trade.contracts, trade.entry_price,
            signal.delta, zone,
        )
    else:
        logger.debug(
            "signal_skip ticker=%s decision=%s reason=%s",
            ticker, signal.decision.value, signal.reasoning,
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
    await asyncio.sleep(5)   # espera inicial para que los precios estén disponibles
    while True:
        try:
            markets = await kfeed.get_active_markets()
            logger.info("rest_poll markets=%d", len(markets))
            for market in markets:
                await _process_market(market, router, pos_mgr, latest_prices, bankroll, max_positions)
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


async def _recal_task(db: Database, router: SignalRouter, bankroll: float) -> None:
    """Recalibra parámetros y bloqueos cada 24h."""
    while True:
        await asyncio.sleep(RECALIBRATE_INTERVAL_S)
        logger.info("recal_task_start")
        _maybe_recalibrate(db, bankroll)
        # Recargar categorías bloqueadas en el router
        router.blocked_categories = set(db.get_blocked_categories())
        logger.info("recal_task_done blocked=%s", router.blocked_categories)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()

    # ── Retrocompatibilidad con --mode ────────────────────────────────────────
    if args.mode is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        if args.mode == "smoke":
            from scripts.e2e_smoke import run_e2e
            asyncio.run(run_e2e())
            return
        if args.mode == "loop":
            from scripts.e2e_smoke import run_e2e

            async def _loop() -> None:
                while True:
                    try:
                        await run_e2e()
                    except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
                        logger.warning("E2E_LOOP_ERROR: %s", exc)
                    await asyncio.sleep(max(2, args.interval))

            asyncio.run(_loop())
            return
        if args.mode == "trading":
            from scripts.trading_loop import run_trading_demo
            bankroll = _resolve_bankroll(args.bankroll)
            asyncio.run(run_trading_demo(
                config_path=args.config,
                bankroll=bankroll,
                max_positions=args.max_positions,
            ))
            return
        if args.mode == "dashboard":
            from scripts.terminal_dashboard import run_dashboard
            asyncio.run(run_dashboard(config_path=args.config, refresh_s=args.refresh))
            return

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
    _maybe_recalibrate(db, bankroll)

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
