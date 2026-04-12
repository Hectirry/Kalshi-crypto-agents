"""
execution/position_manager.py

Gestión de posiciones abiertas, stop loss, take profit y observabilidad.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from core.database import Database
from core.models import Signal, Trade, TradeMode, TradeStatus
from execution.order_executor import PaperOrderExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ManagedClose:
    """Resultado de un cierre gatillado por reglas de gestión."""

    trade: Trade
    reason: str


@dataclass(frozen=True, slots=True)
class GoNoGoStatus:
    """Estado agregado para decidir si seguir operando."""

    allowed: bool
    reason: str
    closed_trades: int
    win_rate: float
    total_pnl: float


class PositionManager:
    """Administra trades abiertos y aplica reglas de salida."""

    def __init__(
        self,
        db: Database,
        executor: PaperOrderExecutor,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.15,
        initial_bankroll: float = 1000.0,
        min_closed_trades: int = 5,
        min_win_rate: float = 0.35,
        min_total_pnl: float = 0.0,
        max_drawdown_pct: float = 0.50,
        time_exit_threshold_s: int = 60,
        time_exit_profit_pct: float = 0.08,
    ) -> None:
        """
        Inicializa el manager de posiciones.

        Args:
            db: base de datos del proyecto.
            executor: ejecutor de órdenes.
            stop_loss_pct: porcentaje de stop loss precio-relativo.
            take_profit_pct: porcentaje de take profit precio-relativo.
            time_exit_threshold_s: segundos restantes por debajo de los cuales
                se activa la salida anticipada por tiempo (default 60 s).
            time_exit_profit_pct: beneficio mínimo relativo al entry para cerrar
                anticipadamente cuando el tiempo es bajo (default 8 %).
        """

        self.db = db
        self.executor = executor
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.time_exit_threshold_s = time_exit_threshold_s
        self.time_exit_profit_pct = time_exit_profit_pct
        self.initial_bankroll = initial_bankroll
        self.min_closed_trades = min_closed_trades
        self.min_win_rate = min_win_rate
        self.min_total_pnl = min_total_pnl
        self.max_drawdown_pct = max_drawdown_pct
        self.max_drawdown_limit = -(initial_bankroll * max_drawdown_pct)
        self.open_positions: dict[int, Trade] = {}
        self.closed_positions: list[Trade] = []
        self.traded_tickers: set[str] = set()
        self.close_reasons: dict[int, str] = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self._latest_marks: dict[int, float] = {}
        self._last_risk_reason: str | None = None
        self._open_lock = asyncio.Lock()
        # Safe mode: si True, ningún trade nuevo puede abrirse
        self._safe_mode: bool = False
        self._safe_mode_reason: str | None = None

    # ── Safe Mode ─────────────────────────────────────────────────────────────

    def enter_safe_mode(self, reason: str) -> None:
        """Activa el modo seguro: bloquea toda apertura de nuevos trades."""
        if not self._safe_mode:
            logger.warning("SAFE_MODE_ACTIVATED reason=%s", reason)
        self._safe_mode = True
        self._safe_mode_reason = reason

    @property
    def is_safe_mode(self) -> bool:
        """True si el sistema está en modo seguro."""
        return self._safe_mode

    def sync_open_trades(self) -> list[Trade]:
        """Sincroniza el estado in-memory desde SQLite."""

        trades = self.db.get_open_trades(mode=TradeMode(self.executor.mode))
        self.open_positions = {
            trade.id: trade for trade in trades if trade.id is not None and trade.status == TradeStatus.OPEN
        }
        self.traded_tickers.update(trade.ticker for trade in self.open_positions.values())
        self._latest_marks = {trade_id: trade.entry_price for trade_id, trade in self.open_positions.items()}
        self._recalculate_unrealized()
        return list(self.open_positions.values())

    async def hydrate_from_db(self, db: Database | None = None, closed_limit: int = 200) -> None:
        """
        Hidrata contexto histórico y PnL realizado desde SQLite.

        Si la hidratación falla, entra en safe mode para evitar operar con
        estado inconsistente. La excepción NO se propaga al llamador.
        """
        try:
            source = db or self.db
            trade_mode = TradeMode(self.executor.mode)
            closed = source.get_closed_trades(limit=closed_limit, mode=trade_mode)
            self.closed_positions = list(reversed(closed))
            self.traded_tickers = {trade.ticker for trade in self.closed_positions}
            self.close_reasons = {
                trade.id: "hydrated"
                for trade in self.closed_positions
                if trade.id is not None
            }
            summary = source.get_trade_pnl_summary(mode=trade_mode)
            self.realized_pnl = float(summary["realized_pnl"])
            self.unrealized_pnl = 0.0
            self.total_pnl = self.realized_pnl
            self.sync_open_trades()
            logger.info(
                "hydration_complete closed=%d pnl=%.4f realized=%.4f open=%d",
                len(self.closed_positions),
                self.total_pnl,
                self.realized_pnl,
                len(self.open_positions),
            )
        except Exception as exc:  # noqa: BLE001
            self.enter_safe_mode(reason=f"hydration_failed:{exc}")
            logger.error(
                "hydration_failed exc=%s — entered safe mode, no trading allowed",
                exc,
            )
            raise

    async def register_trade(self, trade: Trade) -> Trade:
        """Registra un trade abierto dentro del manager."""

        if trade.id is None:
            raise RuntimeError("Trade debe estar persistido antes de registrarse")
        self.open_positions[trade.id] = trade
        self.traded_tickers.add(trade.ticker)
        self._latest_marks[trade.id] = trade.entry_price
        self._recalculate_unrealized()
        return trade

    async def open_from_signal(self, signal: Signal) -> Trade:
        """Abre un trade desde una señal actionable y lo registra."""

        trade = await self.executor.submit(signal)
        if trade.status == TradeStatus.OPEN and trade.id is not None:
            self.open_positions[trade.id] = trade
            self.traded_tickers.add(trade.ticker)
            self._latest_marks[trade.id] = trade.entry_price
            self._recalculate_unrealized()
        return trade

    async def try_open_from_signal(
        self,
        signal: Signal,
        max_positions: int,
    ) -> Trade | None:
        """
        Abre una señal si respeta los límites de riesgo actuales.

        Esta operación es atómica dentro del proceso para evitar que snapshots
        concurrentes de WS y REST abran dos posiciones del mismo ticker.
        """

        async with self._open_lock:
            status = self.go_no_go_status(category=self._infer_category_from_ticker(signal.market_ticker))
            if not status.allowed:
                raise RuntimeError(f"Risk violation: attempted trade while go=False ({status.reason})")
            ticker = signal.market_ticker
            if self.has_traded_ticker(ticker):
                logger.info("open_skip ticker=%s reason=already_traded", ticker)
                return None
            if self.has_open_ticker(ticker):
                logger.info("open_skip ticker=%s reason=already_open", ticker)
                return None
            if len(self.open_positions) >= max_positions:
                logger.info(
                    "open_skip ticker=%s reason=max_positions open=%d max=%d",
                    ticker,
                    len(self.open_positions),
                    max_positions,
                )
                return None
            return await self.open_from_signal(signal)

    def has_open_ticker(self, ticker: str) -> bool:
        """True si ya existe una posición abierta para el ticker."""

        return any(trade.ticker == ticker for trade in self.open_positions.values())

    def has_traded_ticker(self, ticker: str) -> bool:
        """True si este contrato ya se operó y debe evitarse reentrada."""

        return ticker in self.traded_tickers

    async def evaluate_price(
        self,
        ticker: str,
        current_yes_price: float,
        current_no_price: float | None = None,
        time_remaining_s: int | None = None,
    ) -> list[ManagedClose]:
        """
        Evalúa SL/TP para todas las posiciones abiertas de un ticker.

        Args:
            ticker: contrato a evaluar.
            current_yes_price: precio actual del lado YES.
            current_no_price: precio actual del lado NO. Si no se pasa,
                se usa ``1 - current_yes_price``.
            time_remaining_s: segundos restantes hasta expiración. Cuando
                está por debajo de ``time_exit_threshold_s`` y la posición
                tiene beneficio suficiente, se cierra con razón
                ``time_exit_profit``.
        """

        closes: list[ManagedClose] = []
        effective_no_price = current_no_price if current_no_price is not None else 1.0 - current_yes_price
        self._mark_ticker_positions(
            ticker=ticker,
            current_yes_price=current_yes_price,
            current_no_price=effective_no_price,
        )

        near_expiry = (
            time_remaining_s is not None
            and time_remaining_s <= self.time_exit_threshold_s
        )

        for trade_id, trade in list(self.open_positions.items()):
            if trade.ticker != ticker:
                continue

            current_price = current_yes_price if trade.side == "YES" else effective_no_price

            # Salida anticipada por tiempo: si quedan pocos segundos y la posición
            # ya alcanzó el umbral mínimo de beneficio, asegurar la ganancia.
            if near_expiry and current_price >= trade.entry_price * (1.0 + self.time_exit_profit_pct):
                closed = await self.close_trade(trade, current_price, "time_exit_profit")
                closes.append(closed)
                continue

            if current_price <= trade.entry_price * (1.0 - self.stop_loss_pct):
                closed = await self.close_trade(trade, current_price, "stop_loss")
                closes.append(closed)
                continue

            if current_price >= trade.entry_price * (1.0 + self.take_profit_pct):
                closed = await self.close_trade(trade, current_price, "take_profit")
                closes.append(closed)

        return closes

    async def close_trade(self, trade: Trade, exit_price: float, reason: str) -> ManagedClose:
        """Cierra un trade y actualiza observabilidad."""

        closed_trade = await self.executor.close_with_price(trade=trade, exit_price=exit_price)
        if closed_trade.id is None:
            raise RuntimeError("Trade cerrado sin id persistido")
        self.open_positions.pop(closed_trade.id, None)
        self._latest_marks.pop(closed_trade.id, None)
        self.closed_positions.append(closed_trade)
        self.traded_tickers.add(closed_trade.ticker)
        self.close_reasons[closed_trade.id] = reason
        self.realized_pnl += closed_trade.pnl or 0.0
        self._recalculate_unrealized()
        logger.info(
            "position_closed ticker=%s reason=%s pnl=%.4f",
            closed_trade.ticker,
            reason,
            closed_trade.pnl or 0.0,
        )
        return ManagedClose(trade=closed_trade, reason=reason)

    def observability_snapshot(self) -> dict[str, float | int | str]:
        """Retorna snapshot compacto de observabilidad."""

        closed_count = len(self.closed_positions)
        wins = sum(1 for trade in self.closed_positions if (trade.pnl or 0.0) > 0.0)
        total_fees = sum(trade.fee_paid for trade in self.closed_positions) + sum(
            trade.fee_paid for trade in self.open_positions.values()
        )
        return {
            "mode": self.executor.mode,
            "open_positions": len(self.open_positions),
            "closed_positions": closed_count,
            "win_rate": (wins / closed_count) if closed_count else 0.0,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_fees": total_fees,
        }

    def go_no_go_status(
        self,
        min_closed_trades: int | None = None,
        min_win_rate: float | None = None,
        min_total_pnl: float | None = None,
        max_open_positions: int | None = None,
        max_drawdown_limit: float | None = None,
        category: str | None = None,
    ) -> GoNoGoStatus:
        """Determina si el sistema debería seguir operando."""

        # Safe mode is a hard stop — overrides every other check
        if self._safe_mode:
            status = GoNoGoStatus(
                allowed=False,
                reason=f"safe_mode:{self._safe_mode_reason}",
                closed_trades=len(self.closed_positions),
                win_rate=self._win_rate(),
                total_pnl=self.total_pnl,
            )
            self._log_risk_transition(status)
            return status

        min_closed_trades = self.min_closed_trades if min_closed_trades is None else min_closed_trades
        min_win_rate = self.min_win_rate if min_win_rate is None else min_win_rate
        min_total_pnl = self.min_total_pnl if min_total_pnl is None else min_total_pnl
        max_drawdown_limit = self.max_drawdown_limit if max_drawdown_limit is None else max_drawdown_limit
        relevant_closed = self._closed_positions_for_category(category)
        closed_count = len(relevant_closed)
        relevant_total_pnl = sum(trade.pnl or 0.0 for trade in relevant_closed)

        # Hard capital floor: stop trading if effective balance has reached zero
        if self.initial_bankroll + self.total_pnl <= 0.0:
            status = GoNoGoStatus(
                allowed=False,
                reason="account_balance_depleted",
                closed_trades=closed_count,
                win_rate=self._win_rate(relevant_closed),
                total_pnl=self.total_pnl,
            )
            self._log_risk_transition(status)
            return status

        if max_open_positions is not None and len(self.open_positions) >= max_open_positions:
            status = GoNoGoStatus(
                allowed=False,
                reason="open_exposure_limit_reached",
                closed_trades=closed_count,
                win_rate=self._win_rate(relevant_closed),
                total_pnl=relevant_total_pnl if category else self.total_pnl,
            )
            self._log_risk_transition(status)
            return status
        if closed_count < min_closed_trades:
            status = GoNoGoStatus(
                allowed=False,
                reason="insufficient_data",
                closed_trades=closed_count,
                win_rate=self._win_rate(relevant_closed),
                total_pnl=relevant_total_pnl if category else self.total_pnl,
            )
            self._log_risk_transition(status)
            return status

        win_rate = self._win_rate(relevant_closed)
        if win_rate < min_win_rate and relevant_total_pnl < 0.0:
            status = GoNoGoStatus(
                allowed=False,
                reason="win_rate_too_low",
                closed_trades=closed_count,
                win_rate=win_rate,
                total_pnl=relevant_total_pnl if category else self.total_pnl,
            )
            self._log_risk_transition(status)
            return status
        effective_total_pnl = relevant_total_pnl if category else self.total_pnl
        if effective_total_pnl < max_drawdown_limit:
            status = GoNoGoStatus(
                allowed=False,
                reason="max_drawdown_exceeded",
                closed_trades=closed_count,
                win_rate=win_rate,
                total_pnl=effective_total_pnl,
            )
            self._log_risk_transition(status)
            return status
        if effective_total_pnl < min_total_pnl:
            status = GoNoGoStatus(
                allowed=False,
                reason="pnl_too_low",
                closed_trades=closed_count,
                win_rate=win_rate,
                total_pnl=effective_total_pnl,
            )
            self._log_risk_transition(status)
            return status
        status = GoNoGoStatus(
            allowed=True,
            reason="go",
            closed_trades=closed_count,
            win_rate=win_rate,
            total_pnl=effective_total_pnl,
        )
        self._log_risk_transition(status)
        return status

    def _win_rate(self, closed_positions: list[Trade] | None = None) -> float:
        """Calcula win rate sobre el historial cerrado cargado en memoria."""

        relevant_closed = self.closed_positions if closed_positions is None else closed_positions
        effective_closed = len(relevant_closed)
        if effective_closed == 0:
            return 0.0
        wins = sum(1 for trade in relevant_closed if (trade.pnl or 0.0) > 0.0)
        return wins / effective_closed

    def _closed_positions_for_category(self, category: str | None) -> list[Trade]:
        """Filtra trades cerrados por categoría cuando corresponde."""

        if category is None:
            return self.closed_positions
        return [
            trade
            for trade in self.closed_positions
            if self._infer_category_from_ticker(trade.ticker) == category
        ]

    @staticmethod
    def _infer_category_from_ticker(ticker: str) -> str:
        """Infiere una categoría simple desde el ticker."""

        upper_ticker = ticker.upper()
        for candidate in ("BTC", "ETH", "SOL"):
            if candidate in upper_ticker:
                return candidate
        return "UNKNOWN"

    def _mark_ticker_positions(
        self,
        ticker: str,
        current_yes_price: float,
        current_no_price: float,
    ) -> None:
        """Actualiza el último mark conocido para un ticker."""

        for trade_id, trade in self.open_positions.items():
            if trade.ticker != ticker:
                continue
            self._latest_marks[trade_id] = current_yes_price if trade.side == "YES" else current_no_price
        self._recalculate_unrealized()

    def _recalculate_unrealized(self) -> None:
        """Recalcula PnL no realizado a partir de los últimos precios marcados."""

        unrealized = 0.0
        for trade_id, trade in self.open_positions.items():
            mark = self._latest_marks.get(trade_id, trade.entry_price)
            unrealized += (mark - trade.entry_price) * trade.contracts
        self.unrealized_pnl = unrealized
        self.total_pnl = self.realized_pnl + self.unrealized_pnl

    def _log_risk_transition(self, status: GoNoGoStatus) -> None:
        """Emite logs de auditoría solo cuando cambia el estado de riesgo."""

        if status.reason == self._last_risk_reason:
            return
        self._last_risk_reason = status.reason
        if status.reason == "max_drawdown_exceeded":
            logger.warning("RISK_STOP triggered: pnl=%.4f", self.total_pnl)
