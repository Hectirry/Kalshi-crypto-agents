"""
execution/position_manager.py

Gestión de posiciones abiertas, stop loss, take profit y observabilidad.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from core.database import Database
from core.models import Signal, Trade, TradeStatus
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
    ) -> None:
        """
        Inicializa el manager de posiciones.

        Args:
            db: base de datos del proyecto.
            executor: ejecutor de órdenes.
            stop_loss_pct: porcentaje de stop loss.
            take_profit_pct: porcentaje de take profit.
        """

        self.db = db
        self.executor = executor
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.open_positions: dict[int, Trade] = {}
        self.closed_positions: list[Trade] = []
        self.close_reasons: dict[int, str] = {}
        self._open_lock = asyncio.Lock()

    def sync_open_trades(self) -> list[Trade]:
        """Sincroniza el estado in-memory desde SQLite."""

        trades = self.db.get_open_trades()
        self.open_positions = {
            trade.id: trade for trade in trades if trade.id is not None and trade.status == TradeStatus.OPEN
        }
        return list(self.open_positions.values())

    async def register_trade(self, trade: Trade) -> Trade:
        """Registra un trade abierto dentro del manager."""

        if trade.id is None:
            raise RuntimeError("Trade debe estar persistido antes de registrarse")
        self.open_positions[trade.id] = trade
        return trade

    async def open_from_signal(self, signal: Signal) -> Trade:
        """Abre un trade desde una señal actionable y lo registra."""

        trade = await self.executor.submit(signal)
        if trade.status == TradeStatus.OPEN and trade.id is not None:
            self.open_positions[trade.id] = trade
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
            ticker = signal.market_ticker
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

    async def evaluate_price(
        self,
        ticker: str,
        current_yes_price: float,
        current_no_price: float | None = None,
    ) -> list[ManagedClose]:
        """
        Evalúa SL/TP para todas las posiciones abiertas de un ticker.

        Args:
            ticker: contrato a evaluar.
            current_yes_price: precio actual del lado YES.
            current_no_price: precio actual del lado NO. Si no se pasa,
                se usa ``1 - current_yes_price``.
        """

        closes: list[ManagedClose] = []
        effective_no_price = current_no_price if current_no_price is not None else 1.0 - current_yes_price

        for trade_id, trade in list(self.open_positions.items()):
            if trade.ticker != ticker:
                continue

            current_price = current_yes_price if trade.side == "YES" else effective_no_price
            if current_price <= trade.entry_price * (1.0 - self.stop_loss_pct):
                closed = await self._close_trade(trade, current_price, "stop_loss")
                closes.append(closed)
                del self.open_positions[trade_id]
                continue

            if current_price >= trade.entry_price * (1.0 + self.take_profit_pct):
                closed = await self._close_trade(trade, current_price, "take_profit")
                closes.append(closed)
                del self.open_positions[trade_id]

        return closes

    async def _close_trade(self, trade: Trade, exit_price: float, reason: str) -> ManagedClose:
        """Cierra un trade y actualiza observabilidad."""

        closed_trade = await self.executor.close_with_price(trade=trade, exit_price=exit_price)
        if closed_trade.id is None:
            raise RuntimeError("Trade cerrado sin id persistido")
        self.closed_positions.append(closed_trade)
        self.close_reasons[closed_trade.id] = reason
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
        total_pnl = sum(trade.pnl or 0.0 for trade in self.closed_positions)
        total_fees = sum(trade.fee_paid for trade in self.closed_positions) + sum(
            trade.fee_paid for trade in self.open_positions.values()
        )
        return {
            "mode": self.executor.mode,
            "open_positions": len(self.open_positions),
            "closed_positions": closed_count,
            "win_rate": (wins / closed_count) if closed_count else 0.0,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
        }

    def go_no_go_status(
        self,
        min_closed_trades: int = 10,
        min_win_rate: float = 0.52,
        min_total_pnl: float = 0.0,
    ) -> GoNoGoStatus:
        """Determina si el sistema debería seguir operando."""

        closed_count = len(self.closed_positions)
        if closed_count < min_closed_trades:
            return GoNoGoStatus(
                allowed=False,
                reason="insufficient_data",
                closed_trades=closed_count,
                win_rate=0.0,
                total_pnl=sum(trade.pnl or 0.0 for trade in self.closed_positions),
            )

        wins = sum(1 for trade in self.closed_positions if (trade.pnl or 0.0) > 0.0)
        total_pnl = sum(trade.pnl or 0.0 for trade in self.closed_positions)
        win_rate = wins / closed_count
        if win_rate < min_win_rate:
            return GoNoGoStatus(
                allowed=False,
                reason="win_rate_too_low",
                closed_trades=closed_count,
                win_rate=win_rate,
                total_pnl=total_pnl,
            )
        if total_pnl < min_total_pnl:
            return GoNoGoStatus(
                allowed=False,
                reason="pnl_too_low",
                closed_trades=closed_count,
                win_rate=win_rate,
                total_pnl=total_pnl,
            )
        return GoNoGoStatus(
            allowed=True,
            reason="go",
            closed_trades=closed_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
        )
