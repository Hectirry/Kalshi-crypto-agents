"""
execution/order_executor.py

Ejecución de órdenes en modo demo o production.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from typing import Any

from core.database import Database
from core.models import Decision, Signal, Trade, TradeMode, TradeStatus
from engine.ev_calculator import EVCalculator

logger = logging.getLogger(__name__)


class KalshiExecutionClientProtocol:
    """Contrato mínimo esperado de un cliente de ejecución de Kalshi."""

    async def submit_order(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
    ) -> dict[str, Any]:
        """Envía una orden de entrada."""

    async def close_order(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
    ) -> dict[str, Any]:
        """Envía una orden de cierre."""


class PaperOrderExecutor:
    """Ejecutor único para demo y production con persistencia en SQLite."""

    def __init__(
        self,
        db: Database,
        mode: TradeMode | str = TradeMode.DEMO,
        client: KalshiExecutionClientProtocol | None = None,
        contracts_multiplier: int = 100,
    ) -> None:
        """
        Inicializa el ejecutor.

        Args:
            db: base de datos del proyecto.
            mode: ``demo`` o ``production``.
            client: cliente async real de Kalshi para production.
            contracts_multiplier: convierte Kelly size en contratos nominales.
        """

        self.db = db
        self._mode = TradeMode(mode) if isinstance(mode, str) else mode
        self.client = client
        self.contracts_multiplier = contracts_multiplier
        self.fee_calculator = EVCalculator()

    @property
    def mode(self) -> str:
        """Modo actual del ejecutor."""

        return self._mode.value

    async def submit(self, signal: Signal) -> Trade:
        """
        Ejecuta una señal actionable y persiste el trade.

        En demo, señales no actionables se ignoran devolviendo un trade cancelado.
        En production, una señal no actionable lanza ``ValueError``.
        """

        if signal.decision not in (Decision.YES, Decision.NO):
            if self._mode == TradeMode.PRODUCTION:
                raise ValueError(
                    f"Signal no actionable en production: {signal.decision.value}"
                )
            logger.info(
                "demo_signal_ignored ticker=%s decision=%s",
                signal.market_ticker,
                signal.decision.value,
            )
            return self._build_cancelled_trade(signal)

        signal_id = self.db.find_signal_id(signal)
        if signal_id is None:
            signal_id = self.db.save_signal(signal)
        trade = self._build_open_trade(signal, signal_id=signal_id)

        if self._mode == TradeMode.PRODUCTION:
            if self.client is None:
                raise RuntimeError("Production mode requiere cliente Kalshi")
            await self._submit_live(trade)

        trade_id = self.db.save_trade(trade)
        persisted = replace(trade, id=trade_id)
        logger.info(
            "trade_opened mode=%s ticker=%s side=%s contracts=%s entry=%.4f",
            self.mode,
            persisted.ticker,
            persisted.side,
            persisted.contracts,
            persisted.entry_price,
        )
        return persisted

    async def close(self, trade: Trade) -> Trade:
        """
        Cierra un trade usando el precio de entrada como fallback neutral.

        Este método existe para cumplir la interfaz. Para cierres con precio
        explícito, usar ``close_with_price``.
        """

        return await self.close_with_price(trade=trade, exit_price=trade.entry_price)

    async def close_with_price(self, trade: Trade, exit_price: float) -> Trade:
        """
        Cierra un trade persistido con un precio concreto.

        Args:
            trade: trade abierto.
            exit_price: precio del lado del contrato al cerrar.

        Returns:
            Trade actualizado con ``status='closed'``.
        """

        if trade.status != TradeStatus.OPEN:
            return trade

        if not 0.0 <= exit_price <= 1.0:
            raise ValueError(f"exit_price fuera de rango [0,1]: {exit_price}")

        if self._mode == TradeMode.PRODUCTION:
            if self.client is None:
                raise RuntimeError("Production mode requiere cliente Kalshi")
            await self._close_live(trade=trade, exit_price=exit_price)

        fees_total = self._trade_fee(trade.entry_price, trade.contracts) + self._trade_fee(
            exit_price, trade.contracts
        )
        pnl = ((exit_price - trade.entry_price) * trade.contracts) - fees_total
        if trade.id is None:
            raise RuntimeError("Trade no persistido: falta id para cerrar")

        self.db.close_trade(
            trade_id=trade.id,
            exit_price=exit_price,
            pnl=pnl,
            fee_paid=fees_total,
        )
        closed = replace(
            trade,
            exit_price=exit_price,
            pnl=pnl,
            fee_paid=fees_total,
            status=TradeStatus.CLOSED,
            closed_at=time.time(),
        )
        logger.info(
            "trade_closed mode=%s ticker=%s pnl=%.4f exit=%.4f",
            self.mode,
            closed.ticker,
            pnl,
            exit_price,
        )
        return closed

    def _build_open_trade(self, signal: Signal, signal_id: int) -> Trade:
        """Construye el trade abierto a partir de la señal."""

        contracts = max(1, math.ceil(signal.kelly_size * self.contracts_multiplier))
        entry_price = (
            signal.market_probability
            if signal.decision == Decision.YES
            else 1.0 - signal.market_probability
        )
        return Trade(
            ticker=signal.market_ticker,
            side=signal.decision.value,
            contracts=contracts,
            entry_price=entry_price,
            mode=self._mode,
            status=TradeStatus.OPEN,
            opened_at=time.time(),
            signal_id=signal_id,
            fee_paid=self._trade_fee(entry_price, contracts),
        )

    def _build_cancelled_trade(self, signal: Signal) -> Trade:
        """Construye un trade cancelado para señales ignoradas en demo."""

        return Trade(
            ticker=signal.market_ticker,
            side="YES",
            contracts=1,
            entry_price=max(signal.market_probability, 0.01),
            mode=self._mode,
            status=TradeStatus.CANCELLED,
            opened_at=time.time(),
        )

    async def _submit_live(self, trade: Trade) -> None:
        """Envía una orden real de apertura."""

        try:
            await self.client.submit_order(
                ticker=trade.ticker,
                side=trade.side,
                contracts=trade.contracts,
                price=trade.entry_price,
            )
        except (ConnectionError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Kalshi submit failed: {exc}") from exc

    async def _close_live(self, trade: Trade, exit_price: float) -> None:
        """Envía una orden real de cierre."""

        try:
            await self.client.close_order(
                ticker=trade.ticker,
                side=trade.side,
                contracts=trade.contracts,
                price=exit_price,
            )
        except (ConnectionError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Kalshi close failed: {exc}") from exc

    def _trade_fee(self, price: float, contracts: int) -> float:
        """Calcula el fee total de una pierna de la operación."""

        return self.fee_calculator.fee_per_contract(price) * contracts
