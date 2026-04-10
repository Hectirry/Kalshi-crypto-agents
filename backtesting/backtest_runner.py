"""
backtesting/backtest_runner.py

Backtesting histórico sobre señales persistidas en SQLite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.database import Database
from core.interfaces import BacktestSource
from core.models import Decision, Outcome, Signal
from engine.ev_calculator import EVCalculator

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore[import-not-found]
    import vectorbt as vbt  # type: ignore[import-not-found]
except ImportError:
    pd = None
    vbt = None


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Punto de la curva de equity."""

    timestamp: float
    equity: float
    pnl: float
    drawdown: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Resultado agregado del backtest."""

    total_signals: int
    actionable_signals: int
    wins: int
    losses: int
    skipped: int
    win_rate: float
    total_pnl: float
    final_equity: float
    max_drawdown: float
    avg_pnl_per_trade: float
    category: str | None
    vectorbt_available: bool
    vectorbt_used: bool
    equity_curve: list[EquityPoint]


class BacktestRunner(BacktestSource):
    """Ejecuta backtests sobre señales ya guardadas en SQLite."""

    def __init__(self, db: Database, initial_bankroll: float = 1_000.0) -> None:
        """
        Inicializa el runner.

        Args:
            db: base de datos del proyecto.
            initial_bankroll: capital inicial del backtest.
        """

        self.db = db
        self.initial_bankroll = initial_bankroll
        self.ev_calculator = EVCalculator()

    async def load_signals(
        self,
        from_ts: float,
        to_ts: float,
        category: str | None = None,
    ) -> list[Signal]:
        """Carga señales históricas ordenadas por timestamp ascendente."""

        return self.db.get_signals(from_ts=from_ts, to_ts=to_ts, category=category)

    def run(
        self,
        from_ts: float,
        to_ts: float,
        category: str | None = None,
        initial_bankroll: float | None = None,
    ) -> BacktestResult:
        """
        Ejecuta un backtest determinista sobre señales cerradas.

        Args:
            from_ts: inicio del rango.
            to_ts: fin del rango.
            category: categoría opcional, por ejemplo ``BTC`` o ``ETH``.
            initial_bankroll: bankroll inicial opcional.

        Returns:
            BacktestResult con curva de equity y métricas agregadas.
        """

        bankroll = initial_bankroll if initial_bankroll is not None else self.initial_bankroll
        signals = self.db.get_signals(from_ts=from_ts, to_ts=to_ts, category=category)
        actionable = [
            signal
            for signal in signals
            if signal.decision in (Decision.YES, Decision.NO) and signal.outcome is not None
        ]
        skipped = len(signals) - len(actionable)

        if actionable and self._vectorbt_ready():
            try:
                return self._run_vectorbt(
                    signals=signals,
                    actionable=actionable,
                    skipped=skipped,
                    bankroll=bankroll,
                    category=category,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning("vectorbt_backtest_failed category=%s error=%s", category or "ALL", exc)

        equity = bankroll
        peak_equity = bankroll
        wins = 0
        losses = 0
        total_pnl = 0.0
        curve: list[EquityPoint] = []

        for signal in actionable:
            pnl = self._trade_pnl(signal=signal, bankroll=equity)
            equity += pnl
            total_pnl += pnl
            peak_equity = max(peak_equity, equity)
            drawdown = peak_equity - equity
            curve.append(
                EquityPoint(
                    timestamp=signal.timestamp,
                    equity=equity,
                    pnl=pnl,
                    drawdown=drawdown,
                )
            )
            if pnl >= 0.0:
                wins += 1
            else:
                losses += 1

        avg_pnl = total_pnl / len(actionable) if actionable else 0.0
        win_rate = wins / len(actionable) if actionable else 0.0
        max_drawdown = max((point.drawdown for point in curve), default=0.0)

        logger.info(
            "backtest_completed category=%s actionable=%s total_pnl=%.4f win_rate=%.4f",
            category or "ALL",
            len(actionable),
            total_pnl,
            win_rate,
        )

        return BacktestResult(
            total_signals=len(signals),
            actionable_signals=len(actionable),
            wins=wins,
            losses=losses,
            skipped=skipped,
            win_rate=win_rate,
            total_pnl=total_pnl,
            final_equity=equity,
            max_drawdown=max_drawdown,
            avg_pnl_per_trade=avg_pnl,
            category=category,
            vectorbt_available=vbt is not None,
            vectorbt_used=False,
            equity_curve=curve,
        )

    def categories_in_range(self, from_ts: float, to_ts: float) -> set[str]:
        """Infiere categorías a partir del ticker en el rango indicado."""

        signals = self.db.get_signals(from_ts=from_ts, to_ts=to_ts)
        return {
            category
            for category in (_infer_category_from_ticker(signal.market_ticker) for signal in signals)
            if category != "UNKNOWN"
        }

    def _trade_pnl(self, signal: Signal, bankroll: float) -> float:
        """Calcula el P&L realizado de una señal usando su outcome real."""

        stake = bankroll * signal.kelly_size
        if stake <= 0.0:
            return 0.0

        contract_price = self._contract_price(signal)
        contracts = stake / max(contract_price, 0.01)
        exit_price = 1.0 if self._signal_won(signal) else 0.0
        entry_fee = self.ev_calculator.fee_per_contract(contract_price) * contracts
        exit_fee = self.ev_calculator.fee_per_contract(exit_price) * contracts
        fee_paid = entry_fee + exit_fee

        if self._signal_won(signal):
            gross_pnl = contracts * (1.0 - contract_price)
        else:
            gross_pnl = -stake
        return gross_pnl - fee_paid

    def _run_vectorbt(
        self,
        *,
        signals: list[Signal],
        actionable: list[Signal],
        skipped: int,
        bankroll: float,
        category: str | None,
    ) -> BacktestResult:
        """Ejecuta el backtest usando ``vectorbt`` sobre trades secuenciales."""

        equity = bankroll
        peak_equity = bankroll
        wins = 0
        losses = 0
        total_pnl = 0.0
        curve: list[EquityPoint] = []

        for signal in actionable:
            contract_price = self._contract_price(signal)
            stake = equity * signal.kelly_size
            if stake <= 0.0:
                curve.append(
                    EquityPoint(
                        timestamp=signal.timestamp,
                        equity=equity,
                        pnl=0.0,
                        drawdown=peak_equity - equity,
                    )
                )
                continue

            contracts = stake / max(contract_price, 0.01)
            exit_price = 1.0 if self._signal_won(signal) else 0.0
            entry_fee = self.ev_calculator.fee_per_contract(contract_price) * contracts
            exit_fee = self.ev_calculator.fee_per_contract(exit_price) * contracts

            index = pd.Index([signal.timestamp, signal.timestamp + 1e-6], dtype="float64")
            close = pd.Series([contract_price, exit_price], index=index)
            size = pd.Series([contracts, -contracts], index=index)
            fixed_fees = pd.Series([entry_fee, exit_fee], index=index)

            portfolio = vbt.Portfolio.from_orders(
                close=close,
                size=size,
                fixed_fees=fixed_fees,
                init_cash=equity,
            )

            pnl = float(portfolio.total_profit())
            equity_path = [float(value) for value in portfolio.value().tolist()]
            for value in equity_path:
                peak_equity = max(peak_equity, value)
            equity = equity_path[-1]
            total_pnl += pnl
            drawdown = peak_equity - equity
            curve.append(
                EquityPoint(
                    timestamp=signal.timestamp,
                    equity=equity,
                    pnl=pnl,
                    drawdown=drawdown,
                )
            )
            if pnl >= 0.0:
                wins += 1
            else:
                losses += 1

        avg_pnl = total_pnl / len(actionable) if actionable else 0.0
        win_rate = wins / len(actionable) if actionable else 0.0
        max_drawdown = max((point.drawdown for point in curve), default=0.0)

        logger.info(
            "vectorbt_backtest_completed category=%s actionable=%s total_pnl=%.4f win_rate=%.4f",
            category or "ALL",
            len(actionable),
            total_pnl,
            win_rate,
        )

        return BacktestResult(
            total_signals=len(signals),
            actionable_signals=len(actionable),
            wins=wins,
            losses=losses,
            skipped=skipped,
            win_rate=win_rate,
            total_pnl=total_pnl,
            final_equity=equity,
            max_drawdown=max_drawdown,
            avg_pnl_per_trade=avg_pnl,
            category=category,
            vectorbt_available=True,
            vectorbt_used=True,
            equity_curve=curve,
        )

    @staticmethod
    def _vectorbt_ready() -> bool:
        """Retorna True si el entorno tiene lo necesario para ejecutar vectorbt."""

        return vbt is not None and pd is not None

    @staticmethod
    def _contract_price(signal: Signal) -> float:
        """Retorna el precio relevante del lado YES o NO."""

        if signal.decision == Decision.YES:
            return max(signal.market_probability, 0.01)
        return max(1.0 - signal.market_probability, 0.01)

    @staticmethod
    def _signal_won(signal: Signal) -> bool:
        """Determina si la recomendación YES/NO ganó según el outcome real."""

        if signal.outcome is None:
            return False
        if signal.decision == Decision.YES:
            return signal.outcome == Outcome.WIN
        return signal.outcome == Outcome.LOSS


def _infer_category_from_ticker(ticker: str) -> str:
    """Infiere categoría simple desde el ticker persistido."""

    upper_ticker = ticker.upper()
    for candidate in ("BTC", "ETH", "SOL"):
        if candidate in upper_ticker:
            return candidate
    return "UNKNOWN"
