"""
backtesting/backtest_runner.py

Backtesting histórico sobre señales persistidas en SQLite.
"""

from __future__ import annotations

import logging
import math
from math import isfinite
from dataclasses import dataclass

from core.config import EngineCategoryOverride, EngineConfig
from core.database import Database
from core.interfaces import BacktestSource
from core.models import Decision, Outcome, Signal
from engine.ev_calculator import EVCalculator
from engine.probability import classify_time_zone

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

    def __init__(
        self,
        db: Database,
        initial_bankroll: float = 1_000.0,
        contracts_multiplier: int = 100,
        config: EngineConfig | None = None,
    ) -> None:
        """
        Inicializa el runner.

        Args:
            db: base de datos del proyecto.
            initial_bankroll: capital inicial del backtest.
        """

        self.db = db
        self.initial_bankroll = initial_bankroll
        self.contracts_multiplier = contracts_multiplier
        self.config = config
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
        actionable = self._apply_policy_filters(
            [
                signal
                for signal in signals
                if signal.decision in (Decision.YES, Decision.NO) and signal.outcome is not None
            ]
        )
        skipped = len(signals) - len(actionable)

        if actionable and self._vectorbt_ready() and self._vectorbt_prices_supported(actionable):
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
            pnl = self._trade_pnl(signal=signal)
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

    def _trade_pnl(self, signal: Signal) -> float:
        """Calcula el P&L realizado con el mismo sizing del executor."""

        contracts = max(1, math.ceil(signal.kelly_size * self.contracts_multiplier))
        if contracts <= 0:
            return 0.0

        contract_price = self._contract_price(signal)
        exit_price = 1.0 if self._signal_won(signal) else 0.0
        entry_fee = self.ev_calculator.fee_per_contract(contract_price) * contracts
        exit_fee = self.ev_calculator.fee_per_contract(exit_price) * contracts
        fee_paid = entry_fee + exit_fee

        gross_pnl = (exit_price - contract_price) * contracts
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
            contracts = max(1, math.ceil(signal.kelly_size * self.contracts_multiplier))
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
    def _dedupe_signals_by_ticker(signals: list[Signal]) -> list[Signal]:
        """Conserva la primera señal actionable por ticker para evitar churn."""

        deduped: list[Signal] = []
        seen_tickers: set[str] = set()
        for signal in signals:
            if signal.market_ticker in seen_tickers:
                continue
            seen_tickers.add(signal.market_ticker)
            deduped.append(signal)
        return deduped

    def _apply_policy_filters(self, signals: list[Signal]) -> list[Signal]:
        """Aplica dedupe y filtros equivalentes a la política de entrada."""

        if self.config is None:
            return self._dedupe_signals_by_ticker(signals)
        eligible: list[Signal] = []
        resolved_history: list[Signal] = []
        for signal in sorted(signals, key=lambda entry: entry.timestamp):
            if self._passes_policy(signal, resolved_history):
                eligible.append(signal)
            if signal.outcome is not None:
                resolved_history.append(signal)
        return self._dedupe_signals_by_ticker(eligible)

    def _passes_policy(self, signal: Signal, resolved_history: list[Signal] | None = None) -> bool:
        """Replica los thresholds del router para un signal persistido."""

        category = _infer_category_from_ticker(signal.market_ticker)
        params = self.db.get_current_params(category=None if category == "UNKNOWN" else category)
        override = self._category_override(category)
        min_delta = max(
            params.get("min_delta", self.config.min_delta),
            override.min_delta if override.min_delta is not None else self.config.min_delta,
        )
        min_ev = max(
            params.get("min_ev_threshold", self.config.min_ev_threshold),
            override.min_ev_threshold if override.min_ev_threshold is not None else self.config.min_ev_threshold,
        )
        min_time = max(
            self.config.min_time_remaining_s,
            override.min_time_remaining_s if override.min_time_remaining_s is not None else self.config.min_time_remaining_s,
        )
        min_price = max(
            self.config.min_contract_price,
            override.min_contract_price if override.min_contract_price is not None else self.config.min_contract_price,
        )
        max_price = min(
            self.config.max_contract_price,
            override.max_contract_price if override.max_contract_price is not None else self.config.max_contract_price,
        )
        contract_price = self._contract_price(signal)
        return (
            abs(signal.delta) >= min_delta
            and signal.ev_net_fees >= min_ev
            and signal.time_remaining_s >= min_time
            and min_price <= contract_price <= max_price
            and (
                signal.market_overround_bps is None
                or signal.market_overround_bps <= self.config.max_market_overround_bps
            )
            and self._passes_setup_quality_gate(signal, resolved_history or [])
        )

    def _category_override(self, category: str) -> EngineCategoryOverride:
        """Retorna el override de categoría correspondiente."""

        if self.config is None:
            return EngineCategoryOverride()
        return self.config.category_overrides.get(category, EngineCategoryOverride())

    @staticmethod
    def _vectorbt_ready() -> bool:
        """Retorna True si el entorno tiene lo necesario para ejecutar vectorbt."""

        return vbt is not None and pd is not None

    def _vectorbt_prices_supported(self, actionable: list[Signal]) -> bool:
        """Valida si el set de trades puede representarse con precios válidos para vectorbt."""

        for signal in actionable:
            contract_price = self._contract_price(signal)
            exit_price = 1.0 if self._signal_won(signal) else 0.0
            if not isfinite(contract_price) or contract_price <= 0.0:
                logger.info(
                    "vectorbt_skip reason=invalid_entry_price ticker=%s price=%.4f",
                    signal.market_ticker,
                    contract_price,
                )
                return False
            if not isfinite(exit_price) or exit_price <= 0.0:
                logger.info(
                    "vectorbt_skip reason=invalid_exit_price ticker=%s price=%.4f",
                    signal.market_ticker,
                    exit_price,
                )
                return False
        return True

    @staticmethod
    def _contract_price(signal: Signal) -> float:
        """Retorna el precio relevante del lado YES o NO."""

        if signal.contract_price is not None:
            return max(signal.contract_price, 0.01)
        if signal.decision == Decision.YES:
            return max(signal.market_probability, 0.01)
        return max(1.0 - signal.market_probability, 0.01)

    @staticmethod
    def _signal_won(signal: Signal) -> bool:
        """Determina si la recomendación YES/NO ganó según el outcome real."""

        if signal.outcome is None:
            return False
        return signal.outcome == Outcome.WIN

    def _passes_setup_quality_gate(
        self,
        signal: Signal,
        resolved_history: list[Signal],
    ) -> bool:
        """Replica el gate histórico del router para replay consistente."""
        if self.config is None or not self.config.setup_quality_gate_enabled:
            return True

        comparable = [
            entry
            for entry in resolved_history
            if _infer_category_from_ticker(entry.market_ticker) == _infer_category_from_ticker(signal.market_ticker)
            and entry.decision == signal.decision
            and classify_time_zone(entry.time_remaining_s) == classify_time_zone(signal.time_remaining_s)
            and _delta_bucket(entry.delta) == _delta_bucket(signal.delta)
        ][-self.config.setup_quality_history_limit:]

        if len(comparable) < self.config.setup_quality_min_samples:
            return True

        wins = sum(1 for entry in comparable if entry.outcome == Outcome.WIN)
        win_rate = wins / len(comparable)
        return win_rate >= self.config.setup_quality_min_win_rate


def _infer_category_from_ticker(ticker: str) -> str:
    """Infiere categoría simple desde el ticker persistido."""

    upper_ticker = ticker.upper()
    for candidate in ("BTC", "ETH", "SOL"):
        if candidate in upper_ticker:
            return candidate
    return "UNKNOWN"


def _delta_bucket(delta: float) -> str:
    """Agrupa magnitud del edge para comparar setups similares."""
    magnitude = abs(delta)
    if magnitude < 0.08:
        return "small"
    if magnitude < 0.16:
        return "medium"
    return "large"
