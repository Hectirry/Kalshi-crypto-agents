"""
backtesting/category_blocker.py

Bloqueo automático de categorías con rendimiento histórico deficiente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from backtesting.backtest_runner import BacktestRunner
from core.database import Database

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CategoryBlockDecision:
    """Resultado de la evaluación histórica de una categoría."""

    category: str
    win_rate: float
    sample_size: int
    blocked: bool
    reason: str


class CategoryBlocker:
    """Bloquea o desbloquea categorías según win rate histórico."""

    def __init__(
        self,
        db: Database,
        runner: BacktestRunner,
        min_samples: int = 20,
        min_win_rate: float = 0.48,
    ) -> None:
        """
        Inicializa el bloqueador automático.

        Args:
            db: base de datos del proyecto.
            runner: backtest runner para evaluar performance.
            min_samples: muestra mínima para decidir bloqueo.
            min_win_rate: win rate mínimo aceptable.
        """

        self.db = db
        self.runner = runner
        self.min_samples = min_samples
        self.min_win_rate = min_win_rate

    def evaluate_and_apply(
        self,
        from_ts: float,
        to_ts: float,
        categories: set[str] | None = None,
    ) -> list[CategoryBlockDecision]:
        """
        Evalúa categorías y aplica bloqueo/desbloqueo en SQLite.

        Args:
            from_ts: inicio del rango histórico.
            to_ts: fin del rango histórico.
            categories: conjunto opcional de categorías a evaluar.

        Returns:
            Lista de decisiones tomadas por categoría.
        """

        target_categories = categories or self.runner.categories_in_range(from_ts, to_ts)
        decisions: list[CategoryBlockDecision] = []

        for category in sorted(target_categories):
            result = self.runner.run(from_ts=from_ts, to_ts=to_ts, category=category)
            sample_size = result.actionable_signals
            win_rate = result.win_rate

            if sample_size < self.min_samples:
                decision = CategoryBlockDecision(
                    category=category,
                    win_rate=win_rate,
                    sample_size=sample_size,
                    blocked=False,
                    reason="insufficient_sample",
                )
                decisions.append(decision)
                logger.info(
                    "category_check category=%s sample=%s reason=insufficient_sample",
                    category,
                    sample_size,
                )
                continue

            if win_rate < self.min_win_rate:
                self.db.block_category(
                    category=category,
                    win_rate=win_rate,
                    sample_size=sample_size,
                    reason="win_rate_below_threshold",
                )
                decision = CategoryBlockDecision(
                    category=category,
                    win_rate=win_rate,
                    sample_size=sample_size,
                    blocked=True,
                    reason="blocked",
                )
            else:
                self.db.unblock_category(category)
                decision = CategoryBlockDecision(
                    category=category,
                    win_rate=win_rate,
                    sample_size=sample_size,
                    blocked=False,
                    reason="healthy",
                )
            decisions.append(decision)
            logger.info(
                "category_check category=%s sample=%s win_rate=%.4f blocked=%s",
                category,
                sample_size,
                win_rate,
                decision.blocked,
            )

        return decisions
