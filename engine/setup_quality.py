"""
engine/setup_quality.py

Gate conservador basado en calidad histórica de setups comparables.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import EngineConfig
from engine.context_builder import AgentContext, OutcomeStats


@dataclass(frozen=True, slots=True)
class SetupQualityVerdict:
    """Decisión del gate de calidad histórica."""

    allowed: bool
    reason: str
    sample_size: int
    win_rate: float


def evaluate_setup_quality(context: AgentContext, config: EngineConfig) -> SetupQualityVerdict:
    """Evalúa si un setup merece pasar según histórico resuelto comparable.

    Además del umbral de win rate, calcula el EV histórico bruto del setup:
        historical_ev = avg_pnl_win * win_rate - avg_pnl_loss * (1 - win_rate)

    Si el win rate está por debajo del mínimo pero el EV histórico sigue siendo
    positivo, el setup se permite de todas formas.  Esto evita bloquear setups
    asimétricamente rentables (ej. 40 % win rate con ganancia media 2.5× la pérdida).
    """
    if not config.setup_quality_gate_enabled:
        return SetupQualityVerdict(True, "disabled", 0, 0.0)

    stats = _reference_stats(context)
    if stats.sample_size < config.setup_quality_min_samples:
        return SetupQualityVerdict(True, "insufficient_sample", stats.sample_size, stats.win_rate)
    if stats.win_rate < config.setup_quality_min_win_rate:
        historical_ev = (
            stats.avg_pnl_win * stats.win_rate
            - stats.avg_pnl_loss * (1.0 - stats.win_rate)
        )
        if historical_ev > 0.0:
            return SetupQualityVerdict(True, "low_wr_ev_positive", stats.sample_size, stats.win_rate)
        return SetupQualityVerdict(False, "historical_setup_weak", stats.sample_size, stats.win_rate)
    return SetupQualityVerdict(True, "ok", stats.sample_size, stats.win_rate)


def _reference_stats(context: AgentContext) -> OutcomeStats:
    """Usa mismo setup; si falta muestra, no escala a buckets más amplios por seguridad."""
    return context.same_setup
