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
    """Evalúa si un setup merece pasar según histórico resuelto comparable."""
    if not config.setup_quality_gate_enabled:
        return SetupQualityVerdict(True, "disabled", 0, 0.0)

    stats = _reference_stats(context)
    if stats.sample_size < config.setup_quality_min_samples:
        return SetupQualityVerdict(True, "insufficient_sample", stats.sample_size, stats.win_rate)
    if stats.win_rate < config.setup_quality_min_win_rate:
        return SetupQualityVerdict(False, "historical_setup_weak", stats.sample_size, stats.win_rate)
    return SetupQualityVerdict(True, "ok", stats.sample_size, stats.win_rate)


def _reference_stats(context: AgentContext) -> OutcomeStats:
    """Usa mismo setup; si falta muestra, no escala a buckets más amplios por seguridad."""
    return context.same_setup
