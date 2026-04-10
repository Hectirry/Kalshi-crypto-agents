"""
engine/timing.py

Filtro temporal para entradas en contratos crypto de 15 minutos en Kalshi.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.models import Confidence

# Ventana válida de entrada para contratos de 15 min:
#   too_early  → > 840s (más de 14 min restantes: demasiado pronto)
#   too_late   → < min_time_s (default 90s: demasiado cerca del vencimiento)
#   low_conf_late → LOW confidence con < 300s (zona NEAR = demasiado arriesgado)
_MAX_TIME_S: int = 840


@dataclass(frozen=True, slots=True)
class TimingResult:
    """
    Resultado del filtro de timing.

    Attributes:
        allowed: True si todavía conviene considerar entrada.
        reason: motivo corto para logging y análisis.
    """

    allowed: bool
    reason: str


class TimingFilter:
    """Determina si el tiempo restante está dentro de la ventana operable."""

    def should_enter(
        self,
        time_remaining_s: int,
        confidence: Confidence,
        min_time_s: int,
        max_time_s: int = _MAX_TIME_S,
    ) -> TimingResult:
        """
        Evalúa si el timing del mercado permite entrada.

        Args:
            time_remaining_s: segundos hasta expiración.
            confidence: confianza actual de la señal.
            min_time_s: mínimo configurable de tiempo restante (default: 90s).
            max_time_s: máximo configurable — bloquea entrada prematura (default: 840s).

        Returns:
            TimingResult con permiso de entrada y motivo.
        """

        if time_remaining_s > max_time_s:
            return TimingResult(allowed=False, reason="too_early")
        if time_remaining_s < min_time_s:
            return TimingResult(allowed=False, reason="too_late")
        if confidence == Confidence.LOW and time_remaining_s < 300:
            return TimingResult(allowed=False, reason="low_conf_late")
        return TimingResult(allowed=True, reason="ok")
