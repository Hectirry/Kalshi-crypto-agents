"""
engine/probability.py

Estimación probabilística para contratos cripto de Kalshi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.models import Confidence, MarketSnapshot, PriceSnapshot


DEFAULT_VOLATILITY_1M = 0.005
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
MOMENTUM_DRIFT_SCALE = 0.35


class TimeZone:
    """
    Clasificación del mercado por tiempo restante hasta expiración.

    Los límites se alinean con los contratos crypto de 15 minutos de Kalshi:
    - ``too_early`` bloquea > 840 s → FAR cubre hasta ese tope.
    - ``low_conf_late`` bloquea LOW confidence < 300 s → NEAR lo refleja.
    """

    NEAR = "NEAR"   # 0–300 s   (< 5 min)
    MID  = "MID"    # 301–600 s (5–10 min)
    FAR  = "FAR"    # 601–840 s (10–14 min)


def classify_time_zone(time_remaining_s: int) -> str:
    """
    Clasifica el tiempo restante en zona NEAR / MID / FAR.

    Calibrado para contratos crypto de 15 minutos en Kalshi.

    Args:
        time_remaining_s: segundos hasta expiración del contrato.

    Returns:
        Una de las constantes de ``TimeZone``.
    """
    if time_remaining_s <= 300:
        return TimeZone.NEAR
    if time_remaining_s <= 600:
        return TimeZone.MID
    return TimeZone.FAR


@dataclass(frozen=True, slots=True)
class ProbabilityResult:
    """
    Resultado de la estimación probabilística del contrato.

    Attributes:
        my_prob: probabilidad propia estimada.
        market_prob: probabilidad implícita del mercado.
        delta: diferencia entre probabilidad propia y de mercado.
        confidence: fuerza del edge detectado (ya ajustada por zona).
        error: True si hubo un problema al estimar.
        error_msg: detalle del problema si ``error`` es True.
        time_zone: zona temporal del contrato (NEAR / MID / FAR).
    """

    my_prob: float
    market_prob: float
    delta: float
    confidence: Confidence
    error: bool = False
    error_msg: str | None = None
    time_zone: str = TimeZone.MID   # default mantiene compatibilidad con tests existentes


class ProbabilityEngine:
    """Estima la probabilidad real del contrato usando precio spot y strike."""

    def estimate(
        self,
        market: MarketSnapshot,
        price: PriceSnapshot,
        volatility_1m: float | None,
    ) -> ProbabilityResult:
        """
        Estima la probabilidad de que el contrato expire in-the-money.

        Args:
            market: snapshot de Kalshi con strike y tiempo restante.
            price: snapshot de precio spot del subyacente.
            volatility_1m: volatilidad observada del último minuto.

        Returns:
            ProbabilityResult con probabilidad estimada y confianza.
        """
        market_prob = market.implied_prob

        if market.time_to_expiry_s < 30:
            return ProbabilityResult(
                my_prob=market_prob,
                market_prob=market_prob,
                delta=0.0,
                confidence=Confidence.LOW,
            )

        if market.strike is None:
            return ProbabilityResult(
                my_prob=market_prob,
                market_prob=market_prob,
                delta=0.0,
                confidence=Confidence.LOW,
                error=True,
                error_msg="missing_strike",
            )

        if price.price <= 0.0:
            return ProbabilityResult(
                my_prob=market_prob,
                market_prob=market_prob,
                delta=0.0,
                confidence=Confidence.LOW,
                error=True,
                error_msg="invalid_price",
            )

        raw_volatility = (
            volatility_1m
            if volatility_1m is not None and volatility_1m > 0.0
            else DEFAULT_VOLATILITY_1M
        )
        time_scale = math.sqrt(market.time_to_expiry_s / 60.0)
        vol_adjusted = raw_volatility * time_scale

        if vol_adjusted <= 0.0:
            return ProbabilityResult(
                my_prob=market_prob,
                market_prob=market_prob,
                delta=0.0,
                confidence=Confidence.LOW,
                error=True,
                error_msg="invalid_volatility",
            )

        denominator = price.price * vol_adjusted
        if denominator <= 0.0:
            return ProbabilityResult(
                my_prob=market_prob,
                market_prob=market_prob,
                delta=0.0,
                confidence=Confidence.LOW,
                error=True,
                error_msg="invalid_denominator",
            )

        drifted_price = _apply_momentum_drift(
            spot_price=price.price,
            strike=market.strike,
            volatility_1m=raw_volatility,
            time_remaining_s=market.time_to_expiry_s,
        )
        z_score = (market.strike - drifted_price) / denominator
        my_prob = 1.0 - _norm_cdf(z_score)
        my_prob = max(MIN_PROBABILITY, min(my_prob, MAX_PROBABILITY))
        delta = my_prob - market_prob

        zone = classify_time_zone(market.time_to_expiry_s)
        confidence = _confidence_from_delta(delta)
        # Mercados FAR (> 10 min): mayor incertidumbre → bajar confianza un nivel
        # para reducir entradas en señales de largo plazo menos confiables.
        if zone == TimeZone.FAR:
            confidence = _downgrade_confidence(confidence)

        return ProbabilityResult(
            my_prob=my_prob,
            market_prob=market_prob,
            delta=delta,
            confidence=confidence,
            time_zone=zone,
        )


def _confidence_from_delta(delta: float) -> Confidence:
    """Convierte el edge en nivel discreto de confianza."""

    magnitude = abs(delta)
    if magnitude > 0.10:
        return Confidence.HIGH
    if magnitude > 0.05:
        return Confidence.MEDIUM
    return Confidence.LOW


def _downgrade_confidence(confidence: Confidence) -> Confidence:
    """Reduce la confianza un nivel. LOW permanece LOW."""

    if confidence == Confidence.HIGH:
        return Confidence.MEDIUM
    return Confidence.LOW


def _norm_cdf(value: float) -> float:
    """CDF de la normal estándar usando ``math.erf``."""

    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _apply_momentum_drift(
    *,
    spot_price: float,
    strike: float,
    volatility_1m: float,
    time_remaining_s: int,
) -> float:
    """
    Desplaza el precio terminal esperado según momentum relativo al strike.

    La idea no es predecir libremente el mercado, sino reconocer que cuando el
    spot ya se alejó del strike con fuerza relativa a la volatilidad de 1 minuto,
    la masa de probabilidad debe moverse ligeramente en esa dirección.
    """

    if spot_price <= 0.0 or strike <= 0.0 or volatility_1m <= 0.0:
        return spot_price

    price_gap_ratio = (spot_price - strike) / spot_price
    normalized_gap = price_gap_ratio / max(volatility_1m, DEFAULT_VOLATILITY_1M)
    bounded_gap = max(-1.0, min(normalized_gap, 1.0))
    time_weight = min(time_remaining_s / 900.0, 1.0)
    drift_ratio = bounded_gap * volatility_1m * MOMENTUM_DRIFT_SCALE * time_weight
    return spot_price * (1.0 + drift_ratio)
