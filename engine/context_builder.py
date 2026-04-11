"""
engine/context_builder.py

Construye contexto histórico confiable para el agente LLM a partir de
señales con outcome resuelto.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.database import Database
from core.models import MarketSnapshot, Outcome, Signal
from engine.probability import classify_time_zone
from intelligence.social_sentiment import SocialSentimentSnapshot


@dataclass(frozen=True, slots=True)
class OutcomeStats:
    """Agregado simple sobre señales con outcome conocido."""

    sample_size: int
    wins: int
    losses: int
    win_rate: float
    avg_delta: float
    avg_ev_net: float


@dataclass(frozen=True, slots=True)
class ResolvedSignalSummary:
    """Resumen compacto de una señal histórica para prompts."""

    ticker: str
    decision: str
    outcome: str
    delta: float
    ev_net_fees: float
    time_remaining_s: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class LiveSignalFeatures:
    """KPIs vivos para enriquecer la consulta del agente LLM."""

    contract_side: str
    contract_price: float
    spot_price: float
    strike: float | None
    distance_to_strike_pct: float
    strike_distance_sigmas: float
    realized_vol_1m: float
    momentum_15s_pct: float
    momentum_60s_pct: float
    rsi_14: float
    bid_ask_spread_bps: float
    open_interest_proxy: float | None
    market_skew: float
    trend_alignment: str
    regime_label: str


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Contexto histórico agregado para consulta del agente."""

    category: str
    time_zone: str
    overall: OutcomeStats
    same_category: OutcomeStats
    same_time_zone: OutcomeStats
    same_ticker: OutcomeStats
    same_direction: OutcomeStats
    same_setup: OutcomeStats
    recent_same_ticker: list[ResolvedSignalSummary]
    recent_same_category: list[ResolvedSignalSummary]
    recent_same_setup: list[ResolvedSignalSummary]
    live_features: LiveSignalFeatures | None = None
    social_sentiment: SocialSentimentSnapshot | None = None


def build_agent_context(
    *,
    db: Database,
    market: MarketSnapshot,
    signal: Signal,
    history_limit: int = 500,
    recent_limit: int = 5,
    live_features: LiveSignalFeatures | None = None,
    social_sentiment: SocialSentimentSnapshot | None = None,
) -> AgentContext:
    """Construye contexto únicamente desde señales con outcome real."""

    resolved = [
        entry
        for entry in db.get_signals(from_ts=0, to_ts=signal.timestamp, limit=history_limit)
        if entry.outcome is not None and entry.timestamp < signal.timestamp
    ]
    same_category = [entry for entry in resolved if market.category in entry.market_ticker]
    target_zone = classify_time_zone(market.time_to_expiry_s)
    same_time_zone = [
        entry for entry in same_category if classify_time_zone(entry.time_remaining_s) == target_zone
    ]
    same_ticker = [entry for entry in resolved if entry.market_ticker == market.ticker]
    same_direction = [
        entry for entry in same_category if entry.decision == signal.decision
    ]
    target_bucket = _delta_bucket(signal.delta)
    same_setup = [
        entry
        for entry in same_direction
        if classify_time_zone(entry.time_remaining_s) == target_zone
        and _delta_bucket(entry.delta) == target_bucket
    ]

    return AgentContext(
        category=market.category,
        time_zone=target_zone,
        overall=_stats_from_signals(resolved),
        same_category=_stats_from_signals(same_category),
        same_time_zone=_stats_from_signals(same_time_zone),
        same_ticker=_stats_from_signals(same_ticker),
        same_direction=_stats_from_signals(same_direction),
        same_setup=_stats_from_signals(same_setup),
        recent_same_ticker=_recent_summaries(same_ticker, recent_limit),
        recent_same_category=_recent_summaries(same_category, recent_limit),
        recent_same_setup=_recent_summaries(same_setup, recent_limit),
        live_features=live_features,
        social_sentiment=social_sentiment,
    )


def _stats_from_signals(signals: list[Signal]) -> OutcomeStats:
    """Agrega métricas de desempeño sobre señales resueltas."""

    sample_size = len(signals)
    wins = sum(1 for signal in signals if signal.outcome == Outcome.WIN)
    losses = sum(1 for signal in signals if signal.outcome == Outcome.LOSS)
    avg_delta = sum(signal.delta for signal in signals) / sample_size if sample_size else 0.0
    avg_ev_net = sum(signal.ev_net_fees for signal in signals) / sample_size if sample_size else 0.0
    return OutcomeStats(
        sample_size=sample_size,
        wins=wins,
        losses=losses,
        win_rate=(wins / sample_size) if sample_size else 0.0,
        avg_delta=avg_delta,
        avg_ev_net=avg_ev_net,
    )


def _recent_summaries(signals: list[Signal], limit: int) -> list[ResolvedSignalSummary]:
    """Reduce señales históricas a una lista pequeña para el prompt."""

    recent = sorted(signals, key=lambda entry: entry.timestamp, reverse=True)[:limit]
    return [
        ResolvedSignalSummary(
            ticker=entry.market_ticker,
            decision=entry.decision.value,
            outcome=entry.outcome.value,
            delta=entry.delta,
            ev_net_fees=entry.ev_net_fees,
            time_remaining_s=entry.time_remaining_s,
            timestamp=entry.timestamp,
        )
        for entry in recent
        if entry.outcome is not None
    ]


def _delta_bucket(delta: float) -> str:
    """Agrupa magnitud del edge para comparar setups similares."""

    magnitude = abs(delta)
    if magnitude < 0.08:
        return "small"
    if magnitude < 0.16:
        return "medium"
    return "large"
