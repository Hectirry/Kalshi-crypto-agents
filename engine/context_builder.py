"""
engine/context_builder.py

Construye contexto histórico confiable para el agente LLM a partir de
señales con outcome resuelto.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.database import Database
from core.models import Decision, MarketSnapshot, Outcome, Signal
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
    # PnL bruto promedio por resultado (por contrato, sin fees, para EV histórico).
    # avg_pnl_win  = avg(1 - contract_price) para señales WIN
    # avg_pnl_loss = avg(contract_price) para señales LOSS   (magnitud positiva)
    avg_pnl_win: float = 0.0
    avg_pnl_loss: float = 0.0


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
    win_signals = [s for s in signals if s.outcome == Outcome.WIN]
    loss_signals = [s for s in signals if s.outcome == Outcome.LOSS]
    wins = len(win_signals)
    losses = len(loss_signals)
    avg_delta = sum(signal.delta for signal in signals) / sample_size if sample_size else 0.0
    avg_ev_net = sum(signal.ev_net_fees for signal in signals) / sample_size if sample_size else 0.0

    # PnL bruto estimado por contrato (gross, sin fees) para cálculo de EV histórico.
    avg_pnl_win = (
        sum(1.0 - _contract_price_from_signal(s) for s in win_signals) / wins
        if wins else 0.0
    )
    avg_pnl_loss = (
        sum(_contract_price_from_signal(s) for s in loss_signals) / losses
        if losses else 0.0
    )

    return OutcomeStats(
        sample_size=sample_size,
        wins=wins,
        losses=losses,
        win_rate=(wins / sample_size) if sample_size else 0.0,
        avg_delta=avg_delta,
        avg_ev_net=avg_ev_net,
        avg_pnl_win=avg_pnl_win,
        avg_pnl_loss=avg_pnl_loss,
    )


def _contract_price_from_signal(signal: Signal) -> float:
    """Precio de entrada estimado para cálculo de PnL bruto histórico.

    Espeja la lógica de ``ParamInjector._effective_contract_price()`` para
    mantener coherencia entre calibración y gate de calidad.
    """
    if signal.contract_price is not None:
        return max(0.01, min(signal.contract_price, 0.99))
    if signal.decision == Decision.NO:
        return max(0.01, min(1.0 - signal.market_probability, 0.99))
    return max(0.01, min(signal.market_probability, 0.99))


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
