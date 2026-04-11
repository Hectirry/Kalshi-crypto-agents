"""
engine/signal_router.py

Orquestador del engine de señales.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import deque
from dataclasses import replace
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.config import EngineCategoryOverride, EngineConfig
from core.database import Database
from core.models import Confidence, Decision, MarketSnapshot, PriceSnapshot, Signal
from engine.context_builder import LiveSignalFeatures, build_agent_context
from engine.ev_calculator import EVCalculator
from engine.probability import ProbabilityEngine, ProbabilityResult
from engine.setup_quality import evaluate_setup_quality
from engine.timing import TimingFilter

if TYPE_CHECKING:
    from engine.openrouter_agent import OpenRouterAgent
    from intelligence.social_sentiment import SocialSentimentService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ContractSide:
    """Representa el lado del contrato que se va a evaluar económicamente."""

    decision: Decision
    contract_price: float
    my_prob: float


@dataclass(frozen=True, slots=True)
class _DecisionThresholds:
    """Umbrales efectivos para una evaluación concreta."""

    min_delta: float
    min_ev_threshold: float
    min_time_remaining_s: int
    min_contract_price: float
    max_contract_price: float


class SignalRouter:
    """Coordina timing, probabilidad y EV para construir la señal final."""

    def __init__(
        self,
        prob_engine: ProbabilityEngine,
        ev_calc: EVCalculator,
        timing_filter: TimingFilter,
        config: EngineConfig,
        db: Database,
        blocked_categories: set[str],
        openrouter_agent: OpenRouterAgent | None = None,
        social_sentiment_service: SocialSentimentService | None = None,
    ) -> None:
        """
        Inicializa el router del engine.

        Args:
            prob_engine: motor de probabilidad.
            ev_calc: calculadora de EV neto y Kelly.
            timing_filter: filtro de timing.
            config: configuración operativa del engine.
            db: base de datos para persistencia de señales.
            blocked_categories: categorías vetadas al arrancar.
            openrouter_agent: agente LLM opcional para segunda opinión en MEDIUM.
            social_sentiment_service: snapshots secundarios y cacheados de sentimiento.
        """

        self.prob_engine = prob_engine
        self.ev_calc = ev_calc
        self.timing_filter = timing_filter
        self.config = config
        self.db = db
        self.blocked_categories = blocked_categories
        self.openrouter_agent = openrouter_agent
        self.social_sentiment_service = social_sentiment_service
        self._price_memory: dict[str, deque[tuple[float, float]]] = {}
        self._price_memory_maxlen = 240

    def evaluate(
        self,
        market: MarketSnapshot,
        price: PriceSnapshot,
        bankroll: float,
    ) -> Signal:
        """
        Evalúa una oportunidad de trade y retorna una señal segura.

        Nunca propaga excepciones del engine: cualquier error se convierte
        en una señal ``ERROR``.
        """

        try:
            thresholds = self._decision_thresholds(market.category)
            if market.category in self.blocked_categories:
                self._log_skip(market.ticker, "blocked_category")
                return self._skip_signal(
                    market=market,
                    reason="blocked_category",
                    market_probability=market.implied_prob,
                )

            timing = self.timing_filter.should_enter(
                time_remaining_s=market.time_to_expiry_s,
                confidence=Confidence.HIGH,
                min_time_s=thresholds.min_time_remaining_s,
            )
            if not timing.allowed:
                self._log_skip(market.ticker, timing.reason)
                return self._skip_signal(
                    market=market,
                    reason=timing.reason,
                    market_probability=market.implied_prob,
                )

            volatility_1m = self._estimate_volatility_1m(price)
            probability = self.prob_engine.estimate(
                market=market,
                price=price,
                volatility_1m=volatility_1m,
            )
            if probability.error:
                self._log_error(market.ticker, probability.error_msg or "probability_error")
                return Signal.make_error(
                    market_ticker=market.ticker,
                    error_msg=probability.error_msg or "probability_error",
                    timestamp=market.timestamp,
                )

            timing = self.timing_filter.should_enter(
                time_remaining_s=market.time_to_expiry_s,
                confidence=probability.confidence,
                min_time_s=thresholds.min_time_remaining_s,
            )
            if not timing.allowed:
                self._log_skip(market.ticker, timing.reason)
                return self._skip_signal(
                    market=market,
                    reason=timing.reason,
                    market_probability=probability.market_prob,
                    my_probability=probability.my_prob,
                    confidence=probability.confidence,
                    delta=probability.delta,
                )

            if abs(probability.delta) < thresholds.min_delta:
                signal = self._skip_signal(
                    market=market,
                    reason="delta_too_small",
                    market_probability=probability.market_prob,
                    my_probability=probability.my_prob,
                    confidence=probability.confidence,
                    delta=probability.delta,
                )
                self.db.save_signal(signal)
                self._log_skip(market.ticker, "delta_too_small")
                return signal

            side = self._select_contract_side(market=market, probability=probability)
            if not self._is_contract_price_allowed(side.contract_price, thresholds):
                signal = self._skip_signal(
                    market=market,
                    reason="contract_price_out_of_range",
                    market_probability=probability.market_prob,
                    my_probability=probability.my_prob,
                    confidence=probability.confidence,
                    delta=probability.delta,
                )
                self.db.save_signal(signal)
                self._log_skip(market.ticker, "contract_price_out_of_range")
                return signal
            market_overround_bps = self._market_overround_bps(market)
            if market_overround_bps > self.config.max_market_overround_bps:
                signal = self._skip_signal(
                    market=market,
                    reason="market_too_wide",
                    market_probability=probability.market_prob,
                    my_probability=probability.my_prob,
                    confidence=probability.confidence,
                    delta=probability.delta,
                )
                signal.contract_price = side.contract_price
                signal.market_overround_bps = market_overround_bps
                self.db.save_signal(signal)
                self._log_skip(market.ticker, "market_too_wide")
                return signal
            contracts = self._contracts_for_bankroll(
                bankroll=bankroll,
                contract_price=side.contract_price,
            )
            ev_result = self.ev_calc.calculate(
                my_prob=side.my_prob,
                contract_price=side.contract_price,
                contracts=contracts,
                bankroll=bankroll,
            )
            if (not ev_result.is_profitable) or (
                ev_result.ev_net < thresholds.min_ev_threshold
            ):
                signal = self._skip_signal(
                    market=market,
                    reason="ev_negative",
                    market_probability=probability.market_prob,
                    my_probability=probability.my_prob,
                    confidence=probability.confidence,
                    delta=probability.delta,
                    ev_net=ev_result.ev_net,
                )
                self.db.save_signal(signal)
                self._log_skip(market.ticker, "ev_negative")
                return signal

            kelly_size = self.ev_calc.kelly_size(
                my_prob=side.my_prob,
                contract_price=side.contract_price,
                kelly_fraction=self.config.kelly_fraction,
                max_pct=self.config.max_position_pct,
            )

            signal = Signal(
                market_ticker=market.ticker,
                decision=side.decision,
                my_probability=probability.my_prob,
                market_probability=probability.market_prob,
                delta=probability.delta,
                ev_net_fees=ev_result.ev_net,
                kelly_size=kelly_size,
                confidence=probability.confidence,
                time_remaining_s=market.time_to_expiry_s,
                reasoning=self._reasoning_text(
                    decision=side.decision,
                    delta=probability.delta,
                    ev_net=ev_result.ev_net,
                    thresholds=thresholds,
                ),
                timestamp=market.timestamp,
                contract_price=side.contract_price,
                market_overround_bps=market_overround_bps,
            )
            signal = self._apply_setup_quality_gate(signal=signal, market=market)
            self.db.save_signal(signal)
            logger.info(
                "signal_generated ticker=%s decision=%s delta=%.4f ev_net=%.4f",
                market.ticker,
                side.decision.value,
                probability.delta,
                ev_result.ev_net,
            )
            return signal

        except (
            ArithmeticError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            sqlite3.Error,
        ) as exc:
            self._log_error(market.ticker, str(exc))
            return Signal.make_error(
                market_ticker=market.ticker,
                error_msg=str(exc),
                timestamp=market.timestamp,
            )

    async def evaluate_async(
        self,
        market: MarketSnapshot,
        price: PriceSnapshot,
        bankroll: float,
    ) -> Signal:
        """
        Versión async de evaluate que consulta al agente LLM para señales MEDIUM.

        Comportamiento:
          - HIGH confidence: entra directo sin consultar (velocidad).
          - MEDIUM confidence + agente configurado: consulta con timeout 3s.
          - LOW confidence: descartado por evaluate() antes de llegar aquí.
          - Si agent is None: idéntico a evaluate().

        Args:
            market: snapshot del mercado.
            price: snapshot del precio spot.
            bankroll: capital disponible en USD.

        Returns:
            Signal posiblemente modificada por el veredicto del agente.
        """
        signal = self.evaluate(market=market, price=price, bankroll=bankroll)

        if (
            self.openrouter_agent is not None
            and signal.confidence == Confidence.MEDIUM
            and signal.is_actionable
        ):
            live_features = self._build_live_features(
                signal=signal,
                market=market,
                price=price,
            )
            context = build_agent_context(
                db=self.db,
                market=market,
                signal=signal,
                live_features=live_features,
                social_sentiment=(
                    self.social_sentiment_service.get_snapshot(market.category)
                    if self.social_sentiment_service is not None
                    else None
                ),
            )
            verdict = await self.openrouter_agent.consult(
                signal=signal,
                market=market,
                price=price,
                context=context,
            )
            if not verdict.proceed:
                logger.info(
                    "agent_skip ticker=%s reasoning=%s",
                    signal.market_ticker,
                    verdict.reasoning,
                )
                return replace(
                    signal,
                    decision=Decision.SKIP,
                    kelly_size=0.0,
                    reasoning=f"agent_skip: {verdict.reasoning}",
                )
            # Aplicar kelly ajustado (nunca puede subir)
            adjusted = min(verdict.adjusted_kelly, signal.kelly_size)
            if adjusted != signal.kelly_size:
                logger.info(
                    "agent_kelly_adjust ticker=%s from=%.4f to=%.4f",
                    signal.market_ticker,
                    signal.kelly_size,
                    adjusted,
                )
                return replace(signal, kelly_size=adjusted)

        return signal

    def _apply_setup_quality_gate(
        self,
        *,
        signal: Signal,
        market: MarketSnapshot,
    ) -> Signal:
        """Bloquea setups históricamente débiles usando solo histórico resuelto local."""
        if (not self.config.setup_quality_gate_enabled) or (not signal.is_actionable):
            return signal

        context = build_agent_context(
            db=self.db,
            market=market,
            signal=signal,
            history_limit=self.config.setup_quality_history_limit,
            recent_limit=0,
        )
        verdict = evaluate_setup_quality(context, self.config)
        if verdict.allowed:
            return signal

        logger.info(
            "setup_quality_gate_skip ticker=%s n=%d wr=%.3f",
            signal.market_ticker,
            verdict.sample_size,
            verdict.win_rate,
        )
        return replace(
            signal,
            decision=Decision.SKIP,
            kelly_size=0.0,
            reasoning=f"setup_quality_gate:n={verdict.sample_size}:wr={verdict.win_rate:.2f}",
        )

    def _select_contract_side(
        self,
        market: MarketSnapshot,
        probability: ProbabilityResult,
    ) -> _ContractSide:
        """Selecciona YES o NO según el signo del delta."""

        if probability.delta > 0.0:
            return _ContractSide(
                decision=Decision.YES,
                contract_price=max(market.yes_ask, market.implied_prob),
                my_prob=probability.my_prob,
            )
        return _ContractSide(
            decision=Decision.NO,
            contract_price=max(market.no_ask, 1.0 - market.implied_prob),
            my_prob=1.0 - probability.my_prob,
        )

    def _contracts_for_bankroll(self, bankroll: float, contract_price: float) -> int:
        """Convierte el tope de exposición del config en cantidad de contratos."""

        max_notional = bankroll * self.config.max_position_pct
        safe_price = max(contract_price, 0.01)
        return max(1, int(max_notional / safe_price))

    def _decision_thresholds(self, category: str | None) -> _DecisionThresholds:
        """
        Resuelve los umbrales vigentes para una categoría.

        El backtesting escribe parámetros calibrados en SQLite. El router los
        consume aquí con fallback explícito a ``config.json`` para que el bot
        pueda operar aunque todavía no exista suficiente histórico.
        """

        params = self.db.get_current_params(category=category)
        override = self._category_override(category)
        return _DecisionThresholds(
            min_delta=max(
                params.get("min_delta", self.config.min_delta),
                override.min_delta if override.min_delta is not None else self.config.min_delta,
            ),
            min_ev_threshold=max(
                params.get("min_ev_threshold", self.config.min_ev_threshold),
                override.min_ev_threshold if override.min_ev_threshold is not None else self.config.min_ev_threshold,
            ),
            min_time_remaining_s=max(
                self.config.min_time_remaining_s,
                override.min_time_remaining_s if override.min_time_remaining_s is not None else self.config.min_time_remaining_s,
            ),
            min_contract_price=max(
                self.config.min_contract_price,
                override.min_contract_price if override.min_contract_price is not None else self.config.min_contract_price,
            ),
            max_contract_price=min(
                self.config.max_contract_price,
                override.max_contract_price if override.max_contract_price is not None else self.config.max_contract_price,
            ),
        )

    def _category_override(self, category: str | None) -> EngineCategoryOverride:
        """Retorna overrides configurados por categoría si existen."""

        if category is None:
            return EngineCategoryOverride()
        return self.config.category_overrides.get(category.upper(), EngineCategoryOverride())

    def _is_contract_price_allowed(
        self,
        contract_price: float,
        thresholds: _DecisionThresholds,
    ) -> bool:
        """Evita extremos donde fees y microestructura dominan la operación."""

        return thresholds.min_contract_price <= contract_price <= thresholds.max_contract_price

    @staticmethod
    def _skip_signal(
        market: MarketSnapshot,
        reason: str,
        market_probability: float,
        my_probability: float = 0.0,
        confidence: Confidence = Confidence.LOW,
        delta: float = 0.0,
        ev_net: float = 0.0,
    ) -> Signal:
        """Construye una señal SKIP preservando contexto útil."""

        return Signal(
            market_ticker=market.ticker,
            decision=Decision.SKIP,
            my_probability=my_probability,
            market_probability=market_probability,
            delta=delta,
            ev_net_fees=ev_net,
            kelly_size=0.0,
            confidence=confidence,
            time_remaining_s=market.time_to_expiry_s,
            reasoning=reason,
            timestamp=market.timestamp,
        )

    @staticmethod
    def _reasoning_text(
        decision: Decision,
        delta: float,
        ev_net: float,
        thresholds: _DecisionThresholds,
    ) -> str:
        """Construye un texto corto explicando la señal."""

        direction = "undervalued_yes" if decision == Decision.YES else "undervalued_no"
        return (
            f"{direction}|delta={delta:.4f}|ev_net={ev_net:.4f}|"
            f"min_delta={thresholds.min_delta:.4f}|"
            f"min_ev={thresholds.min_ev_threshold:.4f}"
        )

    def _estimate_volatility_1m(self, price: PriceSnapshot) -> float | None:
        """
        Estima volatilidad de 1 minuto a partir de retornos logarítmicos.

        Evita usar ``price.volume_1m`` como proxy de volatilidad porque algunas
        fuentes (ej. Hyperliquid) reportan allí open interest, no varianza.
        """

        series = self._price_memory.setdefault(price.symbol, deque(maxlen=self._price_memory_maxlen))
        series.append((price.timestamp, price.price))

        cutoff = price.timestamp - 60.0
        while series and series[0][0] < cutoff:
            series.popleft()

        return self._volatility_from_series(series)

    @staticmethod
    def _volatility_from_series(series: deque[tuple[float, float]]) -> float | None:
        """Calcula volatilidad logarítmica usando una serie ya preparada."""

        if len(series) < 3:
            return None

        returns: list[float] = []
        prev_price = series[0][1]
        for _ts, current_price in list(series)[1:]:
            if prev_price > 0.0 and current_price > 0.0:
                returns.append(math.log(current_price / prev_price))
            prev_price = current_price

        if len(returns) < 2:
            return None

        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)
        volatility = math.sqrt(variance)
        if volatility <= 0.0:
            return None
        return volatility

    def _volatility_from_memory(self, symbol: str) -> float | None:
        """Lee volatilidad reciente sin modificar la memoria local."""

        series = self._price_memory.get(symbol)
        if not series:
            return None
        return self._volatility_from_series(series)

    def _build_live_features(
        self,
        signal: Signal,
        market: MarketSnapshot,
        price: PriceSnapshot,
    ) -> LiveSignalFeatures:
        """Deriva KPIs vivos y compactos para que el LLM razone mejor."""

        contract_price = (
            max(market.yes_ask, market.implied_prob)
            if signal.decision == Decision.YES
            else max(market.no_ask, 1.0 - market.implied_prob)
        )
        volatility_1m = self._volatility_from_memory(price.symbol) or 0.0
        strike = market.strike
        distance_to_strike_pct = 0.0
        strike_distance_sigmas = 0.0
        if strike is not None and strike > 0.0:
            distance_to_strike_pct = (price.price - strike) / strike
            denom = price.price * max(volatility_1m, 1e-6) * math.sqrt(max(market.time_to_expiry_s, 30) / 60.0)
            if denom > 0.0:
                strike_distance_sigmas = (price.price - strike) / denom

        momentum_15s = self._price_change_pct(price.symbol, price.timestamp, window_s=15.0)
        momentum_60s = self._price_change_pct(price.symbol, price.timestamp, window_s=60.0)
        rsi_14 = self._rsi(price.symbol, period=14)
        spread_bps = self._spread_bps(price)
        trend_alignment = self._trend_alignment(signal.decision, momentum_15s, momentum_60s)
        regime_label = self._regime_label(momentum_60s, volatility_1m, spread_bps)

        return LiveSignalFeatures(
            contract_side=signal.decision.value,
            contract_price=contract_price,
            spot_price=price.price,
            strike=strike,
            distance_to_strike_pct=distance_to_strike_pct,
            strike_distance_sigmas=strike_distance_sigmas,
            realized_vol_1m=volatility_1m,
            momentum_15s_pct=momentum_15s,
            momentum_60s_pct=momentum_60s,
            rsi_14=rsi_14,
            bid_ask_spread_bps=spread_bps,
            open_interest_proxy=price.volume_1m if price.volume_1m is not None and price.volume_1m > 0 else None,
            market_skew=market.implied_prob - 0.5,
            trend_alignment=trend_alignment,
            regime_label=regime_label,
        )

    @staticmethod
    def _market_overround_bps(market: MarketSnapshot) -> float:
        """Proxy simple de anchura del libro usando la suma de asks."""
        return max(0.0, (market.yes_ask + market.no_ask - 1.0) * 10_000.0)

    def _price_change_pct(self, symbol: str, now_ts: float, window_s: float) -> float:
        """Retorna momentum porcentual simple contra un punto pasado cercano."""

        series = self._price_memory.get(symbol)
        if not series:
            return 0.0
        target_ts = now_ts - window_s
        anchor_price = None
        for ts, candidate_price in reversed(series):
            if ts <= target_ts:
                anchor_price = candidate_price
                break
        if anchor_price is None:
            anchor_price = series[0][1]
        current_price = series[-1][1]
        if anchor_price <= 0.0:
            return 0.0
        return (current_price - anchor_price) / anchor_price

    def _rsi(self, symbol: str, period: int = 14) -> float:
        """Calcula RSI corto sobre la memoria local de ticks."""

        series = self._price_memory.get(symbol)
        if not series or len(series) < period + 1:
            return 50.0

        closes = [price for _ts, price in list(series)[-(period + 1):]]
        gains = 0.0
        losses = 0.0
        for prev, current in zip(closes, closes[1:]):
            delta = current - prev
            if delta > 0:
                gains += delta
            elif delta < 0:
                losses += abs(delta)
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0.0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _spread_bps(price: PriceSnapshot) -> float:
        """Spread relativo del feed elegido si bid/ask están disponibles."""

        if price.bid is None or price.ask is None or price.price <= 0.0:
            return 0.0
        return ((price.ask - price.bid) / price.price) * 10_000.0

    @staticmethod
    def _trend_alignment(decision: Decision, momentum_15s: float, momentum_60s: float) -> str:
        """Describe si la dirección sugerida está a favor o en contra del movimiento reciente."""

        short_up = momentum_15s > 0.0
        medium_up = momentum_60s > 0.0
        if decision == Decision.YES:
            return "aligned" if short_up and medium_up else "mixed" if short_up or medium_up else "countertrend"
        return "aligned" if (not short_up and not medium_up) else "mixed" if (not short_up or not medium_up) else "countertrend"

    @staticmethod
    def _regime_label(momentum_60s: float, volatility_1m: float, spread_bps: float) -> str:
        """Resume el régimen actual para el prompt del LLM."""

        vol_bucket = "high_vol" if volatility_1m >= 0.006 else "calm"
        trend_bucket = "uptrend" if momentum_60s > 0.001 else "downtrend" if momentum_60s < -0.001 else "flat"
        micro_bucket = "wide_spread" if spread_bps >= 8.0 else "tight_spread"
        return f"{trend_bucket}|{vol_bucket}|{micro_bucket}"

    @staticmethod
    def _log_skip(ticker: str, reason: str) -> None:
        """Log estructurado de skips."""

        logger.info("signal_skip ticker=%s reason=%s", ticker, reason)

    @staticmethod
    def _log_error(ticker: str, error_msg: str) -> None:
        """Log estructurado de errores."""

        logger.error("signal_error ticker=%s error=%s", ticker, error_msg)
