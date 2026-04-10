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

from core.config import EngineConfig
from core.database import Database
from core.models import Confidence, Decision, MarketSnapshot, PriceSnapshot, Signal
from engine.ev_calculator import EVCalculator
from engine.probability import ProbabilityEngine, ProbabilityResult
from engine.timing import TimingFilter

if TYPE_CHECKING:
    from engine.openrouter_agent import OpenRouterAgent

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
        """

        self.prob_engine = prob_engine
        self.ev_calc = ev_calc
        self.timing_filter = timing_filter
        self.config = config
        self.db = db
        self.blocked_categories = blocked_categories
        self.openrouter_agent = openrouter_agent
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
                min_time_s=self.config.min_time_remaining_s,
            )
            if not timing.allowed:
                self._log_skip(market.ticker, timing.reason)
                return self._skip_signal(
                    market=market,
                    reason=timing.reason,
                    market_probability=market.implied_prob,
                )

            probability = self.prob_engine.estimate(
                market=market,
                price=price,
                volatility_1m=self._estimate_volatility_1m(price),
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
                min_time_s=self.config.min_time_remaining_s,
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

            thresholds = self._decision_thresholds(market.category)

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
            )
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
            recent = self.db.get_signals(
                from_ts=signal.timestamp - 3600,
                to_ts=signal.timestamp,
                category=market.category,
                limit=5,
            )
            verdict = await self.openrouter_agent.consult(
                signal=signal,
                market=market,
                price=price,
                recent_signals=recent,
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
        return _DecisionThresholds(
            min_delta=params.get("min_delta", self.config.min_delta),
            min_ev_threshold=params.get(
                "min_ev_threshold",
                self.config.min_ev_threshold,
            ),
        )

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

    @staticmethod
    def _log_skip(ticker: str, reason: str) -> None:
        """Log estructurado de skips."""

        logger.info("signal_skip ticker=%s reason=%s", ticker, reason)

    @staticmethod
    def _log_error(ticker: str, error_msg: str) -> None:
        """Log estructurado de errores."""

        logger.error("signal_error ticker=%s error=%s", ticker, error_msg)
