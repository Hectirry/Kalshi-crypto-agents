"""
engine/openrouter_agent.py

Agente LLM como segunda opinión vía OpenRouter.
Completamente opcional y no bloqueante.

Reglas de activación:
  - Solo si OPENROUTER_API_KEY está en el entorno
  - Solo si confidence == MEDIUM
  - Timeout 3s — si no responde, la señal pasa sin cambios
  - HIGH confidence: entra directo sin consultar (velocidad)
  - LOW confidence: se descarta sin consultar (ahorro de tokens)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import aiohttp

from core.models import MarketSnapshot, PriceSnapshot, Signal
from engine.context_builder import AgentContext

logger = logging.getLogger(__name__)

AGENT_TIMEOUT_S: float = 3.0
DEFAULT_MODEL: str = "anthropic/claude-haiku-4-5"
OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True, slots=True)
class AgentVerdict:
    """Veredicto del agente LLM sobre una señal MEDIUM."""

    proceed: bool
    adjusted_kelly: float   # puede reducir kelly, nunca aumentar
    reasoning: str
    tokens_used: int


class OpenRouterAgent:
    """
    Agente LLM opcional para segunda opinión en señales MEDIUM.

    Consulta a un modelo vía OpenRouter con timeout estricto de 3 segundos.
    Cualquier fallo (timeout, red, parseo) retorna proceed=True sin cambios.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        """
        Inicializa el agente.

        Args:
            api_key: clave de OpenRouter API.
            model: modelo a consultar (default: anthropic/claude-haiku-4-5).
        """
        self._api_key = api_key
        self._model = model

    async def consult(
        self,
        signal: Signal,
        market: MarketSnapshot,
        price: PriceSnapshot,
        context: AgentContext,
    ) -> AgentVerdict:
        """
        Consulta al agente LLM para segunda opinión sobre una señal MEDIUM.

        Timeout estricto de 3s. Ante cualquier fallo retorna
        proceed=True con kelly sin cambios — la señal pasa sin modificar.

        Args:
            signal: señal del engine a validar.
            market: snapshot del mercado en el momento de la señal.
            price: snapshot del precio spot.
            context: contexto histórico construido desde señales resueltas.

        Returns:
            AgentVerdict con la decisión del agente.
        """
        try:
            return await asyncio.wait_for(
                self._call_api(
                    signal=signal,
                    market=market,
                    price=price,
                    context=context,
                ),
                timeout=AGENT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "openrouter_timeout ticker=%s kelly=%.4f",
                signal.market_ticker,
                signal.kelly_size,
            )
            return AgentVerdict(
                proceed=True,
                adjusted_kelly=signal.kelly_size,
                reasoning="timeout — señal pasa sin cambios",
                tokens_used=0,
            )
        except (aiohttp.ClientError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "openrouter_error ticker=%s exc=%s",
                signal.market_ticker,
                exc,
            )
            return AgentVerdict(
                proceed=True,
                adjusted_kelly=signal.kelly_size,
                reasoning=f"error_fallback: {exc}",
                tokens_used=0,
            )

    def _build_prompt(
        self,
        signal: Signal,
        market: MarketSnapshot,
        price: PriceSnapshot,
        context: AgentContext,
    ) -> list[dict[str, str]]:
        """Construye mensajes compactos y estructurados para el modelo."""
        strike = market.strike or "N/A"
        ticker_examples = ", ".join(
            (
                f"{entry.outcome}/{entry.decision}/delta={entry.delta:.3f}/ev={entry.ev_net_fees:.3f}"
                for entry in context.recent_same_ticker
            )
        ) or "sin historial resuelto"
        category_examples = ", ".join(
            (
                f"{entry.ticker}:{entry.outcome}/{entry.decision}/delta={entry.delta:.3f}"
                for entry in context.recent_same_category
            )
        ) or "sin historial resuelto"
        setup_examples = ", ".join(
            (
                f"{entry.outcome}/{entry.decision}/delta={entry.delta:.3f}/ev={entry.ev_net_fees:.3f}"
                for entry in context.recent_same_setup
            )
        ) or "sin analogos cercanos"

        feature_lines = ["Features vivas: no disponibles"]
        if context.live_features is not None:
            features = context.live_features
            open_interest = (
                f"{features.open_interest_proxy:.2f}"
                if features.open_interest_proxy is not None
                else "n/a"
            )
            feature_lines = [
                "Features vivas:",
                (
                    f"- side={features.contract_side} contract_price={features.contract_price:.3f} "
                    f"market_skew={features.market_skew:.3f}"
                ),
                (
                    f"- distance_to_strike_pct={features.distance_to_strike_pct:.4%} "
                    f"strike_distance_sigmas={features.strike_distance_sigmas:.2f}"
                ),
                (
                    f"- realized_vol_1m={features.realized_vol_1m:.5f} "
                    f"mom_15s={features.momentum_15s_pct:.4%} "
                    f"mom_60s={features.momentum_60s_pct:.4%} "
                    f"rsi_14={features.rsi_14:.1f}"
                ),
                (
                    f"- spread_bps={features.bid_ask_spread_bps:.2f} "
                    f"open_interest_proxy={open_interest}"
                ),
                (
                    f"- trend_alignment={features.trend_alignment} "
                    f"regime_label={features.regime_label}"
                ),
            ]

        social_lines = [
            "Sentimiento social: no disponible o desactivado; ignóralo sin inferir nada."
        ]
        if context.social_sentiment is not None:
            sentiment = context.social_sentiment
            social_lines = [
                "Contexto social secundario:",
                (
                    f"- source={sentiment.source} sentiment_score={sentiment.sentiment_score:.3f} "
                    f"mentions={sentiment.mention_count} confidence={sentiment.confidence:.2f}"
                ),
                (
                    f"- bullish_ratio={sentiment.bullish_ratio:.1%} "
                    f"bearish_ratio={sentiment.bearish_ratio:.1%} "
                    f"acceleration={sentiment.acceleration:.3f}"
                ),
                f"- age_seconds={sentiment.age_seconds}",
            ]

        system_prompt = (
            "Eres un revisor cuantitativo de riesgo para mercados crypto de 15 minutos en Kalshi. "
            "No inventes datos. Usa primero evidencia resuelta comparable y luego features vivas. "
            "Pesa fuerte: delta, EV neto, distancia al strike en sigmas, volatilidad realizada, "
            "momentum 15s/60s, alineación de tendencia, spread y análogos históricos del mismo setup. "
            "Si hay contexto social, úsalo solo como señal secundaria de confirmación o cautela. "
            "Pesa poco RSI si contradice el resto. No aumentes kelly; solo mantenlo o bájalo. "
            "Si faltan datos externos o de sentimiento social, ignóralos en vez de alucinar."
        )
        user_prompt = "\n".join(
            [
                "Contexto: mercado Kalshi crypto de 15 minutos.",
                f"Activo: {market.ticker} | Strike: {strike} | Precio spot: {price.price:.2f}",
                f"Tiempo restante: {market.time_to_expiry_s}s",
                (
                    f"Decision base: {signal.decision.value} | Delta: {signal.delta:.3f} | "
                    f"EV neto: {signal.ev_net_fees:.3f} | Kelly base: {signal.kelly_size:.3f}"
                ),
                "Evalúa solo con evidencia histórica resuelta (WIN/LOSS reales) y KPIs presentes.",
                (
                    f"Global resuelto: n={context.overall.sample_size} wr={context.overall.win_rate:.1%} "
                    f"avg_delta={context.overall.avg_delta:.3f} avg_ev={context.overall.avg_ev_net:.3f}"
                ),
                f"Categoría {context.category}: n={context.same_category.sample_size} wr={context.same_category.win_rate:.1%}",
                f"Misma ventana {context.time_zone}: n={context.same_time_zone.sample_size} wr={context.same_time_zone.win_rate:.1%}",
                f"Mismo ticker: n={context.same_ticker.sample_size} wr={context.same_ticker.win_rate:.1%}",
                (
                    f"Misma dirección {signal.decision.value}: n={context.same_direction.sample_size} "
                    f"wr={context.same_direction.win_rate:.1%}"
                ),
                (
                    f"Mismo setup (dirección + bucket de delta + ventana): "
                    f"n={context.same_setup.sample_size} wr={context.same_setup.win_rate:.1%}"
                ),
                *feature_lines,
                *social_lines,
                f"Últimos resueltos mismo ticker: {ticker_examples}",
                f"Últimos resueltos misma categoría: {category_examples}",
                f"Últimos resueltos mismo setup: {setup_examples}",
                "Responde SOLO con JSON válido:",
                f'{{"proceed": true, "adjusted_kelly": {signal.kelly_size:.2f}, "reasoning": "motivo corto y concreto"}}',
            ]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _call_api(
        self,
        signal: Signal,
        market: MarketSnapshot,
        price: PriceSnapshot,
        context: AgentContext,
    ) -> AgentVerdict:
        """Llama al API de OpenRouter y parsea el veredicto."""
        messages = self._build_prompt(
            signal=signal,
            market=market,
            price=price,
            context=context,
        )
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 140,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kalshi-crypto-agents",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        content = data["choices"][0]["message"]["content"].strip()
        tokens_used: int = data.get("usage", {}).get("total_tokens", 0)

        parsed = json.loads(content)
        proceed = bool(parsed.get("proceed", True))
        # El agente nunca puede aumentar el kelly — solo reducirlo o mantenerlo
        raw_kelly = float(parsed.get("adjusted_kelly", signal.kelly_size))
        adjusted_kelly = min(raw_kelly, signal.kelly_size)
        reasoning = str(parsed.get("reasoning", ""))

        logger.info(
            "openrouter_verdict ticker=%s proceed=%s kelly=%.4f reasoning=%s",
            signal.market_ticker,
            proceed,
            adjusted_kelly,
            reasoning,
        )
        return AgentVerdict(
            proceed=proceed,
            adjusted_kelly=adjusted_kelly,
            reasoning=reasoning,
            tokens_used=tokens_used,
        )
