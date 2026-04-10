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
        recent_signals: list[Signal],
    ) -> AgentVerdict:
        """
        Consulta al agente LLM para segunda opinión sobre una señal MEDIUM.

        Timeout estricto de 3s. Ante cualquier fallo retorna
        proceed=True con kelly sin cambios — la señal pasa sin modificar.

        Args:
            signal: señal del engine a validar.
            market: snapshot del mercado en el momento de la señal.
            price: snapshot del precio spot.
            recent_signals: últimas señales del mismo ticker (máx 5).

        Returns:
            AgentVerdict con la decisión del agente.
        """
        try:
            return await asyncio.wait_for(
                self._call_api(
                    signal=signal,
                    market=market,
                    price=price,
                    recent_signals=recent_signals,
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
        recent_signals: list[Signal],
    ) -> str:
        """Construye el prompt de consulta al modelo en español."""
        ganadas = sum(
            1 for s in recent_signals if s.outcome and s.outcome.value == "WIN"
        )
        perdidas = sum(
            1 for s in recent_signals if s.outcome and s.outcome.value == "LOSS"
        )
        strike = market.strike or "N/A"

        return (
            "Contexto: mercado Kalshi de 15 minutos.\n"
            f"Activo: {market.ticker} | Strike: {strike} | Precio spot: {price.price:.2f}\n"
            f"Tiempo restante: {market.time_to_expiry_s}s\n"
            f"Delta calculado: {signal.delta:.3f} | EV neto: {signal.ev_net_fees:.3f}\n"
            f"Historial reciente ({len(recent_signals)} señales): "
            f"{ganadas} ganadas / {perdidas} perdidas\n"
            "\n"
            f"¿Confirmas la entrada con kelly={signal.kelly_size:.2f}?\n"
            "Responde SOLO con JSON válido:\n"
            f'{{"proceed": true, "adjusted_kelly": {signal.kelly_size:.2f}, "reasoning": "motivo corto"}}'
        )

    async def _call_api(
        self,
        signal: Signal,
        market: MarketSnapshot,
        price: PriceSnapshot,
        recent_signals: list[Signal],
    ) -> AgentVerdict:
        """Llama al API de OpenRouter y parsea el veredicto."""
        prompt = self._build_prompt(
            signal=signal,
            market=market,
            price=price,
            recent_signals=recent_signals,
        )
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 100,
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
