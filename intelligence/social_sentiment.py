"""
intelligence/social_sentiment.py

Capa de sentimiento social desacoplada del loop de trading.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Protocol

from core.config import SocialSentimentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SocialSentimentSnapshot:
    """Resumen compacto y cacheable del sentimiento social por activo."""

    symbol: str
    source: str
    sentiment_score: float
    mention_count: int
    bullish_ratio: float
    bearish_ratio: float
    acceleration: float
    confidence: float
    age_seconds: int
    updated_at: float


class SocialSentimentProvider(Protocol):
    """Interfaz mockeable para recolectores de sentimiento."""

    async def fetch(self, symbols: list[str]) -> dict[str, SocialSentimentSnapshot]:
        """Recolecta snapshots frescos para los símbolos dados."""


class SocialSentimentService:
    """Mantiene snapshots en memoria + disco sin tocar el hot path."""

    def __init__(
        self,
        config: SocialSentimentConfig,
        provider: SocialSentimentProvider,
        *,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self._time_fn = time_fn or time.time
        self._snapshots: dict[str, SocialSentimentSnapshot] = {}
        self._task: asyncio.Task | None = None
        self._load_cache()

    async def start(self) -> None:
        """Arranca el refresco periódico en background si está habilitado."""
        if not self.config.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(), name="social_sentiment")

    async def stop(self) -> None:
        """Detiene la tarea de background sin propagar cancelaciones."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def get_snapshot(self, symbol: str) -> SocialSentimentSnapshot | None:
        """Retorna el snapshot actualizado en memoria, nunca toca la red."""
        if not self.config.enabled:
            return None

        raw = self._snapshots.get(symbol.upper())
        if raw is None:
            return None

        age_seconds = max(0, int(self._time_fn() - raw.updated_at))
        confidence = raw.confidence
        if age_seconds > self.config.ttl_s:
            staleness_penalty = min(0.5, (age_seconds - self.config.ttl_s) / max(self.config.ttl_s, 1))
            confidence = max(0.0, raw.confidence - staleness_penalty)

        return SocialSentimentSnapshot(
            symbol=raw.symbol,
            source=raw.source,
            sentiment_score=raw.sentiment_score,
            mention_count=raw.mention_count,
            bullish_ratio=raw.bullish_ratio,
            bearish_ratio=raw.bearish_ratio,
            acceleration=raw.acceleration,
            confidence=round(confidence, 3),
            age_seconds=age_seconds,
            updated_at=raw.updated_at,
        )

    async def refresh_once(self) -> None:
        """Refresca snapshots; ante error conserva el último estado válido."""
        if not self.config.enabled:
            return

        try:
            fetched = await self.provider.fetch(self.config.supported_assets)
        except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
            logger.warning("social_sentiment_refresh_error exc=%s", exc)
            return

        normalized: dict[str, SocialSentimentSnapshot] = {}
        for symbol in self.config.supported_assets:
            snapshot = fetched.get(symbol.upper())
            if snapshot is not None:
                normalized[symbol.upper()] = snapshot

        if not normalized:
            logger.warning("social_sentiment_refresh_empty")
            return

        self._snapshots.update(normalized)
        self._write_cache()
        logger.info("social_sentiment_refresh_ok symbols=%s", sorted(normalized))

    async def _run_loop(self) -> None:
        """Ejecuta el refresco periódico fuera del loop crítico de señales."""
        await self.refresh_once()
        while True:
            await asyncio.sleep(self.config.refresh_interval_s)
            await self.refresh_once()

    def _load_cache(self) -> None:
        """Carga snapshots previos desde disco si el caché existe."""
        path = self.config.cache_path
        if not path.exists():
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("social_sentiment_cache_load_error path=%s exc=%s", path, exc)
            return

        snapshots: dict[str, SocialSentimentSnapshot] = {}
        for symbol, raw in payload.get("snapshots", {}).items():
            try:
                snapshots[symbol.upper()] = SocialSentimentSnapshot(
                    symbol=str(raw["symbol"]).upper(),
                    source=str(raw["source"]),
                    sentiment_score=float(raw["sentiment_score"]),
                    mention_count=int(raw["mention_count"]),
                    bullish_ratio=float(raw["bullish_ratio"]),
                    bearish_ratio=float(raw["bearish_ratio"]),
                    acceleration=float(raw["acceleration"]),
                    confidence=float(raw["confidence"]),
                    age_seconds=0,
                    updated_at=float(raw["updated_at"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

        self._snapshots = snapshots

    def _write_cache(self) -> None:
        """Persiste el último snapshot con escritura atómica."""
        path = self.config.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "snapshots": {
                symbol: asdict(snapshot)
                for symbol, snapshot in self._snapshots.items()
            }
        }
        tmp_path = Path(f"{path}.tmp")
        tmp_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)


def clamp_ratio(value: float) -> float:
    """Limita un ratio a [0, 1]."""
    return max(0.0, min(1.0, value))


def clamp_score(value: float) -> float:
    """Limita un score de sentimiento a [-1, 1]."""
    return max(-1.0, min(1.0, value))


def confidence_from_counts(
    *,
    mention_count: int,
    opinionated_count: int,
    age_seconds: int,
    ttl_s: int,
) -> float:
    """Heurística simple y transparente de confianza del snapshot."""
    mention_factor = min(1.0, mention_count / 12.0)
    opinion_factor = opinionated_count / mention_count if mention_count else 0.0
    freshness_factor = max(0.0, 1.0 - (age_seconds / max(ttl_s, 1)))
    return round(clamp_ratio((0.55 * mention_factor) + (0.25 * opinion_factor) + (0.20 * freshness_factor)), 3)


def acceleration_from_windows(current_mentions: int, previous_mentions: int) -> float:
    """Mide aceleración relativa vs. ventana previa."""
    if current_mentions == 0 and previous_mentions == 0:
        return 0.0
    baseline = max(previous_mentions, 1)
    return round(clamp_score((current_mentions - previous_mentions) / baseline), 3)


def mean_or_zero(values: list[float]) -> float:
    """Promedio simple con fallback seguro."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def log_scaled_count(value: int) -> float:
    """Factor suave para ponderar scores por cantidad de menciones."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log(16))
