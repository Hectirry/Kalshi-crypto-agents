"""
intelligence/reddit_provider.py

Provider inicial basado en Reddit JSON oficial.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

from core.config import SocialSentimentConfig
from intelligence.social_sentiment import (
    SocialSentimentSnapshot,
    acceleration_from_windows,
    clamp_ratio,
    clamp_score,
    confidence_from_counts,
    log_scaled_count,
    mean_or_zero,
)

logger = logging.getLogger(__name__)

REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("btc", "bitcoin"),
    "ETH": ("eth", "ethereum"),
    "SOL": ("sol", "solana"),
}
POSITIVE_TERMS = {
    "bull",
    "bullish",
    "breakout",
    "moon",
    "rip",
    "long",
    "uptrend",
    "pump",
    "accumulate",
}
NEGATIVE_TERMS = {
    "bear",
    "bearish",
    "dump",
    "short",
    "rug",
    "downtrend",
    "selloff",
    "crash",
    "panic",
}


@dataclass(frozen=True, slots=True)
class RedditSocialSentimentProvider:
    """Consulta Reddit de forma controlada y resumida."""

    config: SocialSentimentConfig
    user_agent: str = "kalshi-crypto-agents/1.0"

    async def fetch(self, symbols: list[str]) -> dict[str, SocialSentimentSnapshot]:
        """Recolecta snapshots por activo usando búsquedas limitadas."""
        headers = {"User-Agent": self.user_agent}
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tasks = [self._fetch_symbol(session, symbol.upper()) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        snapshots: dict[str, SocialSentimentSnapshot] = {}
        for symbol, result in zip(symbols, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("reddit_sentiment_error symbol=%s exc=%s", symbol, result)
                continue
            snapshots[symbol.upper()] = result
        return snapshots

    async def _fetch_symbol(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> SocialSentimentSnapshot:
        """Consulta y resume posts recientes de Reddit para un activo."""
        aliases = ASSET_ALIASES.get(symbol.upper(), (symbol.lower(),))
        query = " OR ".join(aliases)
        subreddit_scope = " OR ".join(self.config.reddit_subreddits)
        if subreddit_scope:
            query = f"({query}) subreddit:({subreddit_scope})"
        params = {
            "q": query,
            "sort": "new",
            "limit": self.config.max_posts_per_asset,
            "type": "link",
            "t": "day",
            "restrict_sr": "false",
        }
        async with session.get(REDDIT_SEARCH_URL, params=params) as response:
            response.raise_for_status()
            payload = await response.json()

        return self._summarize_posts(symbol=symbol, payload=payload, now=time.time())

    def _summarize_posts(
        self,
        *,
        symbol: str,
        payload: dict,
        now: float,
    ) -> SocialSentimentSnapshot:
        """Convierte posts crudos en métricas compactas para el prompt."""
        posts = payload.get("data", {}).get("children", [])
        current_mentions = 0
        previous_mentions = 0
        opinionated_count = 0
        bullish_count = 0
        bearish_count = 0
        scores: list[float] = []
        latest_post_ts = 0.0
        aliases = ASSET_ALIASES.get(symbol.upper(), (symbol.lower(),))
        window_split = now - (self.config.ttl_s / 2.0)

        for item in posts:
            data = item.get("data", {})
            body = f"{data.get('title', '')} {data.get('selftext', '')}".lower()
            if not any(alias in body for alias in aliases):
                continue

            created_at = float(data.get("created_utc", 0.0))
            latest_post_ts = max(latest_post_ts, created_at)
            if created_at >= window_split:
                current_mentions += 1
            else:
                previous_mentions += 1

            post_score = self._score_text(body)
            if post_score > 0:
                bullish_count += 1
                opinionated_count += 1
            elif post_score < 0:
                bearish_count += 1
                opinionated_count += 1
            scores.append(post_score)

        mention_count = current_mentions + previous_mentions
        age_seconds = max(0, int(now - latest_post_ts)) if latest_post_ts > 0 else self.config.ttl_s * 2
        score_mean = mean_or_zero(scores)
        sentiment_score = round(clamp_score(score_mean * log_scaled_count(mention_count)), 3)
        bullish_ratio = clamp_ratio(bullish_count / mention_count) if mention_count else 0.0
        bearish_ratio = clamp_ratio(bearish_count / mention_count) if mention_count else 0.0
        acceleration = acceleration_from_windows(current_mentions, previous_mentions)
        confidence = confidence_from_counts(
            mention_count=mention_count,
            opinionated_count=opinionated_count,
            age_seconds=age_seconds,
            ttl_s=self.config.ttl_s,
        )

        return SocialSentimentSnapshot(
            symbol=symbol.upper(),
            source="reddit",
            sentiment_score=sentiment_score,
            mention_count=mention_count,
            bullish_ratio=round(bullish_ratio, 3),
            bearish_ratio=round(bearish_ratio, 3),
            acceleration=acceleration,
            confidence=confidence,
            age_seconds=age_seconds,
            updated_at=now,
        )

    def _score_text(self, text: str) -> float:
        """Heurística léxica simple: transparente, barata y fácil de testear."""
        positive_hits = sum(1 for term in POSITIVE_TERMS if term in text)
        negative_hits = sum(1 for term in NEGATIVE_TERMS if term in text)
        total_hits = positive_hits + negative_hits
        if total_hits == 0:
            return 0.0
        return clamp_score((positive_hits - negative_hits) / total_hits)
