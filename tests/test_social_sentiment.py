"""
tests/test_social_sentiment.py

Cobertura de la capa de sentimiento social desacoplada.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest

from core.config import SocialSentimentConfig
from intelligence.reddit_provider import RedditSocialSentimentProvider
from intelligence.social_sentiment import SocialSentimentService, SocialSentimentSnapshot


def _sample_snapshot(symbol: str, updated_at: float) -> SocialSentimentSnapshot:
    return SocialSentimentSnapshot(
        symbol=symbol,
        source="reddit",
        sentiment_score=0.42,
        mention_count=14,
        bullish_ratio=0.64,
        bearish_ratio=0.14,
        acceleration=0.30,
        confidence=0.78,
        age_seconds=0,
        updated_at=updated_at,
    )


class TestSocialSentimentService:
    @pytest.mark.asyncio
    async def test_refresh_once_updates_cache_and_memory(self, tmp_path):
        config = SocialSentimentConfig(
            enabled=True,
            cache_path=tmp_path / "social_cache.json",
            supported_assets=["BTC", "ETH", "SOL"],
        )
        now = 1_710_000_000.0
        provider = AsyncMock()
        provider.fetch.return_value = {
            "BTC": _sample_snapshot("BTC", now),
            "ETH": _sample_snapshot("ETH", now),
            "SOL": _sample_snapshot("SOL", now),
        }
        service = SocialSentimentService(config=config, provider=provider, time_fn=lambda: now + 12)

        await service.refresh_once()

        btc = service.get_snapshot("BTC")
        assert btc is not None
        assert btc.age_seconds == 12
        assert btc.mention_count == 14
        assert config.cache_path.exists()
        payload = json.loads(config.cache_path.read_text(encoding="utf-8"))
        assert sorted(payload["snapshots"]) == ["BTC", "ETH", "SOL"]

    @pytest.mark.asyncio
    async def test_refresh_failure_preserves_last_good_snapshot(self, tmp_path):
        config = SocialSentimentConfig(
            enabled=True,
            ttl_s=120,
            cache_path=tmp_path / "social_cache.json",
            supported_assets=["BTC", "ETH", "SOL"],
        )
        cached = _sample_snapshot("BTC", 1_700_000_000.0)
        config.cache_path.write_text(
            json.dumps({"snapshots": {"BTC": asdict(cached)}}),
            encoding="utf-8",
        )
        provider = AsyncMock()
        provider.fetch.side_effect = RuntimeError("reddit down")
        service = SocialSentimentService(
            config=config,
            provider=provider,
            time_fn=lambda: 1_700_000_240.0,
        )

        await service.refresh_once()

        snapshot = service.get_snapshot("BTC")
        assert snapshot is not None
        assert snapshot.mention_count == cached.mention_count
        assert snapshot.age_seconds == 240
        assert snapshot.confidence < cached.confidence


class TestRedditProvider:
    def test_summarize_posts_builds_compact_metrics(self):
        config = SocialSentimentConfig(
            enabled=True,
            ttl_s=900,
            reddit_subreddits=["CryptoCurrency", "Bitcoin"],
        )
        provider = RedditSocialSentimentProvider(config=config)
        now = 1_710_000_000.0
        payload = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "BTC bullish breakout looks strong",
                            "selftext": "bitcoin may moon from here",
                            "created_utc": now - 60,
                        }
                    },
                    {
                        "data": {
                            "title": "bitcoin bearish crash warning",
                            "selftext": "short setup maybe",
                            "created_utc": now - 500,
                        }
                    },
                    {
                        "data": {
                            "title": "macro post unrelated",
                            "selftext": "not about crypto",
                            "created_utc": now - 30,
                        }
                    },
                ]
            }
        }

        snapshot = provider._summarize_posts(symbol="BTC", payload=payload, now=now)

        assert snapshot.symbol == "BTC"
        assert snapshot.source == "reddit"
        assert snapshot.mention_count == 2
        assert 0.0 <= snapshot.bullish_ratio <= 1.0
        assert 0.0 <= snapshot.bearish_ratio <= 1.0
        assert snapshot.age_seconds == 60
        assert -1.0 <= snapshot.sentiment_score <= 1.0
        assert 0.0 <= snapshot.confidence <= 1.0
