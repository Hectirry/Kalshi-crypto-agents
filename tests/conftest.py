"""
tests/conftest.py

Fixtures globales para toda la suite de tests.

Regla crítica: CERO llamadas reales a APIs externas aquí.
Todo lo que necesite red se mockea en este archivo.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Garantizar que el entorno es siempre demo en tests
os.environ.setdefault("ENV", "demo")
os.environ.setdefault("KALSHI_API_KEY", "test-kalshi-key")
os.environ.setdefault("KALSHI_API_SECRET", "test-kalshi-secret")
os.environ.setdefault("LOG_LEVEL", "WARNING")


# ─── Database fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Ruta a una base de datos temporal → se destruye al terminar el test."""
    return tmp_path / "test_trading.db"


@pytest.fixture
def db(db_path: Path):
    """Database inicializada y lista para usar. Se cierra al terminar el test."""
    from core.database import Database
    database = Database(path=db_path)
    database.initialize()
    yield database
    database.close()


# ─── Model factories ──────────────────────────────────────────────────────────

@pytest.fixture
def make_price_snapshot():
    """Factory de PriceSnapshot con valores por defecto sobreescribibles."""
    from core.models import PriceSnapshot

    def _factory(**kwargs):
        defaults = dict(
            symbol="BTC",
            price=95_000.0,
            timestamp=time.time(),
            source="binance",
            bid=94_999.0,
            ask=95_001.0,
        )
        return PriceSnapshot(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_market_snapshot():
    """Factory de MarketSnapshot con valores por defecto sobreescribibles."""
    from core.models import MarketSnapshot

    def _factory(**kwargs):
        defaults = dict(
            ticker="KXBTC-15MIN-B95000",
            implied_prob=0.55,
            yes_ask=0.56,
            no_ask=0.45,
            volume_24h=500,
            time_to_expiry_s=600,
            timestamp=time.time(),
            category="BTC",
            strike=95_000.0,
        )
        return MarketSnapshot(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_signal():
    """Factory de Signal con valores por defecto sobreescribibles."""
    from core.models import Confidence, Decision, Signal

    def _factory(**kwargs):
        defaults = dict(
            market_ticker="KXBTC-15MIN-B95000",
            decision=Decision.YES,
            my_probability=0.70,
            market_probability=0.55,
            delta=0.15,
            ev_net_fees=0.08,
            kelly_size=0.12,
            confidence=Confidence.HIGH,
            time_remaining_s=600,
            reasoning="Test signal",
            timestamp=time.time(),
        )
        return Signal(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_trade():
    """Factory de Trade con valores por defecto sobreescribibles."""
    from core.models import Trade, TradeMode, TradeStatus

    def _factory(**kwargs):
        defaults = dict(
            ticker="KXBTC-15MIN-B95000",
            side="YES",
            contracts=10,
            entry_price=0.56,
            mode=TradeMode.DEMO,
            status=TradeStatus.OPEN,
            opened_at=time.time(),
        )
        return Trade(**{**defaults, **kwargs})

    return _factory


# ─── Config fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def app_config():
    """AppConfig en modo demo → no requiere keys reales."""
    from core.config import load_config
    return load_config()


# ─── Mock feeds ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_price_feed(make_price_snapshot):
    """PriceFeed mockeado → no abre conexiones reales."""
    feed = AsyncMock()
    feed.source_name = "binance_mock"
    feed.is_connected = True
    feed.connect = AsyncMock()
    feed.disconnect = AsyncMock()

    async def _stream():
        for _ in range(5):
            yield make_price_snapshot()

    feed.stream = _stream
    return feed


@pytest.fixture
def mock_market_scanner(make_market_snapshot):
    """MarketScanner mockeado → no llama a Kalshi."""
    scanner = AsyncMock()
    scanner.is_connected = True
    scanner.connect = AsyncMock()
    scanner.disconnect = AsyncMock()
    scanner.get_active_markets = AsyncMock(
        return_value=[make_market_snapshot() for _ in range(3)]
    )

    async def _stream_markets():
        for _ in range(3):
            yield make_market_snapshot()

    scanner.stream_markets = _stream_markets
    return scanner
