"""
tests/test_feeds.py

Tests para BinancePriceFeed, HyperliquidFeed y KalshiFeed.
CERO llamadas reales a APIs — todo mockeado.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets.exceptions

from core.interfaces import EventBus
from core.config import BinanceConfig, HyperliquidConfig


# ══════════════════════════════════════════════════════════════════════════════
# Helpers compartidos
# ══════════════════════════════════════════════════════════════════════════════

def _make_ws_mock(recv_return=None, recv_side_effect=None):
    """Crea un mock de WebSocket con send/recv/close como AsyncMock."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    if recv_side_effect is not None:
        ws.recv = AsyncMock(side_effect=recv_side_effect)
    elif recv_return is not None:
        ws.recv = AsyncMock(return_value=recv_return)
    return ws


def _make_aiohttp_response(json_data, status=200):
    """Crea un mock de respuesta aiohttp compatible con async context manager."""
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status = status

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_cm


def _make_aiohttp_session(json_data=None, post_side_effect=None, get_side_effect=None):
    """Crea un mock de aiohttp.ClientSession."""
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    if post_side_effect is not None:
        session.post = MagicMock(side_effect=post_side_effect)
    elif json_data is not None:
        session.post = MagicMock(return_value=_make_aiohttp_response(json_data))

    if get_side_effect is not None:
        session.get = MagicMock(side_effect=get_side_effect)
    elif json_data is not None:
        session.get = MagicMock(return_value=_make_aiohttp_response(json_data))

    return session


_BOOK_TICKER_BTC = json.dumps({"s": "BTCUSDT", "b": "95000.0", "a": "95001.0"})
_BOOK_TICKER_ETH = json.dumps({"s": "ETHUSDT", "b": "3000.0", "a": "3001.0"})

_HL_RESPONSE = [
    {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
    [
        {"markPx": "95000.0", "openInterest": "1234.5"},
        {"markPx": "3000.0", "openInterest": "567.8"},
    ],
]

_KALSHI_MARKETS_RESPONSE = {
    "markets": [
        {
            "ticker": "KXBTC-15MIN-B95000",
            "yes_ask": 56,
            "no_ask": 44,
            "volume": 500,
            "close_time": "2099-01-01T00:10:00Z",   # muy lejos → ~expiry grande pero ≤ 900 no aplica
        },
    ]
}


# ══════════════════════════════════════════════════════════════════════════════
# BinancePriceFeed
# ══════════════════════════════════════════════════════════════════════════════

class TestBinancePriceFeed:

    def _make_feed(self, ws_url="wss://stream.binance.com:9443/ws"):
        from feeds.binance_feed import BinancePriceFeed
        config = BinanceConfig(
            api_key=None,
            ws_url=ws_url,
            symbols=["BTCUSDT", "ETHUSDT"],
        )
        bus = EventBus()
        return BinancePriceFeed(config=config, bus=bus), bus

    # ── connect ───────────────────────────────────────────────────────────────

    async def test_connect_establishes_websocket(self):
        feed, _ = self._make_feed()
        mock_ws = _make_ws_mock()

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await feed.connect()

        assert feed.is_connected is True
        assert feed._ws is mock_ws

    async def test_connect_subscribes_to_book_ticker(self):
        feed, _ = self._make_feed()
        mock_ws = _make_ws_mock()

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await feed.connect()

        mock_ws.send.assert_called_once()
        sent_msg = json.loads(mock_ws.send.call_args[0][0])
        assert sent_msg["method"] == "SUBSCRIBE"
        assert "btcusdt@bookTicker" in sent_msg["params"]
        assert "ethusdt@bookTicker" in sent_msg["params"]

    async def test_connect_retries_on_failure_then_succeeds(self):
        feed, _ = self._make_feed()
        mock_ws = _make_ws_mock()

        # Falla en el primer intento, éxito en el segundo
        with patch(
            "websockets.connect",
            new_callable=AsyncMock,
            side_effect=[OSError("refused"), mock_ws],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await feed.connect()

        assert feed.is_connected is True

    async def test_connect_raises_connection_error_after_max_attempts(self):
        feed, _ = self._make_feed()

        with patch(
            "websockets.connect",
            new_callable=AsyncMock,
            side_effect=OSError("refused"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ConnectionError, match="5 intentos"):
                    await feed.connect()

        assert feed.is_connected is False

    # ── disconnect ────────────────────────────────────────────────────────────

    async def test_disconnect_closes_websocket(self):
        feed, _ = self._make_feed()
        mock_ws = _make_ws_mock()
        feed._ws = mock_ws
        feed._connected = True

        await feed.disconnect()

        mock_ws.close.assert_called_once()
        assert feed.is_connected is False
        assert feed._ws is None

    async def test_disconnect_does_not_raise_if_ws_close_errors(self):
        feed, _ = self._make_feed()
        mock_ws = _make_ws_mock()
        mock_ws.close = AsyncMock(side_effect=RuntimeError("already closed"))
        feed._ws = mock_ws
        feed._connected = True

        # No debe propagar la excepción
        await feed.disconnect()

        assert feed.is_connected is False

    async def test_disconnect_when_not_connected_is_safe(self):
        feed, _ = self._make_feed()
        await feed.disconnect()  # _ws es None → no explota

    # ── stream ────────────────────────────────────────────────────────────────

    async def test_stream_emits_price_snapshot(self):
        feed, _ = self._make_feed()
        mock_ws = _make_ws_mock(recv_return=_BOOK_TICKER_BTC)
        feed._ws = mock_ws
        feed._connected = True

        gen = feed.stream()
        snapshot = await anext(gen)
        await gen.aclose()

        assert snapshot.symbol == "BTC"
        assert snapshot.source == "binance"
        assert snapshot.bid == pytest.approx(95000.0)
        assert snapshot.ask == pytest.approx(95001.0)
        assert snapshot.price == pytest.approx(95000.5)

    async def test_stream_emits_eth_snapshot(self):
        feed, _ = self._make_feed()
        mock_ws = _make_ws_mock(recv_return=_BOOK_TICKER_ETH)
        feed._ws = mock_ws
        feed._connected = True

        gen = feed.stream()
        snapshot = await anext(gen)
        await gen.aclose()

        assert snapshot.symbol == "ETH"
        assert snapshot.bid == pytest.approx(3000.0)

    async def test_stream_publishes_to_event_bus(self):
        feed, bus = self._make_feed()
        mock_ws = _make_ws_mock(recv_return=_BOOK_TICKER_BTC)
        feed._ws = mock_ws
        feed._connected = True

        gen = feed.stream()
        await anext(gen)
        await gen.aclose()

        assert bus.qsize == 1
        event = bus._queue.get_nowait()
        assert event.symbol == "BTC"
        assert event.source == "binance"

    async def test_stream_skips_subscription_confirmation_messages(self):
        feed, _ = self._make_feed()
        # Mensaje de confirmación (sin "s"), luego mensaje real
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"result": None, "id": 1}),  # confirmación
                _BOOK_TICKER_BTC,
            ]
        )
        feed._ws = mock_ws
        feed._connected = True

        gen = feed.stream()
        snapshot = await anext(gen)
        await gen.aclose()

        assert snapshot.symbol == "BTC"

    async def test_stream_reconnects_on_connection_closed(self):
        feed, _ = self._make_feed()

        ws_after_reconnect = _make_ws_mock(recv_return=_BOOK_TICKER_BTC)

        # ws inicial: recv lanza ConnectionClosed
        feed._ws = _make_ws_mock(
            recv_side_effect=websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)
        )
        feed._connected = True

        with patch(
            "websockets.connect",
            new_callable=AsyncMock,
            return_value=ws_after_reconnect,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                gen = feed.stream()
                snapshot = await anext(gen)
                await gen.aclose()

        assert snapshot.symbol == "BTC"

    async def test_stream_raises_connection_error_after_max_reconnects(self):
        feed, _ = self._make_feed()
        feed._ws = _make_ws_mock(
            recv_side_effect=websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)
        )
        feed._connected = True

        with patch(
            "websockets.connect",
            new_callable=AsyncMock,
            side_effect=OSError("down"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ConnectionError):
                    gen = feed.stream()
                    await anext(gen)

    # ── source_name ───────────────────────────────────────────────────────────

    def test_source_name(self):
        from feeds.binance_feed import BinancePriceFeed
        config = BinanceConfig(api_key=None, ws_url="wss://x", symbols=[])
        feed = BinancePriceFeed(config=config, bus=EventBus())
        assert feed.source_name == "binance"


# ══════════════════════════════════════════════════════════════════════════════
# HyperliquidFeed
# ══════════════════════════════════════════════════════════════════════════════

class TestHyperliquidFeed:

    def _make_feed(self):
        from feeds.hyperliquid_feed import HyperliquidFeed
        config = HyperliquidConfig(
            api_key=None,
            base_url="https://api.hyperliquid.xyz",
            symbols=["BTC", "ETH"],
        )
        bus = EventBus()
        return HyperliquidFeed(config=config, bus=bus), bus

    # ── connect ───────────────────────────────────────────────────────────────

    async def test_connect_succeeds(self):
        feed, _ = self._make_feed()

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session()):
            feed._fetch_once = AsyncMock(return_value=[])
            await feed.connect()

        assert feed._running is True

    async def test_connect_retries_on_failure_then_succeeds(self):
        feed, _ = self._make_feed()

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session()):
            call_count = 0

            async def flaky_fetch():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("timeout")
                return []

            feed._fetch_once = flaky_fetch
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await feed.connect()

        assert feed._running is True
        assert call_count == 2

    async def test_connect_raises_connection_error_after_max_attempts(self):
        feed, _ = self._make_feed()

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session()):
            feed._fetch_once = AsyncMock(side_effect=OSError("down"))
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ConnectionError, match="5 intentos"):
                    await feed.connect()

    # ── disconnect ────────────────────────────────────────────────────────────

    async def test_disconnect_closes_session(self):
        feed, _ = self._make_feed()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        feed._session = mock_session
        feed._running = True

        await feed.disconnect()

        mock_session.close.assert_called_once()
        assert feed._running is False
        assert feed._session is None

    async def test_disconnect_when_not_connected_is_safe(self):
        feed, _ = self._make_feed()
        await feed.disconnect()  # _session es None → no explota

    # ── stream ────────────────────────────────────────────────────────────────

    async def test_stream_emits_price_snapshots(self):
        feed, bus = self._make_feed()
        feed._running = True

        from core.models import PriceSnapshot
        snap_btc = PriceSnapshot(
            symbol="BTC", price=95000.0, timestamp=time.time(), source="hyperliquid"
        )
        snap_eth = PriceSnapshot(
            symbol="ETH", price=3000.0, timestamp=time.time(), source="hyperliquid"
        )

        fetched = []

        async def mock_fetch():
            fetched.append(1)
            if len(fetched) == 1:
                return [snap_btc, snap_eth]
            feed._running = False  # detiene el loop en la segunda llamada
            return []

        feed._fetch_once = mock_fetch

        with patch("asyncio.sleep", new_callable=AsyncMock):
            results = []
            async for snap in feed.stream():
                results.append(snap)

        assert len(results) == 2
        assert results[0].symbol == "BTC"
        assert results[1].symbol == "ETH"

    async def test_stream_publishes_to_event_bus(self):
        feed, bus = self._make_feed()
        feed._running = True

        from core.models import PriceSnapshot
        snap = PriceSnapshot(
            symbol="BTC", price=95000.0, timestamp=time.time(), source="hyperliquid"
        )

        call_count = 0

        async def mock_fetch():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [snap]
            feed._running = False
            return []

        feed._fetch_once = mock_fetch

        with patch("asyncio.sleep", new_callable=AsyncMock):
            async for _ in feed.stream():
                pass

        assert bus.qsize == 1

    async def test_stream_retries_on_single_failure(self):
        feed, _ = self._make_feed()
        feed._running = True

        call_count = 0

        from core.models import PriceSnapshot
        snap = PriceSnapshot(
            symbol="BTC", price=95000.0, timestamp=time.time(), source="hyperliquid"
        )

        async def flaky_fetch():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("timeout")
            if call_count == 2:
                return [snap]
            feed._running = False
            return []

        feed._fetch_once = flaky_fetch

        with patch("asyncio.sleep", new_callable=AsyncMock):
            results = []
            async for s in feed.stream():
                results.append(s)

        assert len(results) == 1
        assert results[0].symbol == "BTC"

    async def test_stream_raises_connection_error_after_max_failures(self):
        feed, _ = self._make_feed()
        feed._running = True
        feed._fetch_once = AsyncMock(side_effect=OSError("down"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError):
                async for _ in feed.stream():
                    pass

    # ── _fetch_once (integración parseo) ─────────────────────────────────────

    async def test_fetch_once_parses_mark_price(self):
        feed, _ = self._make_feed()
        mock_session = _make_aiohttp_session(json_data=_HL_RESPONSE)
        feed._session = mock_session

        snapshots = await feed._fetch_once()

        assert len(snapshots) == 2
        btc = next(s for s in snapshots if s.symbol == "BTC")
        eth = next(s for s in snapshots if s.symbol == "ETH")

        assert btc.price == pytest.approx(95000.0)
        assert btc.source == "hyperliquid"
        assert btc.volume_1m == pytest.approx(1234.5)

        assert eth.price == pytest.approx(3000.0)

    def test_source_name(self):
        from feeds.hyperliquid_feed import HyperliquidFeed
        config = HyperliquidConfig(api_key=None, base_url="https://x", symbols=[])
        feed = HyperliquidFeed(config=config, bus=EventBus())
        assert feed.source_name == "hyperliquid"


# ══════════════════════════════════════════════════════════════════════════════
# KalshiFeed
# ══════════════════════════════════════════════════════════════════════════════

class TestKalshiFeed:

    def _make_feed(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        bus = EventBus()
        return KalshiFeed(config=app_config, bus=bus), bus

    def _market_raw(
        self,
        ticker="KXBTC-15MIN-B95000",
        yes_ask=56,
        no_ask=44,
        volume=500,
        close_time="2099-01-01T00:10:00Z",
    ):
        return {
            "ticker": ticker,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "volume": volume,
            "close_time": close_time,
        }

    # ── connect ───────────────────────────────────────────────────────────────

    async def test_connect_establishes_websocket(self, app_config):
        feed, _ = self._make_feed(app_config)
        mock_ws = _make_ws_mock()

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session()):
            with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
                await feed.connect()

        assert feed._connected is True
        assert feed._ws is mock_ws

    async def test_connect_subscribes_to_ticker_channel(self, app_config):
        feed, _ = self._make_feed(app_config)
        mock_ws = _make_ws_mock()

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session()):
            with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
                await feed.connect()

        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert "ticker" in sent["params"]["channels"]

    async def test_connect_raises_connection_error_after_max_attempts(self, app_config):
        feed, _ = self._make_feed(app_config)

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session()):
            with patch(
                "websockets.connect",
                new_callable=AsyncMock,
                side_effect=OSError("refused"),
            ):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    with pytest.raises(ConnectionError, match="5 intentos"):
                        await feed.connect()

    def test_auth_headers_use_bearer_fallback_without_private_key(self, app_config):
        feed, _ = self._make_feed(app_config)
        feed._private_key = None

        headers = feed._auth_headers(method="GET", path="/markets")

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_auth_headers_use_signed_scheme_with_private_key(self, app_config):
        feed, _ = self._make_feed(app_config)
        mock_key = MagicMock()
        mock_key.sign.return_value = b"signed-bytes"
        feed._private_key = mock_key

        headers = feed._auth_headers(method="GET", path="/markets")

        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert headers["KALSHI-ACCESS-KEY"] == app_config.kalshi.api_key_id

    # ── disconnect ────────────────────────────────────────────────────────────

    async def test_disconnect_closes_ws_and_session(self, app_config):
        feed, _ = self._make_feed(app_config)
        mock_ws = _make_ws_mock()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        feed._ws = mock_ws
        feed._session = mock_session
        feed._connected = True

        await feed.disconnect()

        mock_ws.close.assert_called_once()
        mock_session.close.assert_called_once()
        assert feed._connected is False

    async def test_disconnect_when_not_connected_is_safe(self, app_config):
        feed, _ = self._make_feed(app_config)
        await feed.disconnect()

    # ── get_active_markets ────────────────────────────────────────────────────

    async def test_get_active_markets_returns_filtered_list(self, app_config):
        feed, _ = self._make_feed(app_config)
        # time_to_expiry_s será 0 porque la fecha ya expiró, pero queremos testear
        # el filtro de ticker y volumen; usaremos un close_time futuro cercano (<= 900s)
        from datetime import datetime, timezone, timedelta
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        raw_data = {
            "markets": [
                self._market_raw(ticker="KXBTC-15MIN-B95000", volume=500, close_time=close_soon),
                self._market_raw(ticker="KXETH-15MIN-B3000", volume=200, close_time=close_soon),
                self._market_raw(ticker="KXFOREX-EUR-USD", volume=300, close_time=close_soon),
            ]
        }
        mock_session = _make_aiohttp_session()
        mock_session.get = MagicMock(return_value=_make_aiohttp_response(raw_data))
        feed._session = mock_session

        markets = await feed.get_active_markets()

        tickers = [m.ticker for m in markets]
        assert "KXBTC-15MIN-B95000" in tickers
        assert "KXETH-15MIN-B3000" in tickers
        assert "KXFOREX-EUR-USD" not in tickers

    async def test_get_active_markets_returns_empty_on_error(self, app_config):
        feed, _ = self._make_feed(app_config)
        mock_session = _make_aiohttp_session()
        mock_session.get = MagicMock(side_effect=OSError("network"))
        feed._session = mock_session

        markets = await feed.get_active_markets()

        assert markets == []

    # ── filtros ───────────────────────────────────────────────────────────────

    def test_filter_passes_valid_btc_market(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from core.models import MarketSnapshot

        feed = KalshiFeed(config=app_config, bus=EventBus())
        snap = MarketSnapshot(
            ticker="KXBTC-15MIN-B95000",
            implied_prob=0.56,
            yes_ask=0.56,
            no_ask=0.44,
            volume_24h=500,
            time_to_expiry_s=600,
            timestamp=time.time(),
            category="BTC",
        )
        assert feed._passes_filter(snap) is True

    def test_filter_passes_valid_eth_market(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from core.models import MarketSnapshot

        feed = KalshiFeed(config=app_config, bus=EventBus())
        snap = MarketSnapshot(
            ticker="KXETH-15MIN-B3000",
            implied_prob=0.50,
            yes_ask=0.50,
            no_ask=0.50,
            volume_24h=200,
            time_to_expiry_s=300,
            timestamp=time.time(),
            category="ETH",
        )
        assert feed._passes_filter(snap) is True

    def test_filter_rejects_expired_market(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from core.models import MarketSnapshot

        feed = KalshiFeed(config=app_config, bus=EventBus())
        snap = MarketSnapshot(
            ticker="KXBTC-15MIN-B95000",
            implied_prob=0.56,
            yes_ask=0.56,
            no_ask=0.44,
            volume_24h=500,
            time_to_expiry_s=901,   # > 900 → rechazado
            timestamp=time.time(),
            category="BTC",
        )
        assert feed._passes_filter(snap) is False

    def test_filter_rejects_low_volume_market(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from core.models import MarketSnapshot

        # min_volume_24h default = 100
        feed = KalshiFeed(config=app_config, bus=EventBus())
        snap = MarketSnapshot(
            ticker="KXBTC-15MIN-B95000",
            implied_prob=0.56,
            yes_ask=0.56,
            no_ask=0.44,
            volume_24h=50,   # < 100 → rechazado
            time_to_expiry_s=600,
            timestamp=time.time(),
            category="BTC",
        )
        assert feed._passes_filter(snap) is False

    def test_filter_rejects_non_crypto_ticker(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from core.models import MarketSnapshot

        feed = KalshiFeed(config=app_config, bus=EventBus())
        snap = MarketSnapshot(
            ticker="KXFOREX-EUR-USD",
            implied_prob=0.50,
            yes_ask=0.50,
            no_ask=0.50,
            volume_24h=500,
            time_to_expiry_s=600,
            timestamp=time.time(),
            category="UNKNOWN",
        )
        assert feed._passes_filter(snap) is False

    def test_filter_accepts_market_at_exactly_900s(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from core.models import MarketSnapshot

        feed = KalshiFeed(config=app_config, bus=EventBus())
        snap = MarketSnapshot(
            ticker="KXBTC-15MIN-B95000",
            implied_prob=0.56,
            yes_ask=0.56,
            no_ask=0.44,
            volume_24h=500,
            time_to_expiry_s=900,   # exactamente el límite → acepta
            timestamp=time.time(),
            category="BTC",
        )
        assert feed._passes_filter(snap) is True

    # ── stream_markets ────────────────────────────────────────────────────────

    async def test_stream_markets_emits_market_snapshot(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from datetime import datetime, timezone, timedelta

        feed = KalshiFeed(config=app_config, bus=EventBus())
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        ws_msg = json.dumps({
            "type": "ticker",
            "msg": {
                "ticker": "KXBTC-15MIN-B95000",
                "yes_ask": 56,
                "no_ask": 44,
                "volume": 500,
                "close_time": close_soon,
            },
        })
        mock_ws = _make_ws_mock(recv_return=ws_msg)
        feed._ws = mock_ws
        feed._connected = True

        gen = feed.stream_markets()
        snapshot = await anext(gen)
        await gen.aclose()

        assert snapshot.ticker == "KXBTC-15MIN-B95000"
        assert snapshot.category == "BTC"
        assert snapshot.yes_ask == pytest.approx(0.56)

    async def test_stream_markets_skips_non_ticker_messages(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from datetime import datetime, timezone, timedelta

        feed = KalshiFeed(config=app_config, bus=EventBus())
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        ticker_msg = json.dumps({
            "type": "ticker",
            "msg": {
                "ticker": "KXBTC-15MIN-B95000",
                "yes_ask": 56,
                "no_ask": 44,
                "volume": 500,
                "close_time": close_soon,
            },
        })
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"type": "subscribed", "msg": {}}),  # ignorar
            ticker_msg,
        ])
        feed._ws = mock_ws
        feed._connected = True

        gen = feed.stream_markets()
        snapshot = await anext(gen)
        await gen.aclose()

        assert snapshot.ticker == "KXBTC-15MIN-B95000"

    async def test_stream_markets_reconnects_on_connection_closed(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from datetime import datetime, timezone, timedelta

        feed = KalshiFeed(config=app_config, bus=EventBus())
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        ticker_msg = json.dumps({
            "type": "ticker",
            "msg": {
                "ticker": "KXBTC-15MIN-B95000",
                "yes_ask": 56,
                "no_ask": 44,
                "volume": 500,
                "close_time": close_soon,
            },
        })

        ws_after_reconnect = _make_ws_mock(recv_return=ticker_msg)
        feed._ws = _make_ws_mock(
            recv_side_effect=websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)
        )
        feed._connected = True

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session()):
            with patch(
                "websockets.connect",
                new_callable=AsyncMock,
                return_value=ws_after_reconnect,
            ):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    gen = feed.stream_markets()
                    snapshot = await anext(gen)
                    await gen.aclose()

        assert snapshot.ticker == "KXBTC-15MIN-B95000"

    async def test_stream_markets_publishes_to_event_bus(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from datetime import datetime, timezone, timedelta

        bus = EventBus()
        feed = KalshiFeed(config=app_config, bus=bus)
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        ws_msg = json.dumps({
            "type": "ticker",
            "msg": {
                "ticker": "KXBTC-15MIN-B95000",
                "yes_ask": 56,
                "no_ask": 44,
                "volume": 500,
                "close_time": close_soon,
            },
        })
        mock_ws = _make_ws_mock(recv_return=ws_msg)
        feed._ws = mock_ws
        feed._connected = True

        gen = feed.stream_markets()
        await anext(gen)
        await gen.aclose()

        assert bus.qsize == 1

    # ── parseo de mercados ────────────────────────────────────────────────────

    def test_parse_market_extracts_category_btc(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from datetime import datetime, timezone, timedelta

        feed = KalshiFeed(config=app_config, bus=EventBus())
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        raw = {
            "ticker": "KXBTC-15MIN-B95000",
            "yes_ask": 56, "no_ask": 44, "volume": 100,
            "close_time": close_soon,
        }
        snap = feed._parse_market(raw)
        assert snap is not None
        assert snap.category == "BTC"

    def test_parse_market_converts_prices_from_cents(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from datetime import datetime, timezone, timedelta

        feed = KalshiFeed(config=app_config, bus=EventBus())
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        raw = {
            "ticker": "KXBTC-15MIN-B95000",
            "yes_ask": 60, "no_ask": 40, "volume": 100,
            "close_time": close_soon,
        }
        snap = feed._parse_market(raw)
        assert snap is not None
        assert snap.yes_ask == pytest.approx(0.60)
        assert snap.no_ask == pytest.approx(0.40)

    def test_parse_market_uses_floor_strike_for_15m_crypto(self, app_config):
        from feeds.kalshi_feed import KalshiFeed
        from datetime import datetime, timezone, timedelta

        feed = KalshiFeed(config=app_config, bus=EventBus())
        close_soon = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        raw = {
            "ticker": "KXBTC15M-26APR092215-15",
            "yes_ask_dollars": "0.8600",
            "no_ask_dollars": "0.1500",
            "volume_24h_fp": "131276.51",
            "close_time": close_soon,
            "floor_strike": "72179.72",
        }

        snap = feed._parse_market(raw)

        assert snap is not None
        assert snap.category == "BTC"
        assert snap.strike == pytest.approx(72179.72)
        assert snap.volume_24h == 131276

    def test_parse_market_returns_none_on_missing_ticker(self, app_config):
        from feeds.kalshi_feed import KalshiFeed

        feed = KalshiFeed(config=app_config, bus=EventBus())
        snap = feed._parse_market({"yes_ask": 50, "no_ask": 50})
        assert snap is None


# ══════════════════════════════════════════════════════════════════════════════
# _calc_time_to_expiry (helper de módulo)
# ══════════════════════════════════════════════════════════════════════════════

class TestCalcTimeToExpiry:

    def test_future_timestamp_returns_positive(self):
        from feeds.kalshi_feed import _calc_time_to_expiry
        from datetime import datetime, timezone, timedelta

        future = (datetime.now(tz=timezone.utc) + timedelta(seconds=600)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        result = _calc_time_to_expiry(future)
        assert 590 <= result <= 610  # margen de 10s para ejecución del test

    def test_past_timestamp_returns_zero(self):
        from feeds.kalshi_feed import _calc_time_to_expiry

        result = _calc_time_to_expiry("2020-01-01T00:00:00Z")
        assert result == 0

    def test_empty_string_returns_zero(self):
        from feeds.kalshi_feed import _calc_time_to_expiry

        assert _calc_time_to_expiry("") == 0

    def test_invalid_format_returns_zero(self):
        from feeds.kalshi_feed import _calc_time_to_expiry

        assert _calc_time_to_expiry("not-a-date") == 0
