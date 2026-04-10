"""
feeds/binance_feed.py

WebSocket feed de precios spot desde Binance via bookTicker.
Implementa el protocolo PriceFeed de core/interfaces.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import websockets
import websockets.exceptions

from core.config import BinanceConfig
from core.interfaces import EventBus
from core.models import PriceSnapshot

logger = logging.getLogger(__name__)

_SYMBOL_STRIP: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
}
MAX_RECONNECTS = 5


class BinancePriceFeed:
    """
    Feed de precios spot Binance usando {symbol}@bookTicker por WebSocket.

    Publica PriceSnapshot en el EventBus recibido en el constructor.
    Reconecta automáticamente con backoff exponencial (1s, 2s, 4s, 8s, 16s).
    """

    def __init__(self, config: BinanceConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._ws: object | None = None
        self._connected: bool = False

    # ── Protocolo PriceFeed ───────────────────────────────────────────────────

    @property
    def source_name(self) -> str:
        return "binance"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        """
        Abre la conexión WS y suscribe a bookTicker.

        Reintenta hasta MAX_RECONNECTS veces con backoff exponencial.

        Raises:
            ConnectionError: si agota todos los intentos.
        """
        last_exc: BaseException | None = None
        for attempt in range(1, MAX_RECONNECTS + 1):
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(self._config.ws_url),
                    timeout=10.0,
                )
                self._connected = True
                await self._subscribe()
                logger.info("Binance WS conectado (intento %d/%d)", attempt, MAX_RECONNECTS)
                return
            except Exception as exc:
                last_exc = exc
                self._connected = False
                delay = 2 ** (attempt - 1)  # 1, 2, 4, 8, 16
                logger.warning(
                    "Binance reconexión %d/%d, backoff %ds: %s",
                    attempt, MAX_RECONNECTS, delay, exc,
                )
                if attempt < MAX_RECONNECTS:
                    await asyncio.sleep(delay)

        raise ConnectionError(
            f"Binance: no se pudo conectar tras {MAX_RECONNECTS} intentos. "
            f"Último error: {last_exc}"
        ) from last_exc

    async def disconnect(self) -> None:
        """Cierra el WebSocket limpiamente. No propaga excepciones."""
        self._connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Binance WS desconectado")

    async def stream(self) -> AsyncIterator[PriceSnapshot]:
        """
        Genera PriceSnapshot continuamente desde mensajes bookTicker.

        Reconecta automáticamente si el WS cae. Si los reconectos fallan,
        propaga ConnectionError (desde connect()).
        """
        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError) as exc:
                logger.warning("Binance WS caído: %s. Reconectando...", exc)
                self._connected = False
                await self.connect()  # lanza ConnectionError tras MAX_RECONNECTS
                continue

            data = json.loads(raw)
            # bookTicker: {"s":"BTCUSDT","b":"bid","B":"bidQty","a":"ask","A":"askQty",...}
            # Mensajes de confirmación de suscripción no tienen "s"
            if "s" not in data:
                continue

            raw_sym: str = data["s"]
            symbol = _SYMBOL_STRIP.get(raw_sym, raw_sym.replace("USDT", ""))

            try:
                bid = float(data["b"])
                ask = float(data["a"])
            except (KeyError, ValueError):
                continue

            snapshot = PriceSnapshot(
                symbol=symbol,
                price=(bid + ask) / 2.0,
                timestamp=time.time(),
                source="binance",
                bid=bid,
                ask=ask,
            )
            await self._bus.publish(snapshot)
            yield snapshot

    # ── Privados ──────────────────────────────────────────────────────────────

    async def _subscribe(self) -> None:
        """Envía SUBSCRIBE para todos los símbolos configurados."""
        params = [f"{sym.lower()}@bookTicker" for sym in self._config.symbols]
        msg = json.dumps({"method": "SUBSCRIBE", "params": params, "id": 1})
        await asyncio.wait_for(self._ws.send(msg), timeout=10.0)
