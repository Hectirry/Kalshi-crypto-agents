"""
feeds/hyperliquid_feed.py

REST polling feed desde Hyperliquid (mark price + open interest).
Implementa el protocolo PriceFeed de core/interfaces.py.

No tiene WebSocket público: hace POST a /info cada POLL_INTERVAL segundos.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

import aiohttp

from core.config import HyperliquidConfig
from core.interfaces import EventBus
from core.models import PriceSnapshot

logger = logging.getLogger(__name__)

MAX_RECONNECTS = 5
POLL_INTERVAL = 2.0
REQUEST_TIMEOUT = 5.0


class HyperliquidFeed:
    """
    Feed de mark prices de Hyperliquid via REST polling cada 2s.

    Endpoint: POST {base_url}/info  body: {"type": "metaAndAssetCtxs"}
    Respuesta: [meta, [asset_ctx, ...]]  donde meta["universe"] da los nombres.
    """

    def __init__(self, config: HyperliquidConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._session: aiohttp.ClientSession | None = None
        self._running: bool = False

    # ── Protocolo PriceFeed ───────────────────────────────────────────────────

    @property
    def source_name(self) -> str:
        return "hyperliquid"

    @property
    def is_connected(self) -> bool:
        return (
            self._running
            and self._session is not None
            and not self._session.closed
        )

    async def connect(self) -> None:
        """
        Crea la sesión HTTP y valida conectividad con un request inicial.

        Reintenta hasta MAX_RECONNECTS veces con backoff exponencial.

        Raises:
            ConnectionError: si agota todos los intentos.
        """
        self._session = aiohttp.ClientSession()
        last_exc: BaseException | None = None
        for attempt in range(1, MAX_RECONNECTS + 1):
            try:
                await self._fetch_once()
                self._running = True
                logger.info(
                    "Hyperliquid feed conectado (intento %d/%d)", attempt, MAX_RECONNECTS
                )
                return
            except Exception as exc:
                last_exc = exc
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "Hyperliquid retry %d/%d, backoff %ds: %s",
                    attempt, MAX_RECONNECTS, delay, exc,
                )
                if attempt < MAX_RECONNECTS:
                    await asyncio.sleep(delay)

        await self._close_session()
        raise ConnectionError(
            f"Hyperliquid: no se pudo conectar tras {MAX_RECONNECTS} intentos. "
            f"Último error: {last_exc}"
        ) from last_exc

    async def disconnect(self) -> None:
        """Cierra la sesión HTTP limpiamente. No propaga excepciones."""
        self._running = False
        await self._close_session()
        logger.info("Hyperliquid feed desconectado")

    async def stream(self) -> AsyncIterator[PriceSnapshot]:
        """
        Genera PriceSnapshot cada POLL_INTERVAL segundos.

        Reintenta hasta MAX_RECONNECTS fallos consecutivos antes de lanzar
        ConnectionError. Un éxito reinicia el contador de fallos.
        """
        consecutive_failures = 0
        while self._running:
            try:
                snapshots = await self._fetch_once()
                consecutive_failures = 0
                for snapshot in snapshots:
                    await self._bus.publish(snapshot)
                    yield snapshot
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures > MAX_RECONNECTS:
                    raise ConnectionError(
                        f"Hyperliquid stream: {MAX_RECONNECTS} fallos consecutivos. "
                        f"Último: {exc}"
                    ) from exc
                delay = 2 ** (consecutive_failures - 1)
                logger.warning(
                    "Hyperliquid error %d/%d, retry en %ds: %s",
                    consecutive_failures, MAX_RECONNECTS, delay, exc,
                )
                await asyncio.sleep(delay)

    # ── Privados ──────────────────────────────────────────────────────────────

    async def _fetch_once(self) -> list[PriceSnapshot]:
        """
        Hace POST a metaAndAssetCtxs y retorna PriceSnapshot para los
        símbolos configurados.

        El campo volume_1m almacena open_interest (campo extra disponible).
        """
        url = f"{self._config.base_url}/info"
        payload = {"type": "metaAndAssetCtxs"}

        async def _do_request() -> object:
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

        data = await asyncio.wait_for(_do_request(), timeout=REQUEST_TIMEOUT)
        meta, asset_ctxs = data[0], data[1]
        universe: list[dict] = meta["universe"]

        snapshots: list[PriceSnapshot] = []
        for sym in self._config.symbols:
            idx = next(
                (i for i, asset in enumerate(universe) if asset["name"] == sym),
                None,
            )
            if idx is None or idx >= len(asset_ctxs):
                continue
            ctx = asset_ctxs[idx]
            mark_price = float(ctx["markPx"])
            open_interest_raw = float(ctx.get("openInterest", 0.0))
            open_interest = open_interest_raw if open_interest_raw > 0 else None

            snapshots.append(
                PriceSnapshot(
                    symbol=sym,
                    price=mark_price,
                    timestamp=time.time(),
                    source="hyperliquid",
                    volume_1m=open_interest,
                )
            )
        return snapshots

    async def _close_session(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
