"""
feeds/kalshi_feed.py

Feed de mercados Kalshi: snapshot REST + streaming WS.
Implementa el protocolo MarketScanner de core/interfaces.py.

Filtra SOLO mercados que cumplan los tres criterios:
  1. ticker contiene "BTC", "ETH" o "SOL"
  2. time_to_expiry_s <= 86400  (24 horas — mercados horarios/diarios)
  3. volume_24h >= config.engine.min_volume_24h
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import aiohttp
import websockets
import websockets.exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from core.config import AppConfig
from core.interfaces import EventBus
from core.models import MarketSnapshot

logger = logging.getLogger(__name__)

MAX_RECONNECTS = 5
_CRYPTO_TICKERS = ("BTC", "ETH", "SOL")


class KalshiFeed:
    """
    Scanner de mercados Kalshi con filtro crypto + expiración + volumen.

    Usa REST para snapshot inicial (get_active_markets) y WS para streaming
    continuo (stream_markets). Auth via API key como Bearer token.
    """

    def __init__(self, config: AppConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._ws: object | None = None
        self._session: aiohttp.ClientSession | None = None
        self._connected: bool = False
        self._private_key = self._load_private_key(config.kalshi.private_key_path)
        # Cache ticker → strike: el REST provee floor_strike, el WS no lo manda.
        # Cuando el WS recibe un update y no tiene strike, usamos el cacheado del REST.
        self._strike_cache: dict[str, float] = {}

    # ── Protocolo MarketScanner ───────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Abre sesión HTTP y conexión WS con backoff exponencial, máx 5 intentos.

        Raises:
            ConnectionError: si agota todos los intentos.
        """
        self._session = aiohttp.ClientSession()
        last_exc: BaseException | None = None
        for attempt in range(1, MAX_RECONNECTS + 1):
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self._config.kalshi.ws_url,
                        additional_headers=self._ws_headers(),
                    ),
                    timeout=10.0,
                )
                self._connected = True
                await self._subscribe_ws()
                logger.info("Kalshi WS conectado (intento %d/%d)", attempt, MAX_RECONNECTS)
                return
            except Exception as exc:
                last_exc = exc
                self._connected = False
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "Kalshi reconexión %d/%d, backoff %ds: %s",
                    attempt, MAX_RECONNECTS, delay, exc,
                )
                if attempt < MAX_RECONNECTS:
                    await asyncio.sleep(delay)

        await self._close_session()
        raise ConnectionError(
            f"Kalshi: no se pudo conectar tras {MAX_RECONNECTS} intentos. "
            f"Último error: {last_exc}"
        ) from last_exc

    async def disconnect(self) -> None:
        """Cierra WS y sesión HTTP limpiamente. No propaga excepciones."""
        self._connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        await self._close_session()
        logger.info("Kalshi feed desconectado")

    async def get_active_markets(self) -> list[MarketSnapshot]:
        """
        Snapshot REST de mercados activos, ya filtrados por los tres criterios.

        Hace una request por cada serie crypto conocida para no depender de la
        paginación genérica que puede no retornar BTC/ETH en los primeros 200.

        Returns:
            Lista vacía si no hay mercados disponibles o si la request falla.
        """
        # Series conocidas de Kalshi para cripto
        # KXBTC15M / KXETH15M / KXSOL15M = contratos de 15 minutos (api.elections.kalshi.com)
        # KXBTC / KXETH / KXSOL y variantes D = contratos horarios/diarios
        series = (
            "KXBTC15M", "KXETH15M", "KXSOL15M",
            "KXBTC", "KXBTCD", "KXETH", "KXETHD", "KXSOL", "KXSOLD",
        )
        url = f"{self._config.kalshi.base_url}/markets"
        api_path = urlparse(url).path

        markets: list[MarketSnapshot] = []
        for series_ticker in series:
            params = {"status": "open", "limit": 200, "series_ticker": series_ticker}
            try:
                async def _do_request(p=params) -> object:
                    headers = self._auth_headers(method="GET", path=api_path)
                    async with self._session.get(url, params=p, headers=headers) as resp:
                        resp.raise_for_status()
                        return await resp.json()

                data = await asyncio.wait_for(_do_request(), timeout=10.0)
            except Exception as exc:
                logger.warning("Kalshi get_active_markets falló serie=%s: %s", series_ticker, exc)
                continue

            for raw in data.get("markets", []):
                snapshot = self._parse_market(raw)
                if snapshot is not None:
                    # Cachear strike para que el WS pueda usarlo
                    if snapshot.strike is not None:
                        self._strike_cache[snapshot.ticker] = snapshot.strike
                    if self._passes_filter(snapshot):
                        markets.append(snapshot)

        return markets

    async def stream_markets(self) -> AsyncIterator[MarketSnapshot]:
        """
        Emite MarketSnapshot desde el WS de Kalshi continuamente.

        Reconecta automáticamente si el WS cae. Propaga ConnectionError si
        los reconectos se agotan.
        """
        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError) as exc:
                logger.warning("Kalshi WS caído: %s. Reconectando...", exc)
                self._connected = False
                await self.connect()
                continue

            data = json.loads(raw)
            if data.get("type") not in ("ticker", "orderbook_delta"):
                continue

            msg = data.get("msg", {})
            snapshot = self._parse_market(msg)
            if snapshot is None or not self._passes_filter(snapshot):
                continue

            await self._bus.publish(snapshot)
            yield snapshot

    # ── Filtro ────────────────────────────────────────────────────────────────

    def _passes_filter(self, snapshot: MarketSnapshot) -> bool:
        """True si el mercado cumple los tres criterios de admisión."""
        has_crypto = any(sym in snapshot.ticker for sym in _CRYPTO_TICKERS)
        not_expired = snapshot.time_to_expiry_s <= 900  # contratos de 15 min: max 900s restantes
        enough_volume = snapshot.volume_24h >= self._config.engine.min_volume_24h
        return has_crypto and not_expired and enough_volume

    # ── Parseo ────────────────────────────────────────────────────────────────

    def _parse_market(self, raw: dict) -> MarketSnapshot | None:
        """
        Convierte un dict de la API Kalshi a MarketSnapshot.

        Soporta tanto el formato REST (campo 'ticker', 'close_time') como el
        formato WS (campo 'market_ticker', sin 'close_time' → se deriva del ticker).
        Precios en dólares (0.00–1.00) o centavos (0–99) → normaliza a [0.01, 0.99].
        """
        try:
            # REST usa 'ticker'; WS usa 'market_ticker'
            ticker: str = raw.get("ticker") or raw.get("market_ticker") or ""
            if not ticker:
                return None

            yes_ask = _coerce_price(
                raw.get("yes_ask"),
                raw.get("yes_ask_dollars"),
                default=0.50,
            )
            no_ask = _coerce_price(
                raw.get("no_ask"),
                raw.get("no_ask_dollars"),
                default=max(0.01, min(0.99, 1.0 - yes_ask)),
            )
            implied_prob = max(0.01, min(0.99, yes_ask))

            # WS usa 'volume_fp' (total), REST usa 'volume_24h_fp'
            volume_24h = _coerce_volume(
                raw.get("volume_24h"),
                raw.get("volume_24h_fp"),
                raw.get("volume"),
                raw.get("volume_fp"),
                default=0,
            )

            # REST incluye 'close_time'; WS no → parsear del ticker
            close_time_str: str = raw.get("close_time", "")
            if close_time_str:
                time_to_expiry_s = _calc_time_to_expiry(close_time_str)
            else:
                time_to_expiry_s = _calc_time_to_expiry_from_ticker(ticker)

            category = "UNKNOWN"
            for sym in _CRYPTO_TICKERS:
                if sym in ticker:
                    category = sym
                    break

            # Extraer strike:
            # - mercados 15M: floor_strike/cap_strike/custom_strike
            # - legacy: KXBTC-15MIN-B95000 -> 95000.0
            # - nuevos rangos: KXBTC-...-T80799.99 -> 80799.99
            strike = _coerce_strike(
                raw.get("floor_strike"),
                raw.get("cap_strike"),
                raw.get("custom_strike"),
            )
            if strike is None:
                for part in ticker.split("-"):
                    if part.startswith(("B", "T")):
                        try:
                            strike = float(part[1:].replace(",", ""))
                        except ValueError:
                            strike = None
                        break

            # Fallback: usar strike cacheado del REST si el WS no lo provee
            if strike is None:
                strike = self._strike_cache.get(ticker)

            return MarketSnapshot(
                ticker=ticker,
                implied_prob=implied_prob,
                yes_ask=max(0.01, min(0.99, yes_ask)),
                no_ask=max(0.01, min(0.99, no_ask)),
                volume_24h=volume_24h,
                time_to_expiry_s=time_to_expiry_s,
                timestamp=time.time(),
                category=category,
                strike=strike,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Error parseando mercado Kalshi: %s → %s", raw, exc)
            return None

    # ── Privados ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_private_key(private_key_path: str | None):
        """Carga llave privada PEM si fue configurada."""

        if private_key_path is None:
            return None
        try:
            with open(private_key_path, "rb") as handle:
                return serialization.load_pem_private_key(handle.read(), password=None)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            logger.warning("Kalshi private key no se pudo cargar: %s", exc)
            return None

    def _signed_headers(self, method: str, path: str) -> dict[str, str]:
        """Construye headers RSA-PSS de Kalshi para REST/WS."""

        if self._private_key is None:
            return {}

        timestamp = str(int(time.time() * 1000))
        unsigned_path = path.split("?")[0]
        message = f"{timestamp}{method.upper()}{unsigned_path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        encoded = base64.b64encode(signature).decode("utf-8")
        return {
            "KALSHI-ACCESS-KEY": self._config.kalshi.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": encoded,
        }

    def _bearer_headers(self) -> dict[str, str]:
        """Fallback legacy para entornos aún no migrados."""

        token = self._config.kalshi.api_key or self._config.kalshi.api_key_id
        return {"Authorization": f"Bearer {token}"}

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Selecciona RSA-PSS si hay PEM, sino Bearer legacy."""

        signed = self._signed_headers(method=method, path=path)
        if signed:
            return signed
        return self._bearer_headers()

    def _ws_headers(self) -> dict[str, str]:
        """Headers para websocket Kalshi."""

        return self._auth_headers(method="GET", path="/trade-api/ws/v2")

    async def _subscribe_ws(self) -> None:
        """Suscribe a actualizaciones de ticker en el WS."""
        msg = json.dumps({
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["ticker"]},
        })
        await asyncio.wait_for(self._ws.send(msg), timeout=10.0)

    async def _close_session(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


# ── Helpers de módulo ─────────────────────────────────────────────────────────

def _calc_time_to_expiry(close_time_str: str) -> int:
    """Calcula segundos hasta expiración desde un ISO timestamp. 0 si falla."""
    if not close_time_str:
        return 0
    try:
        close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        now = datetime.now(tz=timezone.utc)
        return max(0, int((close_dt - now).total_seconds()))
    except (ValueError, TypeError):
        return 0


_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
# Kalshi hourly format:   KXBTC-26APR0917-...   → DDMMMYYHH  (day+month+year2+hour)
# Kalshi 15-min format:   KXBTC15M-26APR092215-15 → DDMMMYYHHMM (day+month+year2+hour+min)
_EVENT_RE_HOURLY = re.compile(r"(\d{2})([A-Z]{3})(\d{2})(\d{2})$")
_EVENT_RE_15M    = re.compile(r"(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})$")
# Kalshi settlements use Eastern time. EDT = UTC-4, EST = UTC-5.
# Approximate: assume EDT (UTC-4) for spring/summer markets.
_EDT_TO_UTC = timedelta(hours=4)


def _calc_time_to_expiry_from_ticker(ticker: str) -> int:
    """
    Deriva time-to-expiry en segundos parseando el event part del ticker.

    Formatos soportados:
      - Horario:   KXBTC-26APR0917-...    → '26APR0917'  = 2026-04-09 17:00 EDT
      - 15 minutos: KXBTC15M-26APR092215-15 → '26APR092215' = 2026-04-09 22:15 EDT

    Retorna 0 si no se puede parsear.
    """
    parts = ticker.split("-")
    if len(parts) < 2:
        return 0

    event_part = parts[1]
    hour = 0
    minute = 0

    m15 = _EVENT_RE_15M.match(event_part)
    if m15:
        try:
            year  = 2000 + int(m15.group(1))
            month = _MONTH_MAP.get(m15.group(2), 0)
            day   = int(m15.group(3))
            hour  = int(m15.group(4))
            minute = int(m15.group(5))
            if not month:
                return 0
            close_dt = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc) + _EDT_TO_UTC
            now = datetime.now(tz=timezone.utc)
            return max(0, int((close_dt - now).total_seconds()))
        except (ValueError, OverflowError):
            return 0

    mh = _EVENT_RE_HOURLY.match(event_part)
    if mh:
        try:
            year  = 2000 + int(mh.group(1))
            month = _MONTH_MAP.get(mh.group(2), 0)
            day   = int(mh.group(3))
            hour  = int(mh.group(4))
            if not month:
                return 0
            close_dt = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc) + _EDT_TO_UTC
            now = datetime.now(tz=timezone.utc)
            return max(0, int((close_dt - now).total_seconds()))
        except (ValueError, OverflowError):
            return 0

    return 0


def _coerce_price(*values: object, default: float = 0.50) -> float:
    """Convierte price candidates del payload a rango [0.01, 0.99]."""

    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
            if parsed > 1.0:
                parsed = parsed / 100.0
            return max(0.01, min(0.99, parsed))
        except (TypeError, ValueError):
            continue
    return max(0.01, min(0.99, default))


def _coerce_volume(*values: object, default: int = 0) -> int:
    """Convierte volumen del payload (legacy o fp) a entero no negativo."""

    for value in values:
        if value is None:
            continue
        try:
            return max(0, int(float(value)))
        except (TypeError, ValueError):
            continue
    return max(0, default)


def _coerce_strike(*values: object) -> float | None:
    """Convierte strikes del payload de Kalshi a float positivo."""

    for value in values:
        if value is None:
            continue
        try:
            parsed = float(str(value).replace(",", ""))
            if parsed > 0.0:
                return parsed
        except (TypeError, ValueError):
            continue
    return None
