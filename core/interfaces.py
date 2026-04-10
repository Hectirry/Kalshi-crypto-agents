"""
core/interfaces.py

Protocolos abstractos del sistema. Todos los módulos concretos implementan
uno de estos contratos. Esto permite testear con mocks sin modificar nada.

Regla: ningún módulo fuera de core/ importa de otro módulo concreto directamente
       → solo de interfaces.py y models.py.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol, runtime_checkable

from core.models import MarketSnapshot, PriceSnapshot, Signal, Trade


@runtime_checkable
class PriceFeed(Protocol):
    """
    Fuente de precio spot en tiempo real (Binance, Hyperliquid, etc.).

    Implementaciones concretas: feeds/binance_feed.py, feeds/hyperliquid_feed.py
    """

    @property
    def source_name(self) -> str:
        """Identificador único de la fuente, ej: 'binance', 'hyperliquid'."""
        ...

    async def connect(self) -> None:
        """
        Establece la conexión WebSocket.

        Raises:
            ConnectionError: si no se puede conectar tras los reintentos configurados.
        """
        ...

    async def disconnect(self) -> None:
        """Cierra la conexión limpiamente. No lanza excepciones."""
        ...

    async def stream(self) -> AsyncIterator[PriceSnapshot]:
        """
        Genera snapshots de precio continuamente.

        Yields:
            PriceSnapshot con precio, timestamp y source.

        Raises:
            ConnectionError: si la conexión se pierde y no se puede recuperar.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """True si la conexión WebSocket está activa."""
        ...


@runtime_checkable
class MarketScanner(Protocol):
    """
    Scanner de mercados activos en Kalshi.

    Filtra por crypto, expiración <= 15 min, liquidez mínima.
    Implementación concreta: feeds/kalshi_feed.py
    """

    async def connect(self) -> None:
        """
        Raises:
            ConnectionError: si falla la autenticación o la conexión WS.
        """
        ...

    async def disconnect(self) -> None: ...

    async def stream_markets(self) -> AsyncIterator[MarketSnapshot]:
        """
        Genera snapshots de mercados crypto activos de 15 minutos.

        Solo emite mercados que pasan el filtro de liquidez mínima.

        Yields:
            MarketSnapshot con implied_prob, time_to_expiry_s, volumen.
        """
        ...

    async def get_active_markets(self) -> list[MarketSnapshot]:
        """
        Snapshot puntual de todos los mercados activos (para backtesting y arranque).

        Returns:
            Lista vacía si no hay mercados disponibles → nunca None.
        """
        ...


@runtime_checkable
class SignalEngine(Protocol):
    """
    Motor que convierte un MarketSnapshot + PriceSnapshot en una Signal.

    Implementación concreta: engine/signal_router.py
    """

    def evaluate(
        self,
        market: MarketSnapshot,
        price: PriceSnapshot,
    ) -> Signal:
        """
        Evalúa si un mercado vale la pena tradear.

        Nunca lanza excepciones hacia afuera → retorna Signal con
        decision='ERROR' y error_msg si algo falla internamente.

        Args:
            market: snapshot del contrato Kalshi.
            price: último precio spot de Binance/Hyperliquid.

        Returns:
            Signal con la decisión y todos los parámetros calculados.
        """
        ...


@runtime_checkable
class OrderExecutor(Protocol):
    """
    Ejecuta órdenes en Kalshi.

    En modo demo: registra en SQLite sin llamar a la API real.
    En modo producción: llama a la API y también registra.

    La lógica de decisión (cuándo y qué tradear) vive en SignalEngine,
    no aquí. Este módulo solo sabe cómo ejecutar.

    Implementación concreta: execution/order_executor.py
    """

    @property
    def mode(self) -> str:
        """'demo' o 'production'."""
        ...

    async def submit(self, signal: Signal) -> Trade:
        """
        Ejecuta la orden indicada por signal.decision.

        Args:
            signal: debe tener decision en ('YES', 'NO') → otros valores
                    lanzan ValueError en producción, se ignoran en demo.

        Returns:
            Trade con status='open' si se ejecutó correctamente.

        Raises:
            ValueError: si signal.decision no es YES ni NO.
            RuntimeError: si la API de Kalshi retorna error no recuperable.
        """
        ...

    async def close(self, trade: Trade) -> Trade:
        """
        Cierra una posición abierta.

        Returns:
            Trade actualizado con exit_price, pnl, status='closed'.
        """
        ...


@runtime_checkable
class BacktestSource(Protocol):
    """
    Fuente de datos históricos para backtesting.

    Implementación concreta: backtesting/backtest_runner.py
    """

    async def load_signals(
        self,
        from_ts: float,
        to_ts: float,
        category: str | None = None,
    ) -> list[Signal]:
        """
        Carga señales históricas del rango de tiempo indicado.

        Args:
            from_ts: unix timestamp inicio.
            to_ts: unix timestamp fin.
            category: filtrar por categoría ('BTC', 'ETH') o None para todas.

        Returns:
            Lista de señales ordenadas por timestamp asc. Lista vacía si no hay datos.
        """
        ...


# ─── Bus de eventos interno ───────────────────────────────────────────────────

class EventBus:
    """
    Bus central de eventos async. Todos los feeds publican aquí.
    El SignalEngine y el OrderExecutor consumen de aquí.

    Uso:
        bus = EventBus()
        await bus.publish(snapshot)
        async for event in bus.subscribe():
            ...
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: asyncio.Queue[PriceSnapshot | MarketSnapshot] = asyncio.Queue(
            maxsize=maxsize
        )

    async def publish(self, event: PriceSnapshot | MarketSnapshot) -> None:
        """
        Publica un evento. Si la cola está llena, descarta el más antiguo
        (comportamiento de sliding window → no bloquea a los feeds).
        """
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(event)

    async def subscribe(self) -> AsyncIterator[PriceSnapshot | MarketSnapshot]:
        """Consume eventos del bus indefinidamente."""
        while True:
            event = await self._queue.get()
            yield event
            self._queue.task_done()

    @property
    def qsize(self) -> int:
        """Tamaño actual de la cola → útil para métricas."""
        return self._queue.qsize()
