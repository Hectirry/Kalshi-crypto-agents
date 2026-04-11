"""
core/models.py

Estructuras de datos canónicas. Estas son las únicas clases que viajan
entre módulos. Si necesitas agregar un campo, agrégalo aquí con default=None
para mantener compatibilidad con código existente.

Regla: sin lógica de negocio aquí → solo estructura y validación básica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ─── Enums ────────────────────────────────────────────────────────────────────

class Decision(str, Enum):
    YES  = "YES"
    NO   = "NO"
    WAIT = "WAIT"
    SKIP = "SKIP"
    ERROR = "ERROR"


class Confidence(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class TradeStatus(str, Enum):
    OPEN      = "open"
    CLOSED    = "closed"
    CANCELLED = "cancelled"


class TradeMode(str, Enum):
    DEMO       = "demo"
    PRODUCTION = "production"


class Outcome(str, Enum):
    WIN  = "WIN"
    LOSS = "LOSS"


# ─── Snapshots (datos crudos de feeds) ───────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    """
    Precio spot puntual de un exchange externo (Binance, Hyperliquid).

    Attributes:
        symbol:    símbolo del activo, ej: "BTC", "ETH"
        price:     precio en USD
        timestamp: unix timestamp con precisión de ms
        source:    identificador del feed, ej: "binance", "hyperliquid"
        bid:       mejor oferta de compra (opcional)
        ask:       mejor oferta de venta (opcional)
        volume_1m: volumen en el último minuto en USD (opcional, para volatilidad)
    """
    symbol:    str
    price:     float
    timestamp: float
    source:    str
    bid:       float | None = None
    ask:       float | None = None
    volume_1m: float | None = None

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"price debe ser positivo, recibido: {self.price}")
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if self.bid is not None and self.bid <= 0:
            raise ValueError(f"bid debe ser positivo: {self.bid}")
        if self.ask is not None and self.ask <= 0:
            raise ValueError(f"ask debe ser positivo: {self.ask}")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError(f"bid ({self.bid}) no puede ser mayor que ask ({self.ask})")


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """
    Estado de un contrato de mercado en Kalshi en un momento dado.

    Attributes:
        ticker:          identificador Kalshi, ej: "KXBTC-15MIN-B95000"
        implied_prob:    probabilidad implícita del contrato YES (0.0 → 1.0)
        yes_ask:         precio ask del lado YES (0.0 → 1.0)
        no_ask:          precio ask del lado NO (0.0 → 1.0)
        volume_24h:      contratos operados en las últimas 24h
        time_to_expiry_s: segundos hasta expiración del contrato
        timestamp:       unix timestamp del snapshot
        category:        categoría del activo, ej: "BTC", "ETH"
        strike:          nivel de precio del strike, ej: 95000.0
        event_ticker:    ticker del evento Kalshi contenedor
        title:           título humano del mercado según Kalshi
    """
    ticker:           str
    implied_prob:     float
    yes_ask:          float
    no_ask:           float
    volume_24h:       int
    time_to_expiry_s: int
    timestamp:        float
    category:         str = "UNKNOWN"
    strike:           float | None = None
    event_ticker:     str | None = None
    title:            str | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.implied_prob <= 1.0):
            raise ValueError(f"implied_prob fuera de rango [0,1]: {self.implied_prob}")
        if not (0.0 <= self.yes_ask <= 1.0):
            raise ValueError(f"yes_ask fuera de rango [0,1]: {self.yes_ask}")
        if not (0.0 <= self.no_ask <= 1.0):
            raise ValueError(f"no_ask fuera de rango [0,1]: {self.no_ask}")
        if self.time_to_expiry_s < 0:
            raise ValueError(f"time_to_expiry_s no puede ser negativo: {self.time_to_expiry_s}")
        if self.volume_24h < 0:
            raise ValueError(f"volume_24h no puede ser negativo: {self.volume_24h}")


# ─── Signal (decisión del engine) ─────────────────────────────────────────────

@dataclass(slots=True)
class Signal:
    """
    Decisión del motor de señales para un mercado específico.

    Este objeto es el único que viaja del engine al executor.
    Se guarda completo en SQLite para backtesting posterior.

    Attributes:
        market_ticker:   ticker del contrato Kalshi
        decision:        acción recomendada (YES/NO/WAIT/SKIP/ERROR)
        my_probability:  probabilidad estimada por el modelo propio
        market_probability: probabilidad implícita del mercado
        delta:           my_probability - market_probability (el edge)
        ev_net_fees:     expected value neto después de fees Kalshi
        kelly_size:      fracción de bankroll recomendada (Kelly fraccionario)
        confidence:      nivel de confianza en la señal
        time_remaining_s: segundos hasta expiración al momento de la señal
        reasoning:       texto explicativo para logging/debug
        timestamp:       unix timestamp de la señal
        error_msg:       mensaje de error si decision == ERROR
    """
    market_ticker:      str
    decision:           Decision
    my_probability:     float
    market_probability: float
    delta:              float
    ev_net_fees:        float
    kelly_size:         float
    confidence:         Confidence
    time_remaining_s:   int
    reasoning:          str
    timestamp:          float
    contract_price:     float | None = None
    market_overround_bps: float | None = None
    error_msg:          str | None = None
    # Se rellena después al conocer el outcome real
    outcome:            Outcome | None = None
    outcome_at:         float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.my_probability <= 1.0):
            raise ValueError(f"my_probability fuera de [0,1]: {self.my_probability}")
        if not (0.0 <= self.market_probability <= 1.0):
            raise ValueError(f"market_probability fuera de [0,1]: {self.market_probability}")
        if not (0.0 <= self.kelly_size <= 1.0):
            raise ValueError(f"kelly_size fuera de [0,1]: {self.kelly_size}")
        if self.contract_price is not None and not (0.0 <= self.contract_price <= 1.0):
            raise ValueError(f"contract_price fuera de [0,1]: {self.contract_price}")
        if self.market_overround_bps is not None and self.market_overround_bps < 0.0:
            raise ValueError(
                f"market_overround_bps no puede ser negativo: {self.market_overround_bps}"
            )

    @property
    def is_actionable(self) -> bool:
        """True si la decisión implica ejecutar una orden (YES o NO)."""
        return self.decision in (Decision.YES, Decision.NO)

    @classmethod
    def make_error(
        cls,
        market_ticker: str,
        error_msg: str,
        timestamp: float,
    ) -> Signal:
        """Factory para señales de error → evita repetir valores neutros."""
        return cls(
            market_ticker=market_ticker,
            decision=Decision.ERROR,
            my_probability=0.0,
            market_probability=0.0,
            delta=0.0,
            ev_net_fees=0.0,
            kelly_size=0.0,
            confidence=Confidence.LOW,
            time_remaining_s=0,
            reasoning="",
            timestamp=timestamp,
            contract_price=None,
            market_overround_bps=None,
            error_msg=error_msg,
        )

    @classmethod
    def make_skip(
        cls,
        market_ticker: str,
        reason: str,
        market_probability: float,
        timestamp: float,
    ) -> Signal:
        """Factory para SKIPs → registra el motivo para análisis posterior."""
        return cls(
            market_ticker=market_ticker,
            decision=Decision.SKIP,
            my_probability=0.0,
            market_probability=market_probability,
            delta=0.0,
            ev_net_fees=0.0,
            kelly_size=0.0,
            confidence=Confidence.LOW,
            time_remaining_s=0,
            reasoning=reason,
            timestamp=timestamp,
            contract_price=None,
            market_overround_bps=None,
        )


# ─── Trade (orden ejecutada) ──────────────────────────────────────────────────

@dataclass(slots=True)
class Trade:
    """
    Registro de una orden ejecutada (o simulada en demo).

    Attributes:
        ticker:      ticker del contrato
        side:        lado de la orden (YES o NO)
        contracts:   cantidad de contratos
        entry_price: precio de entrada (0.0 → 1.0)
        mode:        demo o production
        status:      estado actual de la orden
        opened_at:   unix timestamp de apertura
        signal_id:   FK a la tabla signals en SQLite (opcional hasta persistir)
        id:          PK de SQLite (None hasta persistir)
        exit_price:  precio de salida (None si aún abierto)
        fee_paid:    fees pagados a Kalshi
        pnl:         profit/loss neto en USD (None si aún abierto)
        closed_at:   unix timestamp de cierre
    """
    ticker:      str
    side:        Literal["YES", "NO"]
    contracts:   int
    entry_price: float
    mode:        TradeMode
    status:      TradeStatus
    opened_at:   float
    signal_id:   int | None = None
    id:          int | None = None
    exit_price:  float | None = None
    fee_paid:    float = 0.0
    pnl:         float | None = None
    closed_at:   float | None = None

    def __post_init__(self) -> None:
        if self.contracts <= 0:
            raise ValueError(f"contracts debe ser positivo: {self.contracts}")
        if not (0.0 <= self.entry_price <= 1.0):
            raise ValueError(f"entry_price fuera de [0,1]: {self.entry_price}")
        if self.fee_paid < 0:
            raise ValueError(f"fee_paid no puede ser negativo: {self.fee_paid}")

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def gross_value(self) -> float:
        """Valor bruto de la posición al precio de entrada."""
        return self.contracts * self.entry_price
