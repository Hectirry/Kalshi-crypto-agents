"""
engine/price_resolver.py

Resuelve un precio spot de referencia a partir de múltiples feeds.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.models import PriceSnapshot


@dataclass(frozen=True, slots=True)
class PriceResolution:
    """Resultado de combinar múltiples fuentes de precio."""

    snapshot: PriceSnapshot | None
    blocked_reason: str | None
    spread_pct: float | None
    used_sources: tuple[str, ...]


def resolve_reference_price(
    *,
    symbol: str,
    latest_prices: dict[str, dict[str, PriceSnapshot]],
    now_ts: float,
    max_age_s: float = 10.0,
    max_cross_source_spread_pct: float = 0.003,
) -> PriceResolution:
    """Devuelve un precio de referencia usando consenso o fallback por frescura."""

    by_source = latest_prices.get(symbol, {})
    fresh = {
        source: snapshot
        for source, snapshot in by_source.items()
        if now_ts - snapshot.timestamp <= max_age_s
    }
    if not fresh:
        return PriceResolution(
            snapshot=None,
            blocked_reason="missing_or_stale_price",
            spread_pct=None,
            used_sources=tuple(),
        )

    binance = fresh.get("binance")
    hyperliquid = fresh.get("hyperliquid")
    if binance and hyperliquid:
        spread_pct = abs(binance.price - hyperliquid.price) / min(binance.price, hyperliquid.price)
        if spread_pct > max_cross_source_spread_pct:
            return PriceResolution(
                snapshot=None,
                blocked_reason="cross_exchange_divergence",
                spread_pct=spread_pct,
                used_sources=("binance", "hyperliquid"),
            )
        consensus_price = (binance.price + hyperliquid.price) / 2.0
        return PriceResolution(
            snapshot=PriceSnapshot(
                symbol=symbol,
                price=consensus_price,
                timestamp=max(binance.timestamp, hyperliquid.timestamp),
                source="consensus",
                bid=binance.bid,
                ask=binance.ask,
                volume_1m=hyperliquid.volume_1m,
            ),
            blocked_reason=None,
            spread_pct=spread_pct,
            used_sources=("binance", "hyperliquid"),
        )

    preferred = binance or hyperliquid
    return PriceResolution(
        snapshot=preferred,
        blocked_reason=None,
        spread_pct=None,
        used_sources=(preferred.source,),
    )
