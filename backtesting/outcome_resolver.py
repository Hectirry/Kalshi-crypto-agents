"""
backtesting/outcome_resolver.py

Resolución de outcomes para señales expiradas en Kalshi.

Responsabilidad única: después de que un contrato de 15 minutos expira,
consultar la API de Kalshi y escribir WIN/LOSS en signals y trades.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING

from core.database import Database
from core.models import Outcome
from feeds.kalshi_feed import KalshiRateLimitError

if TYPE_CHECKING:
    from feeds.kalshi_feed import KalshiFeed

logger = logging.getLogger(__name__)

# 16 minutos de margen tras expiración del contrato de 15 min
OUTCOME_LOOKBACK_S: int = 960
MAX_SIGNALS_PER_CYCLE: int = 50


class OutcomeResolver:
    """
    Resuelve outcomes de señales expiradas consultando la API REST de Kalshi.

    Uso:
        resolver = OutcomeResolver(db=db, kalshi_client=kfeed)
        n = await resolver.resolve_expired()
    """

    def __init__(self, db: Database, kalshi_client: KalshiFeed) -> None:
        self._db = db
        self._kalshi = kalshi_client

    async def resolve_expired(self) -> int:
        """
        Resuelve outcomes de señales que ya debieron expirar.

        Flujo por señal:
          1. Buscar en DB señales con outcome=NULL, edad > 16 min, decision YES|NO.
          2. Llamar GET /markets/{ticker} en Kalshi.
          3. Si status != 'finalized': skip → reintentar próximo ciclo.
          4. Escribir WIN o LOSS según market.result vs signal.decision.
          5. Si hay trade abierto vinculado: cerrarlo con pnl real.

        Returns:
            Número de outcomes resueltos en este ciclo (0 si ninguno o error).
        """
        cutoff_ts = time.time() - OUTCOME_LOOKBACK_S
        try:
            pending = self._db.get_pending_outcome_signals(
                cutoff_ts=cutoff_ts,
                limit=MAX_SIGNALS_PER_CYCLE,
            )
        except sqlite3.Error as exc:
            logger.error("outcome_resolver: error leyendo señales pendientes: %s", exc)
            return 0

        if not pending:
            return 0

        resolved = 0
        for signal_id, ticker, decision in pending:
            # Consultar Kalshi — detener ciclo si hay rate limit
            try:
                market_data = await self._kalshi.get_market(ticker)
            except KalshiRateLimitError:
                logger.warning(
                    "outcome_resolver: rate_limit después de %d resueltos, parando ciclo",
                    resolved,
                )
                break

            if market_data is None:
                logger.debug(
                    "outcome_resolver: ticker=%s no encontrado (404), skip", ticker
                )
                continue

            status = market_data.get("status", "")
            if status != "finalized":
                # Mercado no finalizado aún — intentar en próximo ciclo
                continue

            market_result = market_data.get("result", "")
            if market_result not in ("yes", "no"):
                logger.warning(
                    "outcome_resolver: ticker=%s resultado desconocido=%r, skip",
                    ticker,
                    market_result,
                )
                continue

            outcome = _determine_outcome(decision=decision, market_result=market_result)

            try:
                self._db.update_signal_outcome(signal_id, outcome, time.time())
            except (ValueError, sqlite3.Error) as exc:
                logger.error(
                    "outcome_resolver: no se pudo actualizar signal_id=%d: %s",
                    signal_id,
                    exc,
                )
                continue

            # Cerrar trade vinculado si existe
            _close_linked_trade(db=self._db, signal_id=signal_id, outcome=outcome)

            logger.info(
                "outcome_resolver: resolved signal_id=%d ticker=%s decision=%s outcome=%s",
                signal_id,
                ticker,
                decision,
                outcome.value,
            )
            resolved += 1

        return resolved


# ── Helpers de módulo ──────────────────────────────────────────────────────────

def _determine_outcome(decision: str, market_result: str) -> Outcome:
    """
    Determina WIN o LOSS según la decisión tomada y el resultado del mercado.

    Args:
        decision:      "YES" o "NO" (lado comprado).
        market_result: "yes" o "no" (resultado del mercado Kalshi).

    Returns:
        Outcome.WIN o Outcome.LOSS.
    """
    if market_result == "yes":
        return Outcome.WIN if decision == "YES" else Outcome.LOSS
    # market_result == "no"
    return Outcome.WIN if decision == "NO" else Outcome.LOSS


def _close_linked_trade(db: Database, signal_id: int, outcome: Outcome) -> None:
    """Cierra el trade abierto vinculado a signal_id, si existe."""
    try:
        trade = db.get_open_trade_by_signal(signal_id)
        if trade is None or trade.id is None:
            return
        exit_price = 1.0 if outcome == Outcome.WIN else 0.0
        pnl = (exit_price - trade.entry_price) * trade.contracts - trade.fee_paid
        db.close_trade(trade.id, exit_price, pnl, trade.fee_paid)
    except (ValueError, sqlite3.Error) as exc:
        logger.error(
            "outcome_resolver: no se pudo cerrar trade para signal_id=%d: %s",
            signal_id,
            exc,
        )
