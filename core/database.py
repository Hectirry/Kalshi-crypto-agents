"""
core/database.py

Capa de acceso a datos. SQLite como única fuente de verdad.

Reglas:
- Todas las escrituras pasan por aquí → ningún módulo ejecuta SQL directo
- Migraciones SOLO aditivas: agregar tablas/columnas, nunca DROP ni ALTER COLUMN
- WAL mode para soportar lecturas concurrentes mientras se escribe
- Parámetros siempre como placeholders (?), nunca f-strings con datos
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from core.models import (
    Confidence,
    Decision,
    Outcome,
    Signal,
    Trade,
    TradeMode,
    TradeStatus,
)

logger = logging.getLogger(__name__)

# Versión actual del schema → incrementar al agregar migraciones
SCHEMA_VERSION = 1


class Database:
    """
    Gestiona la conexión SQLite y las operaciones de persistencia.

    Uso:
        db = Database(path=Path("trading.db"))
        db.initialize()
        signal_id = db.save_signal(signal)

    No es thread-safe por diseño → usar en el loop asyncio principal.
    Para acceso desde otros hilos, crear una instancia separada.
    """

    def __init__(self, path: Path, wal_mode: bool = True, busy_timeout_ms: int = 5000) -> None:
        self._path = path
        self._wal_mode = wal_mode
        self._busy_timeout_ms = busy_timeout_ms
        self._conn: sqlite3.Connection | None = None

    # ─── Ciclo de vida ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Abre la conexión y aplica el schema + migraciones pendientes.

        Debe llamarse una vez al arrancar la aplicación.

        Raises:
            sqlite3.DatabaseError: si el archivo está corrupto.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            timeout=self._busy_timeout_ms / 1000,
        )
        self._conn.row_factory = sqlite3.Row

        with self._transaction() as cur:
            if self._wal_mode:
                cur.execute("PRAGMA journal_mode=WAL")
            cur.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            cur.execute("PRAGMA foreign_keys=ON")
            self._apply_migrations(cur)

        logger.info("Base de datos inicializada en %s (schema v%d)", self._path, SCHEMA_VERSION)

    def close(self) -> None:
        """Cierra la conexión limpiamente."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Conexión SQLite cerrada")

    # ─── Señales ──────────────────────────────────────────────────────────────

    def save_signal(self, signal: Signal) -> int:
        """
        Persiste una señal nueva.

        Returns:
            ID asignado por SQLite (rowid).
        """
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO signals
                    (ticker, decision, my_prob, market_prob, delta,
                     ev_net_fees, kelly_size, confidence, time_remaining_s,
                     reasoning, created_at, outcome, outcome_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.market_ticker,
                    signal.decision.value,
                    signal.my_probability,
                    signal.market_probability,
                    signal.delta,
                    signal.ev_net_fees,
                    signal.kelly_size,
                    signal.confidence.value,
                    signal.time_remaining_s,
                    signal.reasoning,
                    signal.timestamp,
                    signal.outcome.value if signal.outcome else None,
                    signal.outcome_at,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def update_signal_outcome(self, signal_id: int, outcome: Outcome, outcome_at: float) -> None:
        """Registra el resultado real de una señal al expirar el contrato."""
        with self._transaction() as cur:
            cur.execute(
                "UPDATE signals SET outcome=?, outcome_at=? WHERE id=?",
                (outcome.value, outcome_at, signal_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Signal ID {signal_id} no encontrado")

    def get_signals(
        self,
        from_ts: float,
        to_ts: float,
        category: str | None = None,
        decision: Decision | None = None,
        limit: int = 10_000,
    ) -> list[Signal]:
        """
        Carga señales en un rango de tiempo.

        Args:
            from_ts: unix timestamp inicio (inclusivo)
            to_ts:   unix timestamp fin (inclusivo)
            category: filtrar por categoría del ticker, ej: 'BTC'
            decision: filtrar por decisión tomada
            limit:    máximo de registros a retornar

        Returns:
            Lista ordenada por created_at asc. Vacía si no hay datos.
        """
        self._assert_connected()

        query = """
            SELECT * FROM signals
            WHERE created_at BETWEEN ? AND ?
        """
        params: list = [from_ts, to_ts]

        if category is not None:
            query += " AND ticker LIKE ?"
            params.append(f"%{category}%")

        if decision is not None:
            query += " AND decision = ?"
            params.append(decision.value)

        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()  # type: ignore[union-attr]
        return [_row_to_signal(row) for row in rows]

    def find_signal_id(self, signal: Signal) -> int | None:
        """
        Busca el ID de una señal previamente persistida.

        Usa una clave compuesta práctica para evitar duplicar señales cuando
        el executor recibe una señal ya guardada por el router.
        """

        self._assert_connected()
        row = self._conn.execute(  # type: ignore[union-attr]
            """
            SELECT id
            FROM signals
            WHERE ticker = ?
              AND decision = ?
              AND created_at = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (signal.market_ticker, signal.decision.value, signal.timestamp),
        ).fetchone()
        if row is None:
            return None
        return int(row["id"])

    # ─── Trades ───────────────────────────────────────────────────────────────

    def save_trade(self, trade: Trade) -> int:
        """
        Persiste un trade nuevo (apertura).

        Returns:
            ID asignado por SQLite.
        """
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO trades
                    (signal_id, ticker, side, contracts, entry_price,
                     exit_price, fee_paid, pnl, mode, status, opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.signal_id,
                    trade.ticker,
                    trade.side,
                    trade.contracts,
                    trade.entry_price,
                    trade.exit_price,
                    trade.fee_paid,
                    trade.pnl,
                    trade.mode.value,
                    trade.status.value,
                    trade.opened_at,
                    trade.closed_at,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def close_trade(self, trade_id: int, exit_price: float, pnl: float, fee_paid: float) -> None:
        """Actualiza un trade al cerrarse."""
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE trades
                SET exit_price=?, pnl=?, fee_paid=?, status=?, closed_at=?
                WHERE id=?
                """,
                (exit_price, pnl, fee_paid, TradeStatus.CLOSED.value, time.time(), trade_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Trade ID {trade_id} no encontrado")

    def get_open_trades(self, mode: TradeMode | None = None) -> list[Trade]:
        """Retorna todos los trades con status='open'."""
        self._assert_connected()
        query = "SELECT * FROM trades WHERE status = ?"
        params: list = [TradeStatus.OPEN.value]
        if mode is not None:
            query += " AND mode = ?"
            params.append(mode.value)
        rows = self._conn.execute(query, params).fetchall()  # type: ignore[union-attr]
        return [_row_to_trade(row) for row in rows]

    # ─── Parámetros del backtesting ───────────────────────────────────────────

    def upsert_param(
        self,
        key: str,
        value: float,
        category: str | None,
        win_rate: float,
        sample_size: int,
    ) -> None:
        """
        Inserta o actualiza un parámetro calibrado.

        Invalida el parámetro anterior marcando valid_until antes de insertar el nuevo.
        """
        now = time.time()
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE backtest_params
                SET valid_until = ?
                WHERE param_key = ?
                  AND (category IS ? OR category = ?)
                  AND valid_until IS NULL
                """,
                (now, key, category, category),
            )
            cur.execute(
                """
                INSERT INTO backtest_params
                    (param_key, param_value, category, win_rate, sample_size, valid_from)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, value, category, win_rate, sample_size, now),
            )

    def get_current_params(self, category: str | None = None) -> dict[str, float]:
        """
        Retorna los parámetros vigentes para una categoría.

        Primero busca parámetros específicos de la categoría, luego globales.

        Returns:
            Dict {param_key: param_value}. Vacío si no hay parámetros.
        """
        self._assert_connected()
        rows = self._conn.execute(  # type: ignore[union-attr]
            """
            SELECT param_key, param_value
            FROM backtest_params
            WHERE valid_until IS NULL
              AND (category IS NULL OR category = ?)
            ORDER BY category NULLS LAST
            """,
            (category,),
        ).fetchall()

        result: dict[str, float] = {}
        for row in rows:
            if row["param_key"] not in result:
                result[row["param_key"]] = row["param_value"]
        return result

    # ─── Categorías bloqueadas ────────────────────────────────────────────────

    def block_category(self, category: str, win_rate: float, sample_size: int, reason: str) -> None:
        """Bloquea una categoría con upsert (no lanza si ya existe)."""
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO blocked_categories (category, win_rate, sample_size, blocked_at, reason)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(category) DO UPDATE SET
                    win_rate=excluded.win_rate,
                    sample_size=excluded.sample_size,
                    blocked_at=excluded.blocked_at,
                    reason=excluded.reason
                """,
                (category, win_rate, sample_size, time.time(), reason),
            )

    def unblock_category(self, category: str) -> None:
        with self._transaction() as cur:
            cur.execute("DELETE FROM blocked_categories WHERE category=?", (category,))

    def get_blocked_categories(self) -> set[str]:
        self._assert_connected()
        rows = self._conn.execute(  # type: ignore[union-attr]
            "SELECT category FROM blocked_categories"
        ).fetchall()
        return {row["category"] for row in rows}

    # ─── Migraciones ──────────────────────────────────────────────────────────

    def _apply_migrations(self, cur: sqlite3.Cursor) -> None:
        """
        Aplica el schema inicial y migraciones pendientes.

        Solo se ejecuta lo que aún no existe → idempotente.
        NUNCA usar DROP TABLE, ALTER COLUMN, o cambiar tipos existentes.
        """
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  REAL NOT NULL
            )
            """
        )

        current = cur.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0

        if current < 1:
            self._migration_v1(cur)
            cur.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (1, time.time()),
            )
            logger.info("Migración v1 aplicada")

        # Template para futuras migraciones:
        # if current < 2:
        #     self._migration_v2(cur)
        #     cur.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)", (2, time.time()))

    def _migration_v1(self, cur: sqlite3.Cursor) -> None:
        """Schema inicial → todas las tablas y sus índices."""
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker            TEXT    NOT NULL,
                decision          TEXT    NOT NULL,
                my_prob           REAL    NOT NULL,
                market_prob       REAL    NOT NULL,
                delta             REAL    NOT NULL,
                ev_net_fees       REAL    NOT NULL,
                kelly_size        REAL    NOT NULL,
                confidence        TEXT    NOT NULL,
                time_remaining_s  INTEGER NOT NULL,
                reasoning         TEXT,
                created_at        REAL    NOT NULL,
                outcome           TEXT,
                outcome_at        REAL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id    INTEGER REFERENCES signals(id),
                ticker       TEXT    NOT NULL,
                side         TEXT    NOT NULL,
                contracts    INTEGER NOT NULL,
                entry_price  REAL    NOT NULL,
                exit_price   REAL,
                fee_paid     REAL    NOT NULL DEFAULT 0,
                pnl          REAL,
                mode         TEXT    NOT NULL,
                status       TEXT    NOT NULL,
                opened_at    REAL    NOT NULL,
                closed_at    REAL
            );

            CREATE TABLE IF NOT EXISTS backtest_params (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                param_key    TEXT    NOT NULL,
                param_value  REAL    NOT NULL,
                category     TEXT,
                win_rate     REAL,
                sample_size  INTEGER,
                valid_from   REAL    NOT NULL,
                valid_until  REAL
            );

            CREATE TABLE IF NOT EXISTS blocked_categories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                category     TEXT    NOT NULL UNIQUE,
                win_rate     REAL    NOT NULL,
                sample_size  INTEGER NOT NULL,
                blocked_at   REAL    NOT NULL,
                reason       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_signals_ticker  ON signals(ticker);
            CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
            CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome);
            CREATE INDEX IF NOT EXISTS idx_trades_status   ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_ticker   ON trades(ticker);
            CREATE INDEX IF NOT EXISTS idx_params_key      ON backtest_params(param_key, valid_until);
            """
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager que garantiza commit o rollback."""
        self._assert_connected()
        conn = self._conn  # type: ignore[assignment]
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            logger.error("SQLite error → rollback ejecutado: %s", exc)
            raise

    def _assert_connected(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "Database no inicializada → llama a initialize() antes de operar"
            )


# ─── Conversores de filas ─────────────────────────────────────────────────────

def _row_to_signal(row: sqlite3.Row) -> Signal:
    return Signal(
        market_ticker      = row["ticker"],
        decision           = Decision(row["decision"]),
        my_probability     = row["my_prob"],
        market_probability = row["market_prob"],
        delta              = row["delta"],
        ev_net_fees        = row["ev_net_fees"],
        kelly_size         = row["kelly_size"],
        confidence         = Confidence(row["confidence"]),
        time_remaining_s   = row["time_remaining_s"],
        reasoning          = row["reasoning"] or "",
        timestamp          = row["created_at"],
        outcome            = Outcome(row["outcome"]) if row["outcome"] else None,
        outcome_at         = row["outcome_at"],
    )


def _row_to_trade(row: sqlite3.Row) -> Trade:
    return Trade(
        id          = row["id"],
        signal_id   = row["signal_id"],
        ticker      = row["ticker"],
        side        = row["side"],
        contracts   = row["contracts"],
        entry_price = row["entry_price"],
        exit_price  = row["exit_price"],
        fee_paid    = row["fee_paid"],
        pnl         = row["pnl"],
        mode        = TradeMode(row["mode"]),
        status      = TradeStatus(row["status"]),
        opened_at   = row["opened_at"],
        closed_at   = row["closed_at"],
    )
