"""
Limpia trades inconsistentes con la regla actual de un trade por ticker.

Hace backup de la DB antes de borrar.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup de trades inconsistentes")
    parser.add_argument(
        "--db-path",
        default="/root/Kalshi-crypto-agents/data/trading.db",
        help="Ruta a la base SQLite",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica los borrados. Sin esto corre en dry-run.",
    )
    return parser.parse_args()


def _find_trade_ids_to_delete(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT id, ticker, status, opened_at
        FROM trades
        WHERE status IN ('open', 'closed')
        ORDER BY ticker ASC, opened_at ASC, id ASC
        """
    ).fetchall()

    seen_tickers: set[str] = set()
    delete_ids: list[int] = []
    for row in rows:
        ticker = str(row["ticker"])
        if ticker in seen_tickers:
            delete_ids.append(int(row["id"]))
            continue
        seen_tickers.add(ticker)
    return delete_ids


def _backup_db(path: Path) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.stem}.backup-{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    delete_ids = _find_trade_ids_to_delete(conn)
    print(f"duplicate_trades_to_delete={len(delete_ids)}")
    if delete_ids:
        sample = conn.execute(
            f"""
            SELECT id, ticker, side, status, opened_at, pnl
            FROM trades
            WHERE id IN ({','.join('?' for _ in delete_ids[:20])})
            ORDER BY ticker, opened_at, id
            """,
            delete_ids[:20],
        ).fetchall()
        for row in sample:
            print(
                "sample",
                dict(row),
            )

    if not args.apply:
        print("dry_run_only=True")
        conn.close()
        return

    backup_path = _backup_db(db_path)
    with conn:
        conn.executemany("DELETE FROM trades WHERE id = ?", [(trade_id,) for trade_id in delete_ids])
    conn.close()
    print(f"backup_created={backup_path}")
    print("cleanup_applied=True")


if __name__ == "__main__":
    main()
