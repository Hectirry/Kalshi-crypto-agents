"""
memory/openclaw_adapter.py

Integración opcional con el workspace memory de OpenClaw.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.models import Signal, Trade


@dataclass(slots=True)
class OpenClawMemoryAdapter:
    """Escribe eventos del bot en un workspace compatible con OpenClaw."""

    workspace: Path

    def initialize(self) -> None:
        """Crea la estructura mínima del workspace memory."""

        (self.workspace / "memory").mkdir(parents=True, exist_ok=True)
        memory_md = self.workspace / "memory.md"
        if not memory_md.exists():
            memory_md.write_text(
                "# Kalshi Crypto Agents Memory\n\n"
                "- This file stores durable operator-facing notes for the trading bot.\n",
                encoding="utf-8",
            )

    def record_session_start(self, *, mode: str, bankroll: float, total_pnl: float, go_allowed: bool) -> Path:
        """Registra el arranque de una sesión."""

        return self._append(
            section="Session",
            line=(
                f"Started bot session mode={mode} bankroll={bankroll:.2f} "
                f"total_pnl={total_pnl:.4f} go_allowed={go_allowed}"
            ),
            retain=True,
        )

    def record_trade_open(self, trade: Trade, signal: Signal, price_source: str) -> Path:
        """Registra una apertura de trade."""

        return self._append(
            section="Trades",
            line=(
                f"Opened {trade.side} {trade.ticker} contracts={trade.contracts} "
                f"entry={trade.entry_price:.4f} kelly={signal.kelly_size:.4f} "
                f"delta={signal.delta:.4f} ev_net={signal.ev_net_fees:.4f} "
                f"price_source={price_source}"
            ),
            retain=True,
        )

    def record_trade_blocked(self, *, ticker: str, reason: str) -> Path:
        """Registra un bloqueo de trade."""

        return self._append(
            section="Risk",
            line=f"Blocked trade for {ticker} reason={reason}",
            retain=False,
        )

    def record_trade_close(self, *, trade: Trade, reason: str) -> Path:
        """Registra un cierre de trade."""

        return self._append(
            section="Trades",
            line=(
                f"Closed {trade.side} {trade.ticker} reason={reason} "
                f"exit={trade.exit_price or 0.0:.4f} pnl={trade.pnl or 0.0:.4f}"
            ),
            retain=True,
        )

    def _append(self, *, section: str, line: str, retain: bool) -> Path:
        """Agrega una línea al journal diario en Markdown."""

        now = datetime.now(timezone.utc)
        daily = self.workspace / "memory" / f"{now.date().isoformat()}.md"
        if not daily.exists():
            daily.write_text(
                f"# Memory {now.date().isoformat()}\n\n",
                encoding="utf-8",
            )
        with daily.open("a", encoding="utf-8") as handle:
            handle.write(f"## {section}\n")
            handle.write(f"- {now.isoformat()} {line}\n")
            if retain:
                handle.write("## Retain\n")
                handle.write(f"- B @kalshi-bot: {line}\n")
        return daily
