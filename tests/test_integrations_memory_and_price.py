"""
tests/test_integrations_memory_and_price.py

Cobertura para consenso multi-feed y adapter opcional OpenClaw.
"""

from __future__ import annotations

from pathlib import Path

from engine.price_resolver import resolve_reference_price
from memory.openclaw_adapter import OpenClawMemoryAdapter


class TestPriceResolver:
    def test_consensus_uses_binance_and_hyperliquid(self, make_price_snapshot):
        now = make_price_snapshot().timestamp
        resolution = resolve_reference_price(
            symbol="BTC",
            latest_prices={
                "BTC": {
                    "binance": make_price_snapshot(price=95_000.0, timestamp=now, source="binance"),
                    "hyperliquid": make_price_snapshot(
                        price=95_010.0,
                        timestamp=now,
                        source="hyperliquid",
                        bid=None,
                        ask=None,
                    ),
                }
            },
            now_ts=now,
        )

        assert resolution.snapshot is not None
        assert resolution.snapshot.source == "consensus"
        assert resolution.blocked_reason is None
        assert resolution.used_sources == ("binance", "hyperliquid")

    def test_divergence_blocks_reference_price(self, make_price_snapshot):
        now = make_price_snapshot().timestamp
        resolution = resolve_reference_price(
            symbol="BTC",
            latest_prices={
                "BTC": {
                    "binance": make_price_snapshot(price=95_000.0, timestamp=now, source="binance"),
                    "hyperliquid": make_price_snapshot(
                        price=96_500.0,
                        timestamp=now,
                        source="hyperliquid",
                        bid=None,
                        ask=None,
                    ),
                }
            },
            now_ts=now,
        )

        assert resolution.snapshot is None
        assert resolution.blocked_reason == "cross_exchange_divergence"
        assert resolution.spread_pct is not None


class TestOpenClawMemoryAdapter:
    def test_initialize_creates_workspace_files(self, tmp_path: Path):
        adapter = OpenClawMemoryAdapter(workspace=tmp_path / "workspace")

        adapter.initialize()

        assert (adapter.workspace / "memory").exists()
        assert (adapter.workspace / "memory.md").exists()

    def test_records_session_and_trade_events(self, tmp_path: Path, make_signal, make_trade):
        adapter = OpenClawMemoryAdapter(workspace=tmp_path / "workspace")
        adapter.initialize()

        session_path = adapter.record_session_start(
            mode="demo",
            bankroll=100.0,
            total_pnl=-5.0,
            go_allowed=False,
        )
        trade_path = adapter.record_trade_open(
            make_trade(),
            make_signal(),
            "consensus",
        )

        content = session_path.read_text(encoding="utf-8")
        assert "Started bot session" in content
        assert "## Retain" in content
        assert trade_path == session_path
        assert "Opened YES" in trade_path.read_text(encoding="utf-8")
