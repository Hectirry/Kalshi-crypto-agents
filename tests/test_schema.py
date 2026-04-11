"""
tests/test_schema.py

Puerta de salida de Fase 1.
Todos estos tests deben pasar en verde antes de arrancar Fase 2.

Cubre:
- Schema SQLite: tablas, índices, migraciones
- Models: validación de dataclasses y edge cases
- Config: carga desde env vars, fallo explícito si faltan variables
- Interfaces: que los protocolos sean verificables en runtime
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Schema SQLite
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseSchema:
    def test_initialize_creates_all_tables(self, db):
        """Todas las tablas del schema v1 deben existir tras initialize()."""
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "signals"            in tables
        assert "trades"             in tables
        assert "backtest_params"    in tables
        assert "blocked_categories" in tables
        assert "schema_version"     in tables

    def test_schema_version_recorded(self, db):
        """La versión del schema debe estar registrada en schema_version."""
        version = db._conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
        assert version == 2

    def test_initialize_is_idempotent(self, db_path):
        """Llamar initialize() dos veces no debe lanzar excepciones."""
        from core.database import Database
        db1 = Database(path=db_path)
        db1.initialize()
        db1.close()

        db2 = Database(path=db_path)
        db2.initialize()  # no debe fallar
        db2.close()

    def test_required_indexes_exist(self, db):
        """Los índices de rendimiento deben existir."""
        indexes = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_signals_ticker"  in indexes
        assert "idx_signals_created" in indexes
        assert "idx_trades_status"   in indexes

    def test_signals_foreign_key_enabled(self, db):
        """Las foreign keys deben estar activas."""
        result = db._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert result == 1

    def test_operations_before_initialize_raise(self, db_path):
        """Operar sin initialize() debe lanzar RuntimeError."""
        from core.database import Database
        uninitialized = Database(path=db_path)
        with pytest.raises(RuntimeError, match="initialize()"):
            uninitialized.get_open_trades()


class TestSignalPersistence:
    def test_save_and_retrieve_signal(self, db, make_signal):
        signal = make_signal(contract_price=0.57, market_overround_bps=80.0)
        signal_id = db.save_signal(signal)

        assert signal_id > 0

        now = time.time()
        signals = db.get_signals(from_ts=now - 60, to_ts=now + 60)
        assert len(signals) == 1
        saved = signals[0]
        assert saved.market_ticker      == signal.market_ticker
        assert saved.delta              == signal.delta
        assert saved.ev_net_fees        == signal.ev_net_fees
        assert saved.contract_price     == 0.57
        assert saved.market_overround_bps == 80.0
        assert saved.outcome            is None

    def test_migration_v2_adds_signal_microstructure_columns(self, db):
        columns = {
            row["name"]
            for row in db._conn.execute("PRAGMA table_info(signals)").fetchall()
        }
        assert "contract_price" in columns
        assert "market_overround_bps" in columns

    def test_update_signal_outcome(self, db, make_signal):
        from core.models import Outcome
        signal = make_signal()
        signal_id = db.save_signal(signal)
        outcome_ts = time.time()

        db.update_signal_outcome(signal_id, Outcome.WIN, outcome_ts)

        signals = db.get_signals(from_ts=time.time() - 60, to_ts=time.time() + 60)
        assert signals[0].outcome == Outcome.WIN

    def test_update_nonexistent_signal_raises(self, db):
        from core.models import Outcome
        with pytest.raises(ValueError, match="no encontrado"):
            db.update_signal_outcome(99999, Outcome.WIN, time.time())

    def test_filter_signals_by_category(self, db, make_signal):
        db.save_signal(make_signal(market_ticker="KXBTC-15MIN-B95000"))
        db.save_signal(make_signal(market_ticker="KXETH-15MIN-B3000"))

        now = time.time()
        btc_signals = db.get_signals(from_ts=now - 60, to_ts=now + 60, category="BTC")
        eth_signals = db.get_signals(from_ts=now - 60, to_ts=now + 60, category="ETH")

        assert all("BTC" in s.market_ticker for s in btc_signals)
        assert all("ETH" in s.market_ticker for s in eth_signals)

    def test_get_signals_empty_range_returns_empty(self, db):
        result = db.get_signals(from_ts=0, to_ts=1)
        assert result == []

    def test_find_signal_id_returns_rowid(self, db, make_signal):
        signal = make_signal()
        signal_id = db.save_signal(signal)

        found = db.find_signal_id(signal)

        assert found == signal_id


class TestTradePersistence:
    def test_save_and_retrieve_open_trade(self, db, make_trade):
        trade = make_trade()
        trade_id = db.save_trade(trade)
        assert trade_id > 0

        open_trades = db.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].ticker == trade.ticker

    def test_close_trade(self, db, make_trade):
        trade_id = db.save_trade(make_trade())
        db.close_trade(trade_id, exit_price=1.0, pnl=4.4, fee_paid=0.56)

        open_trades = db.get_open_trades()
        assert len(open_trades) == 0

    def test_close_nonexistent_trade_raises(self, db):
        with pytest.raises(ValueError, match="no encontrado"):
            db.close_trade(99999, exit_price=1.0, pnl=0.0, fee_paid=0.0)


class TestParamPersistence:
    def test_upsert_and_get_param(self, db):
        db.upsert_param("min_ev_threshold", 0.05, None, 0.62, 100)
        params = db.get_current_params()
        assert "min_ev_threshold" in params
        assert params["min_ev_threshold"] == 0.05

    def test_upsert_invalidates_previous_param(self, db):
        db.upsert_param("min_ev_threshold", 0.04, None, 0.60, 80)
        db.upsert_param("min_ev_threshold", 0.06, None, 0.65, 120)

        params = db.get_current_params()
        assert params["min_ev_threshold"] == 0.06

        history = db._conn.execute(
            "SELECT COUNT(*) FROM backtest_params WHERE param_key='min_ev_threshold'"
        ).fetchone()[0]
        assert history == 2

    def test_category_specific_param(self, db):
        db.upsert_param("min_delta", 0.05, None,  0.60, 100)
        db.upsert_param("min_delta", 0.08, "BTC", 0.70, 50)

        btc_params = db.get_current_params(category="BTC")
        assert btc_params["min_delta"] == 0.08


class TestBlockedCategories:
    def test_block_and_retrieve(self, db):
        db.block_category("POLITICS", win_rate=0.28, sample_size=25, reason="ROI -40%")
        blocked = db.get_blocked_categories()
        assert "POLITICS" in blocked

    def test_unblock_category(self, db):
        db.block_category("POLITICS", win_rate=0.28, sample_size=25, reason="test")
        db.unblock_category("POLITICS")
        assert "POLITICS" not in db.get_blocked_categories()

    def test_block_upsert_updates_existing(self, db):
        db.block_category("ECON", win_rate=0.30, sample_size=20, reason="initial")
        db.block_category("ECON", win_rate=0.25, sample_size=40, reason="updated")

        blocked = db.get_blocked_categories()
        assert "ECON" in blocked
        row = db._conn.execute(
            "SELECT win_rate, sample_size FROM blocked_categories WHERE category='ECON'"
        ).fetchone()
        assert row["win_rate"]    == 0.25
        assert row["sample_size"] == 40


# ─────────────────────────────────────────────────────────────────────────────
# Models → validación y edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceSnapshot:
    def test_valid_snapshot(self, make_price_snapshot):
        snap = make_price_snapshot()
        assert snap.price > 0

    def test_negative_price_raises(self, make_price_snapshot):
        with pytest.raises(ValueError, match="price"):
            make_price_snapshot(price=-1.0)

    def test_zero_price_raises(self, make_price_snapshot):
        with pytest.raises(ValueError, match="price"):
            make_price_snapshot(price=0.0)

    def test_bid_greater_than_ask_raises(self, make_price_snapshot):
        with pytest.raises(ValueError, match="bid"):
            make_price_snapshot(bid=95_002.0, ask=95_001.0)

    def test_none_bid_ask_valid(self, make_price_snapshot):
        snap = make_price_snapshot(bid=None, ask=None)
        assert snap.bid is None


class TestMarketSnapshot:
    def test_valid_snapshot(self, make_market_snapshot):
        snap = make_market_snapshot()
        assert 0.0 <= snap.implied_prob <= 1.0

    def test_implied_prob_above_one_raises(self, make_market_snapshot):
        with pytest.raises(ValueError, match="implied_prob"):
            make_market_snapshot(implied_prob=1.01)

    def test_negative_time_to_expiry_raises(self, make_market_snapshot):
        with pytest.raises(ValueError, match="time_to_expiry_s"):
            make_market_snapshot(time_to_expiry_s=-1)

    def test_zero_expiry_valid(self, make_market_snapshot):
        snap = make_market_snapshot(time_to_expiry_s=0)
        assert snap.time_to_expiry_s == 0


class TestSignal:
    def test_is_actionable_yes(self, make_signal):
        from core.models import Decision
        s = make_signal(decision=Decision.YES)
        assert s.is_actionable is True

    def test_is_actionable_no(self, make_signal):
        from core.models import Decision
        s = make_signal(decision=Decision.NO)
        assert s.is_actionable is True

    def test_is_not_actionable_skip(self, make_signal):
        from core.models import Decision
        s = make_signal(decision=Decision.SKIP, delta=0.0, ev_net_fees=-0.01, kelly_size=0.0)
        assert s.is_actionable is False

    def test_make_error_factory(self):
        from core.models import Decision, Signal
        s = Signal.make_error("KXBTC-TEST", "Test error", time.time())
        assert s.decision == Decision.ERROR
        assert s.error_msg == "Test error"
        assert s.kelly_size == 0.0

    def test_make_skip_factory(self):
        from core.models import Decision, Signal
        s = Signal.make_skip("KXBTC-TEST", "fees > ev", 0.55, time.time())
        assert s.decision == Decision.SKIP
        assert s.market_probability == 0.55

    def test_kelly_size_above_one_raises(self, make_signal):
        with pytest.raises(ValueError, match="kelly_size"):
            make_signal(kelly_size=1.5)

    def test_negative_market_overround_raises(self, make_signal):
        with pytest.raises(ValueError, match="market_overround_bps"):
            make_signal(market_overround_bps=-1.0)


class TestTrade:
    def test_valid_trade(self, make_trade):
        trade = make_trade()
        assert trade.is_open is True

    def test_zero_contracts_raises(self, make_trade):
        with pytest.raises(ValueError, match="contracts"):
            make_trade(contracts=0)

    def test_negative_contracts_raises(self, make_trade):
        with pytest.raises(ValueError, match="contracts"):
            make_trade(contracts=-5)

    def test_entry_price_above_one_raises(self, make_trade):
        with pytest.raises(ValueError, match="entry_price"):
            make_trade(entry_price=1.1)

    def test_gross_value(self, make_trade):
        trade = make_trade(contracts=10, entry_price=0.56)
        assert abs(trade.gross_value - 5.6) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_loads_in_demo_mode(self, app_config):
        assert app_config.is_demo is True
        assert app_config.is_production is False

    def test_kalshi_key_loaded_from_env(self, app_config):
        assert app_config.kalshi.api_key == "test-kalshi-key"

    def test_engine_defaults_valid(self, app_config):
        cfg = app_config.engine
        assert 0.0 < cfg.min_ev_threshold < 1.0
        assert 0.0 < cfg.kelly_fraction <= 1.0
        assert 0.0 < cfg.min_contract_price < cfg.max_contract_price < 1.0
        assert cfg.max_market_overround_bps >= 0.0
        assert cfg.min_time_remaining_s >= 30

    def test_category_override_loaded_from_config(self, app_config):
        btc = app_config.engine.category_overrides["BTC"]
        assert btc.min_delta == 0.25
        assert btc.min_ev_threshold == 0.30
        assert btc.min_time_remaining_s == 180
        assert btc.max_contract_price == 0.70

    def test_social_sentiment_defaults_loaded_from_config(self, app_config):
        cfg = app_config.social_sentiment
        assert cfg.enabled is False
        assert cfg.provider == "reddit"
        assert cfg.supported_assets == ["BTC", "ETH", "SOL"]
        assert cfg.ttl_s >= cfg.refresh_interval_s

    def test_setup_quality_gate_defaults_loaded_from_config(self, app_config):
        cfg = app_config.engine
        assert cfg.setup_quality_gate_enabled is True
        assert cfg.setup_quality_min_samples == 3
        assert cfg.setup_quality_min_win_rate == pytest.approx(0.40)
        assert cfg.setup_quality_history_limit == 500

    def test_missing_kalshi_key_raises(self):
        from core.config import load_config
        with patch.dict(
            os.environ,
            {"KALSHI_API_KEY": "", "KALSHI_API_KEY_ID": ""},
            clear=False,
        ):
            with pytest.raises(EnvironmentError, match="Kalshi key"):
                load_config()

    def test_invalid_env_value_raises(self):
        from core.config import load_config
        with patch.dict(os.environ, {"ENV": "staging"}, clear=False):
            with pytest.raises(ValueError, match="ENV"):
                load_config()

    def test_invalid_engine_param_raises(self, tmp_path):
        from core.config import load_config
        import json
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"engine": {"min_ev_threshold": 1.5}}))
        with pytest.raises(ValueError, match="min_ev_threshold"):
            load_config(config_file=config_file)

    def test_db_directory_created(self, tmp_path):
        from core.config import load_config
        db_path = tmp_path / "nested" / "dir" / "trading.db"
        with patch.dict(os.environ, {"DB_PATH": str(db_path)}, clear=False):
            config = load_config()
            assert config.database.path.parent.exists()

    def test_social_sentiment_env_override(self):
        from core.config import load_config
        with patch.dict(
            os.environ,
            {
                "SOCIAL_SENTIMENT_ENABLED": "true",
                "SOCIAL_SENTIMENT_TTL_S": "1200",
                "SOCIAL_SENTIMENT_ASSETS": "btc,eth,sol",
            },
            clear=False,
        ):
            config = load_config()
            assert config.social_sentiment.enabled is True
            assert config.social_sentiment.ttl_s == 1200
            assert config.social_sentiment.supported_assets == ["BTC", "ETH", "SOL"]


# ─────────────────────────────────────────────────────────────────────────────
# Interfaces → verificación de protocolos en runtime
# ─────────────────────────────────────────────────────────────────────────────

class TestInterfaces:
    def test_event_bus_publish_and_consume(self):
        """El EventBus debe publicar y entregar mensajes."""
        import asyncio
        from core.interfaces import EventBus

        async def _run():
            bus = EventBus(maxsize=10)
            from core.models import PriceSnapshot
            snap = PriceSnapshot(
                symbol="BTC", price=95_000.0, timestamp=time.time(), source="test"
            )
            await bus.publish(snap)
            assert bus.qsize == 1

        asyncio.run(_run())

    def test_event_bus_sliding_window_on_full(self):
        """Bus lleno debe descartar el más antiguo al publicar."""
        import asyncio
        from core.interfaces import EventBus
        from core.models import PriceSnapshot

        async def _run():
            bus = EventBus(maxsize=2)
            for price in [90_000.0, 91_000.0, 92_000.0]:
                snap = PriceSnapshot(
                    symbol="BTC", price=price, timestamp=time.time(), source="test"
                )
                await bus.publish(snap)
            assert bus.qsize == 2

        asyncio.run(_run())

    def test_mock_feed_satisfies_protocol(self, mock_price_feed):
        """El mock debe satisfacer el protocolo PriceFeed en runtime."""
        from core.interfaces import PriceFeed
        assert isinstance(mock_price_feed, PriceFeed)

    def test_mock_scanner_satisfies_protocol(self, mock_market_scanner):
        """El mock debe satisfacer el protocolo MarketScanner en runtime."""
        from core.interfaces import MarketScanner
        assert isinstance(mock_market_scanner, MarketScanner)
