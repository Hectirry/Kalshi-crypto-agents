"""
tests/test_analytics.py

Cobertura para analytics/execution_quality.py y el endpoint
GET /analytics/execution-quality del dashboard.
"""

from __future__ import annotations

import time

import pytest

from analytics.execution_quality import (
    DELTA_BUCKETS,
    OVERROUND_BUCKETS,
    BucketStats,
    ExecutionQualityAnalyzer,
    ExecutionQualityReport,
    _aggregate,
    _delta_bucket,
    _infer_category,
    _overround_bucket,
    _suggest_overround_threshold,
)
from core.models import Confidence, Decision, Outcome, Trade, TradeMode, TradeStatus


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _seed_resolved_trade(db, *, ticker="KXBTC-15MIN-B95000", outcome=Outcome.WIN,
                          delta=0.20, overround_bps=75.0, contract_price=0.55,
                          pnl=1.5, my_prob=0.70, market_prob=0.55):
    """Inserta una señal resuelta con su trade cerrado."""
    from core.models import Signal

    signal = Signal(
        market_ticker=ticker,
        decision=Decision.YES,
        my_probability=my_prob,
        market_probability=market_prob,
        delta=delta,
        ev_net_fees=0.08,
        kelly_size=0.10,
        confidence=Confidence.HIGH,
        time_remaining_s=300,
        reasoning="test",
        timestamp=time.time(),
        contract_price=contract_price,
        market_overround_bps=overround_bps,
        outcome=outcome,
    )
    signal_id = db.save_signal(signal)

    trade = Trade(
        ticker=ticker,
        side="YES",
        contracts=5,
        entry_price=contract_price,
        mode=TradeMode.DEMO,
        status=TradeStatus.OPEN,
        opened_at=time.time(),
        signal_id=signal_id,
    )
    trade_id = db.save_trade(trade)
    db.close_trade(trade_id, exit_price=1.0 if outcome == Outcome.WIN else 0.0,
                   pnl=pnl, fee_paid=0.05)
    return signal_id, trade_id


# ─── Clasificadores ───────────────────────────────────────────────────────────

class TestClassifiers:
    def test_infer_category_btc(self):
        assert _infer_category("KXBTC-15MIN-B95000") == "BTC"

    def test_infer_category_eth(self):
        assert _infer_category("KXETH-15MIN-B3200") == "ETH"

    def test_infer_category_sol(self):
        assert _infer_category("KXSOL-15MIN-B200") == "SOL"

    def test_infer_category_unknown(self):
        assert _infer_category("KXLINK-15MIN-B20") == "OTHER"

    def test_overround_bucket_low(self):
        assert _overround_bucket(25.0) == "0-50bps"

    def test_overround_bucket_mid(self):
        assert _overround_bucket(75.0) == "50-100bps"

    def test_overround_bucket_high(self):
        assert _overround_bucket(120.0) == "100-150bps"

    def test_overround_bucket_very_high(self):
        assert _overround_bucket(200.0) == "150+bps"

    def test_overround_bucket_boundary_50(self):
        assert _overround_bucket(50.0) == "50-100bps"

    def test_overround_bucket_none(self):
        assert _overround_bucket(None) == "unknown"

    def test_delta_bucket_small(self):
        assert _delta_bucket(0.10) == "0.00-0.15"

    def test_delta_bucket_mid(self):
        assert _delta_bucket(0.20) == "0.15-0.25"

    def test_delta_bucket_large(self):
        assert _delta_bucket(0.30) == "0.25-0.40"

    def test_delta_bucket_xlarge(self):
        assert _delta_bucket(0.50) == "0.40+"

    def test_delta_bucket_negative(self):
        # Usa valor absoluto → -0.20 y 0.20 deben caer en el mismo bucket
        assert _delta_bucket(-0.20) == _delta_bucket(0.20)

    def test_delta_bucket_none(self):
        assert _delta_bucket(None) == "unknown"


# ─── _aggregate ───────────────────────────────────────────────────────────────

class TestAggregate:
    def test_empty_returns_none(self):
        assert _aggregate("test", []) is None

    def test_single_win(self):
        row = {
            "outcome": "WIN",
            "pnl": 2.0,
            "contract_price": 0.55,
            "market_overround_bps": 80.0,
            "delta": 0.20,
            "my_prob": 0.70,
            "entry_price": 0.55,
        }
        stats = _aggregate("test", [row])
        assert stats is not None
        assert stats.sample_size == 1
        assert stats.win_rate == 1.0
        assert stats.total_pnl == 2.0
        assert stats.avg_pnl == 2.0
        assert stats.avg_contract_price == pytest.approx(0.55)
        assert stats.avg_overround_bps == pytest.approx(80.0)
        assert stats.avg_delta == pytest.approx(0.20)
        # entry_edge_bps = (0.70 - 0.55) * 10000 = 1500
        assert stats.avg_entry_edge_bps == pytest.approx(1500.0)

    def test_mixed_win_loss(self):
        rows = [
            {"outcome": "WIN",  "pnl": 3.0, "contract_price": 0.60,
             "market_overround_bps": 60.0, "delta": 0.25, "my_prob": 0.75, "entry_price": 0.60},
            {"outcome": "LOSS", "pnl": -1.0, "contract_price": 0.40,
             "market_overround_bps": 120.0, "delta": 0.15, "my_prob": 0.55, "entry_price": 0.40},
        ]
        stats = _aggregate("mixed", rows)
        assert stats is not None
        assert stats.sample_size == 2
        assert stats.win_rate == 0.5
        assert stats.total_pnl == pytest.approx(2.0)
        assert stats.avg_pnl == pytest.approx(1.0)

    def test_none_pnl_excluded_from_avg(self):
        rows = [
            {"outcome": "WIN", "pnl": 4.0, "contract_price": 0.55,
             "market_overround_bps": 50.0, "delta": 0.20, "my_prob": 0.70, "entry_price": 0.55},
            {"outcome": "LOSS", "pnl": None, "contract_price": 0.55,
             "market_overround_bps": 50.0, "delta": 0.20, "my_prob": 0.70, "entry_price": 0.55},
        ]
        stats = _aggregate("partial_pnl", rows)
        assert stats is not None
        assert stats.total_pnl == 4.0
        assert stats.avg_pnl == 4.0  # Solo cuenta la fila con pnl


# ─── _suggest_overround_threshold ────────────────────────────────────────────

class TestSuggestOverroundThreshold:
    def _make_stats(self, label: str, avg_pnl: float, n: int = 5) -> BucketStats:
        return BucketStats(
            label=label,
            sample_size=n,
            win_rate=0.6 if avg_pnl >= 0 else 0.4,
            total_pnl=avg_pnl * n,
            avg_pnl=avg_pnl,
            avg_contract_price=0.55,
            avg_overround_bps=75.0,
            avg_delta=0.20,
            avg_entry_edge_bps=1500.0,
        )

    def test_all_profitable_returns_none(self):
        by_or = {
            "0-50bps":    self._make_stats("0-50bps",    2.0),
            "50-100bps":  self._make_stats("50-100bps",  1.0),
            "100-150bps": self._make_stats("100-150bps", 0.5),
            "150+bps":    self._make_stats("150+bps",    0.1),
        }
        assert _suggest_overround_threshold(by_or, OVERROUND_BUCKETS) is None

    def test_first_losing_bucket_triggers_suggestion(self):
        by_or = {
            "0-50bps":    self._make_stats("0-50bps",    2.0),
            "50-100bps":  self._make_stats("50-100bps",  1.0),
            "100-150bps": self._make_stats("100-150bps", -0.3),
            "150+bps":    self._make_stats("150+bps",    -1.0),
        }
        result = _suggest_overround_threshold(by_or, OVERROUND_BUCKETS)
        # El primer bucket perdedor empieza en 100 bps
        assert result == pytest.approx(100.0)

    def test_first_bucket_losing_suggests_zero(self):
        by_or = {
            "0-50bps": self._make_stats("0-50bps", -0.5),
        }
        result = _suggest_overround_threshold(by_or, OVERROUND_BUCKETS)
        assert result == pytest.approx(0.0)

    def test_insufficient_samples_ignored(self):
        # El bucket perdedor tiene solo 2 muestras → se ignora
        by_or = {
            "0-50bps":    self._make_stats("0-50bps",    2.0),
            "100-150bps": self._make_stats("100-150bps", -0.5, n=2),
        }
        result = _suggest_overround_threshold(by_or, OVERROUND_BUCKETS, min_samples=3)
        assert result is None

    def test_empty_dict_returns_none(self):
        assert _suggest_overround_threshold({}, OVERROUND_BUCKETS) is None


# ─── ExecutionQualityAnalyzer ─────────────────────────────────────────────────

class TestExecutionQualityAnalyzer:
    def test_empty_db_returns_empty_report(self, db):
        analyzer = ExecutionQualityAnalyzer(db)
        report = analyzer.analyze()
        assert report.total_resolved == 0
        assert report.by_category == {}
        assert report.by_overround_bucket == {}
        assert report.by_delta_bucket == {}
        assert report.overall_pnl == 0.0
        assert report.suggested_max_overround_bps is None

    def test_single_win_trade_counted(self, db):
        _seed_resolved_trade(
            db,
            ticker="KXBTC-15MIN-B95000",
            outcome=Outcome.WIN,
            delta=0.20,
            overround_bps=75.0,
            pnl=2.5,
        )
        analyzer = ExecutionQualityAnalyzer(db)
        report = analyzer.analyze()

        assert report.total_resolved == 1
        assert report.overall_win_rate == 1.0
        assert report.overall_pnl == pytest.approx(2.5)
        assert "BTC" in report.by_category
        assert report.by_category["BTC"].win_rate == 1.0
        assert report.by_category["BTC"].total_pnl == pytest.approx(2.5)

    def test_groups_by_category(self, db):
        _seed_resolved_trade(db, ticker="KXBTC-15MIN-B95000", outcome=Outcome.WIN,  pnl=2.0)
        _seed_resolved_trade(db, ticker="KXETH-15MIN-B3200",  outcome=Outcome.LOSS, pnl=-1.0)
        _seed_resolved_trade(db, ticker="KXSOL-15MIN-B200",   outcome=Outcome.WIN,  pnl=3.0)

        report = ExecutionQualityAnalyzer(db).analyze()

        assert set(report.by_category.keys()) == {"BTC", "ETH", "SOL"}
        assert report.by_category["BTC"].win_rate == 1.0
        assert report.by_category["ETH"].win_rate == 0.0
        assert report.by_category["SOL"].win_rate == 1.0

    def test_groups_by_overround_bucket(self, db):
        _seed_resolved_trade(db, ticker="KXBTC-15MIN-B95000", overround_bps=30.0,  pnl=2.0)
        _seed_resolved_trade(db, ticker="KXETH-15MIN-B3200",  overround_bps=80.0,  pnl=1.0)
        _seed_resolved_trade(db, ticker="KXSOL-15MIN-B200",   overround_bps=130.0, pnl=-0.5)

        report = ExecutionQualityAnalyzer(db).analyze()

        assert "0-50bps"    in report.by_overround_bucket
        assert "50-100bps"  in report.by_overround_bucket
        assert "100-150bps" in report.by_overround_bucket

    def test_groups_by_delta_bucket(self, db):
        _seed_resolved_trade(db, ticker="KXBTC-15MIN-B95000", delta=0.10, pnl=1.0)
        _seed_resolved_trade(db, ticker="KXETH-15MIN-B3200",  delta=0.20, pnl=1.5)
        _seed_resolved_trade(db, ticker="KXSOL-15MIN-B200",   delta=0.35, pnl=2.0)

        report = ExecutionQualityAnalyzer(db).analyze()

        assert "0.00-0.15" in report.by_delta_bucket
        assert "0.15-0.25" in report.by_delta_bucket
        assert "0.25-0.40" in report.by_delta_bucket

    def test_suggests_overround_threshold_when_bucket_losing(self, db):
        # bucket 100-150bps con 5 muestras perdedoras → debe sugerir 100
        for _ in range(5):
            _seed_resolved_trade(
                db, ticker="KXBTC-15MIN-B95000",
                outcome=Outcome.LOSS,
                overround_bps=120.0,
                pnl=-0.5,
            )
        # bucket 50-100bps con 5 muestras ganadoras
        for _ in range(5):
            _seed_resolved_trade(
                db, ticker="KXETH-15MIN-B3200",
                outcome=Outcome.WIN,
                overround_bps=75.0,
                pnl=1.0,
            )

        report = ExecutionQualityAnalyzer(db).analyze()
        assert report.suggested_max_overround_bps == pytest.approx(100.0)

    def test_no_suggestion_when_all_profitable(self, db):
        for _ in range(4):
            _seed_resolved_trade(
                db, ticker="KXBTC-15MIN-B95000",
                outcome=Outcome.WIN,
                overround_bps=120.0,
                pnl=1.0,
            )

        report = ExecutionQualityAnalyzer(db).analyze()
        assert report.suggested_max_overround_bps is None

    def test_unresolved_signals_excluded(self, db):
        """Señales sin outcome no deben aparecer en el análisis."""
        from core.models import Signal

        # Señal sin outcome (no resuelta)
        sig = Signal(
            market_ticker="KXBTC-15MIN-B95000",
            decision=Decision.YES,
            my_probability=0.70,
            market_probability=0.55,
            delta=0.15,
            ev_net_fees=0.08,
            kelly_size=0.10,
            confidence=Confidence.HIGH,
            time_remaining_s=300,
            reasoning="unresolved",
            timestamp=time.time(),
        )
        db.save_signal(sig)

        report = ExecutionQualityAnalyzer(db).analyze()
        assert report.total_resolved == 0

    def test_skip_signals_excluded(self, db):
        """Señales SKIP no deben aparecer en el análisis."""
        from core.models import Signal

        sig = Signal(
            market_ticker="KXBTC-15MIN-B95000",
            decision=Decision.SKIP,
            my_probability=0.0,
            market_probability=0.55,
            delta=0.0,
            ev_net_fees=0.0,
            kelly_size=0.0,
            confidence=Confidence.LOW,
            time_remaining_s=300,
            reasoning="skip test",
            timestamp=time.time(),
        )
        db.save_signal(sig)

        report = ExecutionQualityAnalyzer(db).analyze()
        assert report.total_resolved == 0

    def test_limit_respected(self, db):
        for i in range(10):
            _seed_resolved_trade(db, ticker=f"KXBTC-15MIN-B9{i}000", pnl=1.0)

        report = ExecutionQualityAnalyzer(db).analyze(limit=3)
        assert report.total_resolved == 3


# ─── Database.fetch_resolved_signals_with_trades ─────────────────────────────

class TestFetchResolvedSignalsWithTrades:
    def test_empty_db_returns_empty_list(self, db):
        result = db.fetch_resolved_signals_with_trades()
        assert result == []

    def test_returns_dicts_with_expected_keys(self, db):
        _seed_resolved_trade(db)
        rows = db.fetch_resolved_signals_with_trades()
        assert len(rows) == 1
        row = rows[0]
        assert "ticker" in row
        assert "delta" in row
        assert "outcome" in row
        assert "pnl" in row
        assert "contract_price" in row
        assert "market_overround_bps" in row
        assert "my_prob" in row

    def test_open_trades_excluded(self, db):
        """Solo trades CERRADOS deben aparecer."""
        from core.models import Signal, Trade

        sig = Signal(
            market_ticker="KXBTC-15MIN-B95000",
            decision=Decision.YES,
            my_probability=0.70,
            market_probability=0.55,
            delta=0.15,
            ev_net_fees=0.08,
            kelly_size=0.10,
            confidence=Confidence.HIGH,
            time_remaining_s=300,
            reasoning="open_trade_test",
            timestamp=time.time(),
            outcome=Outcome.WIN,
        )
        signal_id = db.save_signal(sig)
        trade = Trade(
            ticker="KXBTC-15MIN-B95000",
            side="YES",
            contracts=5,
            entry_price=0.55,
            mode=TradeMode.DEMO,
            status=TradeStatus.OPEN,  # ← OPEN, no debe aparecer
            opened_at=time.time(),
            signal_id=signal_id,
        )
        db.save_trade(trade)

        rows = db.fetch_resolved_signals_with_trades()
        assert rows == []


# ─── Dashboard endpoint ───────────────────────────────────────────────────────

class TestExecutionQualityEndpoint:
    """Tests del endpoint /analytics/execution-quality. Requieren httpx."""

    def _get_client(self, db):
        """Devuelve TestClient o salta si httpx no está instalado."""
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        from dashboard.api_server import create_app
        app = create_app(db=db)
        return TestClient(app)

    def test_endpoint_returns_200_empty_db(self, db):
        client = self._get_client(db)
        with client as c:
            response = c.get("/analytics/execution-quality")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total_resolved"] == 0
        assert payload["overall_win_rate"] == 0.0
        assert payload["overall_pnl"] == 0.0
        assert payload["suggested_max_overround_bps"] is None
        assert payload["by_category"] == {}
        assert payload["by_overround_bucket"] == {}
        assert payload["by_delta_bucket"] == {}

    def test_endpoint_returns_correct_structure_with_data(self, db):
        _seed_resolved_trade(db, ticker="KXBTC-15MIN-B95000", outcome=Outcome.WIN,
                              overround_bps=80.0, delta=0.20, pnl=2.0)
        _seed_resolved_trade(db, ticker="KXETH-15MIN-B3200", outcome=Outcome.LOSS,
                              overround_bps=120.0, delta=0.15, pnl=-0.5)

        with self._get_client(db) as client:
            response = client.get("/analytics/execution-quality")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total_resolved"] == 2
        assert "BTC" in payload["by_category"]
        assert "ETH" in payload["by_category"]

        btc = payload["by_category"]["BTC"]
        assert btc["sample_size"] == 1
        assert btc["win_rate"] == 1.0
        assert btc["total_pnl"] == pytest.approx(2.0)

        for key in ("label", "sample_size", "win_rate", "total_pnl", "avg_pnl",
                    "avg_contract_price", "avg_overround_bps", "avg_delta",
                    "avg_entry_edge_bps"):
            assert key in btc, f"Falta campo '{key}' en bucket BTC"

    def test_endpoint_limit_param(self, db):
        for i in range(5):
            _seed_resolved_trade(db, ticker=f"KXBTC-15MIN-B9{i}000", pnl=1.0)

        with self._get_client(db) as client:
            response = client.get("/analytics/execution-quality?limit=2")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total_resolved"] == 2

    def test_endpoint_suggests_overround_threshold(self, db):
        for _ in range(5):
            _seed_resolved_trade(db, ticker="KXBTC-15MIN-B95000",
                                  outcome=Outcome.LOSS, overround_bps=130.0, pnl=-1.0)
        for _ in range(5):
            _seed_resolved_trade(db, ticker="KXETH-15MIN-B3200",
                                  outcome=Outcome.WIN, overround_bps=70.0, pnl=1.5)

        with self._get_client(db) as client:
            response = client.get("/analytics/execution-quality")

        payload = response.json()
        assert payload["suggested_max_overround_bps"] == pytest.approx(100.0)
