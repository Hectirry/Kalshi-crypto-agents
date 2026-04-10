"""
tests/test_backtesting.py

Cobertura de Fase 4: backtesting, bloqueo de categorías y calibración.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

import backtesting.backtest_runner as backtest_runner_module
from backtesting.backtest_runner import BacktestRunner
from backtesting.category_blocker import CategoryBlocker
from backtesting.param_injector import ParamInjector
from core.models import Decision, Outcome


class TestBacktestRunner:
    async def test_load_signals_returns_ordered_history(self, db, make_signal):
        base_ts = time.time()
        later_signal = make_signal(
            market_ticker="KXBTC-15MIN-B96000",
            timestamp=base_ts + 20,
            outcome=Outcome.WIN,
        )
        earlier_signal = make_signal(
            market_ticker="KXBTC-15MIN-B95000",
            timestamp=base_ts + 10,
            outcome=Outcome.LOSS,
        )
        db.save_signal(later_signal)
        db.save_signal(earlier_signal)

        runner = BacktestRunner(db=db)
        signals = await runner.load_signals(
            from_ts=base_ts,
            to_ts=base_ts + 30,
            category="BTC",
        )

        assert [signal.market_ticker for signal in signals] == [
            "KXBTC-15MIN-B95000",
            "KXBTC-15MIN-B96000",
        ]

    def test_run_builds_equity_curve_and_pnl(self, db, make_signal):
        base_ts = time.time()
        win_signal = make_signal(
            market_ticker="KXBTC-15MIN-B95000",
            decision=Decision.YES,
            market_probability=0.55,
            kelly_size=0.10,
            timestamp=base_ts + 10,
            outcome=Outcome.WIN,
        )
        loss_signal = make_signal(
            market_ticker="KXBTC-15MIN-B95500",
            decision=Decision.YES,
            market_probability=0.60,
            kelly_size=0.10,
            timestamp=base_ts + 20,
            outcome=Outcome.LOSS,
        )
        skip_signal = make_signal(
            market_ticker="KXBTC-15MIN-B96000",
            decision=Decision.SKIP,
            kelly_size=0.0,
            timestamp=base_ts + 30,
        )
        db.save_signal(win_signal)
        db.save_signal(loss_signal)
        db.save_signal(skip_signal)

        result = BacktestRunner(db=db, initial_bankroll=1_000.0).run(
            from_ts=base_ts,
            to_ts=base_ts + 40,
            category="BTC",
        )

        assert result.total_signals == 3
        assert result.actionable_signals == 2
        assert result.skipped == 1
        assert result.wins == 1
        assert result.losses == 1
        assert len(result.equity_curve) == 2
        assert result.final_equity > 0.0
        assert result.max_drawdown >= 0.0

    def test_categories_in_range_infers_assets(self, db, make_signal):
        base_ts = time.time()
        db.save_signal(make_signal(market_ticker="KXBTC-15MIN-B95000", timestamp=base_ts + 1))
        db.save_signal(make_signal(market_ticker="KXETH-15MIN-B3200", timestamp=base_ts + 2))

        categories = BacktestRunner(db=db).categories_in_range(
            from_ts=base_ts,
            to_ts=base_ts + 10,
        )

        assert categories == {"BTC", "ETH"}

    def test_run_uses_vectorbt_when_available(self, db, make_signal):
        pytest.importorskip("pandas")
        pytest.importorskip("vectorbt")

        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                decision=Decision.YES,
                market_probability=0.55,
                kelly_size=0.10,
                timestamp=base_ts + 10,
                outcome=Outcome.WIN,
            )
        )

        result = BacktestRunner(db=db, initial_bankroll=1_000.0).run(
            from_ts=base_ts,
            to_ts=base_ts + 20,
            category="BTC",
        )

        assert result.vectorbt_available is True
        assert result.vectorbt_used is True
        assert len(result.equity_curve) == 1
        assert result.final_equity > 1_000.0

    def test_fallback_backtest_charges_entry_and_exit_fees(self, db, make_signal, monkeypatch):
        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                decision=Decision.YES,
                market_probability=0.50,
                kelly_size=0.10,
                timestamp=base_ts + 10,
                outcome=Outcome.WIN,
            )
        )

        monkeypatch.setattr(backtest_runner_module, "pd", None, raising=False)
        monkeypatch.setattr(backtest_runner_module, "vbt", None, raising=False)

        runner = BacktestRunner(db=db, initial_bankroll=1_000.0)
        runner.ev_calculator.fee_per_contract = MagicMock(return_value=0.01)

        result = runner.run(
            from_ts=base_ts,
            to_ts=base_ts + 20,
            category="BTC",
        )

        # stake=100, contracts=200, gross=100, fees=200*0.01*2 => pnl=96
        assert result.vectorbt_used is False
        assert result.total_pnl == pytest.approx(96.0)
        assert result.final_equity == pytest.approx(1_096.0)


class TestCategoryBlocker:
    def test_blocks_low_win_rate_category(self, db, make_signal):
        base_ts = time.time()
        for offset in range(4):
            db.save_signal(
                make_signal(
                    market_ticker=f"KXBTC-15MIN-B95{offset}",
                    decision=Decision.YES,
                    kelly_size=0.1,
                    timestamp=base_ts + offset,
                    outcome=Outcome.LOSS,
                )
            )

        runner = BacktestRunner(db=db)
        blocker = CategoryBlocker(db=db, runner=runner, min_samples=3, min_win_rate=0.50)
        decisions = blocker.evaluate_and_apply(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            categories={"BTC"},
        )

        assert decisions[0].blocked is True
        assert "BTC" in db.get_blocked_categories()

    def test_unblocks_recovered_category(self, db, make_signal):
        base_ts = time.time()
        db.block_category("ETH", win_rate=0.20, sample_size=10, reason="old_bad")
        for offset in range(4):
            db.save_signal(
                make_signal(
                    market_ticker=f"KXETH-15MIN-B3{offset}",
                    decision=Decision.YES,
                    kelly_size=0.1,
                    timestamp=base_ts + offset,
                    outcome=Outcome.WIN,
                )
            )

        runner = BacktestRunner(db=db)
        blocker = CategoryBlocker(db=db, runner=runner, min_samples=3, min_win_rate=0.50)
        decisions = blocker.evaluate_and_apply(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            categories={"ETH"},
        )

        assert decisions[0].blocked is False
        assert "ETH" not in db.get_blocked_categories()

    def test_insufficient_sample_does_not_block(self, db, make_signal):
        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                decision=Decision.YES,
                timestamp=base_ts + 1,
                outcome=Outcome.LOSS,
            )
        )

        runner = BacktestRunner(db=db)
        blocker = CategoryBlocker(db=db, runner=runner, min_samples=3, min_win_rate=0.50)
        decisions = blocker.evaluate_and_apply(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            categories={"BTC"},
        )

        assert decisions[0].reason == "insufficient_sample"
        assert "BTC" not in db.get_blocked_categories()


class TestParamInjector:
    def test_calibrates_and_persists_best_thresholds(self, db, make_signal):
        base_ts = time.time()
        signals = [
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                delta=0.03,
                ev_net_fees=0.01,
                timestamp=base_ts + 1,
                outcome=Outcome.WIN,
            ),
            make_signal(
                market_ticker="KXBTC-15MIN-B95100",
                delta=0.05,
                ev_net_fees=0.03,
                timestamp=base_ts + 2,
                outcome=Outcome.WIN,
            ),
            make_signal(
                market_ticker="KXBTC-15MIN-B95200",
                delta=0.08,
                ev_net_fees=0.06,
                timestamp=base_ts + 3,
                outcome=Outcome.WIN,
            ),
            make_signal(
                market_ticker="KXBTC-15MIN-B95300",
                delta=0.04,
                ev_net_fees=0.02,
                timestamp=base_ts + 4,
                outcome=Outcome.LOSS,
            ),
        ]
        for signal in signals:
            db.save_signal(signal)

        injector = ParamInjector(
            db=db,
            delta_candidates=(0.03, 0.05, 0.08),
            ev_candidates=(0.01, 0.03, 0.06),
        )
        results = injector.calibrate(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            categories={"BTC"},
        )

        assert len(results) == 2
        current = db.get_current_params(category="BTC")
        assert "min_delta" in current
        assert "min_ev_threshold" in current
        assert current["min_delta"] == 0.05
        assert current["min_ev_threshold"] == 0.03

    def test_skips_empty_categories(self, db):
        injector = ParamInjector(db=db)

        results = injector.calibrate(
            from_ts=time.time() - 100,
            to_ts=time.time(),
            categories={"BTC"},
        )

        assert results == []
