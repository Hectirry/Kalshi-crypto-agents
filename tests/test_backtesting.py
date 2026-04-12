"""
tests/test_backtesting.py

Cobertura de Fase 4: backtesting, bloqueo de categorías y calibración.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import backtesting.backtest_runner as backtest_runner_module
from backtesting.backtest_runner import BacktestRunner, ZoneMetrics
from backtesting.category_blocker import CategoryBlocker
from backtesting.outcome_resolver import OutcomeResolver
from backtesting.param_injector import ParamInjector
from core.models import Decision, Outcome, Trade, TradeMode, TradeStatus


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

    def test_run_skips_vectorbt_when_loss_exit_price_is_zero(self, db, make_signal):
        pytest.importorskip("pandas")
        pytest.importorskip("vectorbt")

        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXETH-15MIN-B3200",
                decision=Decision.YES,
                market_probability=0.55,
                kelly_size=0.10,
                timestamp=base_ts + 10,
                outcome=Outcome.LOSS,
            )
        )

        result = BacktestRunner(db=db, initial_bankroll=1_000.0).run(
            from_ts=base_ts,
            to_ts=base_ts + 20,
            category="ETH",
        )

        assert result.vectorbt_available is True
        assert result.vectorbt_used is False

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

        # kelly=0.10 => 10 contratos, gross=5, fees=10*0.01*2 => pnl=4.8
        assert result.vectorbt_used is False
        assert result.total_pnl == pytest.approx(4.8)
        assert result.final_equity == pytest.approx(1_004.8)

    def test_run_dedupes_multiple_actionable_signals_for_same_ticker(self, db, make_signal, monkeypatch):
        base_ts = time.time()
        for offset in (1, 2):
            db.save_signal(
                make_signal(
                    market_ticker="KXBTC-15MIN-B95000",
                    decision=Decision.YES,
                    market_probability=0.50,
                    kelly_size=0.10,
                    timestamp=base_ts + offset,
                    outcome=Outcome.WIN,
                )
            )

        monkeypatch.setattr(backtest_runner_module, "pd", None, raising=False)
        monkeypatch.setattr(backtest_runner_module, "vbt", None, raising=False)

        result = BacktestRunner(db=db, initial_bankroll=1_000.0).run(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            category="BTC",
        )

        assert result.actionable_signals == 1
        assert len(result.equity_curve) == 1

    def test_run_with_config_filters_out_btc_signals_that_fail_override_policy(
        self,
        app_config,
        db,
        make_signal,
        monkeypatch,
    ):
        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                decision=Decision.YES,
                market_probability=0.60,
                delta=0.20,
                ev_net_fees=0.25,
                time_remaining_s=120,
                timestamp=base_ts + 1,
                outcome=Outcome.WIN,
            )
        )

        monkeypatch.setattr(backtest_runner_module, "pd", None, raising=False)
        monkeypatch.setattr(backtest_runner_module, "vbt", None, raising=False)

        result = BacktestRunner(
            db=db,
            initial_bankroll=1_000.0,
            config=app_config.engine,
        ).run(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            category="BTC",
        )

        assert result.actionable_signals == 0
        assert result.total_pnl == pytest.approx(0.0)

    def test_run_with_setup_quality_gate_filters_historically_weak_setup(
        self,
        app_config,
        db,
        make_signal,
        monkeypatch,
    ):
        from dataclasses import replace

        cfg = replace(
            app_config.engine,
            setup_quality_gate_enabled=True,
            setup_quality_min_samples=3,
            setup_quality_min_win_rate=0.40,
        )
        base_ts = time.time()
        for offset in range(3):
            db.save_signal(
                make_signal(
                    market_ticker=f"KXETH-15MIN-B32{offset}",
                    decision=Decision.YES,
                    market_probability=0.50,
                    delta=0.15,
                    ev_net_fees=0.07,
                    time_remaining_s=420,
                    kelly_size=0.10,
                    timestamp=base_ts + offset,
                    outcome=Outcome.LOSS,
                )
            )
        db.save_signal(
            make_signal(
                market_ticker="KXETH-15MIN-B3300",
                decision=Decision.YES,
                market_probability=0.50,
                delta=0.15,
                ev_net_fees=0.07,
                time_remaining_s=420,
                kelly_size=0.10,
                timestamp=base_ts + 10,
                outcome=Outcome.WIN,
            )
        )

        monkeypatch.setattr(backtest_runner_module, "pd", None, raising=False)
        monkeypatch.setattr(backtest_runner_module, "vbt", None, raising=False)

        result = BacktestRunner(
            db=db,
            initial_bankroll=1_000.0,
            config=cfg,
        ).run(
            from_ts=base_ts - 1,
            to_ts=base_ts + 20,
            category="ETH",
        )

        assert result.actionable_signals == 3
        assert result.losses == 3

    def test_backtest_uses_persisted_contract_price_and_overround_filter(
        self,
        app_config,
        db,
        make_signal,
        monkeypatch,
    ):
        from dataclasses import replace

        cfg = replace(app_config.engine, max_market_overround_bps=100.0)
        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXETH-15MIN-B3200",
                decision=Decision.YES,
                market_probability=0.50,
                contract_price=0.65,
                market_overround_bps=80.0,
                kelly_size=0.10,
                timestamp=base_ts + 1,
                outcome=Outcome.WIN,
            )
        )
        db.save_signal(
            make_signal(
                market_ticker="KXETH-15MIN-B3300",
                decision=Decision.YES,
                market_probability=0.50,
                contract_price=0.55,
                market_overround_bps=250.0,
                kelly_size=0.10,
                timestamp=base_ts + 2,
                outcome=Outcome.WIN,
            )
        )

        monkeypatch.setattr(backtest_runner_module, "pd", None, raising=False)
        monkeypatch.setattr(backtest_runner_module, "vbt", None, raising=False)

        result = BacktestRunner(
            db=db,
            initial_bankroll=1_000.0,
            config=cfg,
        ).run(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            category="ETH",
        )

        assert result.actionable_signals == 1
        assert result.total_pnl == pytest.approx(3.34075)


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

    def test_positive_pnl_does_not_block_even_with_low_win_rate(self, db, make_signal):
        base_ts = time.time()
        prices_and_outcomes = [
            (0.05, Outcome.WIN),
            (0.05, Outcome.WIN),
            (0.80, Outcome.LOSS),
            (0.80, Outcome.LOSS),
        ]
        for offset, (market_probability, outcome) in enumerate(prices_and_outcomes, start=1):
            db.save_signal(
                make_signal(
                    market_ticker=f"KXETH-15MIN-B95{offset}",
                    decision=Decision.YES,
                    market_probability=market_probability,
                    kelly_size=0.1,
                    timestamp=base_ts + offset,
                    outcome=outcome,
                )
            )

        runner = BacktestRunner(db=db)
        blocker = CategoryBlocker(db=db, runner=runner, min_samples=3, min_win_rate=0.75)
        decisions = blocker.evaluate_and_apply(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            categories={"ETH"},
        )

        assert decisions[0].blocked is False
        assert decisions[0].win_rate == pytest.approx(0.5)
        assert decisions[0].total_pnl > 0.0
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

    def test_no_decision_with_outcome_win_is_treated_as_win(self, db, make_signal):
        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXETH-15MIN-B3000",
                decision=Decision.NO,
                market_probability=0.60,
                kelly_size=0.10,
                timestamp=base_ts,
                outcome=Outcome.WIN,
                outcome_at=base_ts + 900,
            )
        )

        runner = BacktestRunner(db=db)
        result = runner.run(from_ts=base_ts - 1, to_ts=base_ts + 10, category="ETH")

        assert result.actionable_signals == 1
        assert result.wins == 1
        assert result.losses == 0
        assert result.win_rate == pytest.approx(1.0)

    def test_param_injector_counts_no_win_as_win(self, db, make_signal):
        base_ts = time.time()
        signal = make_signal(
            market_ticker="KXBTC-15MIN-B95000",
            decision=Decision.NO,
            market_probability=0.60,
            timestamp=base_ts,
            outcome=Outcome.WIN,
            outcome_at=base_ts + 900,
        )
        db.save_signal(signal)

        injector = ParamInjector(db=db)

        assert injector._signal_won(signal) is True


    # ── Tests de breakdowns por zona y hora ───────────────────────────────────

    def test_results_by_zone_populated(self, db, make_signal):
        """
        BacktestResult debe tener results_by_zone con métricas por zona NEAR/MID/FAR.
        """
        base_ts = time.time()
        # NEAR (time=200), MID (time=500), FAR (time=700)
        for i, (ticker, time_s, outcome) in enumerate([
            ("KXBTC-15MIN-ZN0", 200, Outcome.WIN),
            ("KXBTC-15MIN-ZM0", 500, Outcome.WIN),
            ("KXBTC-15MIN-ZF0", 700, Outcome.LOSS),
        ]):
            db.save_signal(make_signal(
                market_ticker=ticker,
                time_remaining_s=time_s,
                timestamp=base_ts + i + 1,
                outcome=outcome,
            ))

        runner = BacktestRunner(db=db)
        result = runner.run(from_ts=base_ts, to_ts=base_ts + 10)

        assert isinstance(result.results_by_zone, dict)
        assert "NEAR" in result.results_by_zone
        assert "MID" in result.results_by_zone
        assert "FAR" in result.results_by_zone

        near = result.results_by_zone["NEAR"]
        assert isinstance(near, ZoneMetrics)
        assert near.wins == 1
        assert near.losses == 0
        assert near.win_rate == pytest.approx(1.0)

        far = result.results_by_zone["FAR"]
        assert far.wins == 0
        assert far.losses == 1
        assert far.win_rate == pytest.approx(0.0)

    def test_results_by_hour_populated(self, db, make_signal):
        """
        BacktestResult debe tener results_by_hour con métricas por hora UTC.
        """
        import datetime as dt

        base_ts = time.time()
        # Dos señales en la misma hora UTC, una en otra hora
        ts_h0 = dt.datetime(2025, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        ts_h1 = dt.datetime(2025, 1, 1, 10, 30, 0, tzinfo=dt.timezone.utc).timestamp()
        ts_h2 = dt.datetime(2025, 1, 1, 14, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        for i, (ticker, ts, outcome) in enumerate([
            ("KXETH-15MIN-H0A", ts_h0, Outcome.WIN),
            ("KXETH-15MIN-H0B", ts_h1, Outcome.LOSS),
            ("KXETH-15MIN-H2A", ts_h2, Outcome.WIN),
        ]):
            db.save_signal(make_signal(
                market_ticker=ticker,
                timestamp=ts,
                outcome=outcome,
            ))

        runner = BacktestRunner(db=db)
        result = runner.run(from_ts=ts_h0 - 1, to_ts=ts_h2 + 1)

        assert isinstance(result.results_by_hour, dict)
        assert 10 in result.results_by_hour
        assert 14 in result.results_by_hour

        hour10 = result.results_by_hour[10]
        assert hour10.wins == 1
        assert hour10.losses == 1
        assert hour10.win_rate == pytest.approx(0.5)

        hour14 = result.results_by_hour[14]
        assert hour14.wins == 1
        assert hour14.losses == 0

    def test_empty_backtest_has_empty_breakdowns(self, db):
        """
        Backtest sin señales actionables devuelve dicts vacíos para breakdowns.
        """
        runner = BacktestRunner(db=db)
        result = runner.run(from_ts=0.0, to_ts=1.0)

        assert result.results_by_zone == {}
        assert result.results_by_hour == {}


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

        assert len(results) >= 2
        current = db.get_current_params(category="BTC")
        assert "min_delta" in current
        assert "min_ev_threshold" in current
        assert current["min_delta"] == 0.05
        assert current["min_ev_threshold"] == 0.03

    def test_calibration_dedupes_repeated_ticker(self, db, make_signal):
        base_ts = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                delta=0.12,
                ev_net_fees=0.06,
                timestamp=base_ts + 1,
                outcome=Outcome.LOSS,
            )
        )
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                delta=0.15,
                ev_net_fees=0.10,
                timestamp=base_ts + 2,
                outcome=Outcome.WIN,
            )
        )

        injector = ParamInjector(
            db=db,
            delta_candidates=(0.10, 0.12, 0.15),
            ev_candidates=(0.04, 0.06, 0.10),
        )
        results = injector.calibrate(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            categories={"BTC"},
        )

        assert len(results) >= 2
        current = db.get_current_params(category="BTC")
        assert current["min_delta"] == 0.12
        assert current["min_ev_threshold"] == 0.06

    def test_skips_empty_categories(self, db):
        injector = ParamInjector(db=db)

        results = injector.calibrate(
            from_ts=time.time() - 100,
            to_ts=time.time(),
            categories={"BTC"},
        )

        assert results == []

    # ── _effective_contract_price correctness ──────────────────────────────────

    def test_effective_contract_price_uses_stored_price_for_yes(self, make_signal):
        """Cuando contract_price está guardado, debe usarse en lugar del mid."""
        injector = ParamInjector(db=None)  # type: ignore[arg-type]
        signal = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=0.62,  # spread de 7 puntos sobre el mid
        )
        assert injector._effective_contract_price(signal) == pytest.approx(0.62)

    def test_effective_contract_price_uses_stored_price_for_no(self, make_signal):
        """NO side: usa contract_price (ya normalizado para NO) si está disponible."""
        injector = ParamInjector(db=None)  # type: ignore[arg-type]
        signal = make_signal(
            decision=Decision.NO,
            market_probability=0.60,
            contract_price=0.43,  # no_ask guardado, no el complemento
        )
        assert injector._effective_contract_price(signal) == pytest.approx(0.43)

    def test_effective_contract_price_fallback_yes_when_none(self, make_signal):
        """Fallback a market_probability para YES cuando contract_price es None."""
        injector = ParamInjector(db=None)  # type: ignore[arg-type]
        signal = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=None,
        )
        assert injector._effective_contract_price(signal) == pytest.approx(0.55)

    def test_effective_contract_price_fallback_no_when_none(self, make_signal):
        """Fallback a 1 - market_probability para NO cuando contract_price es None."""
        injector = ParamInjector(db=None)  # type: ignore[arg-type]
        signal = make_signal(
            decision=Decision.NO,
            market_probability=0.60,
            contract_price=None,
        )
        # para NO side: precio efectivo = 1 - market_prob = 0.40
        assert injector._effective_contract_price(signal) == pytest.approx(0.40)

    # ── _signal_realized_pnl afectado por el spread ────────────────────────────

    def test_realized_pnl_lower_when_entry_price_has_spread(self, make_signal):
        """
        Señal YES ganadora: PnL debe ser menor cuando contract_price > market_probability
        (hay bid-ask spread), ya que entramos más caro que el mid.
        """
        injector = ParamInjector(db=None)  # type: ignore[arg-type]
        # Sin spread: entry al mid
        signal_mid = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=None,   # fallback a market_probability = 0.55
            outcome=Outcome.WIN,
        )
        # Con spread: entry al ask, 7 puntos más caro
        signal_ask = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=0.62,
            outcome=Outcome.WIN,
        )
        pnl_mid = injector._signal_realized_pnl(signal_mid)
        pnl_ask = injector._signal_realized_pnl(signal_ask)
        # Pagar más en la entrada siempre reduce la ganancia neta
        assert pnl_ask < pnl_mid

    def test_realized_pnl_same_when_no_spread_stored(self, make_signal):
        """
        Con contract_price=None el PnL es idéntico al comportamiento anterior
        (fallback a market_probability). Compatibilidad con señales pre-v2.
        """
        injector = ParamInjector(db=None)  # type: ignore[arg-type]
        signal = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=None,
            outcome=Outcome.WIN,
        )
        pnl = injector._signal_realized_pnl(signal)
        # gross = 1.0 - 0.55 = 0.45, entry_fee = fee(0.55), exit_fee = fee(1.0)=0
        from engine.ev_calculator import EVCalculator
        expected = (1.0 - 0.55) - EVCalculator().fee_per_contract(0.55)
        assert pnl == pytest.approx(expected)

    def test_realized_pnl_loss_reflects_actual_entry_cost(self, make_signal):
        """LOSS: pérdida debe ser la totalidad del precio de entrada real."""
        injector = ParamInjector(db=None)  # type: ignore[arg-type]
        signal = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=0.60,
            outcome=Outcome.LOSS,
        )
        pnl = injector._signal_realized_pnl(signal)
        # gross = 0.0 - 0.60 = -0.60, entry_fee = fee(0.60), exit_fee = fee(0.0)=0
        from engine.ev_calculator import EVCalculator
        expected = -0.60 - EVCalculator().fee_per_contract(0.60)
        assert pnl == pytest.approx(expected)

    # ── Calibración end-to-end con spreads ────────────────────────────────────

    # ── min_calibration_samples ────────────────────────────────────────────────

    def test_min_samples_rejects_candidates_with_too_few_signals(self, db, make_signal):
        """
        Con min_calibration_samples=3, candidatos con < 3 señales se descartan
        aunque tengan PnL alto. El threshold con muestra suficiente gana aunque
        su PnL promedio sea menor.
        """
        base_ts = time.time()
        # 5 señales con delta ≥ 0.10 → muestra suficiente para cualquier candidato
        for i in range(5):
            db.save_signal(make_signal(
                market_ticker=f"KXETH-15MIN-B32{i}",
                decision=Decision.YES,
                market_probability=0.55,
                delta=0.12,          # pasa threshold 0.10 y 0.12, NO pasa 0.15
                ev_net_fees=0.07,
                timestamp=base_ts + i,
                outcome=Outcome.WIN if i < 3 else Outcome.LOSS,  # win_rate=60%
            ))
        # 1 señal con delta=0.15 → n=1, descartado si min_samples=3
        db.save_signal(make_signal(
            market_ticker="KXETH-15MIN-B3299",
            decision=Decision.YES,
            market_probability=0.55,
            delta=0.15,
            ev_net_fees=0.09,
            timestamp=base_ts + 10,
            outcome=Outcome.WIN,  # 100% win rate pero muestra de 1
        ))

        injector = ParamInjector(
            db=db,
            delta_candidates=(0.10, 0.12, 0.15),
            ev_candidates=(0.04, 0.06),
            min_calibration_samples=3,
        )
        results = injector.calibrate(from_ts=base_ts, to_ts=base_ts + 20, categories={"ETH"})

        assert len(results) >= 2
        current = db.get_current_params(category="ETH")
        # threshold 0.15 tiene n=1 < min_samples=3 → se descarta
        # threshold 0.12 tiene n=5 (todos pasan ≥ 0.10 y ≥ 0.12)
        # threshold 0.10 tiene n=6 (todos los de 0.12 + el de 0.15)
        # entre 0.10 y 0.12, el que tiene mejor avg_pnl gana
        assert current["min_delta"] in (0.10, 0.12)  # nunca 0.15 porque n=1 < 3

    def test_min_samples_default_one_preserves_existing_behavior(self, db, make_signal):
        """Con min_calibration_samples=1 (default), el comportamiento es idéntico al anterior."""
        base_ts = time.time()
        # 1 sola señal con delta=0.15
        db.save_signal(make_signal(
            market_ticker="KXBTC-15MIN-B95000",
            delta=0.15,
            ev_net_fees=0.09,
            timestamp=base_ts + 1,
            outcome=Outcome.WIN,
        ))
        injector = ParamInjector(
            db=db,
            delta_candidates=(0.10, 0.12, 0.15),
            ev_candidates=(0.04, 0.06),
            min_calibration_samples=1,  # default: acepta muestras de 1
        )
        results = injector.calibrate(from_ts=base_ts, to_ts=base_ts + 10, categories={"BTC"})
        assert len(results) >= 2
        current = db.get_current_params(category="BTC")
        # Con n=1 y min_samples=1, threshold 0.15 es válido
        assert current["min_delta"] == 0.15

    def test_min_samples_falls_back_to_first_candidate_when_all_below_threshold(
        self, db, make_signal
    ):
        """Si todos los candidatos quedan bajo min_samples, devuelve el primero (más bajo)."""
        base_ts = time.time()
        # Solo 2 señales → con min_samples=3, todos los candidatos tienen n < 3
        db.save_signal(make_signal(
            market_ticker="KXBTC-15MIN-B95000",
            delta=0.15,
            ev_net_fees=0.09,
            timestamp=base_ts + 1,
            outcome=Outcome.WIN,
        ))
        db.save_signal(make_signal(
            market_ticker="KXBTC-15MIN-B95100",
            delta=0.12,
            ev_net_fees=0.07,
            timestamp=base_ts + 2,
            outcome=Outcome.LOSS,
        ))
        injector = ParamInjector(
            db=db,
            delta_candidates=(0.10, 0.12, 0.15),
            ev_candidates=(0.04, 0.06),
            min_calibration_samples=3,
        )
        results = injector.calibrate(from_ts=base_ts, to_ts=base_ts + 10, categories={"BTC"})
        # Todos los candidatos caen bajo min_samples → fallback al primero
        current = db.get_current_params(category="BTC")
        assert current["min_delta"] == 0.10  # candidates[0]

    def test_calibration_with_contract_price_accounts_for_spread(self, db, make_signal):
        """
        Cuando los signals tienen contract_price guardado (spread real),
        la calibración usa esos precios y no el mid, produciendo estimaciones
        de PnL más conservadoras y thresholds que reflejan el coste real.
        """
        base_ts = time.time()
        # Señal con spread grande: market_prob=0.55 pero pagamos 0.70 (ask)
        # → gross WIN = 1.0 - 0.70 = 0.30 (vs 0.45 sin spread)
        db.save_signal(make_signal(
            market_ticker="KXBTC-15MIN-B95000",
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=0.70,
            delta=0.15,
            ev_net_fees=0.08,
            timestamp=base_ts + 1,
            outcome=Outcome.WIN,
        ))
        db.save_signal(make_signal(
            market_ticker="KXBTC-15MIN-B95100",
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=0.70,
            delta=0.12,
            ev_net_fees=0.06,
            timestamp=base_ts + 2,
            outcome=Outcome.WIN,
        ))
        db.save_signal(make_signal(
            market_ticker="KXBTC-15MIN-B95200",
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=0.70,
            delta=0.20,
            ev_net_fees=0.12,
            timestamp=base_ts + 3,
            outcome=Outcome.LOSS,
        ))

        injector = ParamInjector(
            db=db,
            delta_candidates=(0.10, 0.12, 0.15),
            ev_candidates=(0.04, 0.06, 0.08),
        )
        # _signal_realized_pnl debe usar contract_price=0.70, no market_prob=0.55
        # Se puede verificar que el injector realmente llama _effective_contract_price
        signal = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=0.70,
            outcome=Outcome.WIN,
        )
        pnl_with_price = injector._signal_realized_pnl(signal)

        signal_no_price = make_signal(
            decision=Decision.YES,
            market_probability=0.55,
            contract_price=None,
            outcome=Outcome.WIN,
        )
        pnl_without_price = injector._signal_realized_pnl(signal_no_price)

        # Con precio real de 0.70 en lugar de 0.55, la ganancia neta debe ser menor
        assert pnl_with_price < pnl_without_price
        # La calibración debe completarse sin errores
        results = injector.calibrate(from_ts=base_ts, to_ts=base_ts + 10, categories={"BTC"})
        assert len(results) >= 2

    # ── Tests de calibración de min_time_remaining_s ──────────────────────────

    def test_calibrate_includes_min_time_remaining_s(self, db, make_signal):
        """
        calibrate() debe devolver calibraciones que incluyen ``min_time_remaining_s``
        además de ``min_delta`` y ``min_ev_threshold``.
        """
        base_ts = time.time()
        # Cuatro señales en zona MID (time_remaining_s=500 → 301-600 s)
        for i, (delta, ev, outcome) in enumerate([
            (0.12, 0.06, Outcome.WIN),
            (0.10, 0.05, Outcome.WIN),
            (0.08, 0.04, Outcome.WIN),
            (0.06, 0.03, Outcome.LOSS),
        ]):
            db.save_signal(make_signal(
                market_ticker=f"KXETH-15MIN-B{3000 + i}",
                delta=delta,
                ev_net_fees=ev,
                time_remaining_s=500,
                timestamp=base_ts + i + 1,
                outcome=outcome,
            ))

        injector = ParamInjector(
            db=db,
            delta_candidates=(0.06, 0.08, 0.10, 0.12),
            ev_candidates=(0.03, 0.04, 0.05, 0.06),
            time_candidates=(90, 150, 180, 240, 300),
        )
        results = injector.calibrate(
            from_ts=base_ts,
            to_ts=base_ts + 10,
            categories={"ETH"},
        )

        param_keys = {r.param_key for r in results}
        assert "min_delta" in param_keys
        assert "min_ev_threshold" in param_keys
        assert "min_time_remaining_s" in param_keys

        # Debe también estar persistido en DB
        current = db.get_current_params(category="ETH")
        assert "min_time_remaining_s" in current

    def test_calibrate_zone_far_produces_higher_time_threshold_than_near(self, db, make_signal):
        """
        Señales en zona FAR con peor win rate deben producir un
        ``min_time_remaining_s_FAR`` más alto que el de zona NEAR con buen win rate.

        Setup:
          - NEAR (time=200 s): 4 señales ganadoras → la calibración prefiere el
            umbral de tiempo más bajo posible (90 s), porque hay buen PnL a
            cualquier umbral.
          - FAR (time=700 s): 3 señales perdedoras + 1 ganadora con high time →
            la calibración selecciona un umbral alto (300 s) para filtrar las
            pérdidas de tiempo bajo.
        """
        base_ts = time.time()

        # Zona NEAR: time_remaining_s=200 → classify_time_zone(200) == "NEAR"
        for i in range(4):
            db.save_signal(make_signal(
                market_ticker=f"KXBTC-15MIN-NEAR{i}",
                delta=0.15,
                ev_net_fees=0.08,
                time_remaining_s=200,
                timestamp=base_ts + i + 1,
                outcome=Outcome.WIN,
            ))

        # Zona FAR: time_remaining_s varia
        # Señal perdedora a tiempo bajo (610 s) — FAR pero el umbral 90/150/180/240 las incluye
        for i in range(3):
            db.save_signal(make_signal(
                market_ticker=f"KXBTC-15MIN-FAR-LOW{i}",
                delta=0.10,
                ev_net_fees=0.05,
                time_remaining_s=620,
                timestamp=base_ts + 10 + i,
                outcome=Outcome.LOSS,
            ))
        # Señal ganadora solo a tiempo alto (720 s)
        db.save_signal(make_signal(
            market_ticker="KXBTC-15MIN-FAR-HIGH0",
            delta=0.20,
            ev_net_fees=0.12,
            time_remaining_s=720,
            timestamp=base_ts + 14,
            outcome=Outcome.WIN,
        ))

        injector = ParamInjector(
            db=db,
            delta_candidates=(0.05, 0.10, 0.15),
            ev_candidates=(0.03, 0.06, 0.10),
            time_candidates=(90, 150, 240, 300),
            min_calibration_samples=1,
        )
        results = injector.calibrate(
            from_ts=base_ts,
            to_ts=base_ts + 20,
            categories={"BTC"},
        )

        near_results = [r for r in results if r.param_key == "min_time_remaining_s_NEAR"]
        far_results = [r for r in results if r.param_key == "min_time_remaining_s_FAR"]

        # Debe haber calibraciones para ambas zonas
        assert near_results, "Esperaba calibración para zona NEAR"
        assert far_results, "Esperaba calibración para zona FAR"

        near_threshold = near_results[0].param_value
        far_threshold = far_results[0].param_value

        # FAR con señales perdedoras a tiempo bajo debe calibrar threshold más alto
        assert far_threshold > near_threshold, (
            f"Se esperaba far_threshold({far_threshold}) > near_threshold({near_threshold})"
        )


# ── Fix 3: OutcomeResolver ────────────────────────────────────────────────────

class TestOutcomeResolver:
    """Verifica la resolución de outcomes de señales expiradas."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_kalshi_mock(
        self,
        result: str | None = "yes",
        status: str = "finalized",
    ) -> MagicMock:
        """
        Crea un mock de KalshiFeed con get_market() configurado.
        result=None simula 404 (ticker no encontrado).
        """
        mock = MagicMock()
        if result is None:
            mock.get_market = AsyncMock(return_value=None)
        else:
            mock.get_market = AsyncMock(
                return_value={"result": result, "status": status}
            )
        return mock

    def _save_expired_signal(
        self,
        db,
        make_signal,
        decision: Decision = Decision.YES,
        age_s: float = 1200.0,
    ) -> int:
        """Guarda en DB una señal lo suficientemente antigua para resolver."""
        signal = make_signal(decision=decision, timestamp=time.time() - age_s)
        return db.save_signal(signal)

    def _save_open_trade(
        self,
        db,
        signal_id: int,
        ticker: str = "KXBTC-15MIN-B95000",
        entry_price: float = 0.55,
        contracts: int = 10,
        fee_paid: float = 0.48,
    ) -> int:
        """Guarda en DB un trade abierto vinculado a signal_id."""
        trade = Trade(
            ticker=ticker,
            side="YES",
            contracts=contracts,
            entry_price=entry_price,
            mode=TradeMode.DEMO,
            status=TradeStatus.OPEN,
            opened_at=time.time() - 1200.0,
            signal_id=signal_id,
            fee_paid=fee_paid,
        )
        return db.save_trade(trade)

    # ── Resolución de outcomes ─────────────────────────────────────────────────

    async def test_resolves_yes_win(self, db, make_signal):
        """Signal YES + market result='yes' → outcome=WIN, trade cerrado con exit=1.0."""
        signal_id = self._save_expired_signal(db, make_signal, decision=Decision.YES)
        self._save_open_trade(db, signal_id)
        resolver = OutcomeResolver(db=db, kalshi_client=self._make_kalshi_mock(result="yes"))

        n = await resolver.resolve_expired()

        assert n == 1
        assert db.get_open_trades() == []  # trade cerrado
        signals = db.get_signals(from_ts=0, to_ts=time.time() + 1)
        resolved = [s for s in signals if s.outcome is not None]
        assert len(resolved) == 1
        assert resolved[0].outcome == Outcome.WIN

    async def test_resolves_yes_loss(self, db, make_signal):
        """Signal YES + market result='no' → outcome=LOSS, trade cerrado con exit=0.0."""
        signal_id = self._save_expired_signal(db, make_signal, decision=Decision.YES)
        self._save_open_trade(db, signal_id)
        resolver = OutcomeResolver(db=db, kalshi_client=self._make_kalshi_mock(result="no"))

        n = await resolver.resolve_expired()

        assert n == 1
        assert db.get_open_trades() == []
        signals = db.get_signals(from_ts=0, to_ts=time.time() + 1)
        resolved = [s for s in signals if s.outcome is not None]
        assert resolved[0].outcome == Outcome.LOSS

    async def test_resolves_no_win(self, db, make_signal):
        """Signal NO + market result='no' → outcome=WIN."""
        signal_id = self._save_expired_signal(db, make_signal, decision=Decision.NO)
        resolver = OutcomeResolver(db=db, kalshi_client=self._make_kalshi_mock(result="no"))

        n = await resolver.resolve_expired()

        assert n == 1
        signals = db.get_signals(from_ts=0, to_ts=time.time() + 1)
        resolved = [s for s in signals if s.outcome is not None]
        assert resolved[0].outcome == Outcome.WIN

    async def test_skips_not_finalized(self, db, make_signal):
        """Si el market no está finalized → outcome queda NULL, retorna 0."""
        self._save_expired_signal(db, make_signal)
        resolver = OutcomeResolver(
            db=db,
            kalshi_client=self._make_kalshi_mock(result="yes", status="open"),
        )

        n = await resolver.resolve_expired()

        assert n == 0
        signals = db.get_signals(from_ts=0, to_ts=time.time() + 1)
        assert all(s.outcome is None for s in signals)

    async def test_skips_future_signals(self, db, make_signal):
        """Señales creadas hace 5 min (< 16 min) no se tocan, retorna 0."""
        # Solo 5 min de antigüedad → no supera el cutoff de 16 min
        signal = make_signal(decision=Decision.YES, timestamp=time.time() - 300)
        db.save_signal(signal)
        resolver = OutcomeResolver(
            db=db, kalshi_client=self._make_kalshi_mock(result="yes")
        )

        n = await resolver.resolve_expired()

        assert n == 0
        signals = db.get_signals(from_ts=0, to_ts=time.time() + 1)
        assert all(s.outcome is None for s in signals)

    async def test_handles_404_gracefully(self, db, make_signal):
        """Kalshi retorna 404 (None) → no lanza excepción, retorna 0."""
        self._save_expired_signal(db, make_signal)
        resolver = OutcomeResolver(
            db=db, kalshi_client=self._make_kalshi_mock(result=None)
        )

        n = await resolver.resolve_expired()

        assert n == 0  # no excepción, no outcome escrito

    # ── Cálculo de P&L ────────────────────────────────────────────────────────

    async def test_pnl_calculation_win(self, db, make_signal):
        """WIN: pnl = (1.0 - entry) * contracts - fee_paid = (1.0-0.55)*10-0.48 = $4.02."""
        signal_id = self._save_expired_signal(db, make_signal, decision=Decision.YES)
        trade_id = self._save_open_trade(
            db, signal_id, entry_price=0.55, contracts=10, fee_paid=0.48
        )
        resolver = OutcomeResolver(
            db=db, kalshi_client=self._make_kalshi_mock(result="yes")
        )

        await resolver.resolve_expired()

        closed = db.get_signals(from_ts=0, to_ts=time.time() + 1)
        # Verificar el trade cerrado directamente en DB
        import sqlite3 as _sqlite3
        row = db._conn.execute(
            "SELECT pnl, exit_price FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        assert row is not None
        assert abs(row["exit_price"] - 1.0) < 1e-9
        assert abs(row["pnl"] - 4.02) < 1e-9

    async def test_pnl_calculation_loss(self, db, make_signal):
        """LOSS: pnl = (0.0 - entry) * contracts - fee_paid = (0.0-0.55)*10-0.48 = -$5.98."""
        signal_id = self._save_expired_signal(db, make_signal, decision=Decision.YES)
        trade_id = self._save_open_trade(
            db, signal_id, entry_price=0.55, contracts=10, fee_paid=0.48
        )
        resolver = OutcomeResolver(
            db=db, kalshi_client=self._make_kalshi_mock(result="no")
        )

        await resolver.resolve_expired()

        row = db._conn.execute(
            "SELECT pnl, exit_price FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        assert row is not None
        assert abs(row["exit_price"] - 0.0) < 1e-9
        assert abs(row["pnl"] - (-5.98)) < 1e-9
