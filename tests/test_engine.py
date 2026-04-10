"""
tests/test_engine.py

Cobertura de Fase 3: EV, probabilidad, timing y router.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import pytest

from core.models import Confidence, Decision
from engine.ev_calculator import EVCalculator
from engine.probability import (
    DEFAULT_VOLATILITY_1M,
    MOMENTUM_DRIFT_SCALE,
    ProbabilityEngine,
    TimeZone,
    classify_time_zone,
)
from engine.signal_router import SignalRouter
from engine.timing import TimingFilter


class TestEVCalculator:
    def test_positive_ev_with_real_edge(self):
        calc = EVCalculator()

        result = calc.calculate(
            my_prob=0.70,
            contract_price=0.55,
            contracts=10,
            bankroll=1_000.0,
        )

        assert result.ev_gross > 0.0
        assert result.ev_net > 0.0
        assert result.is_profitable is True

    def test_negative_ev_when_fees_overpower_edge(self):
        calc = EVCalculator()

        result = calc.calculate(
            my_prob=0.56,
            contract_price=0.55,
            contracts=10,
            bankroll=1_000.0,
        )

        assert result.ev_gross > 0.0
        assert result.ev_net < 0.0
        assert result.is_profitable is False

    def test_kelly_size_returns_zero_without_edge(self):
        calc = EVCalculator()

        assert calc.kelly_size(my_prob=0.55, contract_price=0.55) == 0.0
        assert calc.kelly_size(my_prob=0.50, contract_price=0.55) == 0.0

    def test_kelly_size_is_clamped_to_max_pct(self):
        calc = EVCalculator()

        result = calc.kelly_size(
            my_prob=0.95,
            contract_price=0.20,
            kelly_fraction=0.25,
            max_pct=0.05,
        )

        assert result == 0.05

    @pytest.mark.parametrize(
        ("price", "expected"),
        [
            (0.50, 0.50 * 0.50 * 0.07),
            (0.80, 0.80 * 0.20 * 0.07),
            (0.95, 0.95 * 0.05 * 0.07),
        ],
    )
    def test_fee_schedule_matches_formula(self, price, expected):
        calc = EVCalculator()

        assert math.isclose(calc.fee_per_contract(price), expected, rel_tol=1e-9)

    def test_min_prob_to_profit_calculated_correctly(self):
        calc = EVCalculator()

        result = calc.calculate(
            my_prob=0.70,
            contract_price=0.55,
            contracts=5,
            bankroll=500.0,
        )

        expected = 0.55 + (0.55 * 0.45 * 0.07)
        assert math.isclose(result.min_prob_to_profit, expected, rel_tol=1e-9)


class TestProbabilityEngine:
    def test_delta_zero_returns_no_trade_context(
        self,
        make_market_snapshot,
        make_price_snapshot,
    ):
        engine = ProbabilityEngine()
        market = make_market_snapshot(
            strike=95_000.0,
            implied_prob=0.50,
            time_to_expiry_s=600,
        )
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.error is False
        assert math.isclose(result.my_prob, 0.50, abs_tol=1e-9)
        assert math.isclose(result.delta, 0.0, abs_tol=1e-9)
        assert result.confidence == Confidence.LOW

    def test_spot_above_strike_returns_prob_above_half(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=95_000.0, implied_prob=0.52, time_to_expiry_s=600)
        price = make_price_snapshot(price=96_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.error is False
        assert result.my_prob > 0.5

    def test_spot_below_strike_returns_prob_below_half(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=95_000.0, implied_prob=0.48, time_to_expiry_s=600)
        price = make_price_snapshot(price=94_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.error is False
        assert result.my_prob < 0.5

    def test_short_time_falls_back_to_market_prob(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(implied_prob=0.61, time_to_expiry_s=20)
        price = make_price_snapshot()

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.my_prob == market.implied_prob
        assert result.market_prob == market.implied_prob
        assert result.confidence == Confidence.LOW

    def test_missing_strike_returns_error_flag(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=None)
        price = make_price_snapshot()

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.error is True
        assert result.error_msg == "missing_strike"

    def test_probability_is_clamped_to_safe_range(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=95_000.0, time_to_expiry_s=900)

        high = engine.estimate(
            market=market,
            price=make_price_snapshot(price=500_000.0),
            volatility_1m=0.0001,
        )
        low = engine.estimate(
            market=market,
            price=make_price_snapshot(price=1_000.0),
            volatility_1m=0.0001,
        )

        assert 0.01 <= high.my_prob <= 0.99
        assert 0.01 <= low.my_prob <= 0.99

    def test_default_volatility_is_used_when_none(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=95_500.0, time_to_expiry_s=60)
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=None)

        price_gap_ratio = (price.price - market.strike) / price.price
        normalized_gap = price_gap_ratio / DEFAULT_VOLATILITY_1M
        bounded_gap = max(-1.0, min(normalized_gap, 1.0))
        time_weight = min(market.time_to_expiry_s / 900.0, 1.0)
        drift_ratio = bounded_gap * DEFAULT_VOLATILITY_1M * MOMENTUM_DRIFT_SCALE * time_weight
        drifted_price = price.price * (1.0 + drift_ratio)
        expected_z = (market.strike - price.price) / (
            price.price * DEFAULT_VOLATILITY_1M * math.sqrt(market.time_to_expiry_s / 60.0)
        )
        expected_z = (market.strike - drifted_price) / (
            price.price * DEFAULT_VOLATILITY_1M * math.sqrt(market.time_to_expiry_s / 60.0)
        )
        expected_prob = 1.0 - 0.5 * (1.0 + math.erf(expected_z / math.sqrt(2.0)))
        expected_prob = max(0.01, min(expected_prob, 0.99))

        assert math.isclose(result.my_prob, expected_prob, rel_tol=1e-9)


class TestTimeZone:
    """Tests para clasificación NEAR/MID/FAR y downgrade de confianza FAR."""

    # ── classify_time_zone ────────────────────────────────────────────────────

    def test_classify_near_boundary(self):
        assert classify_time_zone(90) == TimeZone.NEAR
        assert classify_time_zone(300) == TimeZone.NEAR

    def test_classify_mid_range(self):
        assert classify_time_zone(301) == TimeZone.MID
        assert classify_time_zone(450) == TimeZone.MID
        assert classify_time_zone(600) == TimeZone.MID

    def test_classify_far_range(self):
        assert classify_time_zone(601) == TimeZone.FAR
        assert classify_time_zone(720) == TimeZone.FAR
        assert classify_time_zone(840) == TimeZone.FAR

    def test_classify_zero_is_near(self):
        assert classify_time_zone(0) == TimeZone.NEAR

    # ── ProbabilityResult lleva time_zone ────────────────────────────────────

    def test_estimate_mid_market_has_mid_zone(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=95_000.0, implied_prob=0.52, time_to_expiry_s=600)
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.time_zone == TimeZone.MID

    def test_estimate_near_market_has_near_zone(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=95_000.0, implied_prob=0.52, time_to_expiry_s=200)
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.time_zone == TimeZone.NEAR

    def test_estimate_far_market_has_far_zone(self, make_market_snapshot, make_price_snapshot):
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=95_000.0, implied_prob=0.52, time_to_expiry_s=700)
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.time_zone == TimeZone.FAR

    # ── FAR aplica downgrade de confianza ─────────────────────────────────────

    def test_far_market_high_delta_gets_medium_confidence(
        self, make_market_snapshot, make_price_snapshot
    ):
        """HIGH delta en FAR → confianza baja a MEDIUM (sería HIGH en MID)."""
        engine = ProbabilityEngine()
        # Delta alto: spot muy por encima del strike → my_prob alta, delta > 0.10
        market = make_market_snapshot(
            strike=90_000.0,
            implied_prob=0.35,
            time_to_expiry_s=700,  # FAR
        )
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        # El delta es grande → sin downgrade sería HIGH, con FAR downgrade es MEDIUM
        assert result.time_zone == TimeZone.FAR
        assert result.confidence in (Confidence.MEDIUM, Confidence.LOW)  # downgraded
        assert result.confidence != Confidence.HIGH

    def test_mid_market_same_delta_keeps_high_confidence(
        self, make_market_snapshot, make_price_snapshot
    ):
        """Mismo setup pero en MID → confianza permanece HIGH."""
        engine = ProbabilityEngine()
        market = make_market_snapshot(
            strike=90_000.0,
            implied_prob=0.35,
            time_to_expiry_s=500,  # MID
        )
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.time_zone == TimeZone.MID
        assert result.confidence == Confidence.HIGH

    def test_far_market_medium_delta_gets_low_confidence(
        self, make_market_snapshot, make_price_snapshot
    ):
        """MEDIUM delta en FAR → confianza baja a LOW."""
        engine = ProbabilityEngine()
        # Delta medio: spot ligeramente sobre strike → my_prob entre 0.55-0.65
        market = make_market_snapshot(
            strike=94_500.0,
            implied_prob=0.50,
            time_to_expiry_s=700,  # FAR
        )
        price = make_price_snapshot(price=95_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        if result.time_zone == TimeZone.FAR:
            assert result.confidence != Confidence.HIGH

    def test_error_result_preserves_default_time_zone(
        self, make_market_snapshot, make_price_snapshot
    ):
        """Resultados de error retornan time_zone=MID por defecto."""
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=None, time_to_expiry_s=700)
        price = make_price_snapshot()

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        assert result.error is True
        assert result.time_zone == TimeZone.MID  # valor por defecto en error paths


class TestTimingFilter:
    def test_too_early(self):
        result = TimingFilter().should_enter(
            time_remaining_s=841,
            confidence=Confidence.HIGH,
            min_time_s=90,
        )

        assert result.allowed is False
        assert result.reason == "too_early"

    def test_too_late(self):
        result = TimingFilter().should_enter(
            time_remaining_s=89,
            confidence=Confidence.HIGH,
            min_time_s=90,
        )

        assert result.allowed is False
        assert result.reason == "too_late"

    def test_low_confidence_late(self):
        result = TimingFilter().should_enter(
            time_remaining_s=299,
            confidence=Confidence.LOW,
            min_time_s=90,
        )

        assert result.allowed is False
        assert result.reason == "low_conf_late"

    def test_valid_window_is_allowed(self):
        result = TimingFilter().should_enter(
            time_remaining_s=600,
            confidence=Confidence.MEDIUM,
            min_time_s=90,
        )

        assert result.allowed is True
        assert result.reason == "ok"


class TestSignalRouter:
    def _make_router(self, app_config, db, blocked_categories=None):
        prob_engine = MagicMock()
        ev_calc = MagicMock()
        timing_filter = MagicMock()
        router = SignalRouter(
            prob_engine=prob_engine,
            ev_calc=ev_calc,
            timing_filter=timing_filter,
            config=app_config.engine,
            db=db,
            blocked_categories=blocked_categories or set(),
        )
        return router, prob_engine, ev_calc, timing_filter

    def test_blocked_category_skips_without_calling_engine(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(
            app_config,
            db,
            blocked_categories={"BTC"},
        )

        signal = router.evaluate(
            market=make_market_snapshot(category="BTC"),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "blocked_category"
        timing_filter.should_enter.assert_not_called()
        prob_engine.estimate.assert_not_called()
        ev_calc.calculate.assert_not_called()

    def test_invalid_timing_skips_without_probability(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.return_value = MagicMock(allowed=False, reason="too_early")

        signal = router.evaluate(
            market=make_market_snapshot(),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "too_early"
        prob_engine.estimate.assert_not_called()
        ev_calc.calculate.assert_not_called()

    def test_small_delta_skips(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.57,
            market_prob=0.55,
            delta=0.02,
            confidence=Confidence.LOW,
        )
        db.save_signal = MagicMock(wraps=db.save_signal)

        signal = router.evaluate(
            market=make_market_snapshot(implied_prob=0.55, time_to_expiry_s=500),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "delta_too_small"
        ev_calc.calculate.assert_not_called()
        db.save_signal.assert_called_once()

    def test_calibrated_min_delta_overrides_config(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        db.upsert_param("min_delta", 0.18, "BTC", win_rate=0.62, sample_size=40)
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.72,
            market_prob=0.55,
            delta=0.17,
            confidence=Confidence.HIGH,
        )

        signal = router.evaluate(
            market=make_market_snapshot(category="BTC"),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "delta_too_small"
        ev_calc.calculate.assert_not_called()

    def test_calibrated_min_ev_threshold_overrides_config(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        db.upsert_param("min_delta", 0.01, "BTC", win_rate=0.62, sample_size=40)
        db.upsert_param("min_ev_threshold", 0.50, "BTC", win_rate=0.62, sample_size=40)
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.72,
            market_prob=0.55,
            delta=0.17,
            confidence=Confidence.HIGH,
        )
        ev_calc.calculate.return_value = MagicMock(
            ev_gross=0.20,
            fee_total=0.02,
            ev_net=0.18,
            is_profitable=True,
            min_prob_to_profit=0.57,
        )

        signal = router.evaluate(
            market=make_market_snapshot(category="BTC"),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "ev_negative"

    def test_router_does_not_use_volume_as_volatility(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.57,
            market_prob=0.55,
            delta=0.02,
            confidence=Confidence.LOW,
        )
        price = make_price_snapshot(price=95_000.0, volume_1m=12345.0)

        router.evaluate(
            market=make_market_snapshot(implied_prob=0.55, time_to_expiry_s=500),
            price=price,
            bankroll=1_000.0,
        )

        used_volatility = prob_engine.estimate.call_args.kwargs["volatility_1m"]
        assert used_volatility is None
        ev_calc.calculate.assert_not_called()

    def test_router_uses_realized_volatility_from_price_history(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, _ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.57,
            market_prob=0.55,
            delta=0.02,
            confidence=Confidence.LOW,
        )

        market = make_market_snapshot(implied_prob=0.55, time_to_expiry_s=500)
        base_ts = market.timestamp
        p1 = make_price_snapshot(price=95_000.0, timestamp=base_ts - 40, volume_1m=99999.0)
        p2 = make_price_snapshot(price=95_080.0, timestamp=base_ts - 20, volume_1m=99999.0)
        p3 = make_price_snapshot(price=95_120.0, timestamp=base_ts - 5, volume_1m=99999.0)

        router.evaluate(market=market, price=p1, bankroll=1_000.0)
        router.evaluate(market=market, price=p2, bankroll=1_000.0)
        router.evaluate(market=market, price=p3, bankroll=1_000.0)

        used_volatility = prob_engine.estimate.call_args.kwargs["volatility_1m"]
        assert used_volatility is not None
        assert used_volatility > 0.0

    def test_delta_zero_skips_as_no_trade(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.50,
            market_prob=0.50,
            delta=0.0,
            confidence=Confidence.LOW,
        )

        signal = router.evaluate(
            market=make_market_snapshot(implied_prob=0.50, time_to_expiry_s=500),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "delta_too_small"
        ev_calc.calculate.assert_not_called()

    def test_negative_ev_skips(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.70,
            market_prob=0.55,
            delta=0.15,
            confidence=Confidence.HIGH,
        )
        ev_calc.calculate.return_value = MagicMock(
            ev_gross=0.05,
            fee_total=0.06,
            ev_net=-0.01,
            is_profitable=False,
            min_prob_to_profit=0.57,
        )
        db.save_signal = MagicMock(wraps=db.save_signal)

        signal = router.evaluate(
            market=make_market_snapshot(implied_prob=0.55, yes_ask=0.56),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "ev_negative"
        db.save_signal.assert_called_once()

    def test_time_remaining_below_90s_closes_window(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.return_value = MagicMock(allowed=False, reason="too_late")

        signal = router.evaluate(
            market=make_market_snapshot(time_to_expiry_s=89),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "too_late"
        prob_engine.estimate.assert_not_called()
        ev_calc.calculate.assert_not_called()

    def test_valid_trade_returns_yes_with_positive_kelly(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.72,
            market_prob=0.55,
            delta=0.17,
            confidence=Confidence.HIGH,
        )
        ev_calc.calculate.return_value = MagicMock(
            ev_gross=12.0,
            fee_total=1.0,
            ev_net=11.0,
            is_profitable=True,
            min_prob_to_profit=0.57,
        )
        ev_calc.kelly_size.return_value = 0.05
        db.save_signal = MagicMock(wraps=db.save_signal)

        signal = router.evaluate(
            market=make_market_snapshot(implied_prob=0.55, yes_ask=0.56, no_ask=0.45),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.YES
        assert signal.kelly_size > 0.0
        assert signal.ev_net_fees == 11.0
        db.save_signal.assert_called_once()

    def test_valid_trade_with_negative_delta_returns_no(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.35,
            market_prob=0.55,
            delta=-0.20,
            confidence=Confidence.HIGH,
        )
        ev_calc.calculate.return_value = MagicMock(
            ev_gross=8.0,
            fee_total=1.0,
            ev_net=7.0,
            is_profitable=True,
            min_prob_to_profit=0.48,
        )
        ev_calc.kelly_size.return_value = 0.04

        signal = router.evaluate(
            market=make_market_snapshot(implied_prob=0.55, yes_ask=0.56, no_ask=0.44),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.NO
        assert signal.kelly_size == 0.04

    def test_internal_exception_returns_error_without_propagation(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = RuntimeError("timing_failed")

        signal = router.evaluate(
            market=make_market_snapshot(),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.ERROR
        assert signal.error_msg == "timing_failed"
        prob_engine.estimate.assert_not_called()
        ev_calc.calculate.assert_not_called()

    def test_signal_persisted_for_delta_skip_ev_skip_and_valid(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        db.save_signal = MagicMock(wraps=db.save_signal)

        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.side_effect = [
            MagicMock(
                error=False,
                error_msg=None,
                my_prob=0.57,
                market_prob=0.55,
                delta=0.02,
                confidence=Confidence.LOW,
            ),
            MagicMock(
                error=False,
                error_msg=None,
                my_prob=0.72,
                market_prob=0.55,
                delta=0.17,
                confidence=Confidence.HIGH,
            ),
            MagicMock(
                error=False,
                error_msg=None,
                my_prob=0.72,
                market_prob=0.55,
                delta=0.17,
                confidence=Confidence.HIGH,
            ),
        ]
        ev_calc.calculate.side_effect = [
            MagicMock(
                ev_gross=0.01,
                fee_total=0.02,
                ev_net=-0.01,
                is_profitable=False,
                min_prob_to_profit=0.57,
            ),
            MagicMock(
                ev_gross=1.0,
                fee_total=0.1,
                ev_net=0.9,
                is_profitable=True,
                min_prob_to_profit=0.57,
            ),
        ]
        ev_calc.kelly_size.return_value = 0.05

        market = make_market_snapshot()
        price = make_price_snapshot()

        skip_delta = router.evaluate(market=market, price=price, bankroll=1_000.0)
        skip_ev = router.evaluate(market=market, price=price, bankroll=1_000.0)
        valid = router.evaluate(market=market, price=price, bankroll=1_000.0)

        assert skip_delta.decision == Decision.SKIP
        assert skip_ev.decision == Decision.SKIP
        assert valid.decision == Decision.YES
        assert db.save_signal.call_count == 3
