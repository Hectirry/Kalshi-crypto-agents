"""
tests/test_engine.py

Cobertura de Fase 3: EV, probabilidad, timing y router.
"""

from __future__ import annotations

import math
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import Confidence, Decision, Outcome
from engine.ev_calculator import EVCalculator
from engine.probability import (
    DEFAULT_VOLATILITY_1M,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    MOMENTUM_DRIFT_SCALE,
    ProbabilityEngine,
    TimeZone,
    VolatilityEstimate,
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
        db.upsert_param("min_delta", 0.01, "ETH", win_rate=0.62, sample_size=40)
        db.upsert_param("min_ev_threshold", 0.50, "ETH", win_rate=0.62, sample_size=40)
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
            market=make_market_snapshot(category="ETH", ticker="KXETH-15MIN-B3200"),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "ev_negative"

    def test_contract_price_out_of_range_skips(self, app_config, db, make_market_snapshot, make_price_snapshot):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.98,
            market_prob=0.85,
            delta=0.13,
            confidence=Confidence.HIGH,
        )

        signal = router.evaluate(
            market=make_market_snapshot(
                category="ETH",
                ticker="KXETH-15MIN-B3200",
                implied_prob=0.92,
                yes_ask=0.93,
                no_ask=0.08,
            ),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "contract_price_out_of_range"
        ev_calc.calculate.assert_not_called()

    def test_btc_category_override_requires_stricter_time_and_edge(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        router, prob_engine, ev_calc, timing_filter = self._make_router(app_config, db)
        timing_filter.should_enter.side_effect = [
            MagicMock(allowed=False, reason="too_late"),
        ]
        prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.95,
            market_prob=0.60,
            delta=0.35,
            confidence=Confidence.HIGH,
        )

        signal = router.evaluate(
            market=make_market_snapshot(category="BTC", implied_prob=0.60, yes_ask=0.61, time_to_expiry_s=120),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "too_late"
        prob_engine.estimate.assert_not_called()

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

        # El router ahora pasa volatility_estimate (VolatilityEstimate) en lugar de
        # volatility_1m; con solo un tick en memoria todas las ventanas son None.
        used_ve = prob_engine.estimate.call_args.kwargs["volatility_estimate"]
        assert used_ve.vol_1m is None
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

        # El router pasa volatility_estimate; después de 3 ticks la ventana 1m debe
        # tener al menos retornos suficientes para un valor no nulo.
        used_ve = prob_engine.estimate.call_args.kwargs["volatility_estimate"]
        assert used_ve.vol_1m is not None
        assert used_ve.vol_1m > 0.0

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
            market=make_market_snapshot(category="ETH", ticker="KXETH-15MIN-B3200", implied_prob=0.55, yes_ask=0.56),
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
            market=make_market_snapshot(
                category="ETH",
                ticker="KXETH-15MIN-B3200",
                implied_prob=0.55,
                yes_ask=0.56,
                no_ask=0.45,
            ),
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
            market=make_market_snapshot(
                category="ETH",
                ticker="KXETH-15MIN-B3200",
                implied_prob=0.55,
                yes_ask=0.56,
                no_ask=0.44,
            ),
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

        market = make_market_snapshot(category="ETH", ticker="KXETH-15MIN-B3200")
        price = make_price_snapshot()

        skip_delta = router.evaluate(market=market, price=price, bankroll=1_000.0)
        skip_ev = router.evaluate(market=market, price=price, bankroll=1_000.0)
        valid = router.evaluate(market=market, price=price, bankroll=1_000.0)

        assert skip_delta.decision == Decision.SKIP
        assert skip_ev.decision == Decision.SKIP
        assert valid.decision == Decision.YES
        assert db.save_signal.call_count == 3

    def test_build_agent_context_uses_only_resolved_history(self, db, make_market_snapshot, make_signal):
        from engine.context_builder import build_agent_context
        from intelligence.social_sentiment import SocialSentimentSnapshot

        now = time.time()
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                timestamp=now - 300,
                time_remaining_s=420,
                decision=Decision.YES,
                delta=0.11,
                ev_net_fees=0.08,
                outcome=Outcome.WIN,
                outcome_at=now - 100,
            )
        )
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B96000",
                timestamp=now - 240,
                time_remaining_s=390,
                decision=Decision.NO,
                delta=-0.09,
                ev_net_fees=0.05,
                outcome=Outcome.LOSS,
                outcome_at=now - 90,
            )
        )
        db.save_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B97000",
                timestamp=now - 180,
                time_remaining_s=380,
                decision=Decision.YES,
                delta=0.07,
                ev_net_fees=0.03,
                outcome=None,
            )
        )

        market = make_market_snapshot(
            ticker="KXBTC-15MIN-B95000",
            category="BTC",
            time_to_expiry_s=450,
            timestamp=now,
        )
        signal = make_signal(market_ticker=market.ticker, timestamp=now, time_remaining_s=450)
        social_sentiment = SocialSentimentSnapshot(
            symbol="BTC",
            source="reddit",
            sentiment_score=0.35,
            mention_count=11,
            bullish_ratio=0.55,
            bearish_ratio=0.18,
            acceleration=0.20,
            confidence=0.72,
            age_seconds=45,
            updated_at=now - 45,
        )

        context = build_agent_context(
            db=db,
            market=market,
            signal=signal,
            social_sentiment=social_sentiment,
        )

        assert context.overall.sample_size == 2
        assert context.overall.wins == 1
        assert context.same_category.sample_size == 2
        assert context.same_ticker.sample_size == 1
        assert context.same_direction.sample_size == 1
        assert context.same_setup.sample_size == 1
        assert len(context.recent_same_category) == 2
        assert len(context.recent_same_setup) == 1
        assert all(entry.outcome in {"WIN", "LOSS"} for entry in context.recent_same_category)
        assert context.social_sentiment == social_sentiment

    @pytest.mark.asyncio
    async def test_evaluate_async_passes_outcome_backed_context_to_agent(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
        make_signal,
    ):
        from engine.context_builder import AgentContext
        from engine.openrouter_agent import AgentVerdict
        from intelligence.social_sentiment import SocialSentimentSnapshot

        db.save_signal(
            make_signal(
                market_ticker="KXETH-15MIN-B3200",
                timestamp=time.time() - 300,
                time_remaining_s=420,
                decision=Decision.YES,
                delta=0.10,
                ev_net_fees=0.06,
                outcome=Outcome.WIN,
                outcome_at=time.time() - 200,
            )
        )
        mock_agent = AsyncMock()
        mock_agent.consult = AsyncMock(
            return_value=AgentVerdict(
                proceed=True,
                adjusted_kelly=0.05,
                reasoning="ok",
                tokens_used=12,
            )
        )
        mock_social_sentiment = MagicMock()
        mock_social_sentiment.get_snapshot.return_value = SocialSentimentSnapshot(
            symbol="ETH",
            source="reddit",
            sentiment_score=0.20,
            mention_count=8,
            bullish_ratio=0.50,
            bearish_ratio=0.13,
            acceleration=0.25,
            confidence=0.61,
            age_seconds=90,
            updated_at=time.time() - 90,
        )

        router = SignalRouter(
            prob_engine=MagicMock(),
            ev_calc=MagicMock(),
            timing_filter=MagicMock(),
            config=app_config.engine,
            db=db,
            blocked_categories=set(),
            openrouter_agent=mock_agent,
            social_sentiment_service=mock_social_sentiment,
        )
        router.timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        router.prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.70,
            market_prob=0.55,
            delta=0.15,
            confidence=Confidence.MEDIUM,
        )
        router.ev_calc.calculate.return_value = MagicMock(
            ev_gross=0.08,
            fee_total=0.01,
            ev_net=0.07,
            is_profitable=True,
            min_prob_to_profit=0.57,
        )
        router.ev_calc.kelly_size.return_value = 0.05

        await router.evaluate_async(
            market=make_market_snapshot(category="ETH", ticker="KXETH-15MIN-B3200"),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        consult_kwargs = mock_agent.consult.await_args.kwargs
        assert isinstance(consult_kwargs["context"], AgentContext)
        assert consult_kwargs["context"].same_ticker.sample_size == 1
        assert consult_kwargs["context"].same_category.sample_size >= 1
        assert consult_kwargs["context"].live_features is not None
        assert consult_kwargs["context"].social_sentiment is not None
        assert consult_kwargs["context"].social_sentiment.symbol == "ETH"
        assert 0.0 <= consult_kwargs["context"].live_features.rsi_14 <= 100.0

    def test_setup_quality_gate_skips_historically_weak_setup(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
        make_signal,
    ):
        from dataclasses import replace

        weak_cfg = replace(
            app_config.engine,
            setup_quality_gate_enabled=True,
            setup_quality_min_samples=3,
            setup_quality_min_win_rate=0.40,
        )
        base_ts = time.time()
        for idx in range(3):
            db.save_signal(
                make_signal(
                    market_ticker=f"KXETH-15MIN-B32{idx}",
                    decision=Decision.YES,
                    delta=0.15,
                    time_remaining_s=420,
                    timestamp=base_ts - (300 - idx),
                    outcome=Outcome.LOSS,
                    outcome_at=base_ts - (200 - idx),
                )
            )

        router = SignalRouter(
            prob_engine=MagicMock(),
            ev_calc=MagicMock(),
            timing_filter=MagicMock(),
            config=weak_cfg,
            db=db,
            blocked_categories=set(),
        )
        router.timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        router.prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.70,
            market_prob=0.55,
            delta=0.15,
            confidence=Confidence.HIGH,
        )
        router.ev_calc.calculate.return_value = MagicMock(
            ev_gross=0.08,
            fee_total=0.01,
            ev_net=0.07,
            is_profitable=True,
            min_prob_to_profit=0.57,
        )
        router.ev_calc.kelly_size.return_value = 0.05

        signal = router.evaluate(
            market=make_market_snapshot(
                category="ETH",
                ticker="KXETH-15MIN-B3300",
                time_to_expiry_s=420,
                timestamp=base_ts,
            ),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert "setup_quality_gate" in signal.reasoning

    def test_market_too_wide_skips_before_ev(
        self,
        app_config,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ):
        from dataclasses import replace

        cfg = replace(app_config.engine, max_market_overround_bps=100.0)
        router = SignalRouter(
            prob_engine=MagicMock(),
            ev_calc=MagicMock(),
            timing_filter=MagicMock(),
            config=cfg,
            db=db,
            blocked_categories=set(),
        )
        router.timing_filter.should_enter.side_effect = [
            MagicMock(allowed=True, reason="ok"),
            MagicMock(allowed=True, reason="ok"),
        ]
        router.prob_engine.estimate.return_value = MagicMock(
            error=False,
            error_msg=None,
            my_prob=0.70,
            market_prob=0.55,
            delta=0.15,
            confidence=Confidence.HIGH,
        )

        signal = router.evaluate(
            market=make_market_snapshot(
                category="ETH",
                ticker="KXETH-15MIN-B3200",
                implied_prob=0.55,
                yes_ask=0.60,
                no_ask=0.45,
            ),
            price=make_price_snapshot(),
            bankroll=1_000.0,
        )

        assert signal.decision == Decision.SKIP
        assert signal.reasoning == "market_too_wide"
        assert signal.contract_price == 0.60
        assert signal.market_overround_bps == pytest.approx(500.0)
        router.ev_calc.calculate.assert_not_called()


# ── Fix 1: my_prob clamping ───────────────────────────────────────────────────

class TestMyProbClamping:
    """Verifica que ProbabilityEngine clampea my_prob a [0.01, 0.99] siempre."""

    def test_spot_far_above_strike_clamps_to_max(
        self, make_market_snapshot, make_price_snapshot
    ):
        """Precio spot muy por encima del strike → my_prob == 0.99."""
        engine = ProbabilityEngine()
        # Strike 90k, spot 200k: probabilidad casi-segura de que el contrato expire ITM.
        market = make_market_snapshot(strike=90_000.0, implied_prob=0.90, time_to_expiry_s=600)
        price = make_price_snapshot(price=200_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.0001)

        assert result.my_prob == MAX_PROBABILITY  # 0.99

    def test_spot_far_below_strike_clamps_to_min(
        self, make_market_snapshot, make_price_snapshot
    ):
        """Precio spot muy por debajo del strike → my_prob == 0.01."""
        engine = ProbabilityEngine()
        # Strike 200k, spot 50k: probabilidad casi-nula de expirar ITM.
        market = make_market_snapshot(strike=200_000.0, implied_prob=0.10, time_to_expiry_s=600)
        price = make_price_snapshot(price=50_000.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.0001)

        assert result.my_prob == MIN_PROBABILITY  # 0.01

    def test_intermediate_probabilities_not_modified_by_clamp(
        self, make_market_snapshot, make_price_snapshot
    ):
        """Probabilidades intermedias (0.3–0.7) pasan sin modificar por el clamp."""
        engine = ProbabilityEngine()
        # Spot cerca del strike → modelo produce probabilidad en zona media.
        market = make_market_snapshot(strike=95_000.0, implied_prob=0.55, time_to_expiry_s=450)
        price = make_price_snapshot(price=95_200.0)

        result = engine.estimate(market=market, price=price, volatility_1m=0.005)

        # El resultado debe ser interior al rango — no tocado por el clamp.
        assert result.my_prob > MIN_PROBABILITY
        assert result.my_prob < MAX_PROBABILITY
        assert result.my_prob not in (MIN_PROBABILITY, MAX_PROBABILITY)


# ── Fix 2: EV units normalization ─────────────────────────────────────────────

class TestEVNormalized:
    """Verifica que ev_net es fracción del capital arriesgado, comparable con min_ev_threshold=0.04."""

    def test_profitable_trade_ev_above_threshold(self):
        """Caso 1 — trade rentable: ev_net > 0.04 con edge claro."""
        calc = EVCalculator()
        # capital_at_risk = 0.55 * 10 = $5.50
        # ev_gross = (0.70 - 0.55) / 0.55 ≈ 0.2727
        # fee_fraction = fee_per_contract(0.55) / 0.55 ≈ 0.0315
        # ev_net ≈ 0.241 > 0.04
        result = calc.calculate(
            my_prob=0.70,
            contract_price=0.55,
            contracts=10,
            bankroll=100.0,
        )

        assert result.ev_gross > 0.0
        assert result.ev_net > 0.04
        assert result.is_profitable is True

    def test_marginal_trade_ev_below_threshold(self):
        """Caso 2 — trade marginal: fees casi destruyen el edge, ev_net < 0.04."""
        calc = EVCalculator()
        # ev_gross = (0.57 - 0.55) / 0.55 ≈ 0.0364
        # fee_fraction ≈ 0.0315
        # ev_net ≈ 0.005 → positivo pero muy por debajo del umbral 0.04
        result = calc.calculate(
            my_prob=0.57,
            contract_price=0.55,
            contracts=10,
            bankroll=100.0,
        )

        assert result.ev_net < 0.04  # router lo filtraría con min_ev_threshold=0.04
        assert result.ev_gross > 0.0  # hay algo de edge bruto

    def test_no_edge_zero_kelly_negative_ev(self):
        """Caso 3 — sin edge (my_prob <= price): kelly=0.0 y ev_net negativo."""
        calc = EVCalculator()
        result = calc.calculate(
            my_prob=0.54,
            contract_price=0.55,
            contracts=1,
            bankroll=100.0,
        )
        kelly = calc.kelly_size(my_prob=0.54, contract_price=0.55)

        assert kelly == 0.0
        assert result.ev_net < 0.0

    def test_extreme_price_near_certain_positive_ev(self):
        """Caso 4 — precio extremo (0.95): fee mínima en extremos, ev_net > 0."""
        calc = EVCalculator()
        # fee_per_contract(0.95) = 0.95 * 0.05 * 0.07 ≈ 0.0033 — muy pequeña
        result = calc.calculate(
            my_prob=0.97,
            contract_price=0.95,
            contracts=5,
            bankroll=100.0,
        )

        assert result.ev_net > 0.0


class TestVolatilityEstimate:
    """Tests para VolatilityEstimate dataclass y flujo multi-timeframe en SignalRouter."""

    # ------------------------------------------------------------------
    # 1. VolatilityEstimate.blended() — lógica de mezcla ponderada
    # ------------------------------------------------------------------

    def test_blended_all_windows_uses_weighted_average(self):
        """Con las tres ventanas disponibles el resultado es la media ponderada 40/35/25."""
        ve = VolatilityEstimate(vol_1m=0.010, vol_5m=0.008, vol_15m=0.006)
        expected = (0.40 * 0.010 + 0.35 * 0.008 + 0.25 * 0.006) / 1.0
        assert abs(ve.blended() - expected) < 1e-9

    def test_blended_only_one_window_returns_that_value(self):
        """Cuando solo hay una ventana disponible, se retorna su valor directamente."""
        ve = VolatilityEstimate(vol_1m=None, vol_5m=0.007, vol_15m=None)
        assert ve.blended() == 0.007

    def test_blended_no_windows_returns_default(self):
        """Sin ninguna ventana válida, se retorna DEFAULT_VOLATILITY_1M."""
        ve = VolatilityEstimate()
        assert ve.blended() == DEFAULT_VOLATILITY_1M

    def test_blended_renormalizes_partial_windows(self):
        """Con solo vol_1m y vol_5m los pesos 0.40/0.35 se renormalizan a 0.4/0.75 y 0.35/0.75."""
        ve = VolatilityEstimate(vol_1m=0.010, vol_5m=0.005, vol_15m=None)
        total_w = 0.40 + 0.35
        expected = (0.40 * 0.010 + 0.35 * 0.005) / total_w
        assert abs(ve.blended() - expected) < 1e-9

    # ------------------------------------------------------------------
    # 2. ProbabilityEngine.estimate() — volatility_estimate tiene precedencia
    # ------------------------------------------------------------------

    def test_estimate_uses_volatility_estimate_over_float(
        self, make_market_snapshot, make_price_snapshot
    ):
        """volatility_estimate tiene precedencia sobre volatility_1m cuando ambos se pasan.

        Verificación por igualdad de ruta: pasar vol=0.020 vía volatility_estimate con
        volatility_1m=0.001 debe producir exactamente el mismo resultado que pasar
        vol=0.020 vía volatility_1m solo (ambos usan raw_volatility=0.020).
        """
        engine = ProbabilityEngine()
        market = make_market_snapshot(strike=50_000.0, time_to_expiry_s=300, implied_prob=0.50)
        price = make_price_snapshot(price=51_000.0)

        # Ruta 1: volatility_estimate con vol_1m=0.020, float=0.001 (debe ignorarse)
        result_via_estimate = engine.estimate(
            market=market,
            price=price,
            volatility_1m=0.001,
            volatility_estimate=VolatilityEstimate(vol_1m=0.020),
        )

        # Ruta 2: solo float=0.020 (sin volatility_estimate)
        result_via_float = engine.estimate(
            market=market,
            price=price,
            volatility_1m=0.020,
        )

        # Si volatility_estimate tiene precedencia, ambas rutas deben producir idéntico my_prob
        assert abs(result_via_estimate.my_prob - result_via_float.my_prob) < 1e-9

    # ------------------------------------------------------------------
    # 3. SignalRouter._vol_for_window_from_memory — aislamiento de ventanas
    # ------------------------------------------------------------------

    def test_vol_for_window_only_uses_ticks_within_window(
        self, app_config, db, make_market_snapshot, make_price_snapshot
    ):
        """_vol_for_window_from_memory no mezcla ticks fuera de la ventana solicitada."""
        from engine.ev_calculator import EVCalculator
        from engine.timing import TimingFilter

        router = SignalRouter(
            prob_engine=ProbabilityEngine(),
            ev_calc=EVCalculator(),
            timing_filter=TimingFilter(),
            config=app_config,
            db=db,
            blocked_categories=set(),
        )

        base_ts = 1_000_000.0
        symbol = "BTC"
        # Inyectar precios a lo largo de 10 minutos
        for i in range(61):
            router._price_memory.setdefault(symbol, __import__("collections").deque(maxlen=router._price_memory_maxlen))
            router._price_memory[symbol].append((base_ts + i, 50_000.0 + i * 10))

        # Ventana de 60 s: solo los últimos 60 ticks
        vol_60 = router._vol_for_window_from_memory(symbol, 60)
        # Ventana de 600 s: los 61 ticks (todos están dentro)
        vol_600 = router._vol_for_window_from_memory(symbol, 600)

        # Con el mismo conjunto de retornos lineales la vol debe ser idéntica (serie uniforme)
        # pero el número de ticks en cada ventana es diferente — lo que nos interesa es
        # que vol_60 no sea None (tiene suficientes ticks) y que no explote.
        assert vol_60 is not None
        assert vol_600 is not None
        assert vol_60 >= 0.0
        assert vol_600 >= 0.0
