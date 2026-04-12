"""
tests/test_execution.py

Cobertura de Fase 5: ejecución y gestión de posiciones.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from main import _process_market
from core.models import Decision, TradeMode, TradeStatus
from execution.order_executor import PaperOrderExecutor
from execution.position_manager import PositionManager


class TestOrderExecutor:
    @pytest.mark.asyncio
    async def test_demo_submit_persists_open_trade(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        signal = make_signal(
            decision=Decision.YES,
            market_probability=0.56,
            kelly_size=0.12,
        )

        trade = await executor.submit(signal)

        assert trade.id is not None
        assert trade.signal_id is not None
        assert trade.status == TradeStatus.OPEN
        assert trade.mode == TradeMode.DEMO
        assert trade.contracts == 12
        open_trades = db.get_open_trades()
        assert len(open_trades) == 1

    @pytest.mark.asyncio
    async def test_demo_ignores_non_actionable_signal(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        signal = make_signal(decision=Decision.SKIP, market_probability=0.40)

        trade = await executor.submit(signal)

        assert trade.status == TradeStatus.CANCELLED
        assert db.get_open_trades() == []

    @pytest.mark.asyncio
    async def test_production_invalid_signal_raises_value_error(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.PRODUCTION, client=AsyncMock())
        signal = make_signal(decision=Decision.WAIT)

        with pytest.raises(ValueError):
            await executor.submit(signal)

    @pytest.mark.asyncio
    async def test_production_submit_calls_client(self, db, make_signal):
        client = AsyncMock()
        client.submit_order.return_value = {"order_id": "abc"}
        executor = PaperOrderExecutor(db=db, mode=TradeMode.PRODUCTION, client=client)
        signal = make_signal(
            decision=Decision.NO,
            market_probability=0.62,
            kelly_size=0.08,
        )

        trade = await executor.submit(signal)

        assert trade.status == TradeStatus.OPEN
        client.submit_order.assert_awaited_once()
        assert trade.mode == TradeMode.PRODUCTION

    @pytest.mark.asyncio
    async def test_production_submit_wraps_client_failure(self, db, make_signal):
        client = AsyncMock()
        client.submit_order.side_effect = TimeoutError("timeout")
        executor = PaperOrderExecutor(db=db, mode=TradeMode.PRODUCTION, client=client)
        signal = make_signal(decision=Decision.YES)

        with pytest.raises(RuntimeError):
            await executor.submit(signal)

    @pytest.mark.asyncio
    async def test_close_with_price_updates_trade_and_db(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        trade = await executor.submit(
            make_signal(decision=Decision.YES, market_probability=0.55, kelly_size=0.10)
        )

        closed = await executor.close_with_price(trade=trade, exit_price=0.70)

        assert closed.status == TradeStatus.CLOSED
        assert closed.closed_at is not None
        assert closed.pnl is not None
        assert db.get_open_trades() == []


class TestPositionManager:
    @pytest.mark.asyncio
    async def test_take_profit_closes_position(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, stop_loss_pct=0.05, take_profit_pct=0.10)
        trade = await manager.open_from_signal(
            make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
        )

        closes = await manager.evaluate_price(
            ticker=trade.ticker,
            current_yes_price=0.60,
        )

        assert len(closes) == 1
        assert closes[0].reason == "take_profit"
        assert manager.observability_snapshot()["open_positions"] == 0

    @pytest.mark.asyncio
    async def test_stop_loss_closes_position(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, stop_loss_pct=0.05, take_profit_pct=0.10)
        trade = await manager.open_from_signal(
            make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
        )

        closes = await manager.evaluate_price(
            ticker=trade.ticker,
            current_yes_price=0.45,
        )

        assert len(closes) == 1
        assert closes[0].reason == "stop_loss"

    @pytest.mark.asyncio
    async def test_no_close_when_thresholds_not_hit(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, stop_loss_pct=0.05, take_profit_pct=0.10)
        trade = await manager.open_from_signal(
            make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
        )

        closes = await manager.evaluate_price(
            ticker=trade.ticker,
            current_yes_price=0.52,
        )

        assert closes == []
        assert manager.observability_snapshot()["open_positions"] == 1

    @pytest.mark.asyncio
    async def test_try_open_from_signal_rejects_duplicate_ticker_once_traded(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, min_closed_trades=0, min_win_rate=0.0)

        first = await manager.try_open_from_signal(
            make_signal(
                market_ticker="KXBTC15M-26APR092215-15",
                decision=Decision.YES,
            ),
            max_positions=3,
        )
        second = await manager.try_open_from_signal(
            make_signal(
                market_ticker="KXBTC15M-26APR092215-15",
                decision=Decision.NO,
            ),
            max_positions=3,
        )

        assert first is not None
        assert second is None
        assert manager.observability_snapshot()["open_positions"] == 1
        assert len(db.get_open_trades()) == 1

    @pytest.mark.asyncio
    async def test_try_open_from_signal_does_not_reopen_closed_ticker(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, min_closed_trades=0, min_win_rate=0.0)

        trade = await manager.try_open_from_signal(
            make_signal(
                market_ticker="KXBTC15M-26APR092215-15",
                decision=Decision.YES,
                market_probability=0.50,
                kelly_size=0.10,
            ),
            max_positions=3,
        )
        assert trade is not None

        await manager.close_trade(trade, exit_price=0.65, reason="take_profit")

        reopened = await manager.try_open_from_signal(
            make_signal(
                market_ticker="KXBTC15M-26APR092215-15",
                decision=Decision.NO,
                market_probability=0.40,
                kelly_size=0.10,
            ),
            max_positions=3,
        )

        assert reopened is None

    @pytest.mark.asyncio
    async def test_try_open_from_signal_serializes_concurrent_duplicate_ticker(
        self,
        db,
        make_signal,
    ):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, min_closed_trades=0, min_win_rate=0.0)
        signal = make_signal(
            market_ticker="KXETH15M-26APR092215-15",
            decision=Decision.YES,
        )

        results = await asyncio.gather(
            manager.try_open_from_signal(signal, max_positions=3),
            manager.try_open_from_signal(signal, max_positions=3),
        )

        opened = [trade for trade in results if trade is not None]
        assert len(opened) == 1
        assert manager.observability_snapshot()["open_positions"] == 1
        assert len(db.get_open_trades()) == 1

    @pytest.mark.asyncio
    async def test_try_open_from_signal_respects_max_positions(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, min_closed_trades=0, min_win_rate=0.0)

        first = await manager.try_open_from_signal(
            make_signal(market_ticker="KXBTC15M-26APR092215-15"),
            max_positions=1,
        )
        second = await manager.try_open_from_signal(
            make_signal(market_ticker="KXETH15M-26APR092215-15"),
            max_positions=1,
        )

        assert first is not None
        assert second is None
        assert manager.observability_snapshot()["open_positions"] == 1
        assert len(db.get_open_trades()) == 1

    @pytest.mark.asyncio
    async def test_try_open_from_signal_raises_when_go_no_go_false(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor)

        with pytest.raises(RuntimeError, match="Risk violation: attempted trade while go=False"):
            await manager.try_open_from_signal(make_signal(), max_positions=3)

    @pytest.mark.asyncio
    async def test_go_no_go_insufficient_data(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor)
        await manager.open_from_signal(
            make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
        )

        status = manager.go_no_go_status(min_closed_trades=2)

        assert status.allowed is False
        assert status.reason == "insufficient_data"

    @pytest.mark.asyncio
    async def test_go_no_go_true_after_profitable_history(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, stop_loss_pct=0.05, take_profit_pct=0.10)
        for _ in range(3):
            trade = await manager.open_from_signal(
                make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
            )
            await manager.evaluate_price(ticker=trade.ticker, current_yes_price=0.60)

        status = manager.go_no_go_status(min_closed_trades=3, min_win_rate=0.50, min_total_pnl=0.0)

        assert status.allowed is True
        assert status.reason == "go"
        assert status.closed_trades == 3

    @pytest.mark.asyncio
    async def test_go_no_go_allows_profitable_history_even_below_win_rate_threshold(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, min_closed_trades=0, min_win_rate=0.75)

        winning = await manager.open_from_signal(
            make_signal(
                market_ticker="KXETH15M-26APR092215-15",
                decision=Decision.YES,
                market_probability=0.05,
                kelly_size=0.10,
            )
        )
        await manager.close_trade(winning, exit_price=1.0, reason="expiry")

        losing = await manager.open_from_signal(
            make_signal(
                market_ticker="KXETH15M-26APR092315-15",
                decision=Decision.YES,
                market_probability=0.80,
                kelly_size=0.10,
            )
        )
        await manager.close_trade(losing, exit_price=0.0, reason="expiry")

        status = manager.go_no_go_status(min_closed_trades=2, min_win_rate=0.75, min_total_pnl=-1.0)

        assert status.allowed is True
        assert status.reason == "go"

    @pytest.mark.asyncio
    async def test_hydrate_from_db_loads_closed_history_and_realized_pnl(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        writer = PositionManager(
            db=db,
            executor=executor,
            min_closed_trades=1,
            min_win_rate=0.0,
        )
        realized = 0.0
        for ticker in ("KXBTC-A", "KXBTC-B", "KXBTC-C"):
            trade = await writer.open_from_signal(
                make_signal(
                    market_ticker=ticker,
                    decision=Decision.YES,
                    market_probability=0.50,
                    kelly_size=0.10,
                )
            )
            closed = await writer.close_trade(trade, 0.60, "take_profit")
            realized += closed.trade.pnl or 0.0

        reader = PositionManager(db=db, executor=executor)
        await reader.hydrate_from_db(closed_limit=10)

        assert len(reader.closed_positions) == 3
        assert reader.realized_pnl == pytest.approx(realized)
        assert reader.total_pnl == pytest.approx(realized)
        assert reader.observability_snapshot()["closed_positions"] == 3

    @pytest.mark.asyncio
    async def test_go_no_go_stops_on_max_drawdown(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(
            db=db,
            executor=executor,
            initial_bankroll=100.0,
            min_closed_trades=1,
            min_win_rate=0.0,
            min_total_pnl=-999.0,
        )
        trade = await manager.open_from_signal(
            make_signal(
                decision=Decision.YES,
                market_probability=0.90,
                kelly_size=1.0,
            )
        )
        await manager.close_trade(trade, 0.0, "stop_loss")

        status = manager.go_no_go_status()

        assert status.allowed is False
        assert status.reason == "max_drawdown_exceeded"

    @pytest.mark.asyncio
    async def test_go_no_go_uses_category_specific_history(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(
            db=db,
            executor=executor,
            initial_bankroll=1_000.0,
            min_closed_trades=1,
            min_win_rate=0.50,
            min_total_pnl=-999.0,
            max_drawdown_pct=1.0,
        )

        btc_trade = await manager.open_from_signal(
            make_signal(
                market_ticker="KXBTC-15MIN-B95000",
                decision=Decision.YES,
                market_probability=0.90,
                kelly_size=0.10,
            )
        )
        await manager.close_trade(btc_trade, 0.0, "stop_loss")

        eth_trade = await manager.open_from_signal(
            make_signal(
                market_ticker="KXETH-15MIN-B3000",
                decision=Decision.NO,
                market_probability=0.60,
                kelly_size=0.10,
            )
        )
        await manager.close_trade(eth_trade, 1.0, "take_profit")

        btc_status = manager.go_no_go_status(category="BTC")
        eth_status = manager.go_no_go_status(category="ETH")

        assert btc_status.allowed is False
        assert btc_status.reason == "win_rate_too_low"
        assert eth_status.allowed is True
        assert eth_status.reason == "go"

    @pytest.mark.asyncio
    async def test_time_exit_profit_triggers_when_near_expiry_and_in_profit(self, db, make_signal):
        """
        evaluate_price con time_remaining_s <= threshold y precio >= entry * (1 + pct)
        debe cerrar la posición con reason ``time_exit_profit``.
        """
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(
            db=db,
            executor=executor,
            stop_loss_pct=0.05,
            take_profit_pct=0.20,  # umbral normal alto (no activado)
            time_exit_threshold_s=60,
            time_exit_profit_pct=0.08,
        )
        trade = await manager.open_from_signal(
            make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
        )

        closes = await manager.evaluate_price(
            ticker=trade.ticker,
            current_yes_price=0.55,  # +10 % sobre entry 0.50 ≥ time_exit_profit_pct 0.08
            time_remaining_s=45,     # por debajo del threshold de 60 s
        )

        assert len(closes) == 1
        assert closes[0].reason == "time_exit_profit"

    @pytest.mark.asyncio
    async def test_time_exit_profit_does_not_trigger_when_not_near_expiry(self, db, make_signal):
        """
        Con time_remaining_s > threshold no debe activarse la salida anticipada
        aunque el beneficio supere time_exit_profit_pct.
        """
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(
            db=db,
            executor=executor,
            stop_loss_pct=0.05,
            take_profit_pct=0.20,
            time_exit_threshold_s=60,
            time_exit_profit_pct=0.08,
        )
        trade = await manager.open_from_signal(
            make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
        )

        closes = await manager.evaluate_price(
            ticker=trade.ticker,
            current_yes_price=0.55,  # +10 % → cumpliría time_exit_profit_pct
            time_remaining_s=200,    # pero tiempo suficiente → no se activa
        )

        assert closes == []  # tampoco alcanza take_profit_pct=0.20

    @pytest.mark.asyncio
    async def test_time_exit_profit_does_not_trigger_when_profit_below_threshold(self, db, make_signal):
        """
        Con tiempo bajo pero beneficio por debajo de time_exit_profit_pct,
        no se cierra la posición.
        """
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(
            db=db,
            executor=executor,
            stop_loss_pct=0.05,
            take_profit_pct=0.20,
            time_exit_threshold_s=60,
            time_exit_profit_pct=0.08,
        )
        trade = await manager.open_from_signal(
            make_signal(decision=Decision.YES, market_probability=0.50, kelly_size=0.10)
        )

        closes = await manager.evaluate_price(
            ticker=trade.ticker,
            current_yes_price=0.52,  # +4 % < time_exit_profit_pct 0.08
            time_remaining_s=30,
        )

        assert closes == []

    @pytest.mark.asyncio
    async def test_process_market_blocks_trade_when_go_no_go_false(
        self,
        db,
        make_signal,
        make_market_snapshot,
        make_price_snapshot,
    ):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor)
        manager.try_open_from_signal = AsyncMock(side_effect=AssertionError("should not trade"))  # type: ignore[method-assign]
        router = AsyncMock()
        router.evaluate_async.return_value = make_signal(decision=Decision.YES)

        await _process_market(
            market=make_market_snapshot(),
            router=router,
            pos_mgr=manager,
            latest_prices={"BTC": {"binance": make_price_snapshot()}},
            bankroll=1000.0,
            max_positions=3,
        )

        manager.try_open_from_signal.assert_not_called()  # type: ignore[attr-defined]

    # ── Safe Mode tests ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_enter_safe_mode_blocks_go_no_go(self, db, make_signal):
        """Cuando safe mode está activo, go_no_go_status devuelve allowed=False."""
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, min_closed_trades=0, min_win_rate=0.0)

        assert not manager.is_safe_mode
        manager.enter_safe_mode(reason="test_trigger")

        status = manager.go_no_go_status()

        assert status.allowed is False
        assert "safe_mode" in status.reason
        assert manager.is_safe_mode

    @pytest.mark.asyncio
    async def test_safe_mode_raises_on_try_open(self, db, make_signal):
        """try_open_from_signal lanza RuntimeError cuando safe mode está activo."""
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor, min_closed_trades=0, min_win_rate=0.0)
        manager.enter_safe_mode(reason="hydration_failed")

        with pytest.raises(RuntimeError, match="Risk violation"):
            await manager.try_open_from_signal(make_signal(), max_positions=3)

    @pytest.mark.asyncio
    async def test_hydration_failure_enters_safe_mode(self, db):
        """Si hydrate_from_db falla, el manager entra en safe mode."""
        import sqlite3

        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor)

        # Cerrar la conexión de DB para forzar el fallo de hidratación
        db.close()

        with pytest.raises(Exception):
            await manager.hydrate_from_db()

        assert manager.is_safe_mode
        assert manager._safe_mode_reason is not None

    @pytest.mark.asyncio
    async def test_account_balance_depleted_blocks_trading(self, db, make_signal):
        """Si total_pnl <= -bankroll, go_no_go_status devuelve account_balance_depleted."""
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(
            db=db,
            executor=executor,
            initial_bankroll=100.0,
            min_closed_trades=0,
            min_win_rate=0.0,
            min_total_pnl=-999.0,
            max_drawdown_pct=2.0,  # drawdown check won't fire first
        )

        # Simulate total loss wiping the bankroll
        manager.realized_pnl = -100.0
        manager.total_pnl = -100.0

        status = manager.go_no_go_status()

        assert status.allowed is False
        assert status.reason == "account_balance_depleted"

    @pytest.mark.asyncio
    async def test_account_balance_zero_exactly_blocks_trading(self, db, make_signal):
        """El chequeo de capital es <= 0, entonces balance exactamente en 0 también bloquea."""
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(
            db=db,
            executor=executor,
            initial_bankroll=100.0,
            min_closed_trades=0,
            min_win_rate=0.0,
            min_total_pnl=-999.0,
            max_drawdown_pct=2.0,
        )

        manager.realized_pnl = -100.0
        manager.total_pnl = -100.0  # 100 - 100 == 0.0 → blocks

        status = manager.go_no_go_status()

        assert status.allowed is False
        assert status.reason == "account_balance_depleted"
