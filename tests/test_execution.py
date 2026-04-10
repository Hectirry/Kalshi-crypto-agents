"""
tests/test_execution.py

Cobertura de Fase 5: ejecución y gestión de posiciones.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

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
    async def test_try_open_from_signal_rejects_duplicate_open_ticker(self, db, make_signal):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor)

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
    async def test_try_open_from_signal_serializes_concurrent_duplicate_ticker(
        self,
        db,
        make_signal,
    ):
        executor = PaperOrderExecutor(db=db, mode=TradeMode.DEMO)
        manager = PositionManager(db=db, executor=executor)
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
        manager = PositionManager(db=db, executor=executor)

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
