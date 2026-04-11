"""
tests/test_dashboard_api.py

Cobertura basica de la API de dashboard.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from core.models import Decision
from dashboard.api_server import create_app

pytest.importorskip("httpx")
from fastapi.testclient import TestClient


class TestDashboardAPI:
    def test_health_endpoint(self, db):
        app = create_app(db=db)
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"

    def test_state_endpoint_returns_snapshot(self, db, make_signal):
        signal = make_signal()
        db.save_signal(signal)
        app = create_app(db=db)
        with TestClient(app) as client:
            response = client.get("/state?limit=5")

        assert response.status_code == 200
        payload = response.json()
        assert "open_trades_count" in payload
        assert "closed_trades_count" in payload
        assert "blocked_categories" in payload
        assert "current_params" in payload
        assert "realized_pnl" in payload
        assert "total_pnl" in payload
        assert "paper_account_negative" in payload
        assert "period_pnl" in payload
        assert "today" in payload["period_pnl"]
        assert "current_week" in payload["period_pnl"]
        assert "previous_week" in payload["period_pnl"]
        assert "current_month" in payload["period_pnl"]
        assert "recent_signals" in payload
        assert len(payload["recent_signals"]) >= 1
        assert "effective_mode" in payload
        assert "signal_decisions" in payload
        assert "skip_reason_summary" in payload

    def test_state_endpoint_reports_negative_paper_pnl(self, db, make_signal):
        from execution.order_executor import PaperOrderExecutor

        executor = PaperOrderExecutor(db=db, mode="demo")
        signal = make_signal(decision=Decision.YES, market_probability=0.90, kelly_size=1.0)
        trade = asyncio.run(executor.submit(signal))
        asyncio.run(executor.close_with_price(trade, 0.0))
        app = create_app(db=db)

        with TestClient(app) as client:
            response = client.get("/state?limit=5")

        assert response.status_code == 200
        payload = response.json()
        assert payload["paper_account_negative"] is True
        assert payload["realized_pnl"] < 0.0
        assert payload["period_pnl"]["today"]["realized_pnl"] < 0.0

    def test_manual_paper_order_creates_open_trade(self, db, app_config, make_market_snapshot):
        class FakeKalshiFeed:
            def __init__(self, config, bus):
                self.config = config
                self.bus = bus

            async def get_active_markets(self):
                return [
                    make_market_snapshot(
                        ticker="KXBTC-15MIN-B95000",
                        category="BTC",
                        implied_prob=0.55,
                        yes_ask=0.56,
                        no_ask=0.45,
                    )
                ]

        app = create_app(db=db, config=app_config)
        with patch("dashboard.api_server.KalshiFeed", FakeKalshiFeed):
            with TestClient(app) as client:
                response = client.post(
                    "/manual/paper-order",
                    json={
                        "ticker": "KXBTC-15MIN-B95000",
                        "side": "YES",
                        "contracts": 5,
                        "note": "test",
                    },
                )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["trade"]["status"] == "open"
        assert len(db.get_open_trades()) == 1

    def test_manual_paper_close_closes_existing_trade(self, db, app_config, make_market_snapshot, make_signal):
        from execution.order_executor import PaperOrderExecutor

        executor = PaperOrderExecutor(db=db, mode="demo")
        trade = asyncio.run(
            executor.submit(
                make_signal(
                    market_ticker="KXETH-15MIN-B3200",
                    decision=Decision.YES,
                    market_probability=0.55,
                    kelly_size=0.05,
                )
            )
        )

        class FakeKalshiFeed:
            def __init__(self, config, bus):
                self.config = config
                self.bus = bus

            async def get_active_markets(self):
                return [
                    make_market_snapshot(
                        ticker="KXETH-15MIN-B3200",
                        category="ETH",
                        implied_prob=0.70,
                        yes_ask=0.71,
                        no_ask=0.30,
                    )
                ]

        app = create_app(db=db, config=app_config)
        with patch("dashboard.api_server.KalshiFeed", FakeKalshiFeed):
            with TestClient(app) as client:
                response = client.post(
                    "/manual/paper-close",
                    json={"trade_id": trade.id, "note": "close"},
                )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["trade"]["status"] == "closed"
        assert db.get_open_trades() == []
