"""
tests/test_dashboard_api.py

Cobertura basica de la API de dashboard.
"""

from __future__ import annotations

import pytest

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
        db.save_signal(make_signal())
        app = create_app(db=db)
        with TestClient(app) as client:
            response = client.get("/state?limit=5")

        assert response.status_code == 200
        payload = response.json()
        assert "open_trades_count" in payload
        assert "blocked_categories" in payload
        assert "current_params" in payload
        assert "recent_signals" in payload
        assert len(payload["recent_signals"]) >= 1
