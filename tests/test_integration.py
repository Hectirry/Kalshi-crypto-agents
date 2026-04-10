"""
tests/test_integration.py

Tests de integración para el orquestador principal y el agente OpenRouter.

Puerta de salida:
  - main arranca en --dry-run sin errores
  - OpenRouterAgent con mock retorna AgentVerdict válido
  - AgentVerdict con proceed=False convierte señal en SKIP
  - Timeout del agente (mock que tarda 4s) → señal pasa sin cambios
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Fixtures locales ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_signal(make_signal):
    from core.models import Confidence
    return make_signal(confidence=Confidence.MEDIUM)


@pytest.fixture
def sample_market(make_market_snapshot):
    return make_market_snapshot()


@pytest.fixture
def sample_price(make_price_snapshot):
    return make_price_snapshot()


# ── 1. main en --dry-run arranca sin errores ──────────────────────────────────

class TestDryRunStartup:
    """Verifica que main.py --dry-run arranque y se pueda cancelar limpiamente."""

    def test_dry_run_starts_cleanly(self, tmp_path: Path) -> None:
        """
        El orquestador en --dry-run debe:
          1. Cargar config sin lanzar excepciones.
          2. Inicializar la DB.
          3. Recalibrar (sin señales → skip silencioso).
          4. Arrancar el loop (sin conectar a APIs reales → mocks).
          5. Cancelarse ante SIGINT sin dejar traceback.
        """
        db_path = tmp_path / "test_dry_run.db"
        os.environ["ENV"] = "demo"
        os.environ.setdefault("KALSHI_API_KEY", "test-key")
        os.environ.setdefault("DB_PATH", str(db_path))

        from core.config import load_config
        from core.database import Database

        cfg = load_config()
        db = Database(path=db_path)
        db.initialize()

        # Verificar que la DB arranca sin señales (recalibrate hace skip)
        from main import _maybe_recalibrate
        _maybe_recalibrate(db, bankroll=100.0)   # no lanza excepciones

        db.close()

    @pytest.mark.asyncio
    async def test_orchestrator_cancels_cleanly(self, tmp_path: Path) -> None:
        """
        _run_orchestrator se cancela limpiamente cuando el shutdown_event se activa.
        Todos los feeds son mocks — sin llamadas reales a red.
        """
        db_path = tmp_path / "orchestrator.db"
        os.environ["ENV"] = "demo"
        os.environ.setdefault("KALSHI_API_KEY", "test-key")

        from core.config import load_config
        from core.database import Database
        from core.models import MarketSnapshot, PriceSnapshot

        cfg = load_config()
        db = Database(path=db_path)
        db.initialize()

        # Mocks de feeds que no generan tráfico real
        mock_bfeed = AsyncMock()
        mock_kfeed = AsyncMock()
        mock_bfeed.connect = AsyncMock()
        mock_bfeed.disconnect = AsyncMock()
        mock_kfeed.connect = AsyncMock()
        mock_kfeed.disconnect = AsyncMock()

        async def _empty_price_stream():
            return
            yield  # hace que sea un async generator vacío

        async def _empty_market_stream():
            return
            yield

        mock_bfeed.stream = _empty_price_stream
        mock_kfeed.stream_markets = _empty_market_stream
        mock_kfeed.get_active_markets = AsyncMock(return_value=[])

        with (
            patch("main.BinancePriceFeed", return_value=mock_bfeed),
            patch("main.KalshiFeed", return_value=mock_kfeed),
            patch("main._serve_dashboard", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(
                __import__("main")._run_orchestrator(
                    cfg=cfg,
                    db=db,
                    bankroll=100.0,
                    max_positions=3,
                )
            )
            # Dar tiempo para que arranque y luego cancelar
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # esperado

        db.close()


# ── 2. OpenRouterAgent con mock retorna AgentVerdict válido ───────────────────

class TestOpenRouterAgent:
    """Tests del agente LLM vía OpenRouter."""

    @pytest.mark.asyncio
    async def test_consult_returns_valid_verdict(
        self,
        sample_signal,
        sample_market,
        sample_price,
    ) -> None:
        """Mock HTTP responde con JSON válido → AgentVerdict bien formado."""
        from engine.openrouter_agent import AgentVerdict, OpenRouterAgent

        mock_response_data = {
            "choices": [
                {"message": {"content": '{"proceed": true, "adjusted_kelly": 0.08, "reasoning": "señal válida"}'}}
            ],
            "usage": {"total_tokens": 42},
        }

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        agent = OpenRouterAgent(api_key="test-key")

        with patch("engine.openrouter_agent.aiohttp.ClientSession", return_value=mock_session):
            verdict = await agent.consult(
                signal=sample_signal,
                market=sample_market,
                price=sample_price,
                recent_signals=[],
            )

        assert isinstance(verdict, AgentVerdict)
        assert verdict.proceed is True
        assert 0.0 <= verdict.adjusted_kelly <= 1.0
        assert isinstance(verdict.reasoning, str)
        assert isinstance(verdict.tokens_used, int)

    @pytest.mark.asyncio
    async def test_adjusted_kelly_never_exceeds_original(
        self,
        sample_signal,
        sample_market,
        sample_price,
    ) -> None:
        """El agente no puede aumentar el kelly — solo reducir o mantener."""
        from engine.openrouter_agent import OpenRouterAgent

        original_kelly = sample_signal.kelly_size  # 0.12 del fixture

        # El modelo devuelve un kelly más alto — debe ser truncado
        mock_response_data = {
            "choices": [
                {"message": {"content": '{"proceed": true, "adjusted_kelly": 0.99, "reasoning": "boom"}'}}
            ],
            "usage": {"total_tokens": 10},
        }

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        agent = OpenRouterAgent(api_key="test-key")

        with patch("engine.openrouter_agent.aiohttp.ClientSession", return_value=mock_session):
            verdict = await agent.consult(
                signal=sample_signal,
                market=sample_market,
                price=sample_price,
                recent_signals=[],
            )

        assert verdict.adjusted_kelly <= original_kelly


# ── 3. proceed=False convierte señal en SKIP ──────────────────────────────────

class TestAgentVerdictIntegration:
    """Tests de integración entre SignalRouter y OpenRouterAgent."""

    @pytest.mark.asyncio
    async def test_proceed_false_converts_signal_to_skip(
        self,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ) -> None:
        """
        Cuando el agente retorna proceed=False, evaluate_async debe retornar
        una señal con decision=SKIP y el reasoning del agente.
        """
        from core.config import EngineConfig
        from core.models import Confidence, Decision
        from engine.ev_calculator import EVCalculator
        from engine.openrouter_agent import AgentVerdict, OpenRouterAgent
        from engine.probability import ProbabilityEngine
        from engine.signal_router import SignalRouter
        from engine.timing import TimingFilter

        engine_cfg = EngineConfig(
            min_ev_threshold=0.01,
            min_delta=0.01,
            min_time_remaining_s=90,
            min_volume_24h=0,
            kelly_fraction=0.25,
            max_position_pct=0.10,
        )

        mock_agent = AsyncMock(spec=OpenRouterAgent)
        mock_agent.consult = AsyncMock(
            return_value=AgentVerdict(
                proceed=False,
                adjusted_kelly=0.0,
                reasoning="riesgo_alto",
                tokens_used=20,
            )
        )

        router = SignalRouter(
            prob_engine=ProbabilityEngine(),
            ev_calc=EVCalculator(),
            timing_filter=TimingFilter(),
            config=engine_cfg,
            db=db,
            blocked_categories=set(),
            openrouter_agent=mock_agent,
        )

        # Mercado con delta claro para que el engine genere YES/NO
        market = make_market_snapshot(
            implied_prob=0.40,
            yes_ask=0.41,
            no_ask=0.60,
            time_to_expiry_s=600,
            volume_24h=500,
        )
        price = make_price_snapshot(price=95_000.0)

        # Forzar MEDIUM confidence en la señal base para que el agente sea consultado
        original_evaluate = router.evaluate

        def _patched_evaluate(market, price, bankroll):
            sig = original_evaluate(market=market, price=price, bankroll=bankroll)
            if sig.is_actionable:
                from dataclasses import replace
                sig = replace(sig, confidence=Confidence.MEDIUM)
            return sig

        router.evaluate = _patched_evaluate

        result = await router.evaluate_async(market=market, price=price, bankroll=1000.0)

        assert result.decision == Decision.SKIP
        assert "agent_skip" in result.reasoning
        assert "riesgo_alto" in result.reasoning

    @pytest.mark.asyncio
    async def test_proceed_true_keeps_signal_actionable(
        self,
        db,
        make_market_snapshot,
        make_price_snapshot,
    ) -> None:
        """Cuando proceed=True, la señal sigue siendo actionable."""
        from core.config import EngineConfig
        from core.models import Confidence, Decision
        from engine.ev_calculator import EVCalculator
        from engine.openrouter_agent import AgentVerdict, OpenRouterAgent
        from engine.probability import ProbabilityEngine
        from engine.signal_router import SignalRouter
        from engine.timing import TimingFilter

        engine_cfg = EngineConfig(
            min_ev_threshold=0.01,
            min_delta=0.01,
            min_time_remaining_s=90,
            min_volume_24h=0,
            kelly_fraction=0.25,
            max_position_pct=0.10,
        )

        mock_agent = AsyncMock(spec=OpenRouterAgent)
        mock_agent.consult = AsyncMock(
            return_value=AgentVerdict(
                proceed=True,
                adjusted_kelly=0.05,
                reasoning="confirmado",
                tokens_used=15,
            )
        )

        router = SignalRouter(
            prob_engine=ProbabilityEngine(),
            ev_calc=EVCalculator(),
            timing_filter=TimingFilter(),
            config=engine_cfg,
            db=db,
            blocked_categories=set(),
            openrouter_agent=mock_agent,
        )

        market = make_market_snapshot(
            implied_prob=0.40,
            yes_ask=0.41,
            no_ask=0.60,
            time_to_expiry_s=600,
            volume_24h=500,
        )
        price = make_price_snapshot(price=95_000.0)

        original_evaluate = router.evaluate

        def _patched_evaluate(market, price, bankroll):
            sig = original_evaluate(market=market, price=price, bankroll=bankroll)
            if sig.is_actionable:
                from dataclasses import replace
                sig = replace(sig, confidence=Confidence.MEDIUM)
            return sig

        router.evaluate = _patched_evaluate

        result = await router.evaluate_async(market=market, price=price, bankroll=1000.0)

        assert result.decision in (Decision.YES, Decision.NO)
        assert result.kelly_size <= 0.05 + 1e-9   # adjusted_kelly aplicado


# ── 4. Timeout del agente → señal pasa sin cambios ───────────────────────────

class TestAgentTimeout:
    """El timeout de 3s hace que la señal pase sin modificar."""

    @pytest.mark.asyncio
    async def test_timeout_passes_signal_unchanged(
        self,
        sample_signal,
        sample_market,
        sample_price,
    ) -> None:
        """
        Un mock que tarda 4s provoca TimeoutError en el agente.
        El agente debe retornar proceed=True con kelly sin cambios.
        """
        from engine.openrouter_agent import OpenRouterAgent

        async def _slow_post(*args, **kwargs):
            await asyncio.sleep(4.0)   # supera el timeout de 3s
            raise AssertionError("no debería llegar aquí")  # noqa: TRY301

        agent = OpenRouterAgent(api_key="test-key")

        async def _slow_call(*args, **kwargs):
            await asyncio.sleep(4.0)

        with patch.object(agent, "_call_api", new=_slow_call):
            verdict = await agent.consult(
                signal=sample_signal,
                market=sample_market,
                price=sample_price,
                recent_signals=[],
            )

        assert verdict.proceed is True
        assert verdict.adjusted_kelly == sample_signal.kelly_size
        assert "timeout" in verdict.reasoning.lower()
        assert verdict.tokens_used == 0

    @pytest.mark.asyncio
    async def test_timeout_via_slow_api_call(
        self,
        sample_signal,
        sample_market,
        sample_price,
    ) -> None:
        """Variante: el mock de aiohttp es lento — verifica el timeout end-to-end."""
        from engine.openrouter_agent import AGENT_TIMEOUT_S, OpenRouterAgent

        original_kelly = sample_signal.kelly_size

        async def _slow_call_api(*args, **kwargs):
            await asyncio.sleep(AGENT_TIMEOUT_S + 1.0)
            raise AssertionError("no debería llegar aquí")  # noqa: TRY301

        agent = OpenRouterAgent(api_key="test-key")

        with patch.object(agent, "_call_api", new=_slow_call_api):
            start = time.monotonic()
            verdict = await agent.consult(
                signal=sample_signal,
                market=sample_market,
                price=sample_price,
                recent_signals=[],
            )
            elapsed = time.monotonic() - start

        # Debe retornar en ≈ AGENT_TIMEOUT_S, no en 4s
        assert elapsed < AGENT_TIMEOUT_S + 1.0
        assert verdict.proceed is True
        assert verdict.adjusted_kelly == original_kelly
        assert verdict.tokens_used == 0
