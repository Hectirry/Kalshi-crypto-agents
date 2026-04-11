"""
backtesting/param_injector.py

Calibración simple de umbrales del engine desde histórico de señales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.database import Database
from core.models import Decision, Outcome, Signal
from engine.ev_calculator import EVCalculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParamCalibration:
    """Resultado de calibración de un parámetro histórico."""

    param_key: str
    param_value: float
    category: str | None
    win_rate: float
    sample_size: int


class ParamInjector:
    """Selecciona e inyecta umbrales calibrados en SQLite."""

    def __init__(
        self,
        db: Database,
        delta_candidates: tuple[float, ...] = (0.05, 0.08, 0.10, 0.12, 0.15),
        ev_candidates: tuple[float, ...] = (0.04, 0.06, 0.10, 0.20),
        min_calibration_samples: int = 1,
    ) -> None:
        """
        Inicializa el inyector de parámetros.

        Args:
            db: base de datos del proyecto.
            delta_candidates: candidatos de ``min_delta`` a evaluar.
            ev_candidates: candidatos de ``min_ev_threshold`` a evaluar.
            min_calibration_samples: muestra mínima para considerar un candidato
                válido. Candidatos con menos muestras se tratan como vacíos y no
                se seleccionan. Con el default=1 se mantiene la conducta anterior.
                Para producción se recomienda un valor de 5 o más para evitar
                sobreajuste a rachas cortas de suerte.
        """

        self.db = db
        self.delta_candidates = delta_candidates
        self.ev_candidates = ev_candidates
        self.min_calibration_samples = min_calibration_samples
        self.ev_calculator = EVCalculator()

    def calibrate(
        self,
        from_ts: float,
        to_ts: float,
        categories: set[str] | None = None,
    ) -> list[ParamCalibration]:
        """
        Calibra parámetros y persiste los ganadores.

        Args:
            from_ts: inicio del rango.
            to_ts: fin del rango.
            categories: categorías opcionales a evaluar.

        Returns:
            Lista de calibraciones persistidas.
        """

        results: list[ParamCalibration] = []
        target_categories = categories or self._infer_categories(from_ts=from_ts, to_ts=to_ts)

        if not target_categories:
            target_categories = {None}

        for category in sorted(target_categories, key=lambda value: "" if value is None else value):
            signals = self._load_actionable_signals(from_ts=from_ts, to_ts=to_ts, category=category)
            if not signals:
                logger.info(
                    "param_calibration_skipped category=%s reason=no_signals",
                    category or "ALL",
                )
                continue

            best_delta = self._best_threshold(
                signals=signals,
                candidates=self.delta_candidates,
                selector=lambda signal, threshold: abs(signal.delta) >= threshold,
            )
            best_ev = self._best_threshold(
                signals=signals,
                candidates=self.ev_candidates,
                selector=lambda signal, threshold: signal.ev_net_fees >= threshold,
            )

            delta_result = ParamCalibration(
                param_key="min_delta",
                param_value=best_delta[0],
                category=category,
                win_rate=best_delta[1],
                sample_size=best_delta[2],
            )
            ev_result = ParamCalibration(
                param_key="min_ev_threshold",
                param_value=best_ev[0],
                category=category,
                win_rate=best_ev[1],
                sample_size=best_ev[2],
            )

            for calibration in (delta_result, ev_result):
                self.db.upsert_param(
                    key=calibration.param_key,
                    value=calibration.param_value,
                    category=calibration.category,
                    win_rate=calibration.win_rate,
                    sample_size=calibration.sample_size,
                )
                results.append(calibration)
                logger.info(
                    "param_calibrated key=%s category=%s value=%.4f win_rate=%.4f sample=%s",
                    calibration.param_key,
                    calibration.category or "ALL",
                    calibration.param_value,
                    calibration.win_rate,
                    calibration.sample_size,
                )

        return results

    def _infer_categories(self, from_ts: float, to_ts: float) -> set[str]:
        """Infiere categorías presentes desde los tickers históricos."""

        signals = self.db.get_signals(from_ts=from_ts, to_ts=to_ts)
        categories: set[str] = set()
        for signal in signals:
            upper_ticker = signal.market_ticker.upper()
            for category in ("BTC", "ETH", "SOL"):
                if category in upper_ticker:
                    categories.add(category)
        return categories

    def _load_actionable_signals(
        self,
        from_ts: float,
        to_ts: float,
        category: str | None,
    ) -> list[Signal]:
        """Carga señales YES/NO con outcome conocido."""

        signals = self.db.get_signals(from_ts=from_ts, to_ts=to_ts, category=category)
        actionable = [
            signal
            for signal in signals
            if signal.decision in (Decision.YES, Decision.NO) and signal.outcome is not None
        ]
        deduped: list[Signal] = []
        seen_tickers: set[str] = set()
        for signal in actionable:
            if signal.market_ticker in seen_tickers:
                continue
            seen_tickers.add(signal.market_ticker)
            deduped.append(signal)
        return deduped

    def _best_threshold(
        self,
        signals: list[Signal],
        candidates: tuple[float, ...],
        selector,
    ) -> tuple[float, float, int]:
        """
        Elige el threshold con mejor score histórico.

        Candidatos con menos de ``min_calibration_samples`` señales se tratan
        como vacíos (avg_pnl = -inf) para evitar sobreajuste a muestras pequeñas.
        """

        ranked: list[tuple[float, float, float, int]] = []
        for threshold in candidates:
            subset = [signal for signal in signals if selector(signal, threshold)]
            sample_size = len(subset)
            if sample_size < max(1, self.min_calibration_samples):
                ranked.append((threshold, float("-inf"), -1.0, 0))
                continue

            wins = sum(1 for signal in subset if self._signal_won(signal))
            win_rate = wins / sample_size
            avg_pnl = sum(self._signal_realized_pnl(signal) for signal in subset) / sample_size
            ranked.append((threshold, avg_pnl, win_rate, sample_size))

        ranked.sort(key=lambda item: (item[1], item[2], item[3], item[0]), reverse=True)
        best_threshold, best_avg_pnl, best_win_rate, best_sample = ranked[0]
        if best_avg_pnl == float("-inf"):
            return candidates[0], 0.0, 0
        return best_threshold, best_win_rate, best_sample

    @staticmethod
    def _signal_won(signal: Signal) -> bool:
        """Evalúa si una señal histórica terminó acertando."""

        if signal.outcome is None:
            return False
        return signal.outcome == Outcome.WIN

    @staticmethod
    def _effective_contract_price(signal: Signal) -> float:
        """
        Retorna el precio de entrada real de la señal para cálculo de PnL.

        Usa ``signal.contract_price`` (precio ask efectivo guardado desde v2)
        cuando está disponible. Si no, cae al precio implícito del mercado como
        aproximación compatible con señales antiguas.

        Espejo de ``BacktestRunner._contract_price()`` para mantener coherencia
        entre calibración y replay histórico.
        """
        if signal.contract_price is not None:
            return max(0.01, min(signal.contract_price, 0.99))
        if signal.decision == Decision.NO:
            return max(0.01, min(1.0 - signal.market_probability, 0.99))
        return max(0.01, min(signal.market_probability, 0.99))

    def _signal_realized_pnl(self, signal: Signal) -> float:
        """
        Estimación de P&L realizado por contrato para ranking de thresholds.

        Usa el precio de entrada real (ask efectivo) en lugar del precio
        implícito de mercado para que la calibración refleje el coste real de
        cada entrada y no sobrestime la rentabilidad histórica.
        """
        contract_price = self._effective_contract_price(signal)

        won = self._signal_won(signal)
        exit_price = 1.0 if won else 0.0
        entry_fee = self.ev_calculator.fee_per_contract(contract_price)
        exit_fee = self.ev_calculator.fee_per_contract(exit_price)
        gross_pnl = (1.0 - contract_price) if won else -contract_price
        return gross_pnl - entry_fee - exit_fee
