"""
engine/ev_calculator.py

Expected value neto de fees para contratos binarios de Kalshi.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EVResult:
    """
    Resultado del cálculo de expected value para un trade.

    Attributes:
        ev_gross: expected value antes de fees.
        fee_total: fees totales del trade.
        ev_net: expected value neto después de fees.
        is_profitable: True si el trade tiene EV neto positivo.
        min_prob_to_profit: probabilidad mínima para tener EV neto positivo.
    """

    ev_gross: float
    fee_total: float
    ev_net: float
    is_profitable: bool
    min_prob_to_profit: float


class EVCalculator:
    """Calcula EV neto de fees y sizing Kelly para contratos Kalshi."""

    FEE_RATE = 0.07

    def calculate(
        self,
        my_prob: float,
        contract_price: float,
        contracts: int,
        bankroll: float,
    ) -> EVResult:
        """
        Calcula el EV esperado neto de fees para una posición.

        Args:
            my_prob: probabilidad propia estimada de payoff del contrato.
            contract_price: precio actual del contrato en Kalshi.
            contracts: número de contratos a comprar.
            bankroll: capital total disponible en USD.

        Returns:
            EVResult con EV bruto, fees, EV neto y umbral de rentabilidad.

        Raises:
            ValueError: si cualquier parámetro está fuera de rango.
        """

        self._validate_inputs(
            my_prob=my_prob,
            contract_price=contract_price,
            contracts=contracts,
            bankroll=bankroll,
        )

        fee_per_contract = self.fee_per_contract(contract_price)
        fee_total = fee_per_contract * contracts
        capital_at_risk = contract_price * contracts
        # ev_gross y ev_net como fracción del capital arriesgado,
        # para que sean comparables con min_ev_threshold (diseñado como 4 %).
        ev_gross = (my_prob - contract_price) / contract_price
        fee_fraction = fee_total / capital_at_risk   # = fee_per_contract / contract_price
        ev_net = ev_gross - fee_fraction
        min_prob_to_profit = min(1.0, contract_price + fee_per_contract)

        return EVResult(
            ev_gross=ev_gross,
            fee_total=fee_total,
            ev_net=ev_net,
            is_profitable=ev_net > 0.0,
            min_prob_to_profit=min_prob_to_profit,
        )

    def kelly_size(
        self,
        my_prob: float,
        contract_price: float,
        kelly_fraction: float = 0.25,
        max_pct: float = 0.05,
    ) -> float:
        """
        Calcula el tamaño Kelly fraccionario como fracción del bankroll.

        Args:
            my_prob: probabilidad propia estimada.
            contract_price: precio del contrato.
            kelly_fraction: fracción del Kelly completo a usar.
            max_pct: tope máximo del bankroll por trade.

        Returns:
            Fracción del bankroll a apostar en rango [0.0, max_pct].

        Raises:
            ValueError: si los inputs están fuera de rango.
        """

        if not 0.0 <= my_prob <= 1.0:
            raise ValueError(f"my_prob fuera de rango [0,1]: {my_prob}")
        if not 0.0 <= contract_price < 1.0:
            raise ValueError(
                f"contract_price debe estar en [0,1), recibido: {contract_price}"
            )
        if kelly_fraction <= 0.0:
            raise ValueError(f"kelly_fraction debe ser positivo: {kelly_fraction}")
        if max_pct <= 0.0:
            raise ValueError(f"max_pct debe ser positivo: {max_pct}")

        full_kelly = (my_prob - contract_price) / (1.0 - contract_price)
        if full_kelly <= 0.0:
            return 0.0

        sized = full_kelly * kelly_fraction
        return max(0.0, min(sized, max_pct))

    @classmethod
    def fee_per_contract(cls, contract_price: float) -> float:
        """
        Retorna el fee por contrato usando el schedule real de Kalshi.

        Args:
            contract_price: precio del contrato en rango [0, 1].

        Returns:
            Fee por contrato en USD.
        """

        if not 0.0 <= contract_price <= 1.0:
            raise ValueError(
                f"contract_price debe estar en rango [0,1], recibido: {contract_price}"
            )
        return contract_price * (1.0 - contract_price) * cls.FEE_RATE

    @staticmethod
    def _validate_inputs(
        *,
        my_prob: float,
        contract_price: float,
        contracts: int,
        bankroll: float,
    ) -> None:
        """Valida inputs del cálculo principal."""

        if not 0.0 <= my_prob <= 1.0:
            raise ValueError(f"my_prob fuera de rango [0,1]: {my_prob}")
        if not 0.0 <= contract_price <= 1.0:
            raise ValueError(
                f"contract_price debe estar en rango [0,1], recibido: {contract_price}"
            )
        if contracts <= 0:
            raise ValueError(f"contracts debe ser positivo, recibido: {contracts}")
        if bankroll <= 0.0:
            raise ValueError(f"bankroll debe ser positivo, recibido: {bankroll}")
