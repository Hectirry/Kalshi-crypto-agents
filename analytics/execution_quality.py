"""
analytics/execution_quality.py

Análisis de calidad de ejecución basado en señales resueltas y trades cerrados.

Sin red, sin dependencias externas.
Todo se calcula desde datos locales de SQLite.

Uso:
    analyzer = ExecutionQualityAnalyzer(db)
    report = analyzer.analyze(limit=500)
    # report.by_category["BTC"].win_rate
    # report.suggested_max_overround_bps  → sugerencia de calibración
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import Database

# ─── Definiciones de buckets ──────────────────────────────────────────────────

# Cada entrada es (lo_inclusive, hi_exclusive, label)
OVERROUND_BUCKETS: list[tuple[float, float, str]] = [
    (0.0,   50.0,  "0-50bps"),
    (50.0,  100.0, "50-100bps"),
    (100.0, 150.0, "100-150bps"),
    (150.0, float("inf"), "150+bps"),
]

DELTA_BUCKETS: list[tuple[float, float, str]] = [
    (0.00, 0.15, "0.00-0.15"),
    (0.15, 0.25, "0.15-0.25"),
    (0.25, 0.40, "0.25-0.40"),
    (0.40, float("inf"), "0.40+"),
]

_KNOWN_CATEGORIES = ("BTC", "ETH", "SOL")


# ─── Helpers de clasificación ─────────────────────────────────────────────────

def _overround_bucket(bps: float | None) -> str:
    """Clasifica un valor de overround en un label de bucket."""
    if bps is None:
        return "unknown"
    for lo, hi, label in OVERROUND_BUCKETS:
        if lo <= bps < hi:
            return label
    return "150+bps"


def _delta_bucket(delta: float | None) -> str:
    """Clasifica el valor absoluto de delta en un label de bucket."""
    if delta is None:
        return "unknown"
    abs_delta = abs(delta)
    for lo, hi, label in DELTA_BUCKETS:
        if lo <= abs_delta < hi:
            return label
    return "0.40+"


def _infer_category(ticker: str) -> str:
    """Infiere la categoría de activo desde el ticker de Kalshi."""
    upper = ticker.upper()
    for cat in _KNOWN_CATEGORIES:
        if cat in upper:
            return cat
    return "OTHER"


# ─── Estructuras de datos ─────────────────────────────────────────────────────

@dataclass
class BucketStats:
    """Métricas agregadas para un grupo (categoría, bucket de overround, etc.)."""

    label: str
    sample_size: int
    win_rate: float            # fracción [0, 1]
    total_pnl: float
    avg_pnl: float
    avg_contract_price: float
    avg_overround_bps: float   # promedio del overround en el bucket
    avg_delta: float           # delta absoluto promedio
    avg_entry_edge_bps: float  # (my_prob - contract_price) * 10000, promedio


@dataclass
class ExecutionQualityReport:
    """
    Informe completo de calidad de ejecución.

    Agrupa métricas por categoría, bucket de overround y bucket de delta.
    Incluye una sugerencia de calibración para max_market_overround_bps basada
    en el primer bucket de overround con avg_pnl negativo.
    """

    by_category: dict[str, BucketStats] = field(default_factory=dict)
    by_overround_bucket: dict[str, BucketStats] = field(default_factory=dict)
    by_delta_bucket: dict[str, BucketStats] = field(default_factory=dict)
    total_resolved: int = 0
    overall_win_rate: float = 0.0
    overall_pnl: float = 0.0
    # None = no hay suficientes datos o todos los buckets son rentables
    suggested_max_overround_bps: float | None = None


# ─── Lógica de agregación ─────────────────────────────────────────────────────

def _aggregate(label: str, rows: list[dict]) -> BucketStats | None:
    """Calcula BucketStats desde una lista de filas de DB."""
    if not rows:
        return None

    total = len(rows)
    wins = sum(1 for r in rows if r.get("outcome") == "WIN")
    pnls = [r["pnl"] for r in rows if r.get("pnl") is not None]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / len(pnls) if pnls else 0.0

    avg_cp = sum(r.get("contract_price") or 0.0 for r in rows) / total
    avg_ob = sum(r.get("market_overround_bps") or 0.0 for r in rows) / total
    avg_d = sum(abs(r.get("delta") or 0.0) for r in rows) / total

    # entry_edge_bps = (my_prob - contract_price) * 10_000
    # mide el edge real pagado en bps respecto a nuestra estimación
    edge_vals = [
        (r.get("my_prob", 0.0) - (r.get("contract_price") or r.get("entry_price") or 0.0)) * 10_000
        for r in rows
        if r.get("contract_price") is not None or r.get("entry_price") is not None
    ]
    avg_entry_edge_bps = sum(edge_vals) / len(edge_vals) if edge_vals else 0.0

    return BucketStats(
        label=label,
        sample_size=total,
        win_rate=wins / total,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        avg_contract_price=avg_cp,
        avg_overround_bps=avg_ob,
        avg_delta=avg_d,
        avg_entry_edge_bps=avg_entry_edge_bps,
    )


def _suggest_overround_threshold(
    by_overround: dict[str, BucketStats],
    bucket_defs: list[tuple[float, float, str]],
    min_samples: int = 3,
) -> float | None:
    """
    Sugiere un techo de max_market_overround_bps basado en histórico local.

    Recorre los buckets de menor a mayor overround. Devuelve el límite inferior
    del primer bucket con avg_pnl negativo (y muestra suficiente). Esto indica
    el punto a partir del cual el overround empieza a erosionar la rentabilidad.

    Devuelve None si todos los buckets rentables o no hay muestra suficiente.
    """
    for lo, _hi, label in sorted(bucket_defs, key=lambda x: x[0]):
        stats = by_overround.get(label)
        if stats is None or stats.sample_size < min_samples:
            continue
        if stats.avg_pnl < 0.0:
            return float(lo)
    return None


# ─── Analizador principal ─────────────────────────────────────────────────────

class ExecutionQualityAnalyzer:
    """
    Calcula métricas de calidad de ejecución desde datos locales de SQLite.

    Sin llamadas de red. Seguro para usar desde cualquier contexto.

    Args:
        db: instancia inicializada de core.database.Database
    """

    def __init__(self, db: "Database") -> None:
        self._db = db

    def analyze(self, limit: int = 500) -> ExecutionQualityReport:
        """
        Construye un ExecutionQualityReport desde señales resueltas con trades.

        Solo analiza señales accionables (YES/NO) con outcome resuelto y trade
        cerrado asociado. Señales sin trade (SKIP/WAIT/ERROR) se excluyen.

        Args:
            limit: máximo de registros a analizar (los más recientes)

        Returns:
            ExecutionQualityReport con desglose por categoría, overround y delta.
        """
        rows = self._db.fetch_resolved_signals_with_trades(limit=limit)
        if not rows:
            return ExecutionQualityReport()

        # Agrupar en los tres ejes
        by_cat: dict[str, list[dict]] = {}
        by_or: dict[str, list[dict]] = {}
        by_d: dict[str, list[dict]] = {}

        for row in rows:
            cat = _infer_category(row["ticker"])
            or_bucket = _overround_bucket(row.get("market_overround_bps"))
            d_bucket = _delta_bucket(row.get("delta"))

            by_cat.setdefault(cat, []).append(row)
            by_or.setdefault(or_bucket, []).append(row)
            by_d.setdefault(d_bucket, []).append(row)

        # Agregar cada grupo
        by_category: dict[str, BucketStats] = {}
        for label, group in by_cat.items():
            stats = _aggregate(label, group)
            if stats:
                by_category[label] = stats

        by_overround_bucket: dict[str, BucketStats] = {}
        for label, group in by_or.items():
            stats = _aggregate(label, group)
            if stats:
                by_overround_bucket[label] = stats

        by_delta_bucket: dict[str, BucketStats] = {}
        for label, group in by_d.items():
            stats = _aggregate(label, group)
            if stats:
                by_delta_bucket[label] = stats

        # Métricas globales
        total = len(rows)
        wins = sum(1 for r in rows if r.get("outcome") == "WIN")
        pnls = [r["pnl"] for r in rows if r.get("pnl") is not None]
        overall_pnl = sum(pnls)
        overall_win_rate = wins / total if total else 0.0

        suggested = _suggest_overround_threshold(by_overround_bucket, OVERROUND_BUCKETS)

        return ExecutionQualityReport(
            by_category=by_category,
            by_overround_bucket=by_overround_bucket,
            by_delta_bucket=by_delta_bucket,
            total_resolved=total,
            overall_win_rate=overall_win_rate,
            overall_pnl=overall_pnl,
            suggested_max_overround_bps=suggested,
        )
