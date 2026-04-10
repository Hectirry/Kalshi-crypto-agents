"""
core/config.py

Carga y valida toda la configuración del sistema.
Única fuente de verdad para parámetros operacionales.

Reglas:
- Secrets SOLO desde variables de entorno → nunca desde config.json
- En modo demo, las API keys de exchanges externos son opcionales
- Falla rápido al arrancar si falta algo crítico
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Secciones de configuración ──────────────────────────────────────────────

@dataclass(frozen=True)
class KalshiConfig:
    api_key_id: str
    private_key_path: str | None
    base_url: str
    ws_url: str
    demo_mode: bool
    # Compatibilidad legacy mientras migramos completamente a RSA-PSS.
    api_key: str | None = None
    api_secret: str | None = None


@dataclass(frozen=True)
class BinanceConfig:
    api_key: str | None   # None en demo
    ws_url:  str
    symbols: list[str]    # ej: ["BTCUSDT", "ETHUSDT"]


@dataclass(frozen=True)
class HyperliquidConfig:
    api_key: str | None   # None en demo
    base_url: str
    symbols: list[str]


@dataclass(frozen=True)
class EngineConfig:
    min_ev_threshold:       float   # EV mínimo neto de fees para entrar (default: 0.04)
    min_delta:              float   # Delta mínimo my_prob - market_prob (default: 0.05)
    min_time_remaining_s:   int     # No entrar si quedan menos de N segundos (default: 90)
    min_volume_24h:         int     # Liquidez mínima en contratos (default: 100)
    kelly_fraction:         float   # Fracción del Kelly completo (default: 0.25)
    max_position_pct:       float   # Máximo % del bankroll por trade (default: 0.05)


@dataclass(frozen=True)
class DatabaseConfig:
    path:            Path
    wal_mode:        bool = True     # WAL para escrituras concurrentes
    busy_timeout_ms: int  = 5000


@dataclass(frozen=True)
class AppConfig:
    env:         str                  # "demo" | "production"
    log_level:   str
    kalshi:      KalshiConfig
    binance:     BinanceConfig
    hyperliquid: HyperliquidConfig
    engine:      EngineConfig
    database:    DatabaseConfig

    @property
    def is_demo(self) -> bool:
        return self.env == "demo"

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_config(config_file: str | Path | None = None) -> AppConfig:
    """
    Carga la configuración combinando config.json + variables de entorno.

    Las variables de entorno siempre tienen prioridad sobre config.json.
    Los secrets (API keys) SOLO se aceptan desde env vars.

    Args:
        config_file: ruta al config.json. Si es None, busca en:
                     1. $CONFIG_PATH
                     2. ./config.json
                     3. /root/Kalshi Trading/config.json

    Returns:
        AppConfig validado y listo para usar.

    Raises:
        EnvironmentError: si falta una variable requerida en modo production.
        FileNotFoundError: si no se encuentra config.json.
        ValueError: si un valor tiene formato incorrecto.
    """
    raw = _load_json(config_file)
    env = os.getenv("ENV", raw.get("env", "demo")).lower()

    if env not in ("demo", "production"):
        raise ValueError(f"ENV debe ser 'demo' o 'production', recibido: '{env}'")

    is_production = env == "production"

    # ── Kalshi (compat dual: RSA-PSS canónico + Bearer legacy) ───────────────
    kalshi_key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    kalshi_private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip() or None

    # Fallback legacy
    legacy_api_key = os.getenv("KALSHI_API_KEY", "").strip() or None
    legacy_api_secret = os.getenv("KALSHI_API_SECRET", "").strip() or None

    if not kalshi_key_id and legacy_api_key:
        kalshi_key_id = legacy_api_key
    if kalshi_private_key_path is None and legacy_api_secret and Path(legacy_api_secret).exists():
        kalshi_private_key_path = legacy_api_secret

    if not kalshi_key_id:
        raise EnvironmentError(
            "Falta Kalshi key. Define KALSHI_API_KEY_ID (recomendado) o KALSHI_API_KEY."
        )

    kalshi_cfg = raw.get("kalshi", {})
    demo_flag  = not is_production

    kalshi = KalshiConfig(
        api_key_id = kalshi_key_id,
        private_key_path = kalshi_private_key_path,
        base_url = kalshi_cfg.get(
            "base_url",
            "https://api.elections.kalshi.com/trade-api/v2" if is_production
            else "https://demo-api.kalshi.co/trade-api/v2",
        ),
        ws_url = kalshi_cfg.get(
            "ws_url",
            "wss://api.elections.kalshi.com/trade-api/ws/v2" if is_production
            else "wss://demo-api.kalshi.co/trade-api/ws/v2",
        ),
        demo_mode = demo_flag,
        api_key = legacy_api_key,
        api_secret = legacy_api_secret,
    )

    if is_production and kalshi.private_key_path is None:
        raise EnvironmentError(
            "KALSHI_PRIVATE_KEY_PATH es requerida en modo production "
            "para autenticacion RSA-PSS."
        )

    # ── Binance (opcional en demo) ────────────────────────────────────────────
    binance_key = os.getenv("BINANCE_API_KEY")
    if is_production and binance_key is None:
        raise EnvironmentError("BINANCE_API_KEY es requerida en modo production")

    binance_cfg = raw.get("binance", {})
    binance = BinanceConfig(
        api_key = binance_key,
        ws_url  = binance_cfg.get("ws_url", "wss://stream.binance.com:9443/ws"),
        symbols = binance_cfg.get("symbols", ["BTCUSDT", "ETHUSDT"]),
    )

    # ── Hyperliquid (opcional en demo) ────────────────────────────────────────
    hl_key = os.getenv("HYPERLIQUID_API_KEY")
    if is_production and hl_key is None:
        raise EnvironmentError("HYPERLIQUID_API_KEY es requerida en modo production")

    hl_cfg = raw.get("hyperliquid", {})
    hyperliquid = HyperliquidConfig(
        api_key  = hl_key,
        base_url = hl_cfg.get("base_url", "https://api.hyperliquid.xyz"),
        symbols  = hl_cfg.get("symbols", ["BTC", "ETH"]),
    )

    # ── Engine ────────────────────────────────────────────────────────────────
    eng_cfg = raw.get("engine", {})
    engine = EngineConfig(
        min_ev_threshold     = float(eng_cfg.get("min_ev_threshold", 0.04)),
        min_delta            = float(eng_cfg.get("min_delta", 0.05)),
        min_time_remaining_s = int(eng_cfg.get("min_time_remaining_s", 90)),
        min_volume_24h       = int(eng_cfg.get("min_volume_24h", 100)),
        kelly_fraction       = float(eng_cfg.get("kelly_fraction", 0.25)),
        max_position_pct     = float(eng_cfg.get("max_position_pct", 0.05)),
    )

    _validate_engine_config(engine)

    # ── Database ──────────────────────────────────────────────────────────────
    db_path_str = os.getenv(
        "DB_PATH",
        raw.get("database", {}).get("path", "./data/trading.db"),
    )
    db_path = Path(db_path_str)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    database = DatabaseConfig(
        path            = db_path,
        wal_mode        = raw.get("database", {}).get("wal_mode", True),
        busy_timeout_ms = int(raw.get("database", {}).get("busy_timeout_ms", 5000)),
    )

    log_level = os.getenv("LOG_LEVEL", raw.get("log_level", "INFO")).upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"LOG_LEVEL inválido: {log_level}")

    config = AppConfig(
        env         = env,
        log_level   = log_level,
        kalshi      = kalshi,
        binance     = binance,
        hyperliquid = hyperliquid,
        engine      = engine,
        database    = database,
    )

    logger.info(
        "Configuración cargada",
        extra={"env": env, "db_path": str(db_path), "log_level": log_level},
    )
    return config


# ─── Helpers privados ─────────────────────────────────────────────────────────

def _require_env(key: str) -> str:
    """
    Lee una variable de entorno requerida.

    Raises:
        EnvironmentError: si la variable no está definida o está vacía.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Variable de entorno requerida no encontrada: {key}\n"
            f"Agrega {key}=<valor> a tu archivo .env o al entorno del sistema."
        )
    return value


def _load_json(config_file: str | Path | None) -> dict:
    """Carga config.json desde la ruta indicada o la busca en ubicaciones default."""
    candidates: list[Path] = []

    if config_file is not None:
        candidates.append(Path(config_file))
    else:
        env_path = os.getenv("CONFIG_PATH")
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend([
            Path("config.json"),
            Path("/root/Kalshi Trading/config.json"),
        ])

    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug("config.json cargado desde %s", path)
                return data
            except json.JSONDecodeError as exc:
                raise ValueError(f"config.json inválido en {path}: {exc}") from exc

    logger.warning(
        "config.json no encontrado en ninguna ruta candidata → usando solo env vars"
    )
    return {}


def _validate_engine_config(cfg: EngineConfig) -> None:
    """Valida rangos y consistencia interna de la configuración del engine."""
    errors: list[str] = []

    if not (0.0 < cfg.min_ev_threshold < 1.0):
        errors.append(f"min_ev_threshold debe estar en (0, 1): {cfg.min_ev_threshold}")
    if not (0.0 < cfg.min_delta < 1.0):
        errors.append(f"min_delta debe estar en (0, 1): {cfg.min_delta}")
    if cfg.min_time_remaining_s < 30:
        errors.append(f"min_time_remaining_s muy bajo (mín 30s): {cfg.min_time_remaining_s}")
    if not (0.0 < cfg.kelly_fraction <= 1.0):
        errors.append(f"kelly_fraction debe estar en (0, 1]: {cfg.kelly_fraction}")
    if not (0.0 < cfg.max_position_pct <= 0.25):
        errors.append(f"max_position_pct demasiado alto (máx 25%): {cfg.max_position_pct}")

    if errors:
        raise ValueError("Configuración de engine inválida:\n" + "\n".join(f"  - {e}" for e in errors))
