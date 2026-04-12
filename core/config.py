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
class EngineCategoryOverride:
    min_ev_threshold: float | None = None
    min_delta: float | None = None
    min_time_remaining_s: int | None = None
    min_contract_price: float | None = None
    max_contract_price: float | None = None


@dataclass(frozen=True)
class EngineConfig:
    min_ev_threshold:       float   # EV mínimo neto de fees para entrar (default: 0.04)
    min_delta:              float   # Delta mínimo my_prob - market_prob (default: 0.05)
    min_time_remaining_s:   int     # No entrar si quedan menos de N segundos (default: 90)
    min_volume_24h:         int     # Liquidez mínima en contratos (default: 100)
    kelly_fraction:         float   # Fracción del Kelly completo (default: 0.25)
    max_position_pct:       float   # Máximo % del bankroll por trade (default: 0.05)
    min_contract_price:     float = 0.10  # Evita contratos demasiado baratos/caros
    max_contract_price:     float = 0.90
    max_market_overround_bps: float = 150.0
    # Overround por encima del cual se consulta el agente LLM incluso para
    # señales HIGH confidence (spread ancho → liquidez reducida → revisión).
    # Debe ser < max_market_overround_bps para que tenga efecto.
    agent_review_overround_bps: float = 100.0
    setup_quality_gate_enabled: bool = False
    setup_quality_min_samples: int = 3
    setup_quality_min_win_rate: float = 0.40
    setup_quality_history_limit: int = 500
    category_overrides:     dict[str, EngineCategoryOverride] = field(default_factory=dict)


@dataclass(frozen=True)
class DatabaseConfig:
    path:            Path
    wal_mode:        bool = True     # WAL para escrituras concurrentes
    busy_timeout_ms: int  = 5000


@dataclass(frozen=True)
class SocialSentimentConfig:
    enabled: bool = False
    provider: str = "reddit"
    refresh_interval_s: int = 300
    ttl_s: int = 900
    request_timeout_s: float = 5.0
    cache_path: Path = Path("./data/social_sentiment_cache.json")
    supported_assets: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    reddit_subreddits: list[str] = field(
        default_factory=lambda: ["CryptoCurrency", "Bitcoin", "ethtrader", "solana"]
    )
    max_posts_per_asset: int = 25


@dataclass(frozen=True)
class AppConfig:
    env:         str                  # "demo" | "production"
    log_level:   str
    kalshi:      KalshiConfig
    binance:     BinanceConfig
    hyperliquid: HyperliquidConfig
    engine:      EngineConfig
    database:    DatabaseConfig
    social_sentiment: SocialSentimentConfig

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
    raw_overrides = eng_cfg.get("category_overrides", {})
    category_overrides: dict[str, EngineCategoryOverride] = {}
    for category, override in raw_overrides.items():
        if not isinstance(override, dict):
            raise ValueError(f"engine.category_overrides.{category} debe ser un objeto")
        category_overrides[str(category).upper()] = EngineCategoryOverride(
            min_ev_threshold=float(override["min_ev_threshold"]) if "min_ev_threshold" in override else None,
            min_delta=float(override["min_delta"]) if "min_delta" in override else None,
            min_time_remaining_s=int(override["min_time_remaining_s"]) if "min_time_remaining_s" in override else None,
            min_contract_price=float(override["min_contract_price"]) if "min_contract_price" in override else None,
            max_contract_price=float(override["max_contract_price"]) if "max_contract_price" in override else None,
        )
    engine = EngineConfig(
        min_ev_threshold     = float(eng_cfg.get("min_ev_threshold", 0.04)),
        min_delta            = float(eng_cfg.get("min_delta", 0.05)),
        min_time_remaining_s = int(eng_cfg.get("min_time_remaining_s", 90)),
        min_volume_24h       = int(eng_cfg.get("min_volume_24h", 100)),
        kelly_fraction       = float(eng_cfg.get("kelly_fraction", 0.25)),
        max_position_pct     = float(eng_cfg.get("max_position_pct", 0.05)),
        min_contract_price   = float(eng_cfg.get("min_contract_price", 0.10)),
        max_contract_price   = float(eng_cfg.get("max_contract_price", 0.90)),
        max_market_overround_bps = float(eng_cfg.get("max_market_overround_bps", 150.0)),
        agent_review_overround_bps = float(eng_cfg.get("agent_review_overround_bps", 100.0)),
        setup_quality_gate_enabled = bool(eng_cfg.get("setup_quality_gate_enabled", False)),
        setup_quality_min_samples = int(eng_cfg.get("setup_quality_min_samples", 3)),
        setup_quality_min_win_rate = float(eng_cfg.get("setup_quality_min_win_rate", 0.40)),
        setup_quality_history_limit = int(eng_cfg.get("setup_quality_history_limit", 500)),
        category_overrides   = category_overrides,
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

    social_raw = raw.get("social_sentiment", {})
    social_enabled = _env_bool(
        "SOCIAL_SENTIMENT_ENABLED",
        bool(social_raw.get("enabled", False)),
    )
    cache_path = Path(
        os.getenv(
            "SOCIAL_SENTIMENT_CACHE_PATH",
            str(social_raw.get("cache_path", "./data/social_sentiment_cache.json")),
        )
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    social_sentiment = SocialSentimentConfig(
        enabled=social_enabled,
        provider=os.getenv(
            "SOCIAL_SENTIMENT_PROVIDER",
            str(social_raw.get("provider", "reddit")),
        ).strip().lower(),
        refresh_interval_s=int(
            os.getenv(
                "SOCIAL_SENTIMENT_REFRESH_INTERVAL_S",
                social_raw.get("refresh_interval_s", 300),
            )
        ),
        ttl_s=int(os.getenv("SOCIAL_SENTIMENT_TTL_S", social_raw.get("ttl_s", 900))),
        request_timeout_s=float(
            os.getenv(
                "SOCIAL_SENTIMENT_REQUEST_TIMEOUT_S",
                social_raw.get("request_timeout_s", 5.0),
            )
        ),
        cache_path=cache_path,
        supported_assets=_env_csv(
            "SOCIAL_SENTIMENT_ASSETS",
            social_raw.get("supported_assets", ["BTC", "ETH", "SOL"]),
        ),
        reddit_subreddits=_env_csv(
            "SOCIAL_SENTIMENT_REDDIT_SUBREDDITS",
            social_raw.get(
                "reddit_subreddits",
                ["CryptoCurrency", "Bitcoin", "ethtrader", "solana"],
            ),
        ),
        max_posts_per_asset=int(
            os.getenv(
                "SOCIAL_SENTIMENT_MAX_POSTS_PER_ASSET",
                social_raw.get("max_posts_per_asset", 25),
            )
        ),
    )

    _validate_social_sentiment_config(social_sentiment)

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
        social_sentiment = social_sentiment,
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
    if not (0.0 < cfg.min_contract_price < 1.0):
        errors.append(
            f"min_contract_price debe estar en (0, 1): {cfg.min_contract_price}"
        )
    if not (0.0 < cfg.max_contract_price < 1.0):
        errors.append(
            f"max_contract_price debe estar en (0, 1): {cfg.max_contract_price}"
        )
    if cfg.max_market_overround_bps < 0.0:
        errors.append(
            "max_market_overround_bps no puede ser negativo: "
            f"{cfg.max_market_overround_bps}"
        )
    if cfg.agent_review_overround_bps < 0.0:
        errors.append(
            "agent_review_overround_bps no puede ser negativo: "
            f"{cfg.agent_review_overround_bps}"
        )
    if cfg.agent_review_overround_bps >= cfg.max_market_overround_bps:
        errors.append(
            "agent_review_overround_bps debe ser menor que max_market_overround_bps "
            f"para tener efecto: {cfg.agent_review_overround_bps} >= "
            f"{cfg.max_market_overround_bps}"
        )
    if cfg.setup_quality_min_samples < 1:
        errors.append(
            "setup_quality_min_samples debe ser >= 1: "
            f"{cfg.setup_quality_min_samples}"
        )
    if not (0.0 <= cfg.setup_quality_min_win_rate <= 1.0):
        errors.append(
            "setup_quality_min_win_rate debe estar en [0, 1]: "
            f"{cfg.setup_quality_min_win_rate}"
        )
    if cfg.setup_quality_history_limit < 50:
        errors.append(
            "setup_quality_history_limit muy bajo (mín 50): "
            f"{cfg.setup_quality_history_limit}"
        )
    if cfg.min_contract_price >= cfg.max_contract_price:
        errors.append(
            "min_contract_price debe ser menor que max_contract_price: "
            f"{cfg.min_contract_price} >= {cfg.max_contract_price}"
        )
    for category, override in cfg.category_overrides.items():
        if override.min_ev_threshold is not None and not (0.0 < override.min_ev_threshold < 1.0):
            errors.append(
                f"category_overrides.{category}.min_ev_threshold inválido: {override.min_ev_threshold}"
            )
        if override.min_delta is not None and not (0.0 < override.min_delta < 1.0):
            errors.append(f"category_overrides.{category}.min_delta inválido: {override.min_delta}")
        if override.min_time_remaining_s is not None and override.min_time_remaining_s < 30:
            errors.append(
                f"category_overrides.{category}.min_time_remaining_s muy bajo: "
                f"{override.min_time_remaining_s}"
            )
        if override.min_contract_price is not None and not (0.0 < override.min_contract_price < 1.0):
            errors.append(
                f"category_overrides.{category}.min_contract_price inválido: "
                f"{override.min_contract_price}"
            )
        if override.max_contract_price is not None and not (0.0 < override.max_contract_price < 1.0):
            errors.append(
                f"category_overrides.{category}.max_contract_price inválido: "
                f"{override.max_contract_price}"
            )
        min_price = override.min_contract_price if override.min_contract_price is not None else cfg.min_contract_price
        max_price = override.max_contract_price if override.max_contract_price is not None else cfg.max_contract_price
        if min_price >= max_price:
            errors.append(
                f"category_overrides.{category} rango de precio inválido: {min_price} >= {max_price}"
            )

    if errors:
        raise ValueError("Configuración de engine inválida:\n" + "\n".join(f"  - {e}" for e in errors))


def _validate_social_sentiment_config(cfg: SocialSentimentConfig) -> None:
    """Valida la configuración de sentimiento social opcional."""
    errors: list[str] = []

    if cfg.provider not in {"reddit", "dummy"}:
        errors.append(f"social_sentiment.provider inválido: {cfg.provider}")
    if cfg.refresh_interval_s <= 0:
        errors.append(
            f"social_sentiment.refresh_interval_s debe ser > 0: {cfg.refresh_interval_s}"
        )
    if cfg.ttl_s <= 0:
        errors.append(f"social_sentiment.ttl_s debe ser > 0: {cfg.ttl_s}")
    if cfg.request_timeout_s <= 0:
        errors.append(
            "social_sentiment.request_timeout_s debe ser > 0: "
            f"{cfg.request_timeout_s}"
        )
    if cfg.max_posts_per_asset <= 0:
        errors.append(
            "social_sentiment.max_posts_per_asset debe ser > 0: "
            f"{cfg.max_posts_per_asset}"
        )
    if not cfg.supported_assets:
        errors.append("social_sentiment.supported_assets no puede estar vacío")
    if cfg.refresh_interval_s > cfg.ttl_s:
        errors.append(
            "social_sentiment.refresh_interval_s no puede exceder ttl_s: "
            f"{cfg.refresh_interval_s} > {cfg.ttl_s}"
        )

    if errors:
        raise ValueError("Configuración de social sentiment inválida:\n" + "\n".join(f"  - {e}" for e in errors))


def _env_bool(key: str, default: bool) -> bool:
    """Parsea un bool desde env con fallback."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(key: str, default: list[str]) -> list[str]:
    """Parsea una lista CSV desde env con fallback."""
    raw = os.getenv(key)
    if raw is None:
        return [str(item).upper() for item in default]
    return [item.strip().upper() for item in raw.split(",") if item.strip()]
