# CLAUDE.md — Kalshi Crypto Trading Bot

> Archivo de contexto para Claude Code y Codex.
> Leer COMPLETO antes de tocar cualquier archivo del proyecto.

---

## Identidad del proyecto

Sistema de trading automatizado para mercados crypto binarios de 15 minutos en Kalshi.
El edge es estructural: detectar el lag entre el precio real en Binance/Hyperliquid
y la probabilidad implícita del contrato en Kalshi, y entrar cuando el delta justifica
el trade después de fees.

**Repositorio:** `/root/Kalshi-crypto-agents`
**Entorno:** VPS Ubuntu 24, Python 3.11+, servicio systemd `kalshi-bot.service`
**Estado:** Todas las fases completas. Bot listo para arrancar con `python main.py`.

---

## Arquitectura de módulos

```
kalshi_trading/
├── core/
│   ├── interfaces.py        # Protocolos abstractos (PriceFeed, MarketScanner, OrderExecutor)
│   ├── models.py            # Dataclasses canónicas (Signal, Trade, MarketSnapshot)
│   ├── config.py            # Carga config desde env vars + config.json (sin secrets)
│   └── database.py          # SQLite — única fuente de verdad, todas las escrituras aquí
├── feeds/
│   ├── binance_feed.py      # BinancePriceFeed: WS async BTC/ETH ticks
│   ├── hyperliquid_feed.py  # HyperliquidFeed: orderbook depth + funding rate
│   └── kalshi_feed.py       # KalshiFeed: WS precios + orderbook contratos 15min
├── engine/
│   ├── probability.py       # ProbabilityEngine: delta prob Kalshi vs Binance
│   ├── timing.py            # TimingFilter: ventana óptima de entrada por tiempo restante
│   ├── ev_calculator.py     # EVCalculator: EV neto de fees + Kelly sizing
│   └── signal_router.py     # SignalRouter: orquesta los tres módulos → decisión final
├── execution/
│   ├── order_executor.py    # OrderExecutor: demo loguea, prod llama API — mismo código
│   └── position_manager.py  # PositionManager: stop loss, take profit, P&L
├── backtesting/
│   ├── backtest_runner.py   # BacktestRunner: VectorBT sobre SQLite histórico
│   ├── category_blocker.py  # CategoryBlocker: auto-block por win rate histórico
│   └── param_injector.py    # ParamInjector: calibra umbrales → inyecta al agente
├── dashboard/
│   └── api_server.py        # FastAPI: endpoints para dashboard HTML existente
├── tests/
│   ├── conftest.py          # Fixtures globales: DB en memoria, feeds mock, config test
│   ├── test_schema.py       # Fase 1: valida schema SQLite y migraciones
│   ├── test_feeds.py        # Fase 2: feeds con mocks, sin llamadas reales
│   ├── test_engine.py       # Fase 3: ProbabilityEngine, EVCalculator — casos edge
│   ├── test_backtesting.py  # Fase 4: BacktestRunner sobre datos sintéticos
│   └── test_execution.py    # Fase 5: OrderExecutor demo vs prod
├── config.json              # Solo estructura — valores reales en env vars
├── CLAUDE.md                # Este archivo
├── PHASES.md                # Estado de cada fase (actualizar al completar)
└── requirements.txt
```

---

## Convenciones de código — NO negociables

### Estilo
- **snake_case** para todo: variables, funciones, módulos, parámetros
- **PascalCase** solo para clases
- **UPPER_SNAKE_CASE** para constantes en `core/config.py`
- Type hints obligatorios en todas las funciones públicas
- Docstrings en todas las clases públicas (formato Google style)
- Máximo 100 caracteres por línea
- `ruff` como linter — configuración en `pyproject.toml`

### Async
- Todo IO es `async/await`. Sin `time.sleep()` — usar `asyncio.sleep()`
- Los feeds publican en `asyncio.Queue` central — nunca llaman directamente al engine
- Timeout explícito en toda llamada de red: `asyncio.wait_for(coro, timeout=10.0)`

### Errores
- **No `except Exception`** — capturar excepciones específicas siempre
- Los módulos de engine nunca lanzan hacia afuera — retornan `Signal` con `status=ERROR`
- Logging estructurado con `structlog` — cada log incluye `module`, `trade_id` si aplica
- Nivel `ERROR` solo para condiciones que requieren intervención humana

### Seguridad
- **Cero secrets en código o config.json** — solo en variables de entorno
- Variables de entorno requeridas: `KALSHI_API_KEY`, `KALSHI_API_SECRET`,
  `BINANCE_API_KEY`, `HYPERLIQUID_API_KEY`, `ENV` (demo|production)
- `config.py` lanza `EnvironmentError` explícito si falta alguna variable requerida en prod
- En demo, las claves de exchanges externos son opcionales (solo Kalshi requerida)

### Tests
- Framework: `pytest` con `pytest-asyncio`
- **Cero llamadas reales a APIs en tests** — todo mockeado con fixtures en `conftest.py`
- Cobertura mínima de módulos `engine/`: 85%
- Cada función pública tiene al menos: caso normal, caso edge (input extremo), caso error

---

## Modelo de datos canónico

```python
# core/models.py — estas son las estructuras que circulan por todo el sistema

@dataclass
class PriceSnapshot:
    symbol: str          # "BTC" | "ETH"
    price: float         # precio spot en USD
    timestamp: float     # unix timestamp con ms
    source: str          # "binance" | "hyperliquid"
    bid: float | None = None
    ask: float | None = None

@dataclass
class MarketSnapshot:
    ticker: str          # ej: "KXBTC-15MIN-B95000"
    implied_prob: float  # precio del contrato Kalshi (0.0 - 1.0)
    yes_ask: float
    no_ask: float
    volume_24h: int
    time_to_expiry_s: int  # segundos hasta expiración
    timestamp: float

@dataclass
class Signal:
    market_ticker: str
    decision: str        # "YES" | "NO" | "WAIT" | "SKIP" | "ERROR"
    my_probability: float
    market_probability: float
    delta: float         # my_probability - market_probability
    ev_net_fees: float   # EV esperado después de fees Kalshi
    kelly_size: float    # fracción de bankroll recomendada
    confidence: str      # "HIGH" | "MEDIUM" | "LOW"
    time_remaining_s: int
    reasoning: str       # texto libre para logging/debug
    timestamp: float
    error_msg: str | None = None
```

---

## Schema SQLite — tablas críticas

El schema vive en `core/database.py`. Las migraciones son aditivas (nunca DROP).

```sql
-- Tabla maestra de señales — base del backtesting
CREATE TABLE signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    decision      TEXT NOT NULL,          -- YES|NO|WAIT|SKIP|ERROR
    my_prob       REAL NOT NULL,
    market_prob   REAL NOT NULL,
    delta         REAL NOT NULL,
    ev_net_fees   REAL NOT NULL,
    kelly_size    REAL NOT NULL,
    confidence    TEXT NOT NULL,
    time_remaining_s INTEGER NOT NULL,
    reasoning     TEXT,
    created_at    REAL NOT NULL,          -- unix timestamp
    outcome       TEXT,                  -- se rellena al expirar: WIN|LOSS|NULL
    outcome_at    REAL                   -- unix timestamp del outcome
);

-- Trades ejecutados (paper y real)
CREATE TABLE trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id     INTEGER REFERENCES signals(id),
    ticker        TEXT NOT NULL,
    side          TEXT NOT NULL,          -- YES|NO
    contracts     INTEGER NOT NULL,
    entry_price   REAL NOT NULL,
    exit_price    REAL,
    fee_paid      REAL NOT NULL DEFAULT 0,
    pnl           REAL,
    mode          TEXT NOT NULL,          -- demo|production
    status        TEXT NOT NULL,          -- open|closed|cancelled
    opened_at     REAL NOT NULL,
    closed_at     REAL
);

-- Parámetros calibrados por el backtesting
CREATE TABLE backtest_params (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    param_key     TEXT NOT NULL,          -- ej: "min_delta_threshold"
    param_value   REAL NOT NULL,
    category      TEXT,                  -- NULL = global, o "BTC"|"ETH"
    win_rate      REAL,                  -- win rate histórico con este param
    sample_size   INTEGER,
    valid_from    REAL NOT NULL,
    valid_until   REAL                   -- NULL = vigente
);

-- Categorías bloqueadas
CREATE TABLE blocked_categories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    category      TEXT NOT NULL UNIQUE,
    win_rate      REAL NOT NULL,
    sample_size   INTEGER NOT NULL,
    blocked_at    REAL NOT NULL,
    reason        TEXT
);

-- Índices de rendimiento
CREATE INDEX idx_signals_ticker ON signals(ticker);
CREATE INDEX idx_signals_created ON signals(created_at);
CREATE INDEX idx_signals_outcome ON signals(outcome);
CREATE INDEX idx_trades_status ON trades(status);
```

---

## Variables de entorno requeridas

```bash
# .env (NUNCA commitear este archivo)
ENV=demo                          # demo | production
KALSHI_API_KEY=your_key
KALSHI_API_SECRET=your_secret
BINANCE_API_KEY=your_key          # opcional en demo
HYPERLIQUID_API_KEY=your_key      # opcional en demo
OPENROUTER_API_KEY=your_key       # para el agente LLM
LOG_LEVEL=INFO                    # DEBUG | INFO | WARNING | ERROR
DB_PATH=/root/kalshi_trading/data/trading.db
```

---

## Fee schedule Kalshi (octubre 2025)

Crítico para EVCalculator — fees destruyen trades marginales.

```
fee = min(0.07 * contracts, 0.07 * contracts * (1 - |price - 0.5| / 0.5))

Simplificado: fee por contrato = precio * (1 - precio) * 0.07 / (precio * 1)
A precio 0.50 → fee máximo ~$0.0175 por contrato
A precio 0.80 → fee ~$0.0112 por contrato
A precio 0.95 → fee ~$0.0033 por contrato

Regla práctica: delta mínimo necesario para ser rentable:
  min_delta = (2 * fee_per_contract) / (1.0 - fee_per_contract)
```

---

## Reglas del agente de decisión

```
decision = SKIP   si  ev_net_fees <= 0
decision = SKIP   si  time_remaining_s < 90       # menos de 90s → demasiado tarde
decision = SKIP   si  categoria en blocked_categories
decision = WAIT   si  0 < ev_net_fees < MIN_EV_THRESHOLD  # señal débil
decision = WAIT   si  confidence == "LOW"
decision = YES    si  ev_net_fees >= MIN_EV_THRESHOLD AND delta > 0 AND confidence in (HIGH, MEDIUM)
decision = NO     si  ev_net_fees >= MIN_EV_THRESHOLD AND delta < 0 AND confidence in (HIGH, MEDIUM)

MIN_EV_THRESHOLD se calibra por VectorBT — valor inicial: 0.04 (4% EV neto)
```

---

## Arranque rápido

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a && source .env && set +a
python main.py --dry-run           # demo, sin credenciales reales
python main.py                     # usa ENV del .env
python main.py --backtest-only     # solo recalibra y sale
```

Dashboard disponible en `http://localhost:8090/health` mientras el bot corre.

---

## Autenticación RSA-PSS (Kalshi)

Variables de entorno requeridas para firma de requests:

```bash
KALSHI_API_KEY_ID=tu_key_id
KALSHI_PRIVATE_KEY_PATH=/root/Kalshi-crypto-agents/KalshiApiSecret.pem
```

El `KalshiFeed` selecciona automáticamente RSA-PSS si `KALSHI_PRIVATE_KEY_PATH` está
presente, y cae al Bearer legacy si no. En producción, `KALSHI_PRIVATE_KEY_PATH` es
obligatorio o el arranque falla con `EnvironmentError`.

---

## Agente OpenRouter (opcional)

Si `OPENROUTER_API_KEY` está en el entorno, el `SignalRouter` consulta al LLM
para señales con `confidence=MEDIUM`. Timeout estricto de 3s — nunca bloquea.

```bash
OPENROUTER_API_KEY=tu_openrouter_key
```

---

## Estado de fases (actualizar aquí al completar)

| Fase | Módulos | Estado | Puerta de salida |
|------|---------|--------|-----------------|
| 1 | interfaces, models, config, database | COMPLETA | 51/51 tests ✓ |
| 2 | binance_feed, hyperliquid_feed, kalshi_feed | COMPLETA | 100/100 tests ✓ |
| 3 | probability, timing, ev_calculator, signal_router | COMPLETA | 126/126 tests ✓ |
| 4 | backtest_runner, category_blocker, param_injector | COMPLETA | 134/134 tests ✓ |
| 5 | order_executor, observabilidad, GO/NO-GO | COMPLETA | 145/145 tests ✓ |
| 6 | main.py, openrouter_agent, integración | COMPLETA | test_integration ✓ |

---

## Archivos del sistema base (NO borrar, solo extender)

```
kalshi_volume_scanner.py   → migrar lógica a feeds/kalshi_feed.py
whale_following_bot.py     → migrar lógica a execution/
crypto_agent.py            → migrar lógica a engine/signal_router.py
openrouter_client.py       → mantener, usar en engine/
kalshi_scanner_dashboard.html → mantener, conectar a dashboard/api_server.py
```

---

## Comando de inicio rápido para cada sesión

```bash
# Verificar que el servicio base sigue corriendo
systemctl status kalshi-bot.service

# Correr tests de la fase actual
cd /root/kalshi_trading && pytest tests/ -v --tb=short

# Ver logs en tiempo real
journalctl -u kalshi-bot.service -f

# Dashboard
curl http://localhost:8010/status
```

---

## Anti-patrones prohibidos

- `except Exception:` o `except:` sin tipo específico
- `print()` para debug — usar `logger.debug()`
- Secrets en cualquier archivo que no sea `.env`
- `time.sleep()` en código async
- Lógica de negocio en `__main__` o scripts sueltos
- Llamadas a API reales en tests
- `if ENV == "demo":` dispersos — solo en `OrderExecutor`
- Modificar schema SQLite con DROP o ALTER COLUMN — solo agregar columnas/tablas