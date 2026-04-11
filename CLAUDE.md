# CLAUDE.md — Kalshi Crypto Agents

Guía de contexto para agentes de código como Codex, Claude Code y Minimax.

Objetivo: entender rápido cómo funciona el proyecto hoy, qué archivos mandan,
qué invariantes no deben romperse y cuál es el estado operativo actual.

---

## 1. Qué es este proyecto

Bot de trading para mercados crypto binarios cortos de Kalshi.

Hace esto:

- lee precios spot desde Binance y Hyperliquid
- lee mercados crypto desde Kalshi por WS + REST
- estima probabilidad propia vs probabilidad implícita del mercado
- filtra por timing, delta, EV neto de fees y precio del contrato
- abre trades paper o production
- persiste señales, trades, outcomes y parámetros calibrados en SQLite
- recalibra thresholds y bloquea/desbloquea categorías con histórico reciente
- sirve dashboard FastAPI en `:8090`

El proyecto hoy opera sobre `BTC`, `ETH` y `SOL`.

---

## 2. Archivos fuente de verdad

Si hay dudas, estos archivos mandan sobre la documentación vieja:

- [main.py](/root/Kalshi-crypto-agents/main.py): orquestador principal
- [config.json](/root/Kalshi-crypto-agents/config.json): defaults públicos
- [core/config.py](/root/Kalshi-crypto-agents/core/config.py): loader y validación real
- [core/database.py](/root/Kalshi-crypto-agents/core/database.py): schema y acceso SQLite
- [engine/signal_router.py](/root/Kalshi-crypto-agents/engine/signal_router.py): política de entrada real
- [execution/order_executor.py](/root/Kalshi-crypto-agents/execution/order_executor.py): sizing y PnL real
- [execution/position_manager.py](/root/Kalshi-crypto-agents/execution/position_manager.py): gestión de riesgo viva
- [backtesting/backtest_runner.py](/root/Kalshi-crypto-agents/backtesting/backtest_runner.py): replay alineado con ejecución
- [backtesting/param_injector.py](/root/Kalshi-crypto-agents/backtesting/param_injector.py): calibración de thresholds
- [backtesting/category_blocker.py](/root/Kalshi-crypto-agents/backtesting/category_blocker.py): bloqueo/desbloqueo de categorías

Si `CLAUDE.md` contradice esos archivos, el código gana.

---

## 3. Estado operativo actual

### Política global actual

Configuración base real en `config.json`:

- `min_ev_threshold = 0.06`
- `min_delta = 0.12`
- `min_time_remaining_s = 90`
- `max_position_pct = 0.05`
- `kelly_fraction = 0.25`
- `min_contract_price = 0.10`
- `max_contract_price = 0.90`

### Override por categoría

Hoy existe override explícito solo para `BTC`:

- `min_delta = 0.25`
- `min_ev_threshold = 0.30`
- `min_time_remaining_s = 180`
- `min_contract_price = 0.10`
- `max_contract_price = 0.70`

Interpretación:

- `BTC` está habilitado otra vez, pero con filtros bastante más duros
- `ETH` y `SOL` usan la política global más los parámetros calibrados desde DB

### Bloqueos de categoría

La tabla `blocked_categories` se maneja por backtesting.

Regla actual:

- no se bloquea una categoría si la muestra es chica
- no se bloquea una categoría solo por win rate bajo
- para bloquear debe haber `win_rate` bajo y `PnL` negativo

### Estado de la base tras limpieza

Se limpió el churn viejo de trades repetidos por `ticker`.

Script usado:

- [scripts/cleanup_inconsistent_history.py](/root/Kalshi-crypto-agents/scripts/cleanup_inconsistent_history.py)

Último backup creado por esa limpieza:

- `/root/Kalshi-crypto-agents/data/trading.backup-20260411T151755Z.db`

La limpieza eliminó trades incompatibles con la política actual de una sola
entrada por contrato.

---

## 4b. Calibración de parámetros — correcciones de sesgo

### Corrección de precio de entrada en `ParamInjector` (corregido)

`_signal_realized_pnl` usaba `signal.market_probability` como precio de entrada
en lugar de `signal.contract_price` (el precio ask real persistido desde v2).

Efecto del bug original: calibración veía trades históricos como más rentables
de lo que fueron (mid-price < ask real) → thresholds calibrados demasiado laxos
→ más señales pasando filtros → peor PnL en live.

Fix: nuevo método `_effective_contract_price(signal)` que espeja `BacktestRunner._contract_price()`:
usa `signal.contract_price` si existe, cae a `market_probability` para señales pre-v2.

### Muestra mínima para calibración (`min_calibration_samples`)

`ParamInjector` ahora acepta `min_calibration_samples` (default=1, backward compatible).
Candidatos de threshold con menos señales de lo requerido se descartan como si estuvieran vacíos.

Uso recomendado en producción con histórico suficiente:
```python
ParamInjector(db=db, min_calibration_samples=5)
```

Previene sobreajuste a rachas cortas de suerte en muestras de 1-2 trades.

---

## 4a. Analytics de calidad de ejecución

Módulo nuevo: `analytics/execution_quality.py`

Qué hace:

- analiza señales resueltas (outcome IS NOT NULL) con su trade cerrado asociado
- agrupa por categoría (BTC/ETH/SOL), bucket de overround (0-50/50-100/100-150/150+bps) y bucket de delta
- métricas por grupo: `sample_size`, `win_rate`, `total_pnl`, `avg_pnl`, `avg_contract_price`, `avg_overround_bps`, `avg_delta`, `avg_entry_edge_bps`
- `avg_entry_edge_bps` = `(my_prob - contract_price) * 10_000` — mide el edge real pagado vs estimación propia
- `suggested_max_overround_bps` — límite inferior del primer bucket de overround con `avg_pnl < 0` y muestra ≥ 3; útil para recalibrar `max_market_overround_bps` en config

Endpoint nuevo: `GET /analytics/execution-quality?limit=500`

- retorna JSON con `total_resolved`, `overall_win_rate`, `overall_pnl`, `suggested_max_overround_bps`, `by_category`, `by_overround_bucket`, `by_delta_bucket`
- estable y testeado

Interfaz de recalibración:

- `_suggest_overround_threshold(by_overround_bucket, OVERROUND_BUCKETS, min_samples=3)` — retorna el umbral candidato o `None`
- los datos disponibles en `by_overround_bucket` permiten identificar manualmente qué rangos destruyen PnL

DB:

- nuevo método `Database.fetch_resolved_signals_with_trades(limit)` — INNER JOIN signals/trades, solo cerrados y resueltos, sin SQL en otros módulos

---

## 4. Arquitectura real

### Config y modelos

- `core/models.py`
  - `PriceSnapshot`, `MarketSnapshot`, `Signal`, `Trade`
- `core/config.py`
  - carga `config.json` + env vars
  - soporta `EngineCategoryOverride`
  - valida modo `demo|production`
  - exige credenciales de Kalshi siempre
  - exige Binance/Hyperliquid solo en production

### Feeds

- `feeds/binance_feed.py`
- `feeds/hyperliquid_feed.py`
- `feeds/kalshi_feed.py`

### Engine

- `engine/probability.py`
  - estima `my_prob`
  - clasifica zonas temporales `NEAR|MID|FAR`
- `engine/timing.py`
  - decide si vale la pena entrar por tiempo restante
- `engine/ev_calculator.py`
  - calcula EV neto y Kelly
- `engine/price_resolver.py`
  - consolida precio de referencia desde múltiples fuentes
- `engine/signal_router.py`
  - aplica política completa de entrada
- `engine/context_builder.py`
  - arma contexto histórico para el agente LLM
- `engine/openrouter_agent.py`
  - segunda opinión opcional solo para señales `MEDIUM`
- `engine/setup_quality.py`
  - gate local por calidad histórica del mismo setup
  - si hay suficiente muestra resuelta y el win rate histórico es malo, se fuerza `SKIP`

### Inteligencia auxiliar

- `intelligence/social_sentiment.py`
  - servicio opcional en background con caché en memoria + disco
  - nunca hace fetch dentro de `SignalRouter.evaluate_async(...)`
  - si falla la fuente, conserva el último snapshot válido y trading sigue
- `intelligence/reddit_provider.py`
  - provider inicial sobre Reddit JSON
  - resume solo `BTC`, `ETH`, `SOL`
  - entrega métricas compactas, no texto crudo

### Ejecución

- `execution/order_executor.py`
  - paper y production comparten la misma estructura
  - el sizing real es `ceil(signal.kelly_size * contracts_multiplier)`
  - `contracts_multiplier` default = `100`
- `execution/position_manager.py`
  - mantiene `open_positions`
  - mantiene `traded_tickers`
  - evita reentrada del mismo `ticker`
  - aplica SL/TP
  - computa snapshot de observabilidad

### Backtesting y recalibración

- `backtesting/backtest_runner.py`
  - replay de señales persistidas
  - usa el mismo sizing por contratos que el executor
  - deduplica por `ticker`
  - si recibe `config`, replica la política activa del router
- `backtesting/param_injector.py`
  - calibra `min_delta` y `min_ev_threshold`
  - deduplica por `ticker`
  - rankea por `avg_pnl` y luego por `win_rate`
- `backtesting/category_blocker.py`
  - bloquea/desbloquea según replay reciente
- `backtesting/outcome_resolver.py`
  - resuelve `outcome` real de señales expiradas

### Observabilidad y memoria

- `dashboard/api_server.py`
  - dashboard FastAPI + frontend estático
- `memory/openclaw_adapter.py`
  - journal opcional si `OPENCLAW_WORKSPACE` está configurado

---

## 5. Flujo principal

El flujo real vive en [main.py](/root/Kalshi-crypto-agents/main.py).

### Arranque

1. parsea CLI
2. resuelve `paper_trade`
3. carga config
4. abre DB
5. lee bloqueos/params actuales
6. corre `_maybe_recalibrate(...)`
7. arranca el orquestador async

### Tareas principales del orquestador

- `_price_task`
- `_market_task`
- `_rest_poll_task`
- `_supervisor_task`
- `_recal_task`
- `_outcome_task`
- `_serve_dashboard`
- `social_sentiment`
  - tarea opcional activada por config/env
  - refresca snapshots fuera del hot path y los persiste en cache local

### Procesamiento de mercado

En `_process_market(...)`:

1. si el contrato va a expirar pronto, cierra posiciones del ticker
2. evalúa SL/TP sobre posiciones abiertas
3. evita reentrada si el ticker ya fue operado
4. resuelve precio de referencia externo
5. genera señal con `SignalRouter`
6. chequea `go/no-go`
7. intenta abrir trade

---

## 6. Semántica de paper mode y production

Esto es importante.

### Resolución del modo real

En `main.py`, `_resolve_execution_mode(cfg, paper_trade)` hace esto:

- si `paper_trade=True` -> executor `demo`
- si `cfg.is_demo` -> executor `demo`
- si `cfg.is_production` y `paper_trade=False` -> executor `production`

Eso significa:

- `--dry-run` fuerza entorno demo completo
- `--paper-trade` usa datos de producción pero NO manda órdenes reales
- `PAPER_TRADE=true` también fuerza executor `demo`

### Invariante

Nunca romper esto:

- `ENV=production` + `paper_trade=True` debe seguir usando executor `demo`

Hay tests cubriendo eso.

---

## 7. Invariantes importantes

### Trading

- un `ticker` solo debe operarse una vez
- no reabrir contratos ya operados
- paper y backtest deben compartir la misma lógica de sizing
- el replay debe parecerse a la ejecución real, no a una simulación abstracta
- sentimiento social nunca puede convertirse en dependencia crítica del loop
- el LLM lo usa solo como contexto secundario y explícitamente puede venir ausente

### Riesgo

- `PositionManager.go_no_go_status()` decide si se puede seguir operando
- no detener categorías sanas solo por win rate bajo si el PnL sigue siendo positivo
- `max_drawdown_pct` se mide contra bankroll inicial
- el gate `setup_quality` usa solo histórico resuelto local; no depende de red

### DB

- todas las escrituras pasan por `core/database.py`
- migraciones son aditivas
- no meter SQL directo en otros módulos
- `signals` y `trades` son la historia operacional

---

## 8. Backtesting: cómo pensar su resultado

El backtest actual no evalúa "todas las señales crudas".

Si se instancia con `config`, el runner:

1. filtra señales por política activa
2. deduplica por `ticker`
3. usa el mismo sizing por contratos del executor
4. calcula PnL neto con fees de entrada y salida

Esto fue una corrección importante respecto a versiones anteriores.

Conclusión:

- si se cambia la política del router, conviene revisar también el replay
- si se cambia el sizing real, conviene revisar también el runner

---

## 9. Base de datos y limpieza

Archivo principal:

- `/root/Kalshi-crypto-agents/data/trading.db`

Script de inspección/limpieza:

- [scripts/cleanup_inconsistent_history.py](/root/Kalshi-crypto-agents/scripts/cleanup_inconsistent_history.py)

Qué hace:

- encuentra trades repetidos por `ticker`
- conserva el primero por orden temporal
- borra el resto solo con `--apply`
- crea backup automático antes de borrar

Uso:

```bash
python scripts/cleanup_inconsistent_history.py
python scripts/cleanup_inconsistent_history.py --apply
```

No usar este script a ciegas en una base nueva sin entender el criterio.

---

## 10. Variables de entorno importantes

Mínimas para arrancar:

- `ENV`
- `KALSHI_API_KEY_ID` o `KALSHI_API_KEY`

En production real además:

- `KALSHI_PRIVATE_KEY_PATH`
- `BINANCE_API_KEY`
- `HYPERLIQUID_API_KEY`

Opcionales:

- `OPENROUTER_API_KEY`
- `OPENCLAW_WORKSPACE`
- `BANKROLL_USD`
- `DB_PATH`
- `PAPER_TRADE`
- `LOG_LEVEL`

---

## 11. Comandos útiles

### Paper trading con datos reales

```bash
cd /root/Kalshi-crypto-agents
set -a && source .env && set +a
python main.py --paper-trade
```

### Demo completo

```bash
python main.py --dry-run
```

### Solo recalibrar y salir

```bash
python main.py --backtest-only
```

### Dashboard

```bash
curl -s http://localhost:8090/health
curl -s http://localhost:8090/state
```

### Tests más relevantes

```bash
pytest tests/test_engine.py tests/test_backtesting.py tests/test_execution.py tests/test_integration.py tests/test_schema.py
```

---

## 12. Qué revisar antes de tocar estrategia

Si el bot vuelve a perder o el replay se ve raro, revisar en este orden:

1. `SignalRouter` y sus thresholds efectivos por categoría
2. `BacktestRunner` y si está alineado con la política activa
3. trades repetidos por `ticker` en la DB
4. si `paper_trade` está usando modo `demo` o no
5. resolución de outcomes
6. diferencias entre `config.json`, params calibrados en DB y bloqueos activos

---

## 13. Qué NO asumir

- no asumir que `BTC` usa la misma política que `ETH`/`SOL`
- no asumir que todos los trades históricos son comparables
- no asumir que `paper_trade` y `production` comparten cliente de ejecución
- no asumir que `CLAUDE.md` está al día sin contrastarlo con el código

---

## 14. Última validación conocida

Última suite relevante ejecutada durante los ajustes recientes:

- `280 passed, 7 skipped` (incluye 38 tests en `tests/test_analytics.py` y 19 tests nuevos en `tests/test_backtesting.py`)

Último replay alineado con política actual:

- `BTC`: 24 trades, `PnL=+4.378`
- `ETH`: 21 trades, `PnL=+6.1655`
- `SOL`: 48 trades, `PnL=+29.6095`

Último estado conocido de `blocked_categories` tras recalibración:

- vacío

Tomar esto como referencia operativa, no como verdad eterna.
