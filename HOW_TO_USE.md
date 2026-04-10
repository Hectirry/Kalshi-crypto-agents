# HOW_TO_USE.md

Guía corta para correr y usar `Kalshi-crypto-agents` en el VPS.

## 1. Activar el entorno

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a
source .env
set +a
```

## 1.1 EntryPoint único (`main.py`)

Desde ahora puedes correr todo desde un solo archivo:

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a
source .env
set +a
PYTHONPATH=. python main.py --mode smoke
```

Modos:

- `--mode smoke` test E2E puntual
- `--mode loop` proceso continuo
- `--mode dashboard` dashboard terminal en vivo

## 2. Ejecutar tests

Suite completa:

```bash
PYTHONPATH=/root/Kalshi-crypto-agents pytest tests/ -v --tb=short
```

Cobertura por fase:

```bash
PYTHONPATH=/root/Kalshi-crypto-agents pytest tests/test_engine.py --cov=engine --cov-report=term-missing
PYTHONPATH=/root/Kalshi-crypto-agents pytest tests/test_backtesting.py --cov=backtesting --cov-report=term-missing
PYTHONPATH=/root/Kalshi-crypto-agents pytest tests/test_execution.py --cov=execution --cov-report=term-missing
```

## 3. Variables de entorno mínimas

En demo:

```bash
set -a
source /root/Kalshi-crypto-agents/.env
set +a
```

Variables Kalshi recomendadas (RSA-PSS):

```bash
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=/ruta/a/tu/clave.pem
```

En producción:

```bash
export ENV=production
export KALSHI_API_KEY=tu_api_key
export KALSHI_API_SECRET=tu_api_secret
export BINANCE_API_KEY=tu_binance_key
export HYPERLIQUID_API_KEY=tu_hyperliquid_key
export LOG_LEVEL=INFO
```

## 4. Cómo usar el engine

Flujo típico:

1. `feeds/` produce `PriceSnapshot` y `MarketSnapshot`
2. `engine/signal_router.py` genera `Signal`
3. `execution/order_executor.py` ejecuta la señal
4. `execution/position_manager.py` maneja SL/TP y estado operativo
5. `backtesting/` recalibra umbrales y bloquea categorías malas

## 5. Demo execution

Ejemplo mínimo:

```python
from core.database import Database
from core.models import Confidence, Decision, Signal
from execution.order_executor import PaperOrderExecutor

db = Database(path=Path("./data/trading.db"))
db.initialize()

executor = PaperOrderExecutor(db=db, mode="demo")

signal = Signal(
    market_ticker="KXBTC-15MIN-B95000",
    decision=Decision.YES,
    my_probability=0.67,
    market_probability=0.55,
    delta=0.12,
    ev_net_fees=0.06,
    kelly_size=0.08,
    confidence=Confidence.HIGH,
    time_remaining_s=420,
    reasoning="demo example",
    timestamp=time.time(),
)

trade = await executor.submit(signal)
```

## 6. Gestión de posiciones

`PositionManager` sirve para:

- sincronizar trades abiertos desde SQLite
- cerrar por `stop_loss`
- cerrar por `take_profit`
- generar snapshot de observabilidad
- generar estado `GO / NO-GO`

Ejemplo:

```python
manager = PositionManager(db=db, executor=executor, stop_loss_pct=0.05, take_profit_pct=0.10)
trade = await manager.open_from_signal(signal)
closes = await manager.evaluate_price(
    ticker=trade.ticker,
    current_yes_price=0.62,
)
status = manager.go_no_go_status(min_closed_trades=10, min_win_rate=0.52)
```

## 7. Backtesting

Ejemplo:

```python
runner = BacktestRunner(db=db, initial_bankroll=1000.0)
result = runner.run(from_ts=start_ts, to_ts=end_ts, category="BTC")
```

Si tienes `vectorbt` instalado en [`.venv`](/root/Kalshi-crypto-agents/.venv),
el runner usará la ruta nativa automáticamente y marcará:

- `result.vectorbt_available == True`
- `result.vectorbt_used == True`

Prueba rápida:

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
PYTHONPATH=/root/Kalshi-crypto-agents pytest tests/test_backtesting.py -v --tb=short
```

Luego puedes recalibrar:

```python
injector = ParamInjector(db=db)
injector.calibrate(from_ts=start_ts, to_ts=end_ts, categories={"BTC", "ETH"})
```

Y bloquear categorías malas:

```python
blocker = CategoryBlocker(db=db, runner=runner)
blocker.evaluate_and_apply(from_ts=start_ts, to_ts=end_ts)
```

## 8. Qué falta para producción real

- Implementar cliente real de Kalshi para `production`
- Conectar `OrderExecutor` al API de órdenes real
- Añadir observabilidad externa más fuerte
- Integrar dashboard/API
- Integrar OpenRouter opcional como validador final

## 9. Estrategia

La tesis operativa del proyecto quedó documentada en:

- [STRATEGY.md](/root/Kalshi-crypto-agents/STRATEGY.md)

## 10. API de dashboard

Levantar API minima:

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a
source .env
set +a
uvicorn dashboard.api_server:create_app --factory --host 0.0.0.0 --port 8090
```

Endpoints disponibles:

- `GET /health`
- `GET /state?limit=20`

## 10. Nota sobre APIs

Este repo ahora soporta ambos formatos:

- canónico: `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH`
- legacy (compat): `KALSHI_API_KEY` + `KALSHI_API_SECRET`

Tu sistema anterior en [Kalshi Trading](/root/Kalshi%20Trading) usa otro modelo:

- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PATH`

Eso significa que no basta con copiar variables entre proyectos. Para usar las
credenciales reales de Kalshi aquí hay que:

1. conseguir el formato de credenciales que este repo espera, o
2. adaptar esta arquitectura nueva para firmar requests como el sistema legacy

Mientras eso no se haga, Binance / Hyperliquid pueden configurarse sin problema,
pero Kalshi real queda pendiente de alinear con el modelo de autenticación.

## 11. Smoke E2E (en vivo)

Para validar flujo completo (Kalshi + Binance + engine + ejecución demo + DB):

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a
source .env
set +a
PYTHONPATH=. python main.py --mode smoke --config config.json
```

Salida esperada: bloque `E2E_RESULT` con `live_signal` y `shadow_signal`.

## 12. Dashboard Terminal (Wallstreet + Matrix)

Vista en tiempo real en terminal con posiciones abiertas, uPnL estimado y señales recientes:

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a
source .env
set +a
PYTHONPATH=. python main.py --mode dashboard --config config.json --refresh 5
```

Controles:

- `Ctrl + C` para salir
- Cambia frecuencia con `--refresh 2` o `--refresh 10`
