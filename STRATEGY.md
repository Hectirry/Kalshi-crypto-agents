# STRATEGY.md

Estrategia base del proyecto `Kalshi-crypto-agents`.

## Tesis

El edge no viene de "adivinar" BTC o ETH. Viene de explotar una desalineación
temporal entre:

- el precio real del subyacente en Binance / Hyperliquid
- la probabilidad implícita del contrato de Kalshi
- el poco tiempo que queda hasta el vencimiento del mercado de 15 minutos

La idea central es simple:

1. Leer el spot externo en tiempo real.
2. Traducir ese spot a una probabilidad propia de que el contrato expire ITM.
3. Comparar esa probabilidad propia con la probabilidad implícita de Kalshi.
4. Solo operar si el delta compensa fees, timing y riesgo.

## Universo operable

Solo mercados cripto de corto vencimiento:

- `BTC`
- `ETH`
- opcionalmente `SOL` si hay suficiente histórico y liquidez

Ventana preferida:

- entre 14 minutos y 90 segundos antes del vencimiento

No operar:

- demasiado temprano
- demasiado tarde
- con liquidez pobre
- con edge pequeño
- con EV neto negativo

## Secuencia de decisión

El flujo correcto del sistema es:

1. `feeds/`
   Obtienen `PriceSnapshot` y `MarketSnapshot`.

2. `ProbabilityEngine`
   Estima `my_prob` usando precio spot, strike y tiempo restante.

3. `TimingFilter`
   Verifica si la ventana temporal todavía es operable.

4. `EVCalculator`
   Resta fees reales de Kalshi y calcula si todavía existe edge neto.

5. `SignalRouter`
   Devuelve:
   - `YES`
   - `NO`
   - `SKIP`
   - `ERROR`

6. `OrderExecutor`
   Ejecuta en demo o producción.

7. `PositionManager`
   Gestiona cierre, stop loss, take profit y estado GO/NO-GO.

8. `BacktestRunner`
   Mide si esta política realmente gana en histórico y recalibra parámetros.

## Reglas de entrada

Una señal solo debería pasar a ejecución si cumple todas:

- categoría no bloqueada
- ventana temporal válida
- `abs(delta) >= min_delta`
- `ev_net_fees > 0`
- `kelly_size > 0`

Interpretación:

- `delta > 0` -> comprar `YES`
- `delta < 0` -> comprar `NO`

## Reglas de sizing

El tamaño de la posición no debe ser fijo.

Se usa Kelly fraccional:

`f = (my_prob - contract_price) / (1 - contract_price)`

Luego:

- aplicar `kelly_fraction`
- capear por `max_pct`
- si `f <= 0`, no operar

Objetivo:

- crecer bankroll cuando hay edge
- limitar sobreexposición cuando la confianza aparente es alta pero el mercado es frágil

## Reglas de salida

En 15 minutos el tiempo destruye edge muy rápido, así que la salida es tan importante
como la entrada.

El sistema actual contempla:

- `take_profit_pct`
- `stop_loss_pct`
- cierre natural por resolución del mercado

En producción real conviene añadir después:

- recorte por pérdida de momentum
- bloqueo de reentrada en el mismo ticker
- salida por falta de liquidez

## Qué significa "sin estrategia no sirve"

Correcto: infraestructura sola no basta.

La estrategia concreta de esta versión es:

- mercados cripto cortos
- señal cuantitativa basada en delta probabilístico
- filtro temporal estricto
- EV neto de fees obligatorio
- sizing con Kelly fraccional
- recalibración por histórico
- bloqueo automático de categorías con mal desempeño

## Qué valida el backtest

El backtest no debe responder solo "ganó o perdió".

Debe responder:

- si el edge existía de verdad después de fees
- si el threshold de `min_delta` está bien calibrado
- si el threshold de `min_ev_threshold` está bien calibrado
- si una categoría debe quedar bloqueada
- si la curva de equity mejora o se degrada

Con `vectorbt` activo, el runner puede modelar una curva de equity más seria
sobre trades secuenciales.

## Qué queda por mejorar

Todavía faltan piezas para una estrategia más fuerte:

- usar order book externo, no solo precio spot
- distinguir mejor regímenes de volatilidad
- separar estrategia por activo (`BTC` != `ETH`)
- validar slippage real de Kalshi
- añadir capa opcional OpenRouter como validador final, nunca como sustituto

## Regla operativa simple

Si una oportunidad no puede explicarse así en una línea, no debería ejecutarse:

`Tengo mejor probabilidad que Kalshi, todavía queda tiempo, el EV neto es positivo y el sizing es razonable.`
