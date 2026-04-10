| Fase | Módulos                              | Estado     | Puerta de salida        |
|------|--------------------------------------|------------|-------------------------|
| 1    | interfaces, models, config, database | COMPLETA   | 51/51 tests ✓ 2026-04-09|
|| 2    | binance_feed, hyperliquid_feed,      | COMPLETA   | 100/100 tests ✓ 2026-04-09|
|      | kalshi_feed                          |            | 20 ticks reales en 30s    |
|      | kalshi_feed                          |            |                         |
| 3    | probability, timing, ev_calculator,  | COMPLETA   | 126/126 tests ✓ 2026-04-09 |
|      | signal_router                        |            | engine/ 90% coverage ✓  |
| 4    | backtest_runner, category_blocker,   | COMPLETA   | 134/134 tests ✓ 2026-04-09 |
|      | param_injector                       |            | backtesting/ 91% coverage ✓ |
|      |                                      |            | vectorbt path verified in .venv ✓ |
| 5    | order_executor, observabilidad       | COMPLETA   | 145/145 tests ✓ 2026-04-09 |
|      | GO/NO-GO                             |            | execution/ 86% coverage ✓ |
