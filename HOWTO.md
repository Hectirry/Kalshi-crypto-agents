# HOWTO — Kalshi Crypto Trading Bot

## Inicio rápido

```bash
cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a && source .env && set +a
```

### Modos de ejecución

| Comando | Descripción |
|---------|-------------|
| `python main.py --dry-run` | Demo completo: API demo de Kalshi, sin órdenes reales |
| `python main.py --paper-trade` | Paper trading: API real de producción, órdenes simuladas |
| `python main.py` | Trading real: órdenes reales (requiere permisos) |
| `python main.py --backtest-only` | Solo recalibrar parámetros y salir |

### Configuración del entorno

En `.env`:
- `ENV=demo` → usa API demo de Kalshi (no necesita keys reales)
- `ENV=production` → usa API real de Kalshi
- `PAPER_TRADE=true` → órdenes simuladas aunque sea ENV=production

## Dashboard

- API health: `http://localhost:8090/health`
- Dashboard web: `http://localhost:8090/`
- Estado del sistema: `http://localhost:8090/state`
- Mercados Kalshi en vivo: `http://localhost:8090/live-markets`

## Troubleshooting

### Puerto 8090 ocupado
```bash
# Ver qué proceso lo usa
ss -tlnp | grep 8090

# Matar el proceso
fuser -k 8090/tcp
```

### Matar el bot
```bash
pkill -f "python main.py"
```

### Ver logs del servicio systemd (si aplica)
```bash
journalctl -u kalshi-bot.service -f
```

### Ver estado por API
```bash
curl -s http://localhost:8090/state | python3 -m json.tool
```

## Archivos importantes

- `.env` — variables de entorno (API keys, modo)
- `config.json` — estructura de configuración
- `data/trading.db` — base de datos SQLite (signals, trades, params)
- `KalshiApiSecret.pem` — clave privada RSA para firma Kalshi
