# Kalshi Crypto Agents

Automated trading system for short-duration crypto prediction markets on Kalshi.

The project watches live BTC, ETH, and SOL markets, compares Kalshi implied
probability against external crypto prices, routes signals through risk filters,
and records every signal/trade in SQLite for review and backtesting.

## What It Does

- Streams spot prices from Binance.
- Polls Hyperliquid as a second spot source and builds a consensus reference price.
- Streams and polls crypto markets from Kalshi.
- Estimates probability edges for 15-minute crypto contracts.
- Applies timing, expected value, fee, Kelly sizing, and category-blocking rules.
- Supports paper trading against live production data.
- Stores signals, trades, outcomes, calibrated params, and blocked categories.
- Serves a FastAPI dashboard for live observability.

## Safety

This project is experimental trading software. Do not run it with real order
execution enabled until you have reviewed the code, tested it with paper trading,
and verified your Kalshi credentials, limits, and risk controls.

Secrets must stay out of Git:

- `.env`
- private key files such as `*.pem`
- SQLite databases under `data/`
- runtime logs under `logs/`

The repository includes `.env.example` as the public template.

## Quick Start

```bash
cd /root/Kalshi-crypto-agents
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Fill in `.env`, then run with live market data and simulated orders:

```bash
set -a && source .env && set +a
python main.py --paper-trade
```

Recommended current paper-trading posture:

- `BTC` is enabled again, but with stricter category-specific filters.
- `ETH` and `SOL` remain enabled.
- The engine now prevents repeat entries on the same contract ticker.
- Historical duplicate trades were cleaned from `data/trading.db` using the
  project cleanup script with backup.

For a full demo run:

```bash
python main.py --dry-run
```

To inspect duplicate-per-ticker history without modifying the DB:

```bash
python scripts/cleanup_inconsistent_history.py
```

## Dashboard

When the main process is running, the dashboard API is served on port `8090`.

- Web dashboard: `http://localhost:8090/`
- Health: `http://localhost:8090/health`
- System state: `http://localhost:8090/state`
- Live markets: `http://localhost:8090/live-markets`

## Run As A Service

Example systemd commands:

```bash
sudo systemctl enable kalshi-bot.service
sudo systemctl start kalshi-bot.service
sudo systemctl status kalshi-bot.service --no-pager
```

Follow logs:

```bash
journalctl -u kalshi-bot.service -f
```

## Project Layout

```text
core/         Canonical models, config, database, interfaces
feeds/        Binance, Hyperliquid, and Kalshi market feeds
engine/       Probability, timing, EV, signal routing, optional LLM validator
execution/    Paper order execution and position management
backtesting/  Historical replay, calibration, and category blocking
dashboard/    FastAPI endpoints and static live dashboard
scripts/      Operational helpers such as database cleanup
tests/        Unit and integration tests with mocked external APIs
```

## Tests

```bash
pytest
```

Linting, if `ruff` is installed:

```bash
ruff check .
```

## Configuration

Public defaults live in `config.json`. Secrets and runtime mode live in `.env`.

Recommended paper-trading setup:

```bash
ENV=production
PAPER_TRADE=true
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=/path/to/private-key.pem
BINANCE_API_KEY=...
HYPERLIQUID_API_KEY=...
```

Optional:

```bash
OPENROUTER_API_KEY=...
OPENCLAW_WORKSPACE=~/.openclaw/workspace
```

If `OPENCLAW_WORKSPACE` is set, the bot appends session and trade notes to a
workspace-compatible Markdown journal under `memory/YYYY-MM-DD.md`.

## Status

The current implementation is intended for supervised paper trading and
observability. Production order execution should be treated as a separate
hardening step.
