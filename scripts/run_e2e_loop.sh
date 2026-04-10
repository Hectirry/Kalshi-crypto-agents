#!/usr/bin/env bash
set -euo pipefail

cd /root/Kalshi-crypto-agents
source .venv/bin/activate
set -a
source .env
set +a

mkdir -p logs

PYTHONPATH=. python main.py --mode loop --config config.json --interval 30
