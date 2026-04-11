#!/usr/bin/env bash
set -euo pipefail

sudo journalctl -u kalshi-bot.service -f
