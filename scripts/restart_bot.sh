#!/usr/bin/env bash
set -euo pipefail

sudo systemctl restart kalshi-bot.service
sudo systemctl status kalshi-bot.service --no-pager -l
