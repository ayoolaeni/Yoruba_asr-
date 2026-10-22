#!/usr/bin/env bash
# Yoruba Speech Recognition - stop (macOS / Linux). Run:  bash stop_mac_linux.sh
cd "$(dirname "$0")" || exit 1
docker compose down
echo "Stopped. Start again with: bash start_mac_linux.sh"
