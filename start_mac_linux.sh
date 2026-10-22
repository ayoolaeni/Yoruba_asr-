#!/usr/bin/env bash
# Yoruba Speech Recognition - start (macOS / Linux). Run:  bash start_mac_linux.sh
cd "$(dirname "$0")" || exit 1
echo "[1/4] Checking that Docker Desktop is running..."
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is NOT running. Start it, wait for 'Engine running', then run this script again."
  exit 1
fi
echo "[2/4] Checking the trained models..."
if [ ! -f models/V4/model.safetensors ]; then
  echo "The folder 'models' is missing or incomplete (needs V1-V4). See HOW_TO_RUN.txt."
  exit 1
fi
echo "[3/4] Building and starting (first time: 5-15 minutes, about 1.5 GB download)..."
docker compose up -d --build || { echo "Docker could not start the application. See HOW_TO_RUN.txt."; exit 1; }
echo "[4/4] Waiting for the application to be ready..."
for i in $(seq 1 60); do
  if curl -s -o /dev/null http://localhost:7870/; then
    echo "READY: http://localhost:7870  (stop later with: bash stop_mac_linux.sh)"
    (command -v open >/dev/null && open http://localhost:7870) || (command -v xdg-open >/dev/null && xdg-open http://localhost:7870) || true
    exit 0
  fi
  sleep 5
done
echo "The application did not answer after 5 minutes. Run: docker compose logs"
exit 1
