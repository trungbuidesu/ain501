#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".git" ]]; then
  echo "Run this script from repo root."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/4] Install runtime dependencies"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "[2/4] Verify runtime models"
"${PYTHON_BIN}" scripts/verify_models.py

echo "[3/4] Optional: build deploy zip"
"${PYTHON_BIN}" scripts/package_models.py --output models_release.zip

echo "[4/4] Dry-run smoke test"
"${PYTHON_BIN}" -m src.app --dry-run --max-frames 20

echo "Setup complete."
echo "Run app: ${PYTHON_BIN} -m src.app --visual --tts"
