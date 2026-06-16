#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"
CONFIG_PATH="${1:-$ROOT_DIR/configs/demo_asro_train.yaml}"
NUM="${NUM:-5}"
read -r -a PYTHON_PARTS <<< "$PYTHON_CMD"

if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  CONFIG_PATH="$ROOT_DIR/configs/demo_asro_train.yaml"
  NUM="$1"
elif [[ $# -ge 2 ]]; then
  NUM="$2"
fi

args=("$ROOT_DIR/ASRO/train.py" --config "$CONFIG_PATH")
if [[ -n "$NUM" ]]; then
  args+=(--num "$NUM")
fi

echo "[INFO] ASRO train demo config: $CONFIG_PATH"
echo "[INFO] NUM=$NUM limits pre-training OCR only (set NUM= to use config value)"
cd "$ROOT_DIR"
"${PYTHON_PARTS[@]}" "${args[@]}"
