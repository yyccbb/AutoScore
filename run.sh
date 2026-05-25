#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs.txt}"
mkdir -p "$(dirname "$LOG_FILE")"

# 每次运行覆盖旧日志；如果想追加日志，注释掉这一行
: > "$LOG_FILE"

# 同时输出到终端和 logs.txt
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "[INFO] AES runner started"
echo "[INFO] Root directory: $ROOT_DIR"
echo "[INFO] Logging to: $LOG_FILE"
echo "[INFO] Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
MODE="${1:-pipeline}"
CONFIG_OVERRIDE="${2:-}"
NUM="${3:-${NUM:-}}"

# 默认使用当前环境里的 python。
# 如需指定：
#   PYTHON_CMD="conda run --no-capture-output -n env python" bash run.sh pipeline
# 或：
#   PYTHON_CMD="python" bash run.sh pipeline
PYTHON_CMD="${PYTHON_CMD:-python}"
read -r -a PYTHON_PARTS <<< "$PYTHON_CMD"

echo "[INFO] Mode: $MODE"
echo "[INFO] Python command: $PYTHON_CMD"

if [[ -n "$CONFIG_OVERRIDE" ]]; then
  echo "[INFO] Config override: $CONFIG_OVERRIDE"
fi

if [[ -n "$NUM" ]]; then
  echo "[INFO] NUM override: $NUM"
fi

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
run_pipeline() {
  local config_path="${CONFIG_OVERRIDE:-$ROOT_DIR/configs/full_pipeline.yaml}"

  echo
  echo "============================================================"
  echo "[INFO] Running pipeline"
  echo "[INFO] Config: $config_path"
  echo "============================================================"

  if [[ -n "$NUM" ]]; then
    "${PYTHON_PARTS[@]}" "$ROOT_DIR/main.py" --config "$config_path" --num "$NUM"
  else
    "${PYTHON_PARTS[@]}" "$ROOT_DIR/main.py" --config "$config_path"
  fi
}

run_asro_train() {
  local config_path="${CONFIG_OVERRIDE:-$ROOT_DIR/configs/full_asro_train.yaml}"

  echo
  echo "============================================================"
  echo "[INFO] Running ASRO training"
  echo "[INFO] Config: $config_path"
  echo "============================================================"

  if [[ -n "$NUM" ]]; then
    echo "[INFO] NUM=$NUM limits pre-training OCR / sample loading if supported by train.py"
    "${PYTHON_PARTS[@]}" "$ROOT_DIR/ASRO/train.py" --config "$config_path" --num "$NUM"
  else
    "${PYTHON_PARTS[@]}" "$ROOT_DIR/ASRO/train.py" --config "$config_path"
  fi
}

print_usage() {
  echo "Usage:"
  echo "  bash run.sh [pipeline|train|asro_train|full] [config_path] [num]"
  echo
  echo "Examples:"
  echo "  bash run.sh"
  echo "  bash run.sh pipeline"
  echo "  bash run.sh train"
  echo "  bash run.sh full"
  echo "  bash run.sh pipeline configs/full_pipeline.yaml"
  echo "  bash run.sh pipeline configs/full_pipeline.yaml 5"
  echo
  echo "Environment overrides:"
  echo "  NUM=5 bash run.sh pipeline"
  echo "  LOG_FILE=logs/pipeline.log bash run.sh pipeline"
  echo "  PYTHON_CMD=\"python\" bash run.sh pipeline"
  echo "  PYTHON_CMD=\"conda run --no-capture-output -n env python\" bash run.sh pipeline"
}

# ------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------
case "$MODE" in
  pipeline)
    run_pipeline
    ;;
  train|asro_train)
    run_asro_train
    ;;
  full)
    run_pipeline
    run_asro_train
    ;;
  -h|--help|help)
    print_usage
    exit 0
    ;;
  *)
    echo "[ERROR] Unknown mode: $MODE"
    echo
    print_usage
    exit 1
    ;;
esac

echo
echo "============================================================"
echo "[INFO] AES runner finished successfully"
echo "[INFO] Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Log saved to: $LOG_FILE"
echo "============================================================"