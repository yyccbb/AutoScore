#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/clear_ocr.sh INPUT_DIR [--yes]

Delete OCR .txt files directly under INPUT_DIR.

By default this script only previews matching files. Add --yes to delete.
It does not recurse into subdirectories.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INPUT_DIR="$1"
CONFIRM="${2:-}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERROR] Not a directory: $INPUT_DIR" >&2
  exit 1
fi

mapfile -d '' TXT_FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.txt' -print0 | sort -z)

COUNT="${#TXT_FILES[@]}"
echo "[INFO] Found $COUNT .txt file(s) under: $INPUT_DIR"

if [[ "$COUNT" -eq 0 ]]; then
  exit 0
fi

printf '%s\n' "${TXT_FILES[@]}"

if [[ "$CONFIRM" != "--yes" ]]; then
  echo "[DRY RUN] Add --yes to delete these files."
  exit 0
fi

for file in "${TXT_FILES[@]}"; do
  rm -- "$file"
done

echo "[INFO] Deleted $COUNT .txt file(s)."
