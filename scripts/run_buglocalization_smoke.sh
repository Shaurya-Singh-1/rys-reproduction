#!/usr/bin/env bash

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Usage: bash scripts/run_buglocalization_smoke.sh /path/to/local-hf-model-or-hf-id" >&2
  exit 1
fi

OUT="${OUT:-results/localization_smoke_$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-datasets/localization_dataset.json}"
NUM_EXAMPLES="${NUM_EXAMPLES:-2}"
BLOCK="${BLOCK:-16,20}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

cd "$REPO"

echo "== Syncing environment =="
uv sync

echo "== Running localization unit tests =="
uv run pytest tests/test_localization_eval.py -q

echo "== Launching localization smoke run =="
uv run python scripts/run_localization_eval.py \
  --dataset "$DATASET" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUT" \
  --limit-examples "$NUM_EXAMPLES" \
  --block "$BLOCK" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  --max-new-tokens "$MAX_NEW_TOKENS"

echo
echo "Done."
echo "Artifacts:"
echo "  $OUT/run_manifest.json"
echo "  $OUT/records.jsonl"
echo "  $OUT/summary.json"
