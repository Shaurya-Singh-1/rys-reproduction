#!/usr/bin/env bash

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ARG="${1:-}"
USE_TINY_FIXTURE=0

if [[ -z "$MODEL_ARG" || "$MODEL_ARG" == "--tiny-fixture" ]]; then
  USE_TINY_FIXTURE=1
fi

DATASET="${DATASET:-datasets/localization_dataset.json}"
FIXTURE_DIR="${FIXTURE_DIR:-local/tiny_llama_localization}"
FIXTURE_NUM_LAYERS="${FIXTURE_NUM_LAYERS:-24}"
FIXTURE_HIDDEN_SIZE="${FIXTURE_HIDDEN_SIZE:-64}"
FIXTURE_INTERMEDIATE_SIZE="${FIXTURE_INTERMEDIATE_SIZE:-128}"
FIXTURE_NUM_HEADS="${FIXTURE_NUM_HEADS:-4}"
FIXTURE_NUM_KV_HEADS="${FIXTURE_NUM_KV_HEADS:-4}"
FIXTURE_TOKENIZER="${FIXTURE_TOKENIZER:-hf-internal-testing/llama-tokenizer}"

if [[ "$USE_TINY_FIXTURE" -eq 1 ]]; then
  MODEL_PATH="$FIXTURE_DIR"
  OUT="${OUT:-results/localization_tiny_smoke_$(date +%Y%m%d_%H%M%S)}"
  NUM_EXAMPLES="${NUM_EXAMPLES:-2}"
  BLOCK="${BLOCK:-16,20}"
  DEVICE_MAP="${DEVICE_MAP:-cpu}"
  DTYPE="${DTYPE:-float32}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
else
  MODEL_PATH="$MODEL_ARG"
  OUT="${OUT:-results/localization_smoke_$(date +%Y%m%d_%H%M%S)}"
  NUM_EXAMPLES="${NUM_EXAMPLES:-2}"
  BLOCK="${BLOCK:-16,20}"
  DEVICE_MAP="${DEVICE_MAP:-auto}"
  DTYPE="${DTYPE:-bfloat16}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

cd "$REPO"

echo "== Syncing environment =="
uv sync

if [[ "$USE_TINY_FIXTURE" -eq 1 ]]; then
  echo "== Building tiny local fixture =="
  uv run python scripts/build_tiny_llama_fixture.py \
    --output-dir "$FIXTURE_DIR" \
    --tokenizer "$FIXTURE_TOKENIZER" \
    --num-layers "$FIXTURE_NUM_LAYERS" \
    --hidden-size "$FIXTURE_HIDDEN_SIZE" \
    --intermediate-size "$FIXTURE_INTERMEDIATE_SIZE" \
    --num-heads "$FIXTURE_NUM_HEADS" \
    --num-kv-heads "$FIXTURE_NUM_KV_HEADS"
fi

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
