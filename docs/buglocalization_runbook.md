# Bug Localization Runbook

This branch adds a compact downstream experiment that is much easier to run locally than the full SWE-agent pipeline.

The task is **fault localization**, not code repair:

- each example contains a bug report
- the model sees a small ranked candidate set of source-file snippets
- the model must output a ranking of the candidate files
- we score whether the true buggy file is ranked first, within the top 3, and by reciprocal rank

## What ships on this branch

- `datasets/localization_dataset.json`
  - curated evaluation set with `13` examples
- `datasets/localization_dataset.json.summary.json`
  - dataset creation summary
- `scripts/run_localization_eval.py`
  - main evaluator
- `scripts/run_buglocalization_smoke.sh`
  - one-command local smoke run
- `src/agent_eval/localization.py`
  - prompt construction, parser, and scoring helpers
- `tests/test_localization_eval.py`
  - unit tests for parsing and aggregation

## Recommended local models

For a normal laptop, prefer a small instruct checkpoint:

- `Qwen/Qwen2.5-3B-Instruct`
- a locally-downloaded equivalent Hugging Face causal LM

For stronger local hardware, a 7B instruct model is also reasonable:

- `Qwen/Qwen2.5-7B-Instruct`

The evaluator accepts either:

- a Hugging Face model id
- or a local model directory

## Fastest way to verify the branch

From the repo root:

```bash
bash scripts/run_buglocalization_smoke.sh --tiny-fixture
```

This does three things:

1. `uv sync`
2. builds a tiny local Llama fixture
3. runs `tests/test_localization_eval.py`
4. evaluates a tiny two-example baseline-vs-RYS smoke run

## Useful overrides

```bash
OUT=results/localization_demo \
NUM_EXAMPLES=4 \
BLOCK=16,20 \
DEVICE_MAP=cpu \
DTYPE=float32 \
bash scripts/run_buglocalization_smoke.sh --tiny-fixture
```

If you already have a real local model:

```bash
bash scripts/run_buglocalization_smoke.sh /path/to/local-hf-model
```

You can also point directly at the Python entrypoint:

```bash
uv run python scripts/run_localization_eval.py \
  --dataset datasets/localization_dataset.json \
  --model-path /path/to/local-hf-model \
  --output-dir results/localization_eval \
  --limit-examples 4 \
  --block 16,20 \
  --dtype bfloat16 \
  --device-map auto
```

## Outputs

The evaluator writes:

- `run_manifest.json`
  - run configuration and selected examples
- `records.jsonl`
  - one row per `(condition, issue_id)` evaluation
- `summary.json`
  - aggregated metrics by condition

The key reported metrics are:

- `top1_accuracy`
- `top3_accuracy`
- `mrr`
- `avg_rank`

## Practical interpretation

This branch is useful when you want a **locally runnable downstream RYS experiment** without:

- full SWE-bench containers
- agent shell loops
- long code-repair trajectories

It is best thought of as a ranking-style proxy task for software engineering rather than a full autonomous repair benchmark.
