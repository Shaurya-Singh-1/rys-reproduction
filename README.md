# RYS

RYS is a reproducibility repo for **relayering** experiments on decoder LLMs.

This `buglocalization` branch keeps the core scan/export workflow and adds the most practical downstream experiment in the repo:

- a **fault-localization evaluation** that can run locally with a standard Hugging Face model
- a packaged localization dataset with candidate file snippets
- a one-command smoke runner for baseline-vs-RYS local evaluation

If you only need one command to verify this repo on a fresh machine, use:

```bash
bash scripts/run_buglocalization_smoke.sh --tiny-fixture
```

That command:

1. installs/syncs dependencies with `uv`
2. builds a tiny local Llama fixture
3. runs the localization unit tests
4. executes a tiny two-example localization comparison using baseline and one RYS block

For the full localization instructions, see:

- `docs/buglocalization_runbook.md`

The core idea is simple: duplicate part of a model's existing layer path without changing any weights.
A standard single-block configuration is written as `(i, j)`:

- run layers `0 .. j-1`
- then jump back and run layers `i .. N-1`
- so layers `i .. j-1` are traversed twice

Baseline is `(0,0)`, which means no duplication.

This repo contains the pieces needed to reproduce the main experimental workflows:

- scanner for full `(i, j)` sweeps
- fixed Math and EQ probe sets
- multi-block beam search
- XGBoost surrogate pipeline
- model exporter for writing relayered Hugging Face checkpoints
- heatmap and balanced Math+EQ analysis code
- a local fault-localization evaluation path

It does not include the private historical runs, blog drafts, ad hoc notebooks, or dataset generation/calibration code.

## Repo Contents

- `datasets/`
  - `math_16.json`
  - `math_120.json`
  - `eq_16.json`
  - `eq_140.json`
  - `manifest.json`
- `src/core/`
  - config parsing
  - layer-list expansion
  - relayer wrappers for dense and MoE-style stacks
- `src/workers/`
  - Math and EQ benchmark workers
  - queue handling
  - model loading helpers
- `src/utils/`
  - balanced Math+EQ analysis
  - heatmap helpers
  - surrogate utilities
- `src/agent_eval/localization.py`
  - prompt construction, ranking parser, and metrics for fault localization
- `scripts/`
  - sweep setup
  - ExLlama workers
  - beam search
  - surrogate pipeline
  - repeat-sweep helpers
  - localization dataset builder and local evaluator
- `hf_export/`
  - checkpoint export
  - HF upload helper
  - Colab notebook

## Branch Focus: Local Bug Localization

The localization branch is the easiest downstream path to run on a normal workstation.

Instead of full repository repair, each example asks the model to:

- read a bug report
- inspect a short list of candidate source files
- rank the files by likelihood of containing the root cause

The shipped dataset is:

- `datasets/localization_dataset.json`

and the summary metadata is:

- `datasets/localization_dataset.json.summary.json`

At the current branch tip, the packaged dataset contains `13` examples.

The main entrypoint is:

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

It writes:

- `run_manifest.json`
- `records.jsonl`
- `summary.json`

with the main metrics:

- `top1_accuracy`
- `top3_accuracy`
- `mrr`
- `avg_rank`

## Probe Sets

This repo ships the fixed benchmark subsets used by the public workflow:

- `datasets/math_16.json`
- `datasets/math_120.json`
- `datasets/eq_16.json`
- `datasets/eq_140.json`

Important notes:

- `eq_16` and `eq_140` are first-pass-only EQ subsets.
- `datasets/manifest.json` records provenance and checksums.
- The file named `eq_140.json` currently contains `139` records; this is documented in the manifest and preserved for continuity with the original naming.

## Setup

Python uses `uv`:

```bash
uv sync
```

For ExLlama scanning you also need:

- a local `exllamav3` checkout
- an EXL3-compatible model directory
- CUDA-capable hardware if you want real scan throughput

Set:

```bash
export EXLLAMAV3_PATH=/path/to/exllamav3
```

For the localization branch, you do **not** need ExLlama just to run the local downstream evaluator. A normal Hugging Face causal LM is enough.

## Local Smoke Demo

If you want to validate the end-to-end pipeline on a normal laptop, run:

```bash
uv sync
bash scripts/run_smoke_demo.sh
```

This builds a tiny local Llama fixture, runs a full six-layer `(i, j)` sweep with the Hugging Face worker on tiny Math/EQ smoke sets, and writes artifacts under:

- `results/smoke/combined_results.pkl`
- `results/smoke/math_results.pkl`
- `results/smoke/eq_results.pkl`
- `results/smoke/analysis/`

The smoke run is only for reproducibility plumbing, not scientific quality. For meaningful RYS results, switch to a real model and the public probe sets.

## Localization Smoke Run

For the localizable downstream experiment on this branch, run:

```bash
bash scripts/run_buglocalization_smoke.sh --tiny-fixture
```

Useful overrides:

```bash
OUT=results/localization_demo \
NUM_EXAMPLES=4 \
BLOCK=16,20 \
DEVICE_MAP=cpu \
DTYPE=float32 \
bash scripts/run_buglocalization_smoke.sh --tiny-fixture
```

Recommended local model sizes:

- zero-setup smoke verification: the built-in tiny fixture
- easiest on consumer hardware: `Qwen/Qwen2.5-3B-Instruct`
- stronger local option: `Qwen/Qwen2.5-7B-Instruct`

The smoke script first verifies the parsing/scoring logic with:

```bash
uv run pytest tests/test_localization_eval.py -q
```

then launches the actual evaluator.

If you already have a real local model, you can pass it directly instead:

```bash
bash scripts/run_buglocalization_smoke.sh /path/to/local-hf-model
```

## Quick Start

### 1. Create a full `(i, j)` sweep queue

Example for a 64-layer model:

```bash
uv run python scripts/init_queue.py \
  --num-layers 64 \
  --queue-file results/demo/queue.json \
  --results-file results/demo/combined_results.pkl
```

This writes canonical layer-list configs, including the baseline `(0,0)`.

### 2. Run the ExLlama combined scanner

This is the main fast scan path in the public repo. It loads the EXL3 model once and scores Math and EQ in one mixed pass per config.

```bash
uv run python scripts/run_exllama_math_eq_combined_worker.py \
  --queue-file results/demo/queue.json \
  --combined-results-file results/demo/combined_results.pkl \
  --math-results-file results/demo/math_results.pkl \
  --eq-results-file results/demo/eq_results.pkl \
  --model-dir /path/to/model.exl3 \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 64 \
  --eq-max-new 64 \
  --auto-cache
```

### 3. Analyze and render heatmaps

```bash
uv run python scripts/analyze_results.py \
  --math-scores results/demo/math_results.pkl \
  --eq-scores results/demo/eq_results.pkl \
  --out-dir results/demo/analysis \
  --num-layers 64
```

This produces:

- top-ranked balanced configs
- scatter plots
- balanced Math+EQ heatmap artifacts

## Containerized ExLlama Run

If you want a simple Docker entrypoint:

```bash
MODEL_DIR=/path/to/model.exl3 \
EXLLAMAV3_PATH=/path/to/exllamav3 \
./scripts/run_exllama_docker.sh
```

Useful overrides:

- `QUEUE_FILE`
- `COMBINED_RESULTS_FILE`
- `MATH_RESULTS_FILE`
- `EQ_RESULTS_FILE`
- `MATH_DATASET`
- `EQ_DATASET`
- `DEVICE`
- `RESERVE_PER_DEVICE`
- `USE_PER_DEVICE`

## Beam Search

Beam search composes multiple repeated blocks and benchmarks only unseen configs.

Example:

```bash
uv run python scripts/beam_search.py \
  --model-path /path/to/hf-model \
  --num-layers 64 \
  --seed-math-results results/demo/math_results.pkl \
  --seed-eq-results results/demo/eq_results.pkl \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --work-dir results/demo/beam-search
```

Notes:

- beam search uses the Hugging Face worker path in `src/workers/`
- seed pickles should come from an already measured single-block scan
- state under `--work-dir` is resume-friendly

## Surrogate Pipeline

The surrogate uses per-layer repeat counts as features. Predicted scores are only for ranking candidates; the final benchmark scores must be measured.

### Train

```bash
uv run python scripts/train_surrogate.py \
  --single-block-math-results results/demo/math_results.pkl \
  --single-block-eq-results results/demo/eq_results.pkl \
  --beam-math-results results/demo/beam-search/beam_math_results.pkl \
  --beam-eq-results results/demo/beam-search/beam_eq_results.pkl \
  --out-dir results/demo/surrogate \
  --num-layers 64
```

### Generate candidates

```bash
uv run python scripts/generate_candidates.py \
  --num-layers 64 \
  --max-extra-layers 12 \
  --count 2000000 \
  --output-csv results/demo/surrogate/candidates.csv
```

### Score candidates and build a top-k config file

```bash
uv run python scripts/score_candidates.py \
  --model-dir results/demo/surrogate \
  --candidates-file results/demo/surrogate/candidates.csv \
  --output-csv results/demo/surrogate/top_scored.csv \
  --top-k 100

uv run python scripts/build_topk_config.py \
  --top-candidates-csv results/demo/surrogate/top_scored.csv \
  --num-layers 64 \
  --output-config results/demo/surrogate/top100.config
```

Then benchmark those configs with the same Math/EQ harness used elsewhere.

## Model Export

`hf_export` writes a relayered Hugging Face checkpoint with the duplicated layers physically materialized into the safetensor shards.

Single block:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id some/model \
  --output exports/model-block-30-34 \
  --blocks "30,34"
```

Multi-block:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id some/model \
  --output exports/model-31_34__43_45 \
  --blocks "31,34;43,45"
```

Upload:

```bash
export HF_TOKEN=...

uv run python -m hf_export.upload_to_hf \
  --folder exports/model-block-30-34 \
  --repo-id your-name/model-block-30-34
```

## Software-Agent Extension

The linked `software-agents` repository from the course proposal is a syllabus repo, not an executable agent framework. This repo therefore targets a real downstream agent stack such as `mini-swe-agent` for the follow-up study.

Useful entrypoints:

- `scripts/run_agent_study_pipeline.py`
  One-command wrapper that builds conditions + task manifests and then runs the experiment, evaluation, run-record export, and summary steps.
- `scripts/build_agent_study_conditions.py`
  Creates a clean baseline-vs-RYS condition manifest with layer specs and overhead.
- `scripts/create_swebench_manifest.py`
  Freezes an exact list of SWE-bench instance IDs for a reproducible subset.
- `scripts/run_mini_swe_experiment.py`
  Runs each condition through `mini-swe-agent` one instance at a time and records trajectories, predictions, and runtime metadata.
- `scripts/evaluate_swebench_runs.py`
  Sends each condition's `preds.json` through the official SWE-bench harness.
- `scripts/build_agent_run_records.py`
  Converts trajectories + evaluation reports into normalized task-level records.
- `scripts/summarize_agent_runs.py`
  Aggregates per-task run records into success rate, step count, execution errors, and runtime summaries.
- `docs/software_agent_extension.md`
  Describes the recommended evaluation flow for the proposal.

Example flow:

```bash
uv run python scripts/run_agent_study_pipeline.py \
  --model-routes-file configs/agent_eval/model_routes.example.json \
  --output-root results/agent_study/demo_run \
  --num-layers 64 \
  --base-model-id your/model \
  --block 24,35 \
  --block 29,34 \
  --instance-id astropy__astropy-12907 \
  --instance-id django__django-11019 \
  --dry-run
```

This writes `conditions.json` and `manifest.json` under `results/agent_study/demo_run/` and then prints the exact downstream commands it would execute.

The export manifest is written to `rys_export_manifest.json`.

A minimal Colab notebook is available at:

- `hf_export/colab/export_upload_minimal.ipynb`

## Heatmaps

The reusable plotting helpers live in `src/utils/heatmaps.py`.

For standard `(i, j)` scans, the normal entrypoint is:

```bash
uv run python scripts/analyze_results.py \
  --math-scores results/demo/math_results.pkl \
  --eq-scores results/demo/eq_results.pkl \
  --out-dir results/demo/analysis \
  --num-layers 64
```

For per-layer repeat sweeps:

```bash
uv run python scripts/plot_repeat_heatmaps.py \
  --results-file results/demo/repeatx8_math_results.pkl \
  --manifest-file results/demo/repeatx8_manifest.json \
  --out-dir results/demo/repeatx8_heatmaps
```

## Current Assumptions and Limits

- The public scan path is ExLlama-first.
- Beam search uses the Hugging Face worker path rather than ExLlama.
- This repo assumes decoder-layer architectures; unsupported architectures should fail explicitly.
- The HF exporter currently detects decoder stacks under:
  - `model.language_model.layers.`
  - `model.layers.`
  - `language_model.layers.`
- Dataset generation and recalibration are intentionally out of scope here.

## Suggested Reproduction Path

If you are starting from scratch, do this in order:

1. run a small `(i, j)` scan with `math_16 + eq_16`
2. inspect the heatmaps and top balanced configs
3. run beam search seeded from the single-block scan
4. train the surrogate on measured configs and benchmark its top candidates
5. rerun the strongest candidates on `math_120 + eq_140`
6. export any final variants you want to share
