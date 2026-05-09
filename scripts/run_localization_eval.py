#!/usr/bin/env python3
"""Run baseline and repeated-layer fault-localization inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.localization import (  # noqa: E402
    build_localization_messages,
    load_localization_dataset,
    parse_ranked_files,
    score_ranking,
    summarize_localization_records,
)
from src.core.layer_config import (  # noqa: E402
    baseline_layers,
    layer_spec_string,
    normalize_to_layers,
    parse_blocks_string,
)
from src.core.layer_duplicator import build_model_with_layers  # noqa: E402
from src.core.layer_duplicator_moe import build_model_with_layers_moe  # noqa: E402
from src.workers.model_utils import (  # noqa: E402
    apply_chat_template_fallback,
    get_text_num_layers,
    is_moe_model,
    load_model_and_tokenizer,
    parse_device_map_arg,
    parse_max_memory_json,
    strip_thinking,
)


def parse_dtype(name: str) -> torch.dtype:
    raw = str(name).strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[raw]


def parse_condition(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError("--condition must be NAME=SPEC, e.g. rys_16_20=blocks:16,20")
    name, spec = raw.split("=", 1)
    name = name.strip()
    spec = spec.strip()
    if not name or not spec:
        raise ValueError("--condition must include non-empty NAME and SPEC")
    return name, spec


def condition_specs_from_args(args: argparse.Namespace, num_layers: int) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    if args.include_baseline:
        layers = baseline_layers(num_layers)
        conditions.append(
            {
                "name": "baseline",
                "spec": layer_spec_string(layers),
                "layers": layers,
                "extra_layers": 0,
                "overhead_fraction": 0.0,
            }
        )

    for raw in args.block:
        blocks = parse_blocks_string(raw)
        if len(blocks) != 1:
            raise ValueError(f"--block expects one block per flag, got: {raw}")
        i, j = blocks[0]
        spec = f"blocks:{i},{j}"
        layers = normalize_to_layers(num_layers, spec)
        conditions.append(
            {
                "name": f"rys_{i}_{j}",
                "spec": spec,
                "layers": layers,
                "extra_layers": len(layers) - num_layers,
                "overhead_fraction": (len(layers) - num_layers) / num_layers,
            }
        )

    for raw in args.condition:
        name, spec = parse_condition(raw)
        layers = normalize_to_layers(num_layers, spec)
        conditions.append(
            {
                "name": name,
                "spec": spec,
                "layers": layers,
                "extra_layers": len(layers) - num_layers,
                "overhead_fraction": (len(layers) - num_layers) / num_layers,
            }
        )

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for condition in conditions:
        name = str(condition["name"])
        if name in seen:
            raise ValueError(f"Duplicate condition name: {name}")
        seen.add(name)
        deduped.append(condition)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fault-localization eval with baseline and RYS layers.")
    parser.add_argument("--dataset", default="datasets/localization_dataset.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--block", action="append", default=[], help="RYS block like 16,20. Repeatable.")
    parser.add_argument("--condition", action="append", default=[], help="Custom NAME=SPEC. Repeatable.")
    parser.add_argument("--no-baseline", dest="include_baseline", action="store_false", default=True)
    parser.add_argument("--start-index", type=int, default=0, help="0-based dataset index to start from.")
    parser.add_argument("--num-examples", type=int, default=None, help="Number of examples to run from --start-index.")
    parser.add_argument("--limit-examples", type=int, default=None, help="Legacy alias for --num-examples from index 0.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-impl", default="sdpa", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-json", default=None)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--redo-existing", action="store_true")
    return parser.parse_args()


def generate_answer(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    prompt = apply_chat_template_fallback(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    started = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    text = strip_thinking(raw_text)
    return text, {
        "elapsed_seconds": time.time() - started,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "completion_tokens": int(generated_ids.shape[0]),
        "prompt_chars": len(prompt),
        "completion_chars": len(text),
    }


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    all_examples = load_localization_dataset(Path(args.dataset))
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.num_examples is not None and args.num_examples < 0:
        raise ValueError("--num-examples must be >= 0")
    if args.limit_examples is not None and args.limit_examples < 0:
        raise ValueError("--limit-examples must be >= 0")
    if args.limit_examples is not None and args.num_examples is not None:
        raise ValueError("Use only one of --num-examples or --limit-examples.")

    num_examples = args.num_examples
    if args.limit_examples is not None:
        if args.start_index != 0:
            raise ValueError("--limit-examples is a legacy alias from index 0; use --num-examples with --start-index.")
        num_examples = args.limit_examples

    end_index = None if num_examples is None else args.start_index + num_examples
    examples = all_examples[args.start_index : end_index]
    if not examples:
        raise ValueError(f"No examples loaded from {args.dataset}")

    resolved_device_map = parse_device_map_arg(args.device_map)
    resolved_max_memory = parse_max_memory_json(args.max_memory_json)
    tokenizer, base_model, load_meta = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=parse_dtype(args.dtype),
        device_map=resolved_device_map,
        attn_implementation=(None if args.attention_impl == "eager" else args.attention_impl),
        max_memory=resolved_max_memory,
        cpu_offload=args.cpu_offload,
        offload_folder=args.offload_folder,
    )
    num_layers = get_text_num_layers(base_model)
    model_is_moe = is_moe_model(base_model)
    layer_builder = build_model_with_layers_moe if model_is_moe else build_model_with_layers
    conditions = condition_specs_from_args(args, num_layers)

    existing = [] if args.redo_existing else load_existing_records(records_path)
    completed = {(row["condition"], row["issue_id"]) for row in existing}
    records: list[dict[str, Any]] = existing

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_total_examples": len(all_examples),
        "dataset_start_index": args.start_index,
        "dataset_num_examples": len(examples),
        "selected_issue_ids": [example.issue_id for example in examples],
        "model_path": args.model_path,
        "load_meta": load_meta,
        "num_layers": num_layers,
        "model_type": "moe" if model_is_moe else "dense",
        "conditions": [
            {
                "name": item["name"],
                "spec": item["spec"],
                "extra_layers": item["extra_layers"],
                "overhead_fraction": item["overhead_fraction"],
            }
            for item in conditions
        ],
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "device_map": args.device_map,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    for condition in conditions:
        condition_name = str(condition["name"])
        layers = list(condition["layers"])
        run_model = base_model if layers == baseline_layers(num_layers) else layer_builder(base_model, layers)
        print(f"\n[{condition_name}] {condition['spec']} ({len(layers)} layers)")

        for idx, example in enumerate(examples, start=1):
            if (condition_name, example.issue_id) in completed:
                print(f"[skip] {condition_name} {example.issue_id}: existing")
                continue
            messages, user_prompt = build_localization_messages(example)
            raw_answer, generation_meta = generate_answer(
                model=run_model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
            )
            ranked_files, parse_meta = parse_ranked_files(raw_answer, example.candidate_paths)
            score = score_ranking(ranked_files, example.ground_truth)
            record = {
                "condition": condition_name,
                "layer_spec": condition["spec"],
                "layer_indices": layers,
                "extra_layers": condition["extra_layers"],
                "overhead_fraction": condition["overhead_fraction"],
                "issue_id": example.issue_id,
                "repo": example.repo,
                "base_commit": example.base_commit,
                "ground_truth": example.ground_truth,
                "candidate_paths": example.candidate_paths,
                "ranked_files": ranked_files,
                "score": score,
                "raw_answer": raw_answer,
                "parse_meta": parse_meta,
                "generation_meta": generation_meta,
                "messages": messages if args.save_prompts else None,
                "user_prompt": user_prompt if args.save_prompts else None,
            }
            append_record(records_path, record)
            records.append(record)
            completed.add((condition_name, example.issue_id))
            print(
                f"[{idx}/{len(examples)}] {example.issue_id} "
                f"rank={score['rank']} top1={score['top1']} "
                f"elapsed={generation_meta['elapsed_seconds']:.1f}s"
            )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_path": str(records_path),
        "summary": summarize_localization_records(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
