from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent_eval.local_repo_bench import extract_first_json_object


SYSTEM_PROMPT = """You are evaluating a software bug report for fault localization.

Given a bug report and candidate source files, rank the files by how likely they are to contain the root cause.

Return exactly one JSON object and no markdown:
{"ranked_files": ["path/to/file.py", "..."], "rationale": "short reason"}

Rules:
- Only use file paths from the candidate list.
- Include every candidate file exactly once in ranked_files.
- Rank most likely first.
"""


@dataclass(frozen=True)
class LocalizationExample:
    issue_id: str
    repo: str
    bug_report: str
    candidate_files: list[dict[str, Any]]
    ground_truth: str
    base_commit: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def candidate_paths(self) -> list[str]:
        return [str(item["file_path"]) for item in self.candidate_files]


def load_localization_dataset(path: Path) -> list[LocalizationExample]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "examples" in raw:
        rows = raw["examples"]
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported localization dataset format in {path}")

    examples: list[LocalizationExample] = []
    for row in rows:
        examples.append(
            LocalizationExample(
                issue_id=str(row["issue_id"]),
                repo=str(row.get("repo", "")),
                base_commit=str(row.get("base_commit", "")),
                bug_report=str(row["bug_report"]),
                candidate_files=list(row["candidate_files"]),
                ground_truth=str(row["ground_truth"]),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return examples


def build_localization_messages(example: LocalizationExample) -> tuple[list[dict[str, str]], str]:
    candidate_blocks: list[str] = []
    for idx, item in enumerate(example.candidate_files, start=1):
        path = str(item["file_path"])
        start = item.get("start_line")
        end = item.get("end_line")
        line_range = f" lines {start}-{end}" if start is not None and end is not None else ""
        snippet = str(item.get("code_snippet", "")).rstrip()
        candidate_blocks.append(
            f"Candidate {idx}: {path}{line_range}\n"
            "```text\n"
            f"{snippet}\n"
            "```"
        )

    candidates_text = "\n\n".join(candidate_blocks)
    user_prompt = (
        f"Issue ID: {example.issue_id}\n"
        f"Repository: {example.repo}\n\n"
        "Bug report:\n"
        f"{example.bug_report.strip()}\n\n"
        "Candidate files:\n\n"
        f"{candidates_text}\n\n"
        "Rank the candidate file paths from most likely root cause to least likely."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], user_prompt


def parse_ranked_files(raw_text: str, candidate_paths: list[str]) -> tuple[list[str], dict[str, Any]]:
    parse_meta: dict[str, Any] = {"parser": "", "errors": []}
    allowed = set(candidate_paths)

    try:
        payload = extract_first_json_object(raw_text)
        ranked_raw = payload.get("ranked_files", payload.get("ranking", []))
        if isinstance(ranked_raw, list):
            ranked = [str(item).strip() for item in ranked_raw]
        else:
            ranked = []
        parse_meta["parser"] = "json"
        parse_meta["payload"] = payload
    except Exception as exc:
        ranked = []
        parse_meta["errors"].append(f"json_parse_failed: {exc}")

    if not ranked:
        parse_meta["parser"] = "path_order_fallback"
        positions: list[tuple[int, str]] = []
        for path in candidate_paths:
            match = re.search(re.escape(path), raw_text)
            if match:
                positions.append((match.start(), path))
        ranked = [path for _, path in sorted(positions)]

    normalized: list[str] = []
    seen: set[str] = set()
    for path in ranked:
        if path in allowed and path not in seen:
            normalized.append(path)
            seen.add(path)

    for path in candidate_paths:
        if path not in seen:
            normalized.append(path)
            seen.add(path)

    parse_meta["valid_ranked_count"] = len([path for path in ranked if path in allowed])
    parse_meta["completed_missing_candidates"] = len(candidate_paths) - parse_meta["valid_ranked_count"]
    return normalized, parse_meta


def score_ranking(ranked_files: list[str], ground_truth: str) -> dict[str, Any]:
    try:
        rank = ranked_files.index(ground_truth) + 1
    except ValueError:
        rank = None
    return {
        "rank": rank,
        "top1": rank == 1,
        "top3": rank is not None and rank <= 3,
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def summarize_localization_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(str(record["condition"]), []).append(record)

    summary: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        n = len(rows)
        if n == 0:
            continue
        top1 = sum(1 for row in rows if row["score"]["top1"])
        top3 = sum(1 for row in rows if row["score"]["top3"])
        mrr = sum(float(row["score"]["mrr"]) for row in rows) / n
        avg_rank_values = [int(row["score"]["rank"]) for row in rows if row["score"]["rank"] is not None]
        avg_rank = sum(avg_rank_values) / len(avg_rank_values) if avg_rank_values else None
        summary.append(
            {
                "condition": condition,
                "n": n,
                "top1_accuracy": top1 / n,
                "top3_accuracy": top3 / n,
                "mrr": mrr,
                "avg_rank": avg_rank,
            }
        )
    return summary
