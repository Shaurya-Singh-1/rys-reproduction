#!/usr/bin/env python3
"""Build a small SWE-bench fault-localization dataset.

The generated examples ask a model to rank candidate files from a bug report and
pre-fix code snippets. The gold patch is used only for dataset construction.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.experiment import resolve_dataset_name


DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$")
HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")
SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
}


def import_load_dataset():
    """Import Hugging Face datasets without being shadowed by ./datasets."""
    original_path = list(sys.path)
    root_resolved = ROOT.resolve()
    sys.path = [
        item
        for item in sys.path
        if Path(item or os.getcwd()).resolve() != root_resolved
    ]
    cached = sys.modules.get("datasets")
    if cached is not None and not hasattr(cached, "load_dataset"):
        sys.modules.pop("datasets", None)
    try:
        return importlib.import_module("datasets").load_dataset
    finally:
        sys.path = original_path


@dataclass(frozen=True)
class ChangedFile:
    path: str
    old_starts: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SWE-bench fault-localization dataset.")
    parser.add_argument("--subset", default="lite", help="Dataset alias or full dataset name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--output", default="datasets/localization_dataset.json")
    parser.add_argument("--repo-cache", default="local/swebench_repos")
    parser.add_argument("--max-examples", type=int, default=15)
    parser.add_argument("--scan-limit", type=int, default=200, help="Maximum dataset rows to inspect.")
    parser.add_argument("--instance-id", action="append", default=[], help="Specific instance id. Repeatable.")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle the full dataset before scanning/filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractors", type=int, default=6, help="Number of distractor files per example.")
    parser.add_argument("--snippet-lines", type=int, default=70, help="Maximum lines per snippet.")
    parser.add_argument("--min-snippet-lines", type=int, default=20, help="Minimum useful lines per snippet.")
    parser.add_argument("--max-changed-files", type=int, default=1)
    parser.add_argument("--include-tests", action="store_true", help="Allow test files as gold/distractors.")
    parser.add_argument("--redo-repos", action="store_true", help="Fetch repos even when cache exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected examples without writing JSON.")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result.stdout


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name.lower()
    return (
        "test" in parts
        or "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".spec.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.ts")
    )


def is_source_path(path: str, *, include_tests: bool) -> bool:
    if Path(path).suffix.lower() not in SOURCE_EXTENSIONS:
        return False
    if not include_tests and is_test_path(path):
        return False
    return True


def parse_changed_files(patch_text: str, *, include_tests: bool) -> list[ChangedFile]:
    files: dict[str, list[int]] = {}
    current: str | None = None

    for line in patch_text.splitlines():
        header = DIFF_HEADER_RE.match(line)
        if header:
            current = header.group(2)
            if is_source_path(current, include_tests=include_tests):
                files.setdefault(current, [])
            continue

        if current is None or current not in files:
            continue
        hunk = HUNK_RE.match(line)
        if hunk:
            files[current].append(int(hunk.group("old_start")))

    return [
        ChangedFile(path=path, old_starts=tuple(starts or [1]))
        for path, starts in files.items()
    ]


def extract_file_patch(patch_text: str, target_path: str) -> str:
    """Return the diff block for one target file from a unified patch."""
    lines = patch_text.splitlines()
    selected: list[str] = []
    in_target = False

    for line in lines:
        header = DIFF_HEADER_RE.match(line)
        if header:
            if in_target and selected:
                break
            in_target = header.group(2) == target_path
        if in_target:
            selected.append(line)

    return "\n".join(selected)


def repo_dir_for(cache_root: Path, repo: str) -> Path:
    return cache_root / repo.replace("/", "__")


def ensure_repo(repo: str, base_commit: str, cache_root: Path, *, redo: bool) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_dir = repo_dir_for(cache_root, repo)
    url = f"https://github.com/{repo}.git"

    if not repo_dir.exists():
        run(["git", "clone", "--filter=blob:none", url, str(repo_dir)])
    elif redo:
        run(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)

    run(["git", "checkout", "--force", base_commit], cwd=repo_dir)
    return repo_dir


def read_repo_file(repo_dir: Path, rel_path: str) -> list[str] | None:
    path = repo_dir / rel_path
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1").splitlines()
        except UnicodeDecodeError:
            return None


def snippet_around(lines: list[str], center_line: int, max_lines: int) -> tuple[str, int, int]:
    if not lines:
        return "", 1, 1
    half = max(1, max_lines // 2)
    center_idx = max(0, min(len(lines) - 1, center_line - 1))
    start = max(0, center_idx - half)
    end = min(len(lines), start + max_lines)
    start = max(0, end - max_lines)
    return "\n".join(lines[start:end]), start + 1, end


def first_interesting_line(lines: list[str]) -> int:
    patterns = [
        re.compile(r"^\s*(def|class|async def)\s+"),
        re.compile(r"^\s*(function|export function|export class|class)\s+"),
        re.compile(r"^\s*(public|private|protected)?\s*(class|interface|void|int|String|boolean)\s+"),
    ]
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            continue
        if any(pattern.match(line) for pattern in patterns):
            return idx
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            return idx
    return 1


def list_candidate_files(repo_dir: Path, correct_path: str, *, include_tests: bool) -> list[str]:
    correct = Path(correct_path)
    directory = repo_dir / correct.parent
    suffix = correct.suffix.lower()
    candidates: list[str] = []

    search_roots = [directory, directory.parent if directory.parent != directory else directory]
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(f"*{suffix}")):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_dir).as_posix()
            if rel == correct_path or rel in seen:
                continue
            if not is_source_path(rel, include_tests=include_tests):
                continue
            seen.add(rel)
            candidates.append(rel)
    return candidates


def choose_distractors(
    repo_dir: Path,
    correct_path: str,
    *,
    count: int,
    include_tests: bool,
) -> list[str]:
    candidates = list_candidate_files(repo_dir, correct_path, include_tests=include_tests)
    correct_parent = Path(correct_path).parent

    def score(path: str) -> tuple[int, int, str]:
        parent = Path(path).parent
        same_dir = 0 if parent == correct_parent else 1
        depth_gap = abs(len(parent.parts) - len(correct_parent.parts))
        return same_dir, depth_gap, path

    return sorted(candidates, key=score)[:count]


def build_example(
    sample: dict[str, Any],
    *,
    repo_cache: Path,
    distractor_count: int,
    snippet_lines: int,
    min_snippet_lines: int,
    max_changed_files: int,
    include_tests: bool,
    redo_repos: bool,
) -> tuple[dict[str, Any] | None, str]:
    patch = str(sample.get("patch", ""))
    changed = parse_changed_files(patch, include_tests=include_tests)
    if len(changed) == 0:
        return None, "no source file in patch"
    if len(changed) > max_changed_files:
        return None, f"patch changes {len(changed)} source files"

    correct = changed[0]
    true_edit = extract_file_patch(patch, correct.path)
    repo = str(sample["repo"])
    base_commit = str(sample["base_commit"])
    repo_dir = ensure_repo(repo, base_commit, repo_cache, redo=redo_repos)

    correct_lines = read_repo_file(repo_dir, correct.path)
    if correct_lines is None:
        return None, f"correct file missing/unreadable: {correct.path}"
    if len(correct_lines) < min_snippet_lines:
        return None, f"correct file too short: {correct.path}"

    center = correct.old_starts[0]
    correct_snippet, correct_start, correct_end = snippet_around(correct_lines, center, snippet_lines)
    if len(correct_snippet.splitlines()) < min_snippet_lines:
        return None, "correct snippet too short"

    distractor_paths = choose_distractors(
        repo_dir,
        correct.path,
        count=distractor_count,
        include_tests=include_tests,
    )
    if len(distractor_paths) < distractor_count:
        return None, f"only found {len(distractor_paths)} distractors"

    candidate_files = [
        {
            "file_path": correct.path,
            "code_snippet": correct_snippet,
            "start_line": correct_start,
            "end_line": correct_end,
        }
    ]

    for distractor in distractor_paths:
        lines = read_repo_file(repo_dir, distractor)
        if lines is None or len(lines) < min_snippet_lines:
            continue
        focus = first_interesting_line(lines)
        snippet, start, end = snippet_around(lines, focus, snippet_lines)
        if len(snippet.splitlines()) < min_snippet_lines:
            continue
        candidate_files.append(
            {
                "file_path": distractor,
                "code_snippet": snippet,
                "start_line": start,
                "end_line": end,
            }
        )

    if len(candidate_files) < distractor_count + 1:
        return None, "not enough usable distractor snippets"

    random.Random(str(sample["instance_id"])).shuffle(candidate_files)
    return {
        "issue_id": str(sample["instance_id"]),
        "repo": repo,
        "base_commit": base_commit,
        "bug_report": str(sample.get("problem_statement", "")),
        "candidate_files": candidate_files,
        "ground_truth": correct.path,
        "true_edit": true_edit,
        "metadata": {
            "source_dataset": "SWE-bench",
            "changed_files_from_patch": [item.path for item in changed],
            "patch_hunk_starts": list(correct.old_starts),
            "construction_note": (
                "Gold patch/true_edit is included for human review and must not be "
                "shown to the model during localization evaluation."
            ),
        },
    }, "ok"


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dataset = import_load_dataset()
    dataset_name = resolve_dataset_name(args.subset)
    rows = list(load_dataset(dataset_name, split=args.split))
    if args.instance_id:
        wanted = set(args.instance_id)
        rows = [dict(row) for row in rows if str(row.get("instance_id")) in wanted]
    else:
        rows = [dict(row) for row in rows]
        if args.shuffle:
            random.Random(args.seed).shuffle(rows)
        rows = rows[: args.scan_limit]
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    repo_cache = Path(args.repo_cache)
    examples: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for idx, sample in enumerate(rows):
        if len(examples) >= args.max_examples:
            break
        issue_id = str(sample.get("instance_id", f"row-{idx}"))
        try:
            example, reason = build_example(
                sample,
                repo_cache=repo_cache,
                distractor_count=args.distractors,
                snippet_lines=args.snippet_lines,
                min_snippet_lines=args.min_snippet_lines,
                max_changed_files=args.max_changed_files,
                include_tests=args.include_tests,
                redo_repos=args.redo_repos,
            )
        except Exception as exc:
            example, reason = None, f"error: {exc}"

        if example is None:
            skipped.append({"issue_id": issue_id, "reason": reason})
            print(f"[skip] {issue_id}: {reason}")
            continue

        examples.append(example)
        print(
            f"[keep] {issue_id}: {example['ground_truth']} "
            f"({len(example['candidate_files'])} candidates)"
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset": args.subset,
        "split": args.split,
        "example_count": len(examples),
        "skipped": skipped,
    }

    if args.dry_run:
        print(json.dumps(examples, indent=2)[:8000])
        print(f"\nSelected {len(examples)} examples; skipped {len(skipped)} rows.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(examples, indent=2))
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(examples)} examples to {output}")
    print(f"Wrote build summary to {summary_path}")
    print(f"Skipped {len(skipped)} rows")


if __name__ == "__main__":
    main()
