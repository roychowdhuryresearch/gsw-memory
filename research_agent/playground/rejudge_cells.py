"""Re-score existing cell logs with the LLM-as-judge.

Walks one or more cell directories under ``logs/``, loads each
``cell_result.json`` (or reconstructs one from per-question traces when
the cell was killed before the harness wrote the aggregate), runs the
``LLMJudge`` on every (question, gold, predicted) triple, and writes a
sibling ``cell_result_judged.json`` with ``judge_correct`` +
``judge_reason`` filled in plus recomputed headline numbers.

Why:
    First pilot cells were scored with exact-match + token F1 only.
    EM is near-zero for long-form FRAMES answers; the real signal is
    the judge verdict. This script is idempotent — rerunning overwrites
    ``cell_result_judged.json`` but leaves the original intact.

Typical usage::

    python playground/rejudge_cells.py --all
    python playground/rejudge_cells.py logs/search_o1__Qwen_QwQ-32B__20260420_090959
    python playground/rejudge_cells.py --glob "logs/vanilla_rag_react*" --judge-model gpt-4o

``--glob`` matches cell directories (not files). ``--all`` = every
subdirectory of ``logs/`` that has a cell_result.json OR a ``traces/``
directory with at least one ``q_*.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_dotenv_files() -> None:
    """Load ``research_agent/.env`` and the parent repo's ``.env``.

    The research_agent workspace's own .env rarely has OPENAI_API_KEY; the
    parent gsw-memory/.env does. Loading both (parent first, then local so
    local can override) + not clobbering already-set env vars mirrors how
    the adapter CLIs bootstrap credentials.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = [
        REPO_ROOT.parent / ".env",  # gsw-memory/.env
        REPO_ROOT / ".env",         # research_agent/.env
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_dotenv_files()

from research_agent.eval.frames_dataset import FramesQuestion, load_frames  # noqa: E402
from research_agent.eval.llm_judge import JudgeVerdict, LLMJudge  # noqa: E402


# ---------------------------------------------------------------------------
# Cell discovery + loading
# ---------------------------------------------------------------------------


@dataclass
class CellPaths:
    root: Path
    cell_result: Path | None
    traces_dir: Path | None

    @property
    def has_cell_result(self) -> bool:
        return self.cell_result is not None and self.cell_result.exists()

    @property
    def has_traces(self) -> bool:
        return self.traces_dir is not None and self.traces_dir.exists()


def discover_cells(logs_dir: Path) -> list[CellPaths]:
    cells: list[CellPaths] = []
    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir():
            continue
        cr = d / "cell_result.json"
        td = d / "traces"
        has_any_trace = td.exists() and any(td.glob("q_*.json"))
        if cr.exists() or has_any_trace:
            cells.append(
                CellPaths(
                    root=d,
                    cell_result=cr if cr.exists() else None,
                    traces_dir=td if td.exists() else None,
                )
            )
    return cells


def _load_cell_result(cell: CellPaths) -> dict[str, Any] | None:
    if not cell.has_cell_result:
        return None
    try:
        return json.loads(cell.cell_result.read_text())
    except json.JSONDecodeError as exc:
        print(f"[warn] {cell.cell_result}: invalid JSON ({exc}); skipping")
        return None


def _reconstruct_from_traces(
    cell: CellPaths, gold_by_id: dict[str, FramesQuestion]
) -> dict[str, Any] | None:
    """Build a minimal cell_result payload from per-question traces.

    Used when the harness never wrote the aggregate (e.g. run was Ctrl-C'd
    mid-cell). We emit only the fields the re-judge pass needs: questions
    with ``question``, ``gold_answer``, ``predicted_answer``, ``trajectory``.
    Missing metadata (system_id, model_id) is inferred from the cell dir
    name as ``<system>__<model>__<timestamp>``.
    """
    if not cell.has_traces:
        return None
    trace_files = sorted(cell.traces_dir.glob("q_*.json"))
    if not trace_files:
        return None

    # Dir name convention: "<system_id>__<model_id_slug>__<timestamp>".
    parts = cell.root.name.split("__")
    system_id = parts[0] if parts else ""
    model_id = parts[1].replace("_", "/") if len(parts) >= 2 else ""
    questions: list[dict[str, Any]] = []
    for tf in trace_files:
        try:
            traj = json.loads(tf.read_text())
        except json.JSONDecodeError:
            continue
        qid = str(traj.get("question_id") or tf.stem.removeprefix("q_"))
        fq = gold_by_id.get(qid)
        if fq is None:
            continue
        questions.append(
            {
                "question_id": qid,
                "question": fq.question,
                "gold_answer": fq.answer,
                "predicted_answer": traj.get("final_answer") or "",
                "trajectory": traj,
                "exact_match": False,
                "f1": 0.0,
                "judge_correct": None,
                "failure_mode": None,
                "notes": "",
            }
        )
    if not questions:
        return None
    return {
        "system_id": system_id,
        "model_id": model_id,
        "benchmark": "frames",
        "subset_id": "reconstructed_from_traces",
        "questions": questions,
        "config": {"reconstructed": True},
    }


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def _judge_one(
    judge: LLMJudge, q: dict[str, Any]
) -> tuple[str, JudgeVerdict]:
    verdict = judge.judge(
        question=q.get("question") or "",
        gold=q.get("gold_answer") or "",
        predicted=q.get("predicted_answer") or "",
    )
    return str(q.get("question_id")), verdict


def rejudge_cell(
    cell: CellPaths,
    *,
    judge: LLMJudge,
    gold_by_id: dict[str, FramesQuestion],
    force: bool,
    concurrency: int,
) -> Path | None:
    payload = _load_cell_result(cell) or _reconstruct_from_traces(cell, gold_by_id)
    if payload is None:
        print(f"[skip] {cell.root.name}: no cell_result.json and no usable traces")
        return None

    qs = payload.get("questions") or []
    if not qs:
        print(f"[skip] {cell.root.name}: 0 questions in payload")
        return None

    # Back-fill gold + question text from the benchmark when the cell
    # payload is thin (reconstructed runs won't have them yet).
    for q in qs:
        qid = str(q.get("question_id") or "")
        fq = gold_by_id.get(qid)
        if fq is None:
            continue
        if not q.get("question"):
            q["question"] = fq.question
        if not q.get("gold_answer"):
            q["gold_answer"] = fq.answer
        if not q.get("predicted_answer"):
            traj = q.get("trajectory") or {}
            q["predicted_answer"] = traj.get("final_answer") or ""

    # Decide which questions still need judging.
    to_judge: list[dict[str, Any]] = []
    for q in qs:
        if force or q.get("judge_correct") is None:
            to_judge.append(q)

    sys_label = payload.get("system_id") or cell.root.name.split("__")[0]
    model_label = payload.get("model_id") or "?"
    print(
        f"[run] {cell.root.name}: system={sys_label} model={model_label} "
        f"n={len(qs)} to_judge={len(to_judge)}"
    )

    # Run the judge concurrently — these are independent API calls.
    if to_judge:
        verdicts: dict[str, JudgeVerdict] = {}
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(_judge_one, judge, q): q for q in to_judge}
            for i, fut in enumerate(as_completed(futures), 1):
                qid, verdict = fut.result()
                verdicts[qid] = verdict
                mark = "✓" if verdict.correct else "✗"
                print(
                    f"  [{i}/{len(to_judge)}] q{qid} {mark} "
                    f"{(verdict.reason or verdict.error or '')[:100]}"
                )

        by_id = {str(q.get("question_id")): q for q in qs}
        for qid, v in verdicts.items():
            q = by_id.get(qid)
            if q is None:
                continue
            q["judge_correct"] = v.correct
            q["judge_reason"] = v.reason
            q["judge_model"] = judge.model
            existing_notes = q.get("notes") or ""
            note = f"judge={v.correct}; reason={v.reason}" if v.reason else f"judge={v.correct}"
            q["notes"] = (existing_notes + " | " if existing_notes else "") + note
            if v.error:
                traj = q.setdefault("trajectory", {})
                extra = traj.setdefault("extra", {})
                extra.setdefault("judge_errors", []).append(v.error)

    # Recompute aggregate stats.
    n_total = len(qs)
    n_correct_em = sum(1 for q in qs if q.get("exact_match"))
    judged_qs = [q for q in qs if q.get("judge_correct") is not None]
    n_judge_seen = len(judged_qs)
    n_judge_correct = sum(1 for q in judged_qs if q.get("judge_correct"))
    payload["n_total"] = n_total
    payload["n_correct"] = n_correct_em
    payload["accuracy"] = (n_correct_em / n_total) if n_total else 0.0
    payload["n_judge_seen"] = n_judge_seen
    payload["n_judge_correct"] = n_judge_correct
    payload["judge_accuracy"] = (n_judge_correct / n_judge_seen) if n_judge_seen else 0.0
    payload["judge_model"] = judge.model
    payload["rejudged_at"] = datetime.now(timezone.utc).isoformat()

    out_path = cell.root / "cell_result_judged.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"[done] {cell.root.name}: em={n_correct_em}/{n_total} "
        f"judge={n_judge_correct}/{n_judge_seen} -> {out_path.name}"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-score existing cell logs with the LLM-as-judge."
    )
    p.add_argument(
        "paths",
        nargs="*",
        help="Cell directories to re-judge. If omitted, use --all or --glob.",
    )
    p.add_argument("--all", action="store_true", help="Re-judge every cell under logs/")
    p.add_argument(
        "--glob",
        default="",
        help="Glob pattern (relative to repo root) matching cell directories",
    )
    p.add_argument(
        "--logs-dir",
        default=str(REPO_ROOT / "logs"),
        help="Root logs directory (default: ./logs)",
    )
    p.add_argument(
        "--judge-model", default="gpt-4o", help="LLM judge model (default: gpt-4o)"
    )
    p.add_argument(
        "--judge-base-url",
        default="",
        help="OpenAI-compatible base URL for a non-OpenAI judge (vLLM, etc.)",
    )
    p.add_argument("--judge-api-key", default="", help="API key for the judge endpoint")
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel judge calls per cell (default: 8)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-judge questions that already have a verdict",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be re-judged without making API calls",
    )
    return p.parse_args(argv)


def _resolve_targets(args: argparse.Namespace) -> list[CellPaths]:
    logs_dir = Path(args.logs_dir)
    all_cells = discover_cells(logs_dir)
    by_name = {c.root.name: c for c in all_cells}

    targets: list[CellPaths] = []
    if args.all:
        targets = list(all_cells)
    if args.glob:
        from fnmatch import fnmatch
        matched = [c for c in all_cells if fnmatch(str(c.root), args.glob)
                   or fnmatch(c.root.name, args.glob)]
        targets.extend(matched)
    for raw in args.paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (REPO_ROOT / raw).resolve() if (REPO_ROOT / raw).exists() else p.resolve()
        match = by_name.get(p.name) or next(
            (c for c in all_cells if c.root == p), None
        )
        if match is None:
            # Allow passing any dir directly even if it's not under logs/.
            cr = p / "cell_result.json"
            td = p / "traces"
            if cr.exists() or (td.exists() and any(td.glob("q_*.json"))):
                match = CellPaths(
                    root=p,
                    cell_result=cr if cr.exists() else None,
                    traces_dir=td if td.exists() else None,
                )
            else:
                print(f"[warn] {raw}: not a cell dir (no cell_result.json / traces/)")
                continue
        targets.append(match)

    # Deduplicate by root while preserving order.
    seen: set[Path] = set()
    deduped: list[CellPaths] = []
    for c in targets:
        if c.root in seen:
            continue
        seen.add(c.root)
        deduped.append(c)
    return deduped


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = _resolve_targets(args)
    if not targets:
        print("No cells selected. Pass paths, --all, or --glob.")
        return 1

    print(f"Cells to process: {len(targets)}")
    for c in targets:
        print(f"  - {c.root.name}")

    if args.dry_run:
        return 0

    gold = {fq.id: fq for fq in load_frames(split="dev")}
    judge = LLMJudge(
        model=args.judge_model,
        base_url=args.judge_base_url or None,
        api_key=args.judge_api_key or None,
    )

    for c in targets:
        try:
            rejudge_cell(
                c,
                judge=judge,
                gold_by_id=gold,
                force=args.force,
                concurrency=args.concurrency,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[err ] {c.root.name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
