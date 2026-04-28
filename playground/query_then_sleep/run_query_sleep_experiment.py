#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.sleep_time import run_bridge_test
from gsw_memory.query_then_sleep.types import QuerySleepBatchRecord
from gsw_memory.sleep_time.curriculum import split_curriculum_batches

QUERY_RUNNER = REPO_ROOT / "playground" / "query_then_sleep" / "run_query_agent.py"
ANALYSIS_RUNNER = REPO_ROOT / "playground" / "query_then_sleep" / "analyze_query_interactions.py"
GUIDED_SLEEP_RUNNER = REPO_ROOT / "playground" / "query_then_sleep" / "run_guided_sleep_batch.py"
PROCESS_TERMINATION_GRACE_SECONDS = 5.0


class _ExperimentTerminationRequested(KeyboardInterrupt):
    def __init__(self, signum: int):
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        super().__init__(f"Experiment interrupted by signal {signal_name}")
        self.signum = signum


def _set_parent_death_sigterm() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_pdeathsig = 1
        result = libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)
        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
    except Exception:
        return


def _child_process_preexec() -> None:
    _set_parent_death_sigterm()


def _install_signal_handlers(handler) -> Dict[int, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous_handlers: Dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        previous_handlers[int(sig)] = signal.getsignal(sig)
        signal.signal(sig, handler)
    return previous_handlers


def _restore_signal_handlers(previous_handlers: Dict[int, Any]) -> None:
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)


def _terminate_process_group(process: Any, *, grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS) -> None:
    if process is None:
        return
    pid = getattr(process, "pid", None)
    if pid is None:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:
                pass
        return

    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = pid

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:
                pass
        return

    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    while time.monotonic() < deadline:
        try:
            if process.poll() is not None:
                return
        except Exception:
            return
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass


def _build_batch_items(questions_path: str, batch_size: int, seed_batch_size: int, corpus_path: Optional[str] = None):
    questions, schema = run_bridge_test.load_questions(questions_path)
    title_to_idx = None
    corpus_path = corpus_path or run_bridge_test.infer_corpus_path(questions_path)
    if schema == "context_titles":
        title_to_idx = run_bridge_test.build_title_to_doc_index(corpus_path or "")
    items = run_bridge_test.build_batch_query_items(questions, schema=schema, title_to_idx=title_to_idx)
    return split_curriculum_batches(items, batch_size=batch_size, seed_batch_size=seed_batch_size)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_experiment_dataset(
    *,
    questions_path: str,
    corpus_path: Optional[str],
    output_root: Path,
) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    source_path = _resolve_repo_path(questions_path)
    payload = _load_json(source_path)

    if isinstance(payload, list):
        resolved_corpus = str(_resolve_repo_path(corpus_path)) if corpus_path else run_bridge_test.infer_corpus_path(str(source_path))
        return str(source_path), resolved_corpus, None

    if not isinstance(payload, dict):
        raise ValueError(f"Expected questions_path to be a JSON list or manifest object, got {type(payload).__name__}")

    manifest_questions = payload.get("questions", [])
    manifest_questions_path = payload.get("questions_path")
    manifest_corpus_path = payload.get("corpus_path")
    if not isinstance(manifest_questions, list) or not manifest_questions:
        raise ValueError("Manifest object must include a non-empty 'questions' list")
    if not manifest_questions_path:
        raise ValueError("Manifest object must include 'questions_path'")

    source_questions_path = _resolve_repo_path(str(manifest_questions_path))
    source_questions = _load_json(source_questions_path)
    if not isinstance(source_questions, list):
        raise ValueError(f"Expected source questions file to be a JSON list, got {type(source_questions).__name__}")

    ordered_subset: List[Dict[str, Any]] = []
    ordered_manifest_questions: List[Dict[str, Any]] = []
    for order_index, entry in enumerate(manifest_questions):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest question entry {order_index} must be an object")
        original_index = int(entry.get("original_index", -1))
        if original_index < 0 or original_index >= len(source_questions):
            raise IndexError(f"Question index {original_index} is out of range for {source_questions_path}")
        question = copy.deepcopy(source_questions[original_index])
        question_id = question.get("_id", question.get("id", ""))
        expected_question_id = str(entry.get("question_id", "") or "")
        if expected_question_id and question_id != expected_question_id:
            raise ValueError(
                f"Manifest question_id mismatch at original_index={original_index}: "
                f"expected {expected_question_id}, found {question_id or '<missing>'}"
            )
        question["_experiment_original_index"] = original_index
        question["_experiment_family"] = entry.get("family", "")
        question["_experiment_split"] = entry.get("split", "train")
        ordered_subset.append(question)
        ordered_manifest_questions.append(
            {
                "order_index": order_index,
                "original_index": original_index,
                "question_id": question_id,
                "question": question.get("question", ""),
                "family": entry.get("family", ""),
                "split": entry.get("split", "train"),
            }
        )

    subset_path = output_root / "selected_questions.json"
    _write_json(subset_path, ordered_subset)
    resolved_corpus = str(_resolve_repo_path(str(corpus_path or manifest_corpus_path))) if (corpus_path or manifest_corpus_path) else None
    _write_json(
        output_root / "resolved_manifest.json",
        {
            "manifest": payload,
            "ordered_questions": ordered_manifest_questions,
            "resolved_questions_path": str(subset_path),
            "resolved_corpus_path": resolved_corpus,
        },
    )
    return str(subset_path), resolved_corpus, payload


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _tail_lines(path: Path, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]) if lines else ""


def _runner_log_paths(stage_output_dir: Path) -> Dict[str, Path]:
    return {
        "stdout": stage_output_dir / "runner_stdout.log",
        "stderr": stage_output_dir / "runner_stderr.log",
    }


def _run_child_stage(stage_name: str, command: List[str], stage_output_dir: Path) -> Dict[str, Path]:
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    log_paths = _runner_log_paths(stage_output_dir)
    print(f"[{stage_name}] output: {stage_output_dir}", flush=True)
    print(f"[{stage_name}] stdout log: {log_paths['stdout']}", flush=True)
    print(f"[{stage_name}] stderr log: {log_paths['stderr']}", flush=True)

    termination_request: Dict[str, Optional[int]] = {"signum": None}

    def _signal_handler(signum, _frame):
        termination_request["signum"] = int(signum)
        raise _ExperimentTerminationRequested(int(signum))

    previous_handlers = _install_signal_handlers(_signal_handler)
    process = None
    try:
        with open(log_paths["stderr"], "w") as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                cwd=str(REPO_ROOT),
                start_new_session=True,
                preexec_fn=_child_process_preexec if sys.platform.startswith("linux") else None,
                text=True,
                bufsize=1,
            )
            _bridge_count = [0]
            _sleep_tokens = [0]
            with open(log_paths["stdout"], "w") as stdout_file:
                for line in iter(process.stdout.readline, ""):
                    stdout_file.write(line)
                    stdout_file.flush()

                    # Query phase: show question start/finish + cost
                    if any(marker in line for marker in [
                        "\U0001f4ca Running:",
                        "\u2753",
                        "\u2705",
                        "\u274c",
                        "\u26a0",
                    ]):
                        sys.stdout.write(line)
                        sys.stdout.flush()

                    # Sleep phase: track bridges from doc-level summaries only
                    if "bridges_created" in line and "total_explored" in line:
                        try:
                            import re as _re
                            m = _re.search(r"'bridges_created':\s*(\d+)", line)
                            if m and int(m.group(1)) > 0:
                                _bridge_count[0] += int(m.group(1))
                        except Exception:
                            pass
                    if "edge_tokens" in line:
                        try:
                            import re as _re
                            m = _re.search(r"'edge_tokens':\s*(\d+)", line)
                            if m:
                                _sleep_tokens[0] += int(m.group(1))
                        except Exception:
                            pass
                    if "tokens_used" in line:
                        try:
                            import re as _re
                            m = _re.search(r"'tokens_used':\s*(\d+)", line)
                            if m:
                                _sleep_tokens[0] += int(m.group(1))
                        except Exception:
                            pass
                    # Print live status on any edge/doc result line
                    if any(k in line for k in ["Result:", "\U0001f9fe Worker"]):
                        t = _sleep_tokens[0]
                        t_str = f"{t/1_000_000:.1f}M" if t >= 1_000_000 else f"{t/1_000:.1f}K" if t >= 1_000 else str(t)
                        cost = (t / 1_000_000) * 0.15  # rough estimate (input-heavy)
                        sys.stdout.write(f"\r    \U0001f4a4 Sleep: {_bridge_count[0]} bridges | {t_str} tokens (~${cost:.2f})  ")
                        sys.stdout.flush()
            returncode = process.wait()
    except (KeyboardInterrupt, _ExperimentTerminationRequested):
        _terminate_process_group(process)
        raise
    finally:
        _restore_signal_handlers(previous_handlers)

    if returncode != 0:
        stderr_tail = _tail_lines(log_paths["stderr"])
        raise RuntimeError(
            f"Stage '{stage_name}' failed with exit code {returncode}.\n"
            f"stdout log: {log_paths['stdout']}\n"
            f"stderr log: {log_paths['stderr']}\n"
            f"stderr tail:\n{stderr_tail}"
        )
    return log_paths


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _snapshot_bridge_count(snapshot_path: Optional[Path]) -> int:
    if snapshot_path is None or not snapshot_path.exists():
        return 0
    payload = _read_json(snapshot_path)
    return int(payload.get("registry_bridge_count", 0) or 0)


def _resolve_query_model(args: argparse.Namespace) -> str:
    return str(args.query_model or args.model)


def _resolve_analysis_model(args: argparse.Namespace) -> Optional[str]:
    value = args.analysis_model or args.query_model or args.model
    return str(value) if value else None


def _resolve_sleep_model(args: argparse.Namespace) -> str:
    return str(args.sleep_model or args.model)


def _resolve_query_base_url(args: argparse.Namespace) -> Optional[str]:
    return args.query_base_url or args.base_url


def _resolve_analysis_base_url(args: argparse.Namespace) -> Optional[str]:
    return args.analysis_base_url or args.query_base_url or args.base_url


def _resolve_sleep_base_url(args: argparse.Namespace) -> Optional[str]:
    return args.sleep_base_url or args.base_url


def _resolve_root_model(args: argparse.Namespace) -> str:
    return str(args.root_model or _resolve_sleep_model(args))


def _resolve_worker_model(args: argparse.Namespace) -> str:
    return str(args.worker_model or _resolve_sleep_model(args))


def _build_query_command(args: argparse.Namespace, *, batch_index: int, batch_dir: Path, registry_snapshot: Optional[Path]) -> List[str]:
    command = [
        sys.executable,
        str(QUERY_RUNNER),
        "--questions_path", str(getattr(args, "resolved_questions_path", args.questions_path)),
        "--batch_index", str(batch_index),
        "--batch_size", str(args.batch_size),
        "--seed_batch_size", str(args.seed_batch_size),
        "--gsw_path", args.gsw_path,
        "--output_dir", str(batch_dir / "query"),
        "--model", _resolve_query_model(args),
        "--max_iterations", str(args.query_max_iterations),
        "--bridge_query_top_k", str(args.bridge_query_top_k),
        "--bridge_inject_min_score", str(getattr(args, "bridge_inject_min_score", 0.30)),
        "--bridge_inject_top_k", str(getattr(args, "bridge_inject_top_k", 5)),
        "--embedding_gpu_memory_utilization", str(args.embedding_gpu_memory_utilization),
        "--reasoning_effort", str(args.reasoning_effort),
    ]
    query_base_url = _resolve_query_base_url(args)
    if query_base_url:
        command.extend(["--base_url", query_base_url])
    query_corpus_path = getattr(args, "resolved_corpus_path", None) or getattr(args, "corpus_path", None)
    if query_corpus_path:
        command.extend(["--corpus_path", str(query_corpus_path)])
    if registry_snapshot is not None and registry_snapshot.exists():
        command.extend(["--bridge_registry_snapshot", str(registry_snapshot)])
    if args.verbose:
        command.append("--verbose")
    if args.show_thinking:
        command.append("--show-thinking")
    return command


def _build_analysis_command(args: argparse.Namespace, *, batch_index: int, batch_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(ANALYSIS_RUNNER),
        "--traces_path", str(batch_dir / "query" / "query_interaction_traces.jsonl"),
        "--output_dir", str(batch_dir / "analysis"),
        "--batch_index", str(batch_index),
        "--reasoning_effort", str(args.reasoning_effort),
    ]

    # Pass the prior batch's guidance so lessons carry forward between batches.
    # batch_dir has the form ".../batch_{N}", so the prior batch is sibling "batch_{N-1}".
    if batch_index > 0:
        prior_guidance_path = (
            batch_dir.parent / f"batch_{batch_index - 1}" / "analysis" / "interaction_guidance.json"
        )
        if prior_guidance_path.exists():
            command.extend(["--prior_guidance_path", str(prior_guidance_path)])

    analysis_model = _resolve_analysis_model(args)
    if analysis_model:
        command.extend(["--model", analysis_model])
    analysis_base_url = _resolve_analysis_base_url(args)
    if analysis_base_url:
        command.extend(["--base_url", analysis_base_url])
    if args.verbose:
        command.append("--verbose")
    if args.show_thinking:
        command.append("--show-thinking")
    return command


def _build_sleep_command(
    args: argparse.Namespace,
    *,
    sleep_batch_index: int,
    batch_dir: Path,
    registry_snapshot: Optional[Path],
) -> List[str]:
    command = [
        sys.executable,
        str(GUIDED_SLEEP_RUNNER),
        "--questions_path", str(getattr(args, "resolved_questions_path", args.questions_path)),
        "--batch_index", str(sleep_batch_index),
        "--batch_size", str(args.batch_size),
        "--seed_batch_size", str(args.seed_batch_size),
        "--gsw_path", args.gsw_path,
        "--output_dir", str(batch_dir / f"sleep_for_batch_{sleep_batch_index}"),
        "--guidance_file", str(batch_dir / "analysis" / "interaction_guidance.json"),
        "--model", _resolve_sleep_model(args),
        "--root_model", _resolve_root_model(args),
        "--worker_model", _resolve_worker_model(args),
        "--embedding_gpu_memory_utilization", str(args.embedding_gpu_memory_utilization),
        "--pipeline_mode", str(args.pipeline_mode),
        "--hybrid_scope", str(args.hybrid_scope),
        "--edge_max_depth", str(args.edge_max_depth),
        "--edge_max_calls", str(args.edge_max_calls),
        "--edge_max_tokens", str(args.edge_max_tokens),
        "--max_optional_docs_per_edge", str(args.max_optional_docs_per_edge),
        "--edge_parallel_workers", str(args.edge_parallel_workers),
        "--verifier_parallel_workers", str(args.verifier_parallel_workers),
        "--edge_max_accepted_bridges", str(args.edge_max_accepted_bridges),
        "--parallel_progress_heartbeat_seconds", str(args.parallel_progress_heartbeat_seconds),
        "--parallel_stuck_warning_seconds", str(args.parallel_stuck_warning_seconds),
        "--curriculum_generation_parallel_workers", str(args.curriculum_generation_parallel_workers),
        "--targeted_edge_ratio", str(getattr(args, "targeted_edge_ratio", 0.7)),
        "--max_tokens", str(args.max_tokens),
        "--max_iterations", str(args.max_iterations),
        "--reasoning_effort", str(args.reasoning_effort),
        "--bridge_verifier_threshold", str(args.bridge_verifier_threshold),
    ]
    sleep_base_url = _resolve_sleep_base_url(args)
    if sleep_base_url:
        command.extend(["--base_url", sleep_base_url])
    sleep_corpus_path = getattr(args, "resolved_corpus_path", None) or getattr(args, "corpus_path", None)
    if sleep_corpus_path:
        command.extend(["--corpus_path", str(sleep_corpus_path)])
    if registry_snapshot is not None and registry_snapshot.exists():
        command.extend(["--bridge_registry_snapshot", str(registry_snapshot)])
    if args.edge_parallel_enabled:
        command.append("--edge_parallel_enabled")
    if args.curriculum_generation_parallel_enabled:
        command.append("--curriculum_generation_parallel_enabled")
    if args.disable_bridge_verifier:
        command.append("--disable_bridge_verifier")
    if args.bridge_verifier_fail_open:
        command.append("--bridge_verifier_fail_open")
    if args.bridge_verifier_model:
        command.extend(["--bridge_verifier_model", str(args.bridge_verifier_model)])
    if args.verbose:
        command.append("--verbose")
    if args.show_thinking:
        command.append("--show-thinking")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated query-then-sleep experiment")
    parser.add_argument("--manifest", "--questions_path", dest="questions_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gsw_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--query_model", default=None)
    parser.add_argument("--query_base_url", default=None)
    parser.add_argument("--analysis_model", default=None)
    parser.add_argument("--analysis_base_url", default=None)
    parser.add_argument("--sleep_model", default=None)
    parser.add_argument("--sleep_base_url", default=None)
    parser.add_argument("--root_model", default=None)
    parser.add_argument("--worker_model", default=None)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--seed_batch_size", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--query_max_iterations", type=int, default=20)
    parser.add_argument("--bridge_query_top_k", type=int, default=5)
    parser.add_argument("--bridge_inject_min_score", type=float, default=0.30)
    parser.add_argument("--bridge_inject_top_k", type=int, default=5)
    parser.add_argument("--embedding_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--pipeline_mode", default="hybrid")
    parser.add_argument("--hybrid_scope", default="doc_edge")
    parser.add_argument("--edge_max_depth", type=int, default=1)
    parser.add_argument("--edge_max_calls", type=int, default=2)
    parser.add_argument("--edge_max_tokens", type=int, default=3000)
    parser.add_argument("--max_optional_docs_per_edge", type=int, default=2)
    parser.add_argument("--edge_parallel_enabled", action="store_true")
    parser.add_argument("--edge_parallel_workers", type=int, default=1)
    parser.add_argument("--verifier_parallel_workers", type=int, default=4)
    parser.add_argument("--edge_max_accepted_bridges", type=int, default=10)
    parser.add_argument("--parallel_progress_heartbeat_seconds", type=float, default=10.0)
    parser.add_argument("--parallel_stuck_warning_seconds", type=float, default=60.0)
    parser.add_argument("--curriculum_generation_parallel_enabled", action="store_true")
    parser.add_argument("--curriculum_generation_parallel_workers", type=int, default=1)
    parser.add_argument("--targeted_edge_ratio", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=10000000)
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--reasoning_effort", default="medium")
    parser.add_argument("--disable_bridge_verifier", action="store_true")
    parser.add_argument("--bridge_verifier_threshold", type=float, default=0.7)
    parser.add_argument("--bridge_verifier_fail_open", action="store_true")
    parser.add_argument("--bridge_verifier_model", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_progress_path = output_root / "experiment_progress.jsonl"
    resolved_questions_path, resolved_corpus_path, resolved_manifest = _resolve_experiment_dataset(
        questions_path=args.questions_path,
        corpus_path=args.corpus_path,
        output_root=output_root,
    )
    args.resolved_questions_path = resolved_questions_path
    args.resolved_corpus_path = resolved_corpus_path
    if int(args.batch_size or 0) <= 0:
        manifest_batch_size = int((resolved_manifest or {}).get("curriculum_batch_size", 8) or 8)
        args.batch_size = manifest_batch_size
    if int(args.seed_batch_size or 0) <= 0:
        manifest_seed_batch_size = int((resolved_manifest or {}).get("curriculum_seed_batch_size", args.batch_size) or args.batch_size)
        args.seed_batch_size = manifest_seed_batch_size

    batches = _build_batch_items(
        args.resolved_questions_path,
        args.batch_size,
        args.seed_batch_size,
        corpus_path=args.resolved_corpus_path,
    )
    if args.max_batches and args.max_batches > 0:
        batches = batches[: args.max_batches]

    learning_curve: List[Dict[str, Any]] = []
    current_registry_snapshot: Optional[Path] = None
    targeted_edge_ratio = max(0.4, min(0.9, float(args.targeted_edge_ratio)))

    for batch_index, _batch_items in enumerate(batches):
        batch_dir = output_root / f"batch_{batch_index}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        _append_jsonl(
            experiment_progress_path,
            {"phase": "batch", "status": "started", "batch_index": batch_index},
        )

        num_batch_qs = len(_batch_items) if _batch_items else "?"
        print(flush=True)
        print(f"\u2550" * 70, flush=True)
        print(f"  BATCH {batch_index}/{len(batches)} \u2014 {num_batch_qs} questions", flush=True)
        print(f"\u2550" * 70, flush=True)

        bridges_before = _snapshot_bridge_count(current_registry_snapshot)

        # ── QUERY PHASE ──
        print(f"\n  \u25b6 QUERY PHASE ({num_batch_qs} questions)", flush=True)
        query_dir = batch_dir / "query"
        _append_jsonl(experiment_progress_path, {"phase": "query", "status": "started", "batch_index": batch_index})
        _run_child_stage("query", _build_query_command(args, batch_index=batch_index, batch_dir=batch_dir, registry_snapshot=current_registry_snapshot), query_dir)
        query_snapshot = query_dir / "bridge_registry_snapshot.json"
        if query_snapshot.exists():
            current_registry_snapshot = query_snapshot
        query_result = _read_json(query_dir / "query_agent_results.json")
        _append_jsonl(experiment_progress_path, {"phase": "query", "status": "completed", "batch_index": batch_index})

        # Print query results summary
        q_metrics = query_result.get("overall_metrics", {})
        q_traces = query_result.get("traces", [])
        print(f"  \u2713 Query complete: F1={q_metrics.get('F1', 0):.3f} | EM={q_metrics.get('ExactMatch', 0):.3f} | {len(q_traces)} questions", flush=True)
        for t in q_traces:
            f1 = float(t.get("f1", 0) or 0)
            icon = "\u2705" if f1 >= 0.8 else ("\u26a0\ufe0f" if f1 > 0 else "\u274c")
            q_text = str(t.get("question", ""))
            ans = str(t.get("final_answer", "")).strip()
            gold_list = t.get("gold_answers", [])
            gold = str(gold_list[0] if gold_list else "")
            bridge = " \U0001f309" if t.get("bridge_evidence_used") else ""
            iters = t.get("tool_iterations", 0)
            missing = t.get("missing_relations", [])
            missing_str = f" | missing: {', '.join(str(m) for m in missing)}" if missing else ""
            print(f"    {icon} F1={f1:.2f} | {q_text}", flush=True)
            print(f"          Answer: {ans or '(empty)'}", flush=True)
            print(f"          Gold:   {gold} | {iters} calls{bridge}{missing_str}", flush=True)

        # ── ANALYSIS PHASE ──
        print(f"\n  \u25b6 ANALYSIS PHASE", flush=True)
        analysis_dir = batch_dir / "analysis"
        _append_jsonl(experiment_progress_path, {"phase": "analysis", "status": "started", "batch_index": batch_index})
        _run_child_stage("analysis", _build_analysis_command(args, batch_index=batch_index, batch_dir=batch_dir), analysis_dir)
        guidance_summary = _read_json(analysis_dir / "interaction_guidance.json")
        _append_jsonl(experiment_progress_path, {"phase": "analysis", "status": "completed", "batch_index": batch_index})

        # Print analysis summary
        gap_profile = guidance_summary.get("relation_gap_profile", [])
        missing_count = sum(int(r.get("missing", 0) or 0) for r in gap_profile)
        nl_summary = str(guidance_summary.get("natural_language_summary", "") or "")
        print(f"  \u2713 Analysis complete: {missing_count} missing relations identified", flush=True)
        if nl_summary:
            print(f"    \U0001f4dd {nl_summary}", flush=True)

        # ── SLEEP PHASE ──
        bridges_generated = 0
        bridges_after_sleep = bridges_before
        if batch_index + 1 < len(batches):
            usage_signal = float(q_metrics.get("bridge_answer_usage_rate", q_metrics.get("bridge_usage_rate", 0.0)) or 0.0)
            if bridges_before > 0:
                previous_ratio = targeted_edge_ratio
                if usage_signal < 0.2:
                    targeted_edge_ratio = max(0.4, targeted_edge_ratio - 0.2)
                elif usage_signal > 0.8:
                    targeted_edge_ratio = min(0.9, targeted_edge_ratio + 0.15)
                if targeted_edge_ratio != previous_ratio:
                    print(
                        f"  Adjusted sleep targeted_edge_ratio: {previous_ratio:.2f} → {targeted_edge_ratio:.2f} "
                        f"(bridge answer usage={usage_signal:.2f})",
                        flush=True,
                    )
            args.targeted_edge_ratio = targeted_edge_ratio
            print(f"\n  \u25b6 SLEEP PHASE (generating bridges for batch {batch_index + 1})", flush=True)
            sleep_dir = batch_dir / f"sleep_for_batch_{batch_index + 1}"
            _append_jsonl(experiment_progress_path, {"phase": "sleep", "status": "started", "batch_index": batch_index, "sleep_batch_index": batch_index + 1})
            _run_child_stage(
                "guided_sleep",
                _build_sleep_command(args, sleep_batch_index=batch_index + 1, batch_dir=batch_dir, registry_snapshot=current_registry_snapshot),
                sleep_dir,
            )
            sleep_result = _read_json(sleep_dir / "guided_sleep_result.json")
            bridges_generated = int(sleep_result.get("bridge_count", 0) or 0)
            snapshot_path = sleep_result.get("snapshot_path")
            if snapshot_path:
                current_registry_snapshot = Path(snapshot_path)
            else:
                candidate_snapshot = sleep_dir / "bridge_registry_snapshot.json"
                if candidate_snapshot.exists():
                    current_registry_snapshot = candidate_snapshot
            bridges_after_sleep = _snapshot_bridge_count(current_registry_snapshot)
            _append_jsonl(experiment_progress_path, {"phase": "sleep", "status": "completed", "batch_index": batch_index, "sleep_batch_index": batch_index + 1, "bridge_count": bridges_generated})
            print(f"  \u2713 Sleep complete: {bridges_generated} bridges generated ({bridges_after_sleep} total)", flush=True)
        else:
            print(f"\n  \u23f9 Last batch \u2014 no sleep phase needed", flush=True)

        # ── BATCH SUMMARY ──
        overall_metrics = dict(query_result.get("overall_metrics", {}) or {})
        print(f"\n  {'='*50}", flush=True)
        print(f"  Batch {batch_index} summary: F1={overall_metrics.get('F1', 0):.3f} | EM={overall_metrics.get('ExactMatch', 0):.3f} | Bridges: {bridges_before}\u2192{bridges_after_sleep} (+{bridges_generated})", flush=True)
        print(f"  {'='*50}", flush=True)

        missing_relation_counts = {
            str(row.get("relation_label", "") or ""): int(row.get("missing", 0) or 0)
            for row in guidance_summary.get("relation_gap_profile", []) or []
            if str(row.get("relation_label", "") or "").strip()
        }
        learning_curve.append(
            QuerySleepBatchRecord(
                batch_index=batch_index,
                questions_answered=int(overall_metrics.get("questions_answered", len(query_result.get("traces", []) or [])) or 0),
                avg_f1=float(overall_metrics.get("F1", 0.0) or 0.0),
                avg_em=float(overall_metrics.get("ExactMatch", 0.0) or 0.0),
                bridges_before_batch=bridges_before,
                bridges_generated_for_next_batch=bridges_generated,
                bridges_after_sleep=bridges_after_sleep,
                bridge_usage_rate=float(overall_metrics.get("bridge_usage_rate", 0.0) or 0.0),
                bridge_answer_usage_rate=float(overall_metrics.get("bridge_answer_usage_rate", 0.0) or 0.0),
                sleep_targeted_edge_ratio=float(targeted_edge_ratio),
                missing_relation_counts=missing_relation_counts,
            ).as_dict()
        )
        _write_json(output_root / "learning_curve.json", learning_curve)
        _append_jsonl(
            experiment_progress_path,
            {
                "phase": "batch",
                "status": "completed",
                "batch_index": batch_index,
                "avg_f1": float(overall_metrics.get("F1", 0.0) or 0.0),
                "bridges_generated_for_next_batch": bridges_generated,
            },
        )


if __name__ == "__main__":
    main()
