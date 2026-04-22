#!/usr/bin/env python3
"""
Run a two-arm curriculum-guidance ablation over a fixed 2Wiki lookup subset.

This script materializes the tracked manifest subset, launches:
- guidance_on
- guidance_off

and writes a comparison report plus retrieved-bridge family exports.
"""

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
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


DEFAULT_MANIFEST_PATH = REPO_ROOT / "playground_data" / "experiments" / "2wiki_lookup_mixed_8q.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "experiments" / "2wiki_lookup_mixed_guidance_ablation"
RUNNER_PATH = REPO_ROOT / "playground" / "sleep_time" / "run_bridge_test.py"
ARM_CONFIGS = {
    "guidance_on": {"disable_guidance": False, "decay_rate": 0.0},
    "guidance_off": {"disable_guidance": True, "decay_rate": 0.0},
    "guidance_decay": {"disable_guidance": False, "decay_rate": 0.5},
}
ARM_NAMES = tuple(ARM_CONFIGS.keys())
PROGRESS_POLL_INTERVAL_SECONDS = 1.0
SHARD_STATUS_TTL_SECONDS = 5.0
PROCESS_TERMINATION_GRACE_SECONDS = 5.0
_LAST_PROGRESS_RENDERED_LINES = 0
_SHARD_STATUS_RENDER_CACHE: Dict[str, Dict[str, Any]] = {}


class _ExperimentTerminationRequested(KeyboardInterrupt):
    """Raised when the parent process receives a termination signal."""

    def __init__(self, signum: int):
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        super().__init__(f"Experiment interrupted by signal {signal_name}")
        self.signum = signum


def _set_parent_death_sigterm() -> None:
    """Ensure a Linux child receives SIGTERM if the parent dies unexpectedly."""
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
        # Best effort only; parent-side process-group cleanup remains the primary path.
        return


def _child_process_preexec() -> None:
    _set_parent_death_sigterm()


def _install_signal_handlers(handler) -> Dict[int, Any]:
    """Install temporary signal handlers from the main thread only."""
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
    """Hard-stop a child process session and any descendants."""
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


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _runner_log_paths(arm_output_dir: Path) -> Dict[str, Path]:
    return {
        "stdout": arm_output_dir / "runner_stdout.log",
        "stderr": arm_output_dir / "runner_stderr.log",
    }


def _tail_lines(path: Path, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _phase_sort_key(phase: str) -> int:
    return {
        "bridge_generation_shards": 0,
        "bridge_pattern_classification": 1,
        "bridge_index_rebuild": 2,
        "query_answering": 3,
    }.get(phase, 99)


def _load_json_lines(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_shard_tool_log(shard_dir: Path, *, min_mtime_ns: Optional[int] = None) -> Optional[Path]:
    run_dirs = sorted(
        [path for path in shard_dir.glob("run_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    for run_dir in reversed(run_dirs):
        tool_log = run_dir / "logs" / "tool_calls.jsonl"
        if not tool_log.exists():
            continue
        if min_mtime_ns is not None:
            try:
                if tool_log.stat().st_mtime_ns < int(min_mtime_ns):
                    continue
            except OSError:
                continue
        return tool_log
    return None


def _short_edge_label(edge: Dict[str, Any]) -> str:
    if not isinstance(edge, dict):
        return ""
    entity = str(edge.get("entity_name", "") or "").strip()
    neighbor = str(edge.get("neighbor_name", "") or "").strip()
    relationship = str(edge.get("relationship", "") or "").strip()
    if not (entity or neighbor or relationship):
        return ""
    label = f"{entity}->{neighbor}"
    if relationship:
        label += f" ({relationship})"
    if len(label) > 54:
        return label[:51] + "..."
    return label


def _shard_cache_key(shard_dir: Path) -> str:
    try:
        return str(shard_dir.resolve())
    except OSError:
        return str(shard_dir)


def _clone_shard_status(shard_status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in shard_status.items()
    }


def _load_shard_live_status(shard_dir: Path, *, now_monotonic: Optional[float] = None) -> Dict[str, Any]:
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    metadata_path = shard_dir / "shard_metadata.json"
    metadata = _load_json_file(metadata_path)
    doc_ids = [str(doc_id) for doc_id in metadata.get("doc_ids", []) or [] if str(doc_id).strip()]
    doc_id_set = set(doc_ids)
    min_mtime_ns: Optional[int] = None
    try:
        min_mtime_ns = metadata_path.stat().st_mtime_ns
    except OSError:
        min_mtime_ns = None
    tool_log = _latest_shard_tool_log(shard_dir, min_mtime_ns=min_mtime_ns)
    events = _load_json_lines(tool_log) if tool_log is not None else []

    completed_docs: set[str] = set()
    started_docs: set[str] = set()
    latest_parallel_event: Optional[Dict[str, Any]] = None
    latest_worker_call: Optional[Dict[str, Any]] = None
    latest_timestamp = ""
    latest_parallel_timestamp = ""
    latest_doc_summary: Optional[Dict[str, Any]] = None
    latest_doc_summary_timestamp = ""
    latest_document_exploration: Optional[Dict[str, Any]] = None

    for event in events:
        event_type = str(event.get("event_type", "") or "")
        data = event.get("data", {}) or {}
        timestamp = str(event.get("timestamp", "") or "")
        if timestamp >= latest_timestamp:
            latest_timestamp = timestamp

        if event_type in {
            "parallel_batch_started",
            "parallel_batch_progress",
            "parallel_batch_heartbeat",
            "parallel_worker_stuck_warning",
            "parallel_batch_completed",
        }:
            latest_parallel_event = {"event_type": event_type, **data}
            latest_parallel_timestamp = timestamp
            doc_id = str(data.get("doc_id", "") or "").strip()
            if doc_id:
                started_docs.add(doc_id)
            if event_type == "parallel_batch_completed" and doc_id:
                completed_docs.add(doc_id)
        elif event_type == "rlm_doc_summary":
            doc_id = str(data.get("entity", "") or data.get("doc_id", "") or "").strip()
            if doc_id:
                completed_docs.add(doc_id)
            latest_doc_summary = data if isinstance(data, dict) else None
            latest_doc_summary_timestamp = timestamp
            document_exploration = data.get("document_exploration") if isinstance(data, dict) else None
            if isinstance(document_exploration, dict):
                latest_document_exploration = document_exploration
        elif event_type == "rlm_worker_call":
            latest_worker_call = data if isinstance(data, dict) else None

    docs_total = len(doc_ids) if doc_ids else len(started_docs | completed_docs)
    docs_completed = len(completed_docs)
    if docs_total > 0:
        docs_completed = min(docs_completed, docs_total)
    docs_remaining = max(docs_total - docs_completed, 0)

    if latest_document_exploration:
        total_docs = latest_document_exploration.get("total_docs")
        total_explored = latest_document_exploration.get("total_explored")
        remaining_docs = latest_document_exploration.get("remaining_docs")
        try:
            if total_docs is not None:
                docs_total = max(0, int(total_docs))
            if total_explored is not None:
                docs_completed = max(0, int(total_explored))
                if docs_total > 0:
                    docs_completed = min(docs_completed, docs_total)
            if isinstance(remaining_docs, list):
                docs_remaining = len(remaining_docs)
            else:
                docs_remaining = max(docs_total - docs_completed, 0)
        except (TypeError, ValueError):
            pass

    current_doc_id = ""
    edge_completed = None
    edge_total = None
    inflight_edge_count = 0
    elapsed_seconds = None
    worker_inflight_seconds = None
    current_edge_label = ""
    state = "pending"

    if tool_log is not None:
        state = "initializing"

    if latest_parallel_event:
        current_doc_id = str(latest_parallel_event.get("doc_id", "") or "").strip()
        if doc_id_set and current_doc_id and current_doc_id not in doc_id_set:
            current_doc_id = ""
        edge_completed = int(latest_parallel_event.get("completed", 0) or 0)
        edge_total = int(latest_parallel_event.get("total", 0) or 0)
        inflight_edges = latest_parallel_event.get("inflight_edges", []) or []
        inflight_edge_count = len(inflight_edges)
        elapsed_seconds = float(latest_parallel_event.get("elapsed_seconds", 0.0) or 0.0)
        worker_inflight_seconds = float(latest_parallel_event.get("worker_inflight_seconds", 0.0) or 0.0)
        if inflight_edges:
            current_edge_label = _short_edge_label(inflight_edges[0])
        elif isinstance(latest_parallel_event.get("completed_edge"), dict):
            current_edge_label = _short_edge_label(latest_parallel_event.get("completed_edge", {}))
        event_type = str(latest_parallel_event.get("event_type", "") or "")
        if event_type == "parallel_worker_stuck_warning":
            state = "stuck"
        elif event_type == "parallel_batch_completed":
            state = "done" if docs_remaining == 0 else "next_doc"
            current_doc_id = ""
        else:
            state = "running"
    elif latest_worker_call:
        edge = latest_worker_call.get("edge", {}) or {}
        current_doc_id = str(edge.get("doc_id", "") or "").strip()
        current_edge_label = _short_edge_label(edge)
        state = "running"

    if docs_total > 0 and docs_completed >= docs_total:
        state = "done"

    completed_doc_id = str((latest_doc_summary or {}).get("entity", "") or (latest_doc_summary or {}).get("doc_id", "") or "").strip()
    if (
        current_doc_id
        and completed_doc_id
        and current_doc_id == completed_doc_id
        and latest_doc_summary_timestamp
        and latest_doc_summary_timestamp >= latest_parallel_timestamp
    ):
        current_doc_id = ""
        if docs_total > 0 and docs_completed >= docs_total:
            state = "done"
        else:
            state = "next_doc"

    return {
        "shard_id": int(metadata.get("shard_id", int(str(shard_dir.name).split("_")[-1]))),
        "question_ids": list(metadata.get("question_ids", []) or []),
        "questions": list(metadata.get("questions", []) or []),
        "doc_ids": doc_ids,
        "docs_total": docs_total,
        "docs_completed": docs_completed,
        "docs_remaining": docs_remaining,
        "current_doc_id": current_doc_id,
        "edge_completed": edge_completed,
        "edge_total": edge_total,
        "inflight_edge_count": inflight_edge_count,
        "elapsed_seconds": elapsed_seconds,
        "worker_inflight_seconds": worker_inflight_seconds,
        "current_edge_label": current_edge_label,
        "state": state,
        "latest_timestamp": latest_timestamp,
        "tool_log_path": str(tool_log) if tool_log is not None else "",
    }


def _apply_shard_status_render_cache(
    shard_dir: Path,
    shard_status: Dict[str, Any],
    *,
    now_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    cache_key = _shard_cache_key(shard_dir)
    cached_entry = _SHARD_STATUS_RENDER_CACHE.get(cache_key)
    current_state = str(shard_status.get("state", "pending") or "pending")
    has_live_event = current_state not in {"pending", "initializing"}

    if has_live_event:
        _SHARD_STATUS_RENDER_CACHE[cache_key] = {
            "last_seen_monotonic": float(now_monotonic),
            "status": _clone_shard_status(shard_status),
        }
        return shard_status

    if cached_entry is None:
        return shard_status

    cached_status = dict(cached_entry.get("status", {}) or {})
    cached_state = str(cached_status.get("state", "pending") or "pending")
    if cached_state in {"pending", "initializing"}:
        return shard_status

    age_seconds = max(0.0, float(now_monotonic) - float(cached_entry.get("last_seen_monotonic", now_monotonic)))
    if age_seconds <= SHARD_STATUS_TTL_SECONDS:
        reused_status = _clone_shard_status(cached_status)
        reused_status["state_stale"] = True
        reused_status["stale_seconds"] = age_seconds
        return reused_status

    waiting_status = _clone_shard_status(cached_status)
    waiting_status["state"] = "waiting"
    waiting_status["state_stale"] = True
    waiting_status["stale_seconds"] = age_seconds
    waiting_status["current_edge_label"] = str(waiting_status.get("current_edge_label", "") or "")
    return waiting_status


def _load_batch_shard_statuses(batch_dir: Path, *, now_monotonic: Optional[float] = None) -> List[Dict[str, Any]]:
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    shard_root = batch_dir / "generation_shards"
    if not shard_root.exists():
        return []
    statuses = [
        _apply_shard_status_render_cache(
            shard_dir,
            _load_shard_live_status(shard_dir, now_monotonic=now_monotonic),
            now_monotonic=now_monotonic,
        )
        for shard_dir in sorted(
            [path for path in shard_root.glob("shard_*") if path.is_dir()],
            key=lambda path: int(path.name.split("_")[1]) if "_" in path.name else -1,
        )
    ]
    return statuses


def _latest_batch_progress_snapshot(arm_output_dir: Path) -> Optional[Dict[str, Any]]:
    snapshot_monotonic = time.monotonic()
    latest_batch_index = -1
    latest_progress_path: Optional[Path] = None
    latest_batch_dir: Optional[Path] = None
    latest_progress_mtime_ns = -1
    batch_dirs = sorted(
        arm_output_dir.glob("batch_*"),
        key=lambda path: int(path.name.split("_")[1]) if "_" in path.name else -1,
    )
    for batch_dir in batch_dirs:
        progress_path = batch_dir / "curriculum_progress.jsonl"
        if not progress_path.exists():
            continue
        batch_index = int(batch_dir.name.split("_")[1])
        try:
            progress_mtime_ns = progress_path.stat().st_mtime_ns
        except OSError:
            progress_mtime_ns = -1
        if (
            progress_mtime_ns > latest_progress_mtime_ns
            or (
                progress_mtime_ns == latest_progress_mtime_ns
                and batch_index >= latest_batch_index
            )
        ):
            latest_progress_mtime_ns = progress_mtime_ns
            latest_batch_index = batch_index
            latest_progress_path = progress_path
            latest_batch_dir = batch_dir

    if latest_progress_path is None:
        return None

    phases: Dict[str, Dict[str, Any]] = {}
    latest_timestamp = ""
    for raw_line in latest_progress_path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        phase = str(event.get("phase", "") or "")
        if not phase:
            continue
        phases[phase] = event
        timestamp = str(event.get("timestamp", "") or "")
        if timestamp >= latest_timestamp:
            latest_timestamp = timestamp

    # Check for probe progress
    probe_progress = None
    probe_progress_path = arm_output_dir / "probe_progress.json"
    if probe_progress_path.exists():
        try:
            probe_progress = json.loads(probe_progress_path.read_text(errors="replace"))
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "batch_index": latest_batch_index,
        "timestamp": latest_timestamp,
        "phases": phases,
        "probe_progress": probe_progress,
        "shards": _load_batch_shard_statuses(
            latest_batch_dir or latest_progress_path.parent,
            now_monotonic=snapshot_monotonic,
        ),
    }


def _format_generation_status(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "shards ?/?"
    completed = int(event.get("completed", 0) or 0)
    total = int(event.get("total", 0) or 0)
    merged_bridge_count = event.get("merged_bridge_count")
    detail = f"shards {completed}/{total}" if total > 0 else "shards"
    if merged_bridge_count is not None:
        detail += f" merged={int(merged_bridge_count)}"
    return detail


def _format_pattern_status(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "patterns ?/?"
    completed = int(event.get("completed", 0) or 0)
    total = int(event.get("total", 0) or 0)
    llm_tagged = int(event.get("llm_tagged_surface_count", 0) or 0)
    uncached = int(event.get("uncached_surface_count", 0) or 0)
    failed = int(event.get("failed_surface_count", 0) or 0)
    detail = f"patterns {completed}/{total}"
    if uncached > 0 or llm_tagged > 0 or failed > 0:
        detail += f" llm={llm_tagged}/{uncached}"
        if failed > 0:
            detail += f" fail={failed}"
    return detail


def _format_index_status(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "index ?"
    status = str(event.get("status", "progress") or "progress")
    surface_count = event.get("surface_count")
    message = str(event.get("message", "") or "")
    detail = "index done" if status == "completed" else "index"
    if surface_count is not None:
        detail += f" surfaces={int(surface_count)}"
    if message:
        detail += f" ({message})"
    return detail


def _format_query_status(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "queries ?/?"
    completed = int(event.get("completed", 0) or 0)
    total = int(event.get("total", 0) or 0)
    detail = f"queries {completed}/{total}" if total > 0 else "queries"
    question_id = str(event.get("question_id", "") or "")
    if question_id:
        detail += f" q={question_id}"
    f1 = event.get("f1")
    exact_match = event.get("exact_match")
    if exact_match is not None and f1 is not None:
        detail += f" em={float(exact_match or 0.0):.2f} f1={float(f1 or 0.0):.2f}"
    bridge_evidence_used = event.get("bridge_evidence_used")
    if bridge_evidence_used is not None:
        detail += f" bridge={'y' if bridge_evidence_used else 'n'}"
    return detail


_STATE_ICONS = {"running": "\u25b6", "done": "\u2713", "pending": "\u23f3", "stuck": "\u26a0", "next_doc": "\u25b6", "initializing": "\u23f3"}
_INPUT_PRICE_PER_M = 0.15
_OUTPUT_PRICE_PER_M = 0.60


def _truncate(text: str, max_len: int = 15) -> str:
    return text[:max_len] if len(text) > max_len else text


def _format_cost(input_tokens: int, output_tokens: int) -> str:
    cost = (input_tokens / 1_000_000) * _INPUT_PRICE_PER_M + (output_tokens / 1_000_000) * _OUTPUT_PRICE_PER_M
    return f"${cost:.2f}"


def _format_tokens_short(total: int) -> str:
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    if total >= 1_000:
        return f"{total / 1_000:.0f}K"
    return str(total)


def _format_shard_status_line(shard_status: Dict[str, Any]) -> str:
    shard_id = int(shard_status.get("shard_id", -1) or -1)
    docs_completed = int(shard_status.get("docs_completed", 0) or 0)
    docs_total = int(shard_status.get("docs_total", 0) or 0)
    state = str(shard_status.get("state", "pending") or "pending")
    elapsed_seconds = shard_status.get("elapsed_seconds")
    current_edge_label = str(shard_status.get("current_edge_label", "") or "").strip()
    stale_seconds = shard_status.get("stale_seconds")
    state_stale = bool(shard_status.get("state_stale", False))
    worker_inflight_seconds = shard_status.get("worker_inflight_seconds")

    icon = _STATE_ICONS.get(state, "?")
    parts = [f"  S{shard_id} {icon} {docs_completed}/{docs_total} docs"]
    if current_edge_label:
        parts.append(f"edge: {current_edge_label}")
    if elapsed_seconds is not None and float(elapsed_seconds or 0.0) > 0:
        parts.append(f"{float(elapsed_seconds or 0.0):.0f}s")
    if state_stale and stale_seconds is not None and float(stale_seconds or 0.0) > 5:
        parts.append(f"stale {float(stale_seconds or 0.0):.0f}s")
    if state == "stuck" and worker_inflight_seconds is not None and float(worker_inflight_seconds or 0.0) > 0:
        parts.append(f"stuck {float(worker_inflight_seconds or 0.0):.0f}s")
    return " | ".join(parts)


def _compute_arm_tokens(arm_output_dir: Path) -> Dict[str, int]:
    """Accumulate token counts from all completed shard results.json files."""
    total_input = 0
    total_output = 0
    import glob as _glob
    for results_file in _glob.glob(str(arm_output_dir / "batch_*" / "generation_shards" / "shard_*" / "run_*" / "results.json")):
        try:
            with open(results_file) as f:
                rd = json.load(f)
            summary = rd.get("summary", {})
            total_input += int(summary.get("input_tokens", 0) or 0)
            total_output += int(summary.get("output_tokens", 0) or 0)
        except Exception:
            pass
    # Also count probe shards
    for results_file in _glob.glob(str(arm_output_dir / "probe_batch_*" / "generation_shards" / "shard_*" / "run_*" / "results.json")):
        try:
            with open(results_file) as f:
                rd = json.load(f)
            summary = rd.get("summary", {})
            total_input += int(summary.get("input_tokens", 0) or 0)
            total_output += int(summary.get("output_tokens", 0) or 0)
        except Exception:
            pass
    return {"input": total_input, "output": total_output, "total": total_input + total_output}


def _format_progress_status(arm_name: str, snapshot: Optional[Dict[str, Any]], arm_output_dir: Optional[Path] = None) -> str:
    if not snapshot:
        return f"[{arm_name}] waiting for first batch progress"
    batch_index_raw = snapshot.get("batch_index", -1)
    batch_index = int(batch_index_raw) if batch_index_raw is not None else -1
    phases = dict(snapshot.get("phases", {}) or {})
    phase_bits = [
        _format_generation_status(phases.get("bridge_generation_shards")),
        _format_pattern_status(phases.get("bridge_pattern_classification")),
        _format_index_status(phases.get("bridge_index_rebuild")),
        _format_query_status(phases.get("query_answering")),
    ]

    # Cost tracking
    cost_bit = ""
    if arm_output_dir is not None:
        tokens = _compute_arm_tokens(arm_output_dir)
        if tokens["total"] > 0:
            cost_str = _format_cost(tokens["input"], tokens["output"])
            cost_bit = (
                f" | {_format_tokens_short(tokens['input'])} in + "
                f"{_format_tokens_short(tokens['output'])} out = "
                f"{_format_tokens_short(tokens['total'])} ({cost_str})"
            )

    # Probe progress
    probe_progress = snapshot.get("probe_progress")
    probe_bit = ""
    if probe_progress and isinstance(probe_progress, dict):
        probe_stage = probe_progress.get("stage", "")
        probe_batch = int(probe_progress.get("batch_index", -1))
        if probe_batch == batch_index:
            if probe_stage == "done":
                pf1 = probe_progress.get("probe_f1", 0)
                pem = probe_progress.get("probe_em", 0)
                pbr = probe_progress.get("probe_bridges", 0)
                probe_bit = f" | probe: F1={pf1:.3f} EM={pem:.3f} br={pbr}"
            else:
                probe_bit = f" | probe: {probe_stage}"

    header = f"[{arm_name}] batch {batch_index}{cost_bit}"
    lines = [header + " | " + " | ".join(phase_bits) + probe_bit]

    shards = snapshot.get("shards", []) or []
    # Deduplicate by shard_id
    seen_ids: set[int] = set()
    unique_shards: list[Dict[str, Any]] = []
    for s in shards:
        sid = int(s.get("shard_id", -1) or -1)
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_shards.append(s)

    # Only show running/stuck shards + count of done/pending
    active_states = {"running", "stuck", "initializing", "next_doc"}
    running = [s for s in unique_shards if str(s.get("state", "")).strip() in active_states]
    done_count = sum(1 for s in unique_shards if str(s.get("state", "")).strip() == "done")
    pending_count = sum(1 for s in unique_shards if str(s.get("state", "")).strip() == "pending")

    for shard_status in running:
        lines.append(_format_shard_status_line(shard_status))
    if done_count > 0 or pending_count > 0:
        summary_parts = []
        if done_count > 0:
            summary_parts.append(f"\u2713 {done_count} done")
        if pending_count > 0:
            summary_parts.append(f"\u23f3 {pending_count} pending")
        lines.append(f"  ({', '.join(summary_parts)})")
    return "\n".join(lines)


def _use_inplace_progress_rendering() -> bool:
    render_mode = os.environ.get("GSW_EXPERIMENT_PROGRESS_MODE", "auto").strip().lower()
    if render_mode in {"plain", "newline"}:
        return False
    if render_mode == "inplace":
        return True
    if not hasattr(sys.stdout, "isatty") or not bool(sys.stdout.isatty()):
        return False
    return str(os.environ.get("TERM", "") or "").strip().lower() not in {"", "dumb"}


def _print_compact_progress(line: str) -> None:
    global _LAST_PROGRESS_RENDERED_LINES
    if not _use_inplace_progress_rendering():
        print(line, flush=True)
        _LAST_PROGRESS_RENDERED_LINES = 0
        return
    lines = line.splitlines() or [line]
    total_lines = max(_LAST_PROGRESS_RENDERED_LINES, len(lines))
    if _LAST_PROGRESS_RENDERED_LINES > 0:
        sys.stdout.write(f"\x1b[{_LAST_PROGRESS_RENDERED_LINES}F")
    for index in range(total_lines):
        sys.stdout.write("\x1b[2K")
        if index < len(lines):
            sys.stdout.write(lines[index])
        if index < total_lines - 1:
            sys.stdout.write("\n")
    sys.stdout.flush()
    _LAST_PROGRESS_RENDERED_LINES = len(lines)


def _finish_compact_progress() -> None:
    global _LAST_PROGRESS_RENDERED_LINES
    if _use_inplace_progress_rendering() and _LAST_PROGRESS_RENDERED_LINES > 0:
        print()
        _LAST_PROGRESS_RENDERED_LINES = 0


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def load_experiment_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest to be a JSON object, got {type(manifest).__name__}")
    questions = manifest.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError("Manifest must include a non-empty 'questions' list")
    required_keys = {"original_index", "question_id", "family", "split"}
    for idx, entry in enumerate(questions):
        missing = required_keys - set(entry)
        if missing:
            raise ValueError(f"Manifest question entry {idx} is missing required keys: {sorted(missing)}")
    return manifest


def materialize_ordered_subset(manifest: Dict[str, Any], output_root: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    questions_path = _resolve_repo_path(manifest["questions_path"])
    source_questions = _load_json(questions_path)
    if not isinstance(source_questions, list):
        raise ValueError(f"Expected question dataset to be a JSON list, got {type(source_questions).__name__}")

    ordered_subset: List[Dict[str, Any]] = []
    ordered_manifest_questions: List[Dict[str, Any]] = []
    for order_index, entry in enumerate(manifest["questions"]):
        original_index = int(entry["original_index"])
        if original_index < 0 or original_index >= len(source_questions):
            raise IndexError(f"Question index {original_index} is out of range for {questions_path}")
        question = copy.deepcopy(source_questions[original_index])
        question_id = question.get("_id", question.get("id", ""))
        if question_id != entry["question_id"]:
            raise ValueError(
                f"Manifest question_id mismatch at original_index={original_index}: "
                f"expected {entry['question_id']}, found {question_id or '<missing>'}"
            )
        question["_experiment_original_index"] = original_index
        question["_experiment_family"] = entry["family"]
        question["_experiment_split"] = entry["split"]
        ordered_subset.append(question)
        ordered_manifest_questions.append(
            {
                "order_index": order_index,
                "original_index": original_index,
                "question_id": question_id,
                "question": question.get("question", ""),
                "family": entry["family"],
                "split": entry["split"],
                "expected_pattern_hint": entry.get("expected_pattern_hint"),
            }
        )

    subset_path = output_root / "selected_questions.json"
    _write_json(subset_path, ordered_subset)
    return subset_path, ordered_manifest_questions


def build_runner_command(
    *,
    subset_questions_path: Path,
    manifest: Dict[str, Any],
    output_dir: Path,
    args,
    disable_guidance: bool,
    decay_rate: float = 0.0,
) -> List[str]:
    curriculum_batch_size = int(
        args.curriculum_batch_size
        if getattr(args, "curriculum_batch_size", None) is not None
        else manifest.get("curriculum_batch_size", 1)
    )
    curriculum_seed_batch_size = int(
        args.curriculum_seed_batch_size
        if getattr(args, "curriculum_seed_batch_size", None) is not None
        else manifest.get("curriculum_seed_batch_size", 1)
    )
    command = [
        args.python_executable,
        str(RUNNER_PATH),
        "--questions",
        str(subset_questions_path),
        "--corpus_path",
        str(_resolve_repo_path(manifest["corpus_path"])),
        "--gsw_path",
        args.gsw_path,
        "--start",
        "0",
        "--end",
        str(len(manifest["questions"])),
        "--orchestration_mode",
        "curriculum",
        "--curriculum_batch_size",
        str(curriculum_batch_size),
        "--curriculum_seed_batch_size",
        str(curriculum_seed_batch_size),
        "--bridge_query_top_k",
        str(args.bridge_query_top_k),
        "--bridge_prompt_exemplar_top_k",
        str(args.bridge_prompt_exemplar_top_k),
        "--pipeline_mode",
        args.pipeline_mode,
        "--hybrid_scope",
        args.hybrid_scope,
        "--model",
        args.model,
        "--max_tokens",
        str(args.max_tokens),
        "--reasoning_effort",
        args.reasoning_effort,
        "--edge_max_depth",
        str(args.edge_max_depth),
        "--edge_max_calls",
        str(args.edge_max_calls),
        "--edge_max_tokens",
        str(args.edge_max_tokens),
        "--max_optional_docs_per_edge",
        str(args.max_optional_docs_per_edge),
        "--edge_parallel_workers",
        str(args.edge_parallel_workers),
        "--parallel_progress_heartbeat_seconds",
        str(args.parallel_progress_heartbeat_seconds),
        "--parallel_stuck_warning_seconds",
        str(args.parallel_stuck_warning_seconds),
        "--verifier_parallel_workers",
        str(args.verifier_parallel_workers),
        "--edge_max_accepted_bridges",
        str(args.edge_max_accepted_bridges),
        "--embedding_gpu_memory_utilization",
        str(args.embedding_gpu_memory_utilization),
        "--output_dir",
        str(output_dir),
    ]

    if args.base_url:
        command.extend(["--base_url", args.base_url])
    if args.root_model:
        command.extend(["--root_model", args.root_model])
    if args.worker_model:
        command.extend(["--worker_model", args.worker_model])
    if args.edge_parallel_enabled:
        command.append("--edge_parallel_enabled")
    if args.verbose:
        command.append("--verbose")
    if args.show_thinking:
        command.append("--show-thinking")
    if disable_guidance:
        command.append("--curriculum_disable_guidance")
    if decay_rate > 0.0:
        command.extend(["--guidance_decay_rate", str(decay_rate)])
    if getattr(args, "probe_eval", False):
        command.append("--probe_eval")
    if args.curriculum_generation_parallel_enabled:
        command.append("--curriculum_generation_parallel_enabled")
        command.extend(
            [
                "--curriculum_generation_parallel_workers",
                str(args.curriculum_generation_parallel_workers),
            ]
        )
    return command


def run_experiment_arm(
    *,
    arm_name: str,
    subset_questions_path: Path,
    manifest: Dict[str, Any],
    output_root: Path,
    args,
) -> Path:
    arm_output_dir = output_root / arm_name
    arm_output_dir.mkdir(parents=True, exist_ok=True)
    log_paths = _runner_log_paths(arm_output_dir)
    arm_config = ARM_CONFIGS.get(arm_name, {"disable_guidance": False, "decay_rate": 0.0})
    command = build_runner_command(
        subset_questions_path=subset_questions_path,
        manifest=manifest,
        output_dir=arm_output_dir,
        args=args,
        disable_guidance=arm_config["disable_guidance"],
        decay_rate=arm_config.get("decay_rate", 0.0),
    )
    print(f"[{arm_name}] output: {arm_output_dir}", flush=True)
    print(f"[{arm_name}] stdout log: {log_paths['stdout']}", flush=True)
    print(f"[{arm_name}] stderr log: {log_paths['stderr']}", flush=True)
    process: Optional[subprocess.Popen] = None
    last_progress_line: Optional[str] = None
    returncode: Optional[int] = None
    previous_signal_handlers: Dict[int, Any] = {}

    def _raise_for_signal(signum, _frame):
        raise _ExperimentTerminationRequested(int(signum))

    try:
        previous_signal_handlers = _install_signal_handlers(_raise_for_signal)
        with open(log_paths["stdout"], "w") as stdout_file, open(log_paths["stderr"], "w") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                text=True,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                preexec_fn=_child_process_preexec if sys.platform.startswith("linux") else None,
            )
            while True:
                returncode = process.poll()
                current_progress_line = _format_progress_status(
                    arm_name,
                    _latest_batch_progress_snapshot(arm_output_dir),
                    arm_output_dir=arm_output_dir,
                )
                if current_progress_line != last_progress_line:
                    _print_compact_progress(current_progress_line)
                    last_progress_line = current_progress_line
                if returncode is not None:
                    break
                time.sleep(PROGRESS_POLL_INTERVAL_SECONDS)
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        _restore_signal_handlers(previous_signal_handlers)
        if last_progress_line is not None:
            _finish_compact_progress()

    # Print final cost summary for this arm
    tokens = _compute_arm_tokens(arm_output_dir)
    if tokens["total"] > 0:
        cost_str = _format_cost(tokens["input"], tokens["output"])
        print(
            f"[{arm_name}] tokens: {_format_tokens_short(tokens['input'])} in + "
            f"{_format_tokens_short(tokens['output'])} out = "
            f"{_format_tokens_short(tokens['total'])} ({cost_str})",
            flush=True,
        )

    if returncode != 0:
        _terminate_process_group(process)
        stderr_tail = _tail_lines(log_paths["stderr"])
        raise RuntimeError(
            f"Experiment arm '{arm_name}' failed with exit code {returncode}.\n"
            f"stdout log: {log_paths['stdout']}\n"
            f"stderr log: {log_paths['stderr']}\n"
            f"stderr tail:\n{stderr_tail}"
        )
    print(f"[{arm_name}] completed", flush=True)
    return arm_output_dir


def _load_batch_query_results(arm_output_dir: Path) -> Dict[str, Dict[str, Any]]:
    query_results_by_id: Dict[str, Dict[str, Any]] = {}
    for batch_dir in sorted(arm_output_dir.glob("batch_*"), key=lambda path: int(path.name.split("_")[1])):
        artifact_path = batch_dir / "query_answer_results.json"
        if not artifact_path.exists():
            continue
        payload = _load_json(artifact_path)
        batch_index = payload.get("batch_index")
        for row in payload.get("query_results", []):
            question_id = str(row.get("question_id", ""))
            if not question_id:
                continue
            query_results_by_id[question_id] = {
                "batch_index": batch_index,
                **row,
            }
    return query_results_by_id


def _load_batch_bridge_counts(arm_output_dir: Path) -> List[Dict[str, Any]]:
    batch_counts: List[Dict[str, Any]] = []
    for batch_dir in sorted(arm_output_dir.glob("batch_*"), key=lambda path: int(path.name.split("_")[1])):
        snapshot_path = batch_dir / "bridge_registry_snapshot.json"
        if not snapshot_path.exists():
            continue
        payload = _load_json(snapshot_path)
        batch_counts.append(
            {
                "batch_index": payload.get("batch_index"),
                "added_bridge_count": payload.get("added_bridge_count", 0),
                "registry_bridge_count": payload.get("registry_bridge_count", 0),
            }
        )
    return batch_counts


def _top_evidence_summary(rows: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for row in list(rows or [])[:top_k]:
        summary.append(
            {
                "source_type": row.get("source_type"),
                "bridge_id": row.get("bridge_id"),
                "doc_id": row.get("doc_id"),
                "question": row.get("question"),
                "final_score": row.get("final_score"),
            }
        )
    return summary


def _family_question_ids(manifest_questions: List[Dict[str, Any]], family: str) -> List[str]:
    return [entry["question_id"] for entry in manifest_questions if entry["family"] == family]


def _build_retrieved_family_bridge_exports(
    *,
    arm_name: str,
    arm_output_dir: Path,
    manifest_questions: List[Dict[str, Any]],
    query_results_by_id: Dict[str, Dict[str, Any]],
    registry_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    export_dir = arm_output_dir / "retrieved_family_bridges"
    export_dir.mkdir(parents=True, exist_ok=True)

    exports: Dict[str, Dict[str, Any]] = {}
    families = sorted({entry["family"] for entry in manifest_questions})
    manifest_by_qid = {entry["question_id"]: entry for entry in manifest_questions}

    for family in families:
        family_question_ids = _family_question_ids(manifest_questions, family)
        bridge_usage: Dict[str, Dict[str, Any]] = {}
        missing_bridge_ids: List[str] = []

        for question_id in family_question_ids:
            query_result = query_results_by_id.get(question_id, {})
            trace = query_result.get("retrieval_trace", {}) or {}
            retrieved_ids = [
                str(row.get("bridge_id"))
                for row in trace.get("raw_bridge_hits", [])
                if row.get("bridge_id")
            ]
            kept_ids = {
                str(row.get("bridge_id"))
                for row in trace.get("kept_candidates", [])
                if row.get("source_type") == "bridge" and row.get("bridge_id")
            }
            exact_match = float(query_result.get("exact_match") or 0.0)
            bridge_evidence_used = bool(query_result.get("bridge_evidence_used", False))
            manifest_entry = manifest_by_qid[question_id]
            question_ref = {
                "question_id": question_id,
                "question": query_result.get("question", manifest_entry.get("question", "")),
                "original_index": manifest_entry["original_index"],
                "split": manifest_entry["split"],
                "batch_index": query_result.get("batch_index"),
            }

            for bridge_id in retrieved_ids:
                usage_entry = bridge_usage.setdefault(
                    bridge_id,
                    {
                        "retrieved_for_questions": [],
                        "kept_for_question_ids": [],
                        "used_in_correct_answer_question_ids": [],
                    },
                )
                usage_entry["retrieved_for_questions"].append(question_ref)
                if bridge_id in kept_ids:
                    usage_entry["kept_for_question_ids"].append(question_id)
                if bridge_evidence_used and exact_match >= 1.0 and bridge_id in kept_ids:
                    usage_entry["used_in_correct_answer_question_ids"].append(question_id)

        bridges_payload: List[Dict[str, Any]] = []
        for bridge_id, usage_entry in sorted(bridge_usage.items()):
            bridge_record = registry_by_id.get(bridge_id)
            if bridge_record is None:
                missing_bridge_ids.append(bridge_id)
                bridge_record = {"bridge_id": bridge_id, "missing_from_registry": True}
            bridges_payload.append(
                {
                    **bridge_record,
                    "retrieved_for_questions": usage_entry["retrieved_for_questions"],
                    "kept_for_question_ids": usage_entry["kept_for_question_ids"],
                    "used_in_correct_answer_question_ids": usage_entry["used_in_correct_answer_question_ids"],
                    "kept_count": len(usage_entry["kept_for_question_ids"]),
                    "used_in_correct_answer_count": len(usage_entry["used_in_correct_answer_question_ids"]),
                }
            )

        export_payload = {
            "arm": arm_name,
            "family": family,
            "question_ids": family_question_ids,
            "bridge_count": len(bridges_payload),
            "missing_bridge_ids": missing_bridge_ids,
            "bridges": bridges_payload,
        }
        export_path = export_dir / f"{family}.json"
        _write_json(export_path, export_payload)
        exports[family] = {"path": str(export_path), **export_payload}

    return exports


def collect_arm_report(
    *,
    arm_name: str,
    arm_output_dir: Path,
    manifest_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    final_results_path = arm_output_dir / "bridge_test_results.json"
    if not final_results_path.exists():
        raise FileNotFoundError(f"Missing bridge test output for {arm_name}: {final_results_path}")
    final_payload = _load_json(final_results_path)
    metadata = final_payload.get("metadata", {})
    summary = final_payload.get("summary", {})
    registry_records = list(summary.get("curriculum_bridge_registry", []) or [])
    registry_by_id = {record.get("bridge_id"): record for record in registry_records if record.get("bridge_id")}

    query_results_by_id = _load_batch_query_results(arm_output_dir)
    batch_bridge_counts = _load_batch_bridge_counts(arm_output_dir)
    family_exports = _build_retrieved_family_bridge_exports(
        arm_name=arm_name,
        arm_output_dir=arm_output_dir,
        manifest_questions=manifest_questions,
        query_results_by_id=query_results_by_id,
        registry_by_id=registry_by_id,
    )

    manifest_by_qid = {entry["question_id"]: entry for entry in manifest_questions}
    test_entries = [entry for entry in manifest_questions if entry["split"] == "test"]
    per_test_question: List[Dict[str, Any]] = []
    for entry in test_entries:
        query_result = query_results_by_id.get(entry["question_id"], {})
        per_test_question.append(
            {
                "question_id": entry["question_id"],
                "question": query_result.get("question", entry["question"]),
                "original_index": entry["original_index"],
                "family": entry["family"],
                "split": entry["split"],
                "batch_index": query_result.get("batch_index"),
                "predicted_answer": query_result.get("predicted_answer"),
                "exact_match": query_result.get("exact_match"),
                "f1": query_result.get("f1"),
                "bridge_evidence_used": query_result.get("bridge_evidence_used", False),
                "retrieved_bridge_hit_count": query_result.get("retrieved_bridge_hit_count", 0),
                "kept_bridge_hit_count": query_result.get("kept_bridge_hit_count", 0),
                "top_evidence": _top_evidence_summary(query_result.get("top_evidence", [])),
            }
        )

    test_exact = [float(item.get("exact_match") or 0.0) for item in per_test_question]
    test_f1 = [float(item.get("f1") or 0.0) for item in per_test_question]
    test_bridge_usage = [1.0 if item.get("bridge_evidence_used") else 0.0 for item in per_test_question]
    test_retrieved = [float(item.get("retrieved_bridge_hit_count") or 0.0) for item in per_test_question]
    test_kept = [float(item.get("kept_bridge_hit_count") or 0.0) for item in per_test_question]

    per_family_test: Dict[str, Dict[str, Any]] = {}
    for family in sorted({entry["family"] for entry in test_entries}):
        family_rows = [item for item in per_test_question if item["family"] == family]
        per_family_test[family] = {
            "num_questions": len(family_rows),
            "exact_match": _safe_mean([float(item.get("exact_match") or 0.0) for item in family_rows]),
            "f1": _safe_mean([float(item.get("f1") or 0.0) for item in family_rows]),
            "bridge_usage_rate": _safe_mean([1.0 if item.get("bridge_evidence_used") else 0.0 for item in family_rows]),
            "avg_retrieved_bridge_hit_count": _safe_mean(
                [float(item.get("retrieved_bridge_hit_count") or 0.0) for item in family_rows]
            ),
            "avg_kept_bridge_hit_count": _safe_mean(
                [float(item.get("kept_bridge_hit_count") or 0.0) for item in family_rows]
            ),
        }

    batch_bridge_counts_with_questions = []
    for row in batch_bridge_counts:
        batch_index = row.get("batch_index")
        manifest_entry = manifest_questions[batch_index] if isinstance(batch_index, int) and batch_index < len(manifest_questions) else None
        batch_bridge_counts_with_questions.append(
            {
                **row,
                "question_id": manifest_entry["question_id"] if manifest_entry else None,
                "family": manifest_entry["family"] if manifest_entry else None,
                "split": manifest_entry["split"] if manifest_entry else None,
                "original_index": manifest_entry["original_index"] if manifest_entry else None,
            }
        )

    return {
        "arm": arm_name,
        "output_dir": str(arm_output_dir),
        "metadata": metadata,
        "overall_test_metrics": {
            "num_questions": len(per_test_question),
            "exact_match": _safe_mean(test_exact),
            "f1": _safe_mean(test_f1),
            "bridge_usage_rate": _safe_mean(test_bridge_usage),
            "avg_retrieved_bridge_hit_count": _safe_mean(test_retrieved),
            "avg_kept_bridge_hit_count": _safe_mean(test_kept),
        },
        "per_family_test_metrics": per_family_test,
        "per_test_question": per_test_question,
        "bridge_inventory": {
            "per_batch_added_bridge_counts": batch_bridge_counts_with_questions,
            "final_registry_size": int(summary.get("total_accepted_bridges", len(registry_records)) or 0),
            "retrieved_family_bridge_counts": {
                family: export["bridge_count"] for family, export in family_exports.items()
            },
            "retrieved_family_bridge_files": {
                family: export["path"] for family, export in family_exports.items()
            },
        },
        "retrieved_family_bridges": family_exports,
        "query_results_by_id": query_results_by_id,
    }


def build_comparison_report(
    *,
    manifest: Dict[str, Any],
    ordered_manifest_questions: List[Dict[str, Any]],
    arm_reports: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    on_report = arm_reports["guidance_on"]
    off_report = arm_reports["guidance_off"]

    delta = {
        "test_exact_match": round(
            float(on_report["overall_test_metrics"]["exact_match"]) - float(off_report["overall_test_metrics"]["exact_match"]),
            4,
        ),
        "test_f1": round(
            float(on_report["overall_test_metrics"]["f1"]) - float(off_report["overall_test_metrics"]["f1"]),
            4,
        ),
        "test_bridge_usage_rate": round(
            float(on_report["overall_test_metrics"]["bridge_usage_rate"])
            - float(off_report["overall_test_metrics"]["bridge_usage_rate"]),
            4,
        ),
        "test_avg_retrieved_bridge_hit_count": round(
            float(on_report["overall_test_metrics"]["avg_retrieved_bridge_hit_count"])
            - float(off_report["overall_test_metrics"]["avg_retrieved_bridge_hit_count"]),
            4,
        ),
        "test_avg_kept_bridge_hit_count": round(
            float(on_report["overall_test_metrics"]["avg_kept_bridge_hit_count"])
            - float(off_report["overall_test_metrics"]["avg_kept_bridge_hit_count"]),
            4,
        ),
        "per_family": {},
    }

    all_families = sorted(
        set(on_report["per_family_test_metrics"].keys()) | set(off_report["per_family_test_metrics"].keys())
    )
    for family in all_families:
        on_family = on_report["per_family_test_metrics"].get(family, {})
        off_family = off_report["per_family_test_metrics"].get(family, {})
        delta["per_family"][family] = {
            "exact_match": round(float(on_family.get("exact_match", 0.0)) - float(off_family.get("exact_match", 0.0)), 4),
            "f1": round(float(on_family.get("f1", 0.0)) - float(off_family.get("f1", 0.0)), 4),
            "bridge_usage_rate": round(
                float(on_family.get("bridge_usage_rate", 0.0)) - float(off_family.get("bridge_usage_rate", 0.0)),
                4,
            ),
            "avg_retrieved_bridge_hit_count": round(
                float(on_family.get("avg_retrieved_bridge_hit_count", 0.0))
                - float(off_family.get("avg_retrieved_bridge_hit_count", 0.0)),
                4,
            ),
            "avg_kept_bridge_hit_count": round(
                float(on_family.get("avg_kept_bridge_hit_count", 0.0))
                - float(off_family.get("avg_kept_bridge_hit_count", 0.0)),
                4,
            ),
        }

    return {
        "manifest": {
            "dataset": manifest["dataset"],
            "family_label": manifest.get("family_label"),
            "questions_path": manifest["questions_path"],
            "corpus_path": manifest["corpus_path"],
            "questions": ordered_manifest_questions,
        },
        "arms": {
            arm_name: {
                key: value
                for key, value in report.items()
                if key != "query_results_by_id"
            }
            for arm_name, report in arm_reports.items()
        },
        "delta": delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-arm curriculum-guidance ablation on a tracked 2Wiki lookup subset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gsw_path", type=str, required=True)
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--model", type=str, default="Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--root_model", type=str, default=None)
    parser.add_argument("--worker_model", type=str, default=None)
    parser.add_argument("--reasoning_effort", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--pipeline_mode", type=str, default="hybrid", choices=["legacy", "rlm", "hybrid"])
    parser.add_argument("--hybrid_scope", type=str, default="doc_edge", choices=["edge", "doc_edge", "corpus_doc_edge"])
    parser.add_argument("--max_tokens", type=int, default=500_000)
    parser.add_argument("--edge_max_depth", type=int, default=1)
    parser.add_argument("--edge_max_calls", type=int, default=2)
    parser.add_argument("--edge_max_tokens", type=int, default=3000)
    parser.add_argument("--max_optional_docs_per_edge", type=int, default=2)
    parser.add_argument("--edge_parallel_enabled", action="store_true")
    parser.add_argument("--edge_parallel_workers", type=int, default=1)
    parser.add_argument("--parallel_progress_heartbeat_seconds", type=float, default=10.0)
    parser.add_argument("--parallel_stuck_warning_seconds", type=float, default=60.0)
    parser.add_argument("--verifier_parallel_workers", type=int, default=4)
    parser.add_argument("--edge_max_accepted_bridges", type=int, default=8)
    parser.add_argument("--bridge_query_top_k", type=int, default=12)
    parser.add_argument("--bridge_prompt_exemplar_top_k", type=int, default=3)
    parser.add_argument("--curriculum_batch_size", type=int, default=None)
    parser.add_argument("--curriculum_seed_batch_size", type=int, default=None)
    parser.add_argument("--curriculum_generation_parallel_enabled", action="store_true")
    parser.add_argument("--curriculum_generation_parallel_workers", type=int, default=1)
    parser.add_argument("--embedding_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--probe_eval", action="store_true",
                        help="Enable probe evaluation after each batch for learning curve analysis")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Which arms to run (default: all). Options: guidance_on, guidance_off, guidance_decay")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = _resolve_repo_path(args.manifest)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_experiment_manifest(manifest_path)
    subset_questions_path, ordered_manifest_questions = materialize_ordered_subset(manifest, output_root)
    _write_json(output_root / "resolved_manifest.json", {"manifest": manifest, "questions": ordered_manifest_questions})

    arm_reports: Dict[str, Dict[str, Any]] = {}
    default_arms = ["guidance_on", "guidance_off"]
    arms_to_run = args.arms if args.arms else default_arms
    for arm_name in arms_to_run:
        arm_output_dir = run_experiment_arm(
            arm_name=arm_name,
            subset_questions_path=subset_questions_path,
            manifest=manifest,
            output_root=output_root,
            args=args,
        )
        arm_reports[arm_name] = collect_arm_report(
            arm_name=arm_name,
            arm_output_dir=arm_output_dir,
            manifest_questions=ordered_manifest_questions,
        )

    comparison = build_comparison_report(
        manifest=manifest,
        ordered_manifest_questions=ordered_manifest_questions,
        arm_reports=arm_reports,
    )
    comparison_path = output_root / "comparison.json"
    _write_json(comparison_path, comparison)
    print(f"Comparison written to {comparison_path}")


if __name__ == "__main__":
    main()
