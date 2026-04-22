#!/usr/bin/env python3
"""
Run a low-load worker-only diagnosis over one curriculum batch.

This script mirrors the curriculum batch setup closely enough to reproduce worker
packet construction, while avoiding the full bridge-generation orchestration.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


from playground.sleep_time import run_bridge_test
from playground.sleep_time import run_curriculum_guidance_experiment as guidance_experiment
from playground.sleep_time.run_sleep_time import SleepTimeRunner
from src.gsw_memory.sleep_time.curriculum import split_curriculum_batches


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _resolve_include_optional(mode: str, prepared: Any) -> bool:
    if mode == "true":
        return True
    if mode == "false":
        return False
    if not getattr(prepared.packet, "optional_contexts", None):
        return False
    signal_tier = str((prepared.signal or {}).get("tier", "") or "")
    path_proof_count = int((prepared.viability or {}).get("path_proof_count", 0) or 0)
    return signal_tier == "low" or path_proof_count == 0


def _matches_edge_filters(edge: Any, args: argparse.Namespace) -> bool:
    if args.edge_filter_doc_id and edge.doc_id != args.edge_filter_doc_id:
        return False
    if args.edge_filter_entity and edge.entity_name != args.edge_filter_entity:
        return False
    if args.edge_filter_neighbor and edge.neighbor_name != args.edge_filter_neighbor:
        return False
    if args.edge_filter_relationship and edge.relationship != args.edge_filter_relationship:
        return False
    return True


def _materialize_batch_items(manifest: Dict[str, Any], output_dir: Path) -> List[Any]:
    selected_questions_path, _manifest_questions = guidance_experiment.materialize_ordered_subset(manifest, output_dir)
    questions, schema = run_bridge_test.load_questions(str(selected_questions_path))
    if schema == "unknown":
        raise ValueError("Unable to infer question schema from selected questions.")

    title_to_idx = None
    if schema == "context_titles":
        corpus_path = guidance_experiment._resolve_repo_path(str(manifest["corpus_path"]))
        title_to_idx = run_bridge_test.build_title_to_doc_index(str(corpus_path))

    return run_bridge_test.build_batch_query_items(questions, schema=schema, title_to_idx=title_to_idx)


def _collect_batch_edges(scheduler: Any, batch_doc_ids: Iterable[str], args: argparse.Namespace) -> List[Any]:
    edges: List[Any] = []
    seen = set()
    for doc_id in batch_doc_ids:
        plan = scheduler._call_tool("plan_document_exploration", {"doc_id": doc_id}, doc_id=doc_id)
        if isinstance(plan, dict) and "error" in plan:
            continue
        for edge in scheduler._iter_pending_edges(doc_id):
            edge_key = edge.as_tuple()
            if edge_key in seen:
                continue
            if not _matches_edge_filters(edge, args):
                continue
            seen.add(edge_key)
            edges.append(edge)
    return edges


def _build_probe_args(args: argparse.Namespace, *, gsw_path: str, num_docs: int, cache_dir: str, output_dir: str) -> argparse.Namespace:
    return run_bridge_test._build_sleep_time_args(
        args,
        gsw_path=gsw_path,
        num_docs=num_docs,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


def _serialize_result(
    *,
    edge: Any,
    prepared: Any,
    include_optional: bool,
    elapsed_seconds: float,
    worker_output: Any,
) -> Dict[str, Any]:
    return {
        "packet": {
            "doc_id": edge.doc_id,
            "entity_name": edge.entity_name,
            "neighbor_name": edge.neighbor_name,
            "relationship": edge.relationship,
            "signal_tier": str((prepared.signal or {}).get("tier", "")),
            "signal_score": (prepared.signal or {}).get("score"),
            "mandatory_docs": list(prepared.packet.mandatory_docs),
            "optional_docs": list(prepared.packet.optional_docs),
            "path_proof_count": int((prepared.viability or {}).get("path_proof_count", 0) or 0),
            "include_optional": bool(include_optional),
        },
        "worker_result": {
            "status": str(worker_output.status),
            "parse_stage": str(worker_output.parse_stage),
            "source_field": str(worker_output.source_field),
            "parse_attempts": int(worker_output.parse_attempts),
            "parse_recovery_used": bool(worker_output.parse_recovery_used),
            "candidates_count": int(len(worker_output.candidates)),
            "need_recursion": bool(worker_output.need_recursion),
            "notes": str(worker_output.notes),
            "finish_reason": worker_output.finish_reason,
            "token_usage": int(getattr(worker_output, "token_usage", 0) or 0),
            "budgeted_token_usage": int(getattr(worker_output, "budgeted_token_usage", 0) or 0),
            "message_fields": dict(getattr(worker_output, "message_field_summary", {}) or {}),
            "attempt_diagnostics": list(getattr(worker_output, "attempt_diagnostics", []) or []),
            "elapsed_seconds": round(float(elapsed_seconds), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose batch-0 Qwen worker behavior on real edge packets.")
    parser.add_argument("--manifest", required=True, help="Path to the curriculum experiment manifest.")
    parser.add_argument("--gsw_path", required=True, help="Path to the GSW networks directory.")
    parser.add_argument("--base_url", required=True, help="Base URL for the local OpenAI-compatible model server.")
    parser.add_argument("--model", required=True, help="Base model name for root/worker unless overridden.")
    parser.add_argument("--root_model", default=None, help="Optional root-stage model override.")
    parser.add_argument("--worker_model", default=None, help="Optional worker-stage model override.")
    parser.add_argument("--batch_index", type=int, default=0, help="Curriculum batch index to diagnose.")
    parser.add_argument("--max_edges", type=int, default=10, help="Maximum number of edges to probe.")
    parser.add_argument(
        "--include_optional",
        choices=["auto", "false", "true"],
        default="auto",
        help="How to include optional cross-doc contexts when running the worker.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for diagnosis artifacts.")
    parser.add_argument("--edge_filter_doc_id", default=None, help="Restrict probes to one doc_id.")
    parser.add_argument("--edge_filter_entity", default=None, help="Restrict probes to one source entity name.")
    parser.add_argument("--edge_filter_neighbor", default=None, help="Restrict probes to one neighbor entity name.")
    parser.add_argument("--edge_filter_relationship", default=None, help="Restrict probes to one relationship label.")
    parser.add_argument("--max_response_tokens", type=int, default=1400, help="Worker max_tokens for each probe call.")
    parser.add_argument("--pipeline_mode", default="hybrid", choices=["rlm", "hybrid"], help="Scheduler mode for packet construction.")
    parser.add_argument("--hybrid_scope", default="doc_edge", help="Hybrid scope passed to the runner.")
    parser.add_argument("--reasoning_effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--edge_max_depth", type=int, default=3)
    parser.add_argument("--edge_max_calls", type=int, default=5)
    parser.add_argument("--edge_max_tokens", type=int, default=12000)
    parser.add_argument("--max_optional_docs_per_edge", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=5_000_000)
    parser.add_argument("--verifier_parallel_workers", type=int, default=4)
    parser.add_argument("--edge_max_accepted_bridges", type=int, default=10)
    parser.add_argument("--embedding_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = guidance_experiment.load_experiment_manifest(Path(args.manifest))
    items = _materialize_batch_items(manifest, output_dir)
    train_items = [item for item in items if item.metadata.get("split") != "probe"]
    batch_size = int(manifest.get("curriculum_batch_size", 4) or 4)
    seed_batch_size = manifest.get("curriculum_seed_batch_size")
    batches = split_curriculum_batches(
        train_items,
        batch_size=batch_size,
        seed_batch_size=int(seed_batch_size or batch_size),
    )
    if args.batch_index < 0 or args.batch_index >= len(batches):
        raise IndexError(f"Batch index {args.batch_index} is out of range for {len(batches)} available batch(es).")

    batch_items = batches[args.batch_index]
    batch_doc_ids = sorted({doc_id for item in batch_items for doc_id in item.doc_ids})
    temp_base = tempfile.mkdtemp(prefix=f"qwen_worker_diag_batch{args.batch_index}_")
    diagnosis_results_path = output_dir / "diagnosis_results.jsonl"
    diagnosis_summary_path = output_dir / "diagnosis_summary.json"

    runner = None
    try:
        temp_gsw_dir = run_bridge_test.setup_temp_gsw_dir(args.gsw_path, batch_doc_ids, temp_base, subdir="networks")
        cache_dir = run_bridge_test._curriculum_cache_dir(str(output_dir), "diagnosis_batch_edges", batch_doc_ids)
        probe_args = _build_probe_args(
            args,
            gsw_path=temp_gsw_dir,
            num_docs=len(batch_doc_ids),
            cache_dir=cache_dir,
            output_dir=str(output_dir / "sleep_time_probe"),
        )

        batch_searcher = run_bridge_test.EntitySearcher(
            num_documents=len(batch_doc_ids),
            path_to_gsw_files=temp_gsw_dir,
            cache_dir=cache_dir,
            rebuild_cache=False,
            verbose=False,
            use_bm25=True,
            use_gpu_for_qa_index=False,
            embedding_gpu_memory_utilization=getattr(args, "embedding_gpu_memory_utilization", 0.5),
        )

        runner = SleepTimeRunner(probe_args, entity_searcher=batch_searcher)
        agent = runner.initialize_agent(display=None)
        if args.pipeline_mode == "hybrid":
            scheduler = agent._create_hybrid_scheduler()
        else:
            scheduler = agent._create_rlm_scheduler()

        if not hasattr(scheduler, "_prepare_edge_work_parallel"):
            raise RuntimeError("Selected scheduler does not expose _prepare_edge_work_parallel for real packet diagnosis.")

        candidate_edges = _collect_batch_edges(scheduler, batch_doc_ids, args)
        selected_edges = candidate_edges[: max(0, int(args.max_edges))]
        results: List[Dict[str, Any]] = []

        for edge in selected_edges:
            prepared = scheduler._prepare_edge_work_parallel(edge)
            if prepared is None:
                result = {
                    "packet": {
                        "doc_id": edge.doc_id,
                        "entity_name": edge.entity_name,
                        "neighbor_name": edge.neighbor_name,
                        "relationship": edge.relationship,
                        "prepared": False,
                    },
                    "worker_result": {
                        "status": "prepare_failed",
                        "parse_stage": "prepare",
                        "source_field": "prepare_failed",
                        "parse_attempts": 0,
                        "parse_recovery_used": False,
                        "candidates_count": 0,
                        "need_recursion": False,
                        "notes": "Failed to prepare edge work packet.",
                        "finish_reason": None,
                        "token_usage": 0,
                        "budgeted_token_usage": 0,
                        "message_fields": {},
                        "elapsed_seconds": 0.0,
                    },
                }
                results.append(result)
                _append_jsonl(diagnosis_results_path, result)
                continue

            include_optional = _resolve_include_optional(args.include_optional, prepared)
            started = time.perf_counter()
            worker_output = scheduler.worker.generate(
                packet=prepared.packet,
                depth=0,
                include_optional=include_optional,
                max_response_tokens=int(args.max_response_tokens),
                retry_brief=prepared.retry_brief,
            )
            elapsed_seconds = time.perf_counter() - started
            result = _serialize_result(
                edge=edge,
                prepared=prepared,
                include_optional=include_optional,
                elapsed_seconds=elapsed_seconds,
                worker_output=worker_output,
            )
            results.append(result)
            _append_jsonl(diagnosis_results_path, result)

        status_counts = Counter(result["worker_result"]["status"] for result in results)
        source_field_counts = Counter(result["worker_result"]["source_field"] for result in results)
        finish_reason_counts = Counter(
            result["worker_result"]["finish_reason"] or "none"
            for result in results
        )
        summary = {
            "manifest": str(Path(args.manifest).resolve()),
            "gsw_path": str(Path(args.gsw_path).resolve()),
            "base_url": args.base_url,
            "model": args.model,
            "root_model": args.root_model or args.model,
            "worker_model": args.worker_model or args.model,
            "batch_index": int(args.batch_index),
            "batch_doc_ids": batch_doc_ids,
            "total_candidate_edges": len(candidate_edges),
            "processed_edges": len(results),
            "include_optional_mode": args.include_optional,
            "max_response_tokens": int(args.max_response_tokens),
            "status_counts": dict(status_counts),
            "source_field_counts": dict(source_field_counts),
            "finish_reason_counts": dict(finish_reason_counts),
            "diagnosis_results_path": str(diagnosis_results_path),
        }
        _write_json(diagnosis_summary_path, summary)
        print(json.dumps(summary, indent=2))
    finally:
        if runner is not None:
            runner._cleanup_logging()
        shutil.rmtree(temp_base, ignore_errors=True)


if __name__ == "__main__":
    main()
