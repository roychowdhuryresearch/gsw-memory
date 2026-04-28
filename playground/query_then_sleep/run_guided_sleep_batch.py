#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.simple_entity_search import EntitySearcher
from playground.sleep_time import run_bridge_test
from playground.sleep_time.run_sleep_time import InteractiveDisplay, SleepTimeRunner, console
from gsw_memory.query_then_sleep import InteractionGuidanceSummary, InteractionGuidedReconciler, build_sleep_guidance_payload
from gsw_memory.sleep_time.curriculum import BatchQueryItem, BridgeRecord, BridgeRegistry, BridgeSurface, split_curriculum_batches


class GuidedSleepTimeRunner(SleepTimeRunner):
    def initialize_agent(self, display: Optional[InteractiveDisplay] = None):
        console.print(f"\nInitializing guided sleep agent with model: {self.args.model}")
        output_callback = self._create_callback_handler(display)

        curriculum_guidance = self.injected_curriculum_guidance
        reasoning_effort = getattr(self.args, "reasoning_effort", "medium")
        base_url = getattr(self.args, "base_url", None)
        disable_bridge_verifier = bool(getattr(self.args, "disable_bridge_verifier", False))
        bridge_verifier_threshold = float(getattr(self.args, "bridge_verifier_threshold", 0.7))
        bridge_verifier_fail_open = bool(getattr(self.args, "bridge_verifier_fail_open", False))
        bridge_verifier_model = getattr(self.args, "bridge_verifier_model", None)
        pipeline_mode = getattr(self.args, "pipeline_mode", "hybrid")
        hybrid_scope = getattr(self.args, "hybrid_scope", "doc_edge")
        root_model = getattr(self.args, "root_model", None)
        worker_model = getattr(self.args, "worker_model", None)
        edge_max_depth = int(getattr(self.args, "edge_max_depth", 1))
        edge_max_calls = int(getattr(self.args, "edge_max_calls", 2))
        edge_max_tokens = int(getattr(self.args, "edge_max_tokens", 3000))
        max_optional_docs_per_edge = int(getattr(self.args, "max_optional_docs_per_edge", 2))
        edge_parallel_enabled = bool(getattr(self.args, "edge_parallel_enabled", False))
        edge_parallel_workers = int(getattr(self.args, "edge_parallel_workers", 1))
        verifier_parallel_workers = int(getattr(self.args, "verifier_parallel_workers", 4))
        edge_max_accepted_bridges = int(getattr(self.args, "edge_max_accepted_bridges", 10))
        parallel_progress_heartbeat_seconds = float(getattr(self.args, "parallel_progress_heartbeat_seconds", 10.0))
        parallel_stuck_warning_seconds = float(getattr(self.args, "parallel_stuck_warning_seconds", 60.0))
        num_entities = int(getattr(self.args, "num_entities", 0))
        num_docs = int(getattr(self.args, "num_docs", len(getattr(self.entity_searcher, "gsw_by_doc_id", {}))))
        max_tokens = int(getattr(self.args, "max_tokens", 500_000))

        agent = InteractionGuidedReconciler(
            entity_searcher=self.entity_searcher,
            model_name=self.args.model,
            budget={"max_entities": num_entities, "max_documents": num_docs, "max_tokens": max_tokens},
            verbose=False,
            output_callback=output_callback,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            bridge_verifier_enabled=not disable_bridge_verifier,
            bridge_verifier_threshold=bridge_verifier_threshold,
            bridge_verifier_fail_open=bridge_verifier_fail_open,
            bridge_verifier_model=bridge_verifier_model,
            pipeline_mode=pipeline_mode,
            hybrid_scope=hybrid_scope,
            root_model=root_model,
            worker_model=worker_model,
            edge_max_depth=edge_max_depth,
            edge_max_calls=edge_max_calls,
            edge_max_tokens=edge_max_tokens,
            max_optional_docs_per_edge=max_optional_docs_per_edge,
            edge_parallel_enabled=edge_parallel_enabled,
            edge_parallel_workers=edge_parallel_workers,
            verifier_parallel_workers=verifier_parallel_workers,
            edge_max_accepted_bridges=edge_max_accepted_bridges,
            parallel_progress_heartbeat_seconds=parallel_progress_heartbeat_seconds,
            parallel_stuck_warning_seconds=parallel_stuck_warning_seconds,
            curriculum_guidance=curriculum_guidance,
        )
        console.print("✓ Guided sleep agent initialized")
        return agent


def _write_progress(progress_path: Path, phase: str, status: str, **data: Any) -> None:
    payload = {"phase": phase, "status": status, **data}
    with open(progress_path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _load_bridge_registry_snapshot(path: Optional[str], entity_searcher: Any) -> BridgeRegistry:
    registry = BridgeRegistry()
    if not path:
        return registry
    payload = json.loads(Path(path).read_text())
    for row in payload.get("all_bridge_records", []) or []:
        if not isinstance(row, dict):
            continue
        bridge_id = str(row.get("bridge_id", "") or "").strip()
        question = str(row.get("question", "") or "").strip()
        reverse_question = str(row.get("reverse_question", "") or "").strip()
        if not bridge_id or not question or not reverse_question:
            continue
        record = BridgeRecord(
            bridge_id=bridge_id,
            question=question,
            reverse_question=reverse_question,
            answer_text=str(row.get("answer_text", "") or ""),
            reverse_answer_text=str(row.get("reverse_answer_text", "") or ""),
            source_docs=list(row.get("source_docs", []) or []),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            pattern_tags=list(row.get("pattern_tags", []) or []),
            forward_pattern=dict(row.get("forward_pattern", {}) or {}),
            reverse_pattern=dict(row.get("reverse_pattern", {}) or {}),
            batch_index=int(payload.get("batch_index", 0) or 0),
            source_relationship=str(row.get("source_relationship", "") or ""),
            source_entity=str(row.get("source_entity", "") or ""),
            source_neighbor=str(row.get("source_neighbor", "") or ""),
            retrieved_count=int(row.get("retrieved_count", 0) or 0),
            helpful_count=int(row.get("helpful_count", 0) or 0),
        )
        registry.records[bridge_id] = record
        registry.surfaces.append(
            BridgeSurface(
                bridge_id=bridge_id,
                orientation="forward",
                question_text=question,
                answer_text=record.answer_text,
                pattern_tags=list(record.pattern_tags),
                question_pattern=dict(record.forward_pattern),
                source_docs=list(record.source_docs),
            )
        )
        reverse_answer_type = str(record.reverse_pattern.get("answer_type", "")).strip().lower()
        if reverse_answer_type in {"person", "organization"}:
            registry.surfaces.append(
                BridgeSurface(
                    bridge_id=bridge_id,
                    orientation="reverse",
                    question_text=reverse_question,
                    answer_text=record.reverse_answer_text,
                    pattern_tags=list(record.pattern_tags),
                    question_pattern=dict(record.reverse_pattern),
                    source_docs=list(record.source_docs),
                )
            )
            registry._surface_keys.add((bridge_id, "reverse"))
        registry._surface_keys.add((bridge_id, "forward"))
    if registry.surfaces:
        embed_fn = getattr(entity_searcher, "_embed_query", None)
        if callable(embed_fn):
            registry._rebuild_indexes(lambda texts: entity_searcher._embed_query(list(texts)))
        else:
            registry._rebuild_indexes(None)
    return registry


def _plan_generation_shards(batch_items: Sequence[BatchQueryItem]) -> List[Dict[str, Any]]:
    shards: List[Dict[str, Any]] = []
    for shard_index, item in enumerate(batch_items):
        shards.append(
            {
                "shard_id": shard_index,
                "question_ids": [str(item.question_id or "")],
                "questions": [str(item.question or "")],
                "doc_ids": list(item.doc_ids),
            }
        )
    return shards


def _write_generation_shard_metadata(shard_output_dir: Path, batch_index: int, shard: Dict[str, Any]) -> None:
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_index": int(batch_index),
        "shard_id": int(shard["shard_id"]),
        "question_ids": list(shard.get("question_ids", [])),
        "questions": list(shard.get("questions", [])),
        "doc_ids": list(shard.get("doc_ids", [])),
    }
    _write_json(shard_output_dir / "shard_metadata.json", payload)


def _merge_bridge_dicts(bridges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for bridge in bridges:
        bridge_id = str(bridge.get("bridge_id", "")).strip()
        if not bridge_id:
            continue
        candidate = dict(bridge)
        candidate_docs = {str(doc_id) for doc_id in candidate.get("source_docs", []) or [] if str(doc_id).strip()}
        if bridge_id not in merged_by_id:
            candidate["source_docs"] = sorted(candidate_docs)
            merged_by_id[bridge_id] = candidate
            continue
        existing = merged_by_id[bridge_id]
        existing_docs = {str(doc_id) for doc_id in existing.get("source_docs", []) or [] if str(doc_id).strip()}
        merged_docs = sorted(existing_docs | candidate_docs)
        existing_confidence = float(existing.get("confidence", 0.0) or 0.0)
        candidate_confidence = float(candidate.get("confidence", 0.0) or 0.0)
        if candidate_confidence > existing_confidence:
            candidate["source_docs"] = merged_docs
            merged_by_id[bridge_id] = candidate
        else:
            existing["source_docs"] = merged_docs
    return [merged_by_id[key] for key in sorted(merged_by_id.keys())]


def _build_sleep_stage_summary(
    *,
    report: Dict[str, Any],
    bridge_count: int,
    generation_mode: str,
    generation_shards: Sequence[Dict[str, Any]],
    run_output_dir: str,
    failure_reason: str = "",
) -> Dict[str, Any]:
    summary = dict(report.get("summary", {}) or {})
    rlm_metrics = dict(summary.get("rlm_metrics", {}) or {})
    errors = list(report.get("errors", []) or [])

    candidate_count = int(rlm_metrics.get("edges_explored", 0) or 0)
    worker_proposals = int(rlm_metrics.get("worker_calls", 0) or 0)
    accepted_bridge_count = int(bridge_count)
    verifier_rejections = int(rlm_metrics.get("verifier_rejected", 0) or 0)
    if verifier_rejections <= 0 and accepted_bridge_count == 0 and worker_proposals > 0 and not errors:
        verifier_rejections = worker_proposals

    failure_text = " ".join(
        str(item.get("error", item)).strip()
        for item in errors
        if str(item).strip()
    )
    if failure_reason:
        failure_text = f"{failure_text} {failure_reason}".strip()
    failure_lower = failure_text.lower()

    likely_zero_bridge_reasons: List[str] = []
    if bridge_count == 0:
        if "empty worker response" in failure_lower or "content_empty" in failure_lower:
            likely_zero_bridge_reasons.append("empty_worker_outputs")
        if any(token in failure_lower for token in ("bedrock", "provider", "invalid model", "api", "timeout")):
            likely_zero_bridge_reasons.append("provider_model_failure")
        if int(rlm_metrics.get("edges_skipped_no_path", 0) or 0) > 0 and worker_proposals == 0:
            likely_zero_bridge_reasons.append("no_path_proof")
        if candidate_count == 0 and worker_proposals == 0:
            likely_zero_bridge_reasons.append("no_viable_edges")
        if verifier_rejections > 0 and "empty_worker_outputs" not in likely_zero_bridge_reasons:
            likely_zero_bridge_reasons.append("verifier_rejection")
        if not likely_zero_bridge_reasons:
            likely_zero_bridge_reasons.append("no_accepted_bridges")

    return {
        "generation_mode": generation_mode,
        "generation_shards": list(generation_shards),
        "candidate_count": candidate_count,
        "worker_proposals_generated": worker_proposals,
        "verifier_rejections": verifier_rejections,
        "accepted_bridge_count": accepted_bridge_count,
        "edges_skipped_no_path": int(rlm_metrics.get("edges_skipped_no_path", 0) or 0),
        "parallel_batches": int(rlm_metrics.get("parallel_batches", 0) or 0),
        "errors_count": len(errors),
        "likely_zero_bridge_reasons": likely_zero_bridge_reasons,
        "failure_reason": str(failure_reason or ""),
        "run_output_dir": str(run_output_dir),
        "rlm_metrics": rlm_metrics,
    }


def _run_guided_sleep_doc_batch(
    *,
    batch_index: int,
    doc_ids: Sequence[str],
    args: argparse.Namespace,
    output_dir: Path,
    guidance_payload: Dict[str, Any],
) -> Dict[str, Any]:
    temp_base = tempfile.mkdtemp(prefix=f"query_then_sleep_sleep_{batch_index}_")
    try:
        temp_gsw_dir = run_bridge_test.setup_temp_gsw_dir(args.gsw_path, list(doc_ids), temp_base, subdir="networks")
        cache_dir = run_bridge_test._curriculum_cache_dir(str(output_dir), "sleep_docs", list(doc_ids))
        entity_searcher = EntitySearcher(
            num_documents=len(doc_ids),
            path_to_gsw_files=temp_gsw_dir,
            cache_dir=cache_dir,
            rebuild_cache=False,
            verbose=False,
            use_bm25=True,
            use_gpu_for_qa_index=False,
            embedding_gpu_memory_utilization=args.embedding_gpu_memory_utilization,
        )
        batch_args = run_bridge_test._build_sleep_time_args(
            args,
            gsw_path=temp_gsw_dir,
            num_docs=len(doc_ids),
            output_dir=str(output_dir),
            cache_dir=cache_dir,
        )
        runner = GuidedSleepTimeRunner(batch_args, entity_searcher=entity_searcher, curriculum_guidance=guidance_payload)
        report = runner.run() or {}
        bridges = list(report.get("bridges", []) or [])
        stage_summary = _build_sleep_stage_summary(
            report=report,
            bridge_count=len(bridges),
            generation_mode="single_batch",
            generation_shards=[],
            run_output_dir=str(output_dir),
        )
        _write_json(output_dir / "sleep_stage_summary.json", stage_summary)
        return {
            "batch_index": int(batch_index),
            "doc_ids": list(doc_ids),
            "bridges": bridges,
            "report": report,
            "run_results": dict(report.get("summary", {}) or {}),
            "stage_summary": stage_summary,
            "output_dir": str(output_dir),
        }
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)


def _run_guided_sleep_shard_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.perf_counter()
    result = _run_guided_sleep_doc_batch(
        batch_index=int(task["batch_index"]),
        doc_ids=list(task["doc_ids"]),
        args=task["args"],
        output_dir=Path(task["output_dir"]),
        guidance_payload=dict(task["guidance_payload"]),
    )
    runtime_seconds = round(time.perf_counter() - started_at, 3)
    return {
        **result,
        "shard_id": int(task["shard_id"]),
        "question_ids": list(task.get("question_ids", [])),
        "questions": list(task.get("questions", [])),
        "runtime_seconds": runtime_seconds,
    }


def _execute_guided_sleep_shard_tasks(
    *,
    tasks: Sequence[Dict[str, Any]],
    max_workers: int,
    progress_path: Path,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    total_tasks = len(tasks)
    _write_progress(progress_path, "bridge_generation_shards", "started", completed=0, total=total_tasks, unit="shards")

    if total_tasks == 1 or max_workers <= 1:
        results: List[Dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            result = _run_guided_sleep_shard_worker(dict(task))
            results.append(result)
            _write_progress(
                progress_path,
                "bridge_generation_shards",
                "progress",
                completed=index,
                total=total_tasks,
                unit="shards",
                shard_id=result["shard_id"],
                question_ids=list(result.get("question_ids", [])),
                accepted_bridges=len(result.get("bridges", [])),
                runtime_seconds=result.get("runtime_seconds"),
            )
        _write_progress(progress_path, "bridge_generation_shards", "completed", completed=total_tasks, total=total_tasks, unit="shards")
        return results

    results_by_shard: Dict[int, Dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
        future_to_task = {
            executor.submit(_run_guided_sleep_shard_worker, dict(task)): dict(task)
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results_by_shard[int(task["shard_id"])] = result
            completed += 1
            _write_progress(
                progress_path,
                "bridge_generation_shards",
                "progress",
                completed=completed,
                total=total_tasks,
                unit="shards",
                shard_id=result["shard_id"],
                question_ids=list(result.get("question_ids", [])),
                accepted_bridges=len(result.get("bridges", [])),
                runtime_seconds=result.get("runtime_seconds"),
            )
    _write_progress(progress_path, "bridge_generation_shards", "completed", completed=total_tasks, total=total_tasks, unit="shards")
    return [results_by_shard[int(task["shard_id"])] for task in tasks]


def _run_guided_sleep_sharded(
    *,
    batch_index: int,
    batch_items: Sequence[BatchQueryItem],
    args: argparse.Namespace,
    output_dir: Path,
    guidance_payload: Dict[str, Any],
    progress_path: Path,
) -> Dict[str, Any]:
    shard_dir_root = output_dir / "generation_shards"
    shard_dir_root.mkdir(parents=True, exist_ok=True)
    shards = _plan_generation_shards(batch_items)
    worker_count = max(1, int(getattr(args, "curriculum_generation_parallel_workers", 1) or 1))

    for shard in shards:
        _write_generation_shard_metadata(shard_dir_root / f"shard_{shard['shard_id']}", batch_index, shard)

    tasks = [
        {
            "batch_index": int(batch_index),
            "shard_id": int(shard["shard_id"]),
            "question_ids": list(shard.get("question_ids", [])),
            "questions": list(shard.get("questions", [])),
            "doc_ids": list(shard.get("doc_ids", [])),
            "args": args,
            "output_dir": str(shard_dir_root / f"shard_{shard['shard_id']}"),
            "guidance_payload": guidance_payload,
        }
        for shard in shards
    ]
    shard_results = _execute_guided_sleep_shard_tasks(tasks=tasks, max_workers=worker_count, progress_path=progress_path)
    merged_bridges = _merge_bridge_dicts([bridge for shard_result in shard_results for bridge in shard_result.get("bridges", [])])

    generation_shards = [
        {
            "shard_id": result["shard_id"],
            "question_ids": list(result.get("question_ids", [])),
            "questions": list(result.get("questions", [])),
            "doc_ids": list(result.get("doc_ids", [])),
            "runtime_seconds": float(result.get("runtime_seconds", 0.0) or 0.0),
            "accepted_bridge_count": len(result.get("bridges", [])),
            "run_results": dict(result.get("run_results", {})),
            "stage_summary": dict(result.get("stage_summary", {})),
            "output_dir": str(result.get("output_dir", "")),
        }
        for result in shard_results
    ]
    stage_summary = {
        "generation_mode": "question_group_sharded",
        "generation_shards": generation_shards,
        "candidate_count": sum(int(row.get("stage_summary", {}).get("candidate_count", 0) or 0) for row in generation_shards),
        "worker_proposals_generated": sum(int(row.get("stage_summary", {}).get("worker_proposals_generated", 0) or 0) for row in generation_shards),
        "verifier_rejections": sum(int(row.get("stage_summary", {}).get("verifier_rejections", 0) or 0) for row in generation_shards),
        "accepted_bridge_count": len(merged_bridges),
        "edges_skipped_no_path": sum(int(row.get("stage_summary", {}).get("edges_skipped_no_path", 0) or 0) for row in generation_shards),
        "parallel_batches": sum(int(row.get("stage_summary", {}).get("parallel_batches", 0) or 0) for row in generation_shards),
        "errors_count": sum(int(row.get("stage_summary", {}).get("errors_count", 0) or 0) for row in generation_shards),
        "likely_zero_bridge_reasons": sorted(
            {
                str(reason)
                for row in generation_shards
                for reason in row.get("stage_summary", {}).get("likely_zero_bridge_reasons", []) or []
                if str(reason).strip()
            }
        ),
        "failure_reason": "",
        "run_output_dir": str(output_dir),
    }
    _write_json(output_dir / "sleep_stage_summary.json", stage_summary)
    return {
        "batch_index": int(batch_index),
        "doc_ids": sorted({doc_id for item in batch_items for doc_id in item.doc_ids}),
        "bridges": merged_bridges,
        "generation_mode": "question_group_sharded",
        "generation_executor": "thread",
        "generation_shards": generation_shards,
        "stage_summary": stage_summary,
    }


def _finalize_success_artifacts(
    *,
    output_dir: Path,
    progress_path: Path,
    batch_index: int,
    batch_doc_ids: Sequence[str],
    bridge_registry: BridgeRegistry,
    snapshot_path: str,
    generation_result: Dict[str, Any],
) -> None:
    bridges = list(generation_result.get("bridges", []) or [])
    stage_summary = dict(generation_result.get("stage_summary", {}) or {})
    summary = {
        "batch_index": int(batch_index),
        "doc_ids": list(batch_doc_ids),
        "bridge_count": len(bridges),
        "registry_size": len(bridge_registry.records),
        "snapshot_path": snapshot_path,
        "generation_mode": str(generation_result.get("generation_mode", stage_summary.get("generation_mode", "single_batch"))),
        "generation_shards": list(generation_result.get("generation_shards", stage_summary.get("generation_shards", [])) or []),
        "candidate_count": int(stage_summary.get("candidate_count", 0) or 0),
        "accepted_bridge_count": int(stage_summary.get("accepted_bridge_count", len(bridges)) or len(bridges)),
        "worker_proposals_generated": int(stage_summary.get("worker_proposals_generated", 0) or 0),
        "verifier_rejections": int(stage_summary.get("verifier_rejections", 0) or 0),
        "likely_zero_bridge_reasons": list(stage_summary.get("likely_zero_bridge_reasons", []) or []),
        "failure_reason": "",
        "run_output_dir": str(output_dir),
        "sleep_stage_summary_path": str(output_dir / "sleep_stage_summary.json"),
    }
    _write_json(output_dir / "guided_sleep_result.json", summary)
    _write_progress(
        progress_path,
        "guided_sleep",
        "completed",
        batch_index=int(batch_index),
        bridge_count=len(bridges),
        registry_size=len(bridge_registry.records),
        snapshot_path=snapshot_path,
        generation_mode=summary["generation_mode"],
    )


def _finalize_failure_artifacts(
    *,
    output_dir: Path,
    progress_path: Path,
    batch_index: int,
    failure_reason: str,
) -> None:
    payload = {
        "batch_index": int(batch_index),
        "failure_reason": str(failure_reason or ""),
        "run_output_dir": str(output_dir),
    }
    _write_json(output_dir / "guided_sleep_failure.json", payload)
    _write_progress(progress_path, "guided_sleep", "failed", batch_index=int(batch_index), error=str(failure_reason or ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guided sleep-time exploration for one next-batch doc set")
    parser.add_argument("--questions_path", "--manifest", dest="questions_path", default="manifests/2wiki_continual_49q_full_docs.json")
    parser.add_argument("--batch_index", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed_batch_size", type=int, default=8)
    parser.add_argument("--gsw_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--guidance_file", required=True)
    parser.add_argument("--bridge_registry_snapshot", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--root_model", default=None)
    parser.add_argument("--worker_model", default=None)
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
    parser.add_argument("--max_tokens", type=int, default=500000)
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--reasoning_effort", default="medium")
    parser.add_argument("--disable_bridge_verifier", action="store_true")
    parser.add_argument("--bridge_verifier_threshold", type=float, default=0.7)
    parser.add_argument("--bridge_verifier_fail_open", action="store_true")
    parser.add_argument("--bridge_verifier_model", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "sleep_progress.jsonl"
    _write_progress(progress_path, "guided_sleep", "started", batch_index=int(args.batch_index), model=args.model)
    console.print(f"\nPreparing guided sleep for batch {args.batch_index}")
    console.print(f"  model: {args.model}")
    if args.base_url:
        console.print(f"  base URL: {args.base_url}")

    guidance_summary = InteractionGuidanceSummary(**json.loads(Path(args.guidance_file).read_text()))
    guidance_payload = build_sleep_guidance_payload(
        guidance_summary,
        targeted_edge_ratio=float(args.targeted_edge_ratio),
    )
    _write_json(output_dir / "sleep_guidance_payload.json", guidance_payload)

    from gsw_memory.query_then_sleep.prompts import render_interaction_guidance_block

    rendered_guidance = render_interaction_guidance_block(guidance_payload)
    (output_dir / "sleep_guidance_rendered.txt").write_text(rendered_guidance)
    console.print(f"  [dim]Rendered guidance saved to: {output_dir / 'sleep_guidance_rendered.txt'}[/dim]")
    if rendered_guidance:
        console.print(f"\n  [bold magenta]Guidance to sleep agent:[/bold magenta]")
        for line in rendered_guidance.split("\n"):
            console.print(f"    [magenta]{line}[/magenta]")

    questions, schema = run_bridge_test.load_questions(args.questions_path)
    title_to_idx = None
    corpus_path = args.corpus_path or run_bridge_test.infer_corpus_path(args.questions_path)
    if schema == "context_titles":
        title_to_idx = run_bridge_test.build_title_to_doc_index(corpus_path or "")
    items = run_bridge_test.build_batch_query_items(questions, schema=schema, title_to_idx=title_to_idx)
    batches = split_curriculum_batches(items, batch_size=args.batch_size, seed_batch_size=args.seed_batch_size)
    batch_items = list(batches[args.batch_index])
    batch_doc_ids = sorted({doc_id for item in batch_items for doc_id in item.doc_ids})

    try:
        use_sharded_generation = bool(args.curriculum_generation_parallel_enabled) and len(batch_items) > 1
        if use_sharded_generation:
            generation_result = _run_guided_sleep_sharded(
                batch_index=args.batch_index,
                batch_items=batch_items,
                args=args,
                output_dir=output_dir,
                guidance_payload=guidance_payload,
                progress_path=progress_path,
            )
        else:
            generation_result = _run_guided_sleep_doc_batch(
                batch_index=args.batch_index,
                doc_ids=batch_doc_ids,
                args=args,
                output_dir=output_dir,
                guidance_payload=guidance_payload,
            )
            generation_result["generation_mode"] = "single_batch"
            generation_result["generation_shards"] = []

        temp_base = tempfile.mkdtemp(prefix=f"query_then_sleep_registry_{args.batch_index}_")
        try:
            temp_gsw_dir = run_bridge_test.setup_temp_gsw_dir(args.gsw_path, batch_doc_ids, temp_base, subdir="networks")
            cache_dir = run_bridge_test._curriculum_cache_dir(str(output_dir), "sleep_docs_registry", batch_doc_ids)
            entity_searcher = EntitySearcher(
                num_documents=len(batch_doc_ids),
                path_to_gsw_files=temp_gsw_dir,
                cache_dir=cache_dir,
                rebuild_cache=False,
                verbose=False,
                use_bm25=True,
                use_gpu_for_qa_index=False,
                embedding_gpu_memory_utilization=args.embedding_gpu_memory_utilization,
            )
            bridge_registry = _load_bridge_registry_snapshot(args.bridge_registry_snapshot, entity_searcher)
            embed_fn = getattr(entity_searcher, "_embed_query", None)
            ingest_result = bridge_registry.ingest_bridges(
                list(generation_result.get("bridges", []) or []),
                batch_index=args.batch_index,
                embed_texts_fn=(lambda texts: entity_searcher._embed_query(list(texts))) if callable(embed_fn) else None,
                pattern_classifier=None,
                strict_pattern_tagging=False,
            )
            snapshot_path = run_bridge_test._write_bridge_registry_snapshot(
                batch_output_dir=str(output_dir),
                batch_index=args.batch_index,
                bridge_registry=bridge_registry,
                ingest_result=ingest_result,
            )
        finally:
            shutil.rmtree(temp_base, ignore_errors=True)

        _finalize_success_artifacts(
            output_dir=output_dir,
            progress_path=progress_path,
            batch_index=args.batch_index,
            batch_doc_ids=batch_doc_ids,
            bridge_registry=bridge_registry,
            snapshot_path=snapshot_path,
            generation_result=generation_result,
        )
    except Exception as exc:
        _finalize_failure_artifacts(
            output_dir=output_dir,
            progress_path=progress_path,
            batch_index=args.batch_index,
            failure_reason=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
