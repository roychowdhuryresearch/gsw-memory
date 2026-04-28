#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.simple_entity_search import EntitySearcher
from playground.sleep_time import run_bridge_test
from gsw_memory.query_then_sleep import run_query_agent_batch
from gsw_memory.query_then_sleep.logging_utils import StageLogger, console
from gsw_memory.sleep_time.curriculum import BridgeRecord, BridgeRegistry, BridgeSurface, split_curriculum_batches


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


def _write_bridge_registry_snapshot(output_dir: Path, batch_index: int, bridge_registry: BridgeRegistry) -> Path:
    payload = {
        "batch_index": int(batch_index),
        "added_bridge_count": 0,
        "added_bridge_ids": [],
        "registry_bridge_count": len(bridge_registry.records),
        "registry_surface_count": len(bridge_registry.surfaces),
        "new_bridge_records": [],
        "all_bridge_records": [
            bridge_registry.records[bridge_id].exemplar()
            for bridge_id in sorted(bridge_registry.records.keys())
        ],
        "ingest_metrics": {},
    }
    path = output_dir / "bridge_registry_snapshot.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated query-time agent on one manifest batch")
    parser.add_argument("--questions_path", "--manifest", dest="questions_path", default="manifests/2wiki_continual_49q_full_docs.json")
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed_batch_size", type=int, default=8)
    parser.add_argument("--gsw_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--bridge_registry_snapshot", default=None)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--bridge_query_top_k", type=int, default=10)
    parser.add_argument("--bridge_inject_min_score", type=float, default=0.30)
    parser.add_argument("--bridge_inject_top_k", type=int, default=5)
    parser.add_argument("--embedding_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--reasoning_effort", default="medium")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_logger = StageLogger(
        stage_name="query",
        output_dir=output_dir,
        verbose=args.verbose,
        show_thinking=args.show_thinking,
    )
    stage_logger.progress("query_batch", "started", batch_index=int(args.batch_index), model=args.model)
    console.print(f"\nInitializing query agent with model: {args.model}")
    if args.base_url:
        console.print(f"  base URL: {args.base_url}")

    questions, schema = run_bridge_test.load_questions(args.questions_path)
    title_to_idx = None
    corpus_path = args.corpus_path or run_bridge_test.infer_corpus_path(args.questions_path)
    if schema == "context_titles":
        title_to_idx = run_bridge_test.build_title_to_doc_index(corpus_path or "")
    items = run_bridge_test.build_batch_query_items(questions, schema=schema, title_to_idx=title_to_idx)
    batches = split_curriculum_batches(items, batch_size=args.batch_size, seed_batch_size=args.seed_batch_size)
    batch_items = list(batches[args.batch_index])
    batch_doc_ids = sorted({doc_id for item in batch_items for doc_id in item.doc_ids})
    stage_logger.info("Loaded %s batch items across %s docs", len(batch_items), len(batch_doc_ids))

    temp_base = tempfile.mkdtemp(prefix=f"query_then_sleep_batch_{args.batch_index}_")
    try:
        temp_gsw_dir = run_bridge_test.setup_temp_gsw_dir(args.gsw_path, batch_doc_ids, temp_base, subdir="networks")
        cache_dir = run_bridge_test._curriculum_cache_dir(str(output_dir), "query_batch", batch_doc_ids)
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
        stage_logger.event(
            "query_registry_loaded",
            {
                "registry_bridge_count": len(bridge_registry.records),
                "registry_surface_count": len(bridge_registry.surfaces),
                "snapshot_path": args.bridge_registry_snapshot or "",
            },
        )
        result = run_query_agent_batch(
            batch_index=args.batch_index,
            batch_items=batch_items,
            entity_searcher=entity_searcher,
            model_name=args.model,
            base_url=args.base_url,
            bridge_registry=bridge_registry,
            max_iterations=args.max_iterations,
            bridge_top_k=args.bridge_query_top_k,
            bridge_inject_min_score=args.bridge_inject_min_score,
            bridge_inject_top_k=args.bridge_inject_top_k,
            reasoning_effort=args.reasoning_effort,
            event_callback=stage_logger.event,
        )
        for trace in result.traces:
            used = list(trace.bridge_ids_used)
            helpful = used if (trace.bridge_evidence_used_in_answer and float(trace.f1 or 0.0) >= 0.5) else []
            bridge_registry.record_query_usage(used, helpful, args.batch_index)
            stage_logger.progress(
                "question",
                "completed",
                question_id=trace.question_id,
                query_status=trace.status,
                f1=trace.f1,
                exact_match=trace.exact_match,
                bridge_evidence_used=trace.bridge_evidence_used,
            )
    except Exception as exc:
        stage_logger.log_exception("Query batch failed", exc)
        stage_logger.progress("query_batch", "failed", error=str(exc), batch_index=int(args.batch_index))
        raise
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)

    (output_dir / "query_agent_results.json").write_text(json.dumps(result.as_dict(), indent=2))
    with open(output_dir / "query_interaction_traces.jsonl", "w") as f:
        for trace in result.traces:
            f.write(json.dumps(trace.as_dict()) + "\n")

    if bridge_registry.records:
        snapshot_path = _write_bridge_registry_snapshot(output_dir, args.batch_index, bridge_registry)
        stage_logger.event(
            "query_registry_saved",
            {
                "snapshot_path": str(snapshot_path),
                "registry_bridge_count": len(bridge_registry.records),
            },
        )

    stage_logger.progress(
        "query_batch",
        "completed",
        batch_index=int(args.batch_index),
        question_count=len(result.traces),
        metrics=result.overall_metrics,
    )
    stage_logger.event(
        "query_finished",
        {
            "batch_index": int(args.batch_index),
            "question_count": len(result.traces),
            "metrics": result.overall_metrics,
        },
    )
    stage_logger.close()


if __name__ == "__main__":
    main()
