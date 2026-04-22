#!/usr/bin/env python3
"""
Bridge Test Runner: run sleep-time agent on bridge-style QA datasets.

Standard mode:
1. Creates a temp directory with symlinks to one question's GSW doc dirs
2. Runs run_sleep_time.py — agent discovers entities itself using discovery tools
3. Collects bridge QAs and tags them with question metadata

Curriculum mode:
1. Groups question+doc manifests into deterministic batches
2. Runs sleep-time bridge creation on the batch doc union
3. Answers batch queries over accumulated bridge + doc evidence without decomposition
4. Feeds demand/gap guidance into the next bridge-generation batch

Supported schemas:
- MuSiQue-style: `paragraphs` list with `idx` and `is_supporting`
- 2Wiki/Hotpot-style: `context` + `supporting_facts` with title-based doc mapping
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

try:
    from bespokelabs import curator
except ImportError:  # pragma: no cover - optional dependency path exercised in integration
    curator = None  # type: ignore[assignment]

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from playground.simple_entity_search import EntitySearcher
from playground.sleep_time.run_sleep_time import run_sleep_time_once
from src.gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler
from src.gsw_memory.sleep_time.curriculum import (
    BatchQueryItem,
    BridgeRegistry,
    QuestionPatternClassifier,
    QuestionPatternResponseSchema,
    QueryAnswerResult,
    QueryBatchResult,
    QueryExperienceState,
    accumulate_query_experience_state,
    build_batch_feedback_brief,
    build_current_batch_demand_sketch,
    build_curriculum_guidance_payload,
    build_panini_answer_messages,
    build_prior_query_demand_sketch,
    build_query_demand_profile,
    build_relationship_to_patterns,
    collect_doc_evidence,
    infer_question_patterns,
    parse_answer_from_response,
    rerank_merged_evidence,
    score_predicted_answer,
    split_curriculum_batches,
    compute_guidance_entropy,
    apply_demand_decay,
)
from src.gsw_memory.sleep_time.tools import normalize_similarity_text

console = Console()

DEFAULT_DOCS_PER_QUESTION = 20
CURATOR_AVAILABLE = curator is not None


def _truncate_for_console(text: Any, limit: int = 80) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _serialize_retrieval_trace_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_type": row.get("source_type"),
        "bridge_id": row.get("bridge_id"),
        "doc_id": row.get("doc_id"),
        "question": row.get("question"),
        "evidence_text": row.get("evidence_text", ""),
        "rerank_text": row.get("rerank_text", ""),
        "score": row.get("score", 0.0),
        "normalized_base_score": row.get("normalized_base_score"),
        "semantic_score": row.get("semantic_score"),
        "embedding_similarity": row.get("embedding_similarity"),
        "final_score": row.get("final_score", row.get("score", 0.0)),
        "rank_before_merge": row.get("rank_before_merge"),
        "rank_after_merge": row.get("rank_after_merge"),
        "is_equivalent_bridge_match": bool(row.get("is_equivalent_bridge_match", False)),
        "protected_keep": bool(row.get("protected_keep", False)),
        "kept": bool(row.get("kept", False)),
    }


def _top_trace_score(rows: List[Dict[str, Any]], source_type: str) -> str:
    scores = [
        float(row.get("final_score"))
        for row in rows
        if str(row.get("source_type", "")) == source_type and row.get("final_score") is not None
    ]
    if not scores:
        return "n/a"
    return f"{max(scores):.4f}"


def _write_query_batch_artifact(
    batch_output_dir: str,
    batch_index: int,
    batch_doc_ids: List[str],
    accumulated_doc_ids: List[str],
    answer_cache_dir: str,
    query_batch_result: QueryBatchResult,
) -> str:
    payload = query_batch_result.as_dict()
    payload["metadata"] = {
        "batch_index": batch_index,
        "batch_doc_ids": list(batch_doc_ids),
        "accumulated_doc_ids": list(accumulated_doc_ids),
        "answer_cache_dir": answer_cache_dir,
    }
    artifact_path = os.path.join(batch_output_dir, "query_answer_results.json")
    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return artifact_path


class CurriculumProgressTracker:
    def __init__(self, batch_index: int, batch_output_dir: str) -> None:
        self.batch_index = batch_index
        self.batch_output_dir = batch_output_dir
        self.artifact_path = os.path.join(batch_output_dir, "curriculum_progress.jsonl")
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self._task_ids: Dict[str, int] = {}

    def __enter__(self) -> "CurriculumProgressTracker":
        os.makedirs(self.batch_output_dir, exist_ok=True)
        with open(self.artifact_path, "w") as f:
            f.write("")
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def emit(self, event: Dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "batch_index": self.batch_index,
            **event,
        }
        os.makedirs(self.batch_output_dir, exist_ok=True)
        with open(self.artifact_path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        self._update_console(payload)

    def callback(self, event: Dict[str, Any]) -> None:
        self.emit(event)

    def _update_console(self, payload: Dict[str, Any]) -> None:
        phase = str(payload.get("phase", "unknown"))
        status = str(payload.get("status", "progress"))
        completed = int(payload.get("completed", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        unit = str(payload.get("unit", "items"))
        description = {
            "bridge_generation_shards": "Generating bridge shards",
            "bridge_pattern_classification": "Classifying bridge patterns",
            "bridge_index_rebuild": "Rebuilding bridge index",
            "query_answering": "Answering queries",
        }.get(phase, phase.replace("_", " ").title())
        if status == "started":
            if phase == "bridge_pattern_classification":
                console.print(f"  {description}: 0/{total} {unit}")
            elif phase == "query_answering":
                console.print(f"  {description}: 0/{total} {unit}")
            else:
                console.print(f"  {description}...")
        elif status == "completed" and phase == "bridge_index_rebuild":
            message = str(payload.get("message", "") or "")
            if message:
                console.print(f"  {description}: {message}")
        if phase not in self._task_ids:
            task_total = total if total > 0 else 1
            self._task_ids[phase] = self._progress.add_task(description, total=task_total)
        task_id = self._task_ids[phase]
        task_total = total if total > 0 else 1
        if status == "started":
            self._progress.update(task_id, description=description, completed=0, total=task_total)
            return
        if status == "completed":
            self._progress.update(task_id, description=description, completed=task_total, total=task_total)
            return
        self._progress.update(task_id, description=description, completed=completed, total=task_total)


def _write_bridge_registry_snapshot(
    batch_output_dir: str,
    batch_index: int,
    bridge_registry: BridgeRegistry,
    ingest_result: Any,
) -> str:
    added_bridge_ids = list(getattr(ingest_result, "added_bridge_ids", []) or [])
    new_bridge_records = [
        bridge_registry.records[bridge_id].exemplar()
        for bridge_id in added_bridge_ids
        if bridge_id in bridge_registry.records
    ]
    all_bridge_records = [
        bridge_registry.records[bridge_id].exemplar()
        for bridge_id in sorted(bridge_registry.records.keys())
    ]
    payload = {
        "batch_index": batch_index,
        "added_bridge_count": int(getattr(ingest_result, "added_count", len(new_bridge_records)) or 0),
        "added_bridge_ids": added_bridge_ids,
        "registry_bridge_count": len(bridge_registry.records),
        "registry_surface_count": len(bridge_registry.surfaces),
        "new_bridge_records": new_bridge_records,
        "all_bridge_records": all_bridge_records,
        "ingest_metrics": ingest_result.as_dict() if hasattr(ingest_result, "as_dict") else {},
    }
    artifact_path = os.path.join(batch_output_dir, "bridge_registry_snapshot.json")
    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return artifact_path


def _write_bridge_pattern_classification_report(
    batch_output_dir: str,
    batch_index: int,
    ingest_result: Any,
) -> str:
    payload = {
        "batch_index": batch_index,
        "status": "failed" if int(getattr(ingest_result, "failed_surface_count", 0) or 0) > 0 else "ok",
        **(ingest_result.as_dict() if hasattr(ingest_result, "as_dict") else {}),
    }
    artifact_path = os.path.join(batch_output_dir, "bridge_pattern_classification_report.json")
    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return artifact_path


def _print_query_batch_summary(batch_index: int, query_batch_result: QueryBatchResult) -> None:
    for query_result in query_batch_result.query_results:
        score_parts: List[str] = []
        if query_result.f1 is not None:
            score_parts.append(f"F1={query_result.f1:.2f}")
        if query_result.exact_match is not None:
            score_parts.append(f"EM={query_result.exact_match:.2f}")
        score_text = " | ".join(score_parts) if score_parts else "unscored"
        bridge_text = "yes" if query_result.bridge_evidence_used else "no"
        trace = query_result.retrieval_trace or {}
        kept_candidates = list(trace.get("kept_candidates", []) or [])
        kept_sources = ",".join(
            sorted({str(row.get("source_type", "unknown")) for row in kept_candidates if row.get("source_type")})
        ) or "none"
        top_bridge_score = _top_trace_score(list(trace.get("merged_candidates", []) or []), "bridge")
        top_doc_score = max(
            (
                _top_trace_score(list(trace.get("merged_candidates", []) or []), "doc_entity"),
                _top_trace_score(list(trace.get("merged_candidates", []) or []), "doc_direct_qa"),
            ),
            key=lambda value: -1.0 if value == "n/a" else float(value),
        )
        console.print(
            "  Query answer: "
            f"batch={batch_index} | "
            f"q={_truncate_for_console(query_result.question, limit=72)!r} | "
            f"pred={_truncate_for_console(query_result.predicted_answer or 'N/A', limit=56)!r} | "
            f"{score_text} | bridge={bridge_text} | "
            f"kept={kept_sources} | "
            f"exact_bridge={query_result.exact_bridge_match_found}/{query_result.exact_bridge_match_kept} | "
            f"equivalent_bridge={query_result.equivalent_bridge_match_found}/{query_result.equivalent_bridge_match_kept} | "
            f"top_bridge={top_bridge_score} | top_doc={top_doc_score}"
        )


def load_questions(path: str, start: int = 0, end: int = None):
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected question file to be a JSON list, got {type(data).__name__}")

    data = data[start:end]
    questions = [(start + i, d) for i, d in enumerate(data) if d.get("answerable", True)]
    if not questions:
        return [], "unknown"

    sample = questions[0][1]
    if "paragraphs" in sample:
        schema = "musique"
    elif "context" in sample and "supporting_facts" in sample:
        schema = "context_titles"
    else:
        schema = "unknown"

    return questions, schema


def infer_corpus_path(questions_path: str) -> str | None:
    qpath = Path(questions_path)
    if qpath.suffix != ".json":
        return None
    candidate = qpath.with_name(f"{qpath.stem}_corpus.json")
    return str(candidate) if candidate.exists() else None


def build_title_to_doc_index(corpus_path: str) -> dict[str, int]:
    with open(corpus_path) as f:
        corpus = json.load(f)
    if not isinstance(corpus, list):
        raise ValueError(f"Expected corpus file to be a JSON list, got {type(corpus).__name__}")

    title_to_idx: dict[str, int] = {}
    for idx, item in enumerate(corpus):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if title:
            title_to_idx[title] = idx
    return title_to_idx


def normalize_question_metadata(
    question: dict,
    q_idx: int,
    schema: str,
    title_to_idx: dict[str, int] | None = None,
):
    if schema == "musique":
        paragraphs = question.get("paragraphs", [])
        if not paragraphs:
            raise ValueError("MuSiQue question is missing non-empty 'paragraphs'")

        docs_per_question = len(paragraphs) if paragraphs else DEFAULT_DOCS_PER_QUESTION
        global_indices = [q_idx * docs_per_question + i for i in range(docs_per_question)]
        supporting_local = [p["idx"] for p in paragraphs if p.get("is_supporting")]
        supporting_global = [q_idx * docs_per_question + local for local in supporting_local]
        supporting_titles = [p.get("title", "") for p in paragraphs if p.get("is_supporting")]
        decomposition = question.get("question_decomposition", [])

        return {
            "question_id": question.get("id", ""),
            "question_text": question.get("question", ""),
            "answer": question.get("answer", ""),
            "answer_aliases": question.get("answer_aliases", []),
            "num_hops": len(decomposition),
            "decomposition": decomposition,
            "global_indices": global_indices,
            "supporting_local": supporting_local,
            "supporting_global": supporting_global,
            "supporting_titles": supporting_titles,
        }

    if schema == "context_titles":
        if title_to_idx is None:
            raise ValueError("Title index is required for context_titles schema")

        context = question.get("context", [])
        supporting_facts = question.get("supporting_facts", [])
        if not context:
            raise ValueError("Question is missing non-empty 'context'")

        supporting_title_set = {fact[0] for fact in supporting_facts if isinstance(fact, list) and fact}

        global_indices = []
        missing_titles = []
        supporting_local = []
        supporting_global = []
        supporting_titles = []

        for local_idx, entry in enumerate(context):
            title = entry[0] if isinstance(entry, list) and entry else None
            if not title:
                continue

            global_idx = title_to_idx.get(title)
            if global_idx is None:
                missing_titles.append(title)
                continue

            global_indices.append(global_idx)
            if title in supporting_title_set:
                supporting_local.append(local_idx)
                supporting_global.append(global_idx)
                supporting_titles.append(title)

        if missing_titles:
            console.print(
                f"  [yellow]Warning: {len(missing_titles)} titles not found in corpus index for q={q_idx}[/yellow]"
            )

        if not global_indices:
            raise ValueError("No mappable context titles found in corpus for this question")

        decomposition = question.get("question_decomposition", question.get("evidences", []))
        answer_aliases = question.get("answer_aliases", [])
        if not answer_aliases and question.get("answer"):
            answer_aliases = [question["answer"]]

        return {
            "question_id": question.get("_id", question.get("id", "")),
            "question_text": question.get("question", ""),
            "answer": question.get("answer", ""),
            "answer_aliases": answer_aliases,
            "num_hops": len(set(supporting_titles)) if supporting_titles else len(supporting_facts),
            "decomposition": decomposition,
            "global_indices": global_indices,
            "supporting_local": supporting_local,
            "supporting_global": supporting_global,
            "supporting_titles": supporting_titles,
        }

    raise ValueError(
        "Unsupported question schema. Expected MuSiQue-style ('paragraphs') "
        "or context-title schema ('context' + 'supporting_facts')."
    )


def _normalize_doc_name(doc_id: Any) -> str:
    text = str(doc_id).strip()
    if text.startswith("doc_"):
        return text
    return f"doc_{text}"


def setup_temp_gsw_dir(gsw_path: str, doc_ids: List[Any], temp_base: str, subdir: str = "networks") -> str:
    temp_dir = os.path.join(temp_base, subdir)
    os.makedirs(temp_dir, exist_ok=True)

    for doc_id in doc_ids:
        doc_name = _normalize_doc_name(doc_id)
        source = os.path.join(gsw_path, doc_name)
        target = os.path.join(temp_dir, doc_name)
        if os.path.exists(target):
            continue
        if os.path.exists(source):
            os.symlink(os.path.abspath(source), target)
        else:
            console.print(f"  [yellow]Warning: {source} not found[/yellow]")

    return temp_dir


def _doc_set_hash(doc_ids: List[Any]) -> str:
    normalized = sorted({_normalize_doc_name(doc_id) for doc_id in doc_ids})
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _curriculum_cache_dir(output_dir: str, namespace: str, doc_ids: List[Any]) -> str:
    normalized = sorted({_normalize_doc_name(doc_id) for doc_id in doc_ids})
    digest = _doc_set_hash(normalized)
    cache_dir = Path(output_dir) / "_curriculum_cache" / namespace / f"{len(normalized)}docs_{digest}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def _build_sleep_time_args(
    args,
    *,
    gsw_path: str,
    num_docs: int,
    output_dir: str,
    cache_dir: str,
):
    return SimpleNamespace(
        num_docs=num_docs,
        gsw_path=gsw_path,
        cache_dir=cache_dir,
        embedding_gpu_memory_utilization=getattr(args, "embedding_gpu_memory_utilization", 0.5),
        num_entities=getattr(args, "num_entities", 0),
        min_docs=getattr(args, "min_docs", 2),
        filter_generic=getattr(args, "filter_generic", False),
        seed_entities_file=getattr(args, "seed_entities_file", None),
        model=args.model,
        base_url=getattr(args, "base_url", None),
        max_tokens=getattr(args, "max_tokens", 500_000),
        max_iterations=getattr(args, "max_iterations", 30),
        reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        disable_bridge_verifier=getattr(args, "disable_bridge_verifier", False),
        bridge_verifier_threshold=getattr(args, "bridge_verifier_threshold", 0.7),
        bridge_verifier_fail_open=getattr(args, "bridge_verifier_fail_open", False),
        bridge_verifier_model=getattr(args, "bridge_verifier_model", None),
        pipeline_mode=getattr(args, "pipeline_mode", "legacy"),
        hybrid_scope=getattr(args, "hybrid_scope", "doc_edge"),
        root_model=getattr(args, "root_model", None),
        worker_model=getattr(args, "worker_model", None),
        edge_max_depth=getattr(args, "edge_max_depth", 1),
        edge_max_calls=getattr(args, "edge_max_calls", 2),
        edge_max_tokens=getattr(args, "edge_max_tokens", 3000),
        max_optional_docs_per_edge=getattr(args, "max_optional_docs_per_edge", 2),
        edge_parallel_enabled=getattr(args, "edge_parallel_enabled", False),
        edge_parallel_workers=getattr(args, "edge_parallel_workers", 1),
        parallel_progress_heartbeat_seconds=getattr(args, "parallel_progress_heartbeat_seconds", 10.0),
        parallel_stuck_warning_seconds=getattr(args, "parallel_stuck_warning_seconds", 60.0),
        verifier_parallel_workers=getattr(args, "verifier_parallel_workers", 4),
        edge_max_accepted_bridges=getattr(args, "edge_max_accepted_bridges", 10),
        curriculum_guidance_file=None,
        output_dir=output_dir,
        checkpoint_interval=getattr(args, "checkpoint_interval", 5),
        verbose=getattr(args, "verbose", False),
        show_thinking=getattr(args, "show_thinking", False),
        resume_from=None,
    )


def build_batch_query_items(
    questions: List[tuple[int, dict]],
    schema: str,
    title_to_idx: Optional[dict[str, int]] = None,
) -> List[BatchQueryItem]:
    items: List[BatchQueryItem] = []
    for q_idx, question in questions:
        meta = normalize_question_metadata(question, q_idx, schema=schema, title_to_idx=title_to_idx)
        gold_answer = meta["answer"] or None
        answer_aliases = list(meta.get("answer_aliases", []))
        if gold_answer and gold_answer not in answer_aliases:
            answer_aliases = [gold_answer] + answer_aliases
        item = BatchQueryItem(
            question=meta["question_text"],
            doc_ids=[_normalize_doc_name(idx) for idx in meta["global_indices"]],
            gold_answer=gold_answer,
            answer_aliases=answer_aliases,
            question_id=meta["question_id"],
            metadata={
                "question_idx": q_idx,
                "num_hops": meta["num_hops"],
                "decomposition": meta["decomposition"],
                "supporting_doc_indices_local": meta["supporting_local"],
                "supporting_doc_indices_global": meta["supporting_global"],
                "supporting_doc_titles": meta["supporting_titles"],
                "all_doc_indices_global": list(meta["global_indices"]),
                "split": question.get("_experiment_split", "train"),
                "family": question.get("_experiment_family", ""),
            },
        )
        items.append(item)
    return items


def _build_embed_texts_fn(entity_searcher: EntitySearcher):
    def _embed(texts: List[str]):
        if not texts or not hasattr(entity_searcher, "_embed_query"):
            return None
        return entity_searcher._embed_query(list(texts))

    return _embed


def _build_answer_reconciler(entity_searcher: EntitySearcher, args) -> AgenticReconciler:
    return AgenticReconciler(
        entity_searcher=entity_searcher,
        model_name=args.model,
        budget={"max_entities": 0, "max_documents": len(entity_searcher.gsw_by_doc_id), "max_tokens": args.max_tokens},
        verbose=False,
        output_callback=None,
        reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        base_url=getattr(args, "base_url", None),
        bridge_verifier_enabled=False,
        bridge_verifier_model=getattr(args, "bridge_verifier_model", None),
        pipeline_mode="legacy",
    )


def _build_pattern_classification_messages(question: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You classify one question for curriculum guidance. "
        "Return strict JSON only with keys: operation, answer_type, domain, relation_label, comparison_target, pattern_key.\n"
        "Allowed operation: lookup | compare | filter | count | boolean | superlative | temporal_order\n"
        "Allowed answer_type: person | place | date | year | number | title | organization | region | other\n"
        "Allowed domain: family | film | history_politics | geography | sports | organization | other\n"
        "Allowed comparison_target: none | earlier | later | before | after | first | last | higher | lower\n"
        "relation_label must be short snake_case and describe the main relation or attribute in the question.\n"
        "pattern_key must be deterministic and formatted as <relation_label>.<operation>.<comparison_target?>.<answer_type> "
        "(omit comparison_target when it is none)."
    )
    user_prompt = f"Question: {question}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


if CURATOR_AVAILABLE:
    class _BridgeSurfacePatternOperator(curator.LLM):
        response_format = QuestionPatternResponseSchema

        def prompt(self, input_row: Dict[str, Any]) -> List[Dict[str, str]]:
            return _build_pattern_classification_messages(str(input_row.get("question", "") or ""))

        def parse(self, input_row: Dict[str, Any], response: QuestionPatternResponseSchema) -> Dict[str, Any]:
            payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            payload["classifier_source"] = "llm"
            return {
                "question": str(input_row.get("question", "") or ""),
                "pattern": payload,
            }


def _classify_question_with_model(classifier_agent: AgenticReconciler, args, question: str) -> Dict[str, Any]:
    selected_model = getattr(args, "worker_model", None) or getattr(args, "model", None)
    if not selected_model:
        raise ValueError("No model configured for question pattern classification")
    message = classifier_agent._call_model_for_stage(
        messages=_build_pattern_classification_messages(question),
        model_name=selected_model,
        max_tokens=500,
        temperature=0.0,
        stage="worker",
        response_format=QuestionPatternResponseSchema,
    )
    raw_text, _source = classifier_agent._extract_message_content_with_source(message)
    parsed = classifier_agent._extract_json_from_text(raw_text)
    validated = QuestionPatternResponseSchema.model_validate(parsed)
    payload = validated.model_dump()
    payload["classifier_source"] = "llm"
    return payload


def _build_pattern_classifier_cache_paths(output_dir: str, model_name: str) -> Dict[str, str]:
    digest = hashlib.sha1(str(model_name or "").encode("utf-8")).hexdigest()[:12]
    base_dir = Path(output_dir) / "_curriculum_cache" / "pattern_classification" / f"model_{digest}"
    base_dir.mkdir(parents=True, exist_ok=True)
    curator_dir = base_dir / "curator_workdir"
    curator_dir.mkdir(parents=True, exist_ok=True)
    return {
        "query_pattern_cache_path": str(base_dir / "query_patterns.json"),
        "bridge_surface_cache_path": str(base_dir / "bridge_surface_patterns.json"),
        "curator_working_dir": str(curator_dir),
    }


def _normalize_curator_dataset_rows(dataset: Any) -> List[Dict[str, Any]]:
    rows = getattr(dataset, "dataset", dataset)
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(dict(row))
    return normalized


def _curator_backend_params(args) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "request_timeout": 300.0,
        "max_concurrent_requests": 64,
        "require_all_responses": False,
    }
    if getattr(args, "base_url", None):
        params["base_url"] = args.base_url
    return params


def _normalize_curator_litellm_model_name(args, model_name: str) -> str:
    selected_model = str(model_name or "").strip()
    if getattr(args, "base_url", None):
        return selected_model if selected_model.startswith("openai/") else f"openai/{selected_model}"
    if selected_model.startswith("bedrock/") and not selected_model.startswith("bedrock/converse/"):
        return selected_model.replace("bedrock/", "bedrock/converse/", 1)
    if "/" in selected_model and not selected_model.startswith(
        ("together_ai/", "openai/", "bedrock/", "anthropic/", "gemini/", "vertex_ai/")
    ):
        return f"together_ai/{selected_model}"
    return selected_model


def _resolve_curator_bridge_classifier_config(args) -> Dict[str, str]:
    model_name = getattr(args, "worker_model", None) or getattr(args, "model", None)
    selected_model = str(model_name or "").strip()
    if not selected_model:
        raise ValueError("No model configured for question pattern classification")
    if getattr(args, "base_url", None):
        return {
            "backend": "litellm",
            "model_name": _normalize_curator_litellm_model_name(args, selected_model),
        }
    if selected_model.startswith("gpt-"):
        return {
            "backend": "openai",
            "model_name": selected_model,
        }
    return {
        "backend": "litellm",
        "model_name": _normalize_curator_litellm_model_name(args, selected_model),
    }


@contextmanager
def _suppress_curator_progress():
    original_value = os.environ.get("CURATOR_DISABLE_PROGRESS")
    os.environ["CURATOR_DISABLE_PROGRESS"] = "1"
    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop("CURATOR_DISABLE_PROGRESS", None)
        else:
            os.environ["CURATOR_DISABLE_PROGRESS"] = original_value


def _classify_questions_with_curator(
    args,
    questions: List[str],
    *,
    working_dir: str,
) -> Dict[str, Dict[str, Any]]:
    if not CURATOR_AVAILABLE:
        raise RuntimeError("Curator is not available")
    classifier_config = _resolve_curator_bridge_classifier_config(args)

    operator = _BridgeSurfacePatternOperator(
        model_name=classifier_config["model_name"],
        backend=classifier_config["backend"],
        generation_params={"temperature": 0.0, "max_tokens": 500},
        backend_params=_curator_backend_params(args),
    )
    with _suppress_curator_progress():
        dataset = operator(
            [{"question": str(question or "")} for question in questions],
            working_dir=working_dir,
        )
    rows = _normalize_curator_dataset_rows(dataset)
    return {
        str(row.get("question", "") or ""): row.get("pattern", {})
        for row in rows
        if str(row.get("question", "") or "").strip()
    }


def _build_question_pattern_classifier(
    entity_searcher: EntitySearcher,
    args,
    persistent_cache_path: Optional[str] = None,
    curator_working_dir: Optional[str] = None,
) -> QuestionPatternClassifier:
    model_name = getattr(args, "worker_model", None) or getattr(args, "model", None)
    if not model_name:
        return QuestionPatternClassifier(persistent_cache_path=persistent_cache_path)
    try:
        classifier_agent = _build_answer_reconciler(entity_searcher, args)
    except Exception:
        classifier_agent = None
    return QuestionPatternClassifier(
        classify_question_fn=(
            (lambda question: _classify_question_with_model(classifier_agent, args, question))
            if classifier_agent is not None
            else None
        ),
        persistent_cache_path=persistent_cache_path,
    )


def _build_bridge_surface_pattern_classifier(
    entity_searcher: EntitySearcher,
    args,
    persistent_cache_path: Optional[str] = None,
    curator_working_dir: Optional[str] = None,
) -> QuestionPatternClassifier:
    batch_classify_fn: Optional[Callable[[List[str]], Dict[str, Dict[str, Any]]]] = None
    single_classify_fn: Optional[Callable[[str], Dict[str, Any]]] = None
    classifier_backend: Optional[str] = None
    classifier_model_name: Optional[str] = None
    model_name = getattr(args, "worker_model", None) or getattr(args, "model", None)
    if str(model_name or "").strip():
        try:
            classifier_agent = _build_answer_reconciler(entity_searcher, args)
        except Exception:
            classifier_agent = None
        else:
            single_classify_fn = lambda question: _classify_question_with_model(classifier_agent, args, question)
    if CURATOR_AVAILABLE and curator_working_dir and str(model_name or "").strip():
        classifier_config = _resolve_curator_bridge_classifier_config(args)
        classifier_backend = classifier_config["backend"]
        classifier_model_name = classifier_config["model_name"]
        batch_classify_fn = lambda questions: _classify_questions_with_curator(
            args,
            list(questions),
            working_dir=curator_working_dir,
        )
    return QuestionPatternClassifier(
        classify_question_fn=single_classify_fn,
        classify_questions_fn=batch_classify_fn,
        persistent_cache_path=persistent_cache_path,
        classifier_backend=classifier_backend,
        classifier_model_name=classifier_model_name,
    )


def _parse_model_answer(answer_agent: AgenticReconciler, question: str, evidence: List[str]) -> Dict[str, str]:
    if not evidence:
        return {"predicted_answer": "", "full_response": "No evidence found to answer the question"}
    messages = build_panini_answer_messages(question, evidence)
    message = answer_agent._call_model_for_stage(
        messages,
        model_name=answer_agent.model_name,
        max_tokens=384,
        temperature=0.0,
        stage="root",
    )
    raw_text, _source = answer_agent._extract_message_content_with_source(message)
    return {
        "predicted_answer": parse_answer_from_response(raw_text),
        "full_response": raw_text,
    }


def answer_query_batch(
    batch_index: int,
    batch_items: List[BatchQueryItem],
    entity_searcher: EntitySearcher,
    bridge_registry: BridgeRegistry,
    args,
    pattern_classifier: Optional[QuestionPatternClassifier] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> QueryBatchResult:
    embed_texts_fn = _build_embed_texts_fn(entity_searcher)
    answer_agent = _build_answer_reconciler(entity_searcher, args)
    fallback_pattern_classifier = pattern_classifier or QuestionPatternClassifier()
    query_results: List[QueryAnswerResult] = []
    total_queries = len(batch_items)

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "query_answering",
                "status": "started",
                "completed": 0,
                "total": total_queries,
                "unit": "queries",
            }
        )

    for query_index, item in enumerate(batch_items, start=1):
        bridge_hits = bridge_registry.search(
            item.question,
            top_k=args.bridge_query_top_k,
            embed_texts_fn=embed_texts_fn,
        )
        bridge_hits = [
            {
                **row,
                "rank_before_merge": index + 1,
            }
            for index, row in enumerate(bridge_hits)
        ]
        doc_hits = collect_doc_evidence(
            entity_searcher,
            item.question,
            top_k_entities=args.bridge_query_top_k,
            top_k_direct_qa=args.bridge_query_top_k,
        )
        doc_hits = [
            {
                **row,
                "rank_before_merge": index + 1,
            }
            for index, row in enumerate(doc_hits)
        ]
        rerank_result = rerank_merged_evidence(
            item.question,
            bridge_hits + doc_hits,
            embed_texts_fn=embed_texts_fn,
            top_k=args.bridge_query_top_k,
        )
        merged_hits = rerank_result.kept_items
        merged_candidates = rerank_result.scored_items
        normalized_query = normalize_similarity_text(str(item.question or ""))
        retrieved_bridge_hit_count = sum(1 for row in bridge_hits if row.get("source_type") == "bridge")
        kept_bridge_hit_count = sum(1 for row in merged_hits if row.get("source_type") == "bridge")
        exact_bridge_match_found = any(
            str(row.get("source_type", "")) == "bridge"
            and normalize_similarity_text(str(row.get("question", "") or "")) == normalized_query
            for row in bridge_hits
        )
        exact_bridge_match_kept = any(
            str(row.get("source_type", "")) == "bridge"
            and normalize_similarity_text(str(row.get("question", "") or "")) == normalized_query
            for row in merged_hits
        )
        equivalent_bridge_candidates = [
            row for row in merged_candidates if bool(row.get("is_equivalent_bridge_match", False))
        ]
        equivalent_bridge_match_found = bool(equivalent_bridge_candidates)
        equivalent_bridge_match_kept = any(
            bool(row.get("is_equivalent_bridge_match", False)) for row in merged_hits
        )
        equivalent_bridge_match_similarity = None
        equivalent_bridge_match_bridge_id = None
        if equivalent_bridge_candidates:
            best_equivalent_bridge = equivalent_bridge_candidates[0]
            equivalent_bridge_match_similarity = float(best_equivalent_bridge.get("embedding_similarity", 0.0) or 0.0)
            equivalent_bridge_match_bridge_id = str(best_equivalent_bridge.get("bridge_id", "") or "") or None
        answer_payload = _parse_model_answer(
            answer_agent,
            item.question,
            [row.get("evidence_text", "") for row in merged_hits],
        )

        gold_answers = list(item.answer_aliases)
        if not gold_answers and item.gold_answer:
            gold_answers = [item.gold_answer]
        exact_match, f1 = score_predicted_answer(gold_answers, answer_payload["predicted_answer"])

        source_mix: Dict[str, int] = {}
        for row in merged_hits:
            source_type = str(row.get("source_type", "unknown"))
            source_mix[source_type] = source_mix.get(source_type, 0) + 1

        bridge_evidence_used = bool(source_mix.get("bridge", 0))
        used_bridge_ids = {
            str(row.get("bridge_id", "")).strip()
            for row in merged_hits
            if row.get("source_type") == "bridge" and str(row.get("bridge_id", "")).strip()
        }
        helpful_bridge_ids = used_bridge_ids if bridge_evidence_used and (f1 or 0.0) >= 0.5 else set()
        bridge_registry.record_query_usage(used_bridge_ids, helpful_bridge_ids, batch_index)

        top_evidence = [
            {
                "source_type": row.get("source_type"),
                "final_score": row.get("final_score", row.get("score", 0.0)),
                "bridge_id": row.get("bridge_id"),
                "doc_id": row.get("doc_id"),
                "question": row.get("question"),
                "evidence_text": row.get("evidence_text", ""),
            }
            for row in merged_hits[: min(5, len(merged_hits))]
        ]
        retrieval_trace = {
            "raw_bridge_hits": [
                _serialize_retrieval_trace_row(row)
                for row in sorted(
                    (row for row in merged_candidates if row.get("source_type") == "bridge"),
                    key=lambda row: int(row.get("rank_before_merge", 0) or 0),
                )
            ],
            "raw_doc_hits": [
                _serialize_retrieval_trace_row(row)
                for row in sorted(
                    (row for row in merged_candidates if row.get("source_type") != "bridge"),
                    key=lambda row: int(row.get("rank_before_merge", 0) or 0),
                )
            ],
            "merged_candidates": [_serialize_retrieval_trace_row(row) for row in merged_candidates],
            "kept_candidates": [_serialize_retrieval_trace_row(row) for row in merged_hits],
        }
        question_pattern = fallback_pattern_classifier.classify(item.question)

        query_results.append(
            QueryAnswerResult(
                question_id=item.question_id,
                question=item.question,
                predicted_answer=answer_payload["predicted_answer"],
                full_response=answer_payload["full_response"],
                exact_match=exact_match,
                f1=f1,
                bridge_evidence_used=bridge_evidence_used,
                answer_source_mix=source_mix,
                top_evidence=top_evidence,
                patterns=[question_pattern.pattern_key],
                question_pattern=question_pattern.as_dict(),
                gold_answers=gold_answers,
                retrieved_bridge_hit_count=retrieved_bridge_hit_count,
                kept_bridge_hit_count=kept_bridge_hit_count,
                exact_bridge_match_found=exact_bridge_match_found,
                exact_bridge_match_kept=exact_bridge_match_kept,
                equivalent_bridge_match_found=equivalent_bridge_match_found,
                equivalent_bridge_match_kept=equivalent_bridge_match_kept,
                equivalent_bridge_match_similarity=equivalent_bridge_match_similarity,
                equivalent_bridge_match_bridge_id=equivalent_bridge_match_bridge_id,
                retrieval_trace=retrieval_trace,
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "query_answering",
                    "status": "progress",
                    "completed": query_index,
                    "total": total_queries,
                    "unit": "queries",
                    "question_id": item.question_id,
                    "question": _truncate_for_console(item.question, limit=96),
                    "bridge_evidence_used": bridge_evidence_used,
                    "predicted_answer": _truncate_for_console(answer_payload["predicted_answer"] or "", limit=64),
                    "f1": f1,
                    "exact_match": exact_match,
                }
            )

    scored = [result for result in query_results if result.f1 is not None]
    if scored:
        overall_metrics = {
            "ExactMatch": round(sum(float(result.exact_match or 0.0) for result in scored) / len(scored), 4),
            "F1": round(sum(float(result.f1 or 0.0) for result in scored) / len(scored), 4),
            "scored_queries": len(scored),
        }
    else:
        overall_metrics = {"ExactMatch": 0.0, "F1": 0.0, "scored_queries": 0}

    feedback_brief = build_batch_feedback_brief(query_results)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "query_answering",
                "status": "completed",
                "completed": total_queries,
                "total": total_queries,
                "unit": "queries",
            }
        )
    return QueryBatchResult(
        batch_index=batch_index,
        query_results=query_results,
        overall_metrics=overall_metrics,
        feedback_brief=feedback_brief,
        )


def _write_curriculum_guidance_artifact(
    batch_output_dir: str,
    batch_index: int,
    guidance_payload: Optional[Dict[str, Any]],
) -> Optional[str]:
    if guidance_payload is None:
        return None
    guidance_path = os.path.join(batch_output_dir, f"curriculum_guidance_batch_{batch_index}.json")
    with open(guidance_path, "w") as f:
        json.dump(guidance_payload, f, indent=2)
    return guidance_path


def _write_generation_shard_metadata(
    shard_output_dir: str,
    batch_index: int,
    shard: Dict[str, Any],
) -> str:
    os.makedirs(shard_output_dir, exist_ok=True)
    payload = {
        "batch_index": int(batch_index),
        "shard_id": int(shard["shard_id"]),
        "question_ids": list(shard.get("question_ids", [])),
        "questions": list(shard.get("questions", [])),
        "doc_ids": list(shard.get("doc_ids", [])),
    }
    artifact_path = os.path.join(shard_output_dir, "shard_metadata.json")
    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2)
    return artifact_path


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


def _merge_bridge_dicts(bridges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for bridge in bridges:
        bridge_id = str(bridge.get("bridge_id", "")).strip()
        if not bridge_id:
            continue
        candidate = dict(bridge)
        candidate_docs = {
            str(doc_id)
            for doc_id in candidate.get("source_docs", []) or []
            if str(doc_id).strip()
        }
        if bridge_id not in merged_by_id:
            candidate["source_docs"] = sorted(candidate_docs)
            merged_by_id[bridge_id] = candidate
            continue
        existing = merged_by_id[bridge_id]
        existing_docs = {
            str(doc_id)
            for doc_id in existing.get("source_docs", []) or []
            if str(doc_id).strip()
        }
        merged_docs = sorted(existing_docs | candidate_docs)
        existing_confidence = float(existing.get("confidence", 0.0) or 0.0)
        candidate_confidence = float(candidate.get("confidence", 0.0) or 0.0)
        if candidate_confidence > existing_confidence:
            candidate["source_docs"] = merged_docs
            merged_by_id[bridge_id] = candidate
        else:
            existing["source_docs"] = merged_docs
    return [merged_by_id[key] for key in sorted(merged_by_id.keys())]


def _run_generation_shard_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.perf_counter()
    run_doc_batch_kwargs = {
        "batch_index": int(task["batch_index"]),
        "doc_ids": list(task["doc_ids"]),
        "args": task["args"],
        "output_dir": str(task["output_dir"]),
        "guidance_payload": task.get("guidance_payload"),
        "batch_output_dir": str(task["batch_output_dir"]),
        "cache_root_dir": str(task.get("cache_root_dir") or task["output_dir"]),
    }
    supported_params = set(inspect.signature(run_doc_batch).parameters)
    result = run_doc_batch(
        **{key: value for key, value in run_doc_batch_kwargs.items() if key in supported_params}
    )
    runtime_seconds = round(time.perf_counter() - started_at, 3)
    return {
        **result,
        "shard_id": int(task["shard_id"]),
        "question_ids": list(task.get("question_ids", [])),
        "questions": list(task.get("questions", [])),
        "runtime_seconds": runtime_seconds,
        "batch_output_dir": str(task["batch_output_dir"]),
    }


def _execute_generation_shard_tasks(
    tasks: Sequence[Dict[str, Any]],
    max_workers: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    total_tasks = len(tasks)
    if len(tasks) == 1 or max_workers <= 1:
        results: List[Dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            result = _run_generation_shard_worker(dict(task))
            results.append(result)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "bridge_generation_shards",
                        "status": "progress",
                        "completed": index,
                        "total": total_tasks,
                        "unit": "shards",
                        "shard_id": result["shard_id"],
                        "question_ids": list(result.get("question_ids", [])),
                        "accepted_bridges": len(result.get("bridges", [])),
                        "runtime_seconds": result.get("runtime_seconds"),
                    }
                )
        return results

    results_by_shard: Dict[int, Dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
        future_to_task = {
            executor.submit(_run_generation_shard_worker, dict(task)): dict(task)
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results_by_shard[int(task["shard_id"])] = result
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "bridge_generation_shards",
                        "status": "progress",
                        "completed": completed,
                        "total": total_tasks,
                        "unit": "shards",
                        "shard_id": result["shard_id"],
                        "question_ids": list(result.get("question_ids", [])),
                        "accepted_bridges": len(result.get("bridges", [])),
                        "runtime_seconds": result.get("runtime_seconds"),
                    }
                )
    return [results_by_shard[int(task["shard_id"])] for task in tasks]


def run_doc_batch(
    batch_index: int,
    doc_ids: List[str],
    args,
    output_dir: str,
    guidance_payload: Optional[Dict[str, Any]] = None,
    batch_output_dir: Optional[str] = None,
    cache_root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    temp_base = tempfile.mkdtemp(prefix=f"bridge_test_batch{batch_index}_")
    try:
        temp_gsw_dir = setup_temp_gsw_dir(args.gsw_path, doc_ids, temp_base, subdir="networks")
        batch_output_dir = batch_output_dir or os.path.join(output_dir, f"batch_{batch_index}")
        os.makedirs(batch_output_dir, exist_ok=True)

        _write_curriculum_guidance_artifact(batch_output_dir, batch_index, guidance_payload)

        cache_dir = _curriculum_cache_dir(cache_root_dir or output_dir, "bridge_batches", doc_ids)
        batch_args = _build_sleep_time_args(
            args,
            gsw_path=temp_gsw_dir,
            num_docs=len(doc_ids),
            output_dir=batch_output_dir,
            cache_dir=cache_dir,
        )

        console.print(
            "  Batch bridge creation: "
            f"docs={len(doc_ids)}, depth={args.edge_max_depth}, calls={args.edge_max_calls}, "
            f"edge_tokens={args.edge_max_tokens}, parallel={'on' if args.edge_parallel_enabled else 'off'}:{args.edge_parallel_workers}"
        )
        console.print(f"  [dim]Bridge cache: {cache_dir}[/dim]")

        batch_searcher = EntitySearcher(
            num_documents=len(doc_ids),
            path_to_gsw_files=temp_gsw_dir,
            cache_dir=cache_dir,
            rebuild_cache=False,
            verbose=False,
            use_bm25=True,
            use_gpu_for_qa_index=False,
            embedding_gpu_memory_utilization=getattr(args, "embedding_gpu_memory_utilization", 0.5),
        )
        console.print(
            "  [dim]Retrieval embedder: "
            f"{getattr(batch_searcher, 'embedding_model_source', 'unknown')}"
            f"{' (shared)' if bool(getattr(batch_searcher, 'embedding_model_reused', False)) else ''}"
            "[/dim]"
        )
        results_json = run_sleep_time_once(
            batch_args,
            entity_searcher=batch_searcher,
            curriculum_guidance=guidance_payload,
        )

        if not results_json:
            raise RuntimeError(f"No results.json produced for batch {batch_index}")

        bridges = results_json.get("bridges", [])
        console.print(f"  [green]Batch {batch_index} produced {len(bridges)} accepted bridges[/green]")
        return {
            "batch_index": batch_index,
            "doc_ids": list(doc_ids),
            "bridges": bridges,
            "run_results": {
                **results_json.get("summary", {}),
                "retrieval_embedding_model_source": getattr(batch_searcher, "embedding_model_source", None),
                "retrieval_embedding_model_reused": bool(getattr(batch_searcher, "embedding_model_reused", False)),
            },
            "results_json": results_json,
            "retrieval_embedding_model_source": getattr(batch_searcher, "embedding_model_source", None),
            "retrieval_embedding_model_reused": bool(getattr(batch_searcher, "embedding_model_reused", False)),
            "generation_executor": "single",
        }
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)


def _run_doc_batch_sharded(
    batch_index: int,
    batch_items: Sequence[BatchQueryItem],
    args,
    batch_output_dir: str,
    output_dir: str,
    guidance_payload: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    shard_dir_root = Path(batch_output_dir) / "generation_shards"
    shard_dir_root.mkdir(parents=True, exist_ok=True)
    shards = _plan_generation_shards(batch_items)
    total_shards = len(shards)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "bridge_generation_shards",
                "status": "started",
                "completed": 0,
                "total": total_shards,
                "unit": "shards",
            }
        )

    _write_curriculum_guidance_artifact(batch_output_dir, batch_index, guidance_payload)
    worker_count = max(1, int(getattr(args, "curriculum_generation_parallel_workers", 1) or 1))
    for shard in shards:
        _write_generation_shard_metadata(
            shard_output_dir=str(shard_dir_root / f"shard_{shard['shard_id']}"),
            batch_index=batch_index,
            shard=shard,
        )
    tasks = [
        {
            "batch_index": batch_index,
            "shard_id": shard["shard_id"],
            "question_ids": list(shard["question_ids"]),
            "questions": list(shard["questions"]),
            "doc_ids": list(shard["doc_ids"]),
            "args": args,
            "output_dir": str(output_dir),
            "batch_output_dir": str(shard_dir_root / f"shard_{shard['shard_id']}"),
            "cache_root_dir": str(output_dir),
            "guidance_payload": guidance_payload,
        }
        for shard in shards
    ]

    shard_results = _execute_generation_shard_tasks(
        tasks=tasks,
        max_workers=worker_count,
        progress_callback=progress_callback,
    )

    merged_bridges = _merge_bridge_dicts(
        [bridge for shard_result in shard_results for bridge in shard_result.get("bridges", [])]
    )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "bridge_generation_shards",
                "status": "completed",
                "completed": total_shards,
                "total": total_shards,
                "unit": "shards",
                "merged_bridge_count": len(merged_bridges),
            }
        )

    shard_metadata = [
        {
            "shard_id": result["shard_id"],
            "question_ids": list(result.get("question_ids", [])),
            "questions": list(result.get("questions", [])),
            "doc_ids": list(result.get("doc_ids", [])),
            "runtime_seconds": float(result.get("runtime_seconds", 0.0) or 0.0),
            "accepted_bridge_count": len(result.get("bridges", [])),
            "run_results": dict(result.get("run_results", {})),
            "retrieval_embedding_model_source": result.get("retrieval_embedding_model_source"),
            "retrieval_embedding_model_reused": bool(result.get("retrieval_embedding_model_reused", False)),
            "output_dir": str(result.get("batch_output_dir", "")),
        }
        for result in shard_results
    ]
    console.print(
        f"  [green]Batch {batch_index} produced {len(merged_bridges)} accepted bridges after merging {len(shard_metadata)} shard(s)[/green]"
    )
    return {
        "batch_index": batch_index,
        "doc_ids": sorted({doc_id for item in batch_items for doc_id in item.doc_ids}),
        "bridges": merged_bridges,
        "run_results": {
            "total_bridges": len(merged_bridges),
            "shard_count": len(shard_metadata),
        },
        "generation_mode": "question_group_sharded",
        "generation_executor": "thread",
        "generation_shards": shard_metadata,
    }


def run_question(
    q_idx: int,
    question: dict,
    args,
    schema: str,
    title_to_idx: dict[str, int] | None = None,
) -> dict:
    meta = normalize_question_metadata(question, q_idx, schema=schema, title_to_idx=title_to_idx)
    global_indices = meta["global_indices"]
    support_global = meta["supporting_global"]
    num_docs = len(global_indices)

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold]Question {q_idx}: {meta['question_text']}[/bold]")
    console.print(f"  Answer: {meta['answer']}")
    console.print(f"  Supporting docs (global): {support_global}")
    console.print(f"  Hops: {meta['num_hops']}")
    console.print(f"  Context docs used: {num_docs}")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]")

    temp_base = tempfile.mkdtemp(prefix=f"bridge_test_q{q_idx}_")

    try:
        temp_gsw_dir = setup_temp_gsw_dir(args.gsw_path, global_indices, temp_base)
        q_output_dir = os.path.join(args.output_dir, f"q_{q_idx}")

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "run_sleep_time.py"),
            "--gsw_path", temp_gsw_dir,
            "--num_docs", str(num_docs),
            "--model", args.model,
            "--max_iterations", str(args.max_iterations),
            "--output_dir", q_output_dir,
            "--max_tokens", str(args.max_tokens),
            "--reasoning_effort", args.reasoning_effort,
            "--pipeline_mode", args.pipeline_mode,
            "--hybrid_scope", args.hybrid_scope,
        ]

        if args.base_url:
            cmd.extend(["--base_url", args.base_url])
        if args.root_model:
            cmd.extend(["--root_model", args.root_model])
        if args.worker_model:
            cmd.extend(["--worker_model", args.worker_model])

        cmd.extend(
            [
                "--edge_max_depth", str(args.edge_max_depth),
                "--edge_max_calls", str(args.edge_max_calls),
                "--edge_max_tokens", str(args.edge_max_tokens),
                "--max_optional_docs_per_edge", str(args.max_optional_docs_per_edge),
                "--edge_parallel_workers", str(args.edge_parallel_workers),
                "--parallel_progress_heartbeat_seconds", str(args.parallel_progress_heartbeat_seconds),
                "--parallel_stuck_warning_seconds", str(args.parallel_stuck_warning_seconds),
                "--verifier_parallel_workers", str(args.verifier_parallel_workers),
                "--edge_max_accepted_bridges", str(args.edge_max_accepted_bridges),
            ]
        )
        if args.edge_parallel_enabled:
            cmd.append("--edge_parallel_enabled")
        if args.verbose or args.show_thinking:
            cmd.append("--verbose")
        if args.show_thinking:
            cmd.append("--show-thinking")

        cache_dir = os.path.join(temp_base, ".gsw_cache")
        cmd.extend(["--cache_dir", cache_dir])

        console.print(
            "  Pipeline knobs: "
            f"depth={args.edge_max_depth}, calls={args.edge_max_calls}, "
            f"edge_tokens={args.edge_max_tokens}, optional_docs={args.max_optional_docs_per_edge}, "
            f"parallel={'on' if args.edge_parallel_enabled else 'off'}:{args.edge_parallel_workers}, "
            f"hb={args.parallel_progress_heartbeat_seconds}s, "
            f"stuck_warn={args.parallel_stuck_warning_seconds}s, "
            f"verifier_workers={args.verifier_parallel_workers}, edge_accept_cap={args.edge_max_accepted_bridges}"
        )
        console.print(f"  Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=not (args.verbose or args.show_thinking), text=True, cwd=str(REPO_ROOT))
        if result.returncode != 0 and not args.verbose:
            console.print(f"  [red]Error (exit {result.returncode}):[/red]")
            console.print(f"  {result.stderr[-500:] if result.stderr else 'No stderr'}")

        results_json = None
        q_output_path = Path(q_output_dir)
        if q_output_path.exists():
            run_dirs = sorted(q_output_path.glob("run_*"))
            if run_dirs:
                results_file = run_dirs[-1] / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results_json = json.load(f)

        bridges = results_json.get("bridges", []) if results_json else []
        console.print(f"  [green]Generated {len(bridges)} bridges[/green]")

        return {
            "question_idx": q_idx,
            "question_id": meta["question_id"],
            "question": meta["question_text"],
            "answer": meta["answer"],
            "answer_aliases": meta["answer_aliases"],
            "num_hops": meta["num_hops"],
            "decomposition": meta["decomposition"],
            "supporting_doc_indices_local": meta["supporting_local"],
            "supporting_doc_indices_global": meta["supporting_global"],
            "supporting_doc_titles": meta["supporting_titles"],
            "all_doc_indices_global": global_indices,
            "bridges": bridges,
            "num_bridges": len(bridges),
            "run_results": results_json.get("summary", {}) if results_json else {},
        }
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)


def run_curriculum(
    items: List[BatchQueryItem],
    args,
) -> Dict[str, Any]:
    # Separate probe items from train items
    probe_eval_enabled = bool(getattr(args, "probe_eval", False))
    probe_items: List[BatchQueryItem] = []
    train_items: List[BatchQueryItem] = []
    for item in items:
        if item.metadata.get("split") == "probe":
            probe_items.append(item)
        else:
            train_items.append(item)

    if probe_eval_enabled and probe_items:
        console.print(f"[bold green]Probe evaluation enabled: {len(probe_items)} probe questions[/bold green]")
    elif probe_eval_enabled:
        console.print("[yellow]Probe eval enabled but no probe items found in manifest[/yellow]")

    batches = split_curriculum_batches(
        train_items,
        batch_size=args.curriculum_batch_size,
        seed_batch_size=args.curriculum_seed_batch_size,
    )
    bridge_registry = BridgeRegistry()
    accumulated_doc_ids: set[str] = set()
    query_experience = QueryExperienceState()
    batch_results: List[Dict[str, Any]] = []
    flat_query_results: List[Dict[str, Any]] = []
    learning_curve: List[Dict[str, Any]] = []

    for batch_index, batch_items in enumerate(batches):
        console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
        console.print(f"[bold]Curriculum batch {batch_index}[/bold]")
        console.print(f"  Questions: {len(batch_items)}")
        console.print(f"[bold cyan]{'='*70}[/bold cyan]")

        batch_output_dir = os.path.join(args.output_dir, f"batch_{batch_index}")
        os.makedirs(batch_output_dir, exist_ok=True)
        current_doc_ids = sorted({doc_id for item in batch_items for doc_id in item.doc_ids})
        guidance_enabled = not bool(getattr(args, "curriculum_disable_guidance", False))
        bridge_exemplars: List[Dict[str, Any]] = []
        guidance_payload: Optional[Dict[str, Any]] = None
        if guidance_enabled:
            prior_query_demand_sketch = build_prior_query_demand_sketch(query_experience)
            prior_feedback_brief = query_experience.last_batch_feedback_brief
            bridge_exemplars = bridge_registry.select_exemplars(
                prior_query_demand_sketch=prior_query_demand_sketch,
                prior_feedback_brief=prior_feedback_brief,
                top_k=args.bridge_prompt_exemplar_top_k,
            )
            demand_profile = build_query_demand_profile(query_experience)
            rel_to_patterns = build_relationship_to_patterns(bridge_registry)
            guidance_payload = build_curriculum_guidance_payload(
                prior_query_demand_sketch=prior_query_demand_sketch,
                prior_batch_feedback_brief=prior_feedback_brief,
                bridge_exemplars=bridge_exemplars,
                prior_batches_seen=query_experience.batches_seen,
                query_demand_profile=demand_profile,
                relationship_to_patterns=rel_to_patterns,
            )

        with CurriculumProgressTracker(batch_index=batch_index, batch_output_dir=batch_output_dir) as progress_tracker:
            use_sharded_generation = bool(getattr(args, "curriculum_generation_parallel_enabled", False)) and len(batch_items) > 1
            if use_sharded_generation:
                batch_bridge_result = _run_doc_batch_sharded(
                    batch_index=batch_index,
                    batch_items=batch_items,
                    args=args,
                    batch_output_dir=batch_output_dir,
                    output_dir=args.output_dir,
                    guidance_payload=guidance_payload,
                    progress_callback=progress_tracker.callback,
                )
                batch_bridge_result.setdefault("generation_executor", "thread")
            else:
                run_doc_batch_kwargs = {
                    "batch_index": batch_index,
                    "doc_ids": current_doc_ids,
                    "args": args,
                    "output_dir": args.output_dir,
                    "guidance_payload": guidance_payload,
                    "batch_output_dir": batch_output_dir,
                    "cache_root_dir": args.output_dir,
                }
                supported_doc_batch_params = set(inspect.signature(run_doc_batch).parameters)
                batch_bridge_result = run_doc_batch(
                    **{
                        key: value
                        for key, value in run_doc_batch_kwargs.items()
                        if key in supported_doc_batch_params
                    }
                )
                batch_bridge_result.setdefault("generation_mode", "single_batch")
                batch_bridge_result.setdefault("generation_executor", "single")
                batch_bridge_result.setdefault("generation_shards", [])

            accumulated_doc_ids.update(current_doc_ids)
            temp_base = tempfile.mkdtemp(prefix=f"bridge_curriculum_answer_{batch_index}_")
            try:
                accumulated_dir = setup_temp_gsw_dir(args.gsw_path, sorted(accumulated_doc_ids), temp_base, subdir="accumulated_networks")
                answer_cache_dir = _curriculum_cache_dir(args.output_dir, "answer_accumulated", sorted(accumulated_doc_ids))
                console.print(f"  [dim]Answer cache: {answer_cache_dir}[/dim]")
                answer_searcher = EntitySearcher(
                    num_documents=-1,
                    path_to_gsw_files=accumulated_dir,
                    cache_dir=answer_cache_dir,
                    rebuild_cache=False,
                    verbose=False,
                    use_bm25=True,
                    use_gpu_for_qa_index=False,
                    embedding_gpu_memory_utilization=getattr(args, "embedding_gpu_memory_utilization", 0.5),
                )
                embed_texts_fn = _build_embed_texts_fn(answer_searcher)
                model_name = getattr(args, "worker_model", None) or getattr(args, "model", None) or "unknown"
                pattern_cache_paths = _build_pattern_classifier_cache_paths(args.output_dir, model_name)
                query_classifier_kwargs = {
                    "entity_searcher": answer_searcher,
                    "args": args,
                    "persistent_cache_path": pattern_cache_paths["query_pattern_cache_path"],
                }
                supported_query_classifier_params = set(inspect.signature(_build_question_pattern_classifier).parameters)
                query_pattern_classifier = _build_question_pattern_classifier(
                    **{
                        key: value
                        for key, value in query_classifier_kwargs.items()
                        if key in supported_query_classifier_params
                    }
                )
                bridge_classifier_kwargs = {
                    "entity_searcher": answer_searcher,
                    "args": args,
                    "persistent_cache_path": pattern_cache_paths["bridge_surface_cache_path"],
                    "curator_working_dir": pattern_cache_paths["curator_working_dir"],
                }
                supported_bridge_classifier_params = set(
                    inspect.signature(_build_bridge_surface_pattern_classifier).parameters
                )
                bridge_pattern_classifier = _build_bridge_surface_pattern_classifier(
                    **{
                        key: value
                        for key, value in bridge_classifier_kwargs.items()
                        if key in supported_bridge_classifier_params
                    }
                )
                ingest_result = bridge_registry.ingest_bridges(
                    batch_bridge_result["bridges"],
                    batch_index=batch_index,
                    embed_texts_fn=embed_texts_fn,
                    pattern_classifier=bridge_pattern_classifier,
                    progress_callback=progress_tracker.callback,
                    strict_pattern_tagging=True,
                )
                pattern_report_path = _write_bridge_pattern_classification_report(
                    batch_output_dir=batch_output_dir,
                    batch_index=batch_index,
                    ingest_result=ingest_result,
                )
                console.print(f"  [dim]Bridge pattern report: {pattern_report_path}[/dim]")
                if int(getattr(ingest_result, "failed_surface_count", 0) or 0) > 0:
                    raise RuntimeError(
                        "Bridge surface LLM tagging failed for uncached surfaces; "
                        f"see {pattern_report_path}"
                    )
                snapshot_path = _write_bridge_registry_snapshot(
                    batch_output_dir=batch_output_dir,
                    batch_index=batch_index,
                    bridge_registry=bridge_registry,
                    ingest_result=ingest_result,
                )
                console.print(f"  [dim]Bridge registry snapshot: {snapshot_path}[/dim]")
                answer_query_kwargs = {
                    "batch_index": batch_index,
                    "batch_items": batch_items,
                    "entity_searcher": answer_searcher,
                    "bridge_registry": bridge_registry,
                    "args": args,
                    "pattern_classifier": query_pattern_classifier,
                    "progress_callback": progress_tracker.callback,
                }
                supported_params = set(inspect.signature(answer_query_batch).parameters)
                query_batch_result = answer_query_batch(
                    **{key: value for key, value in answer_query_kwargs.items() if key in supported_params}
                )
                query_artifact_path = _write_query_batch_artifact(
                    batch_output_dir=batch_output_dir,
                    batch_index=batch_index,
                    batch_doc_ids=current_doc_ids,
                    accumulated_doc_ids=sorted(accumulated_doc_ids),
                    answer_cache_dir=answer_cache_dir,
                    query_batch_result=query_batch_result,
                )
                console.print(f"  [dim]Query answers: {query_artifact_path}[/dim]")
                _print_query_batch_summary(batch_index, query_batch_result)
            finally:
                shutil.rmtree(temp_base, ignore_errors=True)

        post_batch_demand_sketch = build_current_batch_demand_sketch(query_batch_result.query_results)
        accumulate_query_experience_state(
            query_experience,
            batch_items=batch_items,
            batch_result=query_batch_result,
        )

        # Apply demand decay if configured
        decay_rate = float(getattr(args, "guidance_decay_rate", 0.0))
        if decay_rate > 0.0:
            apply_demand_decay(query_experience, decay_rate=decay_rate, current_batch=batch_index)

        # Probe evaluation (isolated sandbox)
        bridges_this_batch = len(batch_bridge_result.get("bridges", []))
        probe_eval_record: Optional[Dict[str, Any]] = None

        def _write_probe_progress(stage: str, detail: str = "", **extra: Any) -> None:
            """Write probe progress file for experiment runner to poll."""
            probe_progress_path = os.path.join(args.output_dir, "probe_progress.json")
            progress = {"batch_index": batch_index, "stage": stage, "detail": detail, "timestamp": datetime.now().isoformat(), **extra}
            try:
                with open(probe_progress_path, "w") as pf:
                    json.dump(progress, pf)
            except Exception:
                pass

        if probe_eval_enabled and probe_items:
            console.print(f"\n  [bold magenta]Running probe evaluation ({len(probe_items)} questions)...[/bold magenta]")
            _write_probe_progress("generating", detail=f"generating bridges for {len(probe_items)} probe questions")
            probe_doc_ids = sorted({doc_id for item in probe_items for doc_id in item.doc_ids})
            probe_temp_base = tempfile.mkdtemp(prefix=f"probe_eval_batch_{batch_index}_")
            try:
                # Generate bridges for probe docs using current guidance
                probe_bridge_result = _run_doc_batch_sharded(
                    batch_index=batch_index,
                    batch_items=probe_items,
                    args=args,
                    batch_output_dir=os.path.join(args.output_dir, f"probe_batch_{batch_index}"),
                    output_dir=args.output_dir,
                    guidance_payload=guidance_payload,
                    progress_callback=None,
                ) if bool(getattr(args, "curriculum_generation_parallel_enabled", False)) else run_doc_batch(
                    batch_index=batch_index,
                    doc_ids=probe_doc_ids,
                    args=args,
                    output_dir=args.output_dir,
                    guidance_payload=guidance_payload,
                    batch_output_dir=os.path.join(args.output_dir, f"probe_batch_{batch_index}"),
                )

                # Build temporary probe registry (isolated)
                probe_registry = BridgeRegistry()
                probe_gsw_dir = setup_temp_gsw_dir(args.gsw_path, probe_doc_ids, probe_temp_base, subdir="probe_networks")
                probe_cache_dir = _curriculum_cache_dir(args.output_dir, "probe_answer", probe_doc_ids)
                probe_searcher = EntitySearcher(
                    num_documents=-1,
                    path_to_gsw_files=probe_gsw_dir,
                    cache_dir=probe_cache_dir,
                    rebuild_cache=False,
                    verbose=False,
                    use_bm25=True,
                    use_gpu_for_qa_index=False,
                    embedding_gpu_memory_utilization=getattr(args, "embedding_gpu_memory_utilization", 0.5),
                )
                probe_embed_fn = _build_embed_texts_fn(probe_searcher)

                # Ingest probe bridges (no pattern classification needed for eval)
                probe_bridge_count = len(probe_bridge_result.get("bridges", []))
                _write_probe_progress("indexing", detail=f"indexing {probe_bridge_count} probe bridges")
                probe_registry.ingest_bridges(
                    probe_bridge_result.get("bridges", []),
                    batch_index=batch_index,
                    embed_texts_fn=probe_embed_fn,
                )

                # Answer probe questions
                probe_answer_kwargs = {
                    "batch_index": batch_index,
                    "batch_items": probe_items,
                    "entity_searcher": probe_searcher,
                    "bridge_registry": probe_registry,
                    "args": args,
                }
                _write_probe_progress("answering", detail=f"answering {len(probe_items)} probe questions")
                supported_params = set(inspect.signature(answer_query_batch).parameters)
                probe_batch_result = answer_query_batch(
                    **{key: value for key, value in probe_answer_kwargs.items() if key in supported_params}
                )

                # Compute probe metrics
                probe_f1s = [r.f1 for r in probe_batch_result.query_results if r.f1 is not None]
                probe_ems = [r.exact_match for r in probe_batch_result.query_results if r.exact_match is not None]
                probe_unique_relations = len({
                    r.source_relationship
                    for r in probe_registry.records.values()
                    if r.source_relationship
                })

                # Per-question probe details
                per_question_details = []
                for p_item, p_result in zip(probe_items, probe_batch_result.query_results):
                    per_question_details.append({
                        "question": p_item.question[:80],
                        "question_id": p_item.question_id,
                        "family": p_item.metadata.get("family", ""),
                        "predicted": (p_result.predicted_answer or "")[:50],
                        "gold": p_item.gold_answer or "",
                        "f1": p_result.f1,
                        "em": p_result.exact_match,
                        "bridge_used": p_result.bridge_evidence_used,
                    })

                probe_eval_record = {
                    "batch_index": batch_index,
                    "probe_f1": sum(probe_f1s) / len(probe_f1s) if probe_f1s else 0.0,
                    "probe_em": sum(probe_ems) / len(probe_ems) if probe_ems else 0.0,
                    "probe_bridges_generated": len(probe_registry.records),
                    "probe_relation_diversity": probe_unique_relations,
                    "guidance_entropy": compute_guidance_entropy(guidance_payload),
                    "main_bridges_this_batch": bridges_this_batch,
                    "main_bridges_cumulative": len(bridge_registry.records),
                    "guidance_batches_seen": query_experience.batches_seen,
                    "per_question": per_question_details,
                }
                learning_curve.append(probe_eval_record)

                # Save probe query results to disk
                probe_batch_dir = os.path.join(args.output_dir, f"probe_batch_{batch_index}")
                os.makedirs(probe_batch_dir, exist_ok=True)
                probe_results_path = os.path.join(probe_batch_dir, "query_answer_results.json")
                with open(probe_results_path, "w") as pf:
                    json.dump({
                        "batch_index": batch_index,
                        "query_results": [r.as_dict() for r in probe_batch_result.query_results],
                    }, pf, indent=2, default=str)

                # Print per-question probe table
                probe_table = Table(title=f"Probe Evaluation \u2014 Batch {batch_index}", box=box.SIMPLE)
                probe_table.add_column("Family", width=16)
                probe_table.add_column("Question", width=45)
                probe_table.add_column("F1", justify="right", width=5)
                probe_table.add_column("EM", justify="center", width=3)
                probe_table.add_column("Predicted", width=30)
                for detail in per_question_details:
                    f1_val = detail["f1"] or 0
                    style = "green" if f1_val >= 0.8 else ("yellow" if f1_val > 0 else "red")
                    probe_table.add_row(
                        detail["family"],
                        detail["question"][:45],
                        f"{f1_val:.2f}",
                        "\u2713" if (detail["em"] or 0) == 1.0 else "\u2717",
                        detail["predicted"][:30],
                        style=style,
                    )
                console.print(probe_table)
                _write_probe_progress(
                    "done",
                    detail=f"F1={probe_eval_record['probe_f1']:.4f} EM={probe_eval_record['probe_em']:.4f}",
                    probe_f1=probe_eval_record["probe_f1"],
                    probe_em=probe_eval_record["probe_em"],
                    probe_bridges=probe_eval_record["probe_bridges_generated"],
                )
                console.print(f"  [magenta]Probe F1: {probe_eval_record['probe_f1']:.4f} | "
                              f"Probe EM: {probe_eval_record['probe_em']:.4f} | "
                              f"Bridges: {probe_eval_record['probe_bridges_generated']} | "
                              f"Entropy: {probe_eval_record['guidance_entropy']:.3f}[/magenta]")
            finally:
                shutil.rmtree(probe_temp_base, ignore_errors=True)

        batch_summary = {
            "batch_index": batch_index,
            "doc_ids": current_doc_ids,
            "num_questions": len(batch_items),
            "generation_mode": batch_bridge_result.get("generation_mode", "single_batch"),
            "generation_executor": batch_bridge_result.get("generation_executor", "single"),
            "generation_shards": list(batch_bridge_result.get("generation_shards", [])),
            "generation_guidance": guidance_payload,
            "post_batch_demand_sketch": post_batch_demand_sketch,
            "bridge_exemplars": bridge_exemplars,
            "bridge_creation": batch_bridge_result,
            "query_batch_result": query_batch_result.as_dict(),
            "query_experience_after_batch": query_experience.as_dict(),
            "probe_eval": probe_eval_record,
        }
        batch_results.append(batch_summary)

        for item, query_result in zip(batch_items, query_batch_result.query_results):
            flat_query_results.append(
                {
                    "batch_index": batch_index,
                    "question_id": item.question_id,
                    "question_idx": item.metadata.get("question_idx"),
                    "question": item.question,
                    "doc_ids": list(item.doc_ids),
                    "gold_answer": item.gold_answer,
                    "answer_aliases": list(item.answer_aliases),
                    **query_result.as_dict(),
                }
            )

    summary = {
        "num_batches": len(batch_results),
        "total_queries": len(flat_query_results),
        "total_accepted_bridges": len(bridge_registry.records),
        "curriculum_bridge_registry": [record.exemplar() for record in bridge_registry.records.values()],
    }

    # Write learning curve if probe evaluation was run
    if learning_curve:
        lc_path = os.path.join(args.output_dir, "learning_curve.json")
        with open(lc_path, "w") as f:
            json.dump(learning_curve, f, indent=2)
        console.print(f"\n[bold green]Learning curve written to {lc_path} ({len(learning_curve)} data points)[/bold green]")

    return {
        "summary": summary,
        "results": flat_query_results,
        "batch_results": batch_results,
        "learning_curve": learning_curve,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run bridge test on first N bridge-style questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--questions", type=str, default="playground_data/musique.json",
                        help="Path to questions JSON (MuSiQue or 2Wiki/Hotpot-style)")
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="Optional corpus JSON path for context-title datasets (title -> global doc mapping)")
    parser.add_argument("--gsw_path", type=str, required=True,
                        help="Path to GSW networks directory (e.g. networks_4_1_mini)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start question index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="End question index (exclusive). Defaults to all.")

    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="Max iterations per entity")
    parser.add_argument("--max_tokens", type=int, default=500_000,
                        help="Max token budget per question/batch")

    parser.add_argument("--model", type=str, default="Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for vLLM/OpenAI-compatible server")
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--pipeline_mode", type=str, default="legacy",
                        choices=["legacy", "rlm", "hybrid"],
                        help="Exploration pipeline for run_sleep_time.py")
    parser.add_argument("--hybrid_scope", type=str, default="doc_edge",
                        choices=["edge", "doc_edge", "corpus_doc_edge"],
                        help="Hybrid autonomy scope when pipeline_mode=hybrid")
    parser.add_argument("--root_model", type=str, default=None,
                        help="Optional root-stage model in RLM mode")
    parser.add_argument("--worker_model", type=str, default=None,
                        help="Optional worker-stage model in RLM mode")
    parser.add_argument("--edge_max_depth", type=int, default=1,
                        help="Maximum recursion depth per edge in RLM mode")
    parser.add_argument("--edge_max_calls", type=int, default=2,
                        help="Maximum worker calls per edge in RLM mode")
    parser.add_argument("--edge_max_tokens", type=int, default=3000,
                        help="Per-edge token budget in RLM mode")
    parser.add_argument("--max_optional_docs_per_edge", type=int, default=2,
                        help="Maximum optional docs per edge packet in RLM mode")
    parser.add_argument("--edge_parallel_enabled", action="store_true",
                        help="Enable doc-local parallel worker generation in hybrid mode")
    parser.add_argument("--edge_parallel_workers", type=int, default=1,
                        help="Number of parallel worker threads per hybrid batch")
    parser.add_argument("--parallel_progress_heartbeat_seconds", type=float, default=10.0,
                        help="Heartbeat interval (seconds) for parallel batch progress events")
    parser.add_argument("--parallel_stuck_warning_seconds", type=float, default=60.0,
                        help="Warning threshold (seconds) for in-flight parallel worker calls")
    parser.add_argument("--verifier_parallel_workers", type=int, default=4,
                        help="Number of parallel verifier calls per edge commit")
    parser.add_argument("--edge_max_accepted_bridges", type=int, default=8,
                        help="Maximum accepted bridges committed per edge")

    parser.add_argument("--orchestration_mode", type=str, default="standard",
                        choices=["standard", "curriculum"],
                        help="Benchmark orchestration mode: per-question standard runner or batch curriculum loop")
    parser.add_argument("--curriculum_batch_size", type=int, default=4,
                        help="Batch size for curriculum question+doc batches")
    parser.add_argument("--curriculum_seed_batch_size", type=int, default=None,
                        help="Optional larger seed batch size for the first curriculum iteration")
    parser.add_argument("--bridge_query_top_k", type=int, default=12,
                        help="Top-k bridge/doc evidence items to keep per curriculum query")
    parser.add_argument("--bridge_prompt_exemplar_top_k", type=int, default=3,
                        help="Maximum prior bridge exemplars injected into the next batch guidance")
    parser.add_argument("--curriculum_disable_guidance", action="store_true",
                        help="Disable prior-query curriculum guidance injection while keeping curriculum batching active")
    parser.add_argument("--probe_eval", action="store_true",
                        help="Enable probe evaluation after each batch (requires probe-split items in manifest)")
    parser.add_argument("--guidance_decay_rate", type=float, default=0.0,
                        help="Exponential decay rate for demand pattern counts (0.0=no decay, 0.5=half-life per batch)")
    parser.add_argument("--curriculum_generation_parallel_enabled", action="store_true",
                        help="Enable question-group sharded bridge generation inside each curriculum batch")
    parser.add_argument("--curriculum_generation_parallel_workers", type=int, default=1,
                        help="Number of worker processes for question-group sharded curriculum generation")
    parser.add_argument("--embedding_gpu_memory_utilization", type=float, default=0.5,
                        help="GPU memory fraction reserved for the retrieval embedding model.")

    parser.add_argument("--output_dir", type=str, default="logs/bridge_test",
                        help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show run_sleep_time.py output")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Show agent reasoning (implies --verbose)")

    args = parser.parse_args()

    if args.orchestration_mode == "curriculum" and args.pipeline_mode == "legacy":
        raise ValueError("curriculum orchestration requires pipeline_mode=rlm or hybrid so worker prompts can consume batch guidance")

    os.makedirs(args.output_dir, exist_ok=True)

    console.print("[cyan]Loading questions...[/cyan]")
    questions, schema = load_questions(args.questions, args.start, args.end)
    console.print(f"  Loaded {len(questions)} questions (range: {args.start}:{args.end})")
    console.print(f"  Detected schema: {schema}")

    if schema == "unknown" and questions:
        raise ValueError(
            "Unsupported question schema. Expected either MuSiQue ('paragraphs') "
            "or context-title format ('context' + 'supporting_facts')."
        )

    title_to_idx = None
    if schema == "context_titles":
        corpus_path = args.corpus_path or infer_corpus_path(args.questions)
        if not corpus_path:
            raise ValueError(
                "Could not infer corpus path for context-title dataset. "
                "Pass --corpus_path explicitly."
            )
        console.print(f"  Loading corpus index from: {corpus_path}")
        title_to_idx = build_title_to_doc_index(corpus_path)
        console.print(f"  Indexed {len(title_to_idx)} titles")

    if args.orchestration_mode == "curriculum":
        items = build_batch_query_items(questions, schema=schema, title_to_idx=title_to_idx)
        curriculum_result = run_curriculum(items, args)
        combined = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "start": args.start,
                "end": args.end,
                "schema": schema,
                "model": args.model,
                "pipeline_mode": args.pipeline_mode,
                "hybrid_scope": args.hybrid_scope,
                "orchestration_mode": args.orchestration_mode,
                "curriculum_batch_size": args.curriculum_batch_size,
                "curriculum_seed_batch_size": args.curriculum_seed_batch_size,
                "bridge_query_top_k": args.bridge_query_top_k,
                "bridge_prompt_exemplar_top_k": args.bridge_prompt_exemplar_top_k,
                "curriculum_disable_guidance": args.curriculum_disable_guidance,
                "curriculum_generation_parallel_enabled": args.curriculum_generation_parallel_enabled,
                "curriculum_generation_parallel_workers": args.curriculum_generation_parallel_workers,
                "curriculum_generation_executor": (
                    "thread" if args.curriculum_generation_parallel_enabled else "single"
                ),
            },
            **curriculum_result,
        }
    else:
        all_results = []
        for q_idx, question in questions:
            try:
                result = run_question(q_idx, question, args, schema=schema, title_to_idx=title_to_idx)
                all_results.append(result)
            except Exception as e:
                console.print(f"  [red]Failed: {e}[/red]")
                all_results.append({
                    "question_idx": q_idx,
                    "question": question.get("question", ""),
                    "error": str(e),
                })

        combined = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "start": args.start,
                "end": args.end,
                "schema": schema,
                "model": args.model,
                "max_iterations": args.max_iterations,
                "pipeline_mode": args.pipeline_mode,
                "hybrid_scope": args.hybrid_scope,
                "edge_parallel_enabled": args.edge_parallel_enabled,
                "edge_parallel_workers": args.edge_parallel_workers,
                "orchestration_mode": args.orchestration_mode,
                "curriculum_disable_guidance": args.curriculum_disable_guidance,
            },
            "results": all_results,
        }

    output_file = os.path.join(args.output_dir, "bridge_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")

    console.print(f"\n[bold]Summary:[/bold]")
    if args.orchestration_mode == "curriculum":
        summary = combined.get("summary", {})
        console.print(f"  Batches: {summary.get('num_batches', 0)}")
        console.print(f"  Queries answered: {summary.get('total_queries', 0)}")
        console.print(f"  Accepted bridges indexed: {summary.get('total_accepted_bridges', 0)}")
    else:
        total_bridges = sum(r.get("num_bridges", 0) for r in combined["results"])
        console.print(f"  Questions tested: {len(combined['results'])}")
        console.print(f"  Total bridges: {total_bridges}")
        avg = (total_bridges / len(combined["results"])) if combined["results"] else 0.0
        console.print(f"  Avg bridges/question: {avg:.1f}")


if __name__ == "__main__":
    main()
