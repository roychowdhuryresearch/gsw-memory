from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - dependency already used elsewhere in repo
    BM25Okapi = None  # type: ignore[assignment]

from gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch


QUESTION_PATTERN_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("burial_place", ("buried", "burial")),
    ("death_date", ("when did", "die")),
    ("death_place", ("where did", "die")),
    ("birth_date", ("when was", "born")),
    ("birth_place", ("where was", "born")),
    ("mother", ("mother",)),
    ("father", ("father",)),
    ("parent", ("parent", "parents")),
    ("spouse", ("spouse", "husband", "wife", "married")),
    ("child", ("child", "children", "son", "daughter")),
    ("reign_start", ("start", "reign")),
    ("reign_end", ("end", "reign")),
    ("title", ("title", "king", "queen", "ruler", "office", "emperor", "duke", "count")),
]


def _tokenize_bm25(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _normalize_pattern_label(label: str) -> str:
    cleaned = re.sub(r"\s+", "_", (label or "").strip().lower())
    return cleaned or "other"


def infer_question_patterns(question: str) -> List[str]:
    lowered = (question or "").strip().lower()
    if not lowered:
        return ["other"]

    patterns: List[str] = []
    for label, required_terms in QUESTION_PATTERN_RULES:
        if all(term in lowered for term in required_terms):
            patterns.append(label)

    if not patterns:
        if lowered.startswith("when "):
            patterns.append("time")
        elif lowered.startswith("where "):
            patterns.append("place")
        elif lowered.startswith("who "):
            patterns.append("person")
        else:
            patterns.append("other")

    deduped: List[str] = []
    seen = set()
    for pattern in patterns:
        normalized = _normalize_pattern_label(pattern)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or ["other"]


@dataclass
class BatchQueryItem:
    question: str
    doc_ids: List[str]
    gold_answer: Optional[str] = None
    answer_aliases: List[str] = field(default_factory=list)
    question_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BridgeRecord:
    bridge_id: str
    question: str
    reverse_question: str
    answer_text: str
    reverse_answer_text: str
    source_docs: List[str]
    confidence: float
    pattern_tags: List[str]
    batch_index: int
    retrieved_count: int = 0
    helpful_count: int = 0
    last_helpful_batch: Optional[int] = None

    def exemplar(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "question": self.question,
            "reverse_question": self.reverse_question,
            "pattern_tags": list(self.pattern_tags),
            "source_docs": list(self.source_docs),
            "confidence": float(self.confidence),
            "retrieved_count": int(self.retrieved_count),
            "helpful_count": int(self.helpful_count),
        }


@dataclass
class BridgeSurface:
    bridge_id: str
    orientation: str
    question_text: str
    answer_text: str
    pattern_tags: List[str]
    source_docs: List[str]

    def evidence_text(self) -> str:
        answer_text = self.answer_text.strip() or "Unknown"
        return f"Q: {self.question_text} A: {answer_text}"


@dataclass
class QueryAnswerResult:
    question_id: str
    question: str
    predicted_answer: str
    full_response: str
    exact_match: Optional[float]
    f1: Optional[float]
    bridge_evidence_used: bool
    answer_source_mix: Dict[str, int]
    top_evidence: List[Dict[str, Any]]
    patterns: List[str]
    gold_answers: List[str]

    def primary_score(self) -> float:
        if self.f1 is not None:
            return float(self.f1)
        return 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchFeedbackBrief:
    high_demand_patterns: List[Dict[str, Any]]
    low_score_patterns: List[Dict[str, Any]]
    doc_dominant_patterns: List[Dict[str, Any]]
    bridge_helpful_patterns: List[Dict[str, Any]]
    summary_lines: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryExperienceState:
    total_questions: int = 0
    batches_seen: int = 0
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    representative_queries: Dict[str, str] = field(default_factory=dict)
    low_score_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    doc_dominant_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bridge_helpful_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_batch_feedback_brief: Optional[BatchFeedbackBrief] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": int(self.total_questions),
            "batches_seen": int(self.batches_seen),
            "pattern_counts": dict(self.pattern_counts),
            "representative_queries": dict(self.representative_queries),
            "low_score_patterns": list(self.low_score_patterns.values()),
            "doc_dominant_patterns": list(self.doc_dominant_patterns.values()),
            "bridge_helpful_patterns": list(self.bridge_helpful_patterns.values()),
            "last_batch_feedback_brief": (
                self.last_batch_feedback_brief.as_dict() if self.last_batch_feedback_brief is not None else None
            ),
        }


@dataclass
class QueryBatchResult:
    batch_index: int
    query_results: List[QueryAnswerResult]
    overall_metrics: Dict[str, float]
    feedback_brief: BatchFeedbackBrief

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "query_results": [result.as_dict() for result in self.query_results],
            "overall_metrics": dict(self.overall_metrics),
            "feedback_brief": self.feedback_brief.as_dict(),
        }


def split_curriculum_batches(
    items: Sequence[BatchQueryItem],
    batch_size: int,
    seed_batch_size: Optional[int] = None,
) -> List[List[BatchQueryItem]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seed_batch_size is None:
        seed_batch_size = batch_size
    if seed_batch_size <= 0:
        raise ValueError("seed_batch_size must be positive")

    batches: List[List[BatchQueryItem]] = []
    cursor = 0
    total = len(items)
    first_batch = min(seed_batch_size, total)
    if first_batch > 0:
        batches.append(list(items[:first_batch]))
        cursor = first_batch

    while cursor < total:
        next_cursor = min(cursor + batch_size, total)
        batches.append(list(items[cursor:next_cursor]))
        cursor = next_cursor

    return batches


def build_current_batch_demand_sketch(
    items: Sequence[BatchQueryItem],
    top_k_patterns: int = 4,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    representatives: Dict[str, str] = {}
    for item in items:
        for pattern in infer_question_patterns(item.question):
            counts[pattern] = counts.get(pattern, 0) + 1
            representatives.setdefault(pattern, item.question)

    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[: max(1, int(top_k_patterns))]
    top_patterns = [
        {
            "pattern": pattern,
            "count": count,
            "representative_query": representatives.get(pattern, ""),
        }
        for pattern, count in ranked
    ]
    return {
        "top_patterns": top_patterns,
        "total_questions": len(items),
    }


def build_prior_query_demand_sketch(
    state: QueryExperienceState,
    top_k_patterns: int = 4,
) -> Optional[Dict[str, Any]]:
    if state.total_questions <= 0 or not state.pattern_counts:
        return None

    ranked = sorted(
        state.pattern_counts.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )[: max(1, int(top_k_patterns))]
    top_patterns = [
        {
            "pattern": pattern,
            "count": count,
            "representative_query": state.representative_queries.get(pattern, ""),
        }
        for pattern, count in ranked
    ]
    return {
        "top_patterns": top_patterns,
        "total_questions": int(state.total_questions),
        "batches_seen": int(state.batches_seen),
    }


def _accumulate_pattern_rows(
    target: Dict[str, Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    representative_queries: Dict[str, str],
) -> None:
    for row in rows:
        pattern = _normalize_pattern_label(str(row.get("pattern", "")))
        if not pattern:
            continue
        count = int(row.get("count", 0) or 0)
        if count <= 0:
            continue
        representative = str(row.get("representative_query", "")).strip() or representative_queries.get(pattern, "")
        stats = target.setdefault(
            pattern,
            {"pattern": pattern, "count": 0, "representative_query": representative},
        )
        stats["count"] += count
        if not stats.get("representative_query") and representative:
            stats["representative_query"] = representative


def accumulate_query_experience_state(
    state: QueryExperienceState,
    batch_items: Sequence[BatchQueryItem],
    batch_result: QueryBatchResult,
) -> QueryExperienceState:
    for item in batch_items:
        state.total_questions += 1
        for pattern in infer_question_patterns(item.question):
            state.pattern_counts[pattern] = state.pattern_counts.get(pattern, 0) + 1
            state.representative_queries.setdefault(pattern, item.question)

    feedback = batch_result.feedback_brief
    _accumulate_pattern_rows(state.low_score_patterns, feedback.low_score_patterns, state.representative_queries)
    _accumulate_pattern_rows(state.doc_dominant_patterns, feedback.doc_dominant_patterns, state.representative_queries)
    _accumulate_pattern_rows(state.bridge_helpful_patterns, feedback.bridge_helpful_patterns, state.representative_queries)

    state.last_batch_feedback_brief = feedback
    state.batches_seen += 1
    return state


def build_batch_feedback_brief(results: Sequence[QueryAnswerResult], top_k_patterns: int = 4) -> BatchFeedbackBrief:
    demand_counts: Dict[str, int] = {}
    low_score: Dict[str, Dict[str, Any]] = {}
    doc_dominant: Dict[str, Dict[str, Any]] = {}
    bridge_helpful: Dict[str, Dict[str, Any]] = {}

    for result in results:
        patterns = result.patterns or ["other"]
        bridge_count = int(result.answer_source_mix.get("bridge", 0))
        doc_count = int(result.answer_source_mix.get("doc_entity", 0)) + int(result.answer_source_mix.get("doc_direct_qa", 0))
        score = result.primary_score()

        for pattern in patterns:
            demand_counts[pattern] = demand_counts.get(pattern, 0) + 1

            if score < 0.5:
                stats = low_score.setdefault(
                    pattern,
                    {"pattern": pattern, "count": 0, "representative_query": result.question, "avg_score_sum": 0.0},
                )
                stats["count"] += 1
                stats["avg_score_sum"] += score

            if doc_count > 0 and bridge_count == 0:
                stats = doc_dominant.setdefault(
                    pattern,
                    {"pattern": pattern, "count": 0, "representative_query": result.question},
                )
                stats["count"] += 1

            if result.bridge_evidence_used and score >= 0.5:
                stats = bridge_helpful.setdefault(
                    pattern,
                    {"pattern": pattern, "count": 0, "representative_query": result.question},
                )
                stats["count"] += 1

    def _rank(items: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked_items = list(items.values())
        for item in ranked_items:
            if "avg_score_sum" in item and item.get("count"):
                item["avg_score"] = round(float(item["avg_score_sum"]) / float(item["count"]), 4)
                item.pop("avg_score_sum", None)
        return sorted(ranked_items, key=lambda row: (-int(row.get("count", 0)), str(row.get("pattern", ""))))[
            : max(1, int(top_k_patterns))
        ]

    high_demand = sorted(demand_counts.items(), key=lambda pair: (-pair[1], pair[0]))[: max(1, int(top_k_patterns))]
    demand_rows = [{"pattern": pattern, "count": count} for pattern, count in high_demand]

    summary_lines: List[str] = []
    if demand_rows:
        summary_lines.append(
            "High demand: " + ", ".join(f"{row['pattern']} x{row['count']}" for row in demand_rows)
        )
    low_rows = _rank(low_score)
    if low_rows:
        summary_lines.append(
            "Weak coverage: " + ", ".join(f"{row['pattern']} x{row['count']}" for row in low_rows)
        )
    doc_rows = _rank(doc_dominant)
    if doc_rows:
        summary_lines.append(
            "Doc-backed more than bridge-backed: "
            + ", ".join(f"{row['pattern']} x{row['count']}" for row in doc_rows)
        )
    helpful_rows = _rank(bridge_helpful)
    if helpful_rows:
        summary_lines.append(
            "Bridge-helpful patterns: "
            + ", ".join(f"{row['pattern']} x{row['count']}" for row in helpful_rows)
        )

    return BatchFeedbackBrief(
        high_demand_patterns=demand_rows,
        low_score_patterns=low_rows,
        doc_dominant_patterns=doc_rows,
        bridge_helpful_patterns=helpful_rows,
        summary_lines=summary_lines,
    )


def _cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    vectors = np.asarray(matrix, dtype=np.float32)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1.0
    vector_norms[vector_norms == 0] = 1.0
    normalized_query = query / query_norm
    normalized_vectors = vectors / vector_norms
    return np.dot(normalized_vectors, normalized_query.T).reshape(-1)


def _embed_texts(
    embed_texts_fn: Optional[Callable[[Sequence[str]], Optional[Sequence[np.ndarray]]]],
    texts: Sequence[str],
) -> Optional[np.ndarray]:
    if not embed_texts_fn or not texts:
        return None
    try:
        embeddings = embed_texts_fn(texts)
    except Exception:
        return None
    if embeddings is None:
        return None
    matrix = np.asarray(list(embeddings), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        return None
    return matrix


class BridgeRegistry:
    def __init__(self) -> None:
        self.records: Dict[str, BridgeRecord] = {}
        self.surfaces: List[BridgeSurface] = []
        self._surface_keys: set[Tuple[str, str]] = set()
        self._bm25 = None
        self._surface_embeddings: Optional[np.ndarray] = None

    def ingest_bridges(
        self,
        bridges: Sequence[Dict[str, Any]],
        batch_index: int,
        embed_texts_fn: Optional[Callable[[Sequence[str]], Optional[Sequence[np.ndarray]]]] = None,
    ) -> int:
        added = 0
        for bridge in bridges:
            bridge_id = str(bridge.get("bridge_id", "")).strip()
            question = str(bridge.get("question", "")).strip()
            reverse_question = str(bridge.get("reverse_question", "")).strip()
            if not bridge_id or not question or not reverse_question:
                continue

            resolved_entities = bridge.get("resolved_entities") or {}
            forward_answers = [
                item.get("name", "")
                for item in resolved_entities.get("answers", [])
                if isinstance(item, dict) and item.get("name")
            ]
            reverse_answers = [
                item.get("name", "")
                for item in resolved_entities.get("reverse_answers", [])
                if isinstance(item, dict) and item.get("name")
            ]

            forward_text = ", ".join(name for name in forward_answers if name)
            reverse_text = ", ".join(name for name in reverse_answers if name)

            record = self.records.get(bridge_id)
            if record is None:
                record = BridgeRecord(
                    bridge_id=bridge_id,
                    question=question,
                    reverse_question=reverse_question,
                    answer_text=forward_text,
                    reverse_answer_text=reverse_text,
                    source_docs=list(bridge.get("source_docs", [])),
                    confidence=float(bridge.get("confidence", 0.0) or 0.0),
                    pattern_tags=sorted(
                        set(infer_question_patterns(question) + infer_question_patterns(reverse_question))
                    ),
                    batch_index=int(batch_index),
                )
                self.records[bridge_id] = record
                added += 1
            else:
                if forward_text:
                    record.answer_text = forward_text
                if reverse_text:
                    record.reverse_answer_text = reverse_text

            for orientation, query_text, answer_text in (
                ("forward", question, record.answer_text),
                ("reverse", reverse_question, record.reverse_answer_text),
            ):
                surface_key = (bridge_id, orientation)
                if surface_key in self._surface_keys:
                    continue
                self._surface_keys.add(surface_key)
                self.surfaces.append(
                    BridgeSurface(
                        bridge_id=bridge_id,
                        orientation=orientation,
                        question_text=query_text,
                        answer_text=answer_text,
                        pattern_tags=list(record.pattern_tags),
                        source_docs=list(record.source_docs),
                    )
                )

        if added > 0:
            self._rebuild_indexes(embed_texts_fn)
        return added

    def _rebuild_indexes(
        self,
        embed_texts_fn: Optional[Callable[[Sequence[str]], Optional[Sequence[np.ndarray]]]] = None,
    ) -> None:
        texts = [surface.question_text for surface in self.surfaces]
        if BM25Okapi is not None and texts:
            self._bm25 = BM25Okapi([_tokenize_bm25(text) for text in texts])
        else:
            self._bm25 = None
        self._surface_embeddings = _embed_texts(embed_texts_fn, texts)

    def search(
        self,
        query: str,
        top_k: int = 10,
        embed_texts_fn: Optional[Callable[[Sequence[str]], Optional[Sequence[np.ndarray]]]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0 or not self.surfaces:
            return []

        scores = np.zeros(len(self.surfaces), dtype=np.float32)
        if self._bm25 is not None:
            bm25_scores = np.asarray(self._bm25.get_scores(_tokenize_bm25(query)), dtype=np.float32)
            if bm25_scores.size:
                max_score = float(np.max(bm25_scores))
                if max_score > 0:
                    scores += bm25_scores / max_score

        if self._surface_embeddings is None and embed_texts_fn is not None:
            self._surface_embeddings = _embed_texts(
                embed_texts_fn, [surface.question_text for surface in self.surfaces]
            )

        if self._surface_embeddings is not None and embed_texts_fn is not None:
            query_matrix = _embed_texts(embed_texts_fn, [query])
            if query_matrix is not None:
                cosine = _cosine_scores(query_matrix[0], self._surface_embeddings)
                if cosine.size:
                    cosine = np.clip(cosine, 0.0, None)
                    max_cosine = float(np.max(cosine))
                    if max_cosine > 0:
                        scores += cosine / max_cosine

        order = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            surface = self.surfaces[int(idx)]
            record = self.records.get(surface.bridge_id)
            results.append(
                {
                    "bridge_id": surface.bridge_id,
                    "orientation": surface.orientation,
                    "question": surface.question_text,
                    "answer_text": surface.answer_text,
                    "pattern_tags": list(surface.pattern_tags),
                    "source_docs": list(surface.source_docs),
                    "score": round(score, 4),
                    "source_type": "bridge",
                    "evidence_text": surface.evidence_text(),
                    "retrieved_count": int(record.retrieved_count) if record else 0,
                    "helpful_count": int(record.helpful_count) if record else 0,
                }
            )
        return results

    def record_query_usage(
        self,
        used_bridge_ids: Iterable[str],
        helpful_bridge_ids: Iterable[str],
        batch_index: int,
    ) -> None:
        seen_used = {str(bridge_id).strip() for bridge_id in used_bridge_ids if str(bridge_id).strip()}
        seen_helpful = {str(bridge_id).strip() for bridge_id in helpful_bridge_ids if str(bridge_id).strip()}

        for bridge_id in seen_used:
            record = self.records.get(bridge_id)
            if record is None:
                continue
            record.retrieved_count += 1
        for bridge_id in seen_helpful:
            record = self.records.get(bridge_id)
            if record is None:
                continue
            record.helpful_count += 1
            record.last_helpful_batch = int(batch_index)

    def select_exemplars(
        self,
        prior_query_demand_sketch: Optional[Dict[str, Any]],
        prior_feedback_brief: Optional[BatchFeedbackBrief],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0 or not self.records:
            return []

        target_weights: Dict[str, float] = {}
        if isinstance(prior_query_demand_sketch, dict):
            for row in prior_query_demand_sketch.get("top_patterns", []):
                pattern = _normalize_pattern_label(str(row.get("pattern", "")))
                if not pattern:
                    continue
                target_weights[pattern] = target_weights.get(pattern, 0.0) + float(row.get("count", 0))

        if prior_feedback_brief is not None:
            for row in prior_feedback_brief.high_demand_patterns:
                pattern = _normalize_pattern_label(str(row.get("pattern", "")))
                target_weights[pattern] = target_weights.get(pattern, 0.0) + 0.5 * float(row.get("count", 0))
            for row in prior_feedback_brief.low_score_patterns:
                pattern = _normalize_pattern_label(str(row.get("pattern", "")))
                target_weights[pattern] = target_weights.get(pattern, 0.0) + 2.0 * float(row.get("count", 0))
            for row in prior_feedback_brief.doc_dominant_patterns:
                pattern = _normalize_pattern_label(str(row.get("pattern", "")))
                target_weights[pattern] = target_weights.get(pattern, 0.0) + 1.5 * float(row.get("count", 0))
            for row in prior_feedback_brief.bridge_helpful_patterns:
                pattern = _normalize_pattern_label(str(row.get("pattern", "")))
                target_weights[pattern] = target_weights.get(pattern, 0.0) + 0.75 * float(row.get("count", 0))

        scored: List[Tuple[float, BridgeRecord]] = []
        for record in self.records.values():
            pattern_score = sum(target_weights.get(pattern, 0.0) for pattern in record.pattern_tags)
            score = (
                pattern_score
                + 1.5 * float(record.helpful_count)
                + 0.5 * float(record.retrieved_count)
                + max(0.0, float(record.confidence))
            )
            scored.append((score, record))

        exemplars: List[Dict[str, Any]] = []
        seen_patterns: set[Tuple[str, ...]] = set()
        for _, record in sorted(scored, key=lambda item: (-item[0], item[1].bridge_id)):
            pattern_key = tuple(sorted(record.pattern_tags))
            if pattern_key in seen_patterns and len(exemplars) < top_k:
                continue
            seen_patterns.add(pattern_key)
            exemplars.append(record.exemplar())
            if len(exemplars) >= top_k:
                break
        return exemplars


def collect_doc_evidence(
    entity_searcher: Any,
    question: str,
    top_k_entities: int = 10,
    top_k_direct_qa: int = 10,
) -> List[Dict[str, Any]]:
    query_embedding = None
    if hasattr(entity_searcher, "_embed_query"):
        try:
            embedded = entity_searcher._embed_query([question])
            if embedded is not None and len(embedded) > 0:
                query_embedding = np.asarray(embedded[0], dtype=np.float32)
        except Exception:
            query_embedding = None

    entity_results = entity_searcher.search(
        query=question,
        query_embedding=query_embedding,
        top_k=top_k_entities,
        verbose=False,
    )

    entity_qa_pairs: List[Dict[str, Any]] = []
    for entity, score in entity_results:
        for qa in entity.get("qa_pairs", []):
            answer_names = qa.get("answer_names", qa.get("answers", []))
            if isinstance(answer_names, str):
                answer_names = [answer_names]
            answer_text = ", ".join(str(name) for name in answer_names if str(name).strip())
            if not qa.get("question") or not answer_text:
                continue
            entity_qa_pairs.append(
                {
                    "question": qa["question"],
                    "answer_text": answer_text,
                    "score": float(score),
                    "source_type": "doc_entity",
                    "source_entity": entity.get("name"),
                    "doc_id": entity.get("doc_id"),
                    "evidence_text": f"Q: {qa['question']} A: {answer_text}",
                }
            )

    direct_results = entity_searcher.search_qa_pairs_direct(
        query=question,
        query_embedding=query_embedding,
        top_k=top_k_direct_qa,
        verbose=False,
    )
    direct_qa_pairs: List[Dict[str, Any]] = []
    for qa in direct_results:
        answer_names = qa.get("answer_names", qa.get("answers", []))
        if isinstance(answer_names, str):
            answer_names = [answer_names]
        answer_text = ", ".join(str(name) for name in answer_names if str(name).strip())
        if not qa.get("question") or not answer_text:
            continue
        direct_qa_pairs.append(
            {
                "question": qa["question"],
                "answer_text": answer_text,
                "score": float(qa.get("similarity_score", 0.0) or 0.0),
                "source_type": "doc_direct_qa",
                "doc_id": qa.get("doc_id"),
                "evidence_text": f"Q: {qa['question']} A: {answer_text}",
            }
        )

    merged: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for item in entity_qa_pairs + direct_qa_pairs:
        key = (str(item.get("question", "")), str(item.get("answer_text", "")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def rerank_merged_evidence(
    question: str,
    evidence_items: Sequence[Dict[str, Any]],
    embed_texts_fn: Optional[Callable[[Sequence[str]], Optional[Sequence[np.ndarray]]]] = None,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    if top_k <= 0 or not evidence_items:
        return []

    base_scores = np.asarray(
        [float(item.get("score", 0.0) or 0.0) for item in evidence_items],
        dtype=np.float32,
    )
    normalized_base = np.zeros_like(base_scores)
    if base_scores.size and float(np.max(base_scores)) > 0:
        normalized_base = base_scores / float(np.max(base_scores))

    semantic_scores = np.zeros_like(base_scores)
    if embed_texts_fn is not None:
        query_matrix = _embed_texts(embed_texts_fn, [question])
        evidence_matrix = _embed_texts(embed_texts_fn, [item.get("evidence_text", "") for item in evidence_items])
        if query_matrix is not None and evidence_matrix is not None:
            semantic_scores = np.clip(_cosine_scores(query_matrix[0], evidence_matrix), 0.0, None)
            if semantic_scores.size and float(np.max(semantic_scores)) > 0:
                semantic_scores = semantic_scores / float(np.max(semantic_scores))

    final_scores = 0.55 * normalized_base + 0.45 * semantic_scores
    order = np.argsort(final_scores)[::-1][:top_k]

    reranked: List[Dict[str, Any]] = []
    for idx in order:
        item = dict(evidence_items[int(idx)])
        item["final_score"] = round(float(final_scores[int(idx)]), 4)
        reranked.append(item)
    return reranked


def parse_answer_from_response(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    if "Answer:" in text:
        answer = text.split("Answer:", 1)[-1].strip()
    else:
        answer = text
    if answer.endswith(".") and answer[:-1].replace(",", "").replace(" ", "").isdigit():
        answer = answer[:-1]
    return answer.strip()


def build_panini_answer_messages(question: str, evidence: Sequence[str]) -> List[Dict[str, str]]:
    evidence_text = "\n".join(str(item) for item in evidence if str(item).strip())
    prompt_text = f"""
{evidence_text}
\n\nQuestion: {question}
\n\nThought:

"""
    one_shot_docs = (
        """ Q: Who directed The Last Horse? A: Edgar Neville
            Q: When was The Last Horse released? A: 1950
            Q: When was the University of Southampton founded? A: 1862
            Q: Where is the University of Southampton located? A: Southampton
            Q: What is the population of Stanton Township? A: 505
            Q: Where is Stanton Township? A: Champaign County, Illinois
            Q: Who is Neville A. Stanton? A: British Professor of Human Factors and Ergonomics
            Q: Where does Neville A. Stanton work? A: University of Southampton
            Q: What is Neville A. Stanton's profession? A: Professor
            Q: Who directed Finding Nemo? A: Andrew Stanton
            Q: When was Finding Nemo released? A: 2003
            Q: What company produced Finding Nemo? A: Pixar Animation Studios"""
    )
    rag_qa_system = (
        'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
        'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
        'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
    )
    one_shot_input = (
        f"{one_shot_docs}"
        "\n\nQuestion: "
        "When was Neville A. Stanton's employer founded?"
        "\nThought: "
    )
    one_shot_output = (
        "From the QA pairs, the employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
        "\nAnswer: 1862."
    )
    return [
        {"role": "system", "content": rag_qa_system},
        {"role": "user", "content": one_shot_input},
        {"role": "assistant", "content": one_shot_output},
        {"role": "user", "content": prompt_text},
    ]


def score_predicted_answer(gold_answers: Sequence[str], predicted_answer: str) -> Tuple[Optional[float], Optional[float]]:
    cleaned_gold = [str(answer).strip() for answer in gold_answers if str(answer).strip()]
    if not cleaned_gold:
        return None, None
    _, per_example = evaluate_qa_batch([cleaned_gold], [predicted_answer or ""])
    if not per_example:
        return None, None
    row = per_example[0]
    return float(row.get("ExactMatch", 0.0)), float(row.get("F1", 0.0))


def build_curriculum_guidance_payload(
    prior_query_demand_sketch: Optional[Dict[str, Any]],
    prior_batch_feedback_brief: Optional[BatchFeedbackBrief],
    bridge_exemplars: Sequence[Dict[str, Any]],
    prior_batches_seen: int = 0,
) -> Dict[str, Any]:
    return {
        "prior_query_demand_sketch": prior_query_demand_sketch,
        "prior_batch_feedback_brief": prior_batch_feedback_brief.as_dict() if prior_batch_feedback_brief else None,
        "bridge_exemplars": list(bridge_exemplars),
        "prior_batches_seen": int(prior_batches_seen),
    }
