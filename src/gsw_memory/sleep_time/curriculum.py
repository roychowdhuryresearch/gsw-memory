from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel

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

QUESTION_PATTERN_OPERATIONS = {
    "lookup",
    "compare",
    "filter",
    "count",
    "boolean",
    "superlative",
    "temporal_order",
}
QUESTION_PATTERN_ANSWER_TYPES = {
    "person",
    "place",
    "date",
    "year",
    "number",
    "title",
    "organization",
    "region",
    "other",
}
QUESTION_PATTERN_DOMAINS = {
    "family",
    "film",
    "history_politics",
    "geography",
    "sports",
    "organization",
    "other",
}
QUESTION_PATTERN_COMPARISON_TARGETS = {
    "none",
    "earlier",
    "later",
    "before",
    "after",
    "first",
    "last",
    "higher",
    "lower",
}
STRICT_PATTERN_FALLBACK_MAX_FAILURES = 8
STRICT_PATTERN_FALLBACK_MAX_FAILURE_RATIO = 0.05

def _tokenize_bm25(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _normalize_pattern_label(label: str) -> str:
    cleaned = re.sub(r"\s+", "_", (label or "").strip().lower())
    return cleaned or "other"


def _normalize_question_cache_key(question: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(question or "").lower()))


def _normalize_enum_label(value: Any, allowed: set[str], default: str) -> str:
    normalized = _normalize_pattern_label(str(value or ""))
    if normalized in allowed:
        return normalized
    return default


def _normalize_relation_label(value: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "other"


def _build_pattern_key(
    relation_label: str,
    operation: str,
    comparison_target: str,
    answer_type: str,
) -> str:
    parts = [relation_label, operation]
    if comparison_target != "none":
        parts.append(comparison_target)
    parts.append(answer_type)
    return ".".join(_normalize_pattern_label(part) for part in parts if part)


class QuestionPatternResponseSchema(BaseModel):
    operation: str
    answer_type: str
    domain: str
    relation_label: str
    comparison_target: str
    pattern_key: Optional[str] = None


@dataclass(frozen=True)
class QuestionPattern:
    operation: str
    answer_type: str
    domain: str
    relation_label: str
    comparison_target: str
    pattern_key: str
    classifier_source: str = "heuristic_fallback"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "answer_type": self.answer_type,
            "domain": self.domain,
            "relation_label": self.relation_label,
            "comparison_target": self.comparison_target,
            "pattern_key": self.pattern_key,
            "classifier_source": self.classifier_source,
        }


def _looks_like_boolean_question(lowered: str) -> bool:
    return lowered.startswith(("is ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ", "can ", "could ", "should ", "would ", "will "))


def _infer_comparison_target(question: str) -> str:
    lowered = (question or "").strip().lower()
    if " released first" in lowered or lowered.startswith("which ") and " first" in lowered:
        return "earlier"
    if " released later" in lowered:
        return "later"
    if " before " in lowered:
        return "before"
    if " after " in lowered:
        return "after"
    if " first" in lowered:
        return "first"
    if " last" in lowered:
        return "last"
    if " higher" in lowered or " more than " in lowered:
        return "higher"
    if " lower" in lowered or " less than " in lowered:
        return "lower"
    return "none"


def infer_question_pattern_structured(question: str) -> QuestionPattern:
    lowered = (question or "").strip().lower()
    patterns = infer_question_patterns(question)
    comparison_target = _infer_comparison_target(question)

    if comparison_target != "none":
        operation = "compare"
    elif lowered.startswith("how many") or " number of " in lowered or lowered.startswith("count "):
        operation = "count"
    elif _looks_like_boolean_question(lowered):
        operation = "boolean"
    elif any(term in lowered for term in (" most ", " least ", " largest ", " smallest ", " oldest ", " youngest ", " highest ", " lowest ")):
        operation = "superlative"
    elif any(term in lowered for term in (" before ", " after ")) and lowered.startswith(("what ", "which ")):
        operation = "temporal_order"
    else:
        operation = "lookup"

    if lowered.startswith("when ") or any(pattern in patterns for pattern in ("birth_date", "death_date", "reign_start", "reign_end")):
        answer_type = "date"
    elif lowered.startswith("where ") or any(pattern in patterns for pattern in ("birth_place", "death_place", "burial_place")):
        answer_type = "place"
    elif lowered.startswith("who "):
        answer_type = "person"
    elif lowered.startswith("how many") or " number of " in lowered or "population" in lowered:
        answer_type = "number"
    elif any(term in lowered for term in ("which film", "which movie", "which song", "which book", "which title")):
        answer_type = "title"
    elif any(term in lowered for term in ("organization", "company", "university", "institution")):
        answer_type = "organization"
    elif "region" in lowered or "province" in lowered or "state" in lowered:
        answer_type = "region"
    else:
        answer_type = "other"

    if any(term in lowered for term in ("film", "movie", "released", "director", "cinema", "box office")):
        domain = "film"
    elif any(pattern in patterns for pattern in ("mother", "father", "parent", "spouse", "child")):
        domain = "family"
    elif any(term in lowered for term in ("king", "queen", "duke", "count", "reign", "dynasty", "emperor", "president", "minister")):
        domain = "history_politics"
    elif any(term in lowered for term in ("river", "mountain", "city", "country", "region", "province", "state", "located")):
        domain = "geography"
    elif any(term in lowered for term in ("team", "club", "league", "season", "tournament", "player", "coach")):
        domain = "sports"
    elif any(term in lowered for term in ("university", "organization", "company", "institution", "founded", "established")):
        domain = "organization"
    else:
        domain = "other"

    relation_label = "other"
    if "release" in lowered or "released" in lowered:
        if lowered.startswith("where "):
            relation_label = "release_place"
        else:
            relation_label = "release_date"
    elif "founded" in lowered or "established" in lowered:
        relation_label = "founding_date" if lowered.startswith("when ") else "founder"
    elif "buried" in lowered or "burial" in lowered:
        relation_label = "burial_place"
    elif "die" in lowered and "mother" in lowered:
        relation_label = "maternal_parent_death_date"
    elif "die" in lowered and "father" in lowered:
        relation_label = "paternal_parent_death_date"
    elif "mother" in lowered:
        relation_label = "maternal_parent"
    elif "father" in lowered:
        relation_label = "paternal_parent"
    elif "parent" in lowered or "parents" in lowered:
        relation_label = "parent"
    elif "spouse" in lowered or "husband" in lowered or "wife" in lowered or "married" in lowered:
        relation_label = "spouse"
    elif "child" in lowered or "children" in lowered or "son" in lowered or "daughter" in lowered:
        relation_label = "child"
    elif "die" in lowered:
        relation_label = "death_date" if lowered.startswith("when ") else "death_place"
    elif "born" in lowered:
        relation_label = "birth_date" if lowered.startswith("when ") else "birth_place"
    elif "reign" in lowered and "start" in lowered:
        relation_label = "reign_start"
    elif "reign" in lowered and "end" in lowered:
        relation_label = "reign_end"
    elif any(term in lowered for term in ("title", "king", "queen", "ruler", "office", "emperor", "duke", "count")):
        relation_label = "title"
    elif lowered.startswith("when "):
        relation_label = "time"
    elif lowered.startswith("where "):
        relation_label = "place"
    elif lowered.startswith("who "):
        relation_label = "person"

    relation_label = _normalize_relation_label(relation_label)
    operation = _normalize_enum_label(operation, QUESTION_PATTERN_OPERATIONS, "lookup")
    answer_type = _normalize_enum_label(answer_type, QUESTION_PATTERN_ANSWER_TYPES, "other")
    domain = _normalize_enum_label(domain, QUESTION_PATTERN_DOMAINS, "other")
    comparison_target = _normalize_enum_label(comparison_target, QUESTION_PATTERN_COMPARISON_TARGETS, "none")
    return QuestionPattern(
        operation=operation,
        answer_type=answer_type,
        domain=domain,
        relation_label=relation_label,
        comparison_target=comparison_target,
        pattern_key=_build_pattern_key(relation_label, operation, comparison_target, answer_type),
        classifier_source="heuristic_fallback",
    )


def normalize_question_pattern_payload(payload: Any) -> QuestionPattern:
    if isinstance(payload, QuestionPattern):
        return payload
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        raise ValueError("Pattern payload must be a dict-like object")

    provided_pattern_key = _normalize_pattern_label(str(payload.get("pattern_key") or payload.get("pattern") or ""))
    parsed_relation = ""
    parsed_operation = ""
    parsed_comparison = "none"
    parsed_answer_type = ""
    if provided_pattern_key:
        parts = [part for part in provided_pattern_key.split(".") if part]
        if len(parts) >= 3:
            parsed_relation = parts[0]
            parsed_operation = parts[1]
            if len(parts) == 4:
                parsed_comparison = parts[2]
                parsed_answer_type = parts[3]
            else:
                parsed_answer_type = parts[2]

    operation = _normalize_enum_label(payload.get("operation") or parsed_operation, QUESTION_PATTERN_OPERATIONS, "lookup")
    answer_type = _normalize_enum_label(payload.get("answer_type") or parsed_answer_type, QUESTION_PATTERN_ANSWER_TYPES, "other")
    domain = _normalize_enum_label(payload.get("domain"), QUESTION_PATTERN_DOMAINS, "other")
    relation_label = _normalize_relation_label(payload.get("relation_label") or parsed_relation)
    comparison_target = _normalize_enum_label(
        payload.get("comparison_target") or parsed_comparison,
        QUESTION_PATTERN_COMPARISON_TARGETS,
        "none",
    )
    pattern_key = _build_pattern_key(relation_label, operation, comparison_target, answer_type)
    return QuestionPattern(
        operation=operation,
        answer_type=answer_type,
        domain=domain,
        relation_label=relation_label,
        comparison_target=comparison_target,
        pattern_key=pattern_key,
        classifier_source=str(payload.get("classifier_source") or "llm"),
    )


class QuestionPatternClassifier:
    def __init__(
        self,
        classify_question_fn: Optional[Callable[[str], Any]] = None,
        classify_questions_fn: Optional[Callable[[Sequence[str]], Any]] = None,
        persistent_cache_path: Optional[str] = None,
        classifier_backend: Optional[str] = None,
        classifier_model_name: Optional[str] = None,
    ) -> None:
        self._classify_question_fn = classify_question_fn
        self._classify_questions_fn = classify_questions_fn
        self._cache: Dict[str, QuestionPattern] = {}
        self._persistent_cache_path = str(persistent_cache_path).strip() if persistent_cache_path else None
        self.classifier_backend = str(classifier_backend or "").strip() or None
        self.classifier_model_name = str(classifier_model_name or "").strip() or None
        self._load_persistent_cache()

    def _load_persistent_cache(self) -> None:
        if not self._persistent_cache_path:
            return
        try:
            with open(self._persistent_cache_path) as f:
                payload = json.load(f)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        for key, raw_pattern in payload.items():
            normalized_key = _normalize_question_cache_key(str(key))
            if not normalized_key:
                continue
            try:
                self._cache[normalized_key] = normalize_question_pattern_payload(raw_pattern)
            except Exception:
                continue

    def _flush_persistent_cache(self) -> None:
        if not self._persistent_cache_path:
            return
        serializable = {
            key: pattern.as_dict()
            for key, pattern in self._cache.items()
        }
        try:
            cache_path = self._persistent_cache_path
            parent = re.sub(r"[/\\\\][^/\\\\]+$", "", cache_path)
            if parent:
                import os

                os.makedirs(parent, exist_ok=True)
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(serializable, f, indent=2)
            import os

            os.replace(temp_path, cache_path)
        except Exception:
            return

    def classify(self, question: str) -> QuestionPattern:
        pattern, _cache_hit, _backend_invoked = self.classify_with_diagnostics(question)
        return pattern

    def classify_with_diagnostics(self, question: str) -> Tuple[QuestionPattern, bool, bool]:
        cache_key = _normalize_question_cache_key(question)
        if cache_key in self._cache:
            return self._cache[cache_key], True, False

        pattern: Optional[QuestionPattern] = None
        backend_invoked = False
        if self._classify_question_fn is not None:
            backend_invoked = True
            try:
                raw_result = self._classify_question_fn(question)
                pattern = normalize_question_pattern_payload(raw_result)
            except Exception:
                pattern = None
        if pattern is None:
            pattern = infer_question_pattern_structured(question)
        self._cache[cache_key] = pattern
        self._flush_persistent_cache()
        return pattern, False, backend_invoked

    def classify_many_with_diagnostics(
        self,
        questions: Sequence[str],
    ) -> Tuple[List[QuestionPattern], int, int, int]:
        if not questions:
            return [], 0, 0, 0

        patterns: List[Optional[QuestionPattern]] = [None] * len(questions)
        cache_hits = 0
        cache_misses = 0
        backend_invocations = 0
        missing_positions: Dict[str, List[int]] = {}
        missing_text_by_key: Dict[str, str] = {}
        missing_order: List[str] = []

        for index, question in enumerate(questions):
            cache_key = _normalize_question_cache_key(question)
            if cache_key in self._cache:
                patterns[index] = self._cache[cache_key]
                cache_hits += 1
                continue
            if cache_key not in missing_positions:
                cache_misses += 1
                missing_positions[cache_key] = []
                missing_text_by_key[cache_key] = str(question or "")
                missing_order.append(cache_key)
            else:
                cache_hits += 1
            missing_positions[cache_key].append(index)

        batch_results_by_key: Dict[str, QuestionPattern] = {}
        if missing_order and self._classify_questions_fn is not None:
            backend_invocations = len(missing_order)
            try:
                raw_batch_result = self._classify_questions_fn(
                    [missing_text_by_key[cache_key] for cache_key in missing_order]
                )
                if isinstance(raw_batch_result, dict):
                    for key, raw_pattern in raw_batch_result.items():
                        normalized_key = _normalize_question_cache_key(str(key))
                        if not normalized_key:
                            continue
                        try:
                            batch_results_by_key[normalized_key] = normalize_question_pattern_payload(raw_pattern)
                        except Exception:
                            continue
                elif isinstance(raw_batch_result, Sequence):
                    for cache_key, raw_pattern in zip(missing_order, raw_batch_result):
                        try:
                            batch_results_by_key[cache_key] = normalize_question_pattern_payload(raw_pattern)
                        except Exception:
                            continue
            except Exception:
                batch_results_by_key = {}

        cache_mutated = False
        for cache_key in missing_order:
            pattern = batch_results_by_key.get(cache_key)
            if pattern is None:
                question = missing_text_by_key[cache_key]
                if self._classify_question_fn is not None and self._classify_questions_fn is None:
                    try:
                        backend_invocations += 1
                        pattern = normalize_question_pattern_payload(self._classify_question_fn(question))
                    except Exception:
                        pattern = None
                if pattern is None:
                    pattern = infer_question_pattern_structured(question)
            self._cache[cache_key] = pattern
            cache_mutated = True
            for index in missing_positions.get(cache_key, []):
                patterns[index] = pattern

        if cache_mutated:
            self._flush_persistent_cache()

        finalized_patterns = [
            pattern if pattern is not None else infer_question_pattern_structured(question)
            for pattern, question in zip(patterns, questions)
        ]
        return finalized_patterns, cache_hits, cache_misses, backend_invocations

    def has_backend_classifier(self) -> bool:
        return self._classify_question_fn is not None or self._classify_questions_fn is not None

    @staticmethod
    def _is_llm_pattern(pattern: Optional["QuestionPattern"]) -> bool:
        return pattern is not None and str(pattern.classifier_source or "").strip() == "llm"

    def classify_many_strict_with_diagnostics(
        self,
        questions: Sequence[str],
    ) -> "StrictPatternBatchClassificationResult":
        if not questions:
            return StrictPatternBatchClassificationResult(
                patterns=[],
                classifier_backend=self.classifier_backend,
                classifier_model_name=self.classifier_model_name,
            )

        patterns: List[Optional[QuestionPattern]] = [None] * len(questions)
        cache_hits = 0
        cache_misses = 0
        llm_calls = 0
        stale_cache_reclassification_count = 0
        missing_positions: Dict[str, List[int]] = {}
        missing_text_by_key: Dict[str, str] = {}
        missing_order: List[str] = []
        failed_surface_examples: List[Dict[str, str]] = []
        successful_by_key: Dict[str, QuestionPattern] = {}
        cache_mutated = False
        backend_exception_message: Optional[str] = None
        initial_failed_surface_count = 0
        fallback_retry_attempted_count = 0
        fallback_retry_succeeded_count = 0
        fallback_retry_failed_count = 0

        for index, question in enumerate(questions):
            cache_key = _normalize_question_cache_key(question)
            cached_pattern = self._cache.get(cache_key)
            if self._is_llm_pattern(cached_pattern):
                patterns[index] = cached_pattern
                cache_hits += 1
                continue
            if cached_pattern is not None and cache_key not in missing_positions:
                stale_cache_reclassification_count += 1
                self._cache.pop(cache_key, None)
                cache_mutated = True
            if cache_key not in missing_positions:
                cache_misses += 1
                missing_positions[cache_key] = []
                missing_text_by_key[cache_key] = str(question or "")
                missing_order.append(cache_key)
            else:
                cache_hits += 1
            missing_positions[cache_key].append(index)

        failure_reason_by_key: Dict[str, str] = {}
        if missing_order:
            if self._classify_questions_fn is None:
                for cache_key in missing_order:
                    failure_reason_by_key[cache_key] = "no_batched_llm_classifier"
            else:
                llm_calls = len(missing_order)
                try:
                    raw_batch_result = self._classify_questions_fn(
                        [missing_text_by_key[cache_key] for cache_key in missing_order]
                    )
                except Exception as exc:
                    reason = f"backend_exception:{type(exc).__name__}"
                    backend_exception_message = str(exc).strip() or repr(exc)
                    for cache_key in missing_order:
                        failure_reason_by_key[cache_key] = reason
                else:
                    raw_rows_by_key: Dict[str, Any] = {}
                    if isinstance(raw_batch_result, dict):
                        for key, raw_pattern in raw_batch_result.items():
                            normalized_key = _normalize_question_cache_key(str(key))
                            if normalized_key:
                                raw_rows_by_key[normalized_key] = raw_pattern
                    elif isinstance(raw_batch_result, Sequence) and not isinstance(raw_batch_result, (str, bytes, bytearray)):
                        for cache_key, raw_pattern in zip(missing_order, raw_batch_result):
                            raw_rows_by_key[cache_key] = raw_pattern
                    else:
                        reason = f"invalid_batch_result_type:{type(raw_batch_result).__name__}"
                        for cache_key in missing_order:
                            failure_reason_by_key[cache_key] = reason

                    for cache_key in missing_order:
                        if cache_key in failure_reason_by_key:
                            continue
                        if cache_key not in raw_rows_by_key:
                            failure_reason_by_key[cache_key] = "missing_row"
                            continue
                        try:
                            normalized_pattern = normalize_question_pattern_payload(raw_rows_by_key[cache_key])
                        except Exception as exc:
                            failure_reason_by_key[cache_key] = f"invalid_row:{type(exc).__name__}"
                            continue
                        if not self._is_llm_pattern(normalized_pattern):
                            failure_reason_by_key[cache_key] = (
                                "non_llm_classifier_source:"
                                f"{str(normalized_pattern.classifier_source or '').strip() or 'unknown'}"
                            )
                            continue
                        successful_by_key[cache_key] = normalized_pattern

        initial_failed_surface_count = len(failure_reason_by_key)
        unresolved_keys = list(failure_reason_by_key)
        fallback_allowed = (
            bool(unresolved_keys)
            and bool(successful_by_key)
            and self._classify_question_fn is not None
            and len(unresolved_keys) <= STRICT_PATTERN_FALLBACK_MAX_FAILURES
            and (len(unresolved_keys) / float(len(missing_order))) <= STRICT_PATTERN_FALLBACK_MAX_FAILURE_RATIO
        )
        if fallback_allowed:
            for cache_key in unresolved_keys:
                question = missing_text_by_key.get(cache_key, "")
                fallback_retry_attempted_count += 1
                llm_calls += 1
                try:
                    fallback_pattern = normalize_question_pattern_payload(self._classify_question_fn(question))
                except Exception as exc:
                    failure_reason_by_key[cache_key] = f"fallback_invalid_row:{type(exc).__name__}"
                    fallback_retry_failed_count += 1
                    continue
                if not self._is_llm_pattern(fallback_pattern):
                    failure_reason_by_key[cache_key] = (
                        "fallback_non_llm_classifier_source:"
                        f"{str(fallback_pattern.classifier_source or '').strip() or 'unknown'}"
                    )
                    fallback_retry_failed_count += 1
                    continue
                successful_by_key[cache_key] = fallback_pattern
                failure_reason_by_key.pop(cache_key, None)
                fallback_retry_succeeded_count += 1

        for cache_key, pattern in successful_by_key.items():
            self._cache[cache_key] = pattern
            cache_mutated = True
            for index in missing_positions.get(cache_key, []):
                patterns[index] = pattern

        for cache_key, reason in failure_reason_by_key.items():
            if len(failed_surface_examples) < 10:
                failed_surface_examples.append(
                    {
                        "question": missing_text_by_key.get(cache_key, ""),
                        "reason": reason,
                    }
                )

        if cache_mutated:
            self._flush_persistent_cache()

        return StrictPatternBatchClassificationResult(
            patterns=patterns,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            llm_calls=llm_calls,
            uncached_surface_count=len(missing_order),
            llm_tagged_surface_count=len(successful_by_key),
            failed_surface_count=len(failure_reason_by_key),
            initial_failed_surface_count=initial_failed_surface_count,
            fallback_retry_attempted_count=fallback_retry_attempted_count,
            fallback_retry_succeeded_count=fallback_retry_succeeded_count,
            fallback_retry_failed_count=fallback_retry_failed_count,
            stale_cache_reclassification_count=stale_cache_reclassification_count,
            failed_surface_examples=failed_surface_examples,
            classifier_backend=self.classifier_backend,
            classifier_model_name=self.classifier_model_name,
            backend_exception_message=backend_exception_message,
        )


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
    forward_pattern: Dict[str, Any]
    reverse_pattern: Dict[str, Any]
    batch_index: int
    source_relationship: str = ""
    source_entity: str = ""
    source_neighbor: str = ""
    retrieved_count: int = 0
    helpful_count: int = 0
    last_helpful_batch: Optional[int] = None

    def exemplar(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "question": self.question,
            "reverse_question": self.reverse_question,
            "answer_text": self.answer_text,
            "reverse_answer_text": self.reverse_answer_text,
            "pattern_tags": list(self.pattern_tags),
            "pattern_key": str(self.forward_pattern.get("pattern_key", "")),
            "relation_label": str(self.forward_pattern.get("relation_label", "")),
            "domain": str(self.forward_pattern.get("domain", "")),
            "operation": str(self.forward_pattern.get("operation", "")),
            "answer_type": str(self.forward_pattern.get("answer_type", "")),
            "comparison_target": str(self.forward_pattern.get("comparison_target", "")),
            "forward_pattern": dict(self.forward_pattern),
            "reverse_pattern": dict(self.reverse_pattern),
            "source_docs": list(self.source_docs),
            "confidence": float(self.confidence),
            "retrieved_count": int(self.retrieved_count),
            "helpful_count": int(self.helpful_count),
            "source_relationship": self.source_relationship,
            "source_entity": self.source_entity,
            "source_neighbor": self.source_neighbor,
        }


@dataclass
class BridgeSurface:
    bridge_id: str
    orientation: str
    question_text: str
    answer_text: str
    pattern_tags: List[str]
    question_pattern: Dict[str, Any]
    source_docs: List[str]

    def evidence_text(self) -> str:
        answer_text = self.answer_text.strip() or "Unknown"
        return f"Q: {self.question_text} A: {answer_text}"

    def rerank_text(self) -> str:
        return self.question_text


@dataclass
class BridgeIngestResult:
    added_count: int
    added_bridge_ids: List[str]
    total_bridges_seen: int
    surfaces_total: int
    cache_hits: int
    cache_misses: int
    llm_calls: int
    uncached_surface_count: int = 0
    llm_tagged_surface_count: int = 0
    failed_surface_count: int = 0
    initial_failed_surface_count: int = 0
    fallback_retry_attempted_count: int = 0
    fallback_retry_succeeded_count: int = 0
    fallback_retry_failed_count: int = 0
    stale_cache_reclassification_count: int = 0
    failed_surface_examples: List[Dict[str, str]] = field(default_factory=list)
    classifier_backend: Optional[str] = None
    classifier_model_name: Optional[str] = None
    backend_exception_message: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "added_count": int(self.added_count),
            "added_bridge_ids": list(self.added_bridge_ids),
            "total_bridges_seen": int(self.total_bridges_seen),
            "surfaces_total": int(self.surfaces_total),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "llm_calls": int(self.llm_calls),
            "uncached_surface_count": int(self.uncached_surface_count),
            "llm_tagged_surface_count": int(self.llm_tagged_surface_count),
            "failed_surface_count": int(self.failed_surface_count),
            "initial_failed_surface_count": int(self.initial_failed_surface_count),
            "fallback_retry_attempted_count": int(self.fallback_retry_attempted_count),
            "fallback_retry_succeeded_count": int(self.fallback_retry_succeeded_count),
            "fallback_retry_failed_count": int(self.fallback_retry_failed_count),
            "stale_cache_reclassification_count": int(self.stale_cache_reclassification_count),
            "failed_surface_examples": list(self.failed_surface_examples),
            "classifier_backend": self.classifier_backend,
            "classifier_model_name": self.classifier_model_name,
            "backend_exception_message": self.backend_exception_message,
        }


@dataclass
class StrictPatternBatchClassificationResult:
    patterns: List[Optional[QuestionPattern]]
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    uncached_surface_count: int = 0
    llm_tagged_surface_count: int = 0
    failed_surface_count: int = 0
    initial_failed_surface_count: int = 0
    fallback_retry_attempted_count: int = 0
    fallback_retry_succeeded_count: int = 0
    fallback_retry_failed_count: int = 0
    stale_cache_reclassification_count: int = 0
    failed_surface_examples: List[Dict[str, str]] = field(default_factory=list)
    classifier_backend: Optional[str] = None
    classifier_model_name: Optional[str] = None
    backend_exception_message: Optional[str] = None


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
    question_pattern: Dict[str, Any] = field(default_factory=dict)
    gold_answers: List[str] = field(default_factory=list)
    retrieved_bridge_hit_count: int = 0
    kept_bridge_hit_count: int = 0
    exact_bridge_match_found: bool = False
    exact_bridge_match_kept: bool = False
    equivalent_bridge_match_found: bool = False
    equivalent_bridge_match_kept: bool = False
    equivalent_bridge_match_similarity: Optional[float] = None
    equivalent_bridge_match_bridge_id: Optional[str] = None
    retrieval_trace: Dict[str, Any] = field(default_factory=dict)

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
    pattern_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    low_score_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    doc_dominant_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bridge_helpful_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_batch_feedback_brief: Optional[BatchFeedbackBrief] = None
    pattern_last_seen_batch: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": int(self.total_questions),
            "batches_seen": int(self.batches_seen),
            "pattern_counts": dict(self.pattern_counts),
            "representative_queries": dict(self.representative_queries),
            "pattern_metadata": dict(self.pattern_metadata),
            "low_score_patterns": list(self.low_score_patterns.values()),
            "doc_dominant_patterns": list(self.doc_dominant_patterns.values()),
            "bridge_helpful_patterns": list(self.bridge_helpful_patterns.values()),
            "last_batch_feedback_brief": (
                self.last_batch_feedback_brief.as_dict() if self.last_batch_feedback_brief is not None else None
            ),
            "pattern_last_seen_batch": dict(self.pattern_last_seen_batch),
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


@dataclass
class RerankEvidenceResult:
    kept_items: List[Dict[str, Any]]
    scored_items: List[Dict[str, Any]]


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


def _pattern_row_from_pattern(
    pattern: QuestionPattern,
    count: int,
    representative_query: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    row = {
        "pattern": pattern.pattern_key,
        "pattern_key": pattern.pattern_key,
        "relation_label": pattern.relation_label,
        "domain": pattern.domain,
        "operation": pattern.operation,
        "answer_type": pattern.answer_type,
        "comparison_target": pattern.comparison_target,
        "count": int(count),
    }
    if representative_query:
        row["representative_query"] = representative_query
    row.update(extra)
    return row


def _normalize_pattern_row(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    pattern_key = _normalize_pattern_label(str(row.get("pattern_key") or row.get("pattern") or "other"))
    pattern = QuestionPattern(
        operation=_normalize_enum_label(row.get("operation"), QUESTION_PATTERN_OPERATIONS, "lookup"),
        answer_type=_normalize_enum_label(row.get("answer_type"), QUESTION_PATTERN_ANSWER_TYPES, "other"),
        domain=_normalize_enum_label(row.get("domain"), QUESTION_PATTERN_DOMAINS, "other"),
        relation_label=_normalize_relation_label(row.get("relation_label") or pattern_key.split(".", 1)[0]),
        comparison_target=_normalize_enum_label(
            row.get("comparison_target"),
            QUESTION_PATTERN_COMPARISON_TARGETS,
            "none",
        ),
        pattern_key=pattern_key,
        classifier_source=str(row.get("classifier_source") or "normalized"),
    )
    normalized_row = _pattern_row_from_pattern(
        pattern,
        count=int(row.get("count", 0) or 0),
        representative_query=str(row.get("representative_query", "") or "").strip(),
    )
    if "avg_score" in row:
        normalized_row["avg_score"] = round(float(row.get("avg_score", 0.0) or 0.0), 4)
    return pattern_key, normalized_row


def _get_query_pattern(result: QueryAnswerResult) -> QuestionPattern:
    pattern_payload = result.question_pattern or {}
    if pattern_payload:
        try:
            return normalize_question_pattern_payload(pattern_payload)
        except Exception:
            pass
    return infer_question_pattern_structured(result.question)


def build_current_batch_demand_sketch(
    items: Sequence[Any],
    top_k_patterns: int = 4,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    representatives: Dict[str, str] = {}
    metadata_by_pattern: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if isinstance(item, QueryAnswerResult):
            pattern = _get_query_pattern(item)
            question = item.question
        else:
            question = str(getattr(item, "question", "") or "")
            pattern = infer_question_pattern_structured(question)
        counts[pattern.pattern_key] = counts.get(pattern.pattern_key, 0) + 1
        representatives.setdefault(pattern.pattern_key, question)
        metadata_by_pattern.setdefault(pattern.pattern_key, pattern.as_dict())

    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[: max(1, int(top_k_patterns))]
    top_patterns = [
        _pattern_row_from_pattern(
            normalize_question_pattern_payload(metadata_by_pattern.get(pattern_key, {"pattern_key": pattern_key})),
            count=count,
            representative_query=representatives.get(pattern_key, ""),
        )
        for pattern_key, count in ranked
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
        _pattern_row_from_pattern(
            normalize_question_pattern_payload(state.pattern_metadata.get(pattern, {"pattern_key": pattern})),
            count=count,
            representative_query=state.representative_queries.get(pattern, ""),
        )
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
    pattern_metadata: Dict[str, Dict[str, Any]],
) -> None:
    for row in rows:
        pattern_key, normalized_row = _normalize_pattern_row(row)
        if not pattern_key:
            continue
        count = int(normalized_row.get("count", 0) or 0)
        if count <= 0:
            continue
        representative = str(normalized_row.get("representative_query", "")).strip() or representative_queries.get(pattern_key, "")
        stats = target.setdefault(
            pattern_key,
            {
                **normalized_row,
                "pattern": pattern_key,
                "pattern_key": pattern_key,
                "count": 0,
                "representative_query": representative,
            },
        )
        stats["count"] += count
        if not stats.get("representative_query") and representative:
            stats["representative_query"] = representative
        if "avg_score" in normalized_row:
            stats["avg_score"] = normalized_row["avg_score"]
        pattern_metadata.setdefault(
            pattern_key,
            {
                key: value
                for key, value in normalized_row.items()
                if key in {"pattern", "pattern_key", "relation_label", "domain", "operation", "answer_type", "comparison_target"}
            },
        )


def accumulate_query_experience_state(
    state: QueryExperienceState,
    batch_items: Sequence[BatchQueryItem],
    batch_result: QueryBatchResult,
) -> QueryExperienceState:
    for item, result in zip(batch_items, batch_result.query_results):
        state.total_questions += 1
        pattern = _get_query_pattern(result)
        state.pattern_counts[pattern.pattern_key] = state.pattern_counts.get(pattern.pattern_key, 0) + 1
        state.representative_queries.setdefault(pattern.pattern_key, item.question)
        state.pattern_metadata.setdefault(pattern.pattern_key, pattern.as_dict())
        state.pattern_last_seen_batch[pattern.pattern_key] = state.batches_seen

    feedback = batch_result.feedback_brief
    _accumulate_pattern_rows(
        state.low_score_patterns,
        feedback.low_score_patterns,
        state.representative_queries,
        state.pattern_metadata,
    )
    _accumulate_pattern_rows(
        state.doc_dominant_patterns,
        feedback.doc_dominant_patterns,
        state.representative_queries,
        state.pattern_metadata,
    )
    _accumulate_pattern_rows(
        state.bridge_helpful_patterns,
        feedback.bridge_helpful_patterns,
        state.representative_queries,
        state.pattern_metadata,
    )

    state.last_batch_feedback_brief = feedback
    state.batches_seen += 1
    return state


def build_batch_feedback_brief(results: Sequence[QueryAnswerResult], top_k_patterns: int = 4) -> BatchFeedbackBrief:
    demand_counts: Dict[str, int] = {}
    pattern_metadata: Dict[str, QuestionPattern] = {}
    low_score: Dict[str, Dict[str, Any]] = {}
    doc_dominant: Dict[str, Dict[str, Any]] = {}
    bridge_helpful: Dict[str, Dict[str, Any]] = {}

    for result in results:
        pattern = _get_query_pattern(result)
        pattern_key = pattern.pattern_key
        pattern_metadata.setdefault(pattern_key, pattern)
        bridge_count = int(result.answer_source_mix.get("bridge", 0))
        doc_count = int(result.answer_source_mix.get("doc_entity", 0)) + int(result.answer_source_mix.get("doc_direct_qa", 0))
        score = result.primary_score()

        demand_counts[pattern_key] = demand_counts.get(pattern_key, 0) + 1

        if score < 0.5:
            stats = low_score.setdefault(
                pattern_key,
                {
                    **_pattern_row_from_pattern(pattern, count=0, representative_query=result.question),
                    "avg_score_sum": 0.0,
                },
            )
            stats["count"] += 1
            stats["avg_score_sum"] += score

        if doc_count > 0 and bridge_count == 0:
            stats = doc_dominant.setdefault(
                pattern_key,
                _pattern_row_from_pattern(pattern, count=0, representative_query=result.question),
            )
            stats["count"] += 1

        if result.bridge_evidence_used and score >= 0.5:
            stats = bridge_helpful.setdefault(
                pattern_key,
                _pattern_row_from_pattern(pattern, count=0, representative_query=result.question),
            )
            stats["count"] += 1

    def _rank(items: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked_items = list(items.values())
        for item in ranked_items:
            if "avg_score_sum" in item and item.get("count"):
                item["avg_score"] = round(float(item["avg_score_sum"]) / float(item["count"]), 4)
                item.pop("avg_score_sum", None)
        return sorted(ranked_items, key=lambda row: (-int(row.get("count", 0)), str(row.get("pattern_key", row.get("pattern", "")))))[
            : max(1, int(top_k_patterns))
        ]

    high_demand = sorted(demand_counts.items(), key=lambda pair: (-pair[1], pair[0]))[: max(1, int(top_k_patterns))]
    demand_rows = [
        _pattern_row_from_pattern(
            pattern_metadata.get(pattern, QuestionPattern("lookup", "other", "other", "other", "none", pattern)),
            count=count,
        )
        for pattern, count in high_demand
    ]

    summary_lines: List[str] = []
    if demand_rows:
        summary_lines.append(
            "High demand: " + ", ".join(f"{row['pattern_key']} x{row['count']}" for row in demand_rows)
        )
    low_rows = _rank(low_score)
    if low_rows:
        summary_lines.append(
            "Weak coverage: " + ", ".join(f"{row['pattern_key']} x{row['count']}" for row in low_rows)
        )
    doc_rows = _rank(doc_dominant)
    if doc_rows:
        summary_lines.append(
            "Doc-backed more than bridge-backed: "
            + ", ".join(f"{row['pattern_key']} x{row['count']}" for row in doc_rows)
        )
    helpful_rows = _rank(bridge_helpful)
    if helpful_rows:
        summary_lines.append(
            "Bridge-helpful patterns: "
            + ", ".join(f"{row['pattern_key']} x{row['count']}" for row in helpful_rows)
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
        pattern_classifier: Optional[QuestionPatternClassifier] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        strict_pattern_tagging: bool = False,
    ) -> BridgeIngestResult:
        valid_bridges = [
            bridge
            for bridge in bridges
            if str(bridge.get("bridge_id", "")).strip()
            and str(bridge.get("question", "")).strip()
            and str(bridge.get("reverse_question", "")).strip()
        ]
        total_bridges_seen = len(valid_bridges)
        surfaces_total = total_bridges_seen * 2
        added = 0
        added_bridge_ids: List[str] = []
        cache_hits = 0
        cache_misses = 0
        llm_calls = 0
        uncached_surface_count = 0
        llm_tagged_surface_count = 0
        failed_surface_count = 0
        initial_failed_surface_count = 0
        fallback_retry_attempted_count = 0
        fallback_retry_succeeded_count = 0
        fallback_retry_failed_count = 0
        stale_cache_reclassification_count = 0
        failed_surface_examples: List[Dict[str, str]] = []
        classifier_backend = getattr(pattern_classifier, "classifier_backend", None) if pattern_classifier is not None else None
        classifier_model_name = getattr(pattern_classifier, "classifier_model_name", None) if pattern_classifier is not None else None
        backend_exception_message: Optional[str] = None
        classified_patterns: Dict[str, Tuple[QuestionPattern, QuestionPattern]] = {}

        def _emit_classification_event(status: str, completed: int) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "phase": "bridge_pattern_classification",
                    "status": status,
                    "completed": completed,
                    "total": total_bridges_seen,
                    "unit": "bridges",
                    "surfaces_total": surfaces_total,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "llm_calls": llm_calls,
                    "uncached_surface_count": uncached_surface_count,
                    "llm_tagged_surface_count": llm_tagged_surface_count,
                    "failed_surface_count": failed_surface_count,
                    "initial_failed_surface_count": initial_failed_surface_count,
                    "fallback_retry_attempted_count": fallback_retry_attempted_count,
                    "fallback_retry_succeeded_count": fallback_retry_succeeded_count,
                    "fallback_retry_failed_count": fallback_retry_failed_count,
                    "stale_cache_reclassification_count": stale_cache_reclassification_count,
                }
            )

        if progress_callback is not None:
            _emit_classification_event("started", 0)

        bridges_to_classify: List[Dict[str, Any]] = []
        surfaces_to_classify: List[str] = []
        for bridge in valid_bridges:
            bridge_id = str(bridge.get("bridge_id", "")).strip()
            if not bridge_id or bridge_id in self.records:
                continue
            question = str(bridge.get("question", "")).strip()
            reverse_question = str(bridge.get("reverse_question", "")).strip()
            if not question or not reverse_question:
                continue
            bridges_to_classify.append(bridge)
            surfaces_to_classify.extend([question, reverse_question])

        if bridges_to_classify:
            if strict_pattern_tagging:
                if pattern_classifier is None:
                    seen_failures: set[str] = set()
                    for question in surfaces_to_classify:
                        cache_key = _normalize_question_cache_key(question)
                        if cache_key in seen_failures:
                            continue
                        seen_failures.add(cache_key)
                        failed_surface_examples.append(
                            {
                                "question": str(question or ""),
                                "reason": "no_bridge_surface_classifier",
                            }
                        )
                    uncached_surface_count = len(seen_failures)
                    cache_misses = uncached_surface_count
                    failed_surface_count = uncached_surface_count
                    _emit_classification_event("completed", total_bridges_seen)
                    return BridgeIngestResult(
                        added_count=0,
                        added_bridge_ids=[],
                        total_bridges_seen=total_bridges_seen,
                        surfaces_total=surfaces_total,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        llm_calls=llm_calls,
                        uncached_surface_count=uncached_surface_count,
                        llm_tagged_surface_count=llm_tagged_surface_count,
                        failed_surface_count=failed_surface_count,
                        initial_failed_surface_count=initial_failed_surface_count,
                        fallback_retry_attempted_count=fallback_retry_attempted_count,
                        fallback_retry_succeeded_count=fallback_retry_succeeded_count,
                        fallback_retry_failed_count=fallback_retry_failed_count,
                        stale_cache_reclassification_count=stale_cache_reclassification_count,
                        failed_surface_examples=failed_surface_examples[:10],
                        classifier_backend=classifier_backend,
                        classifier_model_name=classifier_model_name,
                        backend_exception_message=backend_exception_message,
                    )
                strict_result = pattern_classifier.classify_many_strict_with_diagnostics(surfaces_to_classify)
                cache_hits = strict_result.cache_hits
                cache_misses = strict_result.cache_misses
                llm_calls = strict_result.llm_calls
                uncached_surface_count = strict_result.uncached_surface_count
                llm_tagged_surface_count = strict_result.llm_tagged_surface_count
                failed_surface_count = strict_result.failed_surface_count
                initial_failed_surface_count = strict_result.initial_failed_surface_count
                fallback_retry_attempted_count = strict_result.fallback_retry_attempted_count
                fallback_retry_succeeded_count = strict_result.fallback_retry_succeeded_count
                fallback_retry_failed_count = strict_result.fallback_retry_failed_count
                stale_cache_reclassification_count = strict_result.stale_cache_reclassification_count
                failed_surface_examples = list(strict_result.failed_surface_examples)
                classifier_backend = strict_result.classifier_backend or classifier_backend
                classifier_model_name = strict_result.classifier_model_name or classifier_model_name
                backend_exception_message = strict_result.backend_exception_message
                if failed_surface_count > 0:
                    _emit_classification_event("completed", total_bridges_seen)
                    return BridgeIngestResult(
                        added_count=0,
                        added_bridge_ids=[],
                        total_bridges_seen=total_bridges_seen,
                        surfaces_total=surfaces_total,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        llm_calls=llm_calls,
                        uncached_surface_count=uncached_surface_count,
                        llm_tagged_surface_count=llm_tagged_surface_count,
                        failed_surface_count=failed_surface_count,
                        initial_failed_surface_count=initial_failed_surface_count,
                        fallback_retry_attempted_count=fallback_retry_attempted_count,
                        fallback_retry_succeeded_count=fallback_retry_succeeded_count,
                        fallback_retry_failed_count=fallback_retry_failed_count,
                        stale_cache_reclassification_count=stale_cache_reclassification_count,
                        failed_surface_examples=failed_surface_examples[:10],
                        classifier_backend=classifier_backend,
                        classifier_model_name=classifier_model_name,
                        backend_exception_message=backend_exception_message,
                    )
                batched_patterns = [pattern for pattern in strict_result.patterns if pattern is not None]
            elif pattern_classifier is not None:
                batched_patterns, cache_hits, cache_misses, llm_calls = pattern_classifier.classify_many_with_diagnostics(
                    surfaces_to_classify
                )
                uncached_surface_count = cache_misses
            else:
                batched_patterns = [infer_question_pattern_structured(question) for question in surfaces_to_classify]
                cache_hits = 0
                cache_misses = len(surfaces_to_classify)
                llm_calls = 0
                uncached_surface_count = cache_misses
            for bridge, (forward_pattern, reverse_pattern) in zip(
                bridges_to_classify,
                zip(batched_patterns[0::2], batched_patterns[1::2]),
            ):
                bridge_id = str(bridge.get("bridge_id", "")).strip()
                if bridge_id:
                    classified_patterns[bridge_id] = (forward_pattern, reverse_pattern)

        for index, bridge in enumerate(valid_bridges, start=1):
            bridge_id = str(bridge.get("bridge_id", "")).strip()
            question = str(bridge.get("question", "")).strip()
            reverse_question = str(bridge.get("reverse_question", "")).strip()

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
                if bridge_id in classified_patterns:
                    forward_pattern, reverse_pattern = classified_patterns[bridge_id]
                else:
                    forward_pattern = infer_question_pattern_structured(question)
                    reverse_pattern = infer_question_pattern_structured(reverse_question)
                record = BridgeRecord(
                    bridge_id=bridge_id,
                    question=question,
                    reverse_question=reverse_question,
                    answer_text=forward_text,
                    reverse_answer_text=reverse_text,
                    source_docs=list(bridge.get("source_docs", [])),
                    confidence=float(bridge.get("confidence", 0.0) or 0.0),
                    pattern_tags=sorted({forward_pattern.pattern_key, reverse_pattern.pattern_key}),
                    forward_pattern=forward_pattern.as_dict(),
                    reverse_pattern=reverse_pattern.as_dict(),
                    batch_index=int(batch_index),
                    source_relationship=str(bridge.get("source_relationship", "") or ""),
                    source_entity=str(bridge.get("source_entity", "") or ""),
                    source_neighbor=str(bridge.get("source_neighbor", "") or ""),
                )
                self.records[bridge_id] = record
                added += 1
                added_bridge_ids.append(bridge_id)
            else:
                if forward_text:
                    record.answer_text = forward_text
                if reverse_text:
                    record.reverse_answer_text = reverse_text

            # Gate reverse indexing: only index when the reverse question exists
            # AND the reverse answer is a named entity that users would query by
            # name (person/organization). Skip reverse for date/place/number/
            # other/title — these make the answer an attribute that doesn't match
            # user query phrasing like "Who is French-American?" or "Who died on
            # 24 February 1761?".
            reverse_answer_type = str(record.reverse_pattern.get("answer_type", "")).strip().lower()
            index_reverse = (
                bool(record.reverse_question.strip())
                and reverse_answer_type in {"person", "organization"}
            )

            orientations = [("forward", question, record.answer_text)]
            if index_reverse:
                orientations.append(("reverse", record.reverse_question, record.reverse_answer_text))

            for orientation, query_text, answer_text in orientations:
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
                        question_pattern=dict(record.forward_pattern if orientation == "forward" else record.reverse_pattern),
                        source_docs=list(record.source_docs),
                    )
                )

            if progress_callback is not None and (
                index == 1 or index == total_bridges_seen or index % 10 == 0
            ):
                progress_callback(
                    {
                        "phase": "bridge_pattern_classification",
                        "status": "progress",
                        "completed": index,
                        "total": total_bridges_seen,
                        "unit": "bridges",
                        "surfaces_total": surfaces_total,
                        "cache_hits": cache_hits,
                        "cache_misses": cache_misses,
                        "llm_calls": llm_calls,
                        "uncached_surface_count": uncached_surface_count,
                        "llm_tagged_surface_count": llm_tagged_surface_count,
                        "failed_surface_count": failed_surface_count,
                        "stale_cache_reclassification_count": stale_cache_reclassification_count,
                    }
                )

        if progress_callback is not None:
            _emit_classification_event("completed", total_bridges_seen)

        if added > 0:
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "bridge_index_rebuild",
                        "status": "started",
                        "completed": 0,
                        "total": 1,
                        "unit": "rebuild",
                        "surface_count": len(self.surfaces),
                        "bm25_enabled": BM25Okapi is not None,
                        "embedding_enabled": embed_texts_fn is not None,
                    }
                )
            self._rebuild_indexes(embed_texts_fn)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "bridge_index_rebuild",
                        "status": "completed",
                        "completed": 1,
                        "total": 1,
                        "unit": "rebuild",
                        "surface_count": len(self.surfaces),
                        "bm25_enabled": BM25Okapi is not None,
                        "embedding_enabled": embed_texts_fn is not None,
                    }
                )
        elif progress_callback is not None:
            progress_callback(
                {
                    "phase": "bridge_index_rebuild",
                    "status": "started",
                    "completed": 0,
                    "total": 1,
                    "unit": "rebuild",
                    "surface_count": len(self.surfaces),
                    "bm25_enabled": BM25Okapi is not None,
                    "embedding_enabled": embed_texts_fn is not None,
                    "message": "No new bridges to index",
                }
            )
            progress_callback(
                {
                    "phase": "bridge_index_rebuild",
                    "status": "completed",
                    "completed": 1,
                    "total": 1,
                    "unit": "rebuild",
                    "surface_count": len(self.surfaces),
                    "bm25_enabled": BM25Okapi is not None,
                    "embedding_enabled": embed_texts_fn is not None,
                    "message": "No new bridges to index",
                }
            )
        return BridgeIngestResult(
            added_count=added,
            added_bridge_ids=added_bridge_ids,
            total_bridges_seen=total_bridges_seen,
            surfaces_total=surfaces_total,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            llm_calls=llm_calls,
            uncached_surface_count=uncached_surface_count,
            llm_tagged_surface_count=llm_tagged_surface_count,
            failed_surface_count=failed_surface_count,
            initial_failed_surface_count=initial_failed_surface_count,
            fallback_retry_attempted_count=fallback_retry_attempted_count,
            fallback_retry_succeeded_count=fallback_retry_succeeded_count,
            fallback_retry_failed_count=fallback_retry_failed_count,
            stale_cache_reclassification_count=stale_cache_reclassification_count,
            failed_surface_examples=failed_surface_examples[:10],
            classifier_backend=classifier_backend,
            classifier_model_name=classifier_model_name,
            backend_exception_message=backend_exception_message,
        )

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
                    "pattern_key": str(surface.question_pattern.get("pattern_key", "")),
                    "relation_label": str(surface.question_pattern.get("relation_label", "")),
                    "domain": str(surface.question_pattern.get("domain", "")),
                    "source_docs": list(surface.source_docs),
                    "score": round(score, 4),
                    "source_type": "bridge",
                    "evidence_text": surface.evidence_text(),
                    "rerank_text": surface.rerank_text(),
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

        pattern_weights: Dict[str, float] = {}
        relation_weights: Dict[str, float] = {}
        domain_weights: Dict[str, float] = {}
        answer_type_weights: Dict[str, float] = {}

        def _accumulate_target_weights(row: Dict[str, Any], multiplier: float) -> None:
            pattern_key, normalized_row = _normalize_pattern_row(row)
            count = float(normalized_row.get("count", 0) or 0)
            if count <= 0:
                return
            pattern_weights[pattern_key] = pattern_weights.get(pattern_key, 0.0) + 4.0 * multiplier * count
            relation_label = str(normalized_row.get("relation_label", "")).strip()
            if relation_label:
                relation_weights[relation_label] = relation_weights.get(relation_label, 0.0) + 2.0 * multiplier * count
            domain = str(normalized_row.get("domain", "")).strip()
            if domain:
                domain_weights[domain] = domain_weights.get(domain, 0.0) + 1.25 * multiplier * count
            answer_type = str(normalized_row.get("answer_type", "")).strip()
            if answer_type:
                answer_type_weights[answer_type] = answer_type_weights.get(answer_type, 0.0) + 0.75 * multiplier * count

        if isinstance(prior_query_demand_sketch, dict):
            for row in prior_query_demand_sketch.get("top_patterns", []):
                _accumulate_target_weights(row, multiplier=1.0)

        if prior_feedback_brief is not None:
            for row in prior_feedback_brief.high_demand_patterns:
                _accumulate_target_weights(row, multiplier=0.5)
            for row in prior_feedback_brief.low_score_patterns:
                _accumulate_target_weights(row, multiplier=2.0)
            for row in prior_feedback_brief.doc_dominant_patterns:
                _accumulate_target_weights(row, multiplier=1.5)
            for row in prior_feedback_brief.bridge_helpful_patterns:
                _accumulate_target_weights(row, multiplier=0.75)

        scored: List[Tuple[float, BridgeRecord]] = []
        for record in self.records.values():
            forward_pattern = normalize_question_pattern_payload(record.forward_pattern)
            pattern_score = pattern_weights.get(forward_pattern.pattern_key, 0.0)
            pattern_score += relation_weights.get(forward_pattern.relation_label, 0.0)
            pattern_score += domain_weights.get(forward_pattern.domain, 0.0)
            pattern_score += answer_type_weights.get(forward_pattern.answer_type, 0.0)
            score = (
                pattern_score
                + 1.5 * float(record.helpful_count)
                + 0.5 * float(record.retrieved_count)
                + max(0.0, float(record.confidence))
            )
            scored.append((score, record))

        exemplars: List[Dict[str, Any]] = []
        seen_patterns: set[str] = set()
        for _, record in sorted(scored, key=lambda item: (-item[0], item[1].bridge_id)):
            pattern_key = str(record.forward_pattern.get("pattern_key", "") or "")
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
                    "rerank_text": f"Q: {qa['question']} A: {answer_text}",
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
                "rerank_text": f"Q: {qa['question']} A: {answer_text}",
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


def _rounded_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def _apply_protected_equivalent_keep(
    ranking_order: np.ndarray,
    is_equivalent_bridge_match: np.ndarray,
    top_k: int,
) -> set[int]:
    if top_k <= 0:
        return set()

    kept_indices = set(int(idx) for idx in ranking_order[:top_k])
    equivalent_candidates = [int(idx) for idx in ranking_order if bool(is_equivalent_bridge_match[int(idx)])]
    if not equivalent_candidates:
        return kept_indices

    protected_idx = equivalent_candidates[0]
    if protected_idx in kept_indices:
        return kept_indices

    removable = [int(idx) for idx in ranking_order[:top_k][::-1] if int(idx) not in equivalent_candidates]
    if removable:
        kept_indices.remove(removable[0])
    elif kept_indices:
        kept_indices.remove(int(ranking_order[top_k - 1]))
    kept_indices.add(protected_idx)
    return kept_indices


def rerank_merged_evidence(
    question: str,
    evidence_items: Sequence[Dict[str, Any]],
    embed_texts_fn: Optional[Callable[[Sequence[str]], Optional[Sequence[np.ndarray]]]] = None,
    top_k: int = 12,
) -> RerankEvidenceResult:
    if top_k <= 0 or not evidence_items:
        return RerankEvidenceResult(kept_items=[], scored_items=[])

    base_scores = np.asarray([float(item.get("score", 0.0) or 0.0) for item in evidence_items], dtype=np.float32)
    embedding_similarity = np.zeros_like(base_scores)
    if embed_texts_fn is not None:
        query_matrix = _embed_texts(embed_texts_fn, [question])
        evidence_matrix = _embed_texts(
            embed_texts_fn,
            [str(item.get("rerank_text", item.get("evidence_text", "")) or "") for item in evidence_items],
        )
        if query_matrix is not None and evidence_matrix is not None:
            embedding_similarity = np.clip(_cosine_scores(query_matrix[0], evidence_matrix), 0.0, None)

    final_scores = embedding_similarity
    ranking_order = np.argsort(final_scores)[::-1]
    rank_after_merge = {int(idx): rank + 1 for rank, idx in enumerate(ranking_order)}
    is_equivalent_bridge_match = np.asarray(
        [
            str(item.get("source_type", "")) == "bridge" and float(final_scores[idx]) >= 0.85
            for idx, item in enumerate(evidence_items)
        ],
        dtype=bool,
    )
    kept_indices = _apply_protected_equivalent_keep(ranking_order, is_equivalent_bridge_match, top_k)

    scored_items: List[Dict[str, Any]] = []
    for idx in ranking_order:
        item_idx = int(idx)
        item = dict(evidence_items[item_idx])
        item["normalized_base_score"] = None
        item["semantic_score"] = _rounded_score(embedding_similarity[item_idx])
        item["embedding_similarity"] = _rounded_score(embedding_similarity[item_idx])
        item["final_score"] = _rounded_score(final_scores[item_idx])
        item["rank_before_merge"] = int(item.get("rank_before_merge", item_idx + 1))
        item["rank_after_merge"] = int(rank_after_merge[item_idx])
        item["is_equivalent_bridge_match"] = bool(is_equivalent_bridge_match[item_idx])
        item["protected_keep"] = bool(item_idx in kept_indices and is_equivalent_bridge_match[item_idx])
        item["kept"] = item_idx in kept_indices
        scored_items.append(item)

    kept_items = [dict(item) for item in scored_items if bool(item.get("kept"))]
    return RerankEvidenceResult(kept_items=kept_items, scored_items=scored_items)


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


def compute_guidance_entropy(guidance_payload: Optional[Dict[str, Any]]) -> float:
    """Shannon entropy of the demand distribution in the guidance payload.

    Lower entropy means the demand is concentrated on fewer patterns
    (overfitting signal).  Returns 0.0 when no demand profile exists.
    """
    import math

    if not guidance_payload:
        return 0.0
    profile = guidance_payload.get("query_demand_profile", [])
    counts = [int(p.get("count", 0)) for p in profile if int(p.get("count", 0)) > 0]
    if not counts:
        return 0.0
    total = sum(counts)
    probs = [c / total for c in counts]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def apply_demand_decay(
    state: QueryExperienceState,
    decay_rate: float,
    current_batch: int,
) -> None:
    """Decay pattern counts based on how many batches since they were last seen.

    This prevents old patterns from dominating the demand profile
    (the overfitting fix).  Mutates ``state.pattern_counts`` in place.

    Args:
        state: The query experience state to decay.
        decay_rate: Decay factor per batch (e.g. 0.5 = halve each batch).
                    0.0 means no decay (original behaviour).
        current_batch: Current batch index.
    """
    if decay_rate <= 0.0 or not state.pattern_counts:
        return
    last_seen = getattr(state, "pattern_last_seen_batch", {})
    for pattern_key in list(state.pattern_counts.keys()):
        age = current_batch - last_seen.get(pattern_key, 0)
        if age > 0:
            state.pattern_counts[pattern_key] = max(
                1, int(state.pattern_counts[pattern_key] * (decay_rate ** age))
            )


def build_query_demand_profile(state: QueryExperienceState) -> List[Dict[str, Any]]:
    """Build the full query pattern distribution with counts and success signals.

    Each entry represents a demanded pattern_key with its frequency and
    whether bridges for that pattern were helpful, low-scoring, or
    doc-dominant in prior batches.
    """
    if state.total_questions <= 0 or not state.pattern_counts:
        return []
    profile: List[Dict[str, Any]] = []
    for pattern_key, count in state.pattern_counts.items():
        meta = state.pattern_metadata.get(pattern_key, {})
        profile.append({
            "pattern_key": pattern_key,
            "relation_label": str(meta.get("relation_label", "")),
            "operation": str(meta.get("operation", "")),
            "answer_type": str(meta.get("answer_type", "")),
            "domain": str(meta.get("domain", "")),
            "count": int(count),
            "representative_query": str(state.representative_queries.get(pattern_key, "")),
            "bridge_helpful": pattern_key in state.bridge_helpful_patterns,
            "low_score": pattern_key in state.low_score_patterns,
            "doc_dominant": pattern_key in state.doc_dominant_patterns,
        })
    return sorted(profile, key=lambda x: (-x["count"], x["pattern_key"]))


def build_relationship_to_patterns(registry: BridgeRegistry) -> Dict[str, List[str]]:
    """Map source_relationship → list of pattern_keys it has produced.

    Uses the ``source_relationship`` field stamped on each bridge during
    generation to learn which GSW verb-phrase types produce which query
    patterns.
    """
    mapping: Dict[str, List[str]] = {}
    for record in registry.records.values():
        rel = (record.source_relationship or "").lower().strip()
        if not rel:
            continue
        patterns = mapping.setdefault(rel, [])
        fwd_pk = str(record.forward_pattern.get("pattern_key", ""))
        if fwd_pk and fwd_pk not in patterns:
            patterns.append(fwd_pk)
    return mapping


def build_curriculum_guidance_payload(
    prior_query_demand_sketch: Optional[Dict[str, Any]],
    prior_batch_feedback_brief: Optional[BatchFeedbackBrief],
    bridge_exemplars: Sequence[Dict[str, Any]],
    prior_batches_seen: int = 0,
    query_demand_profile: Optional[List[Dict[str, Any]]] = None,
    relationship_to_patterns: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    return {
        "prior_query_demand_sketch": prior_query_demand_sketch,
        "prior_batch_feedback_brief": prior_batch_feedback_brief.as_dict() if prior_batch_feedback_brief else None,
        "bridge_exemplars": list(bridge_exemplars),
        "prior_batches_seen": int(prior_batches_seen),
        "query_demand_profile": query_demand_profile or [],
        "relationship_to_patterns": relationship_to_patterns or {},
    }
