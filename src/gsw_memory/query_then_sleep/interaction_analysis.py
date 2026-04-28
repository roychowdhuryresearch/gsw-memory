from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gsw_memory.sleep_time.curriculum import QuestionPatternClassifier, infer_question_pattern_structured

from .prompts import INTERACTION_GUIDANCE_SUMMARY_SYSTEM_PROMPT
from .transport import StageTransport
from .types import InteractionGuidanceSummary, QueryInteractionTrace


class _LLMQuestionPatternAdapter:
    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        reasoning_effort: str = "medium",
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.base_url = str(base_url).strip() if base_url else None
        self.event_callback = event_callback
        self.transport = StageTransport(
            self.model_name,
            base_url=self.base_url,
            reasoning_effort=reasoning_effort,
        )

    def __call__(self, question: str) -> Dict[str, Any]:
        system_prompt = (
            "Classify the main relation demand in this query fragment. "
            "Return strict JSON only with key: relation_label. "
            "Use a short snake_case relation label."
        )
        payload, response = self.transport.generate_json(
            instructions=system_prompt,
            user_text=f"Fragment: {question}",
            max_output_tokens=300,
            temperature=0.0,
            stage="analysis_classifier",
        )
        if self.event_callback is not None:
            log_payload = response.as_log_dict()
            log_payload.update({"stage": "analysis_classifier", "fragment": question})
            self.event_callback("model_response", log_payload)
        payload["classifier_source"] = "llm"
        return payload


def _build_pattern_classifier(
    *,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    cache_path: Optional[str] = None,
    reasoning_effort: str = "medium",
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> QuestionPatternClassifier:
    classify_question_fn = None
    if str(model_name or "").strip():
        try:
            classify_question_fn = _LLMQuestionPatternAdapter(
                str(model_name),
                base_url=base_url,
                reasoning_effort=reasoning_effort,
                event_callback=event_callback,
            )
        except Exception as exc:
            if event_callback is not None:
                event_callback(
                    "analysis_classifier_fallback",
                    {
                        "reason": "adapter_init_failed",
                        "error": str(exc),
                        "model_name": str(model_name or ""),
                    },
                )
            classify_question_fn = None
    return QuestionPatternClassifier(
        classify_question_fn=classify_question_fn,
        persistent_cache_path=cache_path,
    )


def _relation_label(
    classifier: QuestionPatternClassifier,
    text: str,
    *,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "other"
    simple_label = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    if simple_label and simple_label == normalized.lower():
        return simple_label
    try:
        return classifier.classify(normalized).relation_label
    except Exception as exc:
        fallback = infer_question_pattern_structured(normalized).relation_label
        if event_callback is not None:
            event_callback(
                "analysis_classifier_fallback",
                {
                    "reason": "classify_failed",
                    "error": str(exc),
                    "fragment": normalized,
                    "fallback_relation_label": fallback,
                },
            )
        return fallback


def _extract_entity_relation_target(
    text: str,
    classifier: QuestionPatternClassifier,
    *,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", "other"

    relation = ""
    entity = ""
    match = re.search(r"(?P<relation>[a-zA-Z_ ]+)\s+for\s+(?P<entity>.+)$", raw)
    if match:
        relation = str(match.group("relation") or "").strip()
        entity = str(match.group("entity") or "").strip()
    elif "::" in raw:
        left, right = raw.split("::", 1)
        entity = left.strip()
        relation = right.strip()
    else:
        relation = raw
    relation_label = _relation_label(classifier, relation, event_callback=event_callback)
    if not entity and relation != raw:
        entity = raw
    return entity, relation_label


def _iter_sub_question_rows(trace: QueryInteractionTrace) -> Iterable[Tuple[str, bool, str]]:
    for row in trace.sub_questions:
        label_source = row.relation_hint or row.searched_for or row.sub_question
        yield label_source, bool(row.answer_found), row.sub_question


def _describe_bridge_chain(question: str, sub_question: str, relation_label: str) -> str:
    question_lower = str(question or "").strip().lower()
    sub_question_lower = str(sub_question or "").strip().lower()
    relation = str(relation_label or "").strip().lower()
    combined = " ".join(part for part in (question_lower, sub_question_lower, relation) if part)

    if "spouse" in combined and ("director" in combined or "film" in combined or "movie" in combined):
        return "spouse links for film-director chains"
    if any(token in combined for token in ("grandfather", "grandmother", "maternal", "paternal", "genealogy")):
        return "grandparent bridge steps in genealogy questions"
    if any(token in combined for token in ("father", "mother", "parent")):
        return "parent bridge steps in genealogy questions"
    if any(token in combined for token in ("award", "grammy", "performer", "song", "album")):
        return "award attribution links for performer-to-award chains"
    if any(token in combined for token in ("death_place", "died", "place of death")):
        return "death-place follow-up links in historical person chains"
    if any(token in combined for token in ("birth", "birthplace", "born")):
        return "birth-place follow-up links in person lookup chains"
    if any(token in combined for token in ("director", "composer")) and ("film" in combined or "movie" in combined):
        return "film-creator identification links"
    readable = relation.replace("_", " ").strip()
    return f"{readable} links in multi-hop chains" if readable else "multi-hop bridge links"


def _build_bridge_chain_profiles(
    traces: Sequence[QueryInteractionTrace],
    classifier: QuestionPatternClassifier,
    *,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    chain_counts: Dict[str, Dict[str, Any]] = {}
    success_counts: Counter[str] = Counter()

    for trace in traces:
        for row in trace.sub_questions:
            label_source = row.relation_hint or row.searched_for or row.sub_question
            relation = _relation_label(classifier, label_source, event_callback=event_callback)
            chain_label = _describe_bridge_chain(trace.question, row.sub_question, relation)
            if chain_label not in chain_counts:
                chain_counts[chain_label] = {
                    "chain_label": chain_label,
                    "relation_labels": [],
                    "asked": 0,
                    "found": 0,
                    "missing": 0,
                    "example_question": trace.question,
                }
            chain_row = chain_counts[chain_label]
            chain_row["asked"] += 1
            if relation and relation not in chain_row["relation_labels"]:
                chain_row["relation_labels"].append(relation)
            if row.answer_found:
                chain_row["found"] += 1
                success_counts[chain_label] += 1
            else:
                chain_row["missing"] += 1

    bridge_chain_profile = list(chain_counts.values())
    bridge_chain_profile.sort(key=lambda row: (-int(row["missing"]), -int(row["asked"]), str(row["chain_label"])))

    undercovered_chain_families = [
        dict(row)
        for row in bridge_chain_profile
        if int(row.get("missing", 0) or 0) > 0
    ][:5]

    deprioritized_chain_families = [
        {
            "chain_label": row["chain_label"],
            "found": int(success_counts.get(row["chain_label"], 0)),
            "relation_labels": list(row.get("relation_labels", [])),
            "example_question": str(row.get("example_question", "") or ""),
        }
        for row in bridge_chain_profile
        if int(row.get("missing", 0) or 0) == 0 and int(row.get("found", 0) or 0) > 0
    ][:5]
    deprioritized_chain_families.sort(key=lambda row: (-int(row["found"]), str(row["chain_label"])))

    return bridge_chain_profile, undercovered_chain_families, deprioritized_chain_families


def _build_natural_language_summary(
    chain_rows: List[Dict[str, Any]],
    deprioritized_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
) -> str:
    if not chain_rows:
        return "No dominant bridge gaps were detected. Keep next-batch sleep exploration broad, but still focus on evidence-grounded connecting hops."

    top_missing = ", ".join(
        f"{row['chain_label']} ({row['missing']} missing)"
        for row in chain_rows[:3]
    )
    summary = (
        f"Recent query traces point to under-covered bridge chains: {top_missing}. "
        "Prioritize next-batch sleep generation toward those missing connecting hops rather than direct fact lookups."
    )
    if deprioritized_rows:
        covered = ", ".join(row["chain_label"] for row in deprioritized_rows[:2])
        summary += f" Direct coverage already looks stronger for {covered}, so those can be deprioritized."
    if target_rows:
        examples = "; ".join(
            str(row.get("question", "") or f"{row.get('entity', 'unknown')}::{row['relation_label']}")
            for row in target_rows[:2]
        )
        summary += f" Concrete bridge opportunities include {examples}."
    return summary


def _build_failed_questions_for_summary(
    traces: Sequence[QueryInteractionTrace],
) -> List[Dict[str, Any]]:
    """Extract raw failed-question details for the guidance summarizer."""
    failed: List[Dict[str, Any]] = []
    for trace in traces:
        f1 = float(trace.f1) if trace.f1 is not None else 0.0
        if f1 >= 0.8:
            continue

        # Grab the top auto-bridge retrieved for this question (if any)
        top_bridge_q = ""
        top_bridge_a = ""
        top_bridge_id = ""
        for tc in trace.tool_calls:
            tool_name = getattr(tc, "tool_name", None) or (tc.get("tool_name") if isinstance(tc, dict) else None)
            args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else None) or {}
            if tool_name == "get_bridge_evidence" and args.get("auto"):
                result = getattr(tc, "result", None) or (tc.get("result") if isinstance(tc, dict) else None)
                if isinstance(result, list) and result:
                    top = result[0]
                    if isinstance(top, dict):
                        top_bridge_q = str(top.get("question", "") or "")
                        top_bridge_a = str(top.get("answer_text", "") or "")
                        top_bridge_id = str(top.get("bridge_id", "") or "")
                break

        # Collect entities the agent searched (from search_entities calls)
        entity_chain: List[str] = []
        for tc in trace.tool_calls:
            tool_name = getattr(tc, "tool_name", None) or (tc.get("tool_name") if isinstance(tc, dict) else None)
            if tool_name == "search_entities":
                args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else None) or {}
                q = str(args.get("query", "") or "")
                if q:
                    entity_chain.append(q)

        gold = trace.gold_answers[0] if trace.gold_answers else ""

        failed.append(
            {
                "question_id": trace.question_id,
                "question": trace.question,
                "gold_answer": gold,
                "predicted_answer": trace.final_answer,
                "f1": round(f1, 3),
                "source_docs": list(trace.doc_ids),
                "top_bridge_retrieved": {
                    "bridge_id": top_bridge_id,
                    "question": top_bridge_q,
                    "answer": top_bridge_a,
                } if top_bridge_q else None,
                "entities_searched": entity_chain[:5],
            }
        )
    return failed


def _generate_natural_language_summary(
    *,
    relation_gap_profile: List[Dict[str, Any]],
    bridge_chain_profile: List[Dict[str, Any]],
    undercovered_chain_families: List[Dict[str, Any]],
    deprioritized_chain_families: List[Dict[str, Any]],
    successful_relation_profile: List[Dict[str, Any]],
    entity_gap_targets: List[Dict[str, Any]],
    supporting_examples: List[Dict[str, Any]],
    questions_answered: int,
    avg_f1: float,
    failed_questions: Optional[List[Dict[str, Any]]] = None,
    prior_guidance_summary: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    reasoning_effort: str = "medium",
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> tuple[str, Dict[str, Any]]:
    fallback = _build_natural_language_summary(
        undercovered_chain_families,
        deprioritized_chain_families,
        entity_gap_targets,
    )
    if not str(model_name or "").strip() and not str(base_url or "").strip():
        return fallback, {"summary_source": "heuristic_no_model"}

    try:
        transport = StageTransport(
            str(model_name or "").strip() or "gpt-4o-mini",
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
        summary_input = {
            "questions_answered": int(questions_answered),
            "avg_f1": float(avg_f1),
            # Raw failed-question details — each entry includes the top bridge
            # that was retrieved during the query phase. The summarizer LLM
            # judges anchor quality from these real bridges directly, without
            # any lexical pattern matching.
            "failed_questions": (failed_questions or [])[:5],
            # Prior batch's guidance — carry forward persistent lessons.
            # Empty string when this is the first batch.
            "prior_guidance_summary": str(prior_guidance_summary or "").strip(),
            # Keep existing abstract profiles as additional context
            "top_missing_relations": relation_gap_profile[:5],
            "bridge_chain_profile": bridge_chain_profile[:8],
            "undercovered_chain_families": undercovered_chain_families[:5],
            "deprioritized_chain_families": deprioritized_chain_families[:5],
            "top_successful_relations": successful_relation_profile[:3],
            "missing_targets": entity_gap_targets[:5],
            "supporting_examples": supporting_examples[:3],
        }
        payload, response = transport.generate_json(
            instructions=INTERACTION_GUIDANCE_SUMMARY_SYSTEM_PROMPT,
            user_text=json.dumps(summary_input, ensure_ascii=False),
            max_output_tokens=2000,
            temperature=0.0,
            stage="analysis_guidance_summary",
        )
        if event_callback is not None:
            log_payload = response.as_log_dict()
            log_payload.update({"stage": "analysis_guidance_summary"})
            event_callback("model_response", log_payload)
        summary = str(payload.get("natural_language_summary", "") or "").strip()
        if summary:
            return summary, {"summary_source": "llm"}
    except Exception as exc:
        if event_callback is not None:
            event_callback(
                "guidance_llm_fallback",
                {
                    "reason": "summary_generation_failed",
                    "error": str(exc),
                },
            )
            event_callback(
                "analysis_guidance_summary_fallback",
                {
                    "reason": "summary_generation_failed",
                    "error": str(exc),
                },
            )
        return fallback, {
            "summary_source": "heuristic_fallback",
            "guidance_llm_fallback": True,
            "guidance_llm_error": str(exc),
        }
    return fallback, {"summary_source": "heuristic_empty_llm_response"}


def analyze_query_interactions(
    traces: Sequence[QueryInteractionTrace],
    *,
    batch_index: int,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    cache_path: Optional[str] = None,
    reasoning_effort: str = "medium",
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    prior_guidance_summary: Optional[str] = None,
) -> InteractionGuidanceSummary:
    classifier = _build_pattern_classifier(
        model_name=model_name,
        base_url=base_url,
        cache_path=cache_path,
        reasoning_effort=reasoning_effort,
        event_callback=event_callback,
    )

    relation_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"asked": 0, "found": 0, "missing": 0})
    success_counts: Counter[str] = Counter()
    entity_gap_counter: Dict[Tuple[str, str], Dict[str, Any]] = {}
    supporting_examples: List[Dict[str, Any]] = []

    for trace in traces:
        for label_source, answer_found, sub_question in _iter_sub_question_rows(trace):
            relation = _relation_label(classifier, label_source, event_callback=event_callback)
            row = relation_counts[relation]
            row["asked"] += 1
            if answer_found:
                row["found"] += 1
                success_counts[relation] += 1
            else:
                row["missing"] += 1
                if len(supporting_examples) < 5:
                    supporting_examples.append(
                        {
                            "question": trace.question,
                            "sub_question": sub_question,
                            "missing_relation": relation,
                            "bridge_chain": _describe_bridge_chain(trace.question, sub_question, relation),
                        }
                    )

        for raw_gap in trace.missing_relations:
            entity, relation = _extract_entity_relation_target(
                raw_gap,
                classifier,
                event_callback=event_callback,
            )
            key = (entity, relation)
            if key not in entity_gap_counter:
                entity_gap_counter[key] = {
                    "entity": entity,
                    "relation_label": relation,
                    "count": 0,
                    "question_ids": [],
                    "question": trace.question,
                }
            entity_gap_counter[key]["count"] += 1
            if trace.question_id and trace.question_id not in entity_gap_counter[key]["question_ids"]:
                entity_gap_counter[key]["question_ids"].append(trace.question_id)

        for raw_found in trace.found_relations:
            relation = _relation_label(classifier, raw_found, event_callback=event_callback)
            success_counts[relation] += 1
            relation_counts[relation]["found"] += 1

    relation_gap_profile = [
        {
            "relation_label": relation,
            "asked": counts["asked"],
            "found": counts["found"],
            "missing": counts["missing"],
        }
        for relation, counts in relation_counts.items()
    ]
    relation_gap_profile.sort(key=lambda row: (-int(row["missing"]), -int(row["asked"]), row["relation_label"]))

    successful_relation_profile = [
        {"relation_label": relation, "found": count}
        for relation, count in success_counts.items()
    ]
    successful_relation_profile.sort(key=lambda row: (-int(row["found"]), row["relation_label"]))

    entity_gap_targets = list(entity_gap_counter.values())
    entity_gap_targets.sort(key=lambda row: (-int(row["count"]), str(row.get("entity", "")), str(row.get("relation_label", ""))))

    bridge_chain_profile, undercovered_chain_families, deprioritized_chain_families = _build_bridge_chain_profiles(
        traces,
        classifier,
        event_callback=event_callback,
    )

    avg_f1 = (
        sum(float(trace.f1) for trace in traces if trace.f1 is not None) /
        max(1, sum(1 for trace in traces if trace.f1 is not None))
    ) if traces else 0.0

    failed_questions_details = _build_failed_questions_for_summary(traces)

    natural_language_summary, summary_metadata = _generate_natural_language_summary(
        relation_gap_profile=relation_gap_profile,
        bridge_chain_profile=bridge_chain_profile,
        undercovered_chain_families=undercovered_chain_families,
        deprioritized_chain_families=deprioritized_chain_families,
        successful_relation_profile=successful_relation_profile,
        entity_gap_targets=entity_gap_targets,
        supporting_examples=supporting_examples,
        questions_answered=len(traces),
        avg_f1=float(avg_f1),
        failed_questions=failed_questions_details,
        prior_guidance_summary=prior_guidance_summary,
        model_name=model_name,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        event_callback=event_callback,
    )

    return InteractionGuidanceSummary(
        guidance_mode="interaction_v1",
        batch_index=int(batch_index),
        questions_answered=len(traces),
        avg_f1=float(avg_f1),
        relation_gap_profile=relation_gap_profile,
        bridge_chain_profile=bridge_chain_profile,
        undercovered_chain_families=undercovered_chain_families,
        deprioritized_chain_families=deprioritized_chain_families,
        entity_gap_targets=entity_gap_targets,
        successful_relation_profile=successful_relation_profile,
        missing_bridge_targets=entity_gap_targets[:10],
        natural_language_summary=natural_language_summary,
        supporting_examples=supporting_examples,
        metadata=summary_metadata,
    )
