import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import playground.simple_entity_search as simple_entity_search
from playground.sleep_time import run_bridge_test
from gsw_memory.sleep_time.curriculum import (
    BatchFeedbackBrief,
    BatchQueryItem,
    BridgeRegistry,
    QuestionPatternClassifier,
    QueryAnswerResult,
    QueryBatchResult,
    QueryExperienceState,
    accumulate_query_experience_state,
    build_batch_feedback_brief,
    build_curriculum_guidance_payload,
    build_current_batch_demand_sketch,
    build_prior_query_demand_sketch,
    infer_question_pattern_structured,
    infer_question_patterns,
    rerank_merged_evidence,
    split_curriculum_batches,
)


def _embed_texts(texts):
    vectors = []
    for text in texts:
        lowered = str(text).lower()
        vectors.append([
            1.0 if "mother" in lowered else 0.0,
            1.0 if "buried" in lowered or "burial" in lowered else 0.0,
            float(len(lowered.split())),
        ])
    return vectors


def _rerank_embed_texts(texts):
    vectors = []
    for text in texts:
        lowered = str(text).lower()
        vectors.append(
            [
                1.0 if "when did lothair ii's mother die" in lowered else 0.0,
                1.0 if "bertha the daughter of" in lowered else 0.0,
                1.0 if "20 march 851" in lowered else 0.0,
            ]
        )
    return vectors


def _equivalent_embed_texts(texts):
    mapping = {
        "when did lothair ii s mother die": [1.0, 0.0],
        "when did the mother of lothair ii die": [0.86, 0.5103],
        "q who was bertha the daughter of a lothair ii": [0.95, 0.3122],
    }
    vectors = []
    for text in texts:
        lowered = " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        vectors.append(mapping.get(lowered, [0.0, 1.0]))
    return vectors


def test_split_curriculum_batches_respects_seed_then_fixed_size():
    items = [BatchQueryItem(question=f"Question {idx}", doc_ids=[f"doc_{idx}"]) for idx in range(7)]

    batches = split_curriculum_batches(items, batch_size=2, seed_batch_size=3)

    assert [len(batch) for batch in batches] == [3, 2, 2]
    assert batches[0][0].question == "Question 0"
    assert batches[-1][-1].question == "Question 6"


def test_question_pattern_classifier_uses_llm_shape_and_caches_by_normalized_question():
    calls = []

    def _classify(question: str):
        calls.append(question)
        return {
            "operation": "compare",
            "answer_type": "title",
            "domain": "film",
            "relation_label": "release_date",
            "comparison_target": "earlier",
            "pattern_key": "release_date.compare.earlier.title",
        }

    classifier = QuestionPatternClassifier(classify_question_fn=_classify)

    first = classifier.classify("Which film was released first, Aas Ka Panchhi or Phoolwari?")
    second = classifier.classify("Which film was released first Aas Ka Panchhi or Phoolwari")

    assert first.pattern_key == "release_date.compare.earlier.title"
    assert first.domain == "film"
    assert first.answer_type == "title"
    assert second.pattern_key == first.pattern_key
    assert len(calls) == 1


def test_question_pattern_classifier_batches_and_persists_cache(tmp_path):
    calls = []

    def _classify_many(questions):
        calls.append(list(questions))
        payload_by_question = {}
        for question in questions:
            lowered = question.lower()
            if "released first" in lowered:
                payload_by_question[question] = {
                    "operation": "compare",
                    "answer_type": "title",
                    "domain": "film",
                    "relation_label": "release_date",
                    "comparison_target": "earlier",
                    "pattern_key": "release_date.compare.earlier.title",
                }
            else:
                payload_by_question[question] = {
                    "operation": "lookup",
                    "answer_type": "place",
                    "domain": "history_politics",
                    "relation_label": "burial_place",
                    "comparison_target": "none",
                    "pattern_key": "burial_place.lookup.place",
                }
        return payload_by_question

    cache_path = tmp_path / "surface_patterns.json"
    classifier = QuestionPatternClassifier(
        classify_questions_fn=_classify_many,
        persistent_cache_path=str(cache_path),
    )

    patterns, cache_hits, cache_misses, llm_calls = classifier.classify_many_with_diagnostics(
        [
            "Which film was released first, Aas Ka Panchhi or Phoolwari?",
            "Which film was released first Aas Ka Panchhi or Phoolwari",
            "Where was Mira buried?",
        ]
    )

    assert [pattern.pattern_key for pattern in patterns] == [
        "release_date.compare.earlier.title",
        "release_date.compare.earlier.title",
        "burial_place.lookup.place",
    ]
    assert cache_hits == 1
    assert cache_misses == 2
    assert llm_calls == 2
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert cache_path.exists()

    def _unexpected_batch_call(_questions):
        raise AssertionError("batched classifier should not be called when persistent cache is warm")

    reloaded_classifier = QuestionPatternClassifier(
        classify_questions_fn=_unexpected_batch_call,
        persistent_cache_path=str(cache_path),
    )
    reloaded_patterns, reloaded_hits, reloaded_misses, reloaded_llm_calls = (
        reloaded_classifier.classify_many_with_diagnostics(
            [
                "Which film was released first Aas Ka Panchhi or Phoolwari",
                "Where was Mira buried?",
            ]
        )
    )

    assert [pattern.pattern_key for pattern in reloaded_patterns] == [
        "release_date.compare.earlier.title",
        "burial_place.lookup.place",
    ]
    assert reloaded_hits == 2
    assert reloaded_misses == 0
    assert reloaded_llm_calls == 0


def test_resolve_curator_bridge_classifier_config_for_bedrock():
    args = SimpleNamespace(
        worker_model="bedrock/openai.gpt-oss-120b-1:0",
        model="unused",
        base_url=None,
    )

    config = run_bridge_test._resolve_curator_bridge_classifier_config(args)

    assert config == {
        "backend": "litellm",
        "model_name": "bedrock/converse/openai.gpt-oss-120b-1:0",
    }


def test_resolve_curator_bridge_classifier_config_for_openai():
    args = SimpleNamespace(
        worker_model="gpt-4o-mini",
        model="unused",
        base_url=None,
    )

    config = run_bridge_test._resolve_curator_bridge_classifier_config(args)

    assert config == {
        "backend": "openai",
        "model_name": "gpt-4o-mini",
    }


def test_resolve_curator_bridge_classifier_config_for_base_url():
    args = SimpleNamespace(
        worker_model="local-model",
        model="unused",
        base_url="http://127.0.0.1:8000/v1",
    )

    config = run_bridge_test._resolve_curator_bridge_classifier_config(args)

    assert config == {
        "backend": "litellm",
        "model_name": "openai/local-model",
    }


def test_classify_questions_with_curator_uses_resolved_backend_and_suppresses_progress(monkeypatch, tmp_path):
    captured = {}

    class _FakeOperator:
        def __init__(self, model_name, backend, generation_params, backend_params):
            captured["model_name"] = model_name
            captured["backend"] = backend
            captured["generation_params"] = dict(generation_params)
            captured["backend_params"] = dict(backend_params)

        def __call__(self, dataset, working_dir):
            captured["curator_disable_progress"] = os.environ.get("CURATOR_DISABLE_PROGRESS")
            captured["dataset"] = list(dataset)
            captured["working_dir"] = working_dir
            return [{"question": dataset[0]["question"], "pattern": {"pattern_key": "x.lookup.other"}}]

    monkeypatch.setattr(run_bridge_test, "_BridgeSurfacePatternOperator", _FakeOperator)
    original_progress = os.environ.get("CURATOR_DISABLE_PROGRESS")
    os.environ["CURATOR_DISABLE_PROGRESS"] = "0"
    try:
        rows = run_bridge_test._classify_questions_with_curator(
            SimpleNamespace(
                worker_model="bedrock/openai.gpt-oss-120b-1:0",
                model="unused",
                base_url=None,
            ),
            ["Where was the director of Gaby: A True Story born?"],
            working_dir=str(tmp_path),
        )
    finally:
        if original_progress is None:
            os.environ.pop("CURATOR_DISABLE_PROGRESS", None)
        else:
            os.environ["CURATOR_DISABLE_PROGRESS"] = original_progress

    assert captured["backend"] == "litellm"
    assert captured["model_name"] == "bedrock/converse/openai.gpt-oss-120b-1:0"
    assert captured["curator_disable_progress"] == "1"
    assert rows["Where was the director of Gaby: A True Story born?"]["pattern_key"] == "x.lookup.other"


def test_question_pattern_classifier_strict_batch_reclassifies_stale_non_llm_cache(tmp_path):
    cache_path = tmp_path / "bridge_surface_patterns.json"
    cache_path.write_text(
        json.dumps(
            {
                "where was mira buried": {
                    "operation": "lookup",
                    "answer_type": "place",
                    "domain": "history_politics",
                    "relation_label": "burial_place",
                    "comparison_target": "none",
                    "pattern_key": "burial_place.lookup.place",
                    "classifier_source": "heuristic_fallback",
                }
            }
        )
    )
    calls = []

    def _classify_many(questions):
        calls.append(list(questions))
        return {
            question: {
                "operation": "lookup",
                "answer_type": "place",
                "domain": "history_politics",
                "relation_label": "burial_place",
                "comparison_target": "none",
                "pattern_key": "burial_place.lookup.place",
                "classifier_source": "llm",
            }
            for question in questions
        }

    classifier = QuestionPatternClassifier(
        classify_questions_fn=_classify_many,
        persistent_cache_path=str(cache_path),
    )

    result = classifier.classify_many_strict_with_diagnostics(
        [
            "Where was Mira buried?",
            "Where was Mira buried",
        ]
    )

    assert result.cache_hits == 1
    assert result.cache_misses == 1
    assert result.uncached_surface_count == 1
    assert result.llm_tagged_surface_count == 1
    assert result.failed_surface_count == 0
    assert result.stale_cache_reclassification_count == 1
    assert len(calls) == 1
    assert result.patterns[0] is not None
    assert result.patterns[0].classifier_source == "llm"
    reloaded = json.loads(cache_path.read_text())
    assert reloaded["where was mira buried"]["classifier_source"] == "llm"


def test_strict_pattern_batch_classifier_recovers_missing_row_with_single_question_fallback(tmp_path):
    cache_path = tmp_path / "bridge_surface_patterns.json"
    questions = [f"Question {index}" for index in range(20)]
    missing_question = questions[-1]
    batch_calls = []
    fallback_calls = []

    def _classify_many(questions):
        batch_calls.append(list(questions))
        return {
            question: {
                "operation": "lookup",
                "answer_type": "date",
                "domain": "history_politics",
                "relation_label": "mother_death",
                "comparison_target": "none",
                "pattern_key": "mother_death.lookup.date",
                "classifier_source": "llm",
            }
            for question in questions[:-1]
        }

    def _classify_one(question):
        fallback_calls.append(question)
        return {
            "operation": "lookup",
            "answer_type": "person",
            "domain": "history_politics",
            "relation_label": "child_of",
            "comparison_target": "none",
            "pattern_key": "child_of.lookup.person",
            "classifier_source": "llm",
        }

    classifier = QuestionPatternClassifier(
        classify_question_fn=_classify_one,
        classify_questions_fn=_classify_many,
        persistent_cache_path=str(cache_path),
    )

    result = classifier.classify_many_strict_with_diagnostics(
        questions
    )

    assert batch_calls == [questions]
    assert fallback_calls == [missing_question]
    assert result.uncached_surface_count == 20
    assert result.initial_failed_surface_count == 1
    assert result.fallback_retry_attempted_count == 1
    assert result.fallback_retry_succeeded_count == 1
    assert result.fallback_retry_failed_count == 0
    assert result.llm_tagged_surface_count == 20
    assert result.failed_surface_count == 0
    assert result.llm_calls == 21
    assert len([pattern for pattern in result.patterns if pattern is not None]) == 20
    reloaded = json.loads(cache_path.read_text())
    assert reloaded["question 0"]["classifier_source"] == "llm"
    assert reloaded["question 19"]["classifier_source"] == "llm"


def test_strict_pattern_batch_classifier_recovers_non_llm_row_with_single_question_fallback():
    questions = [f"Where was Mira buried {index}?" for index in range(20)]
    repaired_question = questions[-1]
    fallback_calls = []

    def _classify_many(_questions):
        return {
            question: {
                "operation": "lookup",
                "answer_type": "place",
                "domain": "history_politics",
                "relation_label": "burial_place",
                "comparison_target": "none",
                "pattern_key": "burial_place.lookup.place",
                "classifier_source": "heuristic_fallback" if question == repaired_question else "llm",
            }
            for question in questions
        }

    def _classify_one(question):
        fallback_calls.append(question)
        return {
            "operation": "lookup",
            "answer_type": "place",
            "domain": "history_politics",
            "relation_label": "burial_place",
            "comparison_target": "none",
            "pattern_key": "burial_place.lookup.place",
            "classifier_source": "llm",
        }

    classifier = QuestionPatternClassifier(
        classify_question_fn=_classify_one,
        classify_questions_fn=_classify_many,
    )

    result = classifier.classify_many_strict_with_diagnostics(questions)

    assert fallback_calls == [repaired_question]
    assert result.initial_failed_surface_count == 1
    assert result.fallback_retry_attempted_count == 1
    assert result.fallback_retry_succeeded_count == 1
    assert result.fallback_retry_failed_count == 0
    assert result.failed_surface_count == 0
    assert len([pattern for pattern in result.patterns if pattern is not None]) == 20
    assert result.patterns[-1] is not None
    assert result.patterns[-1].classifier_source == "llm"


def test_strict_pattern_batch_classifier_does_not_fallback_on_systemic_batch_failure():
    fallback_calls = []

    def _raise_backend_error(_questions):
        raise ValueError("curator backend mismatch")

    def _classify_one(question):
        fallback_calls.append(question)
        return {
            "operation": "lookup",
            "answer_type": "place",
            "domain": "history_politics",
            "relation_label": "burial_place",
            "comparison_target": "none",
            "pattern_key": "burial_place.lookup.place",
            "classifier_source": "llm",
        }

    classifier = QuestionPatternClassifier(
        classify_question_fn=_classify_one,
        classify_questions_fn=_raise_backend_error,
        classifier_backend="litellm",
        classifier_model_name="bedrock/converse/openai.gpt-oss-120b-1:0",
    )

    result = classifier.classify_many_strict_with_diagnostics(
        [
            "Where was Mira buried?",
            "When did Mira's mother die?",
        ]
    )

    assert fallback_calls == []
    assert result.initial_failed_surface_count == 2
    assert result.fallback_retry_attempted_count == 0
    assert result.fallback_retry_succeeded_count == 0
    assert result.fallback_retry_failed_count == 0
    assert result.failed_surface_count == 2
    assert result.backend_exception_message == "curator backend mismatch"


def test_bridge_registry_search_and_exemplar_selection_use_patterns_and_usage():
    registry = BridgeRegistry()
    classifier = QuestionPatternClassifier(
        classify_question_fn=lambda question: {
            "operation": "compare" if "released first" in question.lower() else "lookup",
            "answer_type": "title" if "released first" in question.lower() else ("place" if "where" in question.lower() else "date"),
            "domain": "film" if "released" in question.lower() else ("family" if "mother" in question.lower() else "other"),
            "relation_label": "release_date" if "released" in question.lower() else ("maternal_parent_death_date" if "mother" in question.lower() and "die" in question.lower() else "burial_place"),
            "comparison_target": "earlier" if "released first" in question.lower() else "none",
        }
    )
    mother_pattern = classifier.classify("When did Mira's mother die?")
    bridges = [
        {
            "bridge_id": "bridge_mother",
            "question": "When did Mira's mother die?",
            "reverse_question": "Whose mother died in 912?",
            "source_docs": ["doc_0", "doc_1"],
            "confidence": 0.92,
            "resolved_entities": {
                "answers": [{"name": "912"}],
                "reverse_answers": [{"name": "Mira"}],
            },
        },
        {
            "bridge_id": "bridge_burial",
            "question": "Where was Mira buried?",
            "reverse_question": "Who was buried in Abbey X?",
            "source_docs": ["doc_0", "doc_2"],
            "confidence": 0.88,
            "resolved_entities": {
                "answers": [{"name": "Abbey X"}],
                "reverse_answers": [{"name": "Mira"}],
            },
        },
    ]

    ingest_result = registry.ingest_bridges(
        bridges,
        batch_index=0,
        embed_texts_fn=_embed_texts,
        pattern_classifier=classifier,
    )
    registry.record_query_usage({"bridge_mother"}, {"bridge_mother"}, batch_index=0)

    results = registry.search("When did Mira's mother die?", top_k=2, embed_texts_fn=_embed_texts)
    exemplars = registry.select_exemplars(
        prior_query_demand_sketch={
            "top_patterns": [
                {
                    "pattern": mother_pattern.pattern_key,
                    "pattern_key": mother_pattern.pattern_key,
                    "relation_label": mother_pattern.relation_label,
                    "domain": mother_pattern.domain,
                    "operation": mother_pattern.operation,
                    "answer_type": mother_pattern.answer_type,
                    "comparison_target": mother_pattern.comparison_target,
                    "count": 2,
                    "representative_query": "When did Mira's mother die?",
                }
            ]
        },
        prior_feedback_brief=BatchFeedbackBrief(
            high_demand_patterns=[{"pattern": mother_pattern.pattern_key, "pattern_key": mother_pattern.pattern_key, "relation_label": mother_pattern.relation_label, "domain": mother_pattern.domain, "operation": mother_pattern.operation, "answer_type": mother_pattern.answer_type, "comparison_target": mother_pattern.comparison_target, "count": 2}],
            low_score_patterns=[{"pattern": mother_pattern.pattern_key, "pattern_key": mother_pattern.pattern_key, "relation_label": mother_pattern.relation_label, "domain": mother_pattern.domain, "operation": mother_pattern.operation, "answer_type": mother_pattern.answer_type, "comparison_target": mother_pattern.comparison_target, "count": 1, "representative_query": "When did Mira's mother die?"}],
            doc_dominant_patterns=[],
            bridge_helpful_patterns=[],
            summary_lines=[f"High demand: {mother_pattern.pattern_key} x2"],
        ),
        top_k=1,
    )

    assert ingest_result.added_count == 2
    assert results
    assert results[0]["bridge_id"] == "bridge_mother"
    assert results[0]["source_type"] == "bridge"
    assert results[0]["evidence_text"].startswith("Q: When did Mira's mother die?")
    assert len(exemplars) == 1
    assert exemplars[0]["bridge_id"] == "bridge_mother"


def test_bridge_registry_ingest_reports_progress_and_cache_accounting():
    registry = BridgeRegistry()
    progress_events = []
    classifier = QuestionPatternClassifier(
        classify_question_fn=lambda question: {
            "operation": "lookup",
            "answer_type": "person" if "whose child" in question.lower() else ("date" if "die" in question.lower() else "person"),
            "domain": "history_politics",
            "relation_label": "child_of" if "whose child" in question.lower() else ("mother_death" if "die" in question.lower() else "child_of"),
            "comparison_target": "none",
        }
    )
    bridges = [
        {
            "bridge_id": "bridge_a",
            "question": "When did Mira's mother die?",
            "reverse_question": "Whose child died in 912?",
            "source_docs": ["doc_0", "doc_1"],
            "confidence": 0.9,
            "resolved_entities": {
                "answers": [{"name": "912"}],
                "reverse_answers": [{"name": "Mira"}],
            },
        },
        {
            "bridge_id": "bridge_b",
            "question": "When did Mira's mother die?",
            "reverse_question": "Whose child died in 912?",
            "source_docs": ["doc_2", "doc_3"],
            "confidence": 0.8,
            "resolved_entities": {
                "answers": [{"name": "912"}],
                "reverse_answers": [{"name": "Mira"}],
            },
        },
    ]

    ingest_result = registry.ingest_bridges(
        bridges,
        batch_index=0,
        embed_texts_fn=_embed_texts,
        pattern_classifier=classifier,
        progress_callback=progress_events.append,
    )

    assert ingest_result.added_count == 2
    assert ingest_result.added_bridge_ids == ["bridge_a", "bridge_b"]
    assert ingest_result.total_bridges_seen == 2
    assert ingest_result.surfaces_total == 4
    assert ingest_result.cache_hits == 2
    assert ingest_result.cache_misses == 2
    assert ingest_result.llm_calls == 2
    assert progress_events[0]["phase"] == "bridge_pattern_classification"
    assert progress_events[0]["status"] == "started"
    assert any(
        event["phase"] == "bridge_pattern_classification" and event["status"] == "progress"
        for event in progress_events
    )
    assert progress_events[-1]["phase"] == "bridge_index_rebuild"
    assert progress_events[-1]["status"] == "completed"


def test_rerank_merged_evidence_prefers_bridge_question_surface_for_exact_match():
    query = "When did Lothair Ii's mother die?"
    evidence_items = [
        {
            "source_type": "bridge",
            "bridge_id": "bridge_exact",
            "question": "When did Lothair II's mother die?",
            "score": 0.25,
            "evidence_text": "Q: Who is the son of the person who died on 20 March 851? A: Lothair II",
            "rerank_text": "When did Lothair II's mother die?",
        },
        {
            "source_type": "doc_entity",
            "doc_id": "doc_9",
            "question": "Who was Bertha the daughter of?",
            "score": 1.0,
            "evidence_text": "Q: Who was Bertha the daughter of? A: Lothair II",
            "rerank_text": "Q: Who was Bertha the daughter of? A: Lothair II",
        },
    ]

    rerank_result = rerank_merged_evidence(query, evidence_items, embed_texts_fn=_rerank_embed_texts, top_k=2)
    reranked = rerank_result.kept_items

    assert reranked[0]["source_type"] == "bridge"
    assert reranked[0]["bridge_id"] == "bridge_exact"
    assert reranked[0]["final_score"] > reranked[1]["final_score"]
    assert rerank_result.scored_items[0]["rank_after_merge"] == 1
    assert rerank_result.scored_items[0]["kept"] is True
    assert rerank_result.scored_items[0]["normalized_base_score"] is None
    assert rerank_result.scored_items[0]["embedding_similarity"] == rerank_result.scored_items[0]["final_score"]
    assert rerank_result.scored_items[0]["semantic_score"] == rerank_result.scored_items[0]["final_score"]


def test_feedback_brief_and_guidance_payload_prioritize_low_score_doc_dominant_patterns():
    mother_pattern = infer_question_pattern_structured("When did Mira's mother die?")
    burial_pattern = infer_question_pattern_structured("Where was Mira buried?")
    results = [
        QueryAnswerResult(
            question_id="q1",
            question="When did Mira's mother die?",
            predicted_answer="",
            full_response="",
            exact_match=0.0,
            f1=0.0,
            bridge_evidence_used=False,
            answer_source_mix={"doc_entity": 2},
            top_evidence=[],
            patterns=[mother_pattern.pattern_key],
            question_pattern=mother_pattern.as_dict(),
            gold_answers=["912"],
        ),
        QueryAnswerResult(
            question_id="q2",
            question="Where was Mira buried?",
            predicted_answer="Abbey X",
            full_response="Answer: Abbey X",
            exact_match=1.0,
            f1=1.0,
            bridge_evidence_used=True,
            answer_source_mix={"bridge": 1},
            top_evidence=[],
            patterns=[burial_pattern.pattern_key],
            question_pattern=burial_pattern.as_dict(),
            gold_answers=["Abbey X"],
        ),
    ]

    sketch = build_current_batch_demand_sketch(
        [
            BatchQueryItem(question="When did Mira's mother die?", doc_ids=["doc_0", "doc_1"]),
            BatchQueryItem(question="Where was Mira buried?", doc_ids=["doc_0", "doc_2"]),
        ]
    )
    brief = build_batch_feedback_brief(results)
    payload = build_curriculum_guidance_payload(
        prior_query_demand_sketch=sketch,
        prior_batch_feedback_brief=brief,
        bridge_exemplars=[{"bridge_id": "bridge_burial", "question": "Where was Mira buried?", "reverse_question": "Who was buried in Abbey X?", "pattern_key": burial_pattern.pattern_key, "domain": burial_pattern.domain, "pattern_tags": [burial_pattern.pattern_key]}],
        prior_batches_seen=1,
    )

    assert sketch["top_patterns"]
    assert sketch["top_patterns"][0]["pattern"] in {mother_pattern.pattern_key, burial_pattern.pattern_key}
    assert any(row["pattern"] == mother_pattern.pattern_key for row in brief.low_score_patterns)
    assert any(row["pattern"] == mother_pattern.pattern_key for row in brief.doc_dominant_patterns)
    assert any(row["pattern"] == burial_pattern.pattern_key for row in brief.bridge_helpful_patterns)
    assert payload["prior_query_demand_sketch"] == sketch
    assert payload["prior_batch_feedback_brief"]["summary_lines"] == brief.summary_lines
    assert payload["bridge_exemplars"][0]["bridge_id"] == "bridge_burial"
    assert payload["prior_batches_seen"] == 1



def test_query_experience_state_accumulates_prior_demand_and_feedback():
    state = QueryExperienceState()
    assert build_prior_query_demand_sketch(state) is None

    death_pattern = infer_question_pattern_structured("When did Mira's mother die?")
    burial_pattern = infer_question_pattern_structured("Where was Mira buried?")
    mother_lookup_pattern = infer_question_pattern_structured("Who was Mira's mother?")

    batch_one_items = [
        BatchQueryItem(question="When did Mira's mother die?", doc_ids=["doc_0"]),
        BatchQueryItem(question="Where was Mira buried?", doc_ids=["doc_1"]),
    ]
    batch_one_results = [
        QueryAnswerResult(
            question_id="q1",
            question=batch_one_items[0].question,
            predicted_answer="",
            full_response="",
            exact_match=0.0,
            f1=0.0,
            bridge_evidence_used=False,
            answer_source_mix={"doc_entity": 1},
            top_evidence=[],
            patterns=[death_pattern.pattern_key],
            question_pattern=death_pattern.as_dict(),
            gold_answers=["912"],
        ),
        QueryAnswerResult(
            question_id="q2",
            question=batch_one_items[1].question,
            predicted_answer="Abbey X",
            full_response="Answer: Abbey X",
            exact_match=1.0,
            f1=1.0,
            bridge_evidence_used=True,
            answer_source_mix={"bridge": 1},
            top_evidence=[],
            patterns=[burial_pattern.pattern_key],
            question_pattern=burial_pattern.as_dict(),
            gold_answers=["Abbey X"],
        ),
    ]
    batch_one_brief = build_batch_feedback_brief(batch_one_results)
    batch_one_result = QueryBatchResult(
        batch_index=0,
        query_results=batch_one_results,
        overall_metrics={"ExactMatch": 0.5, "F1": 0.5, "scored_queries": 2},
        feedback_brief=batch_one_brief,
    )
    accumulate_query_experience_state(state, batch_one_items, batch_one_result)

    batch_two_items = [BatchQueryItem(question="Who was Mira's mother?", doc_ids=["doc_2"])]
    batch_two_results = [
        QueryAnswerResult(
            question_id="q3",
            question=batch_two_items[0].question,
            predicted_answer="Ada",
            full_response="Answer: Ada",
            exact_match=1.0,
            f1=1.0,
            bridge_evidence_used=True,
            answer_source_mix={"bridge": 1},
            top_evidence=[],
            patterns=[mother_lookup_pattern.pattern_key],
            question_pattern=mother_lookup_pattern.as_dict(),
            gold_answers=["Ada"],
        )
    ]
    batch_two_result = QueryBatchResult(
        batch_index=1,
        query_results=batch_two_results,
        overall_metrics={"ExactMatch": 1.0, "F1": 1.0, "scored_queries": 1},
        feedback_brief=build_batch_feedback_brief(batch_two_results),
    )
    accumulate_query_experience_state(state, batch_two_items, batch_two_result)

    sketch = build_prior_query_demand_sketch(state)

    assert state.total_questions == 3
    assert state.batches_seen == 2
    assert sketch is not None
    assert sketch["batches_seen"] == 2
    assert any(row["pattern"] == death_pattern.pattern_key and row["count"] == 1 for row in sketch["top_patterns"])
    assert any(row["pattern"] == mother_lookup_pattern.pattern_key and row["count"] == 1 for row in sketch["top_patterns"])
    assert any(row["pattern"] == burial_pattern.pattern_key for row in state.bridge_helpful_patterns.values())
    assert any(row["pattern"] == death_pattern.pattern_key for row in state.low_score_patterns.values())


class _StubEntitySearcher:
    def __init__(self, *args, **kwargs):
        self.gsw_by_doc_id = {}
        self.entities = []

    def _embed_query(self, texts):
        return [[0.0, float(len(str(text)))] for text in texts]


def test_run_doc_batch_sharded_merges_duplicates_and_records_metadata(monkeypatch, tmp_path):
    captured_guidance = []

    def _fake_run_doc_batch(
        *,
        batch_index,
        doc_ids,
        args,
        output_dir,
        guidance_payload=None,
        batch_output_dir=None,
        cache_root_dir=None,
    ):
        Path(batch_output_dir).mkdir(parents=True, exist_ok=True)
        captured_guidance.append(
            {
                "batch_index": batch_index,
                "doc_ids": list(doc_ids),
                "guidance_payload": guidance_payload,
                "batch_output_dir": batch_output_dir,
                "cache_root_dir": cache_root_dir,
            }
        )
        if set(doc_ids) == {"doc_0", "doc_1"}:
            bridges = [
                {
                    "bridge_id": "bridge_shared",
                    "question": "Who directed Film A?",
                    "reverse_question": "Which film did Director A direct?",
                    "source_docs": ["doc_0"],
                    "confidence": 0.4,
                    "resolved_entities": {
                        "answers": [{"name": "Director A"}],
                        "reverse_answers": [{"name": "Film A"}],
                    },
                },
                {
                    "bridge_id": "bridge_left",
                    "question": "Where was Director A born?",
                    "reverse_question": "Who was born in Place A?",
                    "source_docs": ["doc_1"],
                    "confidence": 0.7,
                    "resolved_entities": {
                        "answers": [{"name": "Place A"}],
                        "reverse_answers": [{"name": "Director A"}],
                    },
                },
            ]
        else:
            bridges = [
                {
                    "bridge_id": "bridge_shared",
                    "question": "Who directed Film A?",
                    "reverse_question": "Which film did Director A direct?",
                    "source_docs": ["doc_2"],
                    "confidence": 0.9,
                    "resolved_entities": {
                        "answers": [{"name": "Director A"}],
                        "reverse_answers": [{"name": "Film A"}],
                    },
                },
                {
                    "bridge_id": "bridge_right",
                    "question": "Where was Director B born?",
                    "reverse_question": "Who was born in Place B?",
                    "source_docs": ["doc_2"],
                    "confidence": 0.6,
                    "resolved_entities": {
                        "answers": [{"name": "Place B"}],
                        "reverse_answers": [{"name": "Director B"}],
                    },
                },
            ]
        return {
            "batch_index": batch_index,
            "doc_ids": list(doc_ids),
            "bridges": bridges,
            "run_results": {"accepted_bridges": len(bridges)},
        }

    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _fake_run_doc_batch)

    args = SimpleNamespace(
        curriculum_generation_parallel_workers=1,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
    )
    batch_items = [
        BatchQueryItem(question="Question A", doc_ids=["doc_0", "doc_1"], question_id="q0"),
        BatchQueryItem(question="Question B", doc_ids=["doc_2"], question_id="q1"),
    ]
    progress_events = []
    result = run_bridge_test._run_doc_batch_sharded(
        batch_index=0,
        batch_items=batch_items,
        args=args,
        batch_output_dir=str(tmp_path / "out" / "batch_0"),
        output_dir=str(tmp_path / "out"),
        guidance_payload={"prior_batches_seen": 1},
        progress_callback=progress_events.append,
    )

    assert result["generation_mode"] == "question_group_sharded"
    assert result["generation_executor"] == "thread"
    assert len(result["generation_shards"]) == 2
    assert {metadata["shard_id"] for metadata in result["generation_shards"]} == {0, 1}
    assert len(captured_guidance) == 2
    assert all(call["guidance_payload"] == {"prior_batches_seen": 1} for call in captured_guidance)
    assert (tmp_path / "out" / "batch_0" / "generation_shards" / "shard_0").exists()
    assert (tmp_path / "out" / "batch_0" / "generation_shards" / "shard_1").exists()

    merged_by_id = {bridge["bridge_id"]: bridge for bridge in result["bridges"]}
    assert set(merged_by_id) == {"bridge_left", "bridge_right", "bridge_shared"}
    assert merged_by_id["bridge_shared"]["confidence"] == 0.9
    assert set(merged_by_id["bridge_shared"]["source_docs"]) == {"doc_0", "doc_2"}

    phases = {(event["phase"], event["status"]) for event in progress_events}
    assert ("bridge_generation_shards", "started") in phases
    assert ("bridge_generation_shards", "progress") in phases
    assert ("bridge_generation_shards", "completed") in phases


def test_execute_generation_shard_tasks_runs_in_threads(monkeypatch):
    seen_thread_names = []

    def _fake_run_generation_shard_worker(task):
        seen_thread_names.append(threading.current_thread().name)
        time.sleep(0.01)
        return {"shard_id": int(task["shard_id"])}

    monkeypatch.setattr(run_bridge_test, "_run_generation_shard_worker", _fake_run_generation_shard_worker)

    results = run_bridge_test._execute_generation_shard_tasks(
        [{"shard_id": 0}, {"shard_id": 1}],
        max_workers=2,
    )

    assert [row["shard_id"] for row in results] == [0, 1]
    assert len(seen_thread_names) == 2
    assert all(name != threading.main_thread().name for name in seen_thread_names)


def test_run_curriculum_uses_only_prior_batch_guidance(monkeypatch, tmp_path):
    captured_guidance = []

    def _fake_run_doc_batch(*, batch_index, doc_ids, args, output_dir, guidance_payload=None):
        captured_guidance.append(guidance_payload)
        return {"bridges": [], "num_bridges": 0, "doc_ids": list(doc_ids), "batch_index": batch_index}

    def _fake_answer_query_batch(batch_index, batch_items, entity_searcher, bridge_registry, args):
        query_results = [
            QueryAnswerResult(
                question_id=item.question_id or f"q{batch_index}",
                question=item.question,
                predicted_answer="",
                full_response="",
                exact_match=0.0,
                f1=0.0,
                bridge_evidence_used=False,
                answer_source_mix={"doc_entity": 1},
                top_evidence=[],
                patterns=infer_question_patterns(item.question),
                gold_answers=[],
            )
            for item in batch_items
        ]
        return QueryBatchResult(
            batch_index=batch_index,
            query_results=query_results,
            overall_metrics={"ExactMatch": 0.0, "F1": 0.0, "scored_queries": len(query_results)},
            feedback_brief=build_batch_feedback_brief(query_results),
        )

    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _fake_run_doc_batch)
    monkeypatch.setattr(run_bridge_test, "answer_query_batch", _fake_answer_query_batch)
    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _StubEntitySearcher)

    args = SimpleNamespace(
        curriculum_batch_size=1,
        curriculum_seed_batch_size=1,
        bridge_prompt_exemplar_top_k=3,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
    )
    items = [
        BatchQueryItem(question="When did Mira's mother die?", doc_ids=["doc_0"], question_id="q0"),
        BatchQueryItem(question="Where was Mira buried?", doc_ids=["doc_1"], question_id="q1"),
    ]

    result = run_bridge_test.run_curriculum(items, args)

    assert len(captured_guidance) == 2
    assert captured_guidance[0]["prior_query_demand_sketch"] is None
    assert "current_batch_demand_sketch" not in captured_guidance[0]
    assert captured_guidance[0]["prior_batches_seen"] == 0
    assert captured_guidance[1]["prior_query_demand_sketch"] is not None
    prior_rows = captured_guidance[1]["prior_query_demand_sketch"]["top_patterns"]
    assert any(row["representative_query"] == "When did Mira's mother die?" for row in prior_rows)
    assert all(row["representative_query"] != "Where was Mira buried?" for row in prior_rows)
    assert result["batch_results"][0]["generation_guidance"]["prior_query_demand_sketch"] is None
    assert result["batch_results"][1]["generation_guidance"]["prior_query_demand_sketch"] is not None


def test_run_curriculum_uses_sharded_generation_when_enabled(monkeypatch, tmp_path):
    shard_calls = []

    def _fake_run_doc_batch_sharded(
        *,
        batch_index,
        batch_items,
        args,
        batch_output_dir,
        output_dir,
        guidance_payload=None,
        progress_callback=None,
    ):
        shard_calls.append(
            {
                "batch_index": batch_index,
                "question_ids": [item.question_id for item in batch_items],
                "guidance_payload": guidance_payload,
                "batch_output_dir": batch_output_dir,
                "output_dir": output_dir,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "bridge_generation_shards",
                    "status": "started",
                    "completed": 0,
                    "total": 1,
                    "unit": "shards",
                }
            )
            progress_callback(
                {
                    "phase": "bridge_generation_shards",
                    "status": "completed",
                    "completed": 1,
                    "total": 1,
                    "unit": "shards",
                    "merged_bridge_count": 0,
                }
            )
        return {
            "batch_index": batch_index,
            "doc_ids": sorted({doc_id for item in batch_items for doc_id in item.doc_ids}),
            "bridges": [],
            "run_results": {"total_bridges": 0, "shard_count": len(batch_items)},
            "generation_mode": "question_group_sharded",
            "generation_shards": [
                {
                    "shard_id": index,
                    "question_ids": [item.question_id],
                    "questions": [item.question],
                    "doc_ids": list(item.doc_ids),
                    "runtime_seconds": 0.1,
                    "accepted_bridge_count": 0,
                    "run_results": {},
                    "output_dir": str(Path(batch_output_dir) / "generation_shards" / f"shard_{index}"),
                }
                for index, item in enumerate(batch_items)
            ],
        }

    def _unexpected_run_doc_batch(**_kwargs):
        raise AssertionError(
            "run_doc_batch should not be used when sharded generation is enabled for a multi-question batch"
        )

    def _fake_answer_query_batch(batch_index, batch_items, entity_searcher, bridge_registry, args):
        query_results = [
            QueryAnswerResult(
                question_id=item.question_id or f"q{batch_index}",
                question=item.question,
                predicted_answer="",
                full_response="",
                exact_match=0.0,
                f1=0.0,
                bridge_evidence_used=False,
                answer_source_mix={"doc_entity": 1},
                top_evidence=[],
                patterns=infer_question_patterns(item.question),
                gold_answers=[],
            )
            for item in batch_items
        ]
        return QueryBatchResult(
            batch_index=batch_index,
            query_results=query_results,
            overall_metrics={"ExactMatch": 0.0, "F1": 0.0, "scored_queries": len(query_results)},
            feedback_brief=build_batch_feedback_brief(query_results),
        )

    monkeypatch.setattr(run_bridge_test, "_run_doc_batch_sharded", _fake_run_doc_batch_sharded)
    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _unexpected_run_doc_batch)
    monkeypatch.setattr(run_bridge_test, "answer_query_batch", _fake_answer_query_batch)
    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _StubEntitySearcher)

    args = SimpleNamespace(
        curriculum_batch_size=2,
        curriculum_seed_batch_size=2,
        bridge_prompt_exemplar_top_k=3,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
        curriculum_generation_parallel_enabled=True,
        curriculum_generation_parallel_workers=2,
        model=None,
        worker_model=None,
    )
    items = [
        BatchQueryItem(question="When did Mira's mother die?", doc_ids=["doc_0"], question_id="q0"),
        BatchQueryItem(question="Where was Mira buried?", doc_ids=["doc_1"], question_id="q1"),
    ]

    result = run_bridge_test.run_curriculum(items, args)

    assert len(shard_calls) == 1
    assert shard_calls[0]["question_ids"] == ["q0", "q1"]
    assert result["batch_results"][0]["generation_mode"] == "question_group_sharded"
    assert result["batch_results"][0]["generation_executor"] == "thread"
    assert len(result["batch_results"][0]["generation_shards"]) == 2


def test_build_answer_reconciler_allows_missing_bridge_verifier_arg():
    class _Args:
        model = "bedrock/openai.gpt-oss-120b-1:0"
        max_tokens = 1234
        reasoning_effort = "medium"
        base_url = None

    searcher = _StubEntitySearcher()
    reconciler = run_bridge_test._build_answer_reconciler(searcher, _Args())

    assert reconciler.model_name.endswith("openai.gpt-oss-120b-1:0")
    assert reconciler.bridge_verifier_enabled is False
    assert reconciler.bridge_verifier_model.endswith("openai.gpt-oss-120b-1:0")


def test_answer_query_batch_keeps_exact_bridge_and_records_diagnostics(monkeypatch):
    class _AnswerSearcher:
        gsw_by_doc_id = {"doc_0": object()}

        def _embed_query(self, texts):
            return _rerank_embed_texts(texts)

    class _BridgeRegistry:
        def __init__(self):
            self.used = None
            self.helpful = None

        def search(self, query, top_k, embed_texts_fn=None):
            return [
                {
                    "source_type": "bridge",
                    "bridge_id": "bridge_exact",
                    "question": "When did Lothair II's mother die?",
                    "score": 0.25,
                    "evidence_text": "Q: Who is the son of the person who died on 20 March 851? A: Lothair II",
                    "rerank_text": "When did Lothair II's mother die?",
                }
            ]

        def record_query_usage(self, used_bridge_ids, helpful_bridge_ids, batch_index):
            self.used = set(used_bridge_ids)
            self.helpful = set(helpful_bridge_ids)

    def _fake_collect_doc_evidence(entity_searcher, question, top_k_entities=10, top_k_direct_qa=10):
        return [
            {
                "source_type": "doc_entity",
                "doc_id": "doc_9",
                "question": "Who was Bertha the daughter of?",
                "score": 1.0,
                "evidence_text": "Q: Who was Bertha the daughter of? A: Lothair II",
                "rerank_text": "Q: Who was Bertha the daughter of? A: Lothair II",
            }
        ]

    monkeypatch.setattr(run_bridge_test, "collect_doc_evidence", _fake_collect_doc_evidence)
    monkeypatch.setattr(run_bridge_test, "_build_answer_reconciler", lambda entity_searcher, args: object())
    monkeypatch.setattr(
        run_bridge_test,
        "_parse_model_answer",
        lambda answer_agent, question, evidence: {"predicted_answer": "20 March 851", "full_response": "Answer: 20 March 851"},
    )

    args = SimpleNamespace(model="bedrock/openai.gpt-oss-120b-1:0", max_tokens=1024, bridge_query_top_k=1)
    batch_items = [
        BatchQueryItem(
            question="When did Lothair Ii's mother die?",
            doc_ids=["doc_0"],
            question_id="q0",
            gold_answer="20 March 851",
            answer_aliases=["20 March 851"],
        )
    ]
    bridge_registry = _BridgeRegistry()

    result = run_bridge_test.answer_query_batch(
        batch_index=0,
        batch_items=batch_items,
        entity_searcher=_AnswerSearcher(),
        bridge_registry=bridge_registry,
        args=args,
    )

    query_result = result.query_results[0]
    assert query_result.bridge_evidence_used is True
    assert query_result.retrieved_bridge_hit_count == 1
    assert query_result.kept_bridge_hit_count == 1
    assert query_result.exact_bridge_match_found is True
    assert query_result.exact_bridge_match_kept is True
    assert query_result.equivalent_bridge_match_found is True
    assert query_result.equivalent_bridge_match_kept is True
    assert query_result.equivalent_bridge_match_bridge_id == "bridge_exact"
    assert query_result.top_evidence[0]["bridge_id"] == "bridge_exact"
    assert query_result.top_evidence[0]["question"] == "When did Lothair II's mother die?"
    assert query_result.retrieval_trace["raw_bridge_hits"][0]["bridge_id"] == "bridge_exact"
    assert query_result.retrieval_trace["raw_bridge_hits"][0]["rank_before_merge"] == 1
    assert query_result.retrieval_trace["kept_candidates"][0]["bridge_id"] == "bridge_exact"
    assert query_result.retrieval_trace["merged_candidates"][0]["source_type"] == "bridge"
    assert query_result.retrieval_trace["merged_candidates"][0]["is_equivalent_bridge_match"] is True
    assert bridge_registry.used == {"bridge_exact"}
    assert bridge_registry.helpful == {"bridge_exact"}


def test_answer_query_batch_protects_equivalent_bridge_when_doc_scores_higher(monkeypatch):
    class _AnswerSearcher:
        gsw_by_doc_id = {"doc_0": object()}

        def _embed_query(self, texts):
            return _equivalent_embed_texts(texts)

    class _BridgeRegistry:
        def search(self, query, top_k, embed_texts_fn=None):
            return [
                {
                    "source_type": "bridge",
                    "bridge_id": "bridge_equivalent",
                    "question": "When did the mother of Lothair II die?",
                    "score": 0.25,
                    "evidence_text": "Q: When did the mother of Lothair II die? A: 20 March 851",
                    "rerank_text": "When did the mother of Lothair II die?",
                }
            ]

        def record_query_usage(self, used_bridge_ids, helpful_bridge_ids, batch_index):
            return None

    def _fake_collect_doc_evidence(entity_searcher, question, top_k_entities=10, top_k_direct_qa=10):
        return [
            {
                "source_type": "doc_entity",
                "doc_id": "doc_9",
                "question": "Who was Bertha the daughter of?",
                "score": 1.0,
                "evidence_text": "Q: Who was Bertha the daughter of? A: Lothair II",
                "rerank_text": "Q: Who was Bertha the daughter of? A: Lothair II",
            }
        ]

    monkeypatch.setattr(run_bridge_test, "collect_doc_evidence", _fake_collect_doc_evidence)
    monkeypatch.setattr(run_bridge_test, "_build_answer_reconciler", lambda entity_searcher, args: object())
    monkeypatch.setattr(
        run_bridge_test,
        "_parse_model_answer",
        lambda answer_agent, question, evidence: {"predicted_answer": "851", "full_response": "Answer: 851"},
    )

    args = SimpleNamespace(model="bedrock/openai.gpt-oss-120b-1:0", max_tokens=1024, bridge_query_top_k=1)
    batch_items = [
        BatchQueryItem(
            question="When did Lothair Ii's mother die?",
            doc_ids=["doc_0"],
            question_id="q0",
            gold_answer="20 March 851",
            answer_aliases=["20 March 851"],
        )
    ]

    result = run_bridge_test.answer_query_batch(
        batch_index=0,
        batch_items=batch_items,
        entity_searcher=_AnswerSearcher(),
        bridge_registry=_BridgeRegistry(),
        args=args,
    )

    query_result = result.query_results[0]
    assert query_result.bridge_evidence_used is True
    assert query_result.exact_bridge_match_found is False
    assert query_result.exact_bridge_match_kept is False
    assert query_result.equivalent_bridge_match_found is True
    assert query_result.equivalent_bridge_match_kept is True
    assert query_result.equivalent_bridge_match_bridge_id == "bridge_equivalent"
    assert query_result.top_evidence[0]["source_type"] == "bridge"
    assert query_result.top_evidence[0]["bridge_id"] == "bridge_equivalent"
    assert query_result.retrieval_trace["raw_bridge_hits"][0]["bridge_id"] == "bridge_equivalent"
    assert query_result.retrieval_trace["raw_bridge_hits"][0]["protected_keep"] is True
    assert query_result.retrieval_trace["merged_candidates"][0]["source_type"] == "doc_entity"
    assert query_result.retrieval_trace["merged_candidates"][1]["source_type"] == "bridge"
    assert query_result.retrieval_trace["merged_candidates"][0]["final_score"] > query_result.retrieval_trace["merged_candidates"][1]["final_score"]
    assert query_result.retrieval_trace["kept_candidates"][0]["bridge_id"] == "bridge_equivalent"


def test_answer_query_batch_leaves_doc_top_when_no_equivalent_bridge_exists(monkeypatch):
    class _AnswerSearcher:
        gsw_by_doc_id = {"doc_0": object()}

        def _embed_query(self, texts):
            return _equivalent_embed_texts(texts)

    class _BridgeRegistry:
        def search(self, query, top_k, embed_texts_fn=None):
            return [
                {
                    "source_type": "bridge",
                    "bridge_id": "bridge_other",
                    "question": "When did Lothair II's spouse die?",
                    "score": 0.25,
                    "evidence_text": "Q: When did Lothair II's spouse die? A: 11 November 875",
                    "rerank_text": "When did Lothair II's spouse die?",
                }
            ]

        def record_query_usage(self, used_bridge_ids, helpful_bridge_ids, batch_index):
            return None

    def _fake_collect_doc_evidence(entity_searcher, question, top_k_entities=10, top_k_direct_qa=10):
        return [
            {
                "source_type": "doc_entity",
                "doc_id": "doc_9",
                "question": "Who was Bertha the daughter of?",
                "score": 1.0,
                "evidence_text": "Q: Who was Bertha the daughter of? A: Lothair II",
                "rerank_text": "Q: Who was Bertha the daughter of? A: Lothair II",
            }
        ]

    monkeypatch.setattr(run_bridge_test, "collect_doc_evidence", _fake_collect_doc_evidence)
    monkeypatch.setattr(run_bridge_test, "_build_answer_reconciler", lambda entity_searcher, args: object())
    monkeypatch.setattr(
        run_bridge_test,
        "_parse_model_answer",
        lambda answer_agent, question, evidence: {"predicted_answer": "851", "full_response": "Answer: 851"},
    )

    args = SimpleNamespace(model="bedrock/openai.gpt-oss-120b-1:0", max_tokens=1024, bridge_query_top_k=1)
    batch_items = [
        BatchQueryItem(
            question="When did Lothair Ii's mother die?",
            doc_ids=["doc_0"],
            question_id="q0",
            gold_answer="20 March 851",
            answer_aliases=["20 March 851"],
        )
    ]

    result = run_bridge_test.answer_query_batch(
        batch_index=0,
        batch_items=batch_items,
        entity_searcher=_AnswerSearcher(),
        bridge_registry=_BridgeRegistry(),
        args=args,
    )

    query_result = result.query_results[0]
    assert query_result.bridge_evidence_used is False
    assert query_result.equivalent_bridge_match_found is False
    assert query_result.equivalent_bridge_match_kept is False
    assert query_result.top_evidence[0]["source_type"] == "doc_entity"
    assert query_result.retrieval_trace["raw_bridge_hits"][0]["protected_keep"] is False


def test_run_curriculum_writes_query_answer_artifact_and_console_summary(monkeypatch, tmp_path):
    printed = []

    def _fake_run_doc_batch(*, batch_index, doc_ids, args, output_dir, guidance_payload=None):
        batch_dir = Path(output_dir) / f"batch_{batch_index}" / "run_fake"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "results.json").write_text(
            json.dumps({"bridges": [{"bridge_id": "bridge_keep"}], "summary": {"total_bridges": 1}})
        )
        return {"bridges": [], "num_bridges": 0, "doc_ids": list(doc_ids), "batch_index": batch_index}

    def _fake_answer_query_batch(batch_index, batch_items, entity_searcher, bridge_registry, args):
        query_results = [
            QueryAnswerResult(
                question_id=item.question_id or f"q{batch_index}",
                question=item.question,
                predicted_answer="8 March 925",
                full_response="Answer: 8 March 925",
                exact_match=1.0,
                f1=1.0,
                bridge_evidence_used=True,
                answer_source_mix={"bridge": 2, "doc_entity": 1},
                top_evidence=[
                    {
                        "source_type": "bridge",
                        "final_score": 0.9,
                        "bridge_id": "bridge_x",
                        "doc_id": "doc_0",
                        "evidence_text": "Q: ... A: ...",
                    }
                ],
                patterns=infer_question_patterns(item.question),
                gold_answers=["8 March 925"],
                retrieval_trace={
                    "raw_bridge_hits": [{"source_type": "bridge", "bridge_id": "bridge_x", "final_score": 0.9, "kept": True}],
                    "raw_doc_hits": [{"source_type": "doc_entity", "doc_id": "doc_0", "final_score": 0.4, "kept": False}],
                    "merged_candidates": [
                        {"source_type": "bridge", "bridge_id": "bridge_x", "final_score": 0.9, "kept": True},
                        {"source_type": "doc_entity", "doc_id": "doc_0", "final_score": 0.4, "kept": False},
                    ],
                    "kept_candidates": [{"source_type": "bridge", "bridge_id": "bridge_x", "final_score": 0.9, "kept": True}],
                },
            )
            for item in batch_items
        ]
        return QueryBatchResult(
            batch_index=batch_index,
            query_results=query_results,
            overall_metrics={"ExactMatch": 1.0, "F1": 1.0, "scored_queries": len(query_results)},
            feedback_brief=build_batch_feedback_brief(query_results),
        )

    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _fake_run_doc_batch)
    monkeypatch.setattr(run_bridge_test, "answer_query_batch", _fake_answer_query_batch)
    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _StubEntitySearcher)
    monkeypatch.setattr(run_bridge_test.console, "print", lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)))

    args = SimpleNamespace(
        curriculum_batch_size=1,
        curriculum_seed_batch_size=1,
        bridge_prompt_exemplar_top_k=3,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
    )
    items = [
        BatchQueryItem(question="When did Lothair II's mother die?", doc_ids=["doc_0"], question_id="q0"),
    ]

    result = run_bridge_test.run_curriculum(items, args)

    artifact_path = tmp_path / "out" / "batch_0" / "query_answer_results.json"
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text())
    assert payload["batch_index"] == 0
    assert payload["query_results"][0]["predicted_answer"] == "8 March 925"
    assert payload["query_results"][0]["full_response"] == "Answer: 8 March 925"
    assert payload["query_results"][0]["answer_source_mix"] == {"bridge": 2, "doc_entity": 1}
    assert payload["query_results"][0]["retrieval_trace"]["raw_bridge_hits"][0]["bridge_id"] == "bridge_x"
    assert payload["query_results"][0]["retrieval_trace"]["merged_candidates"][0]["source_type"] == "bridge"
    assert payload["overall_metrics"]["F1"] == 1.0
    assert payload["feedback_brief"]["bridge_helpful_patterns"]
    assert payload["metadata"]["batch_doc_ids"] == ["doc_0"]
    assert payload["metadata"]["accumulated_doc_ids"] == ["doc_0"]
    assert "answer_accumulated" in payload["metadata"]["answer_cache_dir"]
    assert not (tmp_path / "out" / "batch_0" / "run_fake" / "query_answer_results.json").exists()
    assert any("Query answers:" in line for line in printed)
    assert any("Query answer:" in line and "8 March 925" in line and "bridge=yes" in line for line in printed)
    assert any("exact_bridge=" in line and "equivalent_bridge=" in line and "top_bridge=" in line and "top_doc=" in line for line in printed)
    assert result["results"][0]["predicted_answer"] == "8 March 925"


def test_run_curriculum_writes_progress_trace_and_bridge_registry_snapshot(monkeypatch, tmp_path):
    def _fake_run_doc_batch(*, batch_index, doc_ids, args, output_dir, guidance_payload=None):
        batch_dir = Path(output_dir) / f"batch_{batch_index}" / "run_fake"
        batch_dir.mkdir(parents=True, exist_ok=True)
        bridge_payload = {
            "bridge_id": "bridge_snapshot",
            "question": "When did Mira's mother die?",
            "reverse_question": "Whose child died in 912?",
            "source_docs": ["doc_0", "doc_1"],
            "confidence": 0.93,
            "resolved_entities": {
                "answers": [{"name": "912"}],
                "reverse_answers": [{"name": "Mira"}],
            },
        }
        (batch_dir / "results.json").write_text(
            json.dumps({"bridges": [bridge_payload], "summary": {"total_bridges": 1}})
        )
        return {
            "bridges": [bridge_payload],
            "num_bridges": 1,
            "doc_ids": list(doc_ids),
            "batch_index": batch_index,
            "run_results": {"total_bridges": 1},
        }

    def _fake_answer_query_batch(batch_index, batch_items, entity_searcher, bridge_registry, args, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "query_answering",
                    "status": "started",
                    "completed": 0,
                    "total": len(batch_items),
                    "unit": "queries",
                }
            )
        query_results = []
        for index, item in enumerate(batch_items, start=1):
            result = QueryAnswerResult(
                question_id=item.question_id or f"q{batch_index}",
                question=item.question,
                predicted_answer="912",
                full_response="Answer: 912",
                exact_match=1.0,
                f1=1.0,
                bridge_evidence_used=True,
                answer_source_mix={"bridge": 1},
                top_evidence=[
                    {
                        "source_type": "bridge",
                        "final_score": 1.0,
                        "bridge_id": "bridge_snapshot",
                        "doc_id": None,
                        "question": "When did Mira's mother die?",
                        "evidence_text": "Q: When did Mira's mother die? A: 912",
                    }
                ],
                patterns=["mother_death.lookup.date"],
                question_pattern={
                    "operation": "lookup",
                    "answer_type": "date",
                    "domain": "history_politics",
                    "relation_label": "mother_death",
                    "comparison_target": "none",
                    "pattern_key": "mother_death.lookup.date",
                    "classifier_source": "llm",
                },
                gold_answers=["912"],
            )
            query_results.append(result)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "query_answering",
                        "status": "progress",
                        "completed": index,
                        "total": len(batch_items),
                        "unit": "queries",
                        "question_id": result.question_id,
                        "question": item.question,
                        "bridge_evidence_used": True,
                        "predicted_answer": result.predicted_answer,
                        "f1": result.f1,
                        "exact_match": result.exact_match,
                    }
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "query_answering",
                    "status": "completed",
                    "completed": len(batch_items),
                    "total": len(batch_items),
                    "unit": "queries",
                }
            )
        return QueryBatchResult(
            batch_index=batch_index,
            query_results=query_results,
            overall_metrics={"ExactMatch": 1.0, "F1": 1.0, "scored_queries": len(query_results)},
            feedback_brief=build_batch_feedback_brief(query_results),
        )

    query_classifier = QuestionPatternClassifier(
        classify_question_fn=lambda question: {
            "operation": "lookup",
            "answer_type": "person" if "whose child" in question.lower() else ("date" if "die" in question.lower() else "person"),
            "domain": "history_politics",
            "relation_label": "child_of" if "whose child" in question.lower() else ("mother_death" if "die" in question.lower() else "child_of"),
            "comparison_target": "none",
        }
    )
    bridge_classifier = QuestionPatternClassifier(
        classify_questions_fn=lambda questions: {
            question: {
                "operation": "lookup",
                "answer_type": "person" if "whose child" in question.lower() else ("date" if "die" in question.lower() else "person"),
                "domain": "history_politics",
                "relation_label": "child_of" if "whose child" in question.lower() else ("mother_death" if "die" in question.lower() else "child_of"),
                "comparison_target": "none",
                "classifier_source": "llm",
            }
            for question in questions
        }
    )

    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _fake_run_doc_batch)
    monkeypatch.setattr(run_bridge_test, "answer_query_batch", _fake_answer_query_batch)
    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _StubEntitySearcher)
    monkeypatch.setattr(run_bridge_test, "_build_question_pattern_classifier", lambda entity_searcher, args: query_classifier)
    monkeypatch.setattr(
        run_bridge_test,
        "_build_bridge_surface_pattern_classifier",
        lambda entity_searcher, args: bridge_classifier,
    )

    args = SimpleNamespace(
        curriculum_batch_size=1,
        curriculum_seed_batch_size=1,
        bridge_prompt_exemplar_top_k=3,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
        model="stub-model",
    )
    items = [
        BatchQueryItem(question="When did Mira's mother die?", doc_ids=["doc_0"], question_id="q0"),
    ]

    run_bridge_test.run_curriculum(items, args)

    progress_path = tmp_path / "out" / "batch_0" / "curriculum_progress.jsonl"
    report_path = tmp_path / "out" / "batch_0" / "bridge_pattern_classification_report.json"
    snapshot_path = tmp_path / "out" / "batch_0" / "bridge_registry_snapshot.json"
    run_results_path = tmp_path / "out" / "batch_0" / "run_fake" / "results.json"
    assert progress_path.exists()
    assert report_path.exists()
    assert snapshot_path.exists()
    progress_events = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    phases = {(event["phase"], event["status"]) for event in progress_events}
    assert ("bridge_pattern_classification", "started") in phases
    assert ("bridge_pattern_classification", "completed") in phases
    assert ("bridge_index_rebuild", "started") in phases
    assert ("bridge_index_rebuild", "completed") in phases
    assert ("query_answering", "started") in phases
    assert ("query_answering", "progress") in phases
    assert ("query_answering", "completed") in phases
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok"
    assert report["llm_tagged_surface_count"] == 2
    assert report["failed_surface_count"] == 0
    assert report["initial_failed_surface_count"] == 0
    assert report["fallback_retry_attempted_count"] == 0
    assert report["fallback_retry_succeeded_count"] == 0
    assert report["fallback_retry_failed_count"] == 0
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["added_bridge_count"] == 1
    assert snapshot["added_bridge_ids"] == ["bridge_snapshot"]
    assert snapshot["registry_bridge_count"] == 1
    assert snapshot["registry_surface_count"] == 2
    assert snapshot["new_bridge_records"][0]["pattern_tags"]
    assert snapshot["new_bridge_records"][0]["forward_pattern"]["pattern_key"] == "mother_death.lookup.date"
    assert snapshot["new_bridge_records"][0]["reverse_pattern"]["pattern_key"] == "child_of.lookup.person"


def test_run_curriculum_continues_when_strict_bridge_tagging_fallback_recovers_row(monkeypatch, tmp_path):
    generated_bridges = [
        {
            "bridge_id": f"bridge_recovery_{index}",
            "question": f"When did Mira's mother die {index}?",
            "reverse_question": f"Whose child died in 91{index:02d}?",
            "source_docs": ["doc_0", "doc_1"],
            "confidence": 0.9,
            "resolved_entities": {
                "answers": [{"name": f"91{index:02d}"}],
                "reverse_answers": [{"name": "Mira"}],
            },
        }
        for index in range(10)
    ]
    rescued_reverse_question = generated_bridges[-1]["reverse_question"]

    def _fake_run_doc_batch(*, batch_index, doc_ids, args, output_dir, guidance_payload=None):
        batch_dir = Path(output_dir) / f"batch_{batch_index}" / "run_recovery"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "results.json").write_text(json.dumps({"bridges": []}))
        return {
            "batch_index": batch_index,
            "doc_ids": list(doc_ids),
            "bridges": list(generated_bridges),
        }

    query_classifier = QuestionPatternClassifier(
        classify_question_fn=lambda question: {
            "operation": "lookup",
            "answer_type": "date",
            "domain": "history_politics",
            "relation_label": "mother_death" if "die" in question.lower() else "child_of",
            "comparison_target": "none",
            "classifier_source": "llm",
        }
    )

    batch_calls = []
    fallback_calls = []

    def _classify_many(questions):
        batch_calls.append(list(questions))
        return {
            question: {
                "operation": "lookup",
                "answer_type": "person" if "whose child" in question.lower() else "date",
                "domain": "history_politics",
                "relation_label": "child_of" if "whose child" in question.lower() else "mother_death",
                "comparison_target": "none",
                "classifier_source": "llm",
            }
            for question in questions
            if question != rescued_reverse_question
        }

    def _classify_one(question):
        fallback_calls.append(question)
        return {
            "operation": "lookup",
            "answer_type": "person",
            "domain": "history_politics",
            "relation_label": "child_of",
            "comparison_target": "none",
            "classifier_source": "llm",
        }

    bridge_classifier = QuestionPatternClassifier(
        classify_question_fn=_classify_one,
        classify_questions_fn=_classify_many,
        classifier_backend="litellm",
        classifier_model_name="bedrock/converse/openai.gpt-oss-120b-1:0",
    )

    query_calls = []

    def _fake_answer_query_batch(*args, **kwargs):
        query_calls.append(True)
        return QueryBatchResult(
            batch_index=0,
            query_results=[
                QueryAnswerResult(
                    question_id="q0",
                    question="When did Mira's mother die?",
                    predicted_answer="912",
                    full_response="912",
                    exact_match=1.0,
                    f1=1.0,
                    bridge_evidence_used=True,
                    answer_source_mix={"bridge": 1},
                    top_evidence=[],
                    patterns=["mother_death.lookup.date"],
                    question_pattern={"pattern_key": "mother_death.lookup.date", "classifier_source": "llm"},
                    gold_answers=["912"],
                )
            ],
            overall_metrics={"average_f1": 1.0, "average_exact_match": 1.0},
            feedback_brief=BatchFeedbackBrief(
                high_demand_patterns=[],
                low_score_patterns=[],
                doc_dominant_patterns=[],
                bridge_helpful_patterns=[],
                summary_lines=[],
            ),
        )

    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _fake_run_doc_batch)
    monkeypatch.setattr(run_bridge_test, "answer_query_batch", _fake_answer_query_batch)
    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _StubEntitySearcher)
    monkeypatch.setattr(run_bridge_test, "_build_question_pattern_classifier", lambda entity_searcher, args: query_classifier)
    monkeypatch.setattr(
        run_bridge_test,
        "_build_bridge_surface_pattern_classifier",
        lambda entity_searcher, args: bridge_classifier,
    )

    args = SimpleNamespace(
        curriculum_batch_size=1,
        curriculum_seed_batch_size=1,
        bridge_prompt_exemplar_top_k=3,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
        model="stub-model",
    )
    items = [
        BatchQueryItem(question="When did Mira's mother die?", doc_ids=["doc_0"], question_id="q0"),
    ]

    run_bridge_test.run_curriculum(items, args)

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 20
    assert fallback_calls == [rescued_reverse_question]
    assert query_calls == [True]

    report_path = tmp_path / "out" / "batch_0" / "bridge_pattern_classification_report.json"
    snapshot_path = tmp_path / "out" / "batch_0" / "bridge_registry_snapshot.json"
    assert report_path.exists()
    assert snapshot_path.exists()
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok"
    assert report["initial_failed_surface_count"] == 1
    assert report["fallback_retry_attempted_count"] == 1
    assert report["fallback_retry_succeeded_count"] == 1
    assert report["fallback_retry_failed_count"] == 0
    assert report["failed_surface_count"] == 0
    assert report["llm_tagged_surface_count"] == 20


def test_curriculum_progress_tracker_truncates_stale_progress_file(tmp_path):
    batch_dir = tmp_path / "batch_0"
    batch_dir.mkdir(parents=True, exist_ok=True)
    progress_path = batch_dir / "curriculum_progress.jsonl"
    progress_path.write_text('{"batch_index": 7, "phase": "query_answering", "status": "completed"}\n')

    with run_bridge_test.CurriculumProgressTracker(batch_index=0, batch_output_dir=str(batch_dir)) as tracker:
        tracker.emit(
            {
                "phase": "bridge_generation_shards",
                "status": "started",
                "completed": 0,
                "total": 2,
                "unit": "shards",
            }
        )

    lines = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["batch_index"] == 0
    assert lines[0]["phase"] == "bridge_generation_shards"


def test_run_curriculum_aborts_on_strict_bridge_tagging_failure(monkeypatch, tmp_path):
    def _fake_run_doc_batch(*, batch_index, doc_ids, args, output_dir, guidance_payload=None):
        batch_dir = Path(output_dir) / f"batch_{batch_index}" / "run_failure"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "results.json").write_text(json.dumps({"bridges": []}))
        return {
            "batch_index": batch_index,
            "doc_ids": list(doc_ids),
            "bridges": [
                {
                    "bridge_id": "bridge_failure",
                    "question": "When did Mira's mother die?",
                    "reverse_question": "Whose child died in 912?",
                    "source_docs": ["doc_0", "doc_1"],
                    "confidence": 0.9,
                    "resolved_entities": {
                        "answers": [{"name": "912"}],
                        "reverse_answers": [{"name": "Mira"}],
                    },
                }
            ],
        }

    def _unexpected_answer_query_batch(*args, **kwargs):
        raise AssertionError("query answering should not run when strict bridge tagging fails")

    query_classifier = QuestionPatternClassifier(
        classify_question_fn=lambda question: {
            "operation": "lookup",
            "answer_type": "date",
            "domain": "history_politics",
            "relation_label": "mother_death",
            "comparison_target": "none",
            "classifier_source": "llm",
        }
    )
    def _raise_backend_value_error(_questions):
        raise ValueError("curator backend mismatch")

    bridge_classifier = QuestionPatternClassifier(
        classify_questions_fn=_raise_backend_value_error,
        classifier_backend="litellm",
        classifier_model_name="bedrock/converse/openai.gpt-oss-120b-1:0",
    )

    monkeypatch.setattr(run_bridge_test, "run_doc_batch", _fake_run_doc_batch)
    monkeypatch.setattr(run_bridge_test, "answer_query_batch", _unexpected_answer_query_batch)
    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _StubEntitySearcher)
    monkeypatch.setattr(run_bridge_test, "_build_question_pattern_classifier", lambda entity_searcher, args: query_classifier)
    monkeypatch.setattr(
        run_bridge_test,
        "_build_bridge_surface_pattern_classifier",
        lambda entity_searcher, args: bridge_classifier,
    )

    args = SimpleNamespace(
        curriculum_batch_size=1,
        curriculum_seed_batch_size=1,
        bridge_prompt_exemplar_top_k=3,
        output_dir=str(tmp_path / "out"),
        gsw_path=str(tmp_path / "gsw"),
        model="stub-model",
    )
    items = [
        BatchQueryItem(question="When did Mira's mother die?", doc_ids=["doc_0"], question_id="q0"),
    ]

    with pytest.raises(RuntimeError, match="Bridge surface LLM tagging failed"):
        run_bridge_test.run_curriculum(items, args)

    report_path = tmp_path / "out" / "batch_0" / "bridge_pattern_classification_report.json"
    snapshot_path = tmp_path / "out" / "batch_0" / "bridge_registry_snapshot.json"
    assert report_path.exists()
    assert not snapshot_path.exists()
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failed_surface_count"] == 2
    assert report["llm_tagged_surface_count"] == 0
    assert report["initial_failed_surface_count"] == 2
    assert report["fallback_retry_attempted_count"] == 0
    assert report["fallback_retry_succeeded_count"] == 0
    assert report["fallback_retry_failed_count"] == 0
    assert report["failed_surface_examples"]
    assert report["classifier_backend"] == "litellm"
    assert report["classifier_model_name"] == "bedrock/converse/openai.gpt-oss-120b-1:0"
    assert report["backend_exception_message"] == "curator backend mismatch"


class _FakeEmbeddingLLM:
    init_calls = 0

    def __init__(self, *args, **kwargs):
        type(self).init_calls += 1
        self.args = args
        self.kwargs = kwargs


class _ConcurrentFakeEmbeddingLLM:
    init_calls = 0
    active_calls = 0
    max_active_calls = 0
    state_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        type(self).init_calls += 1
        self.args = args
        self.kwargs = kwargs
        time.sleep(0.05)

    def embed(self, texts):
        with type(self).state_lock:
            type(self).active_calls += 1
            type(self).max_active_calls = max(type(self).max_active_calls, type(self).active_calls)
        try:
            time.sleep(0.05)
            return [
                SimpleNamespace(outputs=SimpleNamespace(embedding=[float(index), float(len(str(text))) or 0.0]))
                for index, text in enumerate(texts)
            ]
        finally:
            with type(self).state_lock:
                type(self).active_calls -= 1


def _patch_minimal_entity_searcher(monkeypatch, tmp_path):
    monkeypatch.setattr(simple_entity_search, "FAISS_AVAILABLE", False)
    monkeypatch.setattr(simple_entity_search, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(simple_entity_search, "_SHARED_EMBEDDING_MODELS", {})
    monkeypatch.setattr(simple_entity_search, "_SHARED_EMBEDDING_MODEL_LOCKS", {})
    monkeypatch.setattr(
        simple_entity_search,
        "load_gsw_files",
        lambda *args, **kwargs: ([SimpleNamespace(entity_nodes=[])], ["doc_0"]),
    )
    monkeypatch.setattr(simple_entity_search.EntitySearcher, "_extract_entities_from_gsw", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(simple_entity_search.EntitySearcher, "_load_entity_embeddings_cache", lambda self: True)
    monkeypatch.setattr(simple_entity_search.EntitySearcher, "_load_qa_embeddings_cache", lambda self: True)
    monkeypatch.setattr(simple_entity_search.EntitySearcher, "_build_qa_embeddings_matrix", lambda self: None)
    monkeypatch.setattr(simple_entity_search.EntitySearcher, "_precompute_qa_embeddings", lambda self: None)
    monkeypatch.setattr(simple_entity_search.EntitySearcher, "_build_bm25_index", lambda self: None)
    monkeypatch.setattr(simple_entity_search, "OPENAI_AVAILABLE", False)


def test_curriculum_cache_dir_is_stable_for_same_doc_set(tmp_path):
    first = run_bridge_test._curriculum_cache_dir(str(tmp_path), "bridge_batches", ["doc_2", "doc_1"])
    second = run_bridge_test._curriculum_cache_dir(str(tmp_path), "bridge_batches", ["doc_1", "doc_2"])
    third = run_bridge_test._curriculum_cache_dir(str(tmp_path), "bridge_batches", ["doc_1", "doc_3"])

    assert first == second
    assert first != third
    assert Path(first).exists()


def test_entity_searcher_reuses_embedding_model_singleton(monkeypatch, tmp_path):
    _patch_minimal_entity_searcher(monkeypatch, tmp_path)
    _FakeEmbeddingLLM.init_calls = 0
    monkeypatch.setattr(simple_entity_search, "LLM", _FakeEmbeddingLLM)

    first = simple_entity_search.EntitySearcher(
        num_documents=1,
        path_to_gsw_files=str(tmp_path / "gsw"),
        cache_dir=str(tmp_path / "cache_a"),
        verbose=False,
        use_bm25=False,
    )
    second = simple_entity_search.EntitySearcher(
        num_documents=1,
        path_to_gsw_files=str(tmp_path / "gsw"),
        cache_dir=str(tmp_path / "cache_b"),
        verbose=False,
        use_bm25=False,
    )

    assert _FakeEmbeddingLLM.init_calls == 1
    assert first.embedding_model_reused is False
    assert second.embedding_model_reused is True
    assert second.embedding_model_source == "reused"
    assert first.embedding_model is second.embedding_model
    assert first.embedding_model.kwargs["gpu_memory_utilization"] == 0.5


def test_entity_searcher_shares_singleton_and_serializes_embed_calls_across_threads(monkeypatch, tmp_path):
    _patch_minimal_entity_searcher(monkeypatch, tmp_path)
    _ConcurrentFakeEmbeddingLLM.init_calls = 0
    _ConcurrentFakeEmbeddingLLM.active_calls = 0
    _ConcurrentFakeEmbeddingLLM.max_active_calls = 0
    monkeypatch.setattr(simple_entity_search, "LLM", _ConcurrentFakeEmbeddingLLM)

    def _build_searcher(cache_name: str):
        return simple_entity_search.EntitySearcher(
            num_documents=1,
            path_to_gsw_files=str(tmp_path / "gsw"),
            cache_dir=str(tmp_path / cache_name),
            verbose=False,
            use_bm25=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(_build_searcher, ["cache_a", "cache_b"]))

    assert _ConcurrentFakeEmbeddingLLM.init_calls == 1
    assert first.embedding_model is second.embedding_model
    assert {first.embedding_model_source, second.embedding_model_source} == {"started", "reused"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda searcher: searcher._embed_query(["concurrent query"]), [first, second]))

    assert _ConcurrentFakeEmbeddingLLM.max_active_calls == 1


def test_entity_searcher_raises_clear_error_when_embedding_startup_fails(monkeypatch, tmp_path):
    _patch_minimal_entity_searcher(monkeypatch, tmp_path)

    class _FailingEmbeddingLLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("gpu unavailable")

    monkeypatch.setattr(simple_entity_search, "LLM", _FailingEmbeddingLLM)

    with pytest.raises(ValueError) as exc_info:
        simple_entity_search.EntitySearcher(
            num_documents=1,
            path_to_gsw_files=str(tmp_path / "gsw"),
            cache_dir=str(tmp_path / "cache_fail"),
            verbose=False,
            use_bm25=False,
        )

    message = str(exc_info.value)
    assert "Failed to initialize retrieval embedding model" in message
    assert "No reusable embedding engine available" in message
    assert "gpu unavailable" in message


def test_run_doc_batch_uses_in_process_sleep_time_runner(monkeypatch, tmp_path):
    calls = {}

    def _fake_setup_temp_gsw_dir(*args, **kwargs):
        networks = tmp_path / "temp_networks"
        networks.mkdir(parents=True, exist_ok=True)
        return str(networks)

    class _FakeSearcher:
        def __init__(self, **kwargs):
            calls["searcher_kwargs"] = kwargs
            self.gsw_by_doc_id = {"doc_0": object(), "doc_1": object()}

    def _fake_run_sleep_time_once(args, *, entity_searcher=None, curriculum_guidance=None):
        calls["run_args"] = args
        calls["entity_searcher"] = entity_searcher
        calls["guidance"] = curriculum_guidance
        return {
            "metadata": {"run_id": "test_run"},
            "summary": {"total_bridges": 1},
            "bridges": [{"bridge_id": "bridge_1"}],
        }

    def _unexpected_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called in curriculum mode")

    monkeypatch.setattr(run_bridge_test, "setup_temp_gsw_dir", _fake_setup_temp_gsw_dir)
    monkeypatch.setattr(run_bridge_test, "EntitySearcher", _FakeSearcher)
    monkeypatch.setattr(run_bridge_test, "run_sleep_time_once", _fake_run_sleep_time_once)
    monkeypatch.setattr(run_bridge_test.subprocess, "run", _unexpected_subprocess)

    args = SimpleNamespace(
        gsw_path=str(tmp_path / "gsw"),
        model="bedrock/openai.gpt-oss-120b-1:0",
        max_iterations=10,
        max_tokens=5000,
        reasoning_effort="medium",
        pipeline_mode="hybrid",
        hybrid_scope="doc_edge",
        edge_max_depth=2,
        edge_max_calls=3,
        edge_max_tokens=4000,
        max_optional_docs_per_edge=2,
        edge_parallel_workers=4,
        parallel_progress_heartbeat_seconds=10.0,
        parallel_stuck_warning_seconds=60.0,
        verifier_parallel_workers=2,
        edge_max_accepted_bridges=6,
        edge_parallel_enabled=True,
        embedding_gpu_memory_utilization=0.5,
        verbose=False,
        show_thinking=False,
        base_url=None,
        root_model=None,
        worker_model=None,
    )

    guidance = {"prior_batches_seen": 1}
    result = run_bridge_test.run_doc_batch(
        batch_index=0,
        doc_ids=["doc_1", "doc_0"],
        args=args,
        output_dir=str(tmp_path / "out"),
        guidance_payload=guidance,
    )

    guidance_path = tmp_path / "out" / "batch_0" / "curriculum_guidance_batch_0.json"
    assert guidance_path.exists()
    assert result["bridges"] == [{"bridge_id": "bridge_1"}]
    assert result["run_results"]["total_bridges"] == 1
    assert calls["guidance"] == guidance
    assert calls["run_args"].cache_dir.endswith(run_bridge_test._doc_set_hash(["doc_0", "doc_1"]))
    assert "_curriculum_cache/bridge_batches" in calls["run_args"].cache_dir
    assert isinstance(calls["entity_searcher"], _FakeSearcher)
    assert calls["searcher_kwargs"]["cache_dir"] == calls["run_args"].cache_dir
    assert calls["searcher_kwargs"]["embedding_gpu_memory_utilization"] == 0.5
    assert calls["run_args"].embedding_gpu_memory_utilization == 0.5
