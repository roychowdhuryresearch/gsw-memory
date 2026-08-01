from __future__ import annotations

from dataclasses import replace

from colab_full_solution import (
    RunConfig,
    append_jsonl,
    execute_panini_plan,
    retrieval_components,
    read_jsonl,
    score_trace,
    select_reranker_model,
    stratified_sample,
    validate_plan,
)
from panini_course.qwen_models import QwenAnswerer, QwenDecomposer


def test_t4_uses_reference_reranker_and_smaller_gpu_uses_fallback():
    config = RunConfig()
    assert select_reranker_model(config, 15.0) == "Qwen/Qwen3-Reranker-8B"
    assert select_reranker_model(config, 12.0) == "Qwen/Qwen3-Reranker-4B"
    assert select_reranker_model(config, 24.0) == "Qwen/Qwen3-Reranker-8B"


def test_plan_validation_rejects_future_reference():
    result = validate_plan(
        [{"question": "When did <ENTITY_Q2> die?", "requires_retrieval": True}]
    )
    assert not result["valid"]
    assert "invalid reference Q2" in result["errors"][0]


def test_stratified_ablation_sample_is_reproducible():
    questions = [
        {"question_id": f"{group}-{index}", "type": group}
        for group in ("a", "b", "c", "d")
        for index in range(10)
    ]
    left = stratified_sample(questions, "type", size=20, seed=232)
    right = stratified_sample(questions, "type", size=20, seed=232)
    assert left == right
    assert len(left) == 20


def test_complete_chain_requires_one_consistent_linear_beam():
    question = {
        "answer": "final",
        "answer_aliases": [],
        "context_document_ids": [],
        "evidences": [
            {"question": "first", "answer": "Ada", "document_id": "d1"},
            {"question": "second", "answer": "1843", "document_id": "d2"},
        ],
    }
    trace = {
        "evidence": [],
        "chains": [
            {"steps": [{"qa_uid": "q1", "answer": "Ada"}]},
            {"steps": [{"qa_uid": "q2", "answer": "1843"}]},
        ],
        "branch_traces": [
            {
                "hops": [
                    {
                        "kept": [
                            {"qa_ids": ["q1", "qx"], "answers": ["Ada", "wrong"]},
                            {"qa_ids": ["qy", "q2"], "answers": ["wrong", "1843"]},
                        ]
                    }
                ]
            }
        ],
    }
    assert score_trace(question, trace, [])["complete_chain_recovery"] == 0.0


def test_interrupted_jsonl_tail_is_ignored_and_next_record_is_recoverable(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text('{"question_id":"complete"}\n{"question_id":', encoding="utf-8")
    append_jsonl(path, {"question_id": "resumed"})
    assert [row["question_id"] for row in read_jsonl(path)] == [
        "complete",
        "resumed",
    ]


def _qa(uid, answers, score, *, ids=None, roles=None, question=None):
    return {
        "qa_uid": uid,
        "question": question or uid,
        "document_id": f"doc-{uid}",
        "answer_names": answers,
        "answer_ids": ids or [f"id-{uid}-{i}" for i in range(len(answers))],
        "answer_role_states": roles or [],
        "score": score,
    }


def test_panini_components_topologically_sort_a_converging_dag():
    plan = [
        {"question": "Who directed A?", "requires_retrieval": True},
        {"question": "Who directed B?", "requires_retrieval": True},
        {
            "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": True,
        },
    ]
    assert retrieval_components(plan) == [[1, 2, 3]]


def test_panini_multi_parent_harmonic_combination_substitutes_both_answers():
    plan = [
        {"question": "Root A", "requires_retrieval": True},
        {"question": "Root B", "requires_retrieval": True},
        {
            "question": "Compare <ENTITY_Q1> with <ENTITY_Q2>",
            "requires_retrieval": True,
        },
    ]
    table = {
        "Root A": [_qa("a", ["Ada"], 0.9)],
        "Root B": [_qa("b", ["Grace"], 0.8)],
        "Compare Ada with Grace": [_qa("c", ["Ada"], 0.95)],
    }
    result = execute_panini_plan(
        plan,
        lambda query, k: table[query][:k],
        replace(RunConfig(), beam_width=2, candidates_per_hop=2),
        original_question="Who is older?",
    )
    final_step = result["component_traces"][0]["steps"][-1]
    assert final_step["dependencies"] == [1, 2]
    assert final_step["issued_queries"] == ["Compare Ada with Grace"]
    assert [step["qa_uid"] for step in result["chains"][0]["steps"]] == [
        "a",
        "b",
        "c",
    ]


def test_panini_groups_intermediate_entities_but_not_final_qa_pairs():
    plan = [
        {"question": "Find person", "requires_retrieval": True},
        {"question": "Fact about <ENTITY_Q1>", "requires_retrieval": True},
    ]
    table = {
        "Find person": [
            _qa("weak-ada", ["Ada"], 0.1, ids=["person-ada"]),
            _qa("strong-ada", ["Ada Lovelace"], 0.9, ids=["person-ada"]),
            _qa("grace", ["Grace"], 0.8, ids=["person-grace"]),
        ],
        "Fact about Ada Lovelace": [
            _qa("final-a", ["1843"], 0.9),
            _qa("final-b", ["1843"], 0.8),
        ],
        "Fact about Grace": [_qa("final-c", ["1952"], 0.7)],
    }
    result = execute_panini_plan(
        plan,
        lambda query, k: table[query][:k],
        replace(RunConfig(), beam_width=3, candidates_per_hop=3),
    )
    assert {chain["steps"][0]["qa_uid"] for chain in result["chains"]} == {
        "strong-ada",
        "grace",
    }
    assert {chain["steps"][-1]["qa_uid"] for chain in result["chains"]} >= {
        "final-a",
        "final-b",
    }


def test_panini_evidence_comes_from_every_surviving_final_beam():
    plan = [
        {"question": "Root", "requires_retrieval": True},
        {"question": "Finish <ENTITY_Q1>", "requires_retrieval": True},
    ]
    table = {
        "Root": [_qa("root-a", ["Ada"], 0.9), _qa("root-b", ["Grace"], 0.8)],
        "Finish Ada": [_qa("answer-a", ["1843"], 0.9)],
        "Finish Grace": [_qa("answer-b", ["1952"], 0.8)],
    }
    result = execute_panini_plan(
        plan,
        lambda query, k: table[query][:k],
        replace(RunConfig(), beam_width=2, candidates_per_hop=2),
    )
    assert {row["qa_uid"] for row in result["evidence"]} == {
        "root-a",
        "root-b",
        "answer-a",
        "answer-b",
    }


def test_panini_answer_prompt_matches_research_protocol_without_na_instruction():
    messages = QwenAnswerer.build_messages(
        "When was the institution founded?",
        ["Q: Where does Ada work? A: Example University organization: university"],
    )
    assert len(messages) == 4
    assert messages[0]["content"] == QwenAnswerer.SYSTEM_PROMPT
    assert "Neville A. Stanton" in messages[1]["content"]
    assert messages[2]["content"].endswith("Answer: 1862.")
    assert messages[3]["content"].endswith("Thought: \n\n")
    assert "N/A" not in "\n".join(message["content"] for message in messages)
    assert QwenAnswerer.parse_response("reasoning\nAnswer: 1862.") == "1862"


def test_decomposer_uses_the_research_system_message_and_full_template():
    decomposer = object.__new__(QwenDecomposer)
    decomposer.prompt_template = "Full decomposition prompt for {question}"
    messages = decomposer.build_messages("the test question")
    assert messages == [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that breaks down complex "
                "questions into simple steps."
            ),
        },
        {
            "role": "user",
            "content": "Full decomposition prompt for the test question",
        },
    ]
