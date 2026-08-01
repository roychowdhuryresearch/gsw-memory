from __future__ import annotations

from dataclasses import replace

from colab_full_solution import (
    RunConfig,
    append_jsonl,
    advance_branch_one_hop,
    execute_plan,
    prune_chains,
    retrieval_branches,
    read_jsonl,
    score_trace,
    select_reranker_model,
    stratified_sample,
    validate_plan,
)


def test_t4_uses_smaller_reranker_but_l4_uses_reference_model():
    config = RunConfig()
    assert select_reranker_model(config, 15.0) == "Qwen/Qwen3-Reranker-4B"
    assert select_reranker_model(config, 24.0) == "Qwen/Qwen3-Reranker-8B"


def test_plan_validation_rejects_future_reference():
    result = validate_plan(
        [{"question": "When did <ENTITY_Q2> die?", "requires_retrieval": True}]
    )
    assert not result["valid"]
    assert "invalid reference Q2" in result["errors"][0]


def test_parallel_retrieval_branches_are_extracted_before_reasoning():
    plan = [
        {"question": "Who directed A?", "requires_retrieval": True},
        {"question": "Who directed B?", "requires_retrieval": True},
        {"question": "When did <ENTITY_Q1> die?", "requires_retrieval": True},
        {"question": "When did <ENTITY_Q2> die?", "requires_retrieval": True},
        {
            "question": "Which is later, <ENTITY_Q3> or <ENTITY_Q4>?",
            "requires_retrieval": False,
        },
    ]
    branches, warnings = retrieval_branches(plan)
    assert branches == [[1, 3], [2, 4]]
    assert warnings == []


def test_global_unique_answer_pruning_uses_score_then_stable_ids():
    chains = [
        {"steps": [{"qa_uid": "b", "answer": "Ada"}], "score": 0.8},
        {"steps": [{"qa_uid": "a", "answer": "ada"}], "score": 0.9},
        {"steps": [{"qa_uid": "c", "answer": "Grace"}], "score": 0.85},
    ]
    kept = prune_chains(chains, 2)
    assert [row["steps"][0]["qa_uid"] for row in kept] == ["a", "c"]


def test_toy_plan_substitutes_each_parent_answer_before_global_pruning():
    plan = [
        {"question": "Who?", "requires_retrieval": True},
        {"question": "When did <ENTITY_Q1> publish?", "requires_retrieval": True},
    ]
    candidates = {
        "Who?": [
            {"qa_uid": "q1", "answer": "Ada", "question": "Who?", "score": 0.9},
            {"qa_uid": "q2", "answer": "Grace", "question": "Who?", "score": 0.8},
        ],
        "When did Ada publish?": [
            {"qa_uid": "q3", "answer": "1843", "question": "When?", "score": 0.7}
        ],
        "When did Grace publish?": [
            {"qa_uid": "q4", "answer": "1952", "question": "When?", "score": 0.95}
        ],
    }
    config = replace(RunConfig(), beam_width=2, candidates_per_hop=2)
    result = execute_plan(plan, lambda query, k: candidates[query][:k], config)
    assert [chain["steps"][-1]["answer"] for chain in result["chains"]] == [
        "1952",
        "1843",
    ]


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


def test_depth_scheduled_hops_match_expected_substitution():
    plan = [
        {"question": "Who?", "requires_retrieval": True},
        {"question": "When did <ENTITY_Q1> publish?", "requires_retrieval": True},
    ]
    state = {"step_ids": [1, 2], "hop_index": 0, "beams": [], "trace": []}
    candidates = {
        "Who?": [{"qa_uid": "q1", "answer": "Ada", "question": "Who?", "score": 0.9}],
        "When did Ada publish?": [
            {"qa_uid": "q2", "answer": "1843", "question": "When?", "score": 0.8}
        ],
    }
    for _ in range(2):
        advance_branch_one_hop(
            plan,
            state,
            lambda query, k: candidates[query][:k],
            beam_width=2,
            candidates_per_hop=2,
        )
    assert state["hop_index"] == 2
    assert state["beams"][0]["steps"][-1]["answer"] == "1843"
    assert state["trace"][1]["issued_queries"] == ["When did Ada publish?"]
