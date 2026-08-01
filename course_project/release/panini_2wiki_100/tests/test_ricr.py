from __future__ import annotations

import inspect

import pytest

from panini_course.ricr import (
    Candidate,
    geometric_mean,
    harmonic_mean,
    identify_retrieval_components,
    instantiate_question,
    run_panini_ricr,
)


RICR_IS_SCAFFOLD = "NotImplementedError" in inspect.getsource(run_panini_ricr)
requires_ricr_implementation = pytest.mark.skipif(
    RICR_IS_SCAFFOLD,
    reason="Complete run_panini_ricr to enable this test.",
)


def qa(uid, answers, score, *, ids=()):
    return Candidate(uid, tuple(answers), score, question=uid, answer_ids=tuple(ids))


def test_instantiation_resolves_every_named_dependency():
    assert (
        instantiate_question(
            "Compare <ENTITY_Q1> with <ENTITY_Q2>", {1: "Ada", 2: "Grace"}
        )
        == "Compare Ada with Grace"
    )
    with pytest.raises(KeyError, match="unresolved"):
        instantiate_question("When did <ENTITY_Q1> die?", {})


def test_score_helpers_are_length_normalized_and_penalize_weak_parents():
    assert geometric_mean([0.81, 1.0]) == pytest.approx(0.9)
    assert harmonic_mean([0.9, 0.1]) < 0.2


@requires_ricr_implementation
def test_components_form_one_topologically_sorted_converging_dag():
    plan = [
        {"question": "Root A", "requires_retrieval": True},
        {"question": "Root B", "requires_retrieval": True},
        {
            "question": "Compare <ENTITY_Q1> with <ENTITY_Q2>",
            "requires_retrieval": True,
        },
    ]
    assert identify_retrieval_components(plan) == [[1, 2, 3]]


@requires_ricr_implementation
def test_multi_parent_step_combines_and_substitutes_both_parent_beams():
    plan = [
        {"question": "Root A", "requires_retrieval": True},
        {"question": "Root B", "requires_retrieval": True},
        {
            "question": "Compare <ENTITY_Q1> with <ENTITY_Q2>",
            "requires_retrieval": True,
        },
    ]
    table = {
        "Root A": [qa("a", ["Ada"], 0.9, ids=["doc-a::e1"])],
        "Root B": [qa("b", ["Grace"], 0.8, ids=["doc-b::e1"])],
        "Compare Ada with Grace": [qa("c", ["Ada"], 0.95)],
    }
    result = run_panini_ricr(
        plan,
        lambda query, top_k: table[query][:top_k],
        original_question="Who is older?",
        beam_width=2,
    )
    assert "Compare Ada with Grace" in result.issued_queries
    assert [step.qa_uid for step in result.chains[0].steps] == ["a", "b", "c"]


@requires_ricr_implementation
def test_intermediate_hop_groups_entities_but_final_hop_keeps_qa_alternatives():
    plan = [
        {"question": "Find person", "requires_retrieval": True},
        {"question": "Fact about <ENTITY_Q1>", "requires_retrieval": True},
    ]
    table = {
        "Find person": [
            qa("ada-low", ["Ada"], 0.1, ids=["doc::ada"]),
            qa("ada-high", ["Ada Lovelace"], 0.9, ids=["doc::ada"]),
            qa("grace", ["Grace"], 0.8, ids=["doc::grace"]),
        ],
        "Fact about Ada Lovelace": [
            qa("final-a", ["1843"], 0.9),
            qa("final-b", ["1843"], 0.8),
        ],
        "Fact about Grace": [qa("final-c", ["1952"], 0.7)],
    }
    result = run_panini_ricr(
        plan,
        lambda query, top_k: table[query][:top_k],
        original_question="Find the fact",
        beam_width=3,
        candidates_per_hop=3,
    )
    assert {chain.steps[0].qa_uid for chain in result.chains} == {
        "ada-high",
        "grace",
    }
    assert {chain.steps[-1].qa_uid for chain in result.chains} >= {
        "final-a",
        "final-b",
    }


@requires_ricr_implementation
def test_evidence_is_union_of_all_final_beams_not_only_the_best_chain():
    plan = [
        {"question": "Root", "requires_retrieval": True},
        {"question": "Finish <ENTITY_Q1>", "requires_retrieval": True},
    ]
    table = {
        "Root": [
            qa("root-a", ["Ada"], 0.9, ids=["doc::ada"]),
            qa("root-b", ["Grace"], 0.8, ids=["doc::grace"]),
        ],
        "Finish Ada": [qa("answer-a", ["1843"], 0.9)],
        "Finish Grace": [qa("answer-b", ["1952"], 0.8)],
    }
    result = run_panini_ricr(
        plan,
        lambda query, top_k: table[query][:top_k],
        original_question="When?",
        beam_width=2,
    )
    assert {candidate.qa_uid for candidate in result.evidence} == {
        "root-a",
        "root-b",
        "answer-a",
        "answer-b",
    }


@requires_ricr_implementation
def test_singleton_plan_uses_original_question_fallback():
    calls = []

    def retrieve(query, top_k):
        calls.append(query)
        return [qa("one", ["answer"], 0.9)]

    result = run_panini_ricr(
        [{"question": "A rewritten question", "requires_retrieval": True}],
        retrieve,
        original_question="The original question",
    )
    assert result.fallback
    assert calls == ["The original question"]
