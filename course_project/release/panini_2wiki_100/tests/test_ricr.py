from __future__ import annotations

import inspect

import pytest

from panini_course.ricr import (
    Candidate,
    ChainState,
    geometric_mean,
    instantiate_question,
    prune_unique_answers,
    run_linear_ricr,
)


RICR_IS_SCAFFOLD = (
    "NotImplementedError" in inspect.getsource(prune_unique_answers)
    or "NotImplementedError" in inspect.getsource(run_linear_ricr)
)
requires_ricr_implementation = pytest.mark.skipif(
    RICR_IS_SCAFFOLD,
    reason="Complete the two RICR TODOs to enable this test.",
)


def test_instantiate_question_requires_resolved_dependencies():
    assert (
        instantiate_question("When did <ENTITY_Q1> die?", {1: "Ada"})
        == "When did Ada die?"
    )
    with pytest.raises(KeyError, match="unresolved"):
        instantiate_question("When did <ENTITY_Q1> die?", {})


def test_geometric_mean_is_length_normalized():
    assert geometric_mean([0.81, 1.0]) == pytest.approx(0.9)
    assert geometric_mean([]) == 0.0


@requires_ricr_implementation
def test_pruning_keeps_best_chain_per_normalized_answer():
    def chain(uid: str, answer: str, score: float) -> ChainState:
        candidate = Candidate(uid, answer, score)
        return ChainState((candidate,), {1: answer}, score)

    kept = prune_unique_answers(
        [
            chain("qa-low", "  Ada   Lovelace ", 0.70),
            chain("qa-high", "ada lovelace", 0.90),
            chain("qa-grace", "Grace Hopper", 0.80),
        ],
        beam_width=5,
    )

    assert [item.steps[0].qa_uid for item in kept] == [
        "qa-high",
        "qa-grace",
    ]


@requires_ricr_implementation
def test_pruning_uses_qa_ids_to_break_score_ties():
    left = ChainState(
        (Candidate("qa-b", "Beta", 0.8),),
        {1: "Beta"},
        0.8,
    )
    right = ChainState(
        (Candidate("qa-a", "Alpha", 0.8),),
        {1: "Alpha"},
        0.8,
    )

    kept = prune_unique_answers([left, right], beam_width=1)

    assert kept[0].steps[0].qa_uid == "qa-a"


@requires_ricr_implementation
def test_linear_ricr_returns_empty_when_retrieval_returns_nothing():
    beams = run_linear_ricr(
        [{"question": "Who?", "requires_retrieval": True}],
        lambda query, top_k: [],
    )

    assert beams == []


@requires_ricr_implementation
def test_linear_ricr_propagates_answers_and_prunes_beams():
    candidates = {
        "Who was Lothair II's mother?": [
            Candidate("qa1", "Ermengarde of Tours", 0.92),
            Candidate("qa2", "Ermengarde of Hesbaye", 0.78),
        ],
        "When did Ermengarde of Tours die?": [
            Candidate("qa3", "20 March 851", 0.94),
        ],
        "When did Ermengarde of Hesbaye die?": [
            Candidate("qa4", "3 October 818", 0.60),
        ],
    }

    def retrieve(query: str, top_k: int):
        return candidates[query][:top_k]

    beams = run_linear_ricr(
        [
            {
                "question": "Who was Lothair II's mother?",
                "requires_retrieval": True,
            },
            {
                "question": "When did <ENTITY_Q1> die?",
                "requires_retrieval": True,
            },
        ],
        retrieve,
        beam_width=2,
    )

    assert beams[0].current_answer == "20 March 851"
    assert beams[0].score == pytest.approx(geometric_mean([0.92, 0.94]))


@requires_ricr_implementation
def test_beam_width_one_is_greedy_but_still_runs_all_hops():
    calls = []

    def retrieve(query: str, top_k: int):
        calls.append((query, top_k))
        if query == "Who founded the lab?":
            return [
                Candidate("qa1", "Ada", 0.9),
                Candidate("qa2", "Grace", 0.8),
            ]
        return [Candidate("qa3", "1843", 0.95)]

    beams = run_linear_ricr(
        [
            {"question": "Who founded the lab?", "requires_retrieval": True},
            {
                "question": "When did <ENTITY_Q1> publish?",
                "requires_retrieval": True,
            },
        ],
        retrieve,
        beam_width=1,
        candidates_per_hop=7,
    )

    assert calls == [
        ("Who founded the lab?", 7),
        ("When did Ada publish?", 7),
    ]
    assert len(beams) == 1
    assert len(beams[0].steps) == 2
