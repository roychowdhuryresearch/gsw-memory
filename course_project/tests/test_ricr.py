from __future__ import annotations

import pytest

from panini_course.ricr import (
    Candidate,
    geometric_mean,
    instantiate_question,
    run_linear_ricr,
)


def test_instantiate_question_requires_resolved_dependencies():
    assert (
        instantiate_question("When did <ENTITY_Q1> die?", {1: "Ada"})
        == "When did Ada die?"
    )
    with pytest.raises(KeyError, match="unresolved"):
        instantiate_question("When did <ENTITY_Q1> die?", {})


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
