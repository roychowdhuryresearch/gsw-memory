import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gsw_memory.query_then_sleep.query_agent import QueryAgent


def test_normalize_final_answer_reduces_yes_no_sentences():
    assert (
        QueryAgent._normalize_final_answer(
            "Were Jimmy Santiago Baca and Duane Armstrong from the same country?",
            "Yes, Jimmy Santiago Baca and Duane Armstrong are from the same country – the United States.",
        )
        == "yes"
    )


def test_normalize_final_answer_extracts_answer_marker():
    assert (
        QueryAgent._normalize_final_answer(
            "Where was the director of film En Aasai Rasave born?",
            "Answer: Mallingapuram.",
        )
        == "Mallingapuram"
    )


def test_extract_short_answer_candidate_strips_sentence_and_doc_ref():
    assert (
        QueryAgent._extract_short_answer_candidate(
            "Where did John Ii, Duke Of Cleves's wife die?",
            "John II's wife, Mathilde of Hesse, died in **Cologne**【doc_4626::e5】",
        )
        == "Cologne"
    )


def test_choose_final_answer_prefers_short_candidate_when_structured_answer_is_long():
    assert (
        QueryAgent._choose_final_answer(
            question="Who is the spouse of the director of film Pratisodh (2004 Film)?",
            synthesis_answer="The spouse of the director of Pratisodh (2004 Film) is Piya Sengupta.",
            synthesis_raw_text='{"answer": "The spouse of the director of Pratisodh (2004 Film) is Piya Sengupta."}',
            tool_loop_final_text="",
        )
        == "Piya Sengupta"
    )
