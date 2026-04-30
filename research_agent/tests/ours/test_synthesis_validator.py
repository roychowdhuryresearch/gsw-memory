"""Unit tests for the synthesis validator (Phase-3.7)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from research_agent.adapters.ours._planner_exec import (
    BlankResult,
    Constraint,
    Entity,
    GSWPlan,
    VerbPhrase,
)
from research_agent.adapters.ours._synthesis_validator import (
    ValidatorVerdict,
    auto_repair_candidate,
    build_rejection_message,
    build_validator_messages,
    validate_synthesis,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubChunk:
    chunk_id: str
    title: str
    text: str


class _StubCorpus:
    def __init__(self, chunks: dict[str, _StubChunk]) -> None:
        self._chunks = dict(chunks)

    def get_chunk(self, chunk_id: str):
        return self._chunks.get(chunk_id)


@dataclass
class _StubResponse:
    text: str = ""
    prompt_tokens: int = 5
    completion_tokens: int = 7
    reasoning_content: str = ""


class _ScriptedLLM:
    def __init__(self, script: list[_StubResponse] | None = None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.script:
            return _StubResponse(text="(no script)")
        return self.script.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_blank_plan() -> GSWPlan:
    """Plan: e1 has_year b_yr; e2 won_at_age t. Target=t."""
    return GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Moana", role="subject"),
            Entity(id="b_yr", kind="blank", value_type="number", role="bridge-number"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True, role="target"),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="released_in", subject_id="e1", object_id="b_yr"),
            VerbPhrase(id="vp2", phrase="ws_loser_coach", subject_id="b_yr", object_id="t"),
        ],
    )


def _resolved_state(corpus: _StubCorpus) -> dict[str, BlankResult]:
    return {
        "b_yr": BlankResult(
            blank_id="b_yr", value="2016", status="resolved",
            evidence_chunk_ids=["c_moana_1"],
        ),
        "t": BlankResult(
            blank_id="t", value="Mike Hargrove", status="resolved",
            evidence_chunk_ids=["c_indians_2016"],
        ),
    }


def _stub_corpus() -> _StubCorpus:
    return _StubCorpus({
        "c_moana_1": _StubChunk(
            chunk_id="c_moana_1", title="Moana",
            text="Moana is a 2016 American animated musical adventure film...",
        ),
        "c_indians_2016": _StubChunk(
            chunk_id="c_indians_2016", title="2016 Cleveland Indians",
            text="The 2016 Cleveland Indians lost the World Series. "
                 "Hitting coach was Ty Van Burkleo. "
                 "Mike Hargrove was a former manager of the team but not the 2016 hitting coach.",
        ),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validator_skips_empty_pred():
    """Empty pred → no LLM call, returns 'correct' immediately."""
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    llm = _ScriptedLLM()
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="",
        corpus=corpus, llm=llm,
    )
    assert v.verdict == "correct"
    assert "skipped" in v.reason
    assert llm.calls == []


def test_validator_returns_correct_on_valid_json():
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    llm = _ScriptedLLM([_StubResponse(text=json.dumps({
        "verdict": "correct",
        "reason": "matches retrieved evidence",
    }))])
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="Mike Hargrove",
        corpus=corpus, llm=llm,
    )
    assert v.verdict == "correct"
    assert "matches" in v.reason
    assert len(llm.calls) == 1


def test_validator_returns_wrong_with_flagged_blank_and_correction():
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    llm = _ScriptedLLM([_StubResponse(text=json.dumps({
        "verdict": "wrong",
        "reason": "Mike Hargrove was former manager, not 2016 hitting coach",
        "flagged_blank": "t",
        "suggested_correction": "Ty Van Burkleo",
    }))])
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="Mike Hargrove",
        corpus=corpus, llm=llm,
    )
    assert v.verdict == "wrong"
    assert v.flagged_blank == "t"
    assert v.suggested_correction == "Ty Van Burkleo"
    assert v.corrected_answer == "Ty Van Burkleo"


def test_validator_parses_repair_fields():
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    llm = _ScriptedLLM([_StubResponse(text=json.dumps({
        "verdict": "wrong",
        "reason": "age arithmetic is off",
        "flagged_blank": "t",
        "corrected_answer": "37",
        "confidence": "high",
        "error_type": "arithmetic",
        "evidence_support": "born 1899 and died aged 37",
    }))])
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="38",
        corpus=corpus, llm=llm,
    )
    assert v.verdict == "wrong"
    assert v.corrected_answer == "37"
    assert v.confidence == "high"
    assert v.error_type == "arithmetic"
    assert "born 1899" in (v.evidence_support or "")


def test_auto_repair_candidate_accepts_high_confidence_safe_fix():
    v = ValidatorVerdict(
        verdict="wrong",
        reason="arithmetic is off",
        corrected_answer="37",
        confidence="high",
        error_type="arithmetic",
    )
    repaired, reason = auto_repair_candidate(v, "38")
    assert repaired == "37"
    assert reason == "accepted"


def test_auto_repair_candidate_rejects_low_confidence_or_unsafe_type():
    low = ValidatorVerdict(
        verdict="wrong",
        reason="maybe wrong",
        corrected_answer="37",
        confidence="medium",
        error_type="arithmetic",
    )
    assert auto_repair_candidate(low, "38")[0] is None

    unsafe = ValidatorVerdict(
        verdict="wrong",
        reason="wrong entity",
        corrected_answer="Other Person",
        confidence="high",
        error_type="wrong_entity",
    )
    repaired, reason = auto_repair_candidate(unsafe, "Person")
    assert repaired is None
    assert "unsafe error_type" in reason


def test_validator_falls_back_to_correct_on_unparseable_response():
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    llm = _ScriptedLLM([_StubResponse(text="not json at all")])
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="X",
        corpus=corpus, llm=llm,
    )
    assert v.verdict == "correct"
    assert "unparseable" in v.reason


def test_validator_falls_back_to_correct_on_llm_exception():
    class _BombLLM:
        def chat(self, *a, **k):
            raise RuntimeError("transport failed")
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="X",
        corpus=corpus, llm=_BombLLM(),
    )
    assert v.verdict == "correct"
    assert "validator LLM error" in v.reason


def test_validator_keeps_wrong_when_suggestion_missing():
    """v1 behaviour: a 'wrong' verdict without a suggestion still
    rejects — the orchestrator gets to re-think with whatever reason
    was given. (v2 used to downgrade these to 'correct'; reverted.)"""
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    llm = _ScriptedLLM([_StubResponse(text=json.dumps({
        "verdict": "wrong",
        "reason": "the answer doesn't seem right",
        "suggested_correction": None,
    }))])
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="X",
        corpus=corpus, llm=llm,
    )
    assert v.verdict == "wrong"
    assert v.suggested_correction is None


def test_validator_falls_back_to_correct_on_unexpected_verdict():
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    llm = _ScriptedLLM([_StubResponse(text=json.dumps({
        "verdict": "maybe",
        "reason": "not sure",
    }))])
    v = validate_synthesis(
        question="Q", plan=plan, state=state, proposed_answer="X",
        corpus=corpus, llm=llm,
    )
    assert v.verdict == "correct"


def test_validator_messages_include_evidence_snippets():
    plan = _two_blank_plan()
    corpus = _stub_corpus()
    state = _resolved_state(corpus)
    msgs = build_validator_messages(
        question="In Moana's release year, who was the hitting coach...",
        plan=plan, state=state,
        proposed_answer="Mike Hargrove",
        corpus=corpus,
    )
    user_content = msgs[-1]["content"]
    # Question rendered
    assert "Moana's release year" in user_content
    # Proposed answer rendered
    assert "Mike Hargrove" in user_content
    # Evidence snippets included (full chunk text, no truncation)
    assert "Hitting coach was Ty Van Burkleo" in user_content
    assert "Moana is a 2016" in user_content
    # Target marker
    assert "[TARGET]" in user_content


def test_rejection_message_includes_reason_and_optional_suggestion():
    v = ValidatorVerdict(
        verdict="wrong",
        reason="Hargrove was manager, not coach",
        flagged_blank="t",
        suggested_correction="Ty Van Burkleo",
    )
    msg = build_rejection_message(v, "Mike Hargrove")
    assert "Mike Hargrove" in msg
    assert "REJECTED" in msg
    assert "Hargrove was manager" in msg
    assert "Ty Van Burkleo" in msg
    assert "'t'" in msg
    assert "ONE remaining attempt" in msg
    assert "accepted as-is" in msg


def test_rejection_message_handles_missing_flagged_blank():
    v = ValidatorVerdict(
        verdict="wrong",
        reason="just wrong",
        suggested_correction="42",
    )
    msg = build_rejection_message(v, "X")
    assert "REJECTED" in msg
    assert "just wrong" in msg
    assert "Flagged intermediate" not in msg
    assert "'42'" in msg
