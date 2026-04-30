"""Unit tests for the planner-emit helper (multi-retry repair + exhausted)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from research_agent.adapters.ours._planner_emit import (
    MAX_REPAIR_ATTEMPTS,
    PlanEmitError,
    emit_plan,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubResp:
    text: str = ""
    prompt_tokens: int = 5
    completion_tokens: int = 7
    reasoning_content: str = ""


class _ScriptedLLM:
    """Returns scripted responses one at a time."""

    def __init__(self, script: list[_StubResp]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=None, **kwargs):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, **kwargs})
        if not self.script:
            return _StubResp(text="(out of script)")
        return self.script.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_plan_json(question: str = "What is X?") -> str:
    """A minimal grounded GSWPlan that passes validation.

    Filled name "X" is a substring of the question text so the plan
    grounding test passes if any extra prompt-side checks are layered
    in. (The pydantic-level validators don't actually enforce
    grounding — the planner-side prompt does — so a tiny plan suffices.)
    """
    return json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "X", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "is", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_emit_plan_first_attempt_success():
    llm = _ScriptedLLM([_StubResp(text=_valid_plan_json())])
    plan, meta = emit_plan("What is X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert meta.repair_attempts == []
    assert len(llm.calls) == 1


def test_emit_plan_repairs_after_one_failure():
    """First call returns junk; first repair returns valid JSON."""
    llm = _ScriptedLLM([
        _StubResp(text="not json"),
        _StubResp(text=_valid_plan_json()),
    ])
    plan, meta = emit_plan("What is X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert len(meta.repair_attempts) == 1
    assert meta.repair_attempts[0]["error"] == ""
    assert meta.parse_error  # captured first failure
    assert len(llm.calls) == 2


def test_emit_plan_passes_seed_to_initial_and_repair_calls():
    llm = _ScriptedLLM([
        _StubResp(text="not json"),
        _StubResp(text=_valid_plan_json()),
    ])
    plan, _meta = emit_plan("What is X?", llm, llm_seed=123)
    assert plan.target().id == "t"
    assert [c.get("seed") for c in llm.calls] == [123, 123]


def test_emit_plan_repairs_on_third_attempt():
    """First call + 2 bad repairs, third repair succeeds."""
    llm = _ScriptedLLM([
        _StubResp(text="not json"),
        _StubResp(text="still not json"),
        _StubResp(text="more garbage"),
        _StubResp(text=_valid_plan_json()),
    ])
    plan, meta = emit_plan("What is X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert len(meta.repair_attempts) == 3
    assert meta.repair_attempts[0]["error"]
    assert meta.repair_attempts[1]["error"]
    assert meta.repair_attempts[2]["error"] == ""
    assert len(llm.calls) == 4


def test_emit_plan_raises_exhausted_after_all_retries():
    """First + 3 bad repairs → PlanEmitError(kind='exhausted')."""
    llm = _ScriptedLLM([
        _StubResp(text="garbage 0"),
        _StubResp(text="garbage 1"),
        _StubResp(text="garbage 2"),
        _StubResp(text="garbage 3"),
    ])
    with pytest.raises(PlanEmitError) as exc_info:
        emit_plan("What is X?", llm)
    assert exc_info.value.kind == "exhausted"
    assert "3 repair attempts" in exc_info.value.detail
    assert len(llm.calls) == 1 + MAX_REPAIR_ATTEMPTS  # 4 total


def test_emit_plan_repair_messages_sharpen_each_round():
    """The repair messages on attempts 2 and 3 must mention the attempt
    index so the LLM knows it has already failed earlier rounds."""
    llm = _ScriptedLLM([
        _StubResp(text="g0"),
        _StubResp(text="g1"),
        _StubResp(text="g2"),
        _StubResp(text=_valid_plan_json()),
    ])
    emit_plan("What is X?", llm)

    # Repair-call messages are at call indices 1, 2, 3.
    repair2_user_msg = llm.calls[2]["messages"][-1]["content"]
    repair3_user_msg = llm.calls[3]["messages"][-1]["content"]

    assert "attempt 2" in repair2_user_msg
    assert "FINAL repair attempt" in repair3_user_msg


def test_emit_plan_first_repair_uses_legacy_preface():
    """Attempt 1 of repair preserves the original (non-sharpened) tone."""
    llm = _ScriptedLLM([
        _StubResp(text="garbage"),
        _StubResp(text=_valid_plan_json()),
    ])
    emit_plan("What is X?", llm)
    repair1_user_msg = llm.calls[1]["messages"][-1]["content"]
    # First repair must NOT carry the "attempt N" preface.
    assert "attempt 2" not in repair1_user_msg
    assert "FINAL repair attempt" not in repair1_user_msg
    # But it must still carry the structural rules.
    assert "no dangling entities" in repair1_user_msg


def test_emit_plan_disable_repair_raises_parse_failure():
    """enable_repair=False short-circuits to parse_failure on first bad JSON."""
    llm = _ScriptedLLM([_StubResp(text="not json")])
    with pytest.raises(PlanEmitError) as exc_info:
        emit_plan("What is X?", llm, enable_repair=False)
    assert exc_info.value.kind == "parse_failure"
    assert len(llm.calls) == 1


def test_emit_plan_llm_exception_during_repair():
    """LLM-level exception during a repair call surfaces as llm_error."""

    class _BombingLLM:
        def __init__(self):
            self.calls: list[Any] = []

        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return _StubResp(text="not json")
            raise RuntimeError("transport boom")

    llm = _BombingLLM()
    with pytest.raises(PlanEmitError) as exc_info:
        emit_plan("What is X?", llm)
    assert exc_info.value.kind == "llm_error"
    assert "boom" in exc_info.value.detail
