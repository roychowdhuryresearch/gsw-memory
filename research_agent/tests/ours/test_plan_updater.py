"""Unit tests for the plan-updater agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from research_agent.adapters.ours._plan_updater import (
    PlanReconcileDiff,
    _reconcile_state,
    update_plan,
)
from research_agent.adapters.ours._planner_emit import PlanEmitError
from research_agent.adapters.ours._planner_exec import BlankResult, GSWPlan


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubResp:
    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    prompt_tokens: int = 10
    completion_tokens: int = 20
    reasoning_content: str = ""


class _ScriptedLLM:
    def __init__(self, script: list[_StubResp]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=None, **kwargs):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, **kwargs})
        if not self.script:
            return _StubResp(text="(no script)")
        return self.script.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _old_plan() -> GSWPlan:
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "named", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    })


def _new_plan_same_ids_plus_one() -> dict[str, Any]:
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "b_new", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_new"},
            {"id": "vp3", "phrase": "named", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reconcile_state_preserves_surviving_ids_and_inits_new():
    old = _old_plan()
    new = GSWPlan.model_validate(_new_plan_same_ids_plus_one())
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved", evidence_chunk_ids=["c1"]),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    diff = _reconcile_state(old, new, state)
    assert diff.preserved_ids == ["b_a", "t"]
    assert diff.added_ids == ["b_new"]
    assert diff.dropped_ids == []
    # Resolved value must carry.
    assert state["b_a"].status == "resolved"
    assert state["b_a"].value == "X"
    assert state["b_a"].evidence_chunk_ids == ["c1"]
    # New blank initialised unknown.
    assert state["b_new"].status == "unknown"


def test_reconcile_state_drops_obsolete_ids():
    old = _old_plan()
    # New plan drops b_a entirely (replaces it with b_other).
    new_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_other", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "other", "subject_id": "e1", "object_id": "b_other"},
            {"id": "vp2", "phrase": "named", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }
    new = GSWPlan.model_validate(new_dict)
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    diff = _reconcile_state(old, new, state)
    assert diff.dropped_ids == ["b_a"]
    assert diff.added_ids == ["b_other"]
    assert "b_a" not in state  # dropped
    assert "b_other" in state
    assert state["b_other"].status == "unknown"


def test_update_plan_happy_path():
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    new_plan_json = json.dumps(_new_plan_same_ids_plus_one())
    llm = _ScriptedLLM([_StubResp(text=new_plan_json)])
    new_plan, diff, meta = update_plan(
        old_plan=old,
        state=state,
        question="some question",
        reason="evidence suggests missing attribute",
        evidence="retrieval surfaced a second attribute B",
        llm_client=llm,
    )
    assert "b_new" in {e.id for e in new_plan.blank_entities()}
    assert diff.added_ids == ["b_new"]
    assert state["b_a"].value == "X"  # preserved
    assert state["b_new"].status == "unknown"
    assert meta.completion_tokens == 20


def test_update_plan_repair_retry_fixes_bad_json():
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    # First response bad JSON, second valid.
    llm = _ScriptedLLM(
        [
            _StubResp(text="this is not json"),
            _StubResp(text=json.dumps(_new_plan_same_ids_plus_one())),
        ]
    )
    new_plan, diff, meta = update_plan(
        old_plan=old, state=state, question="Q",
        reason="r", evidence="e", llm_client=llm,
    )
    assert meta.repair_used is True
    assert "b_new" in {e.id for e in new_plan.blank_entities()}


def test_update_plan_raises_when_both_attempts_fail():
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    llm = _ScriptedLLM([_StubResp(text="garbage"), _StubResp(text="still garbage")])
    with pytest.raises(PlanEmitError) as exc_info:
        update_plan(
            old_plan=old, state=state, question="Q",
            reason="r", evidence="e", llm_client=llm,
        )
    assert exc_info.value.kind == "parse_failure"
