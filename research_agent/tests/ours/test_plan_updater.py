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


# Phase-3.6 — diff-op output contract for update_plan().
#
# The plan-updater LLM now returns ONE diff op (e.g. add_blank) which
# is applied deterministically. Earlier tests fed a full revised plan
# JSON; those have been rewritten to feed diff ops. Rejection sampling
# retries up to MAX_REVISION_ATTEMPTS=3 times when the LLM emits a
# trivial / unparseable / no-op-applying op.


def _add_blank_op_json(blank_id: str = "b_new", vp_id: str = "vp3") -> str:
    """A canonical add_blank op that produces a structurally-changed plan."""
    return json.dumps({
        "op": "add_blank",
        "blank": {
            "id": blank_id,
            "value_type": "text",
            "role": "bridge-attribute",
        },
        "wire_via_vp": {
            "id": vp_id,
            "phrase": "has_b",
            "subject_id": "e1",
            "object_id": blank_id,
        },
    })


def test_update_plan_happy_path_diff_op():
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    llm = _ScriptedLLM([_StubResp(text=_add_blank_op_json())])
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
    assert diff.attempts == 1
    assert "add_blank(b_new" in diff.op_summary
    assert state["b_a"].value == "X"  # preserved
    assert state["b_new"].status == "unknown"
    assert meta.completion_tokens == 20
    assert len(llm.calls) == 1


def test_update_plan_retries_after_unparseable_response():
    """First response is junk (no JSON), retry succeeds."""
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    llm = _ScriptedLLM([
        _StubResp(text="this is not json at all"),
        _StubResp(text=_add_blank_op_json()),
    ])
    new_plan, diff, meta = update_plan(
        old_plan=old, state=state, question="Q",
        reason="r", evidence="e", llm_client=llm,
    )
    assert meta.repair_used is True
    assert "b_new" in {e.id for e in new_plan.blank_entities()}
    assert diff.attempts == 2
    assert len(llm.calls) == 2


def test_update_plan_retries_after_trivial_op():
    """First response is a trivial op (existing blank id), retry succeeds."""
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    # Trivial: try to add a blank with an id that already exists.
    trivial_op = json.dumps({
        "op": "add_blank",
        "blank": {"id": "e1", "value_type": "text"},
        "wire_via_vp": {
            "id": "vp_x", "phrase": "x", "subject_id": "e1", "object_id": "t",
        },
    })
    llm = _ScriptedLLM([
        _StubResp(text=trivial_op),
        _StubResp(text=_add_blank_op_json()),
    ])
    new_plan, diff, meta = update_plan(
        old_plan=old, state=state, question="Q",
        reason="r", evidence="e", llm_client=llm,
    )
    assert "b_new" in {e.id for e in new_plan.blank_entities()}
    assert diff.added_ids == ["b_new"]
    assert diff.attempts == 2
    assert state["b_a"].value == "X"
    assert len(llm.calls) == 2


def test_update_plan_raises_after_3_trivial_ops():
    """If LLM emits trivial ops 3 times, update_plan raises."""
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    trivial_op = json.dumps({
        "op": "change_target", "from_id": "t", "to_id": "t",
    })
    llm = _ScriptedLLM([
        _StubResp(text=trivial_op),
        _StubResp(text=trivial_op),
        _StubResp(text=trivial_op),
    ])
    with pytest.raises(PlanEmitError) as exc_info:
        update_plan(
            old_plan=old, state=state, question="Q",
            reason="r", evidence="e", llm_client=llm,
        )
    assert exc_info.value.kind == "parse_failure"
    assert "could not produce a structural change" in exc_info.value.detail
    # State unchanged after failure.
    assert set(state) == {"b_a", "t"}
    assert state["b_a"].value == "X"
    assert len(llm.calls) == 3


def test_update_plan_raises_after_3_unparseable_responses():
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    llm = _ScriptedLLM([
        _StubResp(text="garbage"),
        _StubResp(text="still garbage"),
        _StubResp(text="more garbage"),
    ])
    with pytest.raises(PlanEmitError) as exc_info:
        update_plan(
            old_plan=old, state=state, question="Q",
            reason="r", evidence="e", llm_client=llm,
        )
    assert exc_info.value.kind == "parse_failure"
    assert len(llm.calls) == 3


def test_update_plan_change_target_op():
    """Different op type — change_target — works end-to-end."""
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    op = json.dumps({"op": "change_target", "from_id": "t", "to_id": "b_a"})
    llm = _ScriptedLLM([_StubResp(text=op)])
    new_plan, diff, meta = update_plan(
        old_plan=old, state=state, question="Q",
        reason="target should be b_a", evidence="e", llm_client=llm,
    )
    targets = [e.id for e in new_plan.blank_entities() if e.is_target]
    assert targets == ["b_a"]
    assert "change_target" in diff.op_summary


def test_update_plan_strips_markdown_fences():
    """LLM sometimes wraps JSON in ```json … ```. Parser must handle it."""
    old = _old_plan()
    state = {
        "b_a": BlankResult(blank_id="b_a", value="X", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    op_json = _add_blank_op_json()
    fenced = f"```json\n{op_json}\n```"
    llm = _ScriptedLLM([_StubResp(text=fenced)])
    new_plan, diff, _meta = update_plan(
        old_plan=old, state=state, question="Q",
        reason="r", evidence="e", llm_client=llm,
    )
    assert "b_new" in {e.id for e in new_plan.blank_entities()}
