"""Unit tests for the planner-scaffolded ReAct adapter.

All tests stub the LLM + retriever — no network access, no real BM25
corpus load. The stub LLM returns scripted tool-call sequences so we
can drive the adapter's loop deterministically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from research_agent.adapters.base import AdapterContext
from research_agent.adapters.ours._planner_exec import BlankResult, GSWPlan
from research_agent.adapters.ours._planner_react_prompt import build_system_prompt
from research_agent.adapters.ours.gsw_planner_react_v1 import (
    _TOOLS,
    OursGSWPlannerReactV1Adapter,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubChunk:
    chunk_id: str
    title: str
    text: str


@dataclass
class _StubHit:
    chunk: _StubChunk
    score: float = 1.0


class _StubCorpus:
    """Minimal corpus: `{title: full_text}` plus a chunk lookup."""

    def __init__(self, chunks: dict[str, _StubChunk]) -> None:
        self._chunks = dict(chunks)
        self._titles = {c.title: c.text for c in chunks.values()}

    def get_chunk(self, chunk_id: str):
        return self._chunks.get(chunk_id)

    def article_text(self, title: str) -> str:
        return self._titles.get(title, "")

    def titles(self) -> list[str]:
        return list(self._titles)


class _StubRetriever:
    def __init__(self, mapping: dict[str, list[_StubChunk]]) -> None:
        self.mapping = mapping

    def search(self, query: str, *, top_k: int = 5):
        for key, chunks in self.mapping.items():
            if key.lower() in query.lower():
                return [_StubHit(chunk=c) for c in chunks[:top_k]]
        return []


@dataclass
class _StubResponse:
    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    prompt_tokens: int = 10
    completion_tokens: int = 20
    reasoning_content: str = ""
    finish_reason: str = "stop"

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class _ScriptedLLM:
    """Returns a pre-scripted sequence of responses on each .chat() call."""

    def __init__(self, script: list[_StubResponse]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        if not self.script:
            return _StubResponse(text="(script exhausted — stop)", tool_calls=[])
        return self.script.pop(0)


def _mk_tool_call(tid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": tid, "name": name, "arguments": json.dumps(args)}


# ---------------------------------------------------------------------------
# Plan fixture (Picasso / Pink Floyd temporal bridge — clean 2-blank case)
# ---------------------------------------------------------------------------


def _fixture_plan() -> dict[str, Any]:
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pablo Picasso", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {
                "id": "vp1",
                "phrase": "died_in",
                "subject_id": "e1",
                "object_id": "b_year",
            },
            {
                "id": "vp2",
                "phrase": "released_in_year",
                "subject_id": "e2",
                "object_id": "b_year",
            },
            {"id": "vp3", "phrase": "released", "subject_id": "e2", "object_id": "t"},
        ],
        "constraints": [],
    }


def _mk_adapter(scripted: _ScriptedLLM, corpus=None, retriever=None, max_turns=10):
    ctx = AdapterContext(
        system_id="ours_gsw_planner_react_v1",
        model_id="stub",
        max_turns=max_turns,
        max_completion_tokens=2000,
        extra={"corpus": corpus or _StubCorpus({}), "retriever": retriever or _StubRetriever({})},
    )
    adapter = OursGSWPlannerReactV1Adapter(ctx)
    adapter.llm = scripted  # replace the real LLM with our scripted one
    return adapter


# ---------------------------------------------------------------------------
# Dispatcher-level tests (no full run_question loop)
# ---------------------------------------------------------------------------


def test_update_blank_writes_state():
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, counters = adapter._make_plan_dispatcher(plan, state)

    result, finish_ok = dispatch(
        "update_blank",
        json.dumps({"blank_id": "b_year", "value": "1973", "evidence_chunk_ids": ["chunk_1"]}),
    )

    assert result["ok"] is True
    assert not finish_ok
    assert state["b_year"].status == "resolved"
    assert state["b_year"].value == "1973"
    assert state["b_year"].evidence_chunk_ids == ["chunk_1"]
    assert counters["updates"] == 1


def test_update_blank_rejects_sentinel_as_value():
    """Reject value='insufficient_evidence' — that marker belongs in
    evidence_chunk_ids, never as the actual value. The model sometimes
    mis-binds parameters; the adapter must catch it."""
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    result, finish_ok = dispatch(
        "update_blank",
        json.dumps(
            {
                "blank_id": "t",
                "value": "insufficient_evidence",
                "evidence_chunk_ids": [],
            }
        ),
    )
    assert result["ok"] is False
    assert "best guess" in result["error"] or "best inference" in result["error"]
    assert "insufficient_evidence" in result["error"]  # error names the problem
    assert state["t"].status == "unknown"  # no write happened
    assert not finish_ok

    # Same string with funky casing/whitespace must also reject.
    result2, _ = dispatch(
        "update_blank",
        json.dumps(
            {
                "blank_id": "t",
                "value": "  INSUFFICIENT_EVIDENCE  ",
                "evidence_chunk_ids": ["chunk_1"],
            }
        ),
    )
    assert result2["ok"] is False

    # But a legitimate commit with the sentinel in evidence_chunk_ids
    # must still succeed.
    result3, _ = dispatch(
        "update_blank",
        json.dumps(
            {
                "blank_id": "t",
                "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["insufficient_evidence"],
            }
        ),
    )
    assert result3["ok"] is True
    assert state["t"].status == "resolved"
    assert state["t"].evidence_chunk_ids == ["insufficient_evidence"]


def test_auto_compute_concat_fires_when_inputs_resolved():
    """v5.2 Fix A: when a concat constraint's input blanks are all
    resolved via update_blank, the adapter auto-fills the output blank
    without the model needing to call update_blank on the target."""
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Gymnast", "role": "scope-filter"},
            {"id": "e2", "kind": "filled", "name": "Swimmer", "role": "scope-filter"},
            {"id": "b_mid", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "b_first", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "aggregate-output",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "middle_name_of", "subject_id": "e1", "object_id": "b_mid"},
            {"id": "vp2", "phrase": "first_name_of", "subject_id": "e2", "object_id": "b_first"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "concat",
                "args_blanks": ["b_mid", "b_first"],
                "output_blank_id": "t",
            }
        ],
    }
    plan = GSWPlan.model_validate(plan_dict)
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, counters = adapter._make_plan_dispatcher(plan, state)

    # Resolve first input; target should still be unknown.
    result, _ = dispatch(
        "update_blank",
        json.dumps({"blank_id": "b_mid", "value": "Arianne", "evidence_chunk_ids": ["c1"]}),
    )
    assert result["ok"] is True
    assert state["t"].status == "unknown"
    assert "auto_computed" not in result  # nothing fired yet

    # Resolve second input; target auto-fires.
    result2, _ = dispatch(
        "update_blank",
        json.dumps({"blank_id": "b_first", "value": "Kathleen", "evidence_chunk_ids": ["c2"]}),
    )
    assert result2["ok"] is True
    assert state["t"].status == "resolved"
    # concat op in the executor joins with "; " per _compute_constraint.
    assert "Arianne" in str(state["t"].value)
    assert "Kathleen" in str(state["t"].value)
    assert result2.get("auto_computed") == ["t"]
    assert counters["auto_computed"] == 1


def test_auto_compute_derives_diff_and_allows_finish():
    """After auto-compute fires, finish should be acceptable without the
    model needing to call update_blank on the target."""
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "CityA", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "CityB", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_b", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {
                "id": "t",
                "kind": "blank",
                "role": "aggregate-output",
                "value_type": "number",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_x", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_x", "subject_id": "e2", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "diff",
                "args_blanks": ["b_a", "b_b"],
                "output_blank_id": "t",
            }
        ],
    }
    plan = GSWPlan.model_validate(plan_dict)
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    dispatch("update_blank", json.dumps({"blank_id": "b_a", "value": 820000, "evidence_chunk_ids": []}))
    dispatch("update_blank", json.dumps({"blank_id": "b_b", "value": 615000, "evidence_chunk_ids": []}))
    # Target auto-filled via diff.
    assert state["t"].status == "resolved"
    assert abs(state["t"].value - 205000) < 1e-6

    # finish now works without an explicit update_blank on target.
    fresult, fok = dispatch("finish", json.dumps({"answer": "205000"}))
    assert fresult["ok"] is True
    assert fok


def test_auto_compute_does_not_overwrite_explicit_target_update():
    """If the model committed the target blank manually, cascade must not
    overwrite it even though the constraint's inputs are resolved."""
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "X", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Y", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_b", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {
                "id": "t",
                "kind": "blank",
                "role": "aggregate-output",
                "value_type": "number",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_x", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_x", "subject_id": "e2", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "diff",
                "args_blanks": ["b_a", "b_b"],
                "output_blank_id": "t",
            }
        ],
    }
    plan = GSWPlan.model_validate(plan_dict)
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    # Commit the target FIRST with a manual value.
    dispatch(
        "update_blank",
        json.dumps({"blank_id": "t", "value": 999999, "evidence_chunk_ids": ["manual"]}),
    )
    # Now resolve the inputs.
    dispatch("update_blank", json.dumps({"blank_id": "b_a", "value": 820000, "evidence_chunk_ids": []}))
    dispatch("update_blank", json.dumps({"blank_id": "b_b", "value": 615000, "evidence_chunk_ids": []}))
    # Cascade should NOT overwrite the manually-committed target.
    assert state["t"].value == 999999


def test_update_blank_rejects_unknown_id():
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    result, finish_ok = dispatch(
        "update_blank",
        json.dumps({"blank_id": "bogus", "value": "x", "evidence_chunk_ids": ["chunk_1"]}),
    )

    assert result["ok"] is False
    assert "unknown blank_id" in result["error"]
    assert not finish_ok


def test_update_blank_requires_evidence_ids():
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    result, finish_ok = dispatch(
        "update_blank",
        json.dumps({"blank_id": "b_year", "value": "1973"}),
    )

    assert "bad tool call" in result["error"]
    assert not finish_ok


def test_finish_rejected_when_target_unresolved():
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, counters = adapter._make_plan_dispatcher(plan, state)

    result, finish_ok = dispatch("finish", json.dumps({"answer": "Dark Side"}))

    assert result["ok"] is False
    assert not finish_ok
    assert "t" in result["unresolved_blanks"]
    assert counters["finish_rejected"] == 1


def test_finish_accepted_when_target_resolved():
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    # Resolve target first.
    dispatch(
        "update_blank",
        json.dumps({"blank_id": "t", "value": "Dark Side of the Moon", "evidence_chunk_ids": []}),
    )
    result, finish_ok = dispatch("finish", json.dumps({"answer": "Dark Side of the Moon"}))

    assert result["ok"] is True
    assert finish_ok
    assert result["answer"] == "Dark Side of the Moon"


def test_get_state_shape():
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    result, _ = dispatch("get_state", "{}")
    assert set(result.keys()) == {"blanks", "target_id", "target_status"}
    assert result["target_id"] == "t"
    assert result["target_status"] == "unknown"
    assert len(result["blanks"]) == 2
    blank_ids = sorted(b["blank_id"] for b in result["blanks"])
    assert blank_ids == ["b_year", "t"]


def test_unknown_tool_returns_error():
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    adapter = _mk_adapter(_ScriptedLLM([]))
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)

    result, finish_ok = dispatch("frobnicate", "{}")
    assert "unknown tool" in result["error"]
    assert not finish_ok


def test_find_tool_returns_snippets_around_matches():
    """`find(chunk_id, pattern)` locates substrings in the full article
    and returns ±200-char snippets around up to 3 matches."""
    article = (
        "A" * 300
        + "Ghostface Killah was born Dennis Coles on May 9, 1970."
        + "B" * 300
        + "Ghostface Killah released Fishscale in 2006."
        + "C" * 100
    )
    chunk = _StubChunk(chunk_id="gfk__0", title="Ghostface Killah", text=article)
    corpus = _StubCorpus({"gfk__0": chunk})
    adapter = _mk_adapter(_ScriptedLLM([]), corpus=corpus)

    result = adapter._tool_find(chunk_id="gfk__0", pattern="Ghostface Killah")
    assert result["match_count"] == 2
    assert len(result["snippets"]) == 2
    # Each snippet covers ±200 chars around the match.
    for snip in result["snippets"]:
        assert "ghostface killah" in snip["snippet"].lower()
        assert len(snip["snippet"]) <= 200 + len("Ghostface Killah") + 200

    # Unknown chunk → error.
    miss = adapter._tool_find(chunk_id="nope", pattern="x")
    assert "error" in miss and "not found" in miss["error"]

    # Empty pattern → error.
    empty = adapter._tool_find(chunk_id="gfk__0", pattern="   ")
    assert "error" in empty

    # Miss → match_count=0, no snippets, no error.
    no_hit = adapter._tool_find(chunk_id="gfk__0", pattern="Shaquille")
    assert no_hit["match_count"] == 0
    assert no_hit["snippets"] == []

    # Dispatcher wiring: `find` reachable via dispatch().
    plan = GSWPlan.model_validate(_fixture_plan())
    state = {e.id: BlankResult(blank_id=e.id, status="unknown") for e in plan.blank_entities()}
    dispatch, _ = adapter._make_plan_dispatcher(plan, state)
    disp_result, finish_ok = dispatch(
        "find",
        json.dumps({"chunk_id": "gfk__0", "pattern": "Fishscale"}),
    )
    assert disp_result["match_count"] == 1
    assert not finish_ok


def test_tool_schemas_are_closed_and_update_blank_requires_evidence():
    by_name = {tool["function"]["name"]: tool["function"] for tool in _TOOLS}

    assert by_name["search"]["parameters"]["additionalProperties"] is False
    assert by_name["read"]["parameters"]["additionalProperties"] is False
    assert by_name["find"]["parameters"]["additionalProperties"] is False
    assert set(by_name["find"]["parameters"]["required"]) == {"chunk_id", "pattern"}
    assert by_name["get_state"]["parameters"]["additionalProperties"] is False
    assert by_name["finish"]["parameters"]["additionalProperties"] is False
    assert by_name["update_blank"]["parameters"]["additionalProperties"] is False
    assert by_name["update_blank"]["parameters"]["required"] == [
        "blank_id",
        "value",
        "evidence_chunk_ids",
    ]
    assert by_name["update_blank"]["parameters"]["properties"]["value"]["anyOf"] == [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "array", "items": {"type": "string"}},
    ]


def test_build_system_prompt_uses_protocol_first_wording():
    prompt = build_system_prompt(_fixture_plan(), ["b_year", "t"], {"b_year": 0, "t": 1})

    assert "## Critical protocol" in prompt
    assert "Every assistant turn before completion MUST contain at least one" in prompt
    assert "Never reply with prose-only text or inline JSON" in prompt
    assert "The only valid ending is:" in prompt
    assert "## Tool-error recovery" in prompt
    assert "## Worked examples" in prompt
    assert "### Example 1 — Normal solve path" in prompt
    assert "### Example 2 — Sparse-evidence recovery" in prompt
    assert "turn 1 → search({" not in prompt
    assert 'finish({"answer"' not in prompt
    assert 'search({"query"' not in prompt


def test_render_plan_brief_slice_filters_to_assigned_blanks():
    """With ``slice_blank_ids=['t']`` (only the target), the rendered
    plan brief should mention entity `t` and the VP wiring `t` to
    Pink Floyd, but should drop `b_year` from the topological-order
    section (it belongs to an earlier level)."""
    from research_agent.adapters.ours._planner_react_prompt import render_plan_brief

    plan_dict = _fixture_plan()
    order = ["b_year", "t"]
    levels = {"b_year": 0, "t": 1}

    full = render_plan_brief(plan_dict, order, levels)
    sliced = render_plan_brief(plan_dict, order, levels, slice_blank_ids=["t"])

    # Sanity: full render contains both blanks in topo order
    assert "b_year" in full
    assert "`t`" in full

    # Sliced render: t is there, b_year is NOT listed as a topo step.
    assert "`t`" in sliced
    # b_year should not appear as its own Step heading.
    assert "Step 1" in sliced
    assert "Step 2" not in sliced  # only one step in the slice
    # Filled anchor "Pink Floyd" must survive (it's wired to t).
    assert "Pink Floyd" in sliced
    # Picasso was only wired to b_year (not in slice) → should be dropped.
    assert "Pablo Picasso" not in sliced


def test_render_plan_brief_slice_keeps_constraint_when_output_in_slice():
    """A constraint whose output blank is in the slice must appear
    in the slice rendering (so the researcher knows not to retrieve it)."""
    from research_agent.adapters.ours._planner_react_prompt import render_plan_brief

    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "b_b", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "concat",
                "args_blanks": ["b_a", "b_b"],
                "output_blank_id": "t",
            }
        ],
    }
    order = ["b_a", "b_b", "t"]
    levels = {"b_a": 0, "b_b": 0, "t": 1}

    sliced = render_plan_brief(plan_dict, order, levels, slice_blank_ids=["t"])
    # Target appears in topo order as a constraint-computed step.
    assert "`t`" in sliced
    assert "constraint" in sliced.lower() or "auto-computed" in sliced.lower()
    # b_a / b_b are upstream — they shouldn't show as their own Step headings.
    assert "Step 1" in sliced
    assert "Step 2" not in sliced


# ---------------------------------------------------------------------------
# Full run_question loop tests (with mocked plan-emit path)
# ---------------------------------------------------------------------------


def _patch_emit_plan(monkeypatch, plan_dict: dict[str, Any]):
    """Replace emit_plan to return our fixture plan directly."""
    from research_agent.adapters.ours import gsw_planner_react_v1 as mod
    from research_agent.adapters.ours._planner_emit import PlanEmitMeta

    def _fake_emit(question, llm_client, *, max_tokens=4096, enable_repair=True):
        return GSWPlan.model_validate(plan_dict), PlanEmitMeta(
            prompt_tokens=123, completion_tokens=456, raw_response="{}"
        )

    monkeypatch.setattr(mod, "emit_plan", _fake_emit)


def test_reasoner_loop_happy_path(monkeypatch):
    _patch_emit_plan(monkeypatch, _fixture_plan())

    # Script: turn 1 searches, turn 2 fills b_year, turn 3 fills t, turn 4 finishes.
    script = [
        _StubResponse(
            text=None,
            tool_calls=[_mk_tool_call("c1", "search", {"query": "Pablo Picasso died_in"})],
        ),
        _StubResponse(
            text=None,
            tool_calls=[
                _mk_tool_call(
                    "c2",
                    "update_blank",
                    {
                        "blank_id": "b_year",
                        "value": "1973",
                        "evidence_chunk_ids": ["ch1"],
                    },
                )
            ],
        ),
        _StubResponse(
            text=None,
            tool_calls=[
                _mk_tool_call(
                    "c3",
                    "update_blank",
                    {
                        "blank_id": "t",
                        "value": "Dark Side of the Moon",
                        "evidence_chunk_ids": ["ch2"],
                    },
                )
            ],
        ),
        _StubResponse(
            text=None,
            tool_calls=[_mk_tool_call("c4", "finish", {"answer": "Dark Side of the Moon"})],
        ),
    ]
    adapter = _mk_adapter(
        _ScriptedLLM(script),
        retriever=_StubRetriever(
            {"Pablo Picasso": [_StubChunk("ch1", "Pablo Picasso", "died in 1973")]}
        ),
    )

    traj = adapter.run_question("What album", question_id="q1")

    assert adapter.llm.calls[0]["tool_choice"] == "required"
    assert traj.final_answer == "Dark Side of the Moon"
    assert traj.extra["stopped_reason"] == "finished"
    assert traj.turns == 4
    assert traj.extra["finish_rejections"] == 0
    assert traj.extra["blank_updates"] == 2
    # Target blank shows up in executed_blanks as resolved.
    tgt = next(b for b in traj.extra["executed_blanks"] if b["blank_id"] == "t")
    assert tgt["status"] == "resolved"
    assert tgt["value"] == "Dark Side of the Moon"


def test_finish_rejection_forces_continuation(monkeypatch):
    _patch_emit_plan(monkeypatch, _fixture_plan())

    # Model tries to finish too early → rejection → then fills target → finishes.
    script = [
        _StubResponse(
            text=None,
            tool_calls=[_mk_tool_call("c1", "finish", {"answer": "Dark Side"})],
        ),
        _StubResponse(
            text=None,
            tool_calls=[
                _mk_tool_call(
                    "c2",
                    "update_blank",
                    {"blank_id": "t", "value": "Dark Side of the Moon", "evidence_chunk_ids": []},
                )
            ],
        ),
        _StubResponse(
            text=None,
            tool_calls=[_mk_tool_call("c3", "finish", {"answer": "Dark Side of the Moon"})],
        ),
    ]
    adapter = _mk_adapter(_ScriptedLLM(script))

    traj = adapter.run_question("q", question_id="q2")

    assert traj.extra["stopped_reason"] == "finished"
    assert traj.extra["finish_rejections"] == 1
    assert traj.final_answer == "Dark Side of the Moon"


def test_max_turns_with_resolved_target_still_emits_answer(monkeypatch):
    _patch_emit_plan(monkeypatch, _fixture_plan())

    # Loops searching forever; never calls finish. But target gets filled
    # at turn 2 — when max_turns hits we should still emit the target value.
    script = [
        _StubResponse(
            text=None,
            tool_calls=[_mk_tool_call(f"c{i}", "search", {"query": "x"})],
        )
        for i in range(8)
    ]
    # Insert an update_blank early for target.
    script[0] = _StubResponse(
        text=None,
        tool_calls=[
            _mk_tool_call(
                "u1",
                "update_blank",
                {
                    "blank_id": "t",
                    "value": "FallbackAnswer",
                    "evidence_chunk_ids": ["insufficient_evidence"],
                },
            )
        ],
    )
    adapter = _mk_adapter(_ScriptedLLM(script), max_turns=3)

    traj = adapter.run_question("q", question_id="q3")

    assert traj.extra["stopped_reason"] == "max_turns"
    assert traj.final_answer == "FallbackAnswer"


def test_finish_bypassed_when_model_gives_text_without_tool_call(monkeypatch):
    _patch_emit_plan(monkeypatch, _fixture_plan())

    # Model emits plain text on turn 1 (no tool calls) → stopped_reason=finish_bypassed.
    # Target is NOT resolved → fall through to the model's raw text.
    script = [_StubResponse(text="Just my guess: Dark Side", tool_calls=[])]
    adapter = _mk_adapter(_ScriptedLLM(script))

    traj = adapter.run_question("q", question_id="q4")

    assert traj.extra["stopped_reason"] == "finish_bypassed"
    assert traj.final_answer == "Just my guess: Dark Side"
    assert traj.turns == 1


def test_finish_bypassed_with_resolved_target_prefers_committed_value(monkeypatch):
    """If the model emits text without finish() BUT has already
    committed the target, we prefer the committed value over the text.
    Handles the JSON-as-text failure mode (model says `{"answer": "X"}`
    in content rather than calling the finish tool)."""
    _patch_emit_plan(monkeypatch, _fixture_plan())

    script = [
        _StubResponse(
            text=None,
            tool_calls=[
                _mk_tool_call(
                    "u1",
                    "update_blank",
                    {
                        "blank_id": "t",
                        "value": "Dark Side of the Moon",
                        "evidence_chunk_ids": ["ch2"],
                    },
                )
            ],
        ),
        # Model emits JSON-ish text instead of calling finish.
        _StubResponse(text='{"answer":"Dark Side of the Moon"}', tool_calls=[]),
    ]
    adapter = _mk_adapter(_ScriptedLLM(script))

    traj = adapter.run_question("q", question_id="q4b")

    assert traj.extra["stopped_reason"] == "finish_bypassed"
    assert traj.final_answer == "Dark Side of the Moon"  # from committed state, not text


def test_reasoner_llm_error_returns_partial_trajectory_not_fallback(monkeypatch):
    _patch_emit_plan(monkeypatch, _fixture_plan())

    class _BoomLLM:
        def chat(self, *a, **k):
            raise RuntimeError("boom")

    ctx = AdapterContext(
        system_id="ours_gsw_planner_react_v1",
        model_id="stub",
        max_turns=5,
        max_completion_tokens=2000,
        extra={"corpus": _StubCorpus({}), "retriever": _StubRetriever({})},
    )
    adapter = OursGSWPlannerReactV1Adapter(ctx)
    adapter.llm = _BoomLLM()

    traj = adapter.run_question("q", question_id="q5")

    assert traj.extra["stopped_reason"] == "llm_error"
    assert "boom" in traj.extra["llm_error"]
    # Reasoner failure is NOT a structural failure — no fallback.
    assert not traj.extra.get("fallback_flag", False)


# ---------------------------------------------------------------------------
# Plan-side failure triggers fallback
# ---------------------------------------------------------------------------


def test_planner_parse_failure_triggers_fallback(monkeypatch):
    """Stub emit_plan raising PlanEmitError → fallback fires with matching
    system_id and fallback_flag=True."""
    from research_agent.adapters.ours import gsw_planner_react_v1 as mod
    from research_agent.adapters.ours._planner_emit import PlanEmitError
    from research_agent.models.trace import Trajectory

    def _fake_emit(question, llm_client, **kwargs):
        raise PlanEmitError(kind="parse_failure", detail="malformed JSON")

    monkeypatch.setattr(mod, "emit_plan", _fake_emit)

    # Patch the fallback target to return a scripted trajectory.
    class _StubFallback:
        def run_question(self, question, *, question_id, articles):
            return Trajectory(
                system_id="ours_gsw_v1",
                model_id="stub",
                question_id=question_id,
                final_answer="fallback_answer",
            )

    adapter = _mk_adapter(_ScriptedLLM([]))
    adapter._fallback_adapter = _StubFallback()

    traj = adapter.run_question("q", question_id="q6")

    assert traj.extra["fallback_flag"] is True
    assert traj.extra["fallback_reason"].startswith("parse_failure:")
    assert traj.system_id == "ours_gsw_planner_react_v1"  # stamped back
    assert traj.final_answer == "fallback_answer"
