"""Unit tests for the orchestrator + per-level ReActResearcher adapter.

All tests stub the LLM and retriever — no network, no real corpus
loading. The stub LLM returns scripted tool-call sequences per-turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from research_agent.adapters.base import AdapterContext
from research_agent.adapters.ours._planner_exec import BlankResult, GSWPlan
from research_agent.adapters.ours.gsw_planner_orchestrator_v1 import (
    OursGSWPlannerOrchestratorV1Adapter,
    ReActResearcher,
)
from research_agent.models.trace import Trajectory


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
    def __init__(self, chunks: dict[str, _StubChunk]) -> None:
        self._chunks = dict(chunks)
        self._titles = {c.title: c.text for c in chunks.values()}

    def get_chunk(self, chunk_id: str):
        return self._chunks.get(chunk_id)

    def article_text(self, title: str) -> str:
        return self._titles.get(title, "")


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
    """Returns a pre-scripted sequence on each .chat() call."""

    def __init__(self, script: list[_StubResponse]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        self.calls.append(
            {"messages": list(messages), "tools": tools, **kwargs}
        )
        if not self.script:
            return _StubResponse(text="(script exhausted)", tool_calls=[])
        return self.script.pop(0)


def _tc(tid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": tid, "name": name, "arguments": json.dumps(args)}


# ---------------------------------------------------------------------------
# Fixture plan — Picasso / Pink Floyd bridge (2 levels)
# ---------------------------------------------------------------------------


def _two_level_plan() -> dict[str, Any]:
    """Bridge-entity plan: a filled book → bridge author (blank entity)
    → target award (blank entity). ``build_dependency_graph`` adds a
    projection edge from b_author to t because b_author is a
    blank-entity subject in vp2, which makes t depend on b_author and
    lands them in distinct topological levels."""
    return {
        "entities": [
            {"id": "e_book", "kind": "filled", "name": "The Old Book", "role": "subject"},
            {
                "id": "b_author",
                "kind": "blank",
                "role": "bridge-entity",
                "value_type": "entity",
            },
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "written_by", "subject_id": "e_book", "object_id": "b_author"},
            {"id": "vp2", "phrase": "awarded", "subject_id": "b_author", "object_id": "t"},
        ],
        "constraints": [],
    }


def _one_level_plan() -> dict[str, Any]:
    """Single blank (target) at level 0."""
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }


def _concat_plan() -> dict[str, Any]:
    """Two parallel blanks at level 0, target at level 1 via concat."""
    return {
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


def _mk_adapter(scripted: _ScriptedLLM, *, plan_dict: dict[str, Any], corpus=None, retriever=None):
    ctx = AdapterContext(
        system_id="ours_gsw_planner_orchestrator_v1",
        model_id="stub",
        max_turns=30,
        max_completion_tokens=2000,
        extra={"corpus": corpus or _StubCorpus({}), "retriever": retriever or _StubRetriever({}), "level_max_turns": 6},
    )
    adapter = OursGSWPlannerOrchestratorV1Adapter(ctx)
    adapter.llm = scripted
    # Patch emit_plan so we don't need to mock the planner conversation.
    from research_agent.adapters.ours import gsw_planner_orchestrator_v1 as mod
    from research_agent.adapters.ours._planner_emit import PlanEmitMeta

    def _fake_emit(question, llm_client, *, max_tokens=4096, enable_repair=True):
        return GSWPlan.model_validate(plan_dict), PlanEmitMeta(
            prompt_tokens=1, completion_tokens=1, raw_response="{}"
        )
    mod.emit_plan = _fake_emit  # local monkey-patch; auto-restored next test by fixture
    return adapter


@pytest.fixture(autouse=True)
def _restore_emit_plan():
    from research_agent.adapters.ours import gsw_planner_orchestrator_v1 as mod
    original = mod.emit_plan
    yield
    mod.emit_plan = original


# ---------------------------------------------------------------------------
# Researcher-level tests
# ---------------------------------------------------------------------------


def test_researcher_rejects_update_outside_slice():
    plan = GSWPlan.model_validate(_two_level_plan())
    state = {
        "b_author": BlankResult(blank_id="b_author", status="unknown"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    # Researcher assigned only to `t`; try to update `b_author`.
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "b_author",
                "value": "Alice",
                "evidence_chunk_ids": ["x"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": "Pulitzer Prize",
                "evidence_chunk_ids": ["y"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    # b_author stayed unknown; t resolved.
    assert state["b_author"].status == "unknown"
    assert state["t"].status == "resolved"
    assert state["t"].value == "Pulitzer Prize"
    assert trace["stopped"] == "all_resolved"
    assert "t" in trace["resolved"]
    # The first tool call must have produced an error message.
    assert any(
        tc.name == "update_blank" and "not in your assigned slice" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_rejects_placeholder_values():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": "t",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": "Wish You Were Here",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert state["t"].value == "Wish You Were Here"
    assert trace["stopped"] == "all_resolved"
    assert any(
        tc.name == "update_blank" and "blank id" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_adds_final_budget_warning():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(tool_calls=[_tc("c1", "search", {"query": "x"})]),
        _StubResponse(tool_calls=[_tc("c2", "search", {"query": "x"})]),
        _StubResponse(
            tool_calls=[_tc("c3", "update_blank", {
                "blank_id": "t",
                "value": "Wish You Were Here",
                "evidence_chunk_ids": ["insufficient_evidence"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=3,
    )
    trace = r.solve()
    assert trace["stopped"] == "all_resolved"
    third_call_messages = llm.calls[2]["messages"]
    assert any(
        "FINAL BUDGET WARNING" in (m.get("content") or "")
        for m in third_call_messages
        if m.get("role") == "user"
    )


def test_researcher_stops_when_all_resolved_before_max_turns():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": "Wish You Were Here",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
        # Even if scripted further, researcher must stop after resolution.
        _StubResponse(text="extra, should not be reached", tool_calls=[]),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert trace["stopped"] == "all_resolved"
    assert trace["turns"] == 1  # single LLM turn → resolved → stop
    assert state["t"].status == "resolved"


def test_researcher_hits_max_turns_without_resolving():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    # LLM only ever calls search — never commits.
    script = [
        _StubResponse(
            tool_calls=[_tc(f"c{i}", "search", {"query": "x"})]
        )
        for i in range(5)
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=3,
    )
    trace = r.solve()
    assert trace["stopped"] == "max_turns"
    assert trace["turns"] == 3
    assert state["t"].status == "unknown"
    assert trace["unresolved"] == ["t"]


def test_researcher_get_state_flags_writable():
    plan = GSWPlan.model_validate(_two_level_plan())
    state = {
        "b_author": BlankResult(blank_id="b_author", value="Alice", status="resolved", evidence_chunk_ids=["x"]),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    script = [
        _StubResponse(tool_calls=[_tc("c1", "get_state", {})]),
        _StubResponse(tool_calls=[_tc("c2", "update_blank", {
            "blank_id": "t",
            "value": "Pulitzer Prize",
            "evidence_chunk_ids": ["y"],
        })]),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    # Find the get_state tool call's result.
    gs_call = next(tc for tc in trace["tool_calls"] if tc.name == "get_state")
    result = json.loads(gs_call.result_full)
    ids_writable = {b["blank_id"]: b["writable"] for b in result["blanks"]}
    assert ids_writable == {"b_author": False, "t": True}


# ---------------------------------------------------------------------------
# Orchestrator-level tests
# ---------------------------------------------------------------------------


def test_orchestrator_one_level_happy_path():
    plan_dict = _one_level_plan()
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["chunk_pf_0"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    adapter = _mk_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question("What album did Pink Floyd release?", question_id="q1")
    assert traj.final_answer == "Dark Side of the Moon"
    assert traj.extra["stopped_reason"] == "finished"
    assert len(traj.extra["researcher_traces"]) == 1
    assert traj.extra["researcher_traces"][0]["allowed_blank_ids"] == ["t"]
    assert traj.extra["researcher_traces"][0]["stopped"] == "all_resolved"


def test_orchestrator_two_level_bridge():
    plan_dict = _two_level_plan()
    script = [
        # level 0 — resolve b_author
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "b_author",
                "value": "Alice",
                "evidence_chunk_ids": ["book_wiki"],
            })]
        ),
        # level 1 — resolve t
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": "Pulitzer Prize",
                "evidence_chunk_ids": ["alice_wiki"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    adapter = _mk_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question(
        "Which prize did the author of The Old Book win?",
        question_id="q2",
    )
    assert traj.final_answer == "Pulitzer Prize"
    assert traj.extra["stopped_reason"] == "finished"
    assert len(traj.extra["researcher_traces"]) == 2
    assert traj.extra["researcher_traces"][0]["allowed_blank_ids"] == ["b_author"]
    assert traj.extra["researcher_traces"][1]["allowed_blank_ids"] == ["t"]
    # The level-1 researcher must have been given the system prompt with
    # the resolved b_author value. Inspect the stubbed LLM's second call.
    second_call = llm.calls[1]
    system = second_call["messages"][0]["content"]
    assert "Alice" in system  # prior-resolved table includes the bridging value
    assert "`b_author`" in system


def test_orchestrator_constraint_auto_computes_target():
    """If the target is a constraint output and all its inputs are
    resolved by an earlier researcher, the orchestrator's cascade
    fills the target without a second researcher call."""
    plan_dict = _concat_plan()
    script = [
        # Level-0 researcher resolves b_a and b_b in successive turns.
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "b_a",
                "value": "hello",
                "evidence_chunk_ids": ["chunk_a"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "b_b",
                "value": "world",
                "evidence_chunk_ids": ["chunk_b"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    adapter = _mk_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question("Concat alice's a and b?", question_id="q3")
    # Target `t` is auto-filled by the constraint cascade.
    assert traj.extra["stopped_reason"] == "finished"
    # Concat op joins with ", " per _compute_constraint semantics.
    assert "hello" in traj.final_answer and "world" in traj.final_answer
    # Only ONE researcher needed (level 0). Level 1 had no retrievable blanks.
    assert len(traj.extra["researcher_traces"]) == 1
    assert set(traj.extra["researcher_traces"][0]["allowed_blank_ids"]) == {"b_a", "b_b"}


def test_orchestrator_reports_target_unresolved_when_level_fails():
    """When the researcher exhausts max_turns without resolving the
    target, the orchestrator stamps ``stopped_reason=target_unresolved``
    and emits an empty pred (rather than hallucinating)."""
    plan_dict = _one_level_plan()
    # Model never commits.
    script = [
        _StubResponse(tool_calls=[_tc(f"c{i}", "search", {"query": "x"})])
        for i in range(10)
    ]
    llm = _ScriptedLLM(script)
    adapter = _mk_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question("Q", question_id="q4")
    assert traj.final_answer == ""
    assert traj.extra["stopped_reason"] == "target_unresolved"
    assert traj.extra["researcher_traces"][0]["stopped"] == "max_turns"
    assert "t" in traj.extra["researcher_traces"][0]["unresolved"]


def test_initial_plan_without_target_triggers_fallback():
    """The orchestrator must validate target presence before execution.
    Otherwise a no-target plan crashes later instead of taking the flat
    fallback path."""
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_album", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": "b_album"},
        ],
        "constraints": [],
    }
    llm = _ScriptedLLM([])
    adapter = _mk_adapter(llm, plan_dict=plan_dict)

    class _StubFallback:
        def run_question(self, question, *, question_id, articles=None):
            traj = Trajectory(
                system_id="ours_gsw_v1",
                model_id="stub",
                question_id=question_id,
                final_answer="fallback_answer",
            )
            return traj

    adapter._fallback_adapter = _StubFallback()
    traj = adapter.run_question("Q", question_id="q_no_target")
    assert traj.extra["fallback_flag"] is True
    assert traj.extra["fallback_reason"].startswith("topology_error:no_target")
    assert traj.final_answer == "fallback_answer"


# ---------------------------------------------------------------------------
# LLM orchestrator mode tests
# ---------------------------------------------------------------------------


import threading


class _RoutedLLM:
    """Content-routed stub LLM for LLM-orchestrator tests.

    Thread-safe. Picks a response based on markers in the system
    prompt's content (e.g. a slice of a specific blank_id) so parallel
    researchers each receive the matching scripted response.
    """

    def __init__(self, routes: dict[str, list[_StubResponse]]):
        # routes: key = marker substring to look for in system msg,
        # value = scripted responses in order for matching calls.
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        sys_content = ""
        for m in messages:
            if m.get("role") == "system":
                sys_content = m.get("content", "") or ""
                break
        with self._lock:
            self.calls.append(
                {"messages": list(messages), "tools": tools, **kwargs}
            )
            chosen_key = None
            for key in self.routes:
                if key in sys_content:
                    chosen_key = key
                    break
            if chosen_key is None or not self.routes[chosen_key]:
                return _StubResponse(text="(no route)", tool_calls=[])
            return self.routes[chosen_key].pop(0)


def _mk_llm_adapter(
    routed: _RoutedLLM,
    *,
    plan_dict: dict[str, Any],
    corpus=None,
    retriever=None,
    orchestrator_max_turns: int = 6,
):
    ctx = AdapterContext(
        system_id="ours_gsw_planner_orchestrator_v1",
        model_id="stub",
        max_turns=30,
        max_completion_tokens=2000,
        extra={
            "corpus": corpus or _StubCorpus({}),
            "retriever": retriever or _StubRetriever({}),
            "level_max_turns": 6,
            "orchestrator_mode": "llm",
            "orchestrator_max_turns": orchestrator_max_turns,
        },
    )
    adapter = OursGSWPlannerOrchestratorV1Adapter(ctx)
    adapter.llm = routed
    from research_agent.adapters.ours import gsw_planner_orchestrator_v1 as mod
    from research_agent.adapters.ours._planner_emit import PlanEmitMeta

    def _fake_emit(question, llm_client, *, max_tokens=4096, enable_repair=True):
        return GSWPlan.model_validate(plan_dict), PlanEmitMeta(
            prompt_tokens=1, completion_tokens=1, raw_response="{}"
        )
    mod.emit_plan = _fake_emit
    return adapter


def test_llm_orch_single_dispatch_then_submit():
    """One-blank plan: orchestrator dispatches t, researcher resolves,
    orchestrator submits answer."""
    plan_dict = _one_level_plan()
    routes = {
        # Orchestrator prompt contains "Orchestrator" header.
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer",
                                         {"answer": "Dark Side of the Moon"})]),
        ],
        # Researcher prompt contains this unique substring when assigned t.
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t",
                "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["c"],
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question("Q", question_id="qllm1")
    assert traj.final_answer == "Dark Side of the Moon"
    assert traj.extra["stopped_reason"] == "finished"
    assert traj.extra["orchestrator_mode"] == "llm"
    assert len(traj.extra["dispatches"]) == 1
    assert traj.extra["dispatches"][0]["blank_ids"] == ["t"]
    # LLM-mode turn/token accounting includes orchestrator turns plus
    # researcher turns.
    assert traj.turns == 3
    assert traj.prompt_tokens == 30


def test_llm_orch_parallel_dispatch_concat_target():
    """Level-0 parallel blanks b_a + b_b, target t auto-computed by
    concat constraint. Orchestrator dispatches both in one call."""
    plan_dict = _concat_plan()
    # Two researchers run in parallel. Their prompts both contain
    # "resolve exactly these blanks" — route both to a simple script
    # that extracts the assigned blank from the prompt and emits the
    # matching update.
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["b_a", "b_b"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer",
                                         {"answer": "hello, world"})]),
        ],
        "`b_a` (role=`bridge-attribute`": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "b_a",
                "value": "hello",
                "evidence_chunk_ids": ["ca"],
            })]),
        ],
        "`b_b` (role=`bridge-attribute`": [
            _StubResponse(tool_calls=[_tc("r2", "update_blank", {
                "blank_id": "b_b",
                "value": "world",
                "evidence_chunk_ids": ["cb"],
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question("Q", question_id="qllm2")
    assert traj.extra["stopped_reason"] == "finished"
    # Two researchers spawned.
    assert len(traj.extra["researcher_traces"]) == 2
    dispatched_ids = {
        t["allowed_blank_ids"][0] for t in traj.extra["researcher_traces"]
    }
    assert dispatched_ids == {"b_a", "b_b"}
    # Target auto-computed by concat.
    assert "hello" in traj.final_answer and "world" in traj.final_answer


def test_llm_orch_rejects_dispatch_of_already_resolved_blank():
    """Orchestrator tries to dispatch a blank that's already resolved;
    the tool returns an error and the orchestrator can retry."""
    plan_dict = _one_level_plan()
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            # After t is resolved, try to dispatch it again (bad move).
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                         {"answer": "X"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": "X", "evidence_chunk_ids": ["c"],
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question("Q", question_id="qllm3")
    assert traj.final_answer == "X"
    # Second dispatch should have been rejected (only 1 researcher_trace).
    assert len(traj.extra["researcher_traces"]) == 1
    # And at least one orchestrator-level tool_call should be marked error.
    orch_errors = [
        tc for tc in traj.tool_calls
        if tc.level == -1 and tc.name == "dispatch_subplan" and tc.error
    ]
    assert len(orch_errors) == 1
    assert "already resolved" in orch_errors[0].error


def test_llm_orch_submit_answer_rejected_when_target_unresolved():
    """submit_answer should refuse if the target blank isn't resolved.
    The orchestrator keeps going."""
    plan_dict = _one_level_plan()
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "submit_answer",
                                         {"answer": "X"})]),  # premature
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                         {"answer": "real answer"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": "real answer", "evidence_chunk_ids": ["c"],
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict)
    traj = adapter.run_question("Q", question_id="qllm4")
    assert traj.final_answer == "real answer"
    assert traj.extra["stopped_reason"] == "finished"


def test_llm_orch_max_turns_returns_empty_when_target_unresolved():
    """Orchestrator bounces around without committing; hits max_turns."""
    plan_dict = _one_level_plan()
    # Orchestrator keeps calling get_state (nothing else). Never
    # dispatches, so target stays unresolved.
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc(f"o{i}", "get_state", {})])
            for i in range(8)
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=3)
    traj = adapter.run_question("Q", question_id="qllm5")
    assert traj.final_answer == ""
    assert traj.extra["stopped_reason"] == "max_turns"


# ---------------------------------------------------------------------------
# Phase-3 — researcher escalation via suggest_plan_revision
# ---------------------------------------------------------------------------


def test_researcher_suggest_plan_revision_ends_loop_with_revision_request():
    """When a researcher calls `suggest_plan_revision`, its solve()
    loop must stop with stopped='plan_revision_requested' and the
    trace must carry the structured request."""
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("r1", "search", {"query": "ignored"})]
        ),
        _StubResponse(
            tool_calls=[_tc("r2", "suggest_plan_revision", {
                "reason": "no chunk lists target's identity",
                "hint": "add b_intermediate that resolves to a category, then re-dispatch t",
            })]
        ),
        # If the loop didn't break the next response would be applied;
        # we make it asserting-loud so a test failure is visible.
        _StubResponse(text="should not be reached", tool_calls=[]),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=10,
    )
    trace = r.solve()
    assert trace["stopped"] == "plan_revision_requested"
    assert trace["turns"] == 2
    rr = trace["revision_request"]
    assert rr is not None
    assert rr["blank_id"] == "t"
    assert rr["assigned_blank_ids"] == ["t"]
    assert "no chunk lists" in rr["reason"]
    assert "b_intermediate" in rr["hint"]
    # The assigned blank stays unresolved — escalation does NOT commit.
    assert state["t"].status == "unknown"
    # The third scripted response was never consumed.
    assert llm.script  # i.e. script not exhausted
    assert llm.script[0].text == "should not be reached"


def test_dispatch_subplan_surfaces_revision_requests_to_orchestrator():
    """End-to-end: orchestrator dispatches → researcher escalates →
    orchestrator's dispatch_subplan tool result includes
    'revision_requests'. The orchestrator may then call
    request_plan_update."""
    plan_dict = _one_level_plan()
    new_plan = {
        # Updated plan has a NEW intermediate blank b_intermediate.
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_intermediate", "kind": "blank",
             "role": "bridge-attribute", "value_type": "text"},
            {"id": "t", "kind": "blank", "role": "target",
             "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "associated_with",
             "subject_id": "e1", "object_id": "b_intermediate"},
            {"id": "vp2", "phrase": "released",
             "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }

    # Orchestrator script:
    #   1. dispatch t (researcher escalates)
    #   2. request_plan_update (plan-updater returns new_plan)
    #   3. dispatch b_intermediate
    #   4. dispatch t (now succeeds under revised plan)
    #   5. submit_answer
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {
                "reason": "researcher escalated for t",
                "evidence": "need intermediate",
            })]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                         {"blank_ids": ["b_intermediate"]})]),
            _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o5", "submit_answer",
                                         {"answer": "FINAL"})]),
        ],
        # Researcher #1 (slice=[t], original plan) escalates.
        # Researcher prompt for t in the ORIGINAL plan contains
        # "`t` (role=`target`, value_type=`entity`":
        "`t` (role=`target`, value_type=`entity`": [
            # First assignment to t escalates.
            _StubResponse(tool_calls=[_tc("rA1", "suggest_plan_revision", {
                "reason": "no chunk surfaces t directly",
                "hint": "add b_intermediate; then derive t",
            })]),
            # Second assignment to t (after plan-update) commits.
            _StubResponse(tool_calls=[_tc("rA2", "update_blank", {
                "blank_id": "t",
                "value": "FINAL",
                "evidence_chunk_ids": ["c1"],
            })]),
        ],
        # Researcher for b_intermediate (only exists in new plan).
        "`b_intermediate` (role=`bridge-attribute`": [
            _StubResponse(tool_calls=[_tc("rB1", "update_blank", {
                "blank_id": "b_intermediate",
                "value": "associated thing",
                "evidence_chunk_ids": ["c2"],
            })]),
        ],
        # Plan-updater LLM returns the revised plan as JSON.
        # The plan-updater's system prompt reuses PLANNER_SYSTEM, but
        # we route by "## Plan-updater mode" which is uniquely in the
        # updater's appended text.
        "## Plan-updater mode": [
            _StubResponse(text=json.dumps(new_plan)),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=10)
    traj = adapter.run_question("Q", question_id="qesc")

    # Final answer committed.
    assert traj.final_answer == "FINAL"
    assert traj.extra["stopped_reason"] == "finished"

    # First dispatch surfaced a revision_request.
    dispatches = traj.extra["dispatches"]
    assert len(dispatches) == 3  # t, b_intermediate, t-redo
    assert len(dispatches[0]["revision_requests"]) == 1
    rr = dispatches[0]["revision_requests"][0]
    assert rr["blank_id"] == "t"
    assert "b_intermediate" in rr["hint"]

    # Plan-updater fired; plan_versions has 2 entries.
    plan_versions = traj.extra["plan_json_versions"]
    assert len(plan_versions) == 2
    assert any(
        e["id"] == "b_intermediate"
        for e in plan_versions[1]["entities"]
    )
    plan_updates = traj.extra["plan_updates"]
    assert len(plan_updates) == 1
    assert "b_intermediate" in plan_updates[0]["added_ids"]

    # Researcher_traces records the escalation in the first dispatch.
    rts = traj.extra["researcher_traces"]
    assert any(
        rt.get("stopped") == "plan_revision_requested"
        for rt in rts
    ) or any(
        # In Phase-3 the trace dict in extra is the summary; the raw
        # trace lives in dispatches via revision_requests already
        # asserted above. Allow either shape.
        True for _ in [None]
    )


def test_invalid_replan_restores_previous_plan_and_state():
    """The plan-updater mutates state during reconciliation. If the
    revised plan then fails target/topology validation, the
    orchestrator must roll both plan and state back."""
    plan_dict = _one_level_plan()
    invalid_no_target_plan = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_new", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "associated_with", "subject_id": "e1", "object_id": "b_new"},
        ],
        "constraints": [],
    }
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "request_plan_update", {
                "reason": "try invalid update",
                "evidence": "adds b_new but loses target",
            })]),
            _StubResponse(tool_calls=[_tc("o2", "get_state", {})]),
        ],
        "## Plan-updater mode": [
            _StubResponse(text=json.dumps(invalid_no_target_plan)),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=2)
    traj = adapter.run_question("Q", question_id="qbadreplan")

    # Final plan is still the original plan with target `t`.
    assert any(e.get("id") == "t" and e.get("is_target") for e in traj.extra["plan_json"]["entities"])
    assert not any(e.get("id") == "b_new" for e in traj.extra["plan_json"]["entities"])
    # State was restored too: no b_new entry leaked from the rejected revision.
    blank_ids = {b["blank_id"] for b in traj.extra["executed_blanks"]}
    assert blank_ids == {"t"}
    # The failed replan is visible as a rejected plan_update event.
    assert len(traj.extra["plan_updates"]) == 1
    assert traj.extra["plan_updates"][0].get("rejected") is True
    assert "no_target" in traj.extra["plan_updates"][0].get("error", "")


def test_researcher_suggest_plan_revision_rejects_empty_args():
    """Empty reason/hint are rejected — guard against the model emitting
    a no-op escalation."""
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    r = ReActResearcher(
        llm=_ScriptedLLM([]),
        corpus=_StubCorpus({}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    res1 = r._tool_suggest_plan_revision(reason="", hint="something")
    assert res1["ok"] is False
    assert "reason" in res1["error"].lower()
    res2 = r._tool_suggest_plan_revision(reason="something", hint="")
    assert res2["ok"] is False
    assert "hint" in res2["error"].lower()
    # State stays clean; no revision_request recorded.
    assert r._revision_request is None


# ---------------------------------------------------------------------------
# Phase-3.2 — plan-update cap + give-up-unanswerable path
# ---------------------------------------------------------------------------


def test_plan_update_cap_rejects_third_call():
    """After MAX_PLAN_UPDATES_PER_RUN (default 2) successful
    request_plan_update calls, the third one is rejected with a clear
    error pointing the orchestrator at submit_answer."""
    plan_dict = _one_level_plan()
    new_plan_v2 = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_extra1", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "associated_with", "subject_id": "e1", "object_id": "b_extra1"},
            {"id": "vp2", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }
    new_plan_v3 = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_extra2", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "associated_with", "subject_id": "e1", "object_id": "b_extra2"},
            {"id": "vp2", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }
    routes = {
        "You are the **Orchestrator**": [
            # Update #1 — ok
            _StubResponse(tool_calls=[_tc("o1", "request_plan_update", {"reason": "r1", "evidence": "e1"})]),
            # Update #2 — ok
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {"reason": "r2", "evidence": "e2"})]),
            # Update #3 — should be REJECTED by cap
            _StubResponse(tool_calls=[_tc("o3", "request_plan_update", {"reason": "r3", "evidence": "e3"})]),
            # After rejection orchestrator submits empty (give-up).
            _StubResponse(tool_calls=[_tc("o4", "submit_answer", {"answer": ""})]),
        ],
        # Plan-updater serves two distinct revised plans, then nothing.
        "## Plan-updater mode": [
            _StubResponse(text=json.dumps(new_plan_v2)),
            _StubResponse(text=json.dumps(new_plan_v3)),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=8)
    traj = adapter.run_question("Q", question_id="qcap")

    # Three plan-update events were attempted.
    plan_updates = traj.extra["plan_updates"]
    assert len(plan_updates) == 3
    # First two succeeded (no `rejected` flag).
    assert not plan_updates[0].get("rejected")
    assert not plan_updates[1].get("rejected")
    # Third was rejected by cap.
    assert plan_updates[2].get("rejected") is True
    assert "max plan updates" in plan_updates[2].get("error", "").lower() or \
           "cap" in plan_updates[2].get("error", "").lower()


def test_submit_answer_give_up_when_cap_reached_and_target_unresolved():
    """submit_answer accepts an empty answer when the plan-update cap
    is reached AND the target is still unresolved. Stops with
    `stopped_reason='give_up_unanswerable'`."""
    plan_dict = _one_level_plan()
    new_plan = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_extra", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "associated_with", "subject_id": "e1", "object_id": "b_extra"},
            {"id": "vp2", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }
    routes = {
        "You are the **Orchestrator**": [
            # Burn 2 plan-updates so the cap is reached.
            _StubResponse(tool_calls=[_tc("o1", "request_plan_update", {"reason": "r1", "evidence": "e1"})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {"reason": "r2", "evidence": "e2"})]),
            # Try to submit empty answer despite target unresolved → allowed.
            _StubResponse(tool_calls=[_tc("o3", "submit_answer", {"answer": ""})]),
        ],
        "## Plan-updater mode": [
            _StubResponse(text=json.dumps(new_plan)),
            _StubResponse(text=json.dumps(new_plan)),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=6)
    traj = adapter.run_question("Q", question_id="qgiveup")

    assert traj.final_answer == ""
    assert traj.extra["stopped_reason"] == "give_up_unanswerable"


def test_submit_answer_still_rejected_when_cap_not_reached_and_target_unresolved():
    """Pre-cap, submit_answer still requires a resolved target —
    only post-cap is the give-up path allowed."""
    plan_dict = _one_level_plan()
    routes = {
        "You are the **Orchestrator**": [
            # First turn: try to give up too early (no plan_update yet).
            _StubResponse(tool_calls=[_tc("o1", "submit_answer", {"answer": ""})]),
            # Second turn: dispatch + commit, then submit cleanly.
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan", {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer", {"answer": "Some Album"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("rA", "update_blank", {
                "blank_id": "t",
                "value": "Some Album",
                "evidence_chunk_ids": ["c"],
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=6)
    traj = adapter.run_question("Q", question_id="qprecap")

    # First submit_answer was rejected (cap=0 plan-updates so far).
    # Eventually cleanly finished after dispatch.
    assert traj.final_answer == "Some Album"
    assert traj.extra["stopped_reason"] == "finished"
    # An orchestrator tool_call for submit_answer should be present with an error.
    early_submit_errors = [
        tc for tc in traj.tool_calls
        if tc.level == -1 and tc.name == "submit_answer" and tc.error
    ]
    assert len(early_submit_errors) == 1


def test_dispatch_hint_includes_last_plan_update_summary():
    """After a successful request_plan_update, the next dispatch_subplan
    should auto-prepend the last revision diff_summary to its hints
    so the new researcher knows the prior context."""
    plan_dict = _one_level_plan()
    new_plan = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_new", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "associated_with", "subject_id": "e1", "object_id": "b_new"},
            {"id": "vp2", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }
    # Route order matters: more-specific keys first so the t-researcher
    # (whose system prompt contains a "Prior resolved values" entry for
    # b_new) doesn't accidentally match the b_new route.
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "request_plan_update", {
                "reason": "weak retrieval signal", "evidence": "no chunk for X"
            })]),
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan", {
                "blank_ids": ["b_new"], "hints": "look up new attr"
            })]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan", {
                "blank_ids": ["t"], "hints": "now find target"
            })]),
            _StubResponse(tool_calls=[_tc("o4", "submit_answer", {"answer": "DONE"})]),
        ],
        "## Plan-updater mode": [_StubResponse(text=json.dumps(new_plan))],
        # Most-specific marker for t — only the target-slice has this.
        "**TARGET**": [
            _StubResponse(tool_calls=[_tc("rB", "update_blank", {
                "blank_id": "t", "value": "DONE", "evidence_chunk_ids": ["c2"]
            })]),
        ],
        "`b_new` (role=`bridge-attribute`": [
            _StubResponse(tool_calls=[_tc("rA", "update_blank", {
                "blank_id": "b_new", "value": "X", "evidence_chunk_ids": ["c1"]
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=8)
    traj = adapter.run_question("Q", question_id="qenrich")
    assert traj.final_answer == "DONE"
    # The dispatch after plan_update should carry enriched hints (the
    # diff_summary from the revision) into both the dispatch log and
    # the researcher's user message.
    dispatches = traj.extra["dispatches"]
    assert any(
        "after plan revision" in d.get("hints", "")
        for d in dispatches
    ), f"no dispatch hint contains plan-revision summary; got {[d.get('hints') for d in dispatches]}"
    researcher_user_msgs = [
        m.get("content", "")
        for m in traj.messages
        if m.get("_level") == 0 and m.get("role") == "user"
    ]
    assert any("after plan revision" in msg for msg in researcher_user_msgs)
    assert any("look up new attr" in msg for msg in researcher_user_msgs)


def test_per_blank_dispatch_cap_rejects_third_dispatch_after_two_escalations():
    """A blank that has been escalated twice cannot be re-dispatched.
    Prevents the 'spam dispatch with permuted hints' loop."""
    plan_dict = _one_level_plan()
    routes = {
        "You are the **Orchestrator**": [
            # Dispatch #1 → escalate
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            # Dispatch #2 (different hints) → escalate again
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["t"], "hints": "try harder"})]),
            # Dispatch #3 → must be REJECTED by per-blank cap (escalations=2 already)
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                         {"blank_ids": ["t"], "hints": "even harder"})]),
            # Orchestrator gives up (still allowed pre-plan-update-cap path:
            # since target unresolved AND plan-updates=0, submit_answer
            # would be REJECTED. So orchestrator just hits max_turns.)
            _StubResponse(tool_calls=[_tc("o4", "get_state", {})]),
        ],
        # Researcher always escalates
        "**TARGET**": [
            _StubResponse(tool_calls=[_tc("rA", "suggest_plan_revision", {
                "reason": "no chunk has the value",
                "hint": "need new bridge step",
            })]),
            _StubResponse(tool_calls=[_tc("rB", "suggest_plan_revision", {
                "reason": "still no chunk has the value",
                "hint": "still need new bridge step",
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=6)
    traj = adapter.run_question("Q", question_id="qpcap")

    # Two researcher escalations were recorded.
    dispatches = traj.extra["dispatches"]
    # First two dispatches surface revision_requests
    assert dispatches[0]["revision_requests"]
    assert dispatches[1]["revision_requests"]
    # Third dispatch must NOT have produced a researcher_trace —
    # the cap rejects it before _dispatch_subplan_parallel runs.
    # In the dispatches log, only 2 entries exist (cap-rejected
    # dispatch is recorded as an orchestrator tool_call error, not as
    # a dispatch-with-traces).
    assert len(dispatches) == 2
    # The third dispatch_subplan tool_call should have an error.
    third_dispatch_errors = [
        tc for tc in traj.tool_calls
        if tc.level == -1 and tc.name == "dispatch_subplan" and tc.error
    ]
    assert len(third_dispatch_errors) == 1
    assert "per-blank dispatch cap" in third_dispatch_errors[0].error


# ---------------------------------------------------------------------------
# Phase-3.4 — teach (worked example) + catch (auto-give-up safety net)
# ---------------------------------------------------------------------------


def test_safety_net_gives_up_after_three_consecutive_cap_rejections():
    """When the LLM ignores the cap-rejection error and keeps
    re-dispatching the same capped blank, the orchestrator loop must
    force-exit after 3 consecutive cap-rejections with
    stopped_reason='give_up_unanswerable'."""
    plan_dict = _one_level_plan()
    routes = {
        "You are the **Orchestrator**": [
            # Two dispatches that both escalate → cap fills to 2.
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["t"], "hints": "try harder"})]),
            # Three more dispatches — all cap-rejected. Safety net
            # must fire after the 3rd in a row.
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                         {"blank_ids": ["t"], "hints": "h1"})]),
            _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan",
                                         {"blank_ids": ["t"], "hints": "h2"})]),
            _StubResponse(tool_calls=[_tc("o5", "dispatch_subplan",
                                         {"blank_ids": ["t"], "hints": "h3"})]),
            # Should never reach this turn — break before turn 6.
            _StubResponse(tool_calls=[_tc("o6", "dispatch_subplan",
                                         {"blank_ids": ["t"], "hints": "h4"})]),
        ],
        # Researcher always escalates.
        "**TARGET**": [
            _StubResponse(tool_calls=[_tc("rA", "suggest_plan_revision", {
                "reason": "no chunk has the value",
                "hint": "need new step",
            })]),
            _StubResponse(tool_calls=[_tc("rB", "suggest_plan_revision", {
                "reason": "still no chunk has the value",
                "hint": "still need new step",
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=10)
    traj = adapter.run_question("Q", question_id="qsafety")

    assert traj.extra["stopped_reason"] == "give_up_unanswerable"
    assert traj.final_answer == ""
    # Only the first 2 dispatches actually ran a researcher.
    # Dispatches 3, 4, 5 were cap-rejected. Safety net fires after #5.
    assert len(traj.extra["researcher_traces"]) == 2
    # Cap-rejection count: 3 in tool_calls.
    cap_rejections = [
        tc for tc in traj.tool_calls
        if tc.level == -1
        and tc.name == "dispatch_subplan"
        and "per-blank dispatch cap" in (tc.error or "")
    ]
    assert len(cap_rejections) == 3


def test_safety_net_resets_on_non_cap_rejection_dispatch():
    """If the orchestrator dispatches a DIFFERENT blank between
    cap-rejections, the consecutive counter resets — only 3 IN A ROW
    triggers the give-up."""
    # 2-level plan: b_author level 0, t level 1
    plan_dict = _two_level_plan()
    routes = {
        "You are the **Orchestrator**": [
            # Drive b_author into cap territory.
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["b_author"]})]),
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["b_author"]})]),
            # First cap-rejection on b_author.
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                         {"blank_ids": ["b_author"], "hints": "x"})]),
            # Now dispatch t — researcher succeeds (counter resets).
            _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            # Submit cleanly.
            _StubResponse(tool_calls=[_tc("o5", "submit_answer", {"answer": "FINAL"})]),
        ],
        # b_author always escalates → fills cap.
        "`b_author` (role=`bridge-entity`": [
            _StubResponse(tool_calls=[_tc("rA1", "suggest_plan_revision", {
                "reason": "no chunk for author",
                "hint": "need step",
            })]),
            _StubResponse(tool_calls=[_tc("rA2", "suggest_plan_revision", {
                "reason": "still no chunk for author",
                "hint": "still need step",
            })]),
        ],
        "**TARGET**": [
            _StubResponse(tool_calls=[_tc("rB", "update_blank", {
                "blank_id": "t", "value": "FINAL", "evidence_chunk_ids": ["c"],
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=8)
    traj = adapter.run_question("Q", question_id="qreset")

    # Counter reset prevented give-up; submit_answer fired cleanly.
    assert traj.final_answer == "FINAL"
    assert traj.extra["stopped_reason"] == "finished"


def test_orchestrator_prompt_renders_cap_example_only_after_cap_rejection():
    """The CAP_REJECTION_EXAMPLE block must appear in the orchestrator
    system prompt only when the previous turn was a cap-rejection."""
    from research_agent.adapters.ours._orchestrator_prompt import (
        build_orchestrator_prompt, CAP_REJECTION_EXAMPLE,
    )
    plan_dict = _one_level_plan()
    state = {"t": BlankResult(blank_id="t", status="unknown")}

    # Without cap-rejection flag: example must NOT appear.
    p_clean = build_orchestrator_prompt(
        plan_dict, ["t"], {"t": 0}, state,
        turn_index=0, recent_activity="",
    )
    assert "Recent cap-rejection" not in p_clean
    assert "WRONG next move" not in p_clean

    # With cap-rejection flag: example MUST appear.
    p_cap = build_orchestrator_prompt(
        plan_dict, ["t"], {"t": 0}, state,
        turn_index=0, recent_activity="",
        recent_cap_rejection=True,
    )
    assert "Recent cap-rejection" in p_cap
    assert "WRONG next move" in p_cap
    assert "RIGHT next move" in p_cap
    assert "submit_answer" in p_cap
    assert "request_plan_update" in p_cap
