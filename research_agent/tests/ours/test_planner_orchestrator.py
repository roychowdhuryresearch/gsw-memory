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
