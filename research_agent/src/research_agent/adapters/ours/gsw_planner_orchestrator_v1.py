"""Orchestrator + per-level researcher adapter.

Pipeline per question:

1. **Plan** — one LLM call emits a validated ``GSWPlan`` (same
   ``emit_plan`` path as the flat planner_react adapter).
2. **Topological sort** — ``build_dependency_graph`` +
   ``_compute_levels`` give ``levels: dict[blank_id, int]``.
3. **Orchestrate** — deterministic Python loop over ``levels`` in
   ascending order:

     for lvl in sorted(levels_unique):
         slice = blanks_at(lvl) - constraint_output_blanks
         if not slice: auto-compute, continue
         researcher = ReActResearcher(
             plan, state (read view), allowed_blank_ids=slice, ...)
         researcher.solve()
         cascade_auto_compute(plan, state)

   After the loop, the target blank is checked; its value is the
   final answer or the run stamps ``stopped_reason=target_unresolved``.

In deterministic mode, the orchestrator itself makes no LLM calls.
In LLM-orchestrator mode, the orchestrator also makes control-flow
LLM calls and may invoke the plan-updater/replanner.

Researcher contract:
- ``search``, ``read``, ``find``, ``update_blank``, ``get_state`` tools.
- ``suggest_plan_revision`` escalates to the orchestrator/replanner when
  the plan is missing a retrieval step.
- ``update_blank`` rejects any id outside ``allowed_blank_ids``.
- No ``finish`` tool — researcher ends its own loop when all assigned
  blanks are resolved or max_turns hits.

Falls back to the flat ``ours_gsw_v1`` adapter on planner-side failure
(parse error, topology error). Researcher LLM errors do not trigger
fallback; they return a partial trajectory stamped
``stopped_reason="llm_error"``.
"""

from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Optional

from research_agent.adapters.base import Adapter, AdapterContext, register_adapter
from research_agent.adapters.ours._planner_emit import (
    PlanEmitError,
    _PlannerFallbackMixin,
    emit_plan,
)
from research_agent.adapters.ours._planner_exec import (
    BlankResult,
    ExecutionError,
    GSWPlan,
    _coerce_value,
    build_dependency_graph,
    topological_sort_blanks,
)
from research_agent.adapters.ours._orchestrator_prompt import (
    build_orchestrator_prompt,
)
from research_agent.adapters.ours._plan_updater import (
    PlanReconcileDiff,
    update_plan,
)
from research_agent.adapters.ours._researcher_prompt import build_researcher_prompt
from research_agent.adapters.ours.gsw_planner_react_v1 import (
    _cascade_auto_compute,
    _compute_levels,
    _stringify,
)
from research_agent.models.llm_client import LLMClient
from research_agent.models.trace import ToolCall, Trajectory
from research_agent.retrieval.corpus import load_frames_corpus
from research_agent.retrieval.dense import build_retriever


# ---------------------------------------------------------------------------
# Tool definitions for the researcher (5 tools — no `finish`)
# ---------------------------------------------------------------------------


_ORCHESTRATOR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dispatch_subplan",
            "description": (
                "Dispatch one researcher per blank_id in the list, running "
                "them in parallel (capped at 8 concurrent). Each researcher "
                "retrieves evidence and calls `update_blank` on its assigned "
                "blank. Returns the resolved state for those blanks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blank_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Blank ids to resolve in parallel. 1–8 ids. "
                            "Ids must exist in the current plan and must not "
                            "already be status=resolved, and must not be "
                            "constraint-output blanks."
                        ),
                    },
                    "hints": {
                        "type": "string",
                        "description": (
                            "Optional short note passed to the researchers "
                            "(retrieval query hints, bridging anchors, etc.)."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["blank_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_plan_update",
            "description": (
                "Invoke the plan-updater LLM to revise the GSWPlan given "
                "retrieved evidence. Use only when the plan itself is wrong "
                "or incomplete (not for weak-retrieval cases)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short justification for the update.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "Short string capturing retrieved facts that "
                            "contradict or extend the current plan."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["reason", "evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Return the current plan + blank-fill state. Rarely "
                "needed; the state is already in the system prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Terminate with the final answer. The target blank must "
                "already be status=resolved; pass its committed value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "additionalProperties": False,
                "required": ["answer"],
            },
        },
    },
]


_RESEARCHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Retrieve chunks from the FRAMES article corpus most relevant "
                "to a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "additionalProperties": False,
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Fetch the full article text for a given chunk_id.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "additionalProperties": False,
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": (
                "Search inside a specific chunk's article for a literal "
                "substring (case-insensitive). Returns up to 3 snippets "
                "of ±200 chars around each match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "additionalProperties": False,
                "required": ["chunk_id", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_blank",
            "description": (
                "Record a blank's resolved value into shared state. You may "
                "only update blanks in your assigned slice — other ids will "
                "be rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blank_id": {"type": "string"},
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "evidence_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
                "required": ["blank_id", "value", "evidence_chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Return the current blank-fill state across the whole "
                "plan. Each entry is flagged `writable` — only writable "
                "ids are in your assigned slice."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_plan_revision",
            "description": (
                "Escalate to the orchestrator: the plan is missing a step. "
                "Use ONLY when retrieval reveals that resolving your "
                "assigned blank requires a NEW intermediate blank that "
                "does not exist in the plan yet (not when evidence for "
                "your current blank is just thin — for that, commit your "
                "best-guess value with `evidence_chunk_ids="
                "['insufficient_evidence']` instead). Calling this tool "
                "ENDS your run; your assigned blank stays unresolved and "
                "the orchestrator decides whether to revise the plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short note on what was missing. Example: "
                            "'no chunk in the corpus lists postcodes for "
                            "Liverpool Maternity Hospital'."
                        ),
                    },
                    "hint": {
                        "type": "string",
                        "description": (
                            "Concrete suggestion for the new step, e.g. a "
                            "new blank id and what predicate would link it. "
                            "Example: 'add b_address (entity for the "
                            "hospital address); postcode could be derived "
                            "from address'."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["reason", "hint"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# ReActResearcher — one per topological level
# ---------------------------------------------------------------------------


class ReActResearcher:
    """A single-level researcher agent.

    Runs a ReAct loop with 5 tools until every blank in
    ``allowed_blank_ids`` has ``status=resolved`` in ``state``, or
    until ``max_turns`` is exhausted. Does NOT return a final answer —
    the orchestrator reads the resolved target out of ``state``.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        corpus: Any,
        retriever: Any,
        plan: GSWPlan,
        state: dict[str, BlankResult],
        allowed_blank_ids: list[str],
        system_prompt: str,
        question: str,
        max_turns: int = 15,
        max_completion_tokens: int = 50000,
    ) -> None:
        self.llm = llm
        self.corpus = corpus
        self.retriever = retriever
        self.plan = plan
        self.state = state
        self.allowed: set[str] = set(allowed_blank_ids)
        self.allowed_order: list[str] = list(allowed_blank_ids)
        self.system_prompt = system_prompt
        self.question = question
        self.max_turns = max_turns
        self.max_completion_tokens = max_completion_tokens
        # Set by `_tool_suggest_plan_revision` when the researcher
        # escalates. Surfaced in ``solve()``'s return value.
        self._revision_request: Optional[dict[str, Any]] = None

    # --- tool implementations (bound to this researcher's state) ----

    def _tool_search(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
        hits = self.retriever.search(query, top_k=top_k)
        return {
            "results": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "title": h.chunk.title,
                    "score": round(h.score, 3),
                    "text": h.chunk.text[:800],
                }
                for h in hits
            ]
        }

    def _tool_read(self, *, chunk_id: str) -> dict[str, Any]:
        chunk = self.corpus.get_chunk(chunk_id)
        if chunk is None:
            return {"error": f"chunk_id {chunk_id!r} not found"}
        full = self.corpus.article_text(chunk.title)
        return {
            "chunk_id": chunk_id,
            "title": chunk.title,
            "article_text": full[:12000],
        }

    def _tool_find(self, *, chunk_id: str, pattern: str) -> dict[str, Any]:
        chunk = self.corpus.get_chunk(chunk_id)
        if chunk is None:
            return {"error": f"chunk_id {chunk_id!r} not found"}
        if not pattern or not pattern.strip():
            return {"error": "pattern must be a non-empty string"}
        full = self.corpus.article_text(chunk.title)
        hay = full.lower()
        needle = pattern.lower()
        snippets: list[dict[str, Any]] = []
        start = 0
        while len(snippets) < 3:
            idx = hay.find(needle, start)
            if idx < 0:
                break
            left = max(0, idx - 200)
            right = min(len(full), idx + len(pattern) + 200)
            snippets.append({"offset": idx, "snippet": full[left:right]})
            start = idx + len(pattern)
        return {
            "chunk_id": chunk_id,
            "title": chunk.title,
            "pattern": pattern,
            "match_count": len(snippets),
            "snippets": snippets,
        }

    def _tool_update_blank(
        self,
        *,
        blank_id: str,
        value: Any,
        evidence_chunk_ids: list[str],
    ) -> dict[str, Any]:
        # Slice guard — reject writes outside the assigned slice.
        if blank_id not in self.allowed:
            return {
                "ok": False,
                "error": (
                    f"blank_id {blank_id!r} is not in your assigned slice. "
                    f"You may only update: {sorted(self.allowed)}. "
                    f"Upstream blanks are read-only."
                ),
                "assigned_slice": sorted(self.allowed),
            }
        if blank_id not in self.state:
            return {
                "ok": False,
                "error": f"unknown blank_id {blank_id!r}",
                "known_blank_ids": sorted(self.state.keys()),
            }
        if (
            isinstance(value, str)
            and value.strip().lower() == "insufficient_evidence"
        ):
            return {
                "ok": False,
                "error": (
                    "Passing value='insufficient_evidence' is ALWAYS "
                    "wrong. The string 'insufficient_evidence' is a "
                    "metadata marker for `evidence_chunk_ids`; it is "
                    "never a valid `value`. Retry with your actual "
                    "best-guess as `value`, and put "
                    "'insufficient_evidence' inside `evidence_chunk_ids`."
                ),
            }
        invalid_value_error = self._invalid_update_value_error(value)
        if invalid_value_error:
            return {"ok": False, "error": invalid_value_error}
        ent = self.plan.entity_by_id(blank_id)
        coerced = _coerce_value(value, ent.value_type)
        ev = list(evidence_chunk_ids)
        self.state[blank_id] = BlankResult(
            blank_id=blank_id,
            value=coerced,
            status="resolved" if coerced is not None else "unknown",
            evidence_chunk_ids=ev,
            llm_calls=0,
            wall_time_s=0.0,
        )
        auto_filled = _cascade_auto_compute(self.plan, self.state)
        out: dict[str, Any] = {"ok": True, "state": self._state_view()}
        if auto_filled:
            out["auto_computed"] = auto_filled
        return out

    def _tool_get_state(self) -> dict[str, Any]:
        return self._state_view()

    def _state_view(self) -> dict[str, Any]:
        """State view annotated with ``writable`` per blank."""
        blanks: list[dict[str, Any]] = []
        for b_id, br in self.state.items():
            blanks.append(
                {
                    "blank_id": b_id,
                    "status": br.status,
                    "value": br.value,
                    "evidence_chunk_ids": list(br.evidence_chunk_ids or []),
                    "writable": b_id in self.allowed,
                }
            )
        target_id = self.plan.target().id
        return {
            "blanks": blanks,
            "assigned_slice": sorted(self.allowed),
            "target_blank_id": target_id,
        }

    # --- dispatch + loop --------------------------------------------

    def _invalid_update_value_error(self, value: Any) -> str:
        if value is None:
            return (
                "`value` cannot be null. Retry with an actual best-guess "
                "value and use evidence_chunk_ids=['insufficient_evidence'] "
                "if evidence is weak."
            )
        if isinstance(value, list) and not value:
            return (
                "`value` cannot be an empty list. Retry with a best-effort "
                "list, or use evidence_chunk_ids=['insufficient_evidence'] "
                "if evidence is weak."
            )
        if not isinstance(value, str):
            return ""

        s = value.strip()
        low = s.lower()
        if not s:
            return (
                "`value` cannot be empty. Retry with an actual best-guess "
                "value and use evidence_chunk_ids=['insufficient_evidence'] "
                "if evidence is weak."
            )
        if low in {"unknown", "none", "null", "n/a", "na"}:
            return (
                f"`value={value!r}` is a placeholder, not an answer. Retry "
                "with an actual best-guess value and put "
                "'insufficient_evidence' in evidence_chunk_ids if needed."
            )
        if s in self.state:
            return (
                f"`value={value!r}` is a blank id, not an answer. Retry with "
                "the resolved value for that blank."
            )
        if s[:1] in {"{", "["} and s[-1:] in {"}", "]"}:
            try:
                json.loads(s)
            except json.JSONDecodeError:
                return ""
            return (
                "`value` looks like serialized JSON/tool arguments, not an "
                "answer. Retry with the extracted answer value only."
            )
        return ""

    def _tool_suggest_plan_revision(
        self,
        *,
        reason: str,
        hint: str,
    ) -> dict[str, Any]:
        """Researcher escalation: end the run, ask orchestrator to rethink the plan."""
        if not isinstance(reason, str) or not reason.strip():
            return {
                "ok": False,
                "error": (
                    "`reason` must be a non-empty string describing what "
                    "was missing in retrieval."
                ),
            }
        if not isinstance(hint, str) or not hint.strip():
            return {
                "ok": False,
                "error": (
                    "`hint` must be a non-empty string suggesting a new "
                    "intermediate step (e.g. 'add b_address blank')."
                ),
            }
        # Record the request — the loop checks this each turn and exits.
        # Attach the assigned slice so the orchestrator knows which blank
        # triggered the escalation.
        self._revision_request = {
            "blank_id": self.allowed_order[0] if self.allowed_order else None,
            "assigned_blank_ids": list(self.allowed_order),
            "reason": reason.strip(),
            "hint": hint.strip(),
        }
        return {"ok": True, "escalated": True}

    def _dispatch(self, name: str, args_json: str) -> dict[str, Any]:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return {"error": f"bad tool args: {exc}"}
        try:
            if name == "search":
                return self._tool_search(**args)
            if name == "read":
                return self._tool_read(**args)
            if name == "find":
                return self._tool_find(**args)
            if name == "update_blank":
                return self._tool_update_blank(**args)
            if name == "get_state":
                return self._tool_get_state()
            if name == "suggest_plan_revision":
                return self._tool_suggest_plan_revision(**args)
            return {"error": f"unknown tool {name!r}"}
        except TypeError as exc:
            return {"error": f"bad tool call: {exc}"}

    def _unresolved_assigned(self) -> list[str]:
        return [
            b for b in self.allowed_order
            if self.state.get(b) is None
            or self.state[b].status != "resolved"
        ]

    def _all_assigned_resolved(self) -> bool:
        return not self._unresolved_assigned()

    def _budget_warning(self, unresolved: list[str], remaining: int) -> str:
        plural = "turns" if remaining != 1 else "turn"
        return (
            "FINAL BUDGET WARNING: "
            f"{remaining} tool-call {plural} remain for this researcher. "
            f"Unresolved assigned blanks: {unresolved}. Your next tool call "
            "must be `update_blank` for one unresolved assigned blank. Do "
            "not call `search`, `read`, `find`, or `get_state` now. If "
            "evidence is weak, commit a real best-guess value and set "
            "evidence_chunk_ids=['insufficient_evidence']."
        )

    def solve(self) -> dict[str, Any]:
        """Run the ReAct loop. Returns a trace dict for the orchestrator."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.question},
        ]
        tool_calls: list[ToolCall] = []
        reasoning_chunks: list[str] = []
        prompt_tokens = completion_tokens = 0
        stopped = "max_turns"
        turn = 0

        for turn in range(1, self.max_turns + 1):
            unresolved_before_turn = self._unresolved_assigned()
            if not unresolved_before_turn:
                stopped = "all_resolved"
                break
            remaining = self.max_turns - turn + 1
            if remaining <= len(unresolved_before_turn):
                messages.append(
                    {
                        "role": "user",
                        "content": self._budget_warning(
                            unresolved_before_turn, remaining
                        ),
                    }
                )
            try:
                resp = self.llm.chat(
                    messages,
                    tools=_RESEARCHER_TOOLS,
                    tool_choice="required",
                    max_tokens=self.max_completion_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                stopped = "llm_error"
                reasoning_chunks.append(f"[turn {turn}] LLM error: {exc}")
                break

            prompt_tokens += resp.prompt_tokens
            completion_tokens += resp.completion_tokens
            if resp.reasoning_content:
                reasoning_chunks.append(
                    f"[turn {turn}] {resp.reasoning_content}"
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.text or "",
            }
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if not resp.tool_calls:
                stopped = "no_tool_call"
                break

            for tc in resp.tool_calls:
                t0 = time.time()
                result = self._dispatch(tc["name"], tc["arguments"])
                result_json = json.dumps(result, default=str)
                try:
                    args_parsed = (
                        json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    )
                except json.JSONDecodeError:
                    args_parsed = {"_raw": tc.get("arguments", "")}
                tool_calls.append(
                    ToolCall(
                        turn=turn,
                        name=tc["name"],
                        args=args_parsed,
                        result_preview=result_json[:500],
                        result_full=result_json,
                        duration_s=round(time.time() - t0, 3),
                        error=(
                            result.get("error", "")
                            if isinstance(result, dict)
                            else ""
                        ),
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_json,
                    }
                )
            # Researcher escalation overrides the resolved-check —
            # the loop ends regardless of whether assigned blanks are
            # filled, because the researcher decided the plan is wrong.
            if self._revision_request is not None:
                stopped = "plan_revision_requested"
                break
            if self._all_assigned_resolved():
                stopped = "all_resolved"
                break

        resolved_now = [
            b for b in self.allowed
            if self.state.get(b) is not None
            and self.state[b].status == "resolved"
        ]
        return {
            "allowed_blank_ids": list(self.allowed_order),
            "turns": turn,
            "stopped": stopped,
            "resolved": resolved_now,
            "unresolved": [b for b in self.allowed if b not in set(resolved_now)],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_calls": tool_calls,
            "messages": messages,
            "reasoning": "\n".join(reasoning_chunks),
            "revision_request": self._revision_request,
        }


# ---------------------------------------------------------------------------
# Orchestrator adapter
# ---------------------------------------------------------------------------


@register_adapter
class OursGSWPlannerOrchestratorV1Adapter(_PlannerFallbackMixin, Adapter):
    """Deterministic orchestrator + per-level ReActResearcher."""

    system_id: ClassVar[str] = "ours_gsw_planner_orchestrator_v1"
    display_name: ClassVar[str] = "GSW planner + orchestrated per-level researchers"
    description: ClassVar[str] = (
        "Calls the planner, builds the topological level map, and spawns "
        "one ReActResearcher per level (sliced update_blank, no finish "
        "tool). Orchestrator owns control flow; researchers own retrieval. "
        "Falls back to flat ours_gsw_v1 on planner parse / topology failure."
    )

    def __init__(self, ctx: AdapterContext) -> None:
        super().__init__(ctx)
        self.corpus = ctx.extra.get("corpus") or load_frames_corpus()
        retriever_override = ctx.extra.get("retriever")
        if retriever_override is not None:
            self.retriever = retriever_override
        else:
            retriever_type = ctx.extra.get("retriever_type", "bm25")
            self.retriever = build_retriever(retriever_type, self.corpus)
        self.llm = LLMClient(
            model=ctx.model_name or ctx.model_id,
            base_url=ctx.base_url or None,
            api_key=ctx.api_key or None,
        )
        self._fallback_adapter: Optional[Adapter] = None
        # Per-researcher turn budget (default 20).
        self.level_max_turns = ctx.extra.get("level_max_turns", 20)
        # "deterministic" — Python loop walks topological levels, one
        # researcher per level handling every blank in that level.
        # "llm" — LLM orchestrator decides which blanks to dispatch
        # when, can also invoke the plan-updater. Default stays
        # "deterministic" until LLM mode lands end-to-end.
        self.orchestrator_mode = ctx.extra.get("orchestrator_mode", "deterministic")

    def run_question(
        self,
        question: str,
        *,
        question_id: str,
        articles: Optional[list[dict[str, Any]]] = None,
    ) -> Trajectory:
        traj = Trajectory(
            system_id=self.system_id,
            model_id=self.ctx.model_id,
            question_id=question_id,
        )
        traj.extra["gold_articles"] = articles or []
        traj.extra["fallback_flag"] = False
        start = time.time()

        # --- 1. Plan -------------------------------------------------
        try:
            plan, emit_meta = emit_plan(
                question, self.llm,
                max_tokens=4096, enable_repair=True,
            )
        except PlanEmitError as exc:
            traj.extra["plan_emit_error"] = f"{exc.kind}:{exc.detail}"
            return self._run_fallback(
                question, question_id, articles,
                reason=f"{exc.kind}:{exc.detail}",
            )

        traj.extra["plan_json"] = plan.model_dump()
        traj.extra["raw_planner_output"] = emit_meta.raw_response[:2000]

        # --- 2. Topology --------------------------------------------
        plan_dict = plan.model_dump()
        try:
            plan.target()
            order = topological_sort_blanks(plan)
            deps = build_dependency_graph(plan)
            levels = _compute_levels(deps)
        except ExecutionError as exc:
            traj.extra["topology_error"] = f"{exc.kind}:{exc.detail}"
            traj.extra["stopped_reason"] = "topology_error"
            return self._run_fallback(
                question, question_id, articles,
                reason=f"topology_error:{exc.kind}",
            )
        traj.extra["topological_order"] = order
        traj.extra["topology_levels"] = levels

        # --- 3. Init state ------------------------------------------
        state: dict[str, BlankResult] = {
            e.id: BlankResult(blank_id=e.id, status="unknown")
            for e in plan.blank_entities()
        }

        # --- 4. Dispatch by mode ------------------------------------
        if self.orchestrator_mode == "llm":
            self._run_llm_orchestrator(
                traj=traj, plan=plan, plan_dict=plan_dict,
                state=state, order=order, levels=levels,
                question=question,
            )
        else:
            self._run_deterministic(
                traj=traj, plan=plan, plan_dict=plan_dict,
                state=state, order=order, levels=levels,
                question=question,
            )

        # --- 5. Populate remaining Trajectory fields ----------------
        # In llm mode the plan may have been revised mid-run; the FINAL
        # plan is in ``traj.extra["plan_json"]`` (set by the mode-method).
        # Use it to compute is_target so blanks added by plan-updater
        # don't trigger a "unknown entity id" lookup error.
        final_plan_dict = traj.extra.get("plan_json") or plan.model_dump()
        target_ids: set[str] = {
            e.get("id")
            for e in (final_plan_dict.get("entities") or [])
            if e.get("is_target")
        }
        traj.extra["executed_blanks"] = [
            {
                "blank_id": res.blank_id,
                "value": res.value,
                "status": res.status,
                "evidence_chunk_ids": list(res.evidence_chunk_ids or []),
                "is_target": res.blank_id in target_ids,
            }
            for res in state.values()
        ]
        traj.extra["plan_emit_meta"] = {
            "prompt_tokens": emit_meta.prompt_tokens,
            "completion_tokens": emit_meta.completion_tokens,
            "repair_used": emit_meta.repair_used,
        }
        traj.wall_time_s = round(time.time() - start, 3)
        return traj

    # ------------------------------------------------------------------
    # Mode: deterministic (one researcher per topological level)
    # ------------------------------------------------------------------

    def _run_deterministic(
        self,
        *,
        traj: Trajectory,
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
    ) -> None:
        """Original Phase-1 loop: walk topological levels in order,
        spawn ONE researcher per level whose slice is all of that
        level's non-constraint-output blanks. Populates traj in-place."""
        constraint_outputs: set[str] = {
            c.output_blank_id for c in plan.constraints if c.output_blank_id
        }

        blanks_by_level: dict[int, list[str]] = defaultdict(list)
        for b_id, lvl in levels.items():
            blanks_by_level[lvl].append(b_id)

        researcher_traces: list[dict[str, Any]] = []
        total_prompt_tokens = total_completion_tokens = 0
        all_tool_calls: list[ToolCall] = []
        all_messages: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        llm_error_at_level: Optional[int] = None

        for lvl in sorted(blanks_by_level):
            slice_ids = [
                b for b in blanks_by_level[lvl]
                if b not in constraint_outputs
            ]
            if not slice_ids:
                _cascade_auto_compute(plan, state)
                continue

            system_prompt = build_researcher_prompt(
                plan_dict, order, levels,
                slice_blank_ids=slice_ids,
                state=state,
                level_index=lvl,
            )
            researcher = ReActResearcher(
                llm=self.llm,
                corpus=self.corpus,
                retriever=self.retriever,
                plan=plan,
                state=state,
                allowed_blank_ids=slice_ids,
                system_prompt=system_prompt,
                question=question,
                max_turns=self.level_max_turns,
                max_completion_tokens=self.ctx.max_completion_tokens,
            )
            trace = researcher.solve()
            researcher_traces.append(
                {
                    "level": lvl,
                    "allowed_blank_ids": trace["allowed_blank_ids"],
                    "turns": trace["turns"],
                    "stopped": trace["stopped"],
                    "resolved": trace["resolved"],
                    "unresolved": trace["unresolved"],
                    "prompt_tokens": trace["prompt_tokens"],
                    "completion_tokens": trace["completion_tokens"],
                }
            )
            total_prompt_tokens += trace["prompt_tokens"]
            total_completion_tokens += trace["completion_tokens"]
            for tc in trace["tool_calls"]:
                tc.level = lvl
            all_tool_calls.extend(trace["tool_calls"])
            for m in trace["messages"]:
                m_out = dict(m)
                m_out["_level"] = lvl
                all_messages.append(m_out)
            if trace["reasoning"]:
                reasoning_chunks.append(
                    f"=== level {lvl} ===\n{trace['reasoning']}"
                )
            if trace["stopped"] == "llm_error":
                llm_error_at_level = lvl
                break

            _cascade_auto_compute(plan, state)

        # Final answer
        tgt_id = plan.target().id
        tgt_res = state.get(tgt_id)
        if tgt_res is not None and tgt_res.status == "resolved":
            final_answer = _stringify(tgt_res.value)
            stopped_reason = "finished"
        elif llm_error_at_level is not None:
            final_answer = ""
            stopped_reason = "llm_error"
        else:
            final_answer = ""
            stopped_reason = "target_unresolved"

        traj.turns = sum(t["turns"] for t in researcher_traces)
        traj.prompt_tokens = total_prompt_tokens
        traj.completion_tokens = total_completion_tokens
        traj.tool_calls = all_tool_calls
        traj.messages = all_messages
        traj.final_answer = final_answer
        traj.reasoning = final_answer
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["researcher_traces"] = researcher_traces
        traj.extra["orchestrator_mode"] = "deterministic"

    # ------------------------------------------------------------------
    # Parallel per-blank dispatcher
    # ------------------------------------------------------------------

    MAX_PARALLEL_RESEARCHERS: ClassVar[int] = 8
    # Phase-3.2: cap on plan-updater calls per question. Two revisions
    # is enough for any plausible plan-error recovery; beyond that,
    # the corpus likely lacks the data and further revision is a
    # losing loop. The orchestrator can submit_answer with an empty/
    # best-effort string after the cap is reached.
    MAX_PLAN_UPDATES_PER_RUN: ClassVar[int] = 2
    # Phase-3.2 follow-up: per-blank dispatch cap. If a single blank
    # has triggered ``MAX_ESCALATIONS_PER_BLANK`` researcher
    # escalations (suggest_plan_revision), further dispatches of that
    # blank are REJECTED. Prevents the "spam dispatch with permuted
    # hints" loop where the orchestrator never calls request_plan_update
    # but keeps re-dispatching a blank whose retrieval target isn't in
    # the corpus.
    MAX_ESCALATIONS_PER_BLANK: ClassVar[int] = 2

    def _dispatch_subplan_parallel(
        self,
        *,
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
        blank_ids: list[str],
        hints: str = "",
        level_tag: int = 0,
    ) -> list[dict[str, Any]]:
        """Spawn one ReActResearcher per blank in ``blank_ids`` and run
        them concurrently in a thread pool. Each researcher sees a
        FRESH COPY of the current state (so siblings at the same
        dispatch don't race). After all return, merge each
        researcher's assigned blank back into the shared ``state``,
        then run ``_cascade_auto_compute``.

        Returns a list of per-researcher trace dicts — one per
        blank_id — in the order they were submitted.

        ``level_tag`` is an integer used to label tool_calls +
        messages (the inspector groups by it). In deterministic mode
        this is the topological level; in LLM-orchestrator mode it is
        the dispatch index.
        """
        if not blank_ids:
            return []

        # Snapshot state so sibling researchers at the same dispatch
        # see consistent inputs.
        pre_snapshot: dict[str, BlankResult] = {
            bid: copy.copy(br) for bid, br in state.items()
        }

        question_with_hints = question
        if hints.strip():
            question_with_hints = (
                f"{question}\n\n"
                "Orchestrator hints for this assigned blank:\n"
                f"{hints.strip()}"
            )

        def _solve_one(blank_id: str) -> tuple[str, dict[str, Any], dict[str, BlankResult]]:
            local_state: dict[str, BlankResult] = {
                bid: copy.copy(br) for bid, br in pre_snapshot.items()
            }
            system_prompt = build_researcher_prompt(
                plan_dict, order, levels,
                slice_blank_ids=[blank_id],
                state=local_state,
                level_index=level_tag,
            )
            researcher = ReActResearcher(
                llm=self.llm,
                corpus=self.corpus,
                retriever=self.retriever,
                plan=plan,
                state=local_state,
                allowed_blank_ids=[blank_id],
                system_prompt=system_prompt,
                question=question_with_hints,
                max_turns=self.level_max_turns,
                max_completion_tokens=self.ctx.max_completion_tokens,
            )
            trace = researcher.solve()
            return blank_id, trace, local_state

        # Single-blank dispatch: skip the pool (no point).
        if len(blank_ids) == 1:
            results = [_solve_one(blank_ids[0])]
        else:
            n_workers = min(len(blank_ids), self.MAX_PARALLEL_RESEARCHERS)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_solve_one, blank_ids))

        # Merge each researcher's OWN blank back into shared state.
        # (Sibling writes to other blanks inside local_state are
        # irrelevant — update_blank already rejected them via the
        # allowed_blank_ids guard.)
        out_traces: list[dict[str, Any]] = []
        for blank_id, trace, local_state in results:
            if blank_id in local_state:
                state[blank_id] = local_state[blank_id]
            # Tag tool_calls + messages with the level tag for the inspector.
            for tc in trace["tool_calls"]:
                tc.level = level_tag
            out_traces.append(trace)

        # Fire constraint cascade after all parallel researchers settle.
        _cascade_auto_compute(plan, state)
        return out_traces

    # ------------------------------------------------------------------
    # Mode: llm (orchestrator ReAct loop; parallel per-blank dispatch)
    # ------------------------------------------------------------------

    def _run_llm_orchestrator(
        self,
        *,
        traj: Trajectory,
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
    ) -> None:
        """LLM-driven orchestrator loop.

        Every turn: build the orchestrator system prompt from the
        CURRENT plan + state, make one LLM call with strict
        tool_choice, dispatch the picked tool. The plan, state, order
        and levels are all mutable — `request_plan_update` can swap
        them in mid-run.

        Populates ``traj`` in place. The orchestrator itself does NOT
        call retrieval tools — researchers do.
        """
        max_turns = int(self.ctx.extra.get("orchestrator_max_turns", 12))
        constraint_outputs: set[str] = {
            c.output_blank_id for c in plan.constraints if c.output_blank_id
        }

        researcher_traces: list[dict[str, Any]] = []
        plan_json_versions: list[dict[str, Any]] = [plan_dict]
        plan_updates: list[dict[str, Any]] = []
        dispatches: list[dict[str, Any]] = []
        all_tool_calls: list[ToolCall] = []
        all_messages: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        # Phase-3.2 follow-up: per-blank escalation counter.
        # Incremented every time a researcher returns
        # `revision_request` for that blank.
        escalations_per_blank: dict[str, int] = {}

        total_prompt_tokens = total_completion_tokens = 0
        orchestrator_turns = 0
        dispatch_idx = 0
        recent_activity: str = ""
        final_answer = ""
        stopped_reason = "max_turns"
        target_id = plan.target().id
        # Phase-3.4: track consecutive cap-rejected dispatches. Reset
        # on any non-cap-rejected tool call. After 3 in a row, the loop
        # auto-gives-up — see check at end of turn.
        consecutive_cap_rejections = 0
        recent_cap_rejection = False

        for turn in range(1, max_turns + 1):
            # Fire any pending constraint cascades before building the prompt.
            _cascade_auto_compute(plan, state)

            system_prompt = build_orchestrator_prompt(
                plan_dict, order, levels, state,
                turn_index=turn - 1,
                recent_activity=recent_activity,
                recent_cap_rejection=recent_cap_rejection,
            )
            orch_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            try:
                resp = self.llm.chat(
                    orch_messages,
                    tools=_ORCHESTRATOR_TOOLS,
                    tool_choice="required",
                    max_tokens=self.ctx.max_completion_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                stopped_reason = "llm_error"
                reasoning_chunks.append(
                    f"[orchestrator turn {turn}] LLM error: {exc}"
                )
                break

            orchestrator_turns += 1
            total_prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
            total_completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)
            if resp.reasoning_content:
                reasoning_chunks.append(
                    f"[orchestrator turn {turn}] {resp.reasoning_content}"
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.text or "",
                "_level": -1,  # -1 = orchestrator messages
            }
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in resp.tool_calls
                ]
            all_messages.append(assistant_msg)

            if not resp.tool_calls:
                stopped_reason = "no_tool_call"
                break

            # Process each tool_call (usually exactly one).
            broke_out = False
            for tc in resp.tool_calls:
                name = tc.get("name", "")
                try:
                    args = (
                        json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    )
                except json.JSONDecodeError as exc:
                    result: dict[str, Any] = {
                        "ok": False,
                        "error": f"bad tool args: {exc}",
                    }
                else:
                    # Phase-3.2: pass plan-update count (for cap) + last
                    # update summary (for hint enrichment).
                    last_pu_summary = ""
                    if plan_updates:
                        last = plan_updates[-1]
                        if not last.get("rejected"):
                            last_pu_summary = (
                                f"diff={last.get('diff_summary','')}; "
                                f"reason={str(last.get('reason',''))[:80]}"
                            )
                    pu_count = sum(
                        1 for pu in plan_updates if not pu.get("rejected")
                    )
                    result, action = self._run_orchestrator_tool(
                        name=name,
                        args=args,
                        plan=plan,
                        plan_dict=plan_dict,
                        state=state,
                        order=order,
                        levels=levels,
                        question=question,
                        dispatch_idx=dispatch_idx,
                        constraint_outputs=constraint_outputs,
                        plan_update_count=pu_count,
                        last_plan_update_summary=last_pu_summary,
                        escalations_per_blank=escalations_per_blank,
                    )
                    total_prompt_tokens += action.get("prompt_tokens", 0)
                    total_completion_tokens += action.get("completion_tokens", 0)
                    # Update mutable bookkeeping based on the action.
                    if action.get("dispatch_traces") is not None:
                        traces = action["dispatch_traces"]
                        for t in traces:
                            researcher_traces.append(
                                {
                                    "level": dispatch_idx,
                                    "allowed_blank_ids": t["allowed_blank_ids"],
                                    "turns": t["turns"],
                                    "stopped": t["stopped"],
                                    "resolved": t["resolved"],
                                    "unresolved": t["unresolved"],
                                    "prompt_tokens": t["prompt_tokens"],
                                    "completion_tokens": t["completion_tokens"],
                                }
                            )
                            total_prompt_tokens += t["prompt_tokens"]
                            total_completion_tokens += t["completion_tokens"]
                            all_tool_calls.extend(t["tool_calls"])
                            for m in t["messages"]:
                                m_out = dict(m)
                                m_out["_level"] = dispatch_idx
                                all_messages.append(m_out)
                            if t["reasoning"]:
                                reasoning_chunks.append(
                                    f"=== dispatch {dispatch_idx} ===\n{t['reasoning']}"
                                )
                        dispatches.append(
                            {
                                "dispatch_idx": dispatch_idx,
                                "blank_ids": action["blank_ids"],
                                "hints": action.get("hints", ""),
                                "partial": action.get("partial", False),
                                "revision_requests": action.get(
                                    "revision_requests", []
                                ),
                            }
                        )
                        # Phase-3.2 follow-up: bump per-blank escalation
                        # counter for blanks whose researcher escalated.
                        for rr in action.get("revision_requests") or []:
                            bid = rr.get("blank_id")
                            if bid:
                                escalations_per_blank[bid] = (
                                    escalations_per_blank.get(bid, 0) + 1
                                )
                        dispatch_idx += 1
                    if action.get("new_plan") is not None:
                        # Swap plan + plan_dict + order + levels.
                        plan = action["new_plan"]
                        plan_dict = plan.model_dump()
                        from research_agent.adapters.ours._planner_exec import (
                            build_dependency_graph,
                            topological_sort_blanks,
                        )
                        try:
                            plan.target()
                            order = topological_sort_blanks(plan)
                            deps = build_dependency_graph(plan)
                            levels = _compute_levels(deps)
                        except Exception as exc:  # noqa: BLE001
                            # Topology failed on the new plan — revert and
                            # log the rejection.
                            plan = action["old_plan"]
                            plan_dict = plan.model_dump()
                            state.clear()
                            state.update(action.get("old_state", {}))
                            result = {
                                "ok": False,
                                "error": (
                                    f"new plan has invalid topology: {exc}. "
                                    "Reverting to previous plan."
                                ),
                            }
                            plan_updates.append(
                                {
                                    "turn": turn,
                                    "reason": action["reason"],
                                    "evidence": action["evidence"],
                                    "rejected": True,
                                    "error": result["error"],
                                }
                            )
                        else:
                            constraint_outputs = {
                                c.output_blank_id for c in plan.constraints
                                if c.output_blank_id
                            }
                            target_id = plan.target().id
                            plan_json_versions.append(plan_dict)
                            plan_updates.append(
                                {
                                    "turn": turn,
                                    "reason": action["reason"],
                                    "evidence": action["evidence"],
                                    "diff_summary": action["diff_summary"],
                                    "preserved_ids": action["preserved_ids"],
                                    "added_ids": action["added_ids"],
                                    "dropped_ids": action["dropped_ids"],
                                }
                            )
                    if action.get("plan_update_rejected"):
                        plan_updates.append(
                            {
                                "turn": turn,
                                "reason": action["reason"],
                                "evidence": action["evidence"],
                                "rejected": True,
                                "error": action.get("error", ""),
                            }
                        )
                    if action.get("final_answer") is not None:
                        final_answer = action["final_answer"]
                        stopped_reason = action.get("stopped_reason", "finished")
                        broke_out = True

                result_json = json.dumps(result, default=str)
                try:
                    args_parsed = (
                        json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    )
                except json.JSONDecodeError:
                    args_parsed = {"_raw": tc.get("arguments", "")}
                tcall = ToolCall(
                    turn=turn,
                    name=name,
                    args=args_parsed,
                    result_preview=result_json[:500],
                    result_full=result_json,
                    duration_s=0.0,
                    error=(
                        result.get("error", "")
                        if isinstance(result, dict)
                        else ""
                    ),
                    level=-1,
                )
                all_tool_calls.append(tcall)
                all_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_json,
                        "_level": -1,
                    }
                )
                # Phase-3.4: track consecutive per-blank cap rejections.
                # The dispatcher returns a result with
                # `error` containing "per-blank dispatch cap reached"
                # (and ok:false) when this happens.
                is_cap_rejection = (
                    name == "dispatch_subplan"
                    and isinstance(result, dict)
                    and result.get("ok") is False
                    and "per-blank dispatch cap reached"
                    in str(result.get("error", ""))
                )
                if is_cap_rejection:
                    consecutive_cap_rejections += 1
                else:
                    consecutive_cap_rejections = 0
                # The "recent_cap_rejection" flag is what the next-turn
                # system prompt reads to decide whether to render the
                # worked-example block.
                recent_cap_rejection = is_cap_rejection

                # Refresh the per-turn recent_activity summary the
                # orchestrator sees next turn.
                if name == "dispatch_subplan" and isinstance(result, dict):
                    if is_cap_rejection:
                        recent_activity = (
                            f"dispatch CAP-REJECTED ({consecutive_cap_rejections}"
                            f" consecutive); error: "
                            + str(result.get("error", ""))[:160]
                        )
                    else:
                        resolved = result.get("resolved", {})
                        rrs = result.get("revision_requests") or []
                        recent_activity = (
                            f"dispatch #{dispatch_idx - 1}: blank_ids="
                            f"{list(resolved.keys())} resolved; "
                            f"partial={result.get('partial', False)}"
                        )
                        if rrs:
                            first = rrs[0]
                            recent_activity += (
                                f"; researcher escalated for "
                                f"{first.get('blank_id','?')!r}: "
                                f"{first.get('reason','')[:80]}"
                            )
                elif name == "request_plan_update" and isinstance(result, dict):
                    if result.get("ok"):
                        recent_activity = (
                            f"plan_update: {result.get('diff_summary', '')}"
                        )
                    else:
                        recent_activity = (
                            "plan_update rejected: "
                            + str(result.get("error", ""))[:100]
                        )

            if broke_out:
                break

            # Phase-3.4 safety net: after 3 consecutive cap-rejections,
            # the LLM has clearly ignored the worked example and is
            # stuck. Force-exit instead of burning the whole budget.
            if consecutive_cap_rejections >= 3:
                stopped_reason = "give_up_unanswerable"
                # Best-effort answer: use committed target value if any,
                # else empty.
                tgt_res = state.get(target_id)
                if tgt_res is not None and tgt_res.status == "resolved":
                    final_answer = _stringify(tgt_res.value)
                else:
                    final_answer = ""
                break

        # Final-answer fallback: if orchestrator exhausted its turns
        # but target is resolved, use that committed value.
        tgt_res = state.get(target_id)
        if not final_answer and tgt_res is not None and tgt_res.status == "resolved":
            final_answer = _stringify(tgt_res.value)
            if stopped_reason == "max_turns":
                stopped_reason = "finished_late"

        traj.turns = orchestrator_turns + sum(
            t.get("turns", 0) for t in researcher_traces
        )
        traj.prompt_tokens = total_prompt_tokens
        traj.completion_tokens = total_completion_tokens
        traj.tool_calls = all_tool_calls
        traj.messages = all_messages
        traj.final_answer = final_answer
        traj.reasoning = final_answer
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["researcher_traces"] = researcher_traces
        traj.extra["plan_json_versions"] = plan_json_versions
        traj.extra["plan_updates"] = plan_updates
        traj.extra["dispatches"] = dispatches
        traj.extra["orchestrator_mode"] = "llm"
        # Keep the FINAL plan in plan_json (replaces initial).
        traj.extra["plan_json"] = plan_dict
        traj.extra["topological_order"] = order
        traj.extra["topology_levels"] = levels

    # ------------------------------------------------------------------
    # Orchestrator tool dispatch
    # ------------------------------------------------------------------

    def _run_orchestrator_tool(
        self,
        *,
        name: str,
        args: dict[str, Any],
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
        dispatch_idx: int,
        constraint_outputs: set[str],
        plan_update_count: int = 0,
        last_plan_update_summary: str = "",
        escalations_per_blank: Optional[dict[str, int]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one orchestrator-tool call. Returns ``(result, action)``
        where ``result`` is what the orchestrator sees as the tool
        response, and ``action`` describes mutable side effects that
        the outer loop applies (dispatch traces, new plan, final
        answer)."""
        action: dict[str, Any] = {}

        if name == "get_state":
            # Compact state view.
            view: dict[str, Any] = {"blanks": []}
            for bid, br in state.items():
                view["blanks"].append(
                    {
                        "blank_id": bid,
                        "status": br.status,
                        "value": br.value,
                        "is_target": bid == plan.target().id,
                        "is_constraint_output": bid in constraint_outputs,
                    }
                )
            return view, action

        if name == "dispatch_subplan":
            blank_ids = args.get("blank_ids") or []
            hints = args.get("hints", "")
            # Phase-3.2: when a plan revision has just happened, prepend
            # a summary so the new researcher knows the prior context
            # (avoids the "researcher rediscovers the same gap" loop).
            if last_plan_update_summary:
                hints = (
                    f"[after plan revision: {last_plan_update_summary}] "
                    + hints
                ).strip()
            if not isinstance(blank_ids, list) or not blank_ids:
                return (
                    {"ok": False, "error": "blank_ids must be a non-empty list"},
                    action,
                )
            # Validate each id.
            known_ids = {e.id for e in plan.blank_entities()}
            unknown = [b for b in blank_ids if b not in known_ids]
            if unknown:
                return (
                    {
                        "ok": False,
                        "error": f"unknown blank_ids: {unknown}",
                        "known_blank_ids": sorted(known_ids),
                    },
                    action,
                )
            already = [
                b for b in blank_ids
                if state.get(b) is not None and state[b].status == "resolved"
            ]
            is_output = [b for b in blank_ids if b in constraint_outputs]
            if already:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"blank_ids already resolved: {already}. Use "
                            "`request_plan_update` if their values are wrong."
                        ),
                    },
                    action,
                )
            if is_output:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"blank_ids are constraint outputs (auto-computed): "
                            f"{is_output}. Dispatch their INPUTS instead."
                        ),
                    },
                    action,
                )
            if len(blank_ids) > self.MAX_PARALLEL_RESEARCHERS:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"too many blanks in one dispatch (max "
                            f"{self.MAX_PARALLEL_RESEARCHERS}); got "
                            f"{len(blank_ids)}."
                        ),
                    },
                    action,
                )

            # Phase-3.2 follow-up: reject re-dispatch of blanks that
            # have already triggered MAX_ESCALATIONS_PER_BLANK escalations.
            # Prevents the spam-dispatch-with-permuted-hints loop.
            esc = escalations_per_blank or {}
            burned = [
                b for b in blank_ids
                if esc.get(b, 0) >= self.MAX_ESCALATIONS_PER_BLANK
            ]
            if burned:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"per-blank dispatch cap reached for "
                            f"{burned}: each has triggered "
                            f"{self.MAX_ESCALATIONS_PER_BLANK} researcher "
                            "escalations already. Re-dispatching with "
                            "different hints will not help — the corpus "
                            "lacks this data. Either call "
                            "`request_plan_update` to revise the plan "
                            "(if the cap there hasn't been reached), or "
                            "`submit_answer` (empty answer is acceptable)."
                        ),
                        "escalations_per_blank": {b: esc.get(b, 0) for b in blank_ids},
                    },
                    action,
                )

            traces = self._dispatch_subplan_parallel(
                plan=plan, plan_dict=plan_dict,
                state=state, order=order, levels=levels,
                question=question,
                blank_ids=blank_ids,
                hints=hints,
                level_tag=dispatch_idx,
            )
            # Build orchestrator-visible summary.
            resolved_map: dict[str, Any] = {}
            any_partial = False
            for bid in blank_ids:
                br = state.get(bid)
                if br is not None and br.status == "resolved":
                    resolved_map[bid] = {
                        "value": br.value,
                        "evidence_chunk_ids": list(br.evidence_chunk_ids or []),
                    }
                else:
                    any_partial = True

            # Collect researcher escalations (Phase-3): a researcher may
            # signal that the plan is missing a step by calling
            # suggest_plan_revision. Surface these to the orchestrator.
            revision_requests: list[dict[str, Any]] = []
            for tr in traces:
                rr = tr.get("revision_request")
                if rr:
                    revision_requests.append(rr)

            result = {
                "ok": True,
                "resolved": resolved_map,
                "partial": any_partial,
            }
            if revision_requests:
                result["revision_requests"] = revision_requests
            action["dispatch_traces"] = traces
            action["blank_ids"] = blank_ids
            action["hints"] = hints
            action["partial"] = any_partial
            action["revision_requests"] = revision_requests
            return result, action

        if name == "request_plan_update":
            reason = args.get("reason", "") or ""
            evidence = args.get("evidence", "") or ""
            # Phase-3.2 cap: prevent runaway loops where the corpus
            # genuinely lacks the requested data and every revision
            # leads back to the same gap.
            if plan_update_count >= self.MAX_PLAN_UPDATES_PER_RUN:
                action["plan_update_rejected"] = True
                action["reason"] = reason
                action["evidence"] = evidence
                action["error"] = (
                    f"max plan updates ({self.MAX_PLAN_UPDATES_PER_RUN}) "
                    f"reached for this question"
                )
                return (
                    {
                        "ok": False,
                        "error": (
                            f"plan_update_cap_reached: already used "
                            f"{plan_update_count}/{self.MAX_PLAN_UPDATES_PER_RUN} "
                            "plan-updater calls. Further revisions are unlikely "
                            "to help — the corpus may genuinely lack this data. "
                            "Either submit_answer with a best-effort answer "
                            "(empty string is acceptable) or dispatch a final "
                            "researcher with explicit hints to commit a "
                            "best-guess from chunks already retrieved."
                        ),
                        "plan_update_count": plan_update_count,
                        "max_plan_updates": self.MAX_PLAN_UPDATES_PER_RUN,
                    },
                    action,
                )
            try:
                old_state = {
                    bid: copy.copy(br) for bid, br in state.items()
                }
                new_plan, diff, _meta = update_plan(
                    old_plan=plan,
                    state=state,
                    question=question,
                    reason=reason,
                    evidence=evidence,
                    llm_client=self.llm,
                    max_tokens=4096,
                    enable_repair=True,
                )
            except Exception as exc:  # noqa: BLE001
                action["plan_update_rejected"] = True
                action["reason"] = reason
                action["evidence"] = evidence
                action["error"] = str(exc)
                return (
                    {
                        "ok": False,
                        "error": f"plan_update_rejected: {exc}",
                    },
                    action,
                )
            action["new_plan"] = new_plan
            action["old_plan"] = plan
            action["old_state"] = old_state
            action["reason"] = reason
            action["evidence"] = evidence
            action["diff_summary"] = diff.summary
            action["preserved_ids"] = diff.preserved_ids
            action["added_ids"] = diff.added_ids
            action["dropped_ids"] = diff.dropped_ids
            action["prompt_tokens"] = _meta.prompt_tokens
            action["completion_tokens"] = _meta.completion_tokens
            return (
                {
                    "ok": True,
                    "diff_summary": diff.summary,
                    "preserved_ids": diff.preserved_ids,
                    "added_ids": diff.added_ids,
                    "dropped_ids": diff.dropped_ids,
                },
                action,
            )

        if name == "submit_answer":
            answer = str(args.get("answer", "")).strip()
            target_id = plan.target().id
            tgt_res = state.get(target_id)
            target_resolved = tgt_res is not None and tgt_res.status == "resolved"
            if not target_resolved:
                # Phase-3.2: when the plan-update cap is reached AND the
                # target is still unresolved, the orchestrator is allowed
                # to give up. Stamps a distinct stopped_reason so the
                # trajectory is visibly "we tried, corpus lacks the data"
                # rather than "we never finished".
                if plan_update_count >= self.MAX_PLAN_UPDATES_PER_RUN:
                    action["final_answer"] = answer  # may be ""
                    action["stopped_reason"] = "give_up_unanswerable"
                    return (
                        {
                            "ok": True,
                            "note": (
                                f"target unresolved AND plan-update cap "
                                f"({self.MAX_PLAN_UPDATES_PER_RUN}) reached; "
                                "submitting best-effort answer."
                            ),
                        },
                        action,
                    )
                return (
                    {
                        "ok": False,
                        "error": (
                            f"cannot submit_answer: target blank "
                            f"{target_id!r} is not status=resolved "
                            f"(status={tgt_res.status if tgt_res else 'missing'!r}). "
                            "Dispatch or request_plan_update first."
                        ),
                    },
                    action,
                )
            action["final_answer"] = answer or _stringify(tgt_res.value)
            action["stopped_reason"] = "finished"
            return {"ok": True}, action

        return (
            {"ok": False, "error": f"unknown tool {name!r}"},
            action,
        )
