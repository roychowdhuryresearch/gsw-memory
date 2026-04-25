"""GSW-fragment planner + ReAct reasoner adapter (Phase 1 — prompt-only).

Pipeline per question:

1. **Plan** — one LLM call emits a validated ``GSWPlan`` (same path as
   ``ours_gsw_planner_v1``). Uses ``emit_plan`` from ``_planner_emit``.
2. **Topological sort** — ``compute_topo_info`` gives solve order + levels.
3. **Reason** — a ReAct loop over gpt-oss-120b with five tools:
     - ``search(query, top_k)`` — BM25 over the FRAMES article corpus.
     - ``read(chunk_id)`` — fetch the full article for a chunk's title.
     - ``update_blank(blank_id, value, evidence_chunk_ids)`` — record a
       resolved blank into shared state.
     - ``get_state()`` — inspect the current blank-fill state.
     - ``finish(answer)`` — terminate. Hard-stop: REFUSED if the target
       blank is not resolved.

The hard-stop on ``finish`` is the key experimental lever: the reasoner
cannot emit a final answer until the plan's target blank has been
written to state via ``update_blank``. The harness enforces this at
tool-dispatch time.

Falls back to the flat ``ours_gsw_v1`` adapter on planner parse failure
or topological-sort failure. Reasoner LLM errors mid-loop do NOT trigger
fallback — they return a partial trajectory stamped ``stopped_reason=
"llm_error"``.
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar, Optional

from research_agent.adapters.base import Adapter, AdapterContext, register_adapter
from research_agent.adapters.ours._planner_emit import (
    PlanEmitError,
    _PlannerFallbackMixin,
    emit_plan,
)
from research_agent.adapters.ours._planner_exec import (
    BlankResult,
    Constraint,
    ExecutionError,
    GSWPlan,
    _coerce_value,
    _compute_constraint,
    build_dependency_graph,
    topological_sort_blanks,
)
from research_agent.adapters.ours._planner_react_prompt import build_system_prompt
from research_agent.models.llm_client import LLMClient
from research_agent.models.trace import ToolCall, Trajectory
from research_agent.retrieval.bm25 import BM25Retriever
from research_agent.retrieval.corpus import load_frames_corpus
from research_agent.retrieval.dense import build_retriever


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------


_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Retrieve chunks from the FRAMES article corpus most relevant to a "
                "query. Use filled entity names and verb-phrase predicates "
                "('Pablo Picasso died_in') to target retrieval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search phrase.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5).",
                        "default": 5,
                    },
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
                "properties": {
                    "chunk_id": {"type": "string"},
                },
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
                "Search inside a specific chunk's full article for a literal "
                "pattern (case-insensitive substring). Returns up to 3 snippets "
                "of ±200 chars around each match. Use this instead of `read` "
                "when you only need one fact from a long article."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "Chunk id whose article text to search.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Literal substring to locate (case-insensitive). "
                            "No regex; keep it short and specific."
                        ),
                    },
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
                "Record a blank's resolved value into the plan's state. "
                "value_type coercion is applied automatically (number → float, "
                "etc.). Rejects unknown blank_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blank_id": {
                        "type": "string",
                        "description": "The id of the blank to fill (must exist in plan).",
                    },
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        ],
                        "description": (
                            "The resolved value. Must be a string, number, boolean, "
                            "or array of strings. The tool coerces it to the blank's "
                            "declared value_type."
                        ),
                    },
                    "evidence_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Chunk ids supporting the value. Use 'insufficient_evidence' "
                            "if no chunk clearly supports it."
                        ),
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
                "Return the current blank-fill state: which blanks are resolved, "
                "which are still unknown, and the target blank's status."
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
            "name": "finish",
            "description": (
                "Terminate with a final answer. REFUSED if the target blank is "
                "not status=resolved — call update_blank on the target first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer string shown to the user.",
                    },
                },
                "additionalProperties": False,
                "required": ["answer"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter
class OursGSWPlannerReactV1Adapter(_PlannerFallbackMixin, Adapter):
    """Plan-scaffolded ReAct reasoner with a hard-stop finish rule."""

    system_id: ClassVar[str] = "ours_gsw_planner_react_v1"
    display_name: ClassVar[str] = "GSW-fragment planner + ReAct reasoner"
    description: ClassVar[str] = (
        "Calls the v3 planner, then hands the plan + topological solve order "
        "to a ReAct reasoner with search/read/update_blank/get_state/finish "
        "tools. The reasoner keeps per-blank state; finish is refused until "
        "the target blank is resolved. Falls back to flat ours_gsw_v1 on "
        "planner parse / structural failure. Phase-1 prompt-only."
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

    # ------------------------------------------------------------------
    # Tool dispatchers (bound to per-question state)
    # ------------------------------------------------------------------

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
            snippets.append(
                {
                    "offset": idx,
                    "snippet": full[left:right],
                }
            )
            start = idx + len(pattern)
        return {
            "chunk_id": chunk_id,
            "title": chunk.title,
            "pattern": pattern,
            "match_count": len(snippets),
            "snippets": snippets,
        }

    # -- plan-scoped tools (bound by `_make_plan_dispatcher`) -----------

    @staticmethod
    def _state_view(
        state: dict[str, BlankResult],
        plan: GSWPlan,
    ) -> dict[str, Any]:
        """Canonical JSON view of the blank-fill state."""
        ent_by_id = {e.id: e for e in plan.entities}
        target_id = plan.target().id
        blanks_out: list[dict[str, Any]] = []
        for b_id, res in state.items():
            ent = ent_by_id.get(b_id)
            blanks_out.append(
                {
                    "blank_id": b_id,
                    "role": ent.role if ent else None,
                    "value_type": ent.value_type if ent else None,
                    "is_target": bool(ent and ent.is_target),
                    "status": res.status,
                    "value": res.value,
                    "evidence_chunk_ids": list(res.evidence_chunk_ids or []),
                }
            )
        return {
            "blanks": blanks_out,
            "target_id": target_id,
            "target_status": state[target_id].status,
        }

    def _make_plan_dispatcher(self, plan: GSWPlan, state: dict[str, BlankResult]):
        """Returns a closure that dispatches any of the 5 tools.

        The closure also returns a ``finish_accepted: bool`` flag so the
        outer loop knows when to break. ``finish_rejected_count`` is
        stashed in ``state`` via the closure's nonlocal for trajectory
        bookkeeping.
        """
        counters = {"finish_rejected": 0, "updates": 0, "auto_computed": 0}

        def _tool_update_blank(
            *,
            blank_id: str,
            value: Any,
            evidence_chunk_ids: list[str],
        ) -> dict[str, Any]:
            if blank_id not in state:
                return {
                    "ok": False,
                    "error": f"unknown blank_id {blank_id!r}",
                    "known_blank_ids": sorted(state.keys()),
                }
            # Reject sentinel-as-value: the string "insufficient_evidence"
            # is a marker for the evidence_chunk_ids list, NEVER a valid
            # `value`. v5.1 wording — blunt and instructive.
            if isinstance(value, str) and value.strip().lower() == "insufficient_evidence":
                return {
                    "ok": False,
                    "error": (
                        "Passing value='insufficient_evidence' is ALWAYS "
                        "wrong. The string 'insufficient_evidence' is a "
                        "metadata marker for the `evidence_chunk_ids` list; "
                        "it is never a valid `value`. Retry with your "
                        "actual best-guess (a name, number, or date) as "
                        "`value`, and put 'insufficient_evidence' inside "
                        "`evidence_chunk_ids`. Correct shape: "
                        "update_blank(blank_id='<id>', value=<your best "
                        "guess>, evidence_chunk_ids=['insufficient_evidence'])."
                    ),
                }
            # Coerce to declared value_type.
            ent = plan.entity_by_id(blank_id)
            coerced = _coerce_value(value, ent.value_type)
            ev = list(evidence_chunk_ids)
            state[blank_id] = BlankResult(
                blank_id=blank_id,
                value=coerced,
                status="resolved" if coerced is not None else "unknown",
                evidence_chunk_ids=ev,
                llm_calls=0,
                wall_time_s=0.0,
            )
            counters["updates"] += 1

            # v5.2 Fix A: cascade auto-compute across constraints whose
            # inputs are now resolved. The react adapter used to rely on
            # the LLM to compute concat/diff/argmax manually, which was
            # inconsistent (see q56, q336, q546 in v5.1). With this
            # cascade the adapter produces deterministic output-blank
            # values the moment the inputs are ready.
            auto_filled = _cascade_auto_compute(plan, state)
            counters["auto_computed"] += len(auto_filled)
            out: dict[str, Any] = {"ok": True, "state": self._state_view(state, plan)}
            if auto_filled:
                out["auto_computed"] = auto_filled
            return out

        def _tool_get_state() -> dict[str, Any]:
            return self._state_view(state, plan)

        def _tool_finish(*, answer: str) -> dict[str, Any]:
            target_id = plan.target().id
            res = state.get(target_id)
            if res is None or res.status != "resolved":
                counters["finish_rejected"] += 1
                unresolved = [b for b, r in state.items() if r.status != "resolved"]
                return {
                    "ok": False,
                    "error": (
                        f"Cannot finish: target blank {target_id!r} is not "
                        f"status=resolved (current status="
                        f"{res.status if res else 'missing'!r})."
                    ),
                    "unresolved_blanks": unresolved,
                    "hint": (
                        "Call update_blank on the target with your final answer, "
                        "then retry finish."
                    ),
                }
            return {"ok": True, "answer": answer}

        def dispatch(name: str, args_json: str) -> tuple[dict[str, Any], bool]:
            """Execute a tool. Returns (result, finish_accepted)."""
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError as exc:
                return {"error": f"bad tool args: {exc}"}, False
            try:
                if name == "search":
                    return self._tool_search(**args), False
                if name == "read":
                    return self._tool_read(**args), False
                if name == "find":
                    return self._tool_find(**args), False
                if name == "update_blank":
                    return _tool_update_blank(**args), False
                if name == "get_state":
                    return _tool_get_state(), False
                if name == "finish":
                    out = _tool_finish(**args)
                    return out, bool(out.get("ok"))
                return {"error": f"unknown tool {name!r}"}, False
            except TypeError as exc:
                return {"error": f"bad tool call: {exc}"}, False

        return dispatch, counters

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

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

        # --- 1. Plan --------------------------------------------------
        try:
            plan, emit_meta = emit_plan(
                question,
                llm_client=self.llm,
                max_tokens=self.ctx.max_completion_tokens,
            )
            traj.prompt_tokens += emit_meta.prompt_tokens
            traj.completion_tokens += emit_meta.completion_tokens
            if emit_meta.repair_used:
                traj.extra["repair_used"] = True
                traj.extra["repair_response"] = emit_meta.repair_response[:2000]
            if emit_meta.raw_reasoning:
                traj.hidden_reasoning = emit_meta.raw_reasoning
        except PlanEmitError as exc:
            if exc.kind == "llm_error":
                traj.extra["llm_error"] = f"planner call failed: {exc.detail}"
                traj.extra["stopped_reason"] = "llm_error"
                return self._run_fallback(
                    question, question_id, articles,
                    reason=f"planner_call_error:{exc.detail}",
                )
            # parse_failure
            traj.extra["parse_error"] = exc.detail[:400]
            traj.extra["stopped_reason"] = "parse_failure"
            return self._run_fallback(
                question, question_id, articles,
                reason=f"parse_failure:{exc.detail}",
            )

        traj.extra["plan_json"] = plan.model_dump()
        traj.extra["raw_planner_output"] = emit_meta.raw_response[:2000]

        # --- 2. Topological sort -------------------------------------
        plan_dict = plan.model_dump()
        try:
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

        # --- 3. Initialise blank state -------------------------------
        state: dict[str, BlankResult] = {
            e.id: BlankResult(blank_id=e.id, status="unknown")
            for e in plan.blank_entities()
        }

        # --- 4. Build system prompt + dispatcher ---------------------
        system_prompt = build_system_prompt(plan_dict, order, levels)
        dispatch, counters = self._make_plan_dispatcher(plan, state)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # --- 5. ReAct loop -------------------------------------------
        stopped_reason = "max_turns"
        final_answer = ""
        reasoning_chunks: list[str] = []

        for turn in range(1, self.ctx.max_turns + 1):
            try:
                resp = self.llm.chat(
                    messages,
                    tools=_TOOLS,
                    tool_choice="required",
                    max_tokens=self.ctx.max_completion_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                stopped_reason = "llm_error"
                traj.extra["llm_error"] = str(exc)
                break

            traj.prompt_tokens += resp.prompt_tokens
            traj.completion_tokens += resp.completion_tokens
            if resp.reasoning_content:
                reasoning_chunks.append(f"[turn {turn}] {resp.reasoning_content}")

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.text or None,
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
                # Model decided to answer without calling finish — strict
                # rule says this is an incomplete turn. Treat as "model
                # abandoned the protocol"; stamp and break.
                stopped_reason = "finish_bypassed"
                final_answer = (resp.text or "").strip()
                traj.turns = turn
                break

            finish_accepted_this_turn = False
            finish_answer_this_turn = ""
            for tc in resp.tool_calls:
                t0 = time.time()
                result, finish_ok = dispatch(tc["name"], tc["arguments"])
                if finish_ok:
                    finish_accepted_this_turn = True
                    try:
                        parsed_args = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        parsed_args = {}
                    finish_answer_this_turn = str(parsed_args.get("answer", ""))

                result_json = json.dumps(result, default=str)
                try:
                    args_parsed = (
                        json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    )
                except json.JSONDecodeError:
                    args_parsed = {"_raw": tc.get("arguments", "")}

                traj.tool_calls.append(
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

            traj.turns = turn
            if finish_accepted_this_turn:
                stopped_reason = "finished"
                final_answer = finish_answer_this_turn
                break

        # --- 6. Populate Trajectory ----------------------------------
        # If the loop exited without a clean finish but the target IS
        # resolved, prefer the committed target value over whatever text
        # the model dumped. This handles two failure modes:
        #   - max_turns with a resolved target (model never got to finish).
        #   - finish_bypassed (model emitted text instead of calling finish)
        #     — e.g. the assistant message was `{"answer":"X"}` or prose.
        if stopped_reason in ("max_turns", "finish_bypassed"):
            tgt_id = plan.target().id
            tgt_res = state.get(tgt_id)
            if tgt_res and tgt_res.status == "resolved":
                final_answer = _stringify(tgt_res.value)

        traj.final_answer = final_answer
        traj.reasoning = final_answer  # adapter-synthesised summary
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.messages = messages
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["executed_blanks"] = [
            {
                "blank_id": res.blank_id,
                "value": res.value,
                "status": res.status,
                "evidence_chunk_ids": list(res.evidence_chunk_ids or []),
                "llm_calls": res.llm_calls,
                "wall_time_s": res.wall_time_s,
            }
            for res in state.values()
        ]
        traj.extra["finish_rejections"] = counters["finish_rejected"]
        traj.extra["blank_updates"] = counters["updates"]
        traj.extra["auto_computed_blanks"] = counters["auto_computed"]
        traj.extra["plan_emit_meta"] = {
            "prompt_tokens": emit_meta.prompt_tokens,
            "completion_tokens": emit_meta.completion_tokens,
            "repair_used": emit_meta.repair_used,
        }
        traj.wall_time_s = round(time.time() - start, 3)
        return traj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (list, tuple)):
        return ", ".join(_stringify(x) for x in v)
    return str(v)


def _collect_constraint_inputs(c: Constraint) -> list[str]:
    """Return the list of blank ids a constraint reads from.

    Mirrors the input semantics in ``_planner_exec._compute_constraint``:
    - derived: args_blanks
    - argmax / argmin: sort_by_blank_ids  (candidate_entity_ids are filled
      entities, not retrievable inputs)
    - relational (equals / in_list / gt / lt): left_ref + right_ref
    """
    inputs: list[str] = []
    if c.kind == "derived":
        inputs.extend(c.args_blanks or [])
    elif c.kind in ("argmax", "argmin"):
        inputs.extend(c.sort_by_blank_ids or [])
    elif c.kind in ("equals", "in_list", "gt", "lt"):
        for ref in (c.left_ref, c.right_ref):
            if ref:
                inputs.append(ref)
    return [i for i in inputs if i]


def _cascade_auto_compute(
    plan: GSWPlan,
    state: dict[str, BlankResult],
) -> list[str]:
    """Auto-fire any constraint whose inputs are all resolved.

    Returns the list of blank ids that were newly resolved by the cascade.
    Loops until no more constraints can fire — so a chain like
    ``(diff A B) → t1`` followed by ``(concat t1 C) → t2`` resolves in
    one pass.

    Skips constraints whose output is already resolved; the model may have
    pre-filled the target with its own computed value and we do NOT
    overwrite it.
    """
    newly_filled: list[str] = []
    changed = True
    while changed:
        changed = False
        for c in plan.constraints:
            out_id = c.output_blank_id
            if not out_id:
                continue
            out = state.get(out_id)
            if out is not None and out.status == "resolved":
                continue  # already set (by LLM commit or prior cascade)
            inputs = _collect_constraint_inputs(c)
            if not inputs:
                continue
            if not all(
                state.get(i) is not None
                and state[i].status == "resolved"
                and state[i].value is not None
                for i in inputs
            ):
                continue
            result = _compute_constraint(c, plan, state)
            if result.status == "resolved":
                state[out_id] = result
                newly_filled.append(out_id)
                changed = True
    return newly_filled


def _compute_levels(deps: dict[str, set[str]]) -> dict[str, int]:
    """Blank-id → dependency level via iterative Kahn's layers.

    Level 0 = no blank deps (can run first). Level n = depends on some
    blank at level n-1. Blanks at the same level have no mutual
    dependencies.
    """
    levels: dict[str, int] = {}
    remaining = {b: set(ds) for b, ds in deps.items()}
    lvl = 0
    while remaining:
        frontier = [b for b, ds in remaining.items() if not ds]
        if not frontier:
            break  # cycle — shouldn't happen post-validator
        for b in frontier:
            levels[b] = lvl
            remaining.pop(b)
        for ds in remaining.values():
            ds.difference_update(frontier)
        lvl += 1
    return levels
