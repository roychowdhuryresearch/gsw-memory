"""GSW-fragment question-planner adapter (Phase 1 — prompt-only).

Pipeline per question:

1. **Plan** — one LLM call produces a ``GSWPlan`` JSON (typed blanks +
   verb-phrases + constraints). See ``_planner_prompts.py``.
2. **Execute** — pure Python DAG traversal with per-blank LLM extraction
   for identification / projection passes, and Python-only evaluation
   for derived / argmax / argmin constraints. See ``_planner_exec.py``.
3. **Fallback** — if the plan fails to parse (after 1 repair retry) or
   the executor raises a structural error (cycle / bad ref), delegate
   to the existing flat ``ours_gsw_v1`` adapter.

The adapter emits a standard ``Trajectory`` so the harness treats it
uniformly. Every fallback is flagged in ``trajectory.extra`` so the
downstream aggregate report can separate ``gsw_planner_v1`` performance
from the fallback's performance.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Optional

from research_agent.adapters.base import Adapter, AdapterContext, register_adapter
from research_agent.adapters.ours._planner_emit import (
    PlanEmitError,
    _PlannerFallbackMixin,
    emit_plan,
)
from research_agent.adapters.ours._planner_exec import (
    ExecutionError,
    GSWPlan,
    execute,
)
from research_agent.models.llm_client import LLMClient
from research_agent.models.trace import Trajectory
from research_agent.retrieval.bm25 import BM25Retriever
from research_agent.retrieval.corpus import load_frames_corpus
from research_agent.retrieval.dense import build_retriever


@register_adapter
class OursGSWPlannerV1Adapter(_PlannerFallbackMixin, Adapter):
    """Prompt-only typed-GSW-fragment planner."""

    system_id: ClassVar[str] = "ours_gsw_planner_v1"
    display_name: ClassVar[str] = "Our GSW-fragment planner v1 (prompt-only)"
    description: ClassVar[str] = (
        "Phase-1 typed GSW-fragment planner. One LLM call emits a typed "
        "DAG over the question (filled entities, blank slots, verb-phrase "
        "edges, derived constraints). A pure-Python executor walks the DAG "
        "in topological order, running per-blank BM25 retrieval + one LLM "
        "extraction per blank, and Python-only evaluation for numeric / "
        "selector constraints. Falls back to flat ours_gsw_v1 on parse or "
        "structural failure."
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
            default_temperature=float(ctx.extra.get("llm_temperature", 0.0)),
        )
        self.llm_seed: Optional[int] = ctx.extra.get("llm_seed")
        self._top_k = int(ctx.extra.get("top_k", 8))
        # Lazily constructed fallback adapter (instantiation has side effects).
        self._fallback_adapter: Optional[Adapter] = None

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
                llm_seed=self.llm_seed,
            )
            traj.prompt_tokens += emit_meta.prompt_tokens
            traj.completion_tokens += emit_meta.completion_tokens
            if emit_meta.repair_used:
                traj.extra["repair_used"] = True
                traj.extra["repair_response"] = emit_meta.repair_response[:2000]
            if emit_meta.raw_reasoning:
                traj.hidden_reasoning = emit_meta.raw_reasoning
        except PlanEmitError as exc:
            # Capture whatever was emitted before the failure for inspector audit.
            # exc.__cause__ is the underlying ValidationError/ValueError when
            # parse_failure; for llm_error it's the transport exception.
            if exc.kind == "llm_error":
                traj.extra["llm_error"] = f"planner call failed: {exc.detail}"
                traj.extra["stopped_reason"] = "llm_error"
                return self._run_fallback(
                    question, question_id, articles,
                    reason=f"planner_call_error:{exc.detail}",
                )
            # parse_failure — we don't have emit_meta here, but the detail
            # includes the validator/JSON error text.
            traj.extra["parse_error"] = exc.detail[:400]
            traj.extra["stopped_reason"] = "parse_failure"
            return self._run_fallback(
                question, question_id, articles,
                reason=f"parse_failure:{exc.detail}",
            )

        traj.extra["plan_json"] = plan.model_dump()
        traj.extra["raw_planner_output"] = emit_meta.raw_response[:2000]

        # --- 2. Execute ----------------------------------------------
        try:
            final_answer, exec_trace = execute(
                plan,
                retriever=self.retriever,
                llm_client=self.llm,
                top_k=self._top_k,
            )
        except ExecutionError as exc:
            traj.extra["execution_error"] = f"{exc.kind}:{exc.detail}"
            traj.extra["stopped_reason"] = "execution_error"
            return self._run_fallback(
                question, question_id, articles, reason=f"execution_error:{exc.kind}"
            )

        # --- 3. Populate Trajectory ----------------------------------
        traj.final_answer = final_answer
        traj.tool_calls = exec_trace.tool_calls
        traj.turns = len(exec_trace.tool_calls)
        # Token counts inside the per-blank LLM extractions are not
        # currently threaded through the stub retriever; the planner
        # call totals already flowed into traj above.
        traj.extra["executed_blanks"] = [
            {
                "blank_id": b.blank_id,
                "value": b.value,
                "status": b.status,
                "evidence_chunk_ids": b.evidence_chunk_ids,
                "llm_calls": b.llm_calls,
                "wall_time_s": b.wall_time_s,
            }
            for b in exec_trace.executed_blanks
        ]
        traj.extra["per_blank_wall_times"] = exec_trace.wall_times
        traj.extra["stopped_reason"] = (
            "finished" if final_answer else "finished_unknown"
        )
        traj.wall_time_s = round(time.time() - start, 3)
        return traj
