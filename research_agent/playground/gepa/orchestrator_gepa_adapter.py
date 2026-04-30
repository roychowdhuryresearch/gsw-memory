"""GEPA adapter for the GSW orchestrator decision prompt.

Wraps ``OursGSWPlannerOrchestratorV1Adapter`` so GEPA can search over
candidate texts for the orchestrator's ``ORCHESTRATOR_RULES`` block.

Usage:
    from playground.gepa.orchestrator_gepa_adapter import (
        GSWOrchestratorGEPAAdapter, COMPONENT_NAME, BASELINE_RULES,
    )

    adapter = GSWOrchestratorGEPAAdapter(
        questions=qs, corpus=corpus, retriever=retriever, judge=judge,
        max_concurrent_questions=4,
    )
    result = gepa.optimize(
        seed_candidate={COMPONENT_NAME: BASELINE_RULES},
        trainset=[FramesQuestion(q) for q in qs[:32]],
        adapter=adapter,
        reflection_lm="openai/gpt-4.1",
        max_metric_calls=150,
    )
"""

from __future__ import annotations

import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from gepa import GEPAAdapter, EvaluationBatch

from research_agent.adapters.base import AdapterContext
from research_agent.adapters.ours import _orchestrator_prompt as orch_prompt_mod
from research_agent.adapters.ours.gsw_planner_orchestrator_v1 import (
    OursGSWPlannerOrchestratorV1Adapter,
)
from research_agent.eval.frames_dataset import FramesQuestion
from research_agent.eval.llm_judge import LLMJudge
from research_agent.eval.scoring import token_f1


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost tracking (live)
# ---------------------------------------------------------------------------

# Bedrock gpt-oss-120b us-west-2 pricing
PRICE_TASK_IN_PER_TOK = 0.075 / 1_000_000
PRICE_TASK_OUT_PER_TOK = 0.30 / 1_000_000
# OpenAI gpt-4.1 pricing (judge + reflection)
PRICE_JUDGE_IN_PER_TOK = 2.00 / 1_000_000
PRICE_JUDGE_OUT_PER_TOK = 8.00 / 1_000_000


@dataclass
class CostTracker:
    """Aggregate token usage and dollar cost across all rollouts + judge calls.

    The runner periodically logs deltas to give the user a live tab.
    """

    n_rollouts: int = 0
    n_judge_calls: int = 0
    task_in_tokens: int = 0
    task_out_tokens: int = 0
    judge_in_tokens: int = 0
    judge_out_tokens: int = 0

    def add_rollout(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.n_rollouts += 1
        self.task_in_tokens += int(prompt_tokens or 0)
        self.task_out_tokens += int(completion_tokens or 0)

    def add_judge(self, prompt_tokens: int = 1000, completion_tokens: int = 200) -> None:
        # Judge token counts aren't logged by LLMJudge; use ~1000 in × 200 out estimate.
        self.n_judge_calls += 1
        self.judge_in_tokens += prompt_tokens
        self.judge_out_tokens += completion_tokens

    @property
    def task_cost(self) -> float:
        return (
            self.task_in_tokens * PRICE_TASK_IN_PER_TOK
            + self.task_out_tokens * PRICE_TASK_OUT_PER_TOK
        )

    @property
    def judge_cost(self) -> float:
        return (
            self.judge_in_tokens * PRICE_JUDGE_IN_PER_TOK
            + self.judge_out_tokens * PRICE_JUDGE_OUT_PER_TOK
        )

    @property
    def total_cost(self) -> float:
        return self.task_cost + self.judge_cost

    def summary(self) -> str:
        return (
            f"rollouts={self.n_rollouts} judge={self.n_judge_calls} | "
            f"task_in={self.task_in_tokens:,} task_out={self.task_out_tokens:,} | "
            f"task=${self.task_cost:.2f} judge=${self.judge_cost:.2f} total=${self.total_cost:.2f}"
        )


# The single named component this adapter optimizes.
COMPONENT_NAME = "orchestrator_rules"

# The seed text — current production value of ORCHESTRATOR_RULES.
BASELINE_RULES = orch_prompt_mod.ORCHESTRATOR_RULES


# ---------------------------------------------------------------------------
# Trace / output types
# ---------------------------------------------------------------------------


@dataclass
class OrchTrajectory:
    """Per-question rollout trace, stored opaquely by GEPA and consumed
    only by ``make_reflective_dataset`` below.
    """

    question_id: str
    question: str
    gold: str
    pred: str
    judge_correct: bool
    judge_reason: str
    failure_mode: str
    f1: float
    turns: int
    stopped_reason: str
    orchestrator_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    plan_updates: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class OrchOutput:
    question_id: str
    pred: str
    gold: str
    judge_correct: bool


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patched_rules(text: str):
    """Context manager that swaps ORCHESTRATOR_RULES at module level.

    GEPA serializes adapter.evaluate() calls per-candidate, so the
    monkey-patch is safe across one (candidate × batch) call. Within
    the call we optionally fan out across questions in parallel — they
    share the same patched constant.
    """
    original = orch_prompt_mod.ORCHESTRATOR_RULES
    orch_prompt_mod.ORCHESTRATOR_RULES = text
    try:
        yield
    finally:
        orch_prompt_mod.ORCHESTRATOR_RULES = original


def _summarize_orchestrator_turns(traj_extra: dict) -> list[dict]:
    """Pull the orchestrator-level tool calls into a compact list the
    reflection LLM can read. Skips researcher-level inner work — too
    verbose for prompt budget."""
    out: list[dict] = []
    for tc in traj_extra.get("orchestrator_tool_calls", []) or []:
        name = tc.get("name") or tc.get("tool")
        args = tc.get("arguments") or tc.get("args") or {}
        result = tc.get("result")
        if isinstance(result, str) and len(result) > 200:
            result = result[:200] + "…"
        out.append({"tool": name, "args": args, "result": result})
    return out


class GSWOrchestratorGEPAAdapter(GEPAAdapter):
    """Wraps the orchestrator under a swappable ORCHESTRATOR_RULES."""

    def __init__(
        self,
        *,
        corpus,
        retriever,
        judge: LLMJudge,
        model_id: str,
        base_url: str = "",
        api_key: str = "",
        max_turns: int = 50,
        max_completion_tokens: int = 50000,
        orchestrator_mode: str = "llm",
        max_concurrent_questions: int = 4,
        cost_tracker: CostTracker | None = None,
    ):
        self.corpus = corpus
        self.retriever = retriever
        self.judge = judge
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.max_turns = max_turns
        self.max_completion_tokens = max_completion_tokens
        self.orchestrator_mode = orchestrator_mode
        self.max_concurrent_questions = max_concurrent_questions
        self.cost = cost_tracker if cost_tracker is not None else CostTracker()

    def _build_inner_adapter(self) -> OursGSWPlannerOrchestratorV1Adapter:
        ctx = AdapterContext(
            system_id="ours_gsw_planner_orchestrator_v1",
            model_id=self.model_id,
            model_name=self.model_id,
            base_url=self.base_url,
            api_key=self.api_key,
            max_turns=self.max_turns,
            max_completion_tokens=self.max_completion_tokens,
            extra={
                "corpus": self.corpus,
                "retriever": self.retriever,
                "retriever_type": "hybrid",
                "orchestrator_mode": self.orchestrator_mode,
            },
        )
        return OursGSWPlannerOrchestratorV1Adapter(ctx)

    # ------------------------------------------------------------------
    # GEPAAdapter protocol — evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[FramesQuestion],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[OrchTrajectory, OrchOutput]:
        rules_text = candidate.get(COMPONENT_NAME, BASELINE_RULES)

        outputs: list[OrchOutput] = []
        scores: list[float] = []
        trajectories: list[OrchTrajectory] = []

        with _patched_rules(rules_text):
            adapter = self._build_inner_adapter()

            def _run_one(q: FramesQuestion) -> tuple[OrchOutput, float, OrchTrajectory]:
                try:
                    traj = adapter.run_question(q.question, question_id=q.id)
                    pred = traj.final_answer or ""
                    gold = q.answer or ""
                    # Cost: rollout tokens
                    self.cost.add_rollout(
                        getattr(traj, "prompt_tokens", 0),
                        getattr(traj, "completion_tokens", 0),
                    )
                    verdict = self.judge.judge(question=q.question, gold=gold, predicted=pred)
                    judge_ok = bool(verdict.correct)
                    self.cost.add_judge()
                    extra = traj.extra or {}
                    f1 = token_f1(pred, gold) if pred else 0.0
                    out = OrchOutput(
                        question_id=q.id, pred=pred, gold=gold, judge_correct=judge_ok
                    )
                    score = 1.0 if judge_ok else 0.0
                    tr = OrchTrajectory(
                        question_id=q.id,
                        question=q.question,
                        gold=gold,
                        pred=pred,
                        judge_correct=judge_ok,
                        judge_reason=getattr(verdict, "reason", ""),
                        failure_mode=str(extra.get("stopped_reason", "")),
                        f1=f1,
                        turns=int(extra.get("orchestrator_turns", 0) or 0),
                        stopped_reason=str(extra.get("stopped_reason", "")),
                        orchestrator_tool_calls=_summarize_orchestrator_turns(extra),
                        plan_updates=list(extra.get("plan_updates", []) or []),
                    )
                    return out, score, tr
                except Exception as exc:
                    _log.warning("evaluate q%s failed: %s", q.id, exc)
                    out = OrchOutput(
                        question_id=q.id, pred="", gold=q.answer or "", judge_correct=False
                    )
                    tr = OrchTrajectory(
                        question_id=q.id, question=q.question, gold=q.answer or "",
                        pred="", judge_correct=False, judge_reason="adapter exception",
                        failure_mode="adapter_exception", f1=0.0, turns=0,
                        stopped_reason="adapter_exception", error=str(exc)[:300],
                    )
                    return out, 0.0, tr

            results: list[tuple[int, tuple[OrchOutput, float, OrchTrajectory]]] = []
            with ThreadPoolExecutor(max_workers=self.max_concurrent_questions) as pool:
                fut_to_idx = {pool.submit(_run_one, q): i for i, q in enumerate(batch)}
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    results.append((idx, fut.result()))
            results.sort(key=lambda x: x[0])
            for _, (out, score, tr) in results:
                outputs.append(out)
                scores.append(score)
                trajectories.append(tr)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    # ------------------------------------------------------------------
    # GEPAAdapter protocol — make_reflective_dataset
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[OrchTrajectory, OrchOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Pack each rollout into (input, generated, feedback) dicts for
        the reflection LLM. We surface only what the LLM needs to write
        a better ORCHESTRATOR_RULES — orchestrator-level tool calls,
        plan-update sequence, judge verdict, and gold/pred."""
        if COMPONENT_NAME not in components_to_update:
            return {}

        examples: list[dict[str, Any]] = []
        traj_list = eval_batch.trajectories or []
        for tr in traj_list:
            tools_compact = "\n".join(
                f"- {tc.get('tool')}({_args_compact(tc.get('args'))})"
                + (f" → {tc.get('result')}" if tc.get('result') else "")
                for tc in tr.orchestrator_tool_calls[:30]
            ) or "(no orchestrator tool calls captured)"
            plan_updates_compact = "\n".join(
                f"- attempt={pu.get('attempts')} ok={pu.get('ok')} "
                f"op={(pu.get('op_summary') or '')[:80]} "
                f"err={(pu.get('error') or '')[:80]}"
                for pu in (tr.plan_updates or [])[:6]
            ) or "(no plan updates)"
            verdict_str = "✓ correct" if tr.judge_correct else "✗ wrong"
            examples.append({
                "Inputs": (
                    f"Question (id={tr.question_id}): {tr.question}\n"
                    f"Gold answer: {tr.gold}"
                ),
                "Generated Outputs": (
                    f"Predicted answer: {tr.pred or '(empty)'}\n\n"
                    f"Orchestrator tool sequence:\n{tools_compact}\n\n"
                    f"Plan updates:\n{plan_updates_compact}\n\n"
                    f"Stopped: {tr.stopped_reason}  turns={tr.turns}"
                ),
                "Feedback": (
                    f"Judge verdict: {verdict_str}\n"
                    f"Reason: {tr.judge_reason or '(no reason)'}\n"
                    f"F1: {tr.f1:.2f}\n"
                    + (f"Adapter error: {tr.error}\n" if tr.error else "")
                ),
            })

        return {COMPONENT_NAME: examples}


def _args_compact(args: Any) -> str:
    if not args:
        return ""
    if isinstance(args, dict):
        parts = []
        for k, v in args.items():
            sv = str(v)
            if len(sv) > 60:
                sv = sv[:60] + "…"
            parts.append(f"{k}={sv}")
        return ", ".join(parts)
    s = str(args)
    return s[:120]
