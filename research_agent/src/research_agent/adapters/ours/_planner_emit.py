"""Shared plan-emission helper for the GSW-fragment planner family.

Lifted from ``gsw_planner_v1.py`` so multiple adapters (the deterministic
Python executor in ``gsw_planner_v1`` and the ReAct-reasoner in
``gsw_planner_react_v1``) can share the same "one LLM call → Pydantic
validate → one repair retry" discipline without reimplementing it.

The helper does NOT decide fallback semantics. It raises a structured
``PlanEmitError`` and the caller chooses whether to fall back to a flat
adapter or surface the failure some other way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from research_agent.adapters.ours._planner_exec import (
    GSWPlan,
    _extract_json_object,
)
from research_agent.adapters.ours._planner_prompts import (
    build_planner_messages,
    build_repair_messages,
)


# ---------------------------------------------------------------------------
# Error + metadata shapes
# ---------------------------------------------------------------------------


class PlanEmitError(Exception):
    """Raised when the planner LLM call or plan validation fails terminally.

    ``kind`` is one of:
      - ``"llm_error"`` — network / API / client raised before a response.
      - ``"parse_failure"`` — a single parse attempt failed Pydantic
        validation / JSON extraction. Used by callers (e.g. the plan
        updater) that don't run the multi-retry repair loop.
      - ``"exhausted"`` — the multi-retry repair loop in :func:`emit_plan`
        ran ``MAX_REPAIR_ATTEMPTS`` repair calls and every one of them
        failed validation. Callers should NOT silently fall back to a
        flat reasoner — the planner is broken for this question and a
        sentinel-from-priors answer is worse than an empty submission.
    ``detail`` contains a short human-readable error summary.
    """

    def __init__(
        self,
        kind: Literal["llm_error", "parse_failure", "exhausted"],
        detail: str = "",
    ):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


@dataclass
class PlanEmitMeta:
    """Bookkeeping the caller needs to populate its Trajectory.extra."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""
    raw_reasoning: str = ""
    repair_used: bool = False
    # Most-recent repair attempt — kept for legacy callers / debug UI.
    repair_response: str = ""
    repair_reasoning: str = ""
    # Per-attempt log; index 0 is the first repair, index n-1 is the last.
    # Each entry: {"response": str, "reasoning": str, "error": str}.
    repair_attempts: list[dict[str, str]] = field(default_factory=list)
    parse_error: str = ""  # first-pass validation error, even when repair succeeded
    extras: dict[str, Any] = field(default_factory=dict)


# Number of repair calls to attempt before raising ``exhausted``.
# The first LLM call + this many repairs = 1 + MAX_REPAIR_ATTEMPTS total parses.
MAX_REPAIR_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Main helper
# ---------------------------------------------------------------------------


def emit_plan(
    question: str,
    llm_client: Any,
    *,
    max_tokens: int = 4096,
    enable_repair: bool = True,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    llm_seed: int | None = None,
) -> tuple[GSWPlan, PlanEmitMeta]:
    """Emit + validate a ``GSWPlan`` from a question with multi-retry repair.

    Workflow:
      1. One initial planner LLM call.
      2. If it parses, return.
      3. Otherwise, run up to ``max_repair_attempts`` repair calls. Each
         repair message is sharpened with the attempt index so the LLM
         knows it has already failed earlier rounds.
      4. If every repair fails, raise ``PlanEmitError(kind="exhausted")``
         carrying the last validation error.

    Returns ``(plan, meta)`` on success. Raises ``PlanEmitError`` on
    terminal failure (LLM exception OR every repair retry failing).

    The caller is responsible for any fallback plumbing — this function
    is intentionally unaware of alternate execution paths.
    """
    meta = PlanEmitMeta()

    # ---- First call ----------------------------------------------------
    try:
        resp = llm_client.chat(
            messages=build_planner_messages(question),
            max_tokens=max_tokens,
            seed=llm_seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise PlanEmitError(kind="llm_error", detail=str(exc)) from exc

    text = getattr(resp, "text", "") or ""
    meta.raw_response = text
    meta.raw_reasoning = getattr(resp, "reasoning_content", "") or ""
    meta.prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
    meta.completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)

    # ---- Parse attempt 1 ----------------------------------------------
    try:
        plan = _parse_plan(text, question=question)
        return plan, meta
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        meta.parse_error = str(exc)[:400]
        if not enable_repair:
            raise PlanEmitError(kind="parse_failure", detail=str(exc)) from exc

    # ---- Repair loop (up to max_repair_attempts) ------------------------
    last_bad_response = text
    last_error = meta.parse_error
    last_exc: Exception | None = None
    for attempt_idx in range(max_repair_attempts):
        try:
            repair_resp = llm_client.chat(
                messages=build_repair_messages(
                    question,
                    last_bad_response,
                    last_error,
                    attempt=attempt_idx + 1,
                    max_attempts=max_repair_attempts,
                ),
                max_tokens=max_tokens,
                seed=llm_seed,
            )
        except Exception as exc:  # noqa: BLE001
            raise PlanEmitError(
                kind="llm_error",
                detail=f"repair call {attempt_idx + 1} failed: {exc}",
            ) from exc

        repair_text = getattr(repair_resp, "text", "") or ""
        meta.repair_used = True
        meta.repair_response = repair_text
        meta.repair_reasoning = getattr(repair_resp, "reasoning_content", "") or ""
        meta.prompt_tokens += int(getattr(repair_resp, "prompt_tokens", 0) or 0)
        meta.completion_tokens += int(getattr(repair_resp, "completion_tokens", 0) or 0)

        try:
            plan = _parse_plan(repair_text, question=question)
            meta.repair_attempts.append({
                "response": repair_text,
                "reasoning": meta.repair_reasoning,
                "error": "",
            })
            return plan, meta
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_bad_response = repair_text
            last_error = str(exc)[:400]
            last_exc = exc
            meta.repair_attempts.append({
                "response": repair_text,
                "reasoning": meta.repair_reasoning,
                "error": last_error,
            })

    # All repair attempts exhausted.
    detail = (
        f"after {max_repair_attempts} repair attempts: {last_error}"
        if last_exc is not None
        else f"after {max_repair_attempts} repair attempts"
    )
    raise PlanEmitError(kind="exhausted", detail=detail) from last_exc


def _parse_plan(text: str, *, question: str = "") -> GSWPlan:
    """Balanced-brace JSON extract + Pydantic validate."""
    obj = _extract_json_object(text)
    plan = GSWPlan.model_validate(obj)
    _validate_plan_quality(question, plan)
    return plan


_NUMERIC_DERIVED_OPS = {
    "diff",
    "sum",
    "avg",
    "max",
    "min",
    "mul",
    "div",
    "round_nearest",
}

_LITERAL_VALUE_PHRASES = {
    "value",
    "value_is",
    "equals_number",
    "number_value",
    "year_value",
}

_RISKY_HUB_CUES = re.compile(
    r"\b("
    r"difference|sum|total|average|ratio|percent|percentage|rounded|nearest|"
    r"older|younger|longer|shorter|earlier|later|before|after|"
    r"how many|how much|more|less|count|"
    r"top|leaders?|leaderboard|ranked|ranking|list|lists|both|career|"
    r"same award|same [a-z0-9 .,'-]{0,40} award|"
    r"population|score|head-to-head"
    r")\b",
    flags=re.IGNORECASE,
)

_COMPOUND_ANSWER_CUE = re.compile(
    r"\b(?:what|which|who|name|give|provide)\b.{0,80}\band\b.{0,80}"
    r"\b(?:percentage|percent|population|character|role|movie|film|team|score|number|date|year)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def _validate_plan_quality(question: str, plan: GSWPlan) -> None:
    """Reject structurally valid but low-quality planner patterns.

    These checks intentionally raise ``ValueError`` so ``emit_plan`` uses
    the existing repair loop. They are narrow quality gates, not a full
    semantic validator.
    """
    _reject_literal_constant_blanks(plan)
    _reject_numeric_literal_refs_without_values(plan)
    _reject_risky_target_hub(question, plan)


def _reject_literal_constant_blanks(plan: GSWPlan) -> None:
    literal_ids = {
        e.id
        for e in plan.filled_entities()
        if e.role == "constraint-value" or e.literal_value is not None
    }
    if not literal_ids:
        return
    blanks = {e.id for e in plan.blank_entities()}
    for vp in plan.verb_phrases:
        literal_to_blank = (
            vp.subject_id in literal_ids and vp.object_id in blanks
        ) or (
            vp.object_id in literal_ids and vp.subject_id in blanks
        )
        if not literal_to_blank:
            continue
        phrase = vp.phrase.strip().lower()
        if phrase in _LITERAL_VALUE_PHRASES or "value" in phrase:
            raise ValueError(
                "literal_constant_blank: do not create a blank whose only "
                f"purpose is to hold literal {vp.subject_id!r}/{vp.object_id!r}; "
                "use a filled constraint-value entity with literal_value and "
                "reference it directly in args_refs"
            )


def _reject_numeric_literal_refs_without_values(plan: GSWPlan) -> None:
    ent_by_id = {e.id: e for e in plan.entities}
    for c in plan.constraints:
        if c.kind != "derived" or c.op not in _NUMERIC_DERIVED_OPS:
            continue
        for ref in c.args_refs:
            ent = ent_by_id.get(ref)
            if (
                ent is not None
                and ent.kind == "filled"
                and ent.role == "constraint-value"
                and ent.literal_value is None
            ):
                raise ValueError(
                    "literal_value_missing: numeric derived constraint "
                    f"{c.id!r} references filled constraint-value {ref!r} "
                    "without literal_value; keep name grounded in the "
                    "question and add the typed literal_value"
                )


def _reject_risky_target_hub(question: str, plan: GSWPlan) -> None:
    try:
        target_id = plan.target().id
    except Exception:
        return
    incident = [
        vp
        for vp in plan.verb_phrases
        if target_id in (vp.subject_id, vp.object_id)
    ]
    if len(incident) < 3:
        return
    if any(c.output_blank_id == target_id for c in plan.constraints):
        return
    if not (
        _RISKY_HUB_CUES.search(question or "")
        or _COMPOUND_ANSWER_CUE.search(question or "")
    ):
        return
    raise ValueError(
        "risky_target_hub: target blank has "
        f"{len(incident)} direct verb-phrase clues and the question "
        "contains computation/list/temporal/compound cues; create "
        "intermediate blanks and constraints instead of wiring every "
        "clue directly to TARGET"
    )


# ---------------------------------------------------------------------------
# Shared fallback plumbing for planner-family adapters
# ---------------------------------------------------------------------------


class _PlannerFallbackMixin:
    """Mixin that provides `_get_fallback_adapter` + `_run_fallback`.

    Assumes the including class has:
      - ``self.ctx`` (AdapterContext)
      - ``self.corpus`` (shared article corpus)
      - ``self.retriever`` (shared BM25 retriever)
      - ``self.system_id`` (class-level string)

    and attribute ``_fallback_adapter: Optional[Adapter]`` initialised to
    ``None`` in ``__init__``.
    """

    def _get_fallback_adapter(self):
        """Lazily instantiate the ours_gsw_v1 flat adapter for fallbacks."""
        existing = getattr(self, "_fallback_adapter", None)
        if existing is not None:
            return existing
        # Import lazily to avoid circular-import noise.
        from research_agent.adapters.base import AdapterContext
        from research_agent.adapters.baselines.ours_gsw_v1 import OursGSWv1Adapter

        fallback_ctx = AdapterContext(
            system_id="ours_gsw_v1",
            model_id=self.ctx.model_id,
            model_name=self.ctx.model_name,
            base_url=self.ctx.base_url,
            api_key=self.ctx.api_key,
            max_turns=self.ctx.max_turns,
            max_completion_tokens=self.ctx.max_completion_tokens,
            extra={
                "corpus": self.corpus,
                "retriever": self.retriever,
            },
        )
        self._fallback_adapter = OursGSWv1Adapter(fallback_ctx)
        return self._fallback_adapter

    def _run_fallback(
        self,
        question: str,
        question_id: str,
        articles,
        reason: str,
    ):
        """Delegate to the flat fallback adapter, stamp trajectory flags."""
        traj = self._get_fallback_adapter().run_question(
            question,
            question_id=question_id,
            articles=articles,
        )
        traj.extra["fallback_flag"] = True
        traj.extra["fallback_reason"] = reason
        # Preserve caller's system_id so harness bookkeeping stays consistent.
        traj.system_id = self.system_id
        return traj
