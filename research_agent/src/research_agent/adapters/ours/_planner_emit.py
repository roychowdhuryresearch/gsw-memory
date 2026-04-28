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
      - ``"parse_failure"`` — both the initial call and the repair retry
        failed Pydantic validation / JSON extraction.
    ``detail`` contains a short human-readable error summary.
    """

    def __init__(self, kind: Literal["llm_error", "parse_failure"], detail: str = ""):
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
    repair_response: str = ""
    repair_reasoning: str = ""
    parse_error: str = ""  # first-pass validation error, even when repair succeeded
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main helper
# ---------------------------------------------------------------------------


def emit_plan(
    question: str,
    llm_client: Any,
    *,
    max_tokens: int = 4096,
    enable_repair: bool = True,
) -> tuple[GSWPlan, PlanEmitMeta]:
    """Emit + validate a ``GSWPlan`` from a question. Single repair retry.

    Returns ``(plan, meta)`` on success. Raises ``PlanEmitError`` on
    terminal failure (LLM exception OR two consecutive parse failures).

    The caller is responsible for any fallback plumbing — this function
    is intentionally unaware of alternate execution paths.
    """
    meta = PlanEmitMeta()

    # ---- First call ----------------------------------------------------
    try:
        resp = llm_client.chat(
            messages=build_planner_messages(question),
            max_tokens=max_tokens,
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
        plan = _parse_plan(text)
        return plan, meta
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        meta.parse_error = str(exc)[:400]
        if not enable_repair:
            raise PlanEmitError(kind="parse_failure", detail=str(exc)) from exc

    # ---- Repair retry --------------------------------------------------
    try:
        repair_resp = llm_client.chat(
            messages=build_repair_messages(question, text, meta.parse_error),
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        raise PlanEmitError(
            kind="llm_error",
            detail=f"repair call failed: {exc}",
        ) from exc

    repair_text = getattr(repair_resp, "text", "") or ""
    meta.repair_used = True
    meta.repair_response = repair_text
    meta.repair_reasoning = getattr(repair_resp, "reasoning_content", "") or ""
    meta.prompt_tokens += int(getattr(repair_resp, "prompt_tokens", 0) or 0)
    meta.completion_tokens += int(getattr(repair_resp, "completion_tokens", 0) or 0)

    # ---- Parse attempt 2 ----------------------------------------------
    try:
        plan = _parse_plan(repair_text)
        return plan, meta
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise PlanEmitError(kind="parse_failure", detail=str(exc)) from exc


def _parse_plan(text: str) -> GSWPlan:
    """Balanced-brace JSON extract + Pydantic validate."""
    obj = _extract_json_object(text)
    return GSWPlan.model_validate(obj)


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
