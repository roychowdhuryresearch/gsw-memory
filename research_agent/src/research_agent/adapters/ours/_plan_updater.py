"""Plan-updater agent.

A one-shot LLM call that revises an existing ``GSWPlan`` in light of
retrieved evidence. Used by the LLM-orchestrator when it believes the
current plan is wrong or incomplete.

Inputs:
- ``old_plan``: the current (possibly stale) GSWPlan.
- ``state``: mapping ``blank_id -> BlankResult`` showing what has
  already been resolved.
- ``question``: the original question.
- ``reason``: short note from the orchestrator on WHY an update is
  needed ("the bridge entity I retrieved is a person, but the plan
  thinks it's a company").
- ``evidence``: short string capturing the retrieved facts that
  contradict the current plan.

Output: a revised ``GSWPlan`` plus a ``PlanReconcileDiff`` describing
which blank_ids survived, which were added, and which were dropped.

The updater is allowed to:
- Add or remove verb-phrases and constraints.
- Add new blanks.
- Change which blank is the target.
- Alter value_types / roles.

The updater MUST preserve existing blank_ids wherever the underlying
semantics still apply, because the orchestrator carries resolved
values across the revision by blank_id.

On validation failure the helper mirrors ``emit_plan``: a single
repair retry, then ``PlanEmitError``. The orchestrator catches this
and logs the update as ``plan_update_rejected`` without mutating its
state.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from research_agent.adapters.ours._planner_emit import (
    PlanEmitError,
    PlanEmitMeta,
    _parse_plan,
)
from research_agent.adapters.ours._planner_exec import BlankResult, GSWPlan
from research_agent.adapters.ours._planner_prompts import PLANNER_SYSTEM


# ---------------------------------------------------------------------------
# Reconcile diff
# ---------------------------------------------------------------------------


@dataclass
class PlanReconcileDiff:
    """What changed between the old and new plan, by blank id."""

    preserved_ids: list[str] = field(default_factory=list)
    added_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    # Summary string the orchestrator can surface in its trajectory.
    summary: str = ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


UPDATER_SYSTEM = (
    PLANNER_SYSTEM
    + "\n\n"
    + textwrap.dedent(
        """
        ## Plan-updater mode

        You are revising an EXISTING GSWPlan using retrieved evidence. You
        MUST preserve every existing `blank_id` whose underlying
        semantic slot still exists in the revised plan. Do NOT rename
        blank ids. Only introduce new ids for newly-introduced blank
        slots.

        You MAY:
        - Add or remove verb-phrases and constraints.
        - Add new blank entities (with fresh ids like `b_new1`).
        - Change which blank has `is_target: true`.
        - Change value_types, roles, or aux fields.

        You MUST NOT:
        - Rename existing blank ids (breaks state carryover).
        - Rename filled entities (they are grounded in the question text).
        - Introduce cycles in the dependency graph.

        Return ONLY the revised plan JSON. No prose.
        """
    ).strip()
)


def _state_summary_for_prompt(
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
) -> str:
    """Render the current state for the updater prompt."""
    lines: list[str] = []
    ent_by_id = {e.id: e for e in old_plan.entities}
    for bid, br in state.items():
        ent = ent_by_id.get(bid)
        role = getattr(ent, "role", "—") or "—"
        vt = getattr(ent, "value_type", "text") or "text"
        if br.status == "resolved":
            val = br.value
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            lines.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"RESOLVED): {val!r}"
            )
        else:
            lines.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"status=`{br.status}`)"
            )
    if not lines:
        return "_(no blanks yet)_"
    return "\n".join(lines)


def build_updater_messages(
    question: str,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    reason: str,
    evidence: str,
) -> list[dict[str, Any]]:
    """System + user message list for the one-shot updater call."""
    old_plan_json = json.dumps(old_plan.model_dump(), indent=2, default=str)
    state_block = _state_summary_for_prompt(old_plan, state)
    user = textwrap.dedent(
        f"""
        ## Question

        {question}

        ## Current plan (to revise)

        ```json
        {old_plan_json}
        ```

        ## State so far (from earlier retrieval)

        {state_block}

        ## Reason for update

        {reason.strip() or "_(no reason supplied)_"}

        ## Evidence

        {evidence.strip() or "_(no evidence supplied)_"}

        ## Task

        Emit a revised Plan JSON that reconciles the current plan with
        this evidence. Preserve blank_ids where possible. Return ONLY the
        JSON object.
        """
    ).strip()

    return [
        {"role": "system", "content": UPDATER_SYSTEM},
        {"role": "user", "content": user},
    ]


REPAIR_NOTE = (
    "Your previous response was not valid GSWPlan JSON. "
    "Re-emit a single JSON object that matches the schema and "
    "preserves the existing blank_ids wherever the slots still exist."
)


def build_updater_repair_messages(
    question: str,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    reason: str,
    evidence: str,
    bad_response: str,
    parse_error: str,
) -> list[dict[str, Any]]:
    msgs = build_updater_messages(question, old_plan, state, reason, evidence)
    msgs.append({"role": "assistant", "content": bad_response})
    msgs.append(
        {
            "role": "user",
            "content": (
                f"{REPAIR_NOTE}\n\nValidation error was:\n{parse_error}\n\n"
                "Return the corrected JSON now."
            ),
        }
    )
    return msgs


# ---------------------------------------------------------------------------
# Main helper
# ---------------------------------------------------------------------------


def update_plan(
    *,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    question: str,
    reason: str,
    evidence: str,
    llm_client: Any,
    max_tokens: int = 4096,
    enable_repair: bool = True,
) -> tuple[GSWPlan, PlanReconcileDiff, PlanEmitMeta]:
    """Run the plan-updater LLM once (with one repair retry).

    Returns ``(new_plan, diff, meta)`` on success. Raises
    ``PlanEmitError`` on terminal failure — the orchestrator should
    catch this and continue with the OLD plan.
    """
    meta = PlanEmitMeta()

    # ---- First call ----------------------------------------------------
    try:
        resp = llm_client.chat(
            messages=build_updater_messages(
                question, old_plan, state, reason, evidence
            ),
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        raise PlanEmitError(kind="llm_error", detail=str(exc)) from exc

    text = getattr(resp, "text", "") or ""
    meta.raw_response = text
    meta.raw_reasoning = getattr(resp, "reasoning_content", "") or ""
    meta.prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
    meta.completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)

    # ---- Parse attempt 1 -----------------------------------------------
    try:
        new_plan = _parse_plan(text)
        diff = _reconcile_state(old_plan, new_plan, state)
        return new_plan, diff, meta
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        meta.parse_error = str(exc)[:400]
        if not enable_repair:
            raise PlanEmitError(kind="parse_failure", detail=str(exc)) from exc

    # ---- Repair retry --------------------------------------------------
    try:
        repair_resp = llm_client.chat(
            messages=build_updater_repair_messages(
                question, old_plan, state, reason, evidence,
                bad_response=text, parse_error=meta.parse_error,
            ),
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        raise PlanEmitError(
            kind="llm_error",
            detail=f"updater repair failed: {exc}",
        ) from exc

    repair_text = getattr(repair_resp, "text", "") or ""
    meta.repair_used = True
    meta.repair_response = repair_text
    meta.repair_reasoning = getattr(repair_resp, "reasoning_content", "") or ""
    meta.prompt_tokens += int(getattr(repair_resp, "prompt_tokens", 0) or 0)
    meta.completion_tokens += int(getattr(repair_resp, "completion_tokens", 0) or 0)

    # ---- Parse attempt 2 -----------------------------------------------
    try:
        new_plan = _parse_plan(repair_text)
        diff = _reconcile_state(old_plan, new_plan, state)
        return new_plan, diff, meta
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise PlanEmitError(kind="parse_failure", detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# State reconciliation — mutates ``state`` in place
# ---------------------------------------------------------------------------


def _reconcile_state(
    old_plan: GSWPlan,
    new_plan: GSWPlan,
    state: dict[str, BlankResult],
) -> PlanReconcileDiff:
    """Preserve resolved blanks whose ids survive; drop obsolete ids;
    init ``unknown`` for newly-introduced blank ids. Mutates
    ``state`` in place and returns a summary."""
    old_ids = {e.id for e in old_plan.blank_entities()}
    new_ids = {e.id for e in new_plan.blank_entities()}

    preserved = sorted(old_ids & new_ids)
    added = sorted(new_ids - old_ids)
    dropped = sorted(old_ids - new_ids)

    # Add unknown entries for newly-introduced ids.
    for bid in added:
        if bid not in state:
            state[bid] = BlankResult(blank_id=bid, status="unknown")

    # Drop state for ids no longer in the plan.
    for bid in dropped:
        state.pop(bid, None)

    summary = (
        f"preserved={len(preserved)}, added={len(added)}, dropped={len(dropped)}"
    )
    return PlanReconcileDiff(
        preserved_ids=preserved,
        added_ids=added,
        dropped_ids=dropped,
        summary=summary,
    )
