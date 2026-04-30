"""Plan-updater agent.

A one-shot LLM call that revises an existing ``GSWPlan`` in light of
retrieved evidence. Used by the LLM-orchestrator when it believes the
current plan is wrong or incomplete.

**Phase-3.6 contract**: instead of asking the LLM to emit a full
revised plan JSON (which gpt-oss-120b 100% no-op'd on the Phase-3.4
baseline), the LLM emits ONE structured :class:`PlanDiffOp` describing
a single edit (add a blank, change the target, etc.). The op is
applied deterministically via :func:`apply_diff`. A "do nothing"
response is impossible by construction — the diff vocabulary has no
"keep current plan" op.

Inputs:
- ``old_plan``: the current (possibly stale) GSWPlan.
- ``state``: mapping ``blank_id -> BlankResult`` showing what has
  already been resolved.
- ``question``: the original question.
- ``reason``: short note from the orchestrator on WHY an update is
  needed.
- ``evidence``: short string capturing the retrieved facts that
  contradict the current plan.

Output: a revised ``GSWPlan`` plus a ``PlanReconcileDiff`` describing
which blank_ids survived, which were added, and which were dropped.

The updater MUST preserve existing blank_ids wherever the underlying
semantics still apply, because the orchestrator carries resolved
values across the revision by blank_id. (This is enforced
structurally — the diff vocabulary only allows ``add_blank`` to
introduce new ids; existing blanks keep their ids unless explicitly
``remove_blank``-ed.)

**Rejection sampling**: ``update_plan`` retries up to
``MAX_REVISION_ATTEMPTS`` (default 3) times when the LLM emits a
trivial / structurally-no-op op. After the final attempt fails it
raises :class:`PlanEmitError`. The orchestrator catches this and
treats it as ``ok=False``, surfacing the error so the orchestrator
picks a different recovery (re-dispatch / submit_answer).
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from research_agent.adapters.ours._plan_diff import (
    DiffApplyError,
    PlanDiffOp,
    apply_diff,
    is_trivial_op,
    parse_diff_op,
    plan_signature,
)
from research_agent.adapters.ours._planner_emit import PlanEmitError, PlanEmitMeta
from research_agent.adapters.ours._planner_exec import BlankResult, GSWPlan


# Maximum number of LLM calls per ``update_plan`` request. Each
# attempt that returns a trivial op or applies to a no-op signature
# burns one attempt. Failures DO NOT count against
# ``MAX_PLAN_UPDATES_PER_RUN`` in the orchestrator (that cap is for
# orchestrator-level requests).
MAX_REVISION_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Reconcile diff (unchanged shape from Phase-3.4 — used by orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class PlanReconcileDiff:
    """What changed between the old and new plan, by blank id."""

    preserved_ids: list[str] = field(default_factory=list)
    added_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    # Summary string the orchestrator can surface in its trajectory.
    summary: str = ""
    # Phase-3.6: the structured op the updater emitted, e.g.
    # "add_blank(b_year)". Surfaced in plan_updates trace for debugging.
    op_summary: str = ""
    # Number of LLM attempts used (1..MAX_REVISION_ATTEMPTS).
    attempts: int = 1


# ---------------------------------------------------------------------------
# Prompt — diff-op output contract.
# ---------------------------------------------------------------------------


UPDATER_SYSTEM = textwrap.dedent(
    """
    You are the **Plan-Updater** for a multi-agent question-answering
    system. The orchestrator has decided the current GSWPlan is wrong
    or incomplete and asked you to revise it.

    Your output is **one structured edit operation** in JSON, NOT a
    full revised plan. The edit is applied deterministically by the
    framework, the result is re-validated, and the orchestrator
    continues. Returning the current plan is impossible — the edit
    vocabulary has no "keep plan" op.

    ## Edit ops (pick exactly ONE)

    ```
    {"op": "add_blank",
     "blank": {"id": "b_<new>", "value_type": "<type>", "role": "<role>", "is_target": false},
     "wire_via_vp": {"id": "vp_<new>", "phrase": "<predicate>",
                     "subject_id": "<existing_id>", "object_id": "b_<new>"}}

    {"op": "remove_blank", "blank_id": "<existing_blank>"}

    {"op": "change_target", "from_id": "<current_target>", "to_id": "<other_blank>"}

    {"op": "add_vp",
     "vp": {"id": "vp_<new>", "phrase": "<predicate>",
            "subject_id": "<existing>", "object_id": "<existing>"}}

    {"op": "remove_vp", "vp_id": "<existing_vp>"}

    {"op": "add_constraint",
     "constraint": {"id": "c_<new>", "kind": "derived|argmax|argmin|equals|in_list|gt|lt",
                    "op": "diff|sum|avg|max|min|count|concat|mul|div|round_nearest",
                    "args_blanks": ["<blank>", ...],
                    "candidate_entity_ids": ["<id>", ...],
                    "sort_by_blank_ids": ["<blank>", ...],
                    "left_ref": "<id>", "right_ref": "<id>",
                    "output_blank_id": "<blank>"}}

    {"op": "remove_constraint", "constraint_id": "<existing>"}

    {"op": "change_value_type", "blank_id": "<existing>",
     "to": "date|number|entity|attribute|list|text|bool"}
    ```

    ## Rules

    1. New blank ids MUST be fresh — not present in the current plan.
       Use names like `b_bridge1`, `b_year`, `b_intermediate_X`.
    2. New VP/constraint ids MUST also be fresh.
    3. `add_blank` MUST come with a `wire_via_vp` that connects the
       new blank to an existing entity. A blank with no VP is
       structurally dangling and will be rejected.
    4. `change_target.from_id` MUST be the current target.
       `change_target.to_id` MUST be an existing blank.
    5. `change_value_type.to` MUST differ from the current value_type.
    6. `change_target` and `add_vp`/`remove_vp` must not introduce a
       cycle in the dependency graph.
    7. Filled entities (`kind=filled`) are grounded in the question
       text — never rename or remove them.
    8. For `round_nearest`, one arg rounds to nearest ten; two args use
       the second blank as the interval. For `in_list`, use either
       left_ref/right_ref or args_blanks=[member_blank, list_blank].

    ## Worked examples

    **Example 1 — orchestrator says a bridge step is missing:**
    Current plan: `(e_movie) --[directed_by]--> (t)` where `t` is the
    target director. But the question is "what won the Oscar in the
    year that movie X was directed". Orchestrator's `reason`:
    "missing intermediate step: bridge year between movie X and the
    Oscar winner." Right move:
    ```
    {"op": "add_blank",
     "blank": {"id": "b_year", "value_type": "date", "role": "bridge-date"},
     "wire_via_vp": {"id": "vp_year", "phrase": "released_in",
                     "subject_id": "e_movie", "object_id": "b_year"}}
    ```

    **Example 2 — orchestrator says target value is wrong type:**
    Current plan target `t` has `value_type: number` but the question
    expects an entity name. Orchestrator's `reason`: "committed value
    is a count but question asks for the player name." Right move:
    ```
    {"op": "change_value_type", "blank_id": "t", "to": "entity"}
    ```

    **Example 3 — orchestrator says a constraint dead-ends the search:**
    The plan has `argmax` over a candidate list that the corpus
    doesn't contain. Orchestrator's `reason`: "constraint c1 expects
    candidates that aren't in the corpus." Right move:
    ```
    {"op": "remove_constraint", "constraint_id": "c1"}
    ```

    Return ONLY the JSON op object. No prose, no markdown fences.
    """
).strip()


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


def _render_plan_for_updater(plan: GSWPlan) -> str:
    """Compact textual rendering of the plan for the updater prompt.

    Includes ids, kinds, value_types, and the full VP / constraint
    graph so the updater knows what it's editing without us pasting
    the raw pydantic dump.
    """
    lines: list[str] = ["### Entities"]
    for e in plan.entities:
        flags = []
        if e.is_target:
            flags.append("TARGET")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        if e.kind == "filled":
            lines.append(
                f"- `{e.id}` (filled, role=`{e.role or '—'}`): name={e.name!r}{flag_str}"
            )
        else:
            lines.append(
                f"- `{e.id}` (blank, role=`{e.role or '—'}`, "
                f"value_type=`{e.value_type or '—'}`){flag_str}"
            )

    lines.append("\n### Verb-phrases")
    if plan.verb_phrases:
        for vp in plan.verb_phrases:
            lines.append(
                f"- `{vp.id}`: ({vp.subject_id}) --[{vp.phrase}]--> ({vp.object_id})"
            )
    else:
        lines.append("_(none)_")

    lines.append("\n### Constraints")
    if plan.constraints:
        for c in plan.constraints:
            args = []
            if c.op:
                args.append(f"op={c.op}")
            if c.args_blanks:
                args.append(f"args={c.args_blanks}")
            if c.candidate_entity_ids:
                args.append(f"candidates={c.candidate_entity_ids}")
            if c.sort_by_blank_ids:
                args.append(f"sort_by={c.sort_by_blank_ids}")
            if c.output_blank_id:
                args.append(f"output={c.output_blank_id}")
            lines.append(f"- `{c.id}` (kind={c.kind}): " + ", ".join(args))
    else:
        lines.append("_(none)_")
    return "\n".join(lines)


def build_updater_messages(
    question: str,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    reason: str,
    evidence: str,
) -> list[dict[str, Any]]:
    """System + user message list for the diff-op updater call."""
    plan_render = _render_plan_for_updater(old_plan)
    state_block = _state_summary_for_prompt(old_plan, state)
    user = textwrap.dedent(
        f"""
        ## Question

        {question}

        ## Current plan

        {plan_render}

        ## State so far

        {state_block}

        ## Orchestrator's reason for asking for a revision

        {reason.strip() or "_(no reason supplied)_"}

        ## Evidence

        {evidence.strip() or "_(no evidence supplied)_"}

        ## Task

        Emit ONE diff op (JSON object) that reconciles the plan with the
        evidence. Pick the smallest meaningful change — `add_blank`
        for a missing intermediate, `change_target` for a wrong target
        blank, `add_constraint` for an aggregation/filter, etc. Return
        ONLY the JSON object.
        """
    ).strip()

    return [
        {"role": "system", "content": UPDATER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_updater_retry_messages(
    question: str,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    reason: str,
    evidence: str,
    bad_response: str,
    failure_detail: str,
) -> list[dict[str, Any]]:
    """Sharpened retry — appends the failure detail and demands a
    different op."""
    msgs = build_updater_messages(question, old_plan, state, reason, evidence)
    msgs.append({"role": "assistant", "content": bad_response})
    msgs.append(
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Your previous response failed: **{failure_detail}**

                Emit a DIFFERENT diff op that:
                - Has fresh ids for any new blank/VP/constraint
                - Actually changes the plan structure (not a no-op shape)
                - Addresses the orchestrator's reason above

                If `add_blank` was rejected because the id collides,
                pick a different fresh id. If `change_target` was
                rejected, you may instead want `change_value_type` or
                `add_blank` for a missing bridge step. Return ONLY the
                JSON op object.
                """
            ).strip(),
        }
    )
    return msgs


# ---------------------------------------------------------------------------
# Main entry point — rejection sampling over diff ops.
# ---------------------------------------------------------------------------


def _try_parse_op(text: str) -> tuple[Optional[PlanDiffOp], str]:
    """Try to extract a PlanDiffOp from raw LLM text.

    Returns ``(op, "")`` on success or ``(None, error_msg)``.
    """
    if not text:
        return None, "empty response"
    # Strip common markdown fences.
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the first line and the trailing fence.
        first_nl = stripped.find("\n")
        if first_nl >= 0:
            stripped = stripped[first_nl + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    # Some models emit a JSON object embedded in prose; find the first {...}.
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        # Last-ditch: scan for the first '{' through the last '}'.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            return None, f"no JSON object found in response: {stripped[:120]!r}"
        try:
            raw = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"JSON decode failed: {exc}"
    if not isinstance(raw, dict):
        return None, f"expected JSON object, got {type(raw).__name__}"
    try:
        op = parse_diff_op(raw)
    except ValidationError as exc:
        return None, f"diff op validation failed: {exc.errors()[0].get('msg', exc)}"
    return op, ""


def _op_summary(op: PlanDiffOp) -> str:
    """Short human-readable summary of the op for trace logs."""
    name = op.op
    if name == "add_blank":
        return f"add_blank({op.blank.id} via {op.wire_via_vp.id})"
    if name == "remove_blank":
        return f"remove_blank({op.blank_id})"
    if name == "change_target":
        return f"change_target({op.from_id} -> {op.to_id})"
    if name == "add_vp":
        return f"add_vp({op.vp.id})"
    if name == "remove_vp":
        return f"remove_vp({op.vp_id})"
    if name == "add_constraint":
        return f"add_constraint({op.constraint.id})"
    if name == "remove_constraint":
        return f"remove_constraint({op.constraint_id})"
    if name == "change_value_type":
        return f"change_value_type({op.blank_id} -> {op.to})"
    return name


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
    llm_seed: int | None = None,
) -> tuple[GSWPlan, PlanReconcileDiff, PlanEmitMeta]:
    """Rejection-sampling loop over diff ops.

    Up to :data:`MAX_REVISION_ATTEMPTS` LLM calls. The first call
    builds the system/user prompt; each subsequent call appends a
    sharpened retry message describing why the previous attempt was
    rejected (parse failure / trivial op / DiffApplyError /
    no-op-after-apply).

    Returns ``(new_plan, diff, meta)`` on success. Raises
    :class:`PlanEmitError` on terminal failure — the orchestrator
    catches and surfaces ``{"ok": false, "error": ...}``.

    The ``enable_repair`` flag is kept for API compat with the
    Phase-3.4 caller; the new contract retries on any failure regardless,
    and the orchestrator's ``max_attempts`` is governed by
    ``MAX_REVISION_ATTEMPTS``.
    """
    meta = PlanEmitMeta()
    last_bad_response = ""
    last_failure_detail = ""
    old_signature = plan_signature(old_plan)

    for attempt in range(1, MAX_REVISION_ATTEMPTS + 1):
        if attempt == 1:
            messages = build_updater_messages(
                question, old_plan, state, reason, evidence
            )
        else:
            messages = build_updater_retry_messages(
                question, old_plan, state, reason, evidence,
                bad_response=last_bad_response,
                failure_detail=last_failure_detail,
            )

        try:
            resp = llm_client.chat(
                messages=messages,
                max_tokens=max_tokens,
                seed=llm_seed,
            )
        except Exception as exc:  # noqa: BLE001
            raise PlanEmitError(kind="llm_error", detail=str(exc)) from exc

        text = getattr(resp, "text", "") or ""
        if attempt == 1:
            meta.raw_response = text
            meta.raw_reasoning = getattr(resp, "reasoning_content", "") or ""
        else:
            meta.repair_used = True
            meta.repair_response = text
            meta.repair_reasoning = getattr(resp, "reasoning_content", "") or ""
        meta.prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
        meta.completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)
        last_bad_response = text

        # Parse + validate + apply, in order. Any failure → retry.
        op, parse_err = _try_parse_op(text)
        if op is None:
            last_failure_detail = f"parse failure: {parse_err}"
            meta.parse_error = parse_err[:400]
            continue

        trivial, trivial_reason = is_trivial_op(op, old_plan)
        if trivial:
            last_failure_detail = f"trivial op rejected: {trivial_reason}"
            continue

        try:
            new_plan = apply_diff(old_plan, op)
        except DiffApplyError as exc:
            last_failure_detail = f"apply_diff failed: {exc}"
            continue

        # Belt-and-suspenders: even if we got here, double-check the
        # signature actually changed. This handles the (unlikely)
        # case of a diff that mutates fields the signature ignores.
        if plan_signature(new_plan) == old_signature:
            last_failure_detail = (
                "post-apply signature equals input plan (no-op)"
            )
            continue

        # Success.
        diff = _reconcile_state(old_plan, new_plan, state)
        diff.op_summary = _op_summary(op)
        diff.attempts = attempt
        diff.summary = (
            f"{diff.summary} via {diff.op_summary}"
            f"{f' (after {attempt} attempts)' if attempt > 1 else ''}"
        )
        return new_plan, diff, meta

    # All attempts exhausted.
    raise PlanEmitError(
        kind="parse_failure",
        detail=(
            f"plan-updater could not produce a structural change after "
            f"{MAX_REVISION_ATTEMPTS} attempts. last failure: "
            f"{last_failure_detail}"
        ),
    )


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
