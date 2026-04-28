"""System-prompt construction for the LLM orchestrator.

The orchestrator does not retrieve directly; it decides which blanks
to dispatch to a per-blank researcher, when to ask the plan-updater
to revise the plan, and when to submit the final answer.

Prompt structure, top-to-bottom:

1. **Orchestrator mission** — what its role is.
2. **Plan schema legend** (reuse of SYSTEM_PROMPT_LEGEND, so the
   orchestrator and the researchers speak the same language).
3. **Current plan** (full, not sliced — orchestrator needs the whole
   picture).
4. **Current state** (every blank with status + value).
5. **Tools** — dispatch_subplan / request_plan_update / get_state /
   submit_answer.
6. **Rules + stop conditions**.

Only sections (3) and (4) vary per turn. Sections (1), (2), (5), (6)
are static and cacheable.
"""

from __future__ import annotations

import textwrap
from typing import Any

from research_agent.adapters.ours._planner_exec import BlankResult
from research_agent.adapters.ours._planner_react_prompt import (
    SYSTEM_PROMPT_LEGEND,
    render_plan_brief,
)


ORCHESTRATOR_MISSION = textwrap.dedent(
    """
    You are the **Orchestrator** of a multi-agent research team
    answering one question.

    You hold:
    - The question (in the user message).
    - The current **GSWPlan**: a typed DAG of entities, verb-phrases,
      and constraints. One blank entity is the target whose value is
      the final answer.
    - The **global state**: which blanks are resolved, their committed
      values, and their evidence chunk ids.

    Each turn you pick EXACTLY ONE of four tools:

    - `dispatch_subplan(blank_ids: list[str], hints?: str)` — ask a
      fan-out of researchers to retrieve and commit values for a
      specific set of blanks. Each blank in the list is resolved by
      its OWN researcher in parallel. Use this for the main
      retrieval work.
    - `request_plan_update(reason: str, evidence: str)` — the plan is
      wrong or missing something. A plan-updater LLM rewrites the
      plan from the evidence you provide; the revised plan replaces
      the current one and resolved blanks are carried over by id
      where possible.
    - `get_state()` — re-read the current state. Rarely needed;
      state is already in this prompt.
    - `submit_answer(answer: str)` — terminate with the final answer.
      Only allowed when the TARGET blank is resolved in state. The
      answer should match the committed target value.

    The run ends when you call `submit_answer` (success) or hit the
    turn budget (fail).
    """
).strip()


ORCHESTRATOR_RULES = textwrap.dedent(
    """
    ## Rules

    1. Every turn MUST be exactly one tool call. Never reply in prose.
    2. `dispatch_subplan` may include 1 to 8 blank_ids. Include
       multiple ids ONLY when they have no mutual dependency
       (different rows in the topological level chart below). When in
       doubt, dispatch fewer — the researchers are isolated and will
       not help each other solve siblings.
    3. Do not dispatch a blank that is already `status=resolved` in
       state. Use `request_plan_update` for a wrong value ONLY when
       the evidence shows a schema-supported graph change is needed.
    4. Only call `request_plan_update` when you have concrete
       evidence that the plan structure is wrong. It is for structural
       edits: adding/dropping blanks, changing relations, changing the
       target blank, or changing constraints supported by the current
       schema. A single weak retrieval is NOT enough; it just means
       "dispatch again with different blanks".
       Do NOT use `request_plan_update` for answer formatting only
       (for example semicolon vs space), for replacing a committed
       value while leaving the graph unchanged, or for operations the
       schema/executor cannot express. Rounding to nearest ten,
       multiplication, and division are supported as structural
       constraint fixes. If a plan update is rejected as a no-op, do
       not retry the same request.
    5. Constraint-output blanks fill themselves once their input
       blanks are resolved. Never dispatch a blank that is the output
       of a constraint; dispatch its INPUTS.
    6. `submit_answer(answer)` is the ONLY way to end. Pass the
       committed target value verbatim. If retrieval genuinely
       cannot find the target, you may still submit a best-guess
       answer — but prefer `request_plan_update` first if the plan
       itself looks wrong.
    7. If a `dispatch_subplan` result contains `revision_requests`
       (a researcher escalated because the plan is missing a step),
       your **default next move** is `request_plan_update` using the
       first request's `reason` as the update reason and `hint` as
       the evidence. Override this default ONLY if you have a concrete
       cheaper recovery (e.g. re-dispatching the same blank with
       targeted hints will plausibly succeed). Do NOT ignore an
       escalation silently.

    8. **Plan-update cap**. You may call `request_plan_update` at
       most TWICE per question. After the second call, further
       `request_plan_update` calls will be REJECTED. If two revisions
       have not unlocked the answer, the corpus likely lacks the
       required data — do NOT keep trying. Either:
         (a) `submit_answer("")` — accept the question is unanswerable
             from the corpus (the harness will treat this as
             `stopped_reason="give_up_unanswerable"`); or
         (b) dispatch one more researcher with explicit hints asking
             it to commit a best-guess from chunks already retrieved
             (use only if you have a plausible best-guess on hand).
    9. **Per-blank dispatch cap**. After a blank has triggered TWO
       researcher escalations (`suggest_plan_revision`), further
       `dispatch_subplan` calls including that blank will be REJECTED.
       The corpus genuinely lacks the data; permuting hints will not
       help. Your options:
         (a) call `request_plan_update` (if cap above not reached) to
             revise the plan with new intermediate blanks;
         (b) dispatch a DIFFERENT (still-unresolved) blank;
         (c) `submit_answer("")` (give up, acceptable when the corpus
             gap is structural — see rule 8 path (a)).
       Do NOT keep dispatching the same blank with new hints once it
       has hit this cap; that loop is what the cap is built to prevent.

    ## Answer format for `submit_answer`

    - Person / entity: just the name.
    - Year / date: just the date value.
    - Number: just the number.
    - Bool: `True` or `False`.
    - List: comma-separated values, no surrounding prose.
    - Never prefix with "Answer:", "Final:", etc.
    """
).strip()


# Worked example shown ONLY when the most recent dispatch was
# cap-rejected. Helps the LLM recover from the per-blank cap loop
# instead of just retrying with permuted hints. Inserted between the
# rules and the answer-format section.
CAP_REJECTION_EXAMPLE = textwrap.dedent(
    """
    ## ⚠️ Recent cap-rejection — read this before your next move

    Your most recent `dispatch_subplan` call was REJECTED by the
    per-blank dispatch cap. The blank has already triggered 2
    researcher escalations; the corpus does not contain the data and
    permuting `hints` will NOT help. Re-dispatching the same blank
    again will be rejected again with the same error.

    ### Worked example — what to do

    You called:
        dispatch_subplan(["b_X"], hints="…")
    Got back:
        {"ok": false, "error": "per-blank dispatch cap reached for ['b_X']: …"}

    ❌ WRONG next move — re-dispatching the same blank with a new hint:
        dispatch_subplan(["b_X"], hints="<different wording>")
        # → will be rejected again with the same error.

    ✅ RIGHT next move (a) — if you have NOT yet called
       `request_plan_update` twice, ask for a plan revision:
        request_plan_update(
            reason="researcher escalated twice on b_X; corpus lacks this data",
            evidence="<paste the cap-rejection error here>"
        )

    ✅ RIGHT next move (b) — if both caps are reached OR plan-update
       is unlikely to help, give up cleanly:
        submit_answer("")
        # → trajectory will be stamped stopped_reason="give_up_unanswerable"

    ✅ RIGHT next move (c) — if a different unresolved blank still has
       budget, dispatch THAT blank instead.

    Do NOT call `dispatch_subplan` with the capped blank ids again.
    """
).strip()


def _state_table_for_prompt(
    plan_dict: dict[str, Any],
    state: dict[str, BlankResult],
) -> str:
    """Render every blank in the state as a row the orchestrator can read.

    Constraint-output blanks are marked so the orchestrator doesn't
    try to dispatch them.
    """
    ent_by_id = {e.get("id"): e for e in plan_dict.get("entities", [])}
    constraint_outputs = {
        c.get("output_blank_id")
        for c in plan_dict.get("constraints", [])
        if c.get("output_blank_id")
    }
    target_id = next(
        (
            e.get("id")
            for e in plan_dict.get("entities", [])
            if e.get("is_target")
        ),
        None,
    )
    rows: list[str] = []
    for bid, br in state.items():
        ent = ent_by_id.get(bid, {})
        role = ent.get("role", "—")
        vt = ent.get("value_type", "text")
        flags: list[str] = []
        if bid == target_id:
            flags.append("TARGET")
        if bid in constraint_outputs:
            flags.append("auto-computed (do not dispatch)")
        flag_str = "  [" + ", ".join(flags) + "]" if flags else ""
        if br.status == "resolved":
            val = br.value
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            rows.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"**RESOLVED**): {val!r}{flag_str}"
            )
        else:
            rows.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"status=`{br.status}`){flag_str}"
            )
    if not rows:
        return "_(no blanks in this plan)_"
    return "\n".join(rows)


def build_orchestrator_prompt(
    plan_dict: dict[str, Any],
    order: list[str],
    levels: dict[str, int],
    state: dict[str, BlankResult],
    *,
    turn_index: int = 0,
    recent_activity: str = "",
    recent_cap_rejection: bool = False,
) -> str:
    """Assemble the full system prompt for the orchestrator.

    Called every turn — sections 3 and 4 reflect the latest state and
    any plan revisions.

    ``recent_activity`` is a short orchestrator-local log (last
    dispatch's resolved ids, last plan-update summary, etc.) that the
    prompt surfaces right before the tools list so the orchestrator
    sees what it just did.

    ``recent_cap_rejection`` — when ``True``, the
    ``CAP_REJECTION_EXAMPLE`` worked-example block is inserted near
    the rules. The orchestrator-loop sets this only on turns
    immediately following a cap-rejected dispatch; it disappears as
    soon as the orchestrator picks a different tool (so the prompt
    isn't permanently bloated).
    """
    brief = render_plan_brief(plan_dict, order, levels)  # full, no slice
    state_table = _state_table_for_prompt(plan_dict, state)
    activity_section = (
        f"## Recent activity\n\n{recent_activity.strip()}\n"
        if recent_activity.strip()
        else ""
    )
    cap_section = (
        f"\n\n{CAP_REJECTION_EXAMPLE}" if recent_cap_rejection else ""
    )
    return (
        f"{ORCHESTRATOR_MISSION}\n\n"
        f"{SYSTEM_PROMPT_LEGEND}\n\n"
        f"{brief}\n\n"
        f"## Current state\n\n{state_table}\n\n"
        f"{activity_section}"
        f"{ORCHESTRATOR_RULES}"
        f"{cap_section}\n\n"
        f"(Turn {turn_index + 1}.)"
    )
