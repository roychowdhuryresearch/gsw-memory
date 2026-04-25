"""System-prompt construction for a single per-level researcher.

The researcher only resolves a SLICE of the plan's blanks. Its system
prompt differs from the full planner-react prompt in three ways:

1. **Scope banner** at the top that names the exact blank ids it may
   update.
2. **Upstream resolved values** table, so bridging values from earlier
   levels are visible (but not writable).
3. **No finish tool instructions.** The researcher ends its own loop
   by resolving all its assigned blanks; the orchestrator owns final
   answer emission.

The base protocol (v5.1 wording — `insufficient_evidence` as a marker
never as a value, evidence_chunk_ids required, tool-call every turn) is
inherited unchanged from ``_planner_react_prompt.py``. We rely on
``render_plan_brief(..., slice_blank_ids=...)`` to filter the plan to
the relevant subgraph.
"""

from __future__ import annotations

import textwrap
from typing import Any, Optional

from research_agent.adapters.ours._planner_exec import BlankResult
from research_agent.adapters.ours._planner_react_prompt import (
    render_plan_brief,
)


# Researcher-scoped legend. Mirrors ``SYSTEM_PROMPT_LEGEND`` in
# ``_planner_react_prompt.py`` but with the Critical-protocol bullets
# rewritten for the no-finish researcher contract (only update_blank
# ends the researcher's work; the orchestrator emits the final answer).
RESEARCHER_LEGEND = textwrap.dedent(
    """
    ## Critical protocol

    - Every assistant turn MUST contain at least one real tool call.
    - Never reply with prose-only text or inline JSON that imitates a
      tool call.
    - Fill blanks by calling `update_blank(blank_id, value,
      evidence_chunk_ids)`; `evidence_chunk_ids` is REQUIRED on every
      update.
    - Your run ENDS when every assigned blank listed above has
      `status=resolved` in state — no explicit termination call is
      needed. There is no `finish` tool at your level.

    ## Plan schema legend — read this before the plan below

    A plan is a small graph describing what the question is asking. It
    has three arrays: `entities`, `verb_phrases`, `constraints`.

    ### Entity

    - `kind="filled"` means the entity is named in the question and
      has a `name`.
    - `kind="blank"` means you must resolve it. Blank `value_type` is
      one of: `date`, `number`, `entity`, `attribute`, `list`, `text`,
      `bool`.
    - Exactly one blank has `is_target: true`; its resolved value is
      the final answer for the overall question.

    `role` explains why an entity exists in the plan.
    - Common filled roles: `subject`, `scope-filter`, `year-anchor`,
      `date-anchor`, `as-of-date`, `list-header`, `candidate`.
    - Common blank roles: `target`, `bridge-entity`, `bridge-date`,
      `bridge-number`, `bridge-attribute`, `list-member`,
      `comparison-output`, `aggregate-output`.

    ### VerbPhrase  `(subject, phrase, object)`

    A typed binary relation between two entities. The `phrase` is a
    short snake_case predicate like `died_in`, `has_population`,
    `entered_service_in`. Verb-phrases do two jobs:
    - Identify a blank from a filled anchor plus a predicate.
    - Project from a resolved blank to a downstream blank.

    Temporal predicates like `in_year`, `signed_in_year`,
    `released_in_year`, and `as_of_date` should be used literally as
    retrieval hints.

    ### Constraint

    A constraint computes a blank without retrieval.
    - `derived` applies an op such as `diff`, `sum`, `avg`, `max`,
      `min`, `count`, or `concat` to input blanks.
    - `argmax` / `argmin` pick the winning entity from candidates
      using aligned ranking blanks.

    Every constraint has an `output_blank_id`. When all input blanks
    are resolved, the ORCHESTRATOR automatically computes the output.
    You do NOT call `update_blank` on constraint-output blanks — even
    if one is in your assigned slice, just make sure the input blanks
    are resolved and the orchestrator will auto-fill it.
    """
).strip()


RESEARCHER_TASK_BANNER = textwrap.dedent(
    """
    ## Your task (this researcher only)

    You are one of several researchers resolving a larger question.
    An orchestrator has assigned you a SLICE of the plan — a specific
    set of blanks to fill.

    ### Hard contract

    - **Every assigned blank in your slice MUST receive an
      `update_blank` call before you stop.** This is non-negotiable.
      If retrieval is weak after 2 tool calls for a blank, your next
      action for that blank must be `update_blank` with your best-guess
      value and `evidence_chunk_ids=["insufficient_evidence"]`.
      A noisy commit is ALWAYS better than leaving a blank unresolved.
    - Leaving an assigned blank unresolved means the whole question
      fails downstream. The only way your run ends successfully is
      when every blank in your slice has `status=resolved` in state.
    - You MUST only call `update_blank` on the ids listed in the
      assigned slice below. The tool will reject writes to any other
      blank with an error.
    - Upstream bridging values (from earlier researchers) are visible
      in the "Prior resolved values" table and via `get_state`, but
      they are read-only for you.
    - There is no `finish` tool at your level. Your run ends when
      every assigned blank has `status=resolved` in state.

    ### Workflow

    1. For each assigned blank: call `search` (and `find`/`read` if
       needed) to gather evidence. Search-result snippets count as
       evidence; you do not need to read the full article when the
       snippet already contains a plausible value.
    2. As soon as you have a plausible value, call `update_blank`.
    3. If after 2 retrieval tool calls for the same blank you still
       cannot find the answer, commit your best-guess value with
       `evidence_chunk_ids=["insufficient_evidence"]` and move on.
    4. Do not spend more than 3 consecutive tool turns on one blank
       while another assigned blank is still unresolved. Commit the
       current blank, then move on.
    5. Repeat for every assigned blank. Do NOT stop early — a
       researcher that ends with any assigned blank unresolved has
       failed.
    """
).strip()


RESEARCHER_RULES = textwrap.dedent(
    """
    ## Retrieval policy

    - `search` is the default retrieval step.
    - Prefer short queries built from a filled entity name + a
      verb-phrase predicate (e.g., `<entity-name> <predicate>`).
    - Use upstream resolved values from "Prior resolved values" as
      retrieval anchors when an assigned blank depends on them.
    - If snippets already contain the fact, commit from the snippets.
      Do not repeat the same search or find call just to get more
      confidence.
    - After at most 2 weak retrieval calls for the same blank
      (`search`, or `search` + `read`/`find`), commit your best value
      with `evidence_chunk_ids=["insufficient_evidence"]`. Your best
      value is an actual guess (name / number / date / list / etc.) —
      the string `"insufficient_evidence"` goes ONLY inside the
      `evidence_chunk_ids` list, never as the `value`. A noisy commit
      beats leaving an assigned blank unresolved.
    - For `value_type="list"`, a partial best-effort list is better than no update.
      Store the list as an array of short strings and correct it later
      if better evidence appears.

    ## State-update rule

    Your job is to RECORD values via `update_blank`. A value you
    inferred mentally but never committed does not count.

    - Every `update_blank` call MUST include `evidence_chunk_ids`.
    - If later evidence contradicts a blank you already committed,
      call `update_blank` again with the corrected value.
    - Multiple blanks in the same slice are independent. You may
      update one blank as soon as it is plausible; do not wait until
      you have solved every blank in the slice.
    - `get_state` shows ALL blanks in the larger plan — the ones
      marked `writable=false` are already resolved by earlier
      researchers; DO NOT try to update them.
    - Passing `value="insufficient_evidence"` is ALWAYS wrong. It is a
      metadata marker for `evidence_chunk_ids`; it is never a valid
      `value`.
    - Passing placeholder values is also wrong: never use `value=""`,
      `value="unknown"`, `value="None"`, `value="null"`, a blank id
      such as `"t"` or `"b_year"`, or a JSON/tool-call string as the
      value. If evidence is weak, use a real best-guess value and put
      `"insufficient_evidence"` in `evidence_chunk_ids`.

    ## Tool-error recovery

    If any tool returns `error` or `ok:false`, issue a corrected tool
    call next. Do not explain the failure in prose.

    ## Tools

    - `search(query, top_k=5)` — retrieve top-k chunks for a query.
    - `read(chunk_id)` — full article text for a returned chunk.
    - `find(chunk_id, pattern)` — locate a literal substring inside a
      chunk's article and return ±200-char snippets around up to 3
      matches. Prefer this over `read` when you only need one fact
      from a long article.
    - `update_blank(blank_id, value, evidence_chunk_ids)` — record a
      blank's resolved value. Rejects ids outside your assigned slice.
    - `get_state()` — inspect the current blank-fill state across the
      whole plan (with a writable flag per entry).

    (There is no `finish` tool at this level. Your run ends when every
    assigned blank is resolved, or when you exhaust your turn budget.)
    """
).strip()


def _format_resolved_row(
    blank_id: str,
    ent: dict[str, Any],
    br: BlankResult,
) -> str:
    role = ent.get("role", "—")
    vt = ent.get("value_type", "text")
    val = br.value
    if isinstance(val, str) and len(val) > 80:
        val = val[:77] + "..."
    return f"- `{blank_id}` (role=`{role}`, value_type=`{vt}`): {val!r}"


def build_researcher_prompt(
    plan_dict: dict[str, Any],
    order: list[str],
    levels: dict[str, int],
    slice_blank_ids: list[str],
    state: dict[str, BlankResult],
    level_index: int,
) -> str:
    """Assemble the full system prompt for a single researcher.

    Sections, top-to-bottom:
    1. Scope banner + assigned slice id list.
    2. Plan schema legend (static, shared with the flat adapter).
    3. Plan brief FILTERED to the slice (entities + VPs + constraints
       touching slice blanks; filled anchors kept).
    4. Prior resolved values table.
    5. Researcher rules (retrieval / state-update / tool list).
    """
    ent_by_id = {e.get("id"): e for e in plan_dict.get("entities", [])}

    # --- Prior-resolved section --------------------------------------
    resolved_rows: list[str] = []
    for b_id, br in state.items():
        if b_id in set(slice_blank_ids):
            continue  # assigned to me, not upstream
        if br.status != "resolved":
            continue
        ent = ent_by_id.get(b_id, {})
        resolved_rows.append(_format_resolved_row(b_id, ent, br))

    if resolved_rows:
        prior_section = (
            "## Prior resolved values (read-only)\n\n"
            "These blanks were filled by earlier researchers. They are "
            "visible via `get_state()` and can be used as retrieval "
            "anchors, but you cannot update them.\n\n"
            + "\n".join(resolved_rows)
        )
    else:
        prior_section = (
            "## Prior resolved values (read-only)\n\n"
            "_(none — you are the first researcher)_"
        )

    # --- Slice summary -----------------------------------------------
    slice_rows = []
    for b_id in slice_blank_ids:
        ent = ent_by_id.get(b_id, {})
        slice_rows.append(
            f"- `{b_id}` (role=`{ent.get('role','—')}`, "
            f"value_type=`{ent.get('value_type','text')}`"
            + (", **TARGET**" if ent.get("is_target") else "")
            + ")"
        )
    scope = (
        f"{RESEARCHER_TASK_BANNER}\n\n"
        f"**Level {level_index + 1}** — resolve exactly these blanks:\n\n"
        + "\n".join(slice_rows)
    )

    # --- Filtered plan brief -----------------------------------------
    brief = render_plan_brief(plan_dict, order, levels, slice_blank_ids=slice_blank_ids)

    return (
        f"{scope}\n\n"
        f"{RESEARCHER_LEGEND}\n\n"
        f"{brief}\n\n"
        f"{prior_section}\n\n"
        f"{RESEARCHER_RULES}"
    )
