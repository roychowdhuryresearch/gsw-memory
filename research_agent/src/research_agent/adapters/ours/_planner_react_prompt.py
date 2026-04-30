"""System-prompt construction for the planner-scaffolded ReAct reasoner.

The reasoner reads THREE things before it acts:

1. A **schema legend** — what `argmin`, `bridge-date`, `category=true`,
   `state="expanded_from_collective:..."` etc. actually mean. Without this the
   model cannot interpret the plan we hand it. Written as a static string so
   it can sit in a prompt-prefix cache.
2. **The plan for this question** — a compact markdown rendering of the
   emitted ``GSWPlan`` (entities + roles + VPs + constraints).
3. **The topological solve order** — the plan's blanks in dependency order,
   annotated with retrieval signals the reasoner should use.

The rules + tool summary follow after. Only sections 2 and 3 vary per
question; the rest is static.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Static prefix: legend + rules + tool summary
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_LEGEND = textwrap.dedent(
    """
    You are answering a research question using a pre-emitted plan.

    ## Critical protocol

    - Every assistant turn before completion MUST contain at least one
      real tool call.
    - Never reply with prose-only text or inline JSON that imitates a
      tool call.
    - Fill blanks by calling `update_blank(blank_id, value,
      evidence_chunk_ids)`; `evidence_chunk_ids` is REQUIRED on every
      update.
    - The only valid ending is:
      `update_blank(target_id, target_value, evidence_chunk_ids)` then
      `finish(answer=target_value)`.
    - The `finish` tool will REFUSE if the target blank is unresolved.

    ## Plan schema legend — read this before the plan below

    A plan is a small graph describing what the question is asking. It has
    three arrays: `entities`, `verb_phrases`, `constraints`.

    ### Entity

    - `kind="filled"` means the entity is named in the question and has
      a `name`.
    - `kind="blank"` means you must resolve it. Blank `value_type` is one
      of: `date`, `number`, `entity`, `attribute`, `list`, `text`, `bool`.
    - Exactly one blank has `is_target: true`; its resolved value is the
      final answer.

    `role` explains why an entity exists in the plan.
    - Common filled roles: `subject`, `scope-filter`, `year-anchor`,
      `date-anchor`, `as-of-date`, `list-header`, `candidate`.
    - Common blank roles: `target`, `bridge-entity`, `bridge-date`,
      `bridge-number`, `bridge-attribute`, `list-member`,
      `comparison-output`, `aggregate-output`.

    Auxiliary fields:
    - `category: true` means the filled entity is a category or role,
      not a named individual.
    - `state="expanded_from_collective:<group>"` marks a valid candidate
      expanded from a collective named in the question.

    ### VerbPhrase  `(subject, phrase, object)`

    A typed binary relation between two entities. The `phrase` is a short
    snake_case predicate like `died_in`, `has_population`,
    `entered_service_in`. Verb-phrases do two jobs:

    - Identify a blank from a filled anchor plus a predicate.
    - Project from a resolved entity-blank to a downstream blank.

    Temporal predicates like `in_year`, `signed_in_year`,
    `released_in_year`, and `as_of_date` should be used literally as
    retrieval hints.

    ### Constraint

    A constraint computes a blank without retrieval.

    - `derived` applies an operation such as `diff`, `sum`, `avg`,
      `max`, `min`, `count`, `concat`, `mul`, `div`, or
      `round_nearest` to input blanks.
    - `argmax` / `argmin` pick the winning entity from candidates using
      aligned ranking blanks.
    - Relational kinds like `equals`, `in_list`, `gt`, and `lt` produce
      bool outputs when their inputs are resolved.

    Every constraint has an `output_blank_id`. When all of the
    constraint's input blanks are resolved, the adapter AUTOMATICALLY
    computes the output blank — you do NOT need to call `update_blank`
    for constraint outputs. Just make sure the INPUT blanks are resolved
    (by retrieval + `update_blank` on them); the output will appear in
    state as soon as the last input lands.
    """
).strip()


SYSTEM_PROMPT_RULES = textwrap.dedent(
    """
    ## Your job

    Walk the topological solve order below. For each blank in order:
    1. If the blank is the output of a constraint (derived / argmax /
       argmin / etc.), SKIP it — the adapter auto-computes it once the
       input blanks are resolved. You'll see it appear in state.
    2. Otherwise, start with `search`.
    3. Use `read` only when a returned chunk is ambiguous and you need
       the full article text.
    4. As soon as you have a usable value, call `update_blank`.
    5. Move on to the next blank.

    When the TARGET blank shows `status=resolved` in state (whether you
    committed it directly or a constraint auto-produced it), call
    `finish(answer=<the target value>)`.

    ## Retrieval policy

    - `search` is the default retrieval step.
    - Prefer short queries built from the filled entity name and the
      verb-phrase predicate, for example `Pablo Picasso died_in`.
    - If the `search` snippets already contain the fact, commit from the
      snippets; do not read the full article just to feel safer.
    - After at most 2 weak retrieval calls for the same blank
      (`search`, or `search` + `read`), commit your best value with
      `evidence_chunk_ids=["insufficient_evidence"]`. Your best value
      is an actual guess (name / number / date / etc.) — the string
      `"insufficient_evidence"` goes ONLY inside the
      `evidence_chunk_ids` list, never as the `value`. A noisy commit
      beats leaving the blank unresolved.

    ## State-update rule

    Your job is to RECORD values into state via `update_blank`. A value
    that you inferred mentally but never committed does not count.

    - Every `update_blank` call MUST include `evidence_chunk_ids`.
    - If later evidence contradicts a blank, call `update_blank` again
      with the corrected value.
    - `get_state` is only for contradiction recovery or after a refused
      `finish`.

    ## Finish rule

    - Only the `finish` tool ends the run.
    - Do not call `finish()` until the target blank is resolved.
    - As soon as the target blank is resolved, call `finish()` with the
      same value you stored for the target.
    - If you are stuck and retrieval stays weak, call `update_blank`
      on the target with your best inference as the `value` and
      `["insufficient_evidence"]` as the `evidence_chunk_ids`, then
      call finish. A noisy commit is better than an empty answer.
      Passing `value="insufficient_evidence"` is ALWAYS wrong — it
      means you committed the marker string, not your actual guess.

    ## Tool-error recovery

    If any tool returns `error` or `ok:false`, issue a corrected tool
    call next. Do not explain the failure in prose.

    ## Answer format

    When you call `finish(answer=...)`, the answer should be short and
    direct:
    - Person/entity: just the name.
    - Year/date: just the date value.
    - Number: just the number.
    - Bool: `True` or `False`.
    - List: comma-separated values, no surrounding prose.
    - Never prefix with "Answer:", "Final:", etc.

    ## Tools

    - `search(query, top_k=5)` — retrieve top-k chunks for a query.
    - `read(chunk_id)` — full article text for a returned chunk.
    - `find(chunk_id, pattern)` — locate a literal substring inside a
      chunk's article and return ±200-char snippets around up to 3
      matches. Prefer this over `read` when you only need one fact
      from a long article.
    - `update_blank(blank_id, value, evidence_chunk_ids)` — record a
      blank's resolved value. Rejects unknown `blank_id`.
    - `get_state()` — inspect the current blank-fill state.
    - `finish(answer)` — terminate if the target is resolved.

    ## Worked examples

    ### Example 1 — Normal solve path

    Temporal bridge with blanks `b_year` then `t`.
    - Turn 1: call `search` using the year-identification query.
    - Turn 2: call `update_blank` for `b_year` with the supporting
      chunk id.
    - Turn 3: call `search` using the resolved year plus the target
      entity.
    - Turn 4: call `update_blank` for `t` with the supporting chunk id.
    - Turn 5: call `finish` with the same value stored for `t`.

    ### Example 2 — Sparse-evidence recovery

    Target blank `t` (say it's a person's name); first 2 retrieval
    calls stay weak.
    - Turn 1: call `search` with the most direct query.
    - Turn 2: call `read` or a second `search` only if the first result
      is still ambiguous.
    - Turn 3: call `update_blank` with
          blank_id   = "t"
          value      = "Jane Smith"          ← your best inference
          evidence_chunk_ids = ["insufficient_evidence"]
    - Turn 4: call `finish(answer="Jane Smith")` — same value as in
      step 3.

    Common mistake to avoid:
    - WRONG  ❌  `update_blank(blank_id="t",
                             value="insufficient_evidence",
                             evidence_chunk_ids=[])`
    - RIGHT  ✅  `update_blank(blank_id="t",
                             value="Jane Smith",
                             evidence_chunk_ids=["insufficient_evidence"])`

    The string `"insufficient_evidence"` is a metadata marker inside
    `evidence_chunk_ids`. It is never a valid `value`. If your retrieval
    was weak, the `value` is still your real best-guess (a name, a
    number, a date), and the `evidence_chunk_ids` list carries the
    marker.
    """
).strip()


# ---------------------------------------------------------------------------
# Per-question plan rendering
# ---------------------------------------------------------------------------


def _format_entity(e: dict[str, Any]) -> str:
    """One compact line per entity."""
    parts = [f"`{e.get('id','?')}`"]
    parts.append(f"kind={e.get('kind','?')}")
    if e.get("role"):
        parts.append(f"role={e['role']}")
    if e.get("value_type"):
        parts.append(f"value_type={e['value_type']}")
    if e.get("name"):
        parts.append(f"name={e['name']!r}")
    if e.get("category"):
        parts.append("category=true")
    if e.get("is_target"):
        parts.append("**is_target=true**")
    if e.get("state"):
        parts.append(f"state={e['state']!r}")
    return "- " + "  ".join(parts)


def _format_vp(vp: dict[str, Any]) -> str:
    return (
        f"- `{vp.get('id','?')}`: "
        f"({vp.get('subject_id','?')}) --[{vp.get('phrase','?')}]--> "
        f"({vp.get('object_id','?')})"
    )


def _format_constraint(c: dict[str, Any]) -> str:
    kind = c.get("kind", "?")
    parts = [f"`{c.get('id','?')}`", f"kind={kind}"]
    if c.get("op"):
        parts.append(f"op={c['op']}")
    if c.get("args_blanks"):
        parts.append(f"args_blanks={c['args_blanks']}")
    if c.get("candidate_entity_ids"):
        parts.append(f"candidates={c['candidate_entity_ids']}")
    if c.get("sort_by_blank_ids"):
        parts.append(f"sort_by={c['sort_by_blank_ids']}")
    if c.get("left_ref") or c.get("right_ref"):
        parts.append(f"left={c.get('left_ref')}  right={c.get('right_ref')}")
    parts.append(f"output={c.get('output_blank_id','?')}")
    return "- " + "  ".join(parts)


def render_plan_brief(
    plan_dict: dict[str, Any],
    order: list[str],
    levels: dict[str, int],
    slice_blank_ids: Optional[list[str]] = None,
) -> str:
    """Render the per-question plan + topological-order section.

    Output is plain markdown. Each blank in the topological order is
    annotated with its retrieval signals (incoming VPs) so the reasoner
    knows what to search for.

    When ``slice_blank_ids`` is given, the output is filtered to the
    subset of entities / VPs / constraints / topo-steps relevant to
    that slice — PLUS any filled entities those slice entities refer
    to (so upstream anchors stay visible for retrieval). Blanks that
    are upstream dependencies of slice blanks are rendered as context
    rows in a separate "upstream bridging values" section (no topo
    step for them; they will already be resolved by the time this
    researcher runs).
    """
    ent_by_id: dict[str, dict[str, Any]] = {
        e.get("id"): e for e in plan_dict.get("entities", [])
    }
    entities = plan_dict.get("entities", [])
    vps = plan_dict.get("verb_phrases", [])
    constraints = plan_dict.get("constraints", [])
    constraint_by_output = {
        c.get("output_blank_id"): c for c in constraints if c.get("output_blank_id")
    }

    vp_by_blank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vp in vps:
        for side in ("subject_id", "object_id"):
            eid = vp.get(side)
            if eid in ent_by_id and ent_by_id[eid].get("kind") == "blank":
                vp_by_blank[eid].append(vp)

    # ----- Slice filtering ------------------------------------------
    slice_set: Optional[set[str]] = set(slice_blank_ids) if slice_blank_ids is not None else None
    if slice_set is not None:
        # Entities to keep: the slice blanks themselves, plus every
        # filled entity connected to a slice blank via a VP (these are
        # the retrieval anchors the researcher needs). Upstream blank
        # entities (already resolved by earlier researchers) are
        # dropped from the primary entity list and exposed separately.
        keep_ent_ids: set[str] = set(slice_set)
        kept_vps: list[dict[str, Any]] = []
        for vp in vps:
            sid = vp.get("subject_id")
            oid = vp.get("object_id")
            if sid in slice_set or oid in slice_set:
                kept_vps.append(vp)
                for eid in (sid, oid):
                    if eid in ent_by_id and ent_by_id[eid].get("kind") == "filled":
                        keep_ent_ids.add(eid)
        # Constraints: keep if their output is in slice OR any input is in slice
        kept_constraints: list[dict[str, Any]] = []
        for c in constraints:
            ins = set(c.get("args_blanks") or []) | set(c.get("sort_by_blank_ids") or [])
            if c.get("output_blank_id") in slice_set or (ins & slice_set):
                kept_constraints.append(c)
                # Input blanks of a slice-constraint are upstream — we'll render them below.

        entities = [e for e in entities if e.get("id") in keep_ent_ids]
        vps = kept_vps
        constraints = kept_constraints
        constraint_by_output = {
            c.get("output_blank_id"): c for c in constraints if c.get("output_blank_id")
        }
        # Rebuild vp_by_blank over filtered VPs
        vp_by_blank = defaultdict(list)
        for vp in vps:
            for side in ("subject_id", "object_id"):
                eid = vp.get(side)
                if eid in ent_by_id and ent_by_id[eid].get("kind") == "blank" and eid in slice_set:
                    vp_by_blank[eid].append(vp)
        # Filter topo order to slice blanks (preserving original order)
        order = [b for b in order if b in slice_set]

    out: list[str] = []
    out.append("## The plan for this question")
    out.append("")
    out.append("### Entities")
    filled = [e for e in entities if e.get("kind") == "filled"]
    blank = [e for e in entities if e.get("kind") == "blank"]
    out.append("**Filled** (drawn from the question text):")
    for e in filled:
        out.append(_format_entity(e))
    out.append("")
    out.append("**Blanks** (you must fill these):")
    for e in blank:
        out.append(_format_entity(e))
    out.append("")
    if vps:
        out.append("### Verb-phrases")
        for vp in vps:
            out.append(_format_vp(vp))
        out.append("")
    if constraints:
        out.append("### Constraints")
        for c in constraints:
            out.append(_format_constraint(c))
        out.append("")

    out.append("## Topological solve order")
    out.append("")
    out.append(
        "Fill blanks in this order. Blanks at the same level have no "
        "dependencies on each other and can be resolved in any order."
    )
    out.append("")

    if not order:
        out.append("_(no blanks to solve)_")
        out.append("")
    else:
        # Group by level for readability.
        by_level: dict[int, list[str]] = defaultdict(list)
        for b_id in order:
            by_level[levels.get(b_id, 0)].append(b_id)
        step = 1
        for lvl in sorted(by_level):
            ids = by_level[lvl]
            header = f"### Level {lvl + 1}"
            if len(ids) > 1:
                header += "  _(parallel — blanks here have no mutual dependencies)_"
            out.append(header)
            for b_id in ids:
                ent = ent_by_id.get(b_id, {})
                vt = ent.get("value_type") or "text"
                role = ent.get("role") or "—"
                is_target = " **(TARGET)**" if ent.get("is_target") else ""
                out.append(
                    f"**Step {step}** — `{b_id}` (value_type=`{vt}`, role=`{role}`){is_target}"
                )
                if b_id in constraint_by_output:
                    c = constraint_by_output[b_id]
                    kind = c.get("kind")
                    op = c.get("op")
                    method = f"constraint `{kind}`" + (f" (`{op}`)" if op else "")
                    if kind == "derived":
                        method += f"; inputs: `{c.get('args_blanks', [])}`"
                    elif kind in ("argmax", "argmin"):
                        method += (
                            f"; candidates: `{c.get('candidate_entity_ids', [])}`, "
                            f"ranked by: `{c.get('sort_by_blank_ids', [])}`"
                        )
                    out.append(f"    — method: {method}")
                    out.append(
                        "    — no retrieval; auto-computed by the adapter once "
                        "all input blanks above are resolved"
                    )
                else:
                    signals: list[str] = []
                    for vp in vp_by_blank.get(b_id, []):
                        s, o = vp.get("subject_id"), vp.get("object_id")
                        other = o if s == b_id else s
                        other_ent = ent_by_id.get(other, {})
                        other_name = other_ent.get("name") or other
                        signals.append(f"`({other_name}, {vp.get('phrase')})`")
                    sig_text = ", ".join(signals) if signals else "_(no wired VPs — fall back to role+question keywords)_"
                    out.append(f"    — method: retrieval + LLM extract")
                    out.append(f"    — retrieval signals: {sig_text}")
                step += 1
            out.append("")

    return "\n".join(out).rstrip()


def render_plan_topo_only(
    plan_dict: dict[str, Any],
    order: list[str],
    levels: dict[str, int],
) -> str:
    """Render the plan as ONLY a topological numbered step list.

    This is the stripped-down sibling of :func:`render_plan_brief` —
    used by the topo-only ablation on ``ours_gsw_planner_react_v1``.
    Drops the entities / verb-phrases / constraints structured dump
    and exposes only:

    - One section "## Solve order (topological)" with numbered Steps
      grouped by level (parallel siblings within a level).
    - Per step: blank id, value_type, role, target tag, "depends on
      Step N" annotations (so the dependency graph is visible without
      needing the full graph), and retrieval signals (subject +
      relation pairs derived from the wired VPs).
    - Constraint-output blanks are flagged as auto-computed (do NOT
      dispatch).

    The legend (``SYSTEM_PROMPT_LEGEND``) is still prepended by
    :func:`build_system_prompt`, so the LLM still has the schema
    vocabulary if it needs to interpret the per-step annotations.
    """
    ent_by_id: dict[str, dict[str, Any]] = {
        e.get("id"): e for e in plan_dict.get("entities", [])
    }
    constraint_by_output = {
        c.get("output_blank_id"): c
        for c in plan_dict.get("constraints", [])
        if c.get("output_blank_id")
    }
    vp_by_blank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vp in plan_dict.get("verb_phrases", []):
        for side in ("subject_id", "object_id"):
            eid = vp.get(side)
            if eid in ent_by_id and ent_by_id[eid].get("kind") == "blank":
                vp_by_blank[eid].append(vp)

    # Reverse-dependency map (blank → blanks it depends on). Used to
    # render "depends on Step N" inline so the LLM sees bridging
    # relationships without the structured graph.
    blank_set = {
        e.get("id") for e in plan_dict.get("entities", [])
        if e.get("kind") == "blank"
    }
    depends_on: dict[str, list[str]] = defaultdict(list)
    for c in plan_dict.get("constraints", []):
        out_id = c.get("output_blank_id")
        if not out_id or out_id not in blank_set:
            continue
        for inp_field in ("args_blanks", "sort_by_blank_ids"):
            for inp in c.get(inp_field) or []:
                if inp in blank_set and inp != out_id:
                    depends_on[out_id].append(inp)
    for vp in plan_dict.get("verb_phrases", []):
        s, o = vp.get("subject_id"), vp.get("object_id")
        s_ent, o_ent = ent_by_id.get(s, {}), ent_by_id.get(o, {})
        if (s_ent.get("kind") == "blank"
                and s_ent.get("value_type") == "entity"
                and o_ent.get("kind") == "blank"):
            depends_on[o].append(s)

    out: list[str] = []
    out.append("## Solve order (topological)")
    out.append("")
    out.append(
        "Each step below resolves ONE blank. Steps in the same "
        "level have no mutual dependencies and can be solved in any "
        "order. Use the listed retrieval signals (subject + relation) "
        "to query for evidence."
    )
    out.append("")

    if not order:
        out.append("_(no blanks to solve)_")
        return "\n".join(out).rstrip()

    by_level: dict[int, list[str]] = defaultdict(list)
    for b_id in order:
        by_level[levels.get(b_id, 0)].append(b_id)

    # Map blank_id → step number (1-indexed), so dependency
    # annotations can reference earlier steps.
    step_of: dict[str, int] = {}
    n = 1
    for lvl in sorted(by_level):
        for b_id in by_level[lvl]:
            step_of[b_id] = n
            n += 1

    step = 1
    for lvl in sorted(by_level):
        ids = by_level[lvl]
        header = f"### Level {lvl + 1}"
        if len(ids) > 1:
            header += "  _(parallel)_"
        out.append(header)
        for b_id in ids:
            ent = ent_by_id.get(b_id, {})
            vt = ent.get("value_type") or "text"
            role = ent.get("role") or "—"
            target_tag = " **(TARGET)**" if ent.get("is_target") else ""
            line = (
                f"**Step {step}** — fill `{b_id}` "
                f"(value_type=`{vt}`, role=`{role}`){target_tag}"
            )
            deps = depends_on.get(b_id) or []
            dep_steps = sorted({step_of[d] for d in deps if d in step_of})
            if dep_steps:
                plural = "s" if len(dep_steps) > 1 else ""
                line += (
                    f" — depends on Step{plural} "
                    + ", ".join(str(s) for s in dep_steps)
                )
            out.append(line)

            if b_id in constraint_by_output:
                c = constraint_by_output[b_id]
                kind = c.get("kind")
                op_name = c.get("op")
                method = (
                    f"computed by constraint `{kind}`"
                    + (f" (`{op_name}`)" if op_name else "")
                )
                if kind == "derived":
                    method += f" over `{c.get('args_blanks', [])}`"
                elif kind in ("argmax", "argmin"):
                    method += (
                        f" over candidates `{c.get('candidate_entity_ids', [])}`, "
                        f"ranked by `{c.get('sort_by_blank_ids', [])}`"
                    )
                out.append(f"    — {method} (auto-computed; do NOT dispatch)")
            else:
                signals: list[str] = []
                for vp in vp_by_blank.get(b_id, []):
                    s, o = vp.get("subject_id"), vp.get("object_id")
                    other = o if s == b_id else s
                    other_ent = ent_by_id.get(other, {})
                    other_name = (
                        other_ent.get("name")
                        if other_ent.get("kind") == "filled"
                        else other
                    )
                    signals.append(f"`({other_name}, {vp.get('phrase')})`")
                sig_text = (
                    ", ".join(signals)
                    if signals
                    else "_(no wired VPs — fall back to role + question keywords)_"
                )
                out.append(f"    — retrieval signals: {sig_text}")
            step += 1
        out.append("")

    return "\n".join(out).rstrip()


def build_system_prompt(
    plan_dict: dict[str, Any],
    order: list[str],
    levels: dict[str, int],
    slice_blank_ids: Optional[list[str]] = None,
    *,
    topo_only: bool = False,
) -> str:
    """Assemble the full system prompt for the reasoner.

    When ``topo_only=True``, the plan brief is replaced with a
    stripped-down topological-step list (no entities / VPs /
    constraints structured dump). Used by the topo-only ablation on
    ``ours_gsw_planner_react_v1``.
    """
    if topo_only:
        if slice_blank_ids is not None:
            # Slice-then-topo-only is not currently used. The
            # orchestrator dispatches per-blank slices via the
            # multi-agent path; the flat react adapter never slices.
            # If we need this later, derive a sliced order/levels and
            # call render_plan_topo_only on them.
            raise NotImplementedError(
                "topo_only with slice_blank_ids is not supported"
            )
        brief = render_plan_topo_only(plan_dict, order, levels)
    else:
        brief = render_plan_brief(
            plan_dict, order, levels, slice_blank_ids=slice_blank_ids
        )
    return (
        f"{SYSTEM_PROMPT_LEGEND}\n\n"
        f"{brief}\n\n"
        f"{SYSTEM_PROMPT_RULES}"
    )
