"""Shared visualisation helpers for the planner Streamlit inspectors.

Both `planner_dump_inspector.py` and `pilot_run_inspector.py` need to
render the same kind of GSW-fragment plan — as a Graphviz DOT graph,
as dataframes, and as a narrated solve-step walkthrough. This module
centralises those helpers so they don't drift between the two apps.

Public surface:
  - `compute_topo_info(plan_dict)`     → (order, levels, validation_error)
  - `plan_to_dot(plan_dict, order, levels)` → Graphviz DOT string
  - `entities_df / vps_df / constraints_df` → pandas DataFrames
  - `solve_steps_markdown(plan, order, levels)` → narrative walkthrough
  - `dangling_badge(validation_error)`  → "✓" / "⚠" / "—"
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Make the src/ tree importable even when Streamlit is launched with a
# Python that doesn't have the editable install on its path.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
from pydantic import ValidationError as _PydanticValidationError

from research_agent.adapters.ours._planner_exec import (
    ExecutionError,
    GSWPlan,
    build_dependency_graph,
    topological_sort_blanks,
)


# ---------------------------------------------------------------------------
# Topological info
# ---------------------------------------------------------------------------


def compute_topo_info(
    plan_dict: Optional[dict],
) -> tuple[list[str], dict[str, int], Optional[str]]:
    """Return (order, levels, error).

    - `order`: topological linearisation of all blank entity ids.
    - `levels`: blank_id → dep-level (0 = no blank deps; 1 = depends on some
      level-0 blank; ...). Blanks at the same level can run in parallel.
    - `error`: non-None when the plan fails schema validation (e.g.
      dangling entities) or has a dependency cycle.
    """
    if not plan_dict or not plan_dict.get("entities"):
        return [], {}, None
    try:
        plan_obj = GSWPlan.model_validate(plan_dict)
    except _PydanticValidationError as e:
        # Surface the first message (typically the dangling-entity one).
        msg = str(e.errors()[0]["msg"]) if e.errors() else str(e)
        return [], {}, msg
    try:
        order = topological_sort_blanks(plan_obj)
        deps = build_dependency_graph(plan_obj)
    except ExecutionError as e:
        return [], {}, f"{e.kind}: {e.detail}"

    levels: dict[str, int] = {}
    remaining: dict[str, set[str]] = {b: set(ds) for b, ds in deps.items()}
    level = 0
    while remaining:
        frontier = [b for b, ds in remaining.items() if not ds]
        if not frontier:
            break
        for b in frontier:
            levels[b] = level
            remaining.pop(b)
        for ds in remaining.values():
            ds.difference_update(frontier)
        level += 1
    return order, levels, None


def dangling_badge(validation_error: Optional[str]) -> str:
    if validation_error is None:
        return "✓"
    if "dangling" in (validation_error or "").lower():
        return "⚠ dangling"
    return f"⚠ {validation_error[:40]}"


# ---------------------------------------------------------------------------
# Graphviz DOT
# ---------------------------------------------------------------------------


_VT_COLOUR = {
    "date": "#A7D8F5",
    "number": "#FFCDD2",
    "entity": "#C8E6C9",
    "attribute": "#FFE0B2",
    "list": "#D1C4E9",
    "bool": "#F0F4C3",
    "text": "#ECEFF1",
}

_KIND_COLOUR = {
    "derived": "#6A1B9A",
    "argmax": "#BF360C",
    "argmin": "#BF360C",
    "equals": "#1565C0",
    "in_list": "#1565C0",
    "gt": "#1565C0",
    "lt": "#1565C0",
}


def plan_to_dot(
    plan: Optional[dict],
    order: Optional[list[str]] = None,
    levels: Optional[dict[str, int]] = None,
) -> str:
    """DOT string for `plan`, optionally annotated with topological order.

    When `order` + `levels` are non-empty:
    - Blank labels are prefixed with their 1-based solve step number.
    - `penwidth` grows with dep-level so deeper-in-DAG blanks visually
      differ from root blanks (subtle depth indicator).
    - `{ rank=same; ... }` hints group blanks at the same level so the
      layout reads left-to-right by solve order.
    """
    if not plan or not plan.get("entities"):
        return 'digraph G { label="(no plan)"; }'

    order = order or []
    levels = levels or {}
    step_num = {b_id: i + 1 for i, b_id in enumerate(order)}
    max_level = max(levels.values()) if levels else 0

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        "  node [fontname=Helvetica fontsize=10];",
        "  edge [fontname=Helvetica fontsize=9];",
    ]

    for e in plan["entities"]:
        eid = e.get("id", "")
        kind = e.get("kind", "filled")
        name = (e.get("name") or "").replace('"', r"\"")
        vt = e.get("value_type") or ""
        is_target = e.get("is_target", False)
        category = e.get("category", False)
        role = (e.get("role") or "").replace('"', r"\"")

        if kind == "filled":
            label = f"{eid}\\n{name}" if name else eid
            if role:
                label += f"\\n[{role}]"
            attrs = 'shape=box style="filled" fillcolor="#FFF9C4"'
            if category:
                attrs += ' fontname="Helvetica-Oblique"'
        else:
            step = step_num.get(eid)
            prefix = f"{step}. " if step else ""
            label = f"{prefix}{eid}\\n?{vt or 'blank'}"
            if role:
                label += f"\\n[{role}]"
            fill = _VT_COLOUR.get(vt, "#E1F5FE")
            # Deeper blanks get thicker border as a subtle depth cue.
            pen = 1 + 0.4 * levels.get(eid, 0)
            attrs = (
                f'shape="box" style="filled,rounded" fillcolor="{fill}" '
                f'penwidth={pen:.1f}'
            )
            if is_target:
                attrs += ' color="#2E7D32" fillcolor="#A5D6A7"'
                label += "\\n(TARGET)"
        lines.append(f'  "{eid}" [label="{label}" {attrs}];')

    for vp in plan.get("verb_phrases", []) or []:
        s = vp.get("subject_id", "")
        o = vp.get("object_id", "")
        phrase = (vp.get("phrase") or "").replace('"', r"\"")
        if s and o:
            lines.append(f'  "{s}" -> "{o}" [label="{phrase}"];')

    for c in plan.get("constraints", []) or []:
        out = c.get("output_blank_id")
        if not out:
            continue
        kind = c.get("kind", "")
        op = c.get("op")
        colour = _KIND_COLOUR.get(kind, "#6A1B9A")
        label = kind + (f" ({op})" if op else "")
        if kind == "derived":
            inputs = list(c.get("args_refs", []) or c.get("args_blanks", []) or [])
        elif kind in ("argmax", "argmin"):
            inputs = list(c.get("sort_by_blank_ids", []) or [])
        else:
            inputs = [x for x in (c.get("left_ref"), c.get("right_ref")) if x]
        for inp in inputs:
            lines.append(
                f'  "{inp}" -> "{out}" [label="{label}" '
                f'style=dashed color="{colour}" fontcolor="{colour}"];'
            )

    # Rank hints: blanks at the same level align vertically.
    if levels:
        by_level: dict[int, list[str]] = defaultdict(list)
        for b_id, lvl in levels.items():
            by_level[lvl].append(b_id)
        for lvl, ids in sorted(by_level.items()):
            if len(ids) > 1:
                rank_ids = "; ".join(f'"{i}"' for i in ids)
                lines.append(f"  {{ rank=same; {rank_ids} }}")

    _ = max_level  # kept for future gradient fills
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataFrame helpers (with role column)
# ---------------------------------------------------------------------------


def entities_df(plan: Optional[dict]) -> pd.DataFrame:
    if not plan:
        return pd.DataFrame()
    rows = [
        {
            "id": e.get("id", ""),
            "kind": e.get("kind", ""),
            "role": e.get("role") or "—",
            "name": e.get("name") or "—",
            "value_type": e.get("value_type") or "—",
            "target?": "✓" if e.get("is_target") else "",
            "category?": "✓" if e.get("category") else "",
        }
        for e in plan.get("entities", [])
    ]
    return pd.DataFrame(rows)


def vps_df(plan: Optional[dict]) -> pd.DataFrame:
    if not plan:
        return pd.DataFrame()
    rows = [
        {
            "id": v.get("id", ""),
            "subject_id": v.get("subject_id", ""),
            "phrase": v.get("phrase", ""),
            "object_id": v.get("object_id", ""),
        }
        for v in plan.get("verb_phrases", [])
    ]
    return pd.DataFrame(rows)


def constraints_df(plan: Optional[dict]) -> pd.DataFrame:
    if not plan:
        return pd.DataFrame()
    rows = []
    for c in plan.get("constraints", []):
        kind = c.get("kind", "")
        if kind == "derived":
            inputs = ", ".join(c.get("args_refs") or c.get("args_blanks", []))
        elif kind in ("argmax", "argmin"):
            inputs = (
                f"cands={c.get('candidate_entity_ids', [])} "
                f"sort={c.get('sort_by_blank_ids', [])}"
            )
        else:
            inputs = f"L={c.get('left_ref')} R={c.get('right_ref')}"
        rows.append(
            {
                "id": c.get("id", ""),
                "kind": kind,
                "op": c.get("op") or "—",
                "inputs": inputs,
                "output": c.get("output_blank_id") or "—",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Solve-step narration
# ---------------------------------------------------------------------------


def solve_steps_markdown(
    plan: Optional[dict],
    order: list[str],
    levels: dict[str, int],
) -> str:
    """Render each topological step as a markdown paragraph."""
    if not plan or not order:
        return "_(no topological order — plan missing or invalid)_"

    ent_by_id = {e.get("id"): e for e in plan.get("entities", [])}
    con_by_out = {
        c.get("output_blank_id"): c for c in plan.get("constraints", []) if c.get("output_blank_id")
    }
    vp_by_blank: dict[str, list[dict]] = defaultdict(list)
    for vp in plan.get("verb_phrases", []):
        for side in ("subject_id", "object_id"):
            eid = vp.get(side)
            if eid in ent_by_id and ent_by_id[eid].get("kind") == "blank":
                vp_by_blank[eid].append(vp)

    by_level: dict[int, list[str]] = defaultdict(list)
    for b_id in order:
        by_level[levels.get(b_id, 0)].append(b_id)

    blocks: list[str] = []
    step = 1
    for lvl in sorted(by_level):
        blanks = by_level[lvl]
        header = f"### Level {lvl + 1}"
        if len(blanks) > 1:
            header += "  _(parallel — these blanks have no mutual dependencies)_"
        blocks.append(header)
        for b_id in blanks:
            ent = ent_by_id.get(b_id, {})
            vt = ent.get("value_type") or "text"
            role = ent.get("role") or "—"
            target_tag = "  **(TARGET)**" if ent.get("is_target") else ""
            line = f"**Step {step}** — `{b_id}` (`?{vt}`, role=`{role}`){target_tag}"
            if b_id in con_by_out:
                c = con_by_out[b_id]
                kind = c.get("kind")
                op = c.get("op")
                detail = f"constraint `{kind}`" + (f" (`{op}`)" if op else "")
                if kind == "derived":
                    detail += (
                        f"; inputs: `{c.get('args_refs') or c.get('args_blanks', [])}`"
                    )
                elif kind in ("argmax", "argmin"):
                    detail += (
                        f"; candidates: `{c.get('candidate_entity_ids', [])}`, "
                        f"sort by: `{c.get('sort_by_blank_ids', [])}`"
                    )
                line += f"\n    — method: {detail}"
            else:
                signals = []
                for vp in vp_by_blank.get(b_id, []):
                    s, o = vp.get("subject_id"), vp.get("object_id")
                    other = o if s == b_id else s
                    other_ent = ent_by_id.get(other, {})
                    other_name = other_ent.get("name") or other
                    signals.append(f"`({other_name}, {vp.get('phrase')})`")
                sig_text = ", ".join(signals) if signals else "_(none)_"
                line += f"\n    — method: retrieval + LLM extract\n    — signals: {sig_text}"
            blocks.append(line)
            step += 1
    return "\n\n".join(blocks)
