"""Streamlit inspector for `ours_gsw_planner_react_v1` cells.

Points at any `research_agent/logs/ours_gsw_planner_react_v1__*/` cell
written by `run_substitution.py`. Specialised views:

- **Cell summary**: judge/EM/F1, stopped-reason histogram, finish-rejection
  rate, target-filled rate, blank-update stats, mean search-only turns
  before first commit.
- **Per-Q drilldown**: plan graph (via `_planner_viz`), topological solve
  order, tool-call timeline colour-coded by tool name, executed-blanks
  table with role + status + value + evidence, full messages tab,
  reasoning tab.

Run with::

    .venv/bin/streamlit run playground/planner_react_run_inspector.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

# Make src importable regardless of how Streamlit is launched.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _planner_viz import (
    compute_topo_info,
    constraints_df as _constraints_df,
    dangling_badge,
    entities_df as _entities_df,
    plan_to_dot as _plan_to_dot,
    solve_steps_markdown,
    vps_df as _vps_df,
)


_LOGS = _ROOT / "logs"
_TARGET_SYSTEM_ID = "ours_gsw_planner_react_v1"


# ---------------------------------------------------------------------------
# Discovery + load
# ---------------------------------------------------------------------------


@st.cache_data
def _discover_cells() -> list[Path]:
    if not _LOGS.exists():
        return []
    cells: list[Path] = []
    for d in _LOGS.iterdir():
        if not d.is_dir():
            continue
        cr = d / "cell_result.json"
        if cr.exists():
            try:
                meta = json.loads(cr.read_text())
            except Exception:  # noqa: BLE001
                continue
            if meta.get("system_id") == _TARGET_SYSTEM_ID:
                cells.append(d)
    cells.sort(key=lambda p: p.name, reverse=True)
    return cells


@st.cache_data
def _load_cell(cell_dir: str) -> dict[str, Any]:
    return json.loads((Path(cell_dir) / "cell_result.json").read_text())


def _question_by_id(cell: dict, qid: str) -> Optional[dict]:
    for q in cell.get("questions", []):
        if str(q.get("question_id")) == str(qid):
            return q
    return None


# ---------------------------------------------------------------------------
# Metric aggregation (cell-level)
# ---------------------------------------------------------------------------


def _cell_metrics(cell: dict) -> dict[str, Any]:
    qs = cell.get("questions", [])
    n = len(qs)
    if n == 0:
        return {}
    reasons = Counter()
    finish_rejections: list[int] = []
    updates: list[int] = []
    target_resolved = 0
    fallback_fired = 0
    search_only_prefix: list[int] = []
    turns: list[int] = []
    walls: list[float] = []
    empty_answers = 0

    for q in qs:
        traj = q.get("trajectory") or {}
        extra = traj.get("extra") or {}
        reasons[extra.get("stopped_reason", "?")] += 1
        finish_rejections.append(int(extra.get("finish_rejections", 0) or 0))
        updates.append(int(extra.get("blank_updates", 0) or 0))
        if extra.get("fallback_flag"):
            fallback_fired += 1
        # Target resolved?
        for b in extra.get("executed_blanks", []) or []:
            if b.get("blank_id") and _is_target(extra, b["blank_id"]):
                if b.get("status") == "resolved":
                    target_resolved += 1
                break
        # Search-only prefix length.
        tool_seq = [tc.get("name", "") for tc in traj.get("tool_calls") or []]
        prefix = 0
        for name in tool_seq:
            if name in ("search", "read"):
                prefix += 1
            else:
                break
        search_only_prefix.append(prefix)
        turns.append(int(traj.get("turns", 0) or 0))
        walls.append(float(traj.get("wall_time_s", 0) or 0))
        if not (q.get("predicted_answer") or "").strip():
            empty_answers += 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        "n": n,
        "stopped_reasons": dict(reasons),
        "target_resolved_pct": round(100 * target_resolved / n, 1),
        "fallback_pct": round(100 * fallback_fired / n, 1),
        "empty_answer_pct": round(100 * empty_answers / n, 1),
        "mean_finish_rejections": mean(finish_rejections),
        "mean_updates_per_q": mean(updates),
        "mean_search_only_prefix": mean(search_only_prefix),
        "mean_turns": mean(turns),
        "mean_wall_s": mean(walls),
    }


def _is_target(extra: dict, blank_id: str) -> bool:
    plan = extra.get("plan_json") or {}
    for e in plan.get("entities", []):
        if e.get("id") == blank_id and e.get("is_target"):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-Q helpers
# ---------------------------------------------------------------------------


def _tool_palette(name: str) -> str:
    return {
        "search": "#1976D2",         # blue
        "read": "#7B1FA2",           # purple
        "update_blank": "#2E7D32",   # green
        "get_state": "#616161",      # grey
        "finish": "#E65100",         # orange
    }.get(name, "#455A64")


def _tool_timeline_df(traj: dict) -> pd.DataFrame:
    rows = []
    for i, tc in enumerate(traj.get("tool_calls") or [], start=1):
        name = tc.get("name", "?")
        args = tc.get("args", {}) or {}
        # Short summary: first useful arg.
        if name == "search":
            summary = f"query={args.get('query','')!r}"
        elif name == "read":
            summary = f"chunk={args.get('chunk_id','')!r}"
        elif name == "update_blank":
            summary = f"blank={args.get('blank_id','')!r}  value={str(args.get('value',''))[:40]!r}"
        elif name == "finish":
            summary = f"answer={str(args.get('answer',''))[:50]!r}"
        else:
            summary = json.dumps(args, default=str)[:80]
        rows.append(
            {
                "#": i,
                "turn": tc.get("turn", "?"),
                "tool": name,
                "args_summary": summary,
                "duration_s": round(tc.get("duration_s", 0), 3),
                "error": tc.get("error", "") or "",
            }
        )
    return pd.DataFrame(rows)


def _executed_blanks_df(extra: dict) -> pd.DataFrame:
    plan = extra.get("plan_json") or {}
    role_by_id = {e.get("id"): e.get("role") for e in plan.get("entities", [])}
    target_by_id = {e.get("id"): bool(e.get("is_target")) for e in plan.get("entities", [])}
    vt_by_id = {e.get("id"): e.get("value_type") for e in plan.get("entities", [])}
    rows = []
    for b in extra.get("executed_blanks", []) or []:
        bid = b.get("blank_id", "?")
        rows.append(
            {
                "blank_id": bid,
                "role": role_by_id.get(bid) or "—",
                "value_type": vt_by_id.get(bid) or "—",
                "target?": "✓" if target_by_id.get(bid) else "",
                "status": b.get("status"),
                "value": str(b.get("value", ""))[:120],
                "evidence": ", ".join(b.get("evidence_chunk_ids", []) or []),
            }
        )
    return pd.DataFrame(rows)


def _judge_icon(q: dict) -> str:
    if q.get("judge_correct") is True:
        return "✓"
    if q.get("judge_correct") is False:
        return "✗"
    return "—"


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Planner-React run inspector", layout="wide")
st.title("🧭 ours_gsw_planner_react_v1 — run inspector")

cells = _discover_cells()
if not cells:
    st.error(f"No `{_TARGET_SYSTEM_ID}` cells found under {_LOGS}.")
    st.stop()

# Sidebar: cell picker
cell_meta = []
for d in cells:
    try:
        c = _load_cell(str(d))
        cell_meta.append(
            {
                "dir": str(d),
                "name": d.name,
                "model": c.get("model_id", "?"),
                "n": c.get("n_total", len(c.get("questions", []))),
                "judge": c.get("judge_accuracy", 0) or 0,
                "em": c.get("accuracy", 0) or 0,
            }
        )
    except Exception:  # noqa: BLE001
        continue

labels = [
    f"{m['name']}  ·  {m['model'][-30:]}  ·  judge={m['judge']:.2f}  n={m['n']}"
    for m in cell_meta
]
picked_label = st.sidebar.selectbox("Cell", labels, index=0)
picked = cell_meta[labels.index(picked_label)]
cell = _load_cell(picked["dir"])

# Cell summary header
st.header(f"Cell: {picked['model']}")
metrics = _cell_metrics(cell)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("n", cell.get("n_total", 0))
c2.metric("judge", f"{cell.get('judge_accuracy', 0):.3f}")
c3.metric("EM", f"{cell.get('accuracy', 0):.3f}")
c4.metric("F1", f"{cell.get('mean_f1', 0):.3f}")
c5.metric("wall/Q", f"{metrics.get('mean_wall_s', 0)}s")

# Row 2 — react-specific
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("target-filled %", f"{metrics.get('target_resolved_pct', 0)}%")
c2.metric("empty-answer %", f"{metrics.get('empty_answer_pct', 0)}%")
c3.metric("finish-reject/Q", metrics.get("mean_finish_rejections", 0))
c4.metric("updates/Q", metrics.get("mean_updates_per_q", 0))
c5.metric("search-only prefix/Q", metrics.get("mean_search_only_prefix", 0))

# Stopped-reason histogram
reasons_df = pd.DataFrame(
    [{"reason": k, "count": v} for k, v in (metrics.get("stopped_reasons") or {}).items()]
).sort_values("count", ascending=False)
if not reasons_df.empty:
    with st.expander("Stopped-reason histogram", expanded=False):
        st.bar_chart(reasons_df.set_index("reason"))
        st.dataframe(reasons_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Per-Q table
st.subheader("Questions")
rows = []
for q in cell.get("questions", []):
    traj = q.get("trajectory") or {}
    extra = traj.get("extra") or {}
    rows.append(
        {
            "id": q.get("question_id"),
            "hops": q.get("num_hops", "—"),
            "judge": _judge_icon(q),
            "stopped": extra.get("stopped_reason", "?"),
            "turns": traj.get("turns", 0),
            "updates": int(extra.get("blank_updates", 0) or 0),
            "fin_rej": int(extra.get("finish_rejections", 0) or 0),
            "wall_s": round(traj.get("wall_time_s", 0), 1),
            "pred": (q.get("predicted_answer", "") or "")[:80],
            "gold": (q.get("gold_answer", "") or "")[:60],
            "question": (q.get("question", "") or "")[:80],
        }
    )
qs_df = pd.DataFrame(rows)

# Sidebar filters on the questions table.
judge_filter = st.sidebar.multiselect(
    "Judge verdict", ["✓", "✗", "—"], default=["✓", "✗", "—"]
)
stop_values = sorted({r["stopped"] for r in rows if r["stopped"]})
stop_filter = st.sidebar.multiselect("Stopped reason", stop_values, default=stop_values)

filt = qs_df[qs_df["judge"].isin(judge_filter) & qs_df["stopped"].isin(stop_filter)].reset_index(drop=True)
st.dataframe(filt, use_container_width=True, hide_index=True)

if filt.empty:
    st.info("No questions match current filters.")
    st.stop()

# Drilldown
st.markdown("---")
st.subheader("Drilldown")

qid_opts = filt["id"].tolist()
_session_key = f"planner_react_idx__{picked['dir']}"
if _session_key not in st.session_state or st.session_state[_session_key] >= len(qid_opts):
    st.session_state[_session_key] = 0

prev_col, sel_col, next_col = st.columns([1, 6, 1])
if prev_col.button("⬅ Prev", use_container_width=True, disabled=st.session_state[_session_key] == 0):
    st.session_state[_session_key] = max(0, st.session_state[_session_key] - 1)
    st.rerun()
if next_col.button(
    "Next ➡",
    use_container_width=True,
    disabled=st.session_state[_session_key] >= len(qid_opts) - 1,
):
    st.session_state[_session_key] = min(len(qid_opts) - 1, st.session_state[_session_key] + 1)
    st.rerun()

with sel_col:
    qid = st.selectbox(
        "Question",
        options=qid_opts,
        index=st.session_state[_session_key],
        format_func=lambda x: f"q{x} · {filt.loc[filt['id']==x, 'question'].iloc[0][:80]}",
        label_visibility="collapsed",
    )
st.session_state[_session_key] = qid_opts.index(qid)
st.caption(f"Question {st.session_state[_session_key] + 1} / {len(qid_opts)}")

q = _question_by_id(cell, qid)
if not q:
    st.error("Question not found.")
    st.stop()

traj = q.get("trajectory") or {}
extra = traj.get("extra") or {}

st.markdown(f"**Question**: {q.get('question', '')}")
st.markdown(f"**Gold answer**: {q.get('gold_answer', '')}")
st.markdown(
    f"**Predicted**: `{(q.get('predicted_answer', '') or '(empty)')[:500]}`  "
    f"— **judge**: {_judge_icon(q)}  ·  **stopped**: `{extra.get('stopped_reason','?')}`"
)
st.markdown(
    f"**Turns**: {traj.get('turns', 0)}  ·  **Finish rejections**: "
    f"{extra.get('finish_rejections', 0)}  ·  **Blank updates**: "
    f"{extra.get('blank_updates', 0)}"
)
if extra.get("fallback_flag"):
    st.warning(f"**Fallback fired** — reason: `{extra.get('fallback_reason','?')}`")

tabs = st.tabs(["Plan", "Tool timeline", "Executed blanks", "Messages", "Reasoning"])

# --- Tab: Plan ---------------------------------------------------------
with tabs[0]:
    plan_json = extra.get("plan_json")
    if not plan_json:
        st.info("No plan_json in trajectory.")
    else:
        order, levels, topo_error = compute_topo_info(plan_json)
        st.markdown(f"**Graph validation**: {dangling_badge(topo_error)}")
        if topo_error:
            st.warning(f"Topology error: {topo_error}")
        with st.expander("📊 Visual graph (topological)", expanded=True):
            st.graphviz_chart(
                _plan_to_dot(plan_json, order=order, levels=levels),
                use_container_width=True,
            )
        with st.expander("🪜 Solve steps", expanded=False):
            st.markdown(solve_steps_markdown(plan_json, order, levels))
        cL, cM, cR = st.columns(3)
        cL.subheader("Entities")
        cL.dataframe(_entities_df(plan_json), use_container_width=True, hide_index=True)
        cM.subheader("VerbPhrases")
        cM.dataframe(_vps_df(plan_json), use_container_width=True, hide_index=True)
        cR.subheader("Constraints")
        cR.dataframe(_constraints_df(plan_json), use_container_width=True, hide_index=True)
        with st.expander("Raw plan JSON"):
            st.code(json.dumps(plan_json, indent=2, default=str))

# --- Tab: Tool timeline -----------------------------------------------
with tabs[1]:
    tl = _tool_timeline_df(traj)
    if tl.empty:
        st.info("No tool calls recorded.")
    else:
        # Colour chip rendering via Styler.
        def _style_tool(val):
            return f"background-color: {_tool_palette(val)}; color: white; font-weight: bold; text-align: center;"

        styled = tl.style.map(_style_tool, subset=["tool"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        # Per-call full result expanders.
        st.caption("Click to inspect each tool call's full result:")
        for i, tc in enumerate(traj.get("tool_calls") or [], start=1):
            with st.expander(
                f"#{i} · turn {tc.get('turn')} · {tc.get('name')} · "
                f"{json.dumps(tc.get('args', {}), default=str)[:80]}",
                expanded=False,
            ):
                st.code(tc.get("result_full") or tc.get("result_preview") or "(empty)")
                if tc.get("error"):
                    st.error(f"error: {tc['error']}")

# --- Tab: Executed blanks ---------------------------------------------
with tabs[2]:
    eb = _executed_blanks_df(extra)
    if eb.empty:
        st.info("No executed_blanks (adapter didn't emit state, or fallback fired).")
    else:
        st.dataframe(eb, use_container_width=True, hide_index=True)

# --- Tab: Messages -----------------------------------------------------
with tabs[3]:
    msgs = traj.get("messages") or []
    if not msgs:
        st.info("No full-message log captured.")
    else:
        for i, m in enumerate(msgs):
            role = m.get("role", "?")
            content = m.get("content")
            preview_len = len(str(content)) if content else 0
            hdr = f"[{i}] {role} ({preview_len} chars)"
            if m.get("tool_calls"):
                hdr += f"  +{len(m['tool_calls'])} tool_calls"
            with st.expander(hdr):
                if isinstance(content, str):
                    st.code(content)
                else:
                    st.code(json.dumps(content, indent=2, default=str))
                if m.get("tool_calls"):
                    st.caption("tool_calls:")
                    st.code(json.dumps(m["tool_calls"], indent=2, default=str))

# --- Tab: Reasoning ---------------------------------------------------
with tabs[4]:
    hidden = traj.get("hidden_reasoning") or ""
    summary = traj.get("reasoning") or ""
    if not (hidden.strip() or summary.strip()):
        st.info("(No reasoning captured.)")
    else:
        if summary.strip():
            st.markdown("**Adapter reasoning summary**")
            st.code(summary)
        if hidden.strip():
            st.markdown("**Hidden reasoning (per-turn, concatenated)**")
            st.code(hidden)
