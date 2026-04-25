"""Streamlit inspector for harness cell_result.json runs.

Browses any ``research_agent/logs/<system>__<model>__<ts>/`` cell written
by ``run_substitution.py`` (works for E1–E13 baselines, `ours_gsw_v1`,
`rule_decomp_gsw`, and `ours_gsw_planner_v1`). Per-question drilldown
shows the trajectory (tool_calls, messages), the emitted `plan_json`
(if planner adapter), executed blanks (if planner adapter), judge
verdict, and any captured `reasoning` / `hidden_reasoning`.

Run with::

    .venv/bin/streamlit run playground/pilot_run_inspector.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from _planner_viz import (
    compute_topo_info,
    constraints_df as _constraints_df,
    dangling_badge,
    entities_df as _entities_df,
    plan_to_dot as _plan_to_dot,
    solve_steps_markdown,
    vps_df as _vps_df,
)


_ROOT = Path(__file__).resolve().parent.parent
_LOGS = _ROOT / "logs"


# ---------------------------------------------------------------------------
# Discovery + load
# ---------------------------------------------------------------------------


@st.cache_data
def _discover_cells() -> list[Path]:
    """Return all `<system>__<model>__<ts>/cell_result.json` under logs/, newest first."""
    if not _LOGS.exists():
        return []
    cells = []
    for d in _LOGS.iterdir():
        if not d.is_dir():
            continue
        cr = d / "cell_result.json"
        if cr.exists():
            cells.append(d)
    cells.sort(key=lambda p: p.name, reverse=True)
    return cells


@st.cache_data
def _load_cell(cell_dir_str: str) -> dict[str, Any]:
    path = Path(cell_dir_str) / "cell_result.json"
    return json.loads(path.read_text())


def _question_by_id(cell: dict, qid: str) -> Optional[dict]:
    for q in cell.get("questions", []):
        if str(q.get("question_id")) == str(qid):
            return q
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _judge_icon(q: dict) -> str:
    if q.get("judge_correct") is True:
        return "✓"
    if q.get("judge_correct") is False:
        return "✗"
    return "—"


def _questions_table(cell: dict) -> pd.DataFrame:
    rows = []
    for q in cell.get("questions", []):
        traj = q.get("trajectory") or {}
        rows.append(
            {
                "id": q.get("question_id", ""),
                "hops": q.get("num_hops", "—"),
                "judge": _judge_icon(q),
                "mode": q.get("failure_mode", "?"),
                "turns": traj.get("turns", 0),
                "wall_s": round(traj.get("wall_time_s", 0), 1),
                "f1": round(q.get("f1", 0.0), 2),
                "pred": (q.get("predicted_answer", "") or "")[:80],
                "gold": (q.get("gold_answer", "") or "")[:60],
                "question": (q.get("question", "") or "")[:80],
            }
        )
    return pd.DataFrame(rows)


def _tool_calls_df(traj: dict) -> pd.DataFrame:
    calls = traj.get("tool_calls", []) or []
    rows = []
    for tc in calls:
        rows.append(
            {
                "turn": tc.get("turn", ""),
                "name": tc.get("name", ""),
                "args": json.dumps(tc.get("args", {}), default=str)[:300],
                "result_preview": (tc.get("result_preview", "") or "")[:300],
                "duration_s": round(tc.get("duration_s", 0), 2),
                "error": tc.get("error", "") or "",
            }
        )
    return pd.DataFrame(rows)


def _executed_blanks_df(traj_extra: dict) -> pd.DataFrame:
    blanks = traj_extra.get("executed_blanks", []) or []
    rows = []
    for b in blanks:
        rows.append(
            {
                "blank_id": b.get("blank_id", ""),
                "status": b.get("status", ""),
                "value": str(b.get("value", ""))[:120],
                "evidence_chunks": ", ".join(b.get("evidence_chunk_ids", []) or []),
                "llm_calls": b.get("llm_calls", 0),
                "wall_s": round(b.get("wall_time_s", 0), 2),
            }
        )
    return pd.DataFrame(rows)


def _failure_histogram(cell: dict) -> pd.DataFrame:
    hist = cell.get("failure_histogram", {}) or {}
    if not hist:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"mode": k, "count": v} for k, v in hist.items()]
    ).sort_values("count", ascending=False)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Planner-v1 run inspector", layout="wide")
st.title("🔬 ours_gsw_planner_v1 — run inspector")

cells = _discover_cells()
if not cells:
    st.error(f"No cell_result.json files found under {_LOGS}.")
    st.stop()

# Restrict to ours_gsw_planner_v1 only (per user request).
cell_meta = []
for d in cells:
    try:
        c = _load_cell(str(d))
        if c.get("system_id") != "ours_gsw_planner_v1":
            continue
        cell_meta.append(
            {
                "dir": str(d),
                "name": d.name,
                "system_id": c.get("system_id", "?"),
                "model_id": c.get("model_id", "?"),
                "n": c.get("n_total", len(c.get("questions", []))),
                "judge": c.get("judge_accuracy", 0) or 0,
                "em": c.get("accuracy", 0) or 0,
            }
        )
    except Exception:  # noqa: BLE001
        continue

if not cell_meta:
    st.error("No `ours_gsw_planner_v1` cells found. Run the planner adapter first.")
    st.stop()

# Secondary filter: model only (system is locked to planner_v1).
models = sorted({m["model_id"] for m in cell_meta})
st.sidebar.markdown("### Filters")
mod_filter = st.sidebar.multiselect("Model", models, default=models)

visible = [m for m in cell_meta if m["model_id"] in mod_filter]

if not visible:
    st.warning("No cells match these filters.")
    st.stop()

cell_labels = [
    f"{m['name']}  ·  {m['system_id']} / {m['model_id'][-25:]}  ·  judge={m['judge']:.2f}  n={m['n']}"
    for m in visible
]
picked_label = st.sidebar.selectbox("Cell", cell_labels, index=0)
picked = visible[cell_labels.index(picked_label)]
cell = _load_cell(picked["dir"])

# Top-of-page cell summary
st.header(f"Cell: {picked['system_id']} / {picked['model_id']}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("n", cell.get("n_total", 0))
c2.metric("judge", f"{cell.get('judge_accuracy', 0):.3f}")
c3.metric("EM", f"{cell.get('accuracy', 0):.3f}")
c4.metric("F1", f"{cell.get('mean_f1', 0):.3f}")
c5.metric("wall/Q", f"{cell.get('mean_wall_time_s', 0):.1f}s")

# Failure histogram
hist_df = _failure_histogram(cell)
if not hist_df.empty:
    with st.expander("Failure histogram", expanded=False):
        st.bar_chart(hist_df.set_index("mode"))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# Per-Q table
st.markdown("---")
st.subheader("Questions")
qs_df = _questions_table(cell)

judge_filter = st.sidebar.multiselect(
    "Judge verdict",
    ["✓", "✗", "—"],
    default=["✓", "✗", "—"],
)
mode_values = sorted({q.get("failure_mode", "?") for q in cell.get("questions", [])})
mode_filter = st.sidebar.multiselect("Failure mode", mode_values, default=mode_values)

filtered_df = qs_df[
    qs_df["judge"].isin(judge_filter) & qs_df["mode"].isin(mode_filter)
].reset_index(drop=True)

st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# Per-Q drilldown
st.markdown("---")
st.subheader("Drilldown")
if filtered_df.empty:
    st.info("No questions match filter; drilldown inactive.")
    st.stop()

qid_options = filtered_df["id"].tolist()

_session_key = f"pilot_idx__{picked['dir']}"
if _session_key not in st.session_state or st.session_state[_session_key] >= len(qid_options):
    st.session_state[_session_key] = 0

prev_col, sel_col, next_col = st.columns([1, 6, 1])
if prev_col.button("⬅ Prev", use_container_width=True, disabled=st.session_state[_session_key] == 0):
    st.session_state[_session_key] = max(0, st.session_state[_session_key] - 1)
    st.rerun()
if next_col.button(
    "Next ➡",
    use_container_width=True,
    disabled=st.session_state[_session_key] >= len(qid_options) - 1,
):
    st.session_state[_session_key] = min(len(qid_options) - 1, st.session_state[_session_key] + 1)
    st.rerun()

with sel_col:
    qid = st.selectbox(
        "Question",
        options=qid_options,
        index=st.session_state[_session_key],
        format_func=lambda x: f"q{x} · {filtered_df.loc[filtered_df['id'] == x, 'question'].iloc[0][:80]}",
        label_visibility="collapsed",
        key=f"{_session_key}_select",
    )
st.session_state[_session_key] = qid_options.index(qid)
st.caption(f"Question {st.session_state[_session_key] + 1} / {len(qid_options)}")

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
    f"— **judge**: {_judge_icon(q)}  · **mode**: `{q.get('failure_mode', '?')}`"
)
st.markdown(
    f"**Turns**: {traj.get('turns', 0)}  · **Wall**: "
    f"{traj.get('wall_time_s', 0):.1f}s  · **Tokens**: "
    f"in={traj.get('prompt_tokens', 0)} out={traj.get('completion_tokens', 0)}"
)
if extra.get("fallback_flag"):
    st.warning(
        f"**Fallback fired** — reason: `{extra.get('fallback_reason', '?')}`"
    )

# Tabs
tab_names = ["Trajectory", "Plan (if planner)", "Executed blanks", "Messages", "Reasoning"]
if q.get("judge_reason"):
    tab_names.append("Judge reason")
tabs = st.tabs(tab_names)

with tabs[0]:
    st.write(f"**{len(traj.get('tool_calls') or [])} tool calls**")
    tc_df = _tool_calls_df(traj)
    if not tc_df.empty:
        st.dataframe(tc_df, use_container_width=True, hide_index=True)
        # Per-call full result expanders.
        st.caption("Click a call to see its full result_full:")
        for i, tc in enumerate(traj.get("tool_calls", []) or []):
            with st.expander(
                f"turn {tc.get('turn')} · {tc.get('name')} · {str(tc.get('args'))[:80]}",
                expanded=False,
            ):
                st.code(tc.get("result_full", tc.get("result_preview", "")) or "(empty)")
                if tc.get("error"):
                    st.error(f"error: {tc['error']}")
    else:
        st.info("No tool calls recorded.")

with tabs[1]:
    plan_json = extra.get("plan_json")
    if plan_json:
        order, levels, topo_error = compute_topo_info(plan_json)
        st.markdown(f"**Graph validation**: {dangling_badge(topo_error)}")
        if topo_error:
            st.warning(f"Plan failed schema/topo validation — {topo_error}")
        with st.expander("📊 Visual graph (topological)", expanded=True):
            st.graphviz_chart(
                _plan_to_dot(plan_json, order=order, levels=levels),
                use_container_width=True,
            )
        with st.expander("🪜 Solve steps (topological order)", expanded=False):
            st.markdown(solve_steps_markdown(plan_json, order, levels))
        c_left, c_mid, c_right = st.columns(3)
        with c_left:
            st.subheader("Entities")
            st.dataframe(_entities_df(plan_json), use_container_width=True, hide_index=True)
        with c_mid:
            st.subheader("VerbPhrases")
            st.dataframe(_vps_df(plan_json), use_container_width=True, hide_index=True)
        with c_right:
            st.subheader("Constraints")
            st.dataframe(_constraints_df(plan_json), use_container_width=True, hide_index=True)
        with st.expander("Raw plan JSON"):
            st.code(json.dumps(plan_json, indent=2, default=str))
    else:
        st.info("No plan_json in trajectory (adapter not a planner, or fallback fired).")

with tabs[2]:
    plan_json = extra.get("plan_json")
    eb_df = _executed_blanks_df(extra)
    if eb_df.empty:
        st.info("No executed_blanks recorded (adapter not a planner, or fallback fired).")
    elif plan_json:
        # Side-by-side "planned vs actual" walkthrough: align the topo order
        # with what the executor actually produced.
        order, levels, _err = compute_topo_info(plan_json)
        ent_by_id = {e.get("id"): e for e in plan_json.get("entities", [])}
        exec_by_id = {
            b.get("blank_id"): b
            for b in (extra.get("executed_blanks") or [])
            if b.get("blank_id")
        }
        rows = []
        for step, b_id in enumerate(order, start=1):
            ent = ent_by_id.get(b_id, {})
            executed = exec_by_id.get(b_id, {})
            rows.append(
                {
                    "step": step,
                    "blank_id": b_id,
                    "planned role": ent.get("role") or "—",
                    "value_type": ent.get("value_type") or "—",
                    "target?": "✓" if ent.get("is_target") else "",
                    "actual status": executed.get("status", "(not run)"),
                    "actual value": str(executed.get("value", ""))[:120],
                    "evidence chunks": ", ".join(
                        executed.get("evidence_chunk_ids", []) or []
                    ),
                    "wall_s": round(executed.get("wall_time_s", 0), 2),
                }
            )
        if rows:
            st.caption("Planned (topological order) vs actual (executor output):")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Raw executed_blanks (no topological alignment possible):")
            st.dataframe(eb_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(eb_df, use_container_width=True, hide_index=True)

with tabs[3]:
    msgs = traj.get("messages") or []
    if not msgs:
        st.info("No full-message log captured for this trajectory.")
    else:
        for i, m in enumerate(msgs):
            with st.expander(f"[{i}] {m.get('role','?')} ({len(str(m.get('content','')))} chars)"):
                content = m.get("content")
                if isinstance(content, str):
                    st.code(content)
                else:
                    st.code(json.dumps(content, indent=2, default=str))
                if m.get("tool_calls"):
                    st.caption("tool_calls:")
                    st.code(json.dumps(m["tool_calls"], indent=2, default=str))

with tabs[4]:
    hidden = traj.get("hidden_reasoning") or ""
    summary = traj.get("reasoning") or ""
    if hidden.strip() or summary.strip():
        if summary.strip():
            st.markdown("**Adapter reasoning summary**:")
            st.code(summary)
        if hidden.strip():
            st.markdown("**Hidden reasoning (raw)**:")
            st.code(hidden)
    else:
        st.info("(No reasoning captured for this run.)")

if q.get("judge_reason"):
    with tabs[-1]:
        st.code(q["judge_reason"])
