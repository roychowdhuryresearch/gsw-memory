"""Streamlit inspector for ``ours_gsw_planner_orchestrator_v1`` cells.

Complementary to ``planner_react_run_inspector.py``. Specialised for the
hierarchical orchestrator adapter:

- **Cell summary**: judge/EM/F1, stopped-reason histogram, target-filled
  rate, empty-pred rate, mean level count per Q, mean turns per level,
  mean wall per Q.
- **Per-Q drilldown**:
    • Plan tab (same as the flat inspector — shows full plan graph).
    • **Levels tab** (new) — one card per researcher: allowed blanks,
      stopped-reason, resolved / unresolved, turns, tokens, per-level
      tool timeline, per-level messages, per-level system prompt.
    • Executed blanks tab (final state across all levels).
    • Messages tab (all messages across levels, grouped by ``_level``).
    • Reasoning tab (hidden_reasoning already comes level-split).

Run with::

    .venv/bin/streamlit run playground/planner_orchestrator_run_inspector.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

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
_TARGET_SYSTEM_ID = "ours_gsw_planner_orchestrator_v1"


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
# Metric aggregation
# ---------------------------------------------------------------------------


def _cell_metrics(cell: dict) -> dict[str, Any]:
    qs = cell.get("questions", [])
    n = len(qs)
    if n == 0:
        return {}
    reasons: Counter = Counter()
    level_counts: list[int] = []
    empty_answers = 0
    turns_per_q: list[int] = []
    walls: list[float] = []
    fallback_fired = 0
    researcher_stops: Counter = Counter()
    all_tokens_in: list[int] = []
    all_tokens_out: list[int] = []

    for q in qs:
        traj = q.get("trajectory") or {}
        extra = traj.get("extra") or {}
        reasons[extra.get("stopped_reason", "?")] += 1
        rts = extra.get("researcher_traces", []) or []
        level_counts.append(len(rts))
        for rt in rts:
            researcher_stops[rt.get("stopped", "?")] += 1
        if extra.get("fallback_flag"):
            fallback_fired += 1
        turns_per_q.append(int(traj.get("turns", 0) or 0))
        walls.append(float(traj.get("wall_time_s", 0) or 0))
        all_tokens_in.append(int(traj.get("prompt_tokens", 0) or 0))
        all_tokens_out.append(int(traj.get("completion_tokens", 0) or 0))
        if not (q.get("predicted_answer") or "").strip():
            empty_answers += 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        "n": n,
        "stopped_reasons": dict(reasons),
        "researcher_stops": dict(researcher_stops),
        "mean_levels_per_q": mean(level_counts),
        "empty_answer_pct": round(100 * empty_answers / n, 1),
        "fallback_pct": round(100 * fallback_fired / n, 1),
        "mean_turns": mean(turns_per_q),
        "mean_wall_s": mean(walls),
        "mean_prompt_tokens": mean(all_tokens_in),
        "mean_completion_tokens": mean(all_tokens_out),
    }


# ---------------------------------------------------------------------------
# Per-Q helpers
# ---------------------------------------------------------------------------


def _parse_reasoning(hidden: str) -> dict:
    """Parse ``traj.hidden_reasoning`` into ``{(level, turn): text}``.

    Level -1 = orchestrator (llm mode); N>=0 = per-level / per-dispatch
    researcher. Turn numbers are per-section, 1-based.

    Tolerates both formats:
      * deterministic:  ``=== level 0 ===\\n[turn 1] ...``
      * llm-orch mix:   ``[orchestrator turn 1] ...\\n=== dispatch 0 ===\\n[turn 1] ...``
    """
    import re

    out: dict = {}
    if not hidden:
        return out
    section = -2  # -2 = "unknown / pre-header"; set properly as we see markers
    # Start buffer for the current (section, turn) entry.
    cur_section = None
    cur_turn = None
    buf: list[str] = []

    def _flush():
        nonlocal buf, cur_section, cur_turn
        if cur_section is not None and cur_turn is not None and buf:
            key = (cur_section, cur_turn)
            out.setdefault(key, "")
            out[key] = (out[key] + "\n" + "\n".join(buf)).strip()
        buf = []

    sect_re = re.compile(r"^===\s*(?:level|dispatch)\s+(\d+)\s*===\s*$")
    turn_re = re.compile(r"^\[(?:orchestrator\s+)?turn\s+(\d+)\]\s*(.*)$")

    for line in hidden.splitlines():
        m_sect = sect_re.match(line)
        if m_sect:
            _flush()
            section = int(m_sect.group(1))
            cur_section = section
            cur_turn = None
            continue
        m_turn = turn_re.match(line)
        if m_turn:
            _flush()
            is_orch = "orchestrator" in line
            cur_section = -1 if is_orch else (section if section >= 0 else 0)
            cur_turn = int(m_turn.group(1))
            tail = m_turn.group(2).strip()
            if tail:
                buf.append(tail)
            continue
        # Plain line — append to current buffer
        buf.append(line)
    _flush()
    return out


def _tool_palette(name: str) -> str:
    return {
        # Researcher tools
        "search": "#1976D2",
        "read": "#7B1FA2",
        "find": "#00838F",
        "update_blank": "#2E7D32",
        "get_state": "#616161",
        "suggest_plan_revision": "#FF6F00",  # amber-strong (escalation)
        # Orchestrator tools (llm mode)
        "dispatch_subplan": "#C62828",       # red-strong
        "request_plan_update": "#AD1457",    # pink-strong
        "submit_answer": "#EF6C00",          # orange-strong
    }.get(name, "#455A64")


def _judge_icon(q: dict) -> str:
    if q.get("judge_correct") is True:
        return "✓"
    if q.get("judge_correct") is False:
        return "✗"
    return "—"


def _tool_timeline_df(tool_calls: list[dict], level: Optional[int] = None) -> pd.DataFrame:
    """Build a timeline for a specific level (or all if level is None)."""
    rows = []
    for i, tc in enumerate(tool_calls, start=1):
        if level is not None and tc.get("level") != level:
            continue
        name = tc.get("name", "?")
        args = tc.get("args", {}) or {}
        if name == "search":
            summary = f"query={str(args.get('query',''))[:80]!r}"
        elif name == "read":
            summary = f"chunk={args.get('chunk_id','')!r}"
        elif name == "find":
            summary = f"chunk={args.get('chunk_id','')!r}  pattern={str(args.get('pattern',''))[:40]!r}"
        elif name == "update_blank":
            summary = (
                f"blank={args.get('blank_id','')!r}  "
                f"value={str(args.get('value',''))[:40]!r}  "
                f"ev={str(args.get('evidence_chunk_ids',[]))[:40]}"
            )
        else:
            summary = json.dumps(args, default=str)[:80]
        rows.append(
            {
                "#": i,
                "turn": tc.get("turn", "?"),
                "level": tc.get("level", "—"),
                "tool": name,
                "args_summary": summary,
                "duration_s": round(tc.get("duration_s", 0), 3),
                "error": tc.get("error", "") or "",
            }
        )
    return pd.DataFrame(rows)


def _question_tags(q: dict) -> list[str]:
    """Derived filters for common orchestrator failure/recovery paths."""
    traj = q.get("trajectory") or {}
    extra = traj.get("extra") or {}
    tool_calls = traj.get("tool_calls") or []
    plan_updates = extra.get("plan_updates") or []
    tags: set[str] = set()

    if q.get("judge_correct") is True:
        tags.add("judge_correct")
    elif q.get("judge_correct") is False:
        tags.add("judge_wrong")
    if not (q.get("predicted_answer") or "").strip():
        tags.add("empty_pred")

    if extra.get("fallback_flag"):
        tags.add("fallback")
    if extra.get("planner_failed") or extra.get("plan_emit_error"):
        tags.add("planner_failed")
    if extra.get("topology_error"):
        tags.add("topology_error")

    if plan_updates:
        tags.add("plan_update_used")
    if any(pu.get("rejected") for pu in plan_updates):
        tags.add("plan_update_rejected")
    if any(
        pu.get("rejected")
        and "max plan updates" in str(pu.get("error", "")).lower()
        for pu in plan_updates
    ):
        tags.add("plan_update_cap")

    verdicts = extra.get("validator_verdicts") or []
    if verdicts:
        tags.add("validator_used")
    if any(v.get("verdict") == "wrong" for v in verdicts):
        tags.add("validator_rejected")
        if q.get("judge_correct") is True:
            tags.add("validator_rejected_recovered")
        elif q.get("judge_correct") is False:
            tags.add("validator_rejected_failed")
    if extra.get("validator_repairs_used"):
        tags.add("validator_repaired")

    dispatches = extra.get("dispatches") or []
    if any(d.get("revision_requests") for d in dispatches):
        tags.add("researcher_requested_revision")
    if any(
        "per-blank dispatch cap" in str(tc.get("error", "")).lower()
        for tc in tool_calls
    ):
        tags.add("per_blank_dispatch_cap")
    if any(
        tc.get("name") == "submit_answer" and tc.get("error")
        for tc in tool_calls
    ):
        tags.add("submit_rejected")
    if any(tc.get("error") for tc in tool_calls):
        tags.add("tool_error")

    stopped = extra.get("stopped_reason")
    if stopped:
        tags.add(f"stopped:{stopped}")
    return sorted(tags)


def _issue_type(q: dict) -> str:
    """Small, user-facing diagnostic bucket for sidebar filtering."""
    tags = set(_question_tags(q))
    if "planner_failed" in tags:
        return "Planner failed"
    if "fallback" in tags:
        return "Fallback"
    if "topology_error" in tags:
        return "Topology error"
    if "validator_rejected_failed" in tags:
        return "Validator rejected → failed"
    if "validator_rejected_recovered" in tags:
        return "Validator rejected → recovered"
    if "validator_repaired" in tags:
        return "Validator repaired"
    if "plan_update_cap" in tags:
        return "Plan-update cap"
    if "plan_update_rejected" in tags:
        return "Plan update rejected"
    if "per_blank_dispatch_cap" in tags:
        return "Per-blank dispatch cap"
    if "submit_rejected" in tags:
        return "Submit rejected"
    if "empty_pred" in tags:
        return "Empty prediction"
    if "tool_error" in tags:
        return "Tool error"
    if "plan_update_used" in tags:
        return "Plan update used"
    if q.get("judge_correct") is True:
        return "Correct"
    if q.get("judge_correct") is False:
        return "Wrong answer"
    return "Other"


def _executed_blanks_df(extra: dict) -> pd.DataFrame:
    plan = extra.get("plan_json") or {}
    role_by_id = {e.get("id"): e.get("role") for e in plan.get("entities", [])}
    target_by_id = {e.get("id"): bool(e.get("is_target")) for e in plan.get("entities", [])}
    vt_by_id = {e.get("id"): e.get("value_type") for e in plan.get("entities", [])}
    levels = extra.get("topology_levels", {}) or {}
    rows = []
    for b in extra.get("executed_blanks", []) or []:
        bid = b.get("blank_id", "?")
        rows.append(
            {
                "blank_id": bid,
                "level": levels.get(bid, "—"),
                "role": role_by_id.get(bid) or "—",
                "value_type": vt_by_id.get(bid) or "—",
                "target?": "✓" if target_by_id.get(bid) else "",
                "status": b.get("status"),
                "value": str(b.get("value", ""))[:120],
                "evidence": ", ".join(b.get("evidence_chunk_ids", []) or []),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["level", "blank_id"], kind="stable").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Orchestrator run inspector", layout="wide")
st.title("🧭 ours_gsw_planner_orchestrator_v1 — run inspector")

cells = _discover_cells()
if not cells:
    st.error(f"No `{_TARGET_SYSTEM_ID}` cells found under {_LOGS}.")
    st.stop()

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

# Cell summary
st.header(f"Cell: {picked['model']}")
metrics = _cell_metrics(cell)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("n", cell.get("n_total", 0))
c2.metric("judge", f"{cell.get('judge_accuracy', 0):.3f}")
c3.metric("EM", f"{cell.get('accuracy', 0):.3f}")
c4.metric("F1", f"{cell.get('mean_f1', 0):.3f}")
c5.metric("wall/Q", f"{metrics.get('mean_wall_s', 0)}s")

# Row 2 — orchestrator-specific
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("avg levels/Q", metrics.get("mean_levels_per_q", 0))
c2.metric("empty-pred %", f"{metrics.get('empty_answer_pct', 0)}%")
c3.metric("fallback %", f"{metrics.get('fallback_pct', 0)}%")
c4.metric("tokens_in/Q", f"{int(metrics.get('mean_prompt_tokens', 0)):,}")
c5.metric("tokens_out/Q", f"{int(metrics.get('mean_completion_tokens', 0)):,}")

with st.expander("Stopped-reason histograms", expanded=False):
    cL, cR = st.columns(2)
    with cL:
        st.caption("Orchestrator-level stopped_reason")
        df = pd.DataFrame(
            [{"reason": k, "count": v} for k, v in (metrics.get("stopped_reasons") or {}).items()]
        ).sort_values("count", ascending=False)
        st.bar_chart(df.set_index("reason"))
        st.dataframe(df, use_container_width=True, hide_index=True)
    with cR:
        st.caption("Per-researcher stopped")
        df = pd.DataFrame(
            [{"reason": k, "count": v} for k, v in (metrics.get("researcher_stops") or {}).items()]
        ).sort_values("count", ascending=False)
        st.bar_chart(df.set_index("reason"))
        st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# Questions table
st.subheader("Questions")
rows = []
for q in cell.get("questions", []):
    traj = q.get("trajectory") or {}
    extra = traj.get("extra") or {}
    rts = extra.get("researcher_traces", []) or []
    rows.append(
        {
            "id": q.get("question_id"),
            "hops": q.get("num_hops", "—"),
            "judge": _judge_icon(q),
            "failure_mode": q.get("failure_mode", "—") or "—",
            "issue_type": _issue_type(q),
            "mode": extra.get("orchestrator_mode", "—"),
            "stopped": extra.get("stopped_reason", "?"),
            "researchers": len(rts),
            "plan_vers": len(extra.get("plan_json_versions") or []) or 1,
            "plan_upd": len(extra.get("plan_updates") or []),
            "validator": int(extra.get("validator_retries_used") or 0),
            "turns": traj.get("turns", 0),
            "wall_s": round(traj.get("wall_time_s", 0), 1),
            "tokens_out": traj.get("completion_tokens", 0),
            "pred": (q.get("predicted_answer", "") or "")[:80],
            "gold": (q.get("gold_answer", "") or "")[:60],
            "question": (q.get("question", "") or "")[:80],
        }
    )
qs_df = pd.DataFrame(rows)

judge_filter = st.sidebar.multiselect(
    "Judge verdict", ["✓", "✗", "—"], default=["✓", "✗", "—"]
)
failure_values = sorted({r["failure_mode"] for r in rows if r["failure_mode"]})
failure_filter = st.sidebar.multiselect(
    "Failure mode",
    failure_values,
    default=failure_values,
)
stop_values = sorted({r["stopped"] for r in rows if r["stopped"]})
stop_filter = st.sidebar.multiselect("Stopped reason", stop_values, default=stop_values)
issue_values = sorted({r["issue_type"] for r in rows if r["issue_type"]})
issue_filter = st.sidebar.multiselect(
    "Issue type",
    issue_values,
    default=issue_values,
    help="Small derived buckets for common planner/orchestrator/validator paths.",
)

filt = qs_df[
    qs_df["judge"].isin(judge_filter)
    & qs_df["failure_mode"].isin(failure_filter)
    & qs_df["stopped"].isin(stop_filter)
    & qs_df["issue_type"].isin(issue_filter)
].reset_index(drop=True)

st.caption(
    f"Showing {len(filt)} / {len(qs_df)} questions after filters."
)
if not filt.empty:
    with st.expander("Filtered subset summary", expanded=False):
        cL, cR = st.columns(2)
        with cL:
            st.caption("failure_mode")
            fmode_df = (
                filt["failure_mode"]
                .value_counts(dropna=False)
                .rename_axis("failure_mode")
                .reset_index(name="count")
            )
            st.dataframe(fmode_df, use_container_width=True, hide_index=True)
        with cR:
            st.caption("issue_type")
            issue_df = (
                filt["issue_type"]
                .value_counts(dropna=False)
                .rename_axis("issue_type")
                .reset_index(name="count")
            )
            st.dataframe(issue_df, use_container_width=True, hide_index=True)
st.dataframe(filt, use_container_width=True, hide_index=True)

if filt.empty:
    st.info("No questions match current filters.")
    st.stop()

# Drilldown
st.markdown("---")
st.subheader("Drilldown")

qid_opts = filt["id"].tolist()
_session_key = f"orch_idx__{picked['dir']}"
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
researcher_traces = extra.get("researcher_traces", []) or []
all_tool_calls = traj.get("tool_calls") or []
all_messages = traj.get("messages") or []

mode = extra.get("orchestrator_mode", "—")
plan_versions = extra.get("plan_json_versions") or []
plan_updates = extra.get("plan_updates") or []
dispatches = extra.get("dispatches") or []

st.markdown(f"**Question**: {q.get('question', '')}")
st.markdown(f"**Gold answer**: {q.get('gold_answer', '')}")
st.markdown(
    f"**Predicted**: `{(q.get('predicted_answer', '') or '(empty)')[:500]}`  "
    f"— **judge**: {_judge_icon(q)}  ·  **stopped**: `{extra.get('stopped_reason','?')}`  "
    f"·  **mode**: `{mode}`"
)
st.markdown(
    f"**Researchers**: {len(researcher_traces)}  ·  "
    f"**Plan revisions**: {max(len(plan_versions), 1)}  ·  "
    f"**Total turns**: {traj.get('turns', 0)}  ·  "
    f"**Wall**: {traj.get('wall_time_s', 0):.1f}s  ·  "
    f"**Tokens out**: {traj.get('completion_tokens', 0):,}"
)
if extra.get("fallback_flag"):
    st.warning(f"**Fallback fired** — reason: `{extra.get('fallback_reason','?')}`")

# Tabs: Orchestrator tab only shown for llm mode.
if mode == "llm":
    tabs = st.tabs([
        "Plan",
        "Orchestrator",
        "Researchers",
        "Validator",
        "Plan versions",
        "Executed blanks",
        "Messages",
        "Reasoning",
    ])
    orch_tab = tabs[1]
    researchers_tab = tabs[2]
    validator_tab = tabs[3]
    plan_versions_tab = tabs[4]
    executed_tab = tabs[5]
    messages_tab = tabs[6]
    reasoning_tab = tabs[7]
else:
    tabs = st.tabs([
        "Plan",
        "Researchers",
        "Validator",
        "Executed blanks",
        "Messages",
        "Reasoning",
    ])
    orch_tab = None
    researchers_tab = tabs[1]
    validator_tab = tabs[2]
    plan_versions_tab = None
    executed_tab = tabs[3]
    messages_tab = tabs[4]
    reasoning_tab = tabs[5]

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
            st.json(plan_json, expanded=False)

# --- Tab: Orchestrator (llm mode only) --------------------------------
if orch_tab is not None:
    with orch_tab:
        orch_calls = [tc for tc in all_tool_calls if tc.get("level") == -1]
        if not orch_calls:
            st.info("No orchestrator-level tool calls recorded.")
        else:
            # Top-level orchestrator timeline
            rows_orch = []
            for i, tc in enumerate(orch_calls, start=1):
                name = tc.get("name", "?")
                args = tc.get("args", {}) or {}
                if name == "dispatch_subplan":
                    summary = f"blank_ids={args.get('blank_ids')}"
                elif name == "request_plan_update":
                    summary = (
                        f"reason={str(args.get('reason',''))[:60]!r}  "
                        f"evidence={str(args.get('evidence',''))[:40]!r}"
                    )
                elif name == "submit_answer":
                    summary = f"answer={str(args.get('answer',''))[:60]!r}"
                elif name == "get_state":
                    summary = "(inspect state)"
                else:
                    summary = json.dumps(args, default=str)[:80]
                rows_orch.append(
                    {
                        "#": i,
                        "turn": tc.get("turn", "?"),
                        "tool": name,
                        "args": summary,
                        "duration_s": round(tc.get("duration_s", 0), 3),
                        "error": tc.get("error", "") or "",
                    }
                )
            odf = pd.DataFrame(rows_orch)

            def _style_tool(val):
                return (
                    f"background-color: {_tool_palette(val)}; "
                    f"color: white; font-weight: bold; text-align: center;"
                )
            st.dataframe(
                odf.style.map(_style_tool, subset=["tool"]),
                use_container_width=True, hide_index=True,
            )

            # Per-turn orchestrator tool call detail
            st.caption("Per-turn orchestrator tool-call detail:")
            for tc in orch_calls:
                name = tc.get("name", "?")
                with st.expander(
                    f"turn {tc.get('turn')} · {name} · "
                    f"{json.dumps(tc.get('args', {}), default=str)[:120]}",
                    expanded=False,
                ):
                    _result = tc.get("result_full") or tc.get("result_preview") or ""
                    _stripped = _result.strip() if _result else ""
                    if _stripped and (
                        (_stripped.startswith("{") and _stripped.endswith("}"))
                        or (_stripped.startswith("[") and _stripped.endswith("]"))
                    ):
                        try:
                            st.json(json.loads(_stripped), expanded=False)
                        except Exception:  # noqa: BLE001
                            st.code(_result or "(empty)", language=None, wrap_lines=True)
                    else:
                        st.code(_result or "(empty)", language=None, wrap_lines=True)
                    if tc.get("error"):
                        st.error(f"error: {tc['error']}")

            # Orchestrator system prompts (last system message shown).
            # In llm mode each orchestrator turn rebuilds the system prompt;
            # we show the first (turn 1) system message for inspection.
            sys_msgs = [
                m for m in all_messages
                if m.get("role") == "system" and m.get("_level") == -1
            ]
            if sys_msgs:
                with st.expander("📜 Orchestrator system prompt (turn 1 snapshot)"):
                    st.code(sys_msgs[0].get("content", ""), language=None, wrap_lines=True)

        # Plan-update events (if any fired)
        if plan_updates:
            st.markdown("### Plan-update events")
            for i, pu in enumerate(plan_updates, start=1):
                icon = "❌" if pu.get("rejected") else "🔄"
                header = (
                    f"{icon} #{i} turn={pu.get('turn','?')}  "
                    f"reason={str(pu.get('reason',''))[:40]!r}"
                )
                with st.expander(header, expanded=False):
                    st.markdown(f"**Reason**: {pu.get('reason','')}")
                    st.markdown(f"**Evidence**: {pu.get('evidence','')}")
                    if pu.get("rejected"):
                        st.error(f"REJECTED: {pu.get('error','')}")
                    else:
                        st.success(f"Diff: {pu.get('diff_summary','')}")
                        st.markdown(
                            f"- preserved: `{pu.get('preserved_ids',[])}`\n"
                            f"- added: `{pu.get('added_ids',[])}`\n"
                            f"- dropped: `{pu.get('dropped_ids',[])}`"
                        )
        else:
            st.caption("No plan-updater invocations on this question.")

        # Dispatches list
        if dispatches:
            st.markdown("### Dispatches")
            # Flatten revision_requests for the table view; full
            # detail surfaces in the per-dispatch expander below.
            disp_rows = []
            for d in dispatches:
                rrs = d.get("revision_requests") or []
                disp_rows.append({
                    "dispatch_idx": d.get("dispatch_idx"),
                    "blank_ids": d.get("blank_ids"),
                    "hints": (d.get("hints") or "")[:60],
                    "partial": d.get("partial"),
                    "revisions_requested": len(rrs),
                })
            st.dataframe(
                pd.DataFrame(disp_rows),
                use_container_width=True, hide_index=True,
            )
            # Surface any escalations prominently.
            for d in dispatches:
                rrs = d.get("revision_requests") or []
                for rr in rrs:
                    st.warning(
                        f"🆙 dispatch #{d.get('dispatch_idx')} — researcher "
                        f"escalated for `{rr.get('blank_id','?')}`:\n\n"
                        f"**reason**: {rr.get('reason','')}\n\n"
                        f"**hint**: {rr.get('hint','')}"
                    )


# --- Tab: Validator ---------------------------------------------------
with validator_tab:
    validator_verdicts = extra.get("validator_verdicts") or []
    validator_mode = extra.get("synthesis_validator_mode", "—")
    validator_retries = int(extra.get("validator_retries_used") or 0)
    validator_repairs = int(extra.get("validator_repairs_used") or 0)
    validator_calls = [
        tc for tc in all_tool_calls
        if tc.get("level") == -1 and tc.get("name") == "submit_answer"
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mode", validator_mode)
    c2.metric("verdicts", len(validator_verdicts))
    c3.metric("rejections", validator_retries)
    c4.metric("repairs", validator_repairs)

    if not validator_verdicts:
        st.info("No validator verdicts recorded for this question.")
    else:
        rows_v = []
        for i, v in enumerate(validator_verdicts, start=1):
            rows_v.append(
                {
                    "#": i,
                    "mode": v.get("mode", validator_mode),
                    "verdict": v.get("verdict"),
                    "proposed": v.get("proposed_answer"),
                    "suggested": v.get("suggested_correction"),
                    "corrected": v.get("corrected_answer"),
                    "confidence": v.get("confidence"),
                    "error_type": v.get("error_type"),
                    "auto_repair": v.get("auto_repair_accepted", False),
                    "repair_reason": v.get("auto_repair_reason", ""),
                    "flagged_blank": v.get("flagged_blank"),
                    "reason": v.get("reason"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows_v),
            use_container_width=True,
            hide_index=True,
        )

        for i, v in enumerate(validator_verdicts, start=1):
            verdict = v.get("verdict", "?")
            icon = "✅" if verdict == "correct" else "❌"
            repaired = " · repaired" if v.get("auto_repair_accepted") else ""
            with st.expander(
                f"{icon} verdict #{i}: {verdict}{repaired} · proposed={v.get('proposed_answer')!r}",
                expanded=(verdict == "wrong"),
            ):
                st.markdown(f"**Reason**: {v.get('reason', '')}")
                if v.get("flagged_blank"):
                    st.markdown(f"**Flagged blank**: `{v.get('flagged_blank')}`")
                if v.get("suggested_correction") is not None:
                    st.markdown(
                        f"**Suggested correction**: `{v.get('suggested_correction')}`"
                    )
                if v.get("corrected_answer") is not None:
                    st.markdown(
                        f"**Corrected answer**: `{v.get('corrected_answer')}`"
                    )
                if v.get("confidence") or v.get("error_type"):
                    st.markdown(
                        f"**Repair gate**: confidence=`{v.get('confidence')}` · "
                        f"error_type=`{v.get('error_type')}` · "
                        f"accepted=`{bool(v.get('auto_repair_accepted'))}`"
                    )
                if v.get("auto_repair_reason"):
                    st.caption(f"auto_repair_reason: {v.get('auto_repair_reason')}")
                if v.get("evidence_support"):
                    st.markdown(f"**Evidence support**: {v.get('evidence_support')}")

    st.markdown("### Submit / Validator Flow")
    if not validator_calls:
        st.caption("No submit_answer calls recorded.")
    else:
        submit_rows = []
        for tc in validator_calls:
            args = tc.get("args", {}) or {}
            result_text = tc.get("result_full") or tc.get("result_preview") or ""
            result_obj = None
            if result_text.strip().startswith("{"):
                try:
                    result_obj = json.loads(result_text)
                except Exception:  # noqa: BLE001
                    result_obj = None
            submit_rows.append(
                {
                    "turn": tc.get("turn"),
                    "answer_arg": args.get("answer"),
                    "ok": (
                        result_obj.get("ok")
                        if isinstance(result_obj, dict)
                        else None
                    ),
                    "validator_repaired": (
                        result_obj.get("validator_repaired")
                        if isinstance(result_obj, dict)
                        else None
                    ),
                    "error": tc.get("error", "") or "",
                    "result_preview": result_text[:180],
                }
            )
        st.dataframe(
            pd.DataFrame(submit_rows),
            use_container_width=True,
            hide_index=True,
        )

        for tc in validator_calls:
            with st.expander(
                f"submit_answer turn {tc.get('turn')} · answer={((tc.get('args') or {}).get('answer'))!r}",
                expanded=bool(tc.get("error")),
            ):
                st.json(tc.get("args", {}) or {}, expanded=False)
                result_text = tc.get("result_full") or tc.get("result_preview") or ""
                stripped = result_text.strip()
                if stripped and (
                    (stripped.startswith("{") and stripped.endswith("}"))
                    or (stripped.startswith("[") and stripped.endswith("]"))
                ):
                    try:
                        st.json(json.loads(stripped), expanded=False)
                    except Exception:  # noqa: BLE001
                        st.code(result_text, language=None, wrap_lines=True)
                else:
                    st.code(result_text or "(empty)", language=None, wrap_lines=True)


# --- Tab: Researchers -------------------------------------------------
with researchers_tab:
    if not researcher_traces:
        st.info("No researcher_traces in trajectory (adapter failed before spawning any).")
    else:
        st.caption(
            f"{len(researcher_traces)} researcher(s) were spawned — one per topological level."
        )
        for rt in researcher_traces:
            lvl = rt.get("level", "?")
            allowed = rt.get("allowed_blank_ids", [])
            stopped = rt.get("stopped", "?")
            turns = rt.get("turns", 0)
            resolved = rt.get("resolved", [])
            unresolved = rt.get("unresolved", [])
            status_icon = (
                "✅" if stopped == "all_resolved"
                else "⏱️" if stopped == "max_turns"
                else "❌" if stopped == "llm_error"
                else "🛑" if stopped == "no_tool_call"
                else "🆙" if stopped == "plan_revision_requested"
                else "❔"
            )
            header = (
                f"{status_icon} Level {lvl}  ·  stopped=`{stopped}`  ·  "
                f"turns={turns}  ·  "
                f"resolved {len(resolved)}/{len(allowed)}  ·  "
                f"tokens_out={rt.get('completion_tokens', 0):,}"
            )
            with st.expander(header, expanded=(stopped != "all_resolved")):
                cL, cR = st.columns([2, 3])
                with cL:
                    st.markdown(f"**Assigned slice**: `{allowed}`")
                    st.markdown(f"**Resolved**: `{resolved}`")
                    if unresolved:
                        st.warning(f"Unresolved: `{unresolved}`")
                    st.markdown(
                        f"**Prompt tokens**: {rt.get('prompt_tokens', 0):,}  ·  "
                        f"**Completion tokens**: {rt.get('completion_tokens', 0):,}"
                    )
                with cR:
                    level_tl = _tool_timeline_df(all_tool_calls, level=lvl)
                    if level_tl.empty:
                        st.caption("(no tool calls recorded for this level)")
                    else:
                        def _style_tool(val):
                            return (
                                f"background-color: {_tool_palette(val)}; "
                                f"color: white; font-weight: bold; text-align: center;"
                            )
                        styled = level_tl.style.map(_style_tool, subset=["tool"])
                        st.dataframe(styled, use_container_width=True, hide_index=True)

                # --- Turn-by-turn narrative for THIS researcher -------
                # Pair each assistant message with its reasoning + tool
                # calls + matching tool results. This is where you see
                # what actually happened inside the researcher.
                level_msgs = [m for m in all_messages if m.get("_level") == lvl]
                reasoning_by_key_r = _parse_reasoning(
                    traj.get("hidden_reasoning") or ""
                )

                # Build a tool_call_id -> result lookup.
                tool_result_by_id: dict[str, str] = {}
                for tm in level_msgs:
                    if tm.get("role") == "tool" and tm.get("tool_call_id"):
                        tool_result_by_id[tm["tool_call_id"]] = tm.get("content", "")

                st.markdown("**Turn-by-turn narrative**")
                assistant_turn = 0
                for m in level_msgs:
                    role = m.get("role", "?")
                    if role != "assistant":
                        continue
                    assistant_turn += 1
                    reasoning_text = reasoning_by_key_r.get((lvl, assistant_turn), "")
                    tcalls = m.get("tool_calls") or []
                    tool_names = [tc["function"]["name"] for tc in tcalls]
                    hdr_bits = [f"🤖 turn {assistant_turn}"]
                    if tool_names:
                        hdr_bits.append(f"tools=[{', '.join(tool_names)}]")
                    if reasoning_text:
                        hdr_bits.append(f"💭 {len(reasoning_text)}c")
                    with st.expander(" · ".join(hdr_bits), expanded=True):
                        if reasoning_text:
                            st.markdown("**💭 Reasoning**")
                            st.code(reasoning_text, language=None, wrap_lines=True)
                        text = m.get("content") or ""
                        if text.strip():
                            st.markdown("**Assistant text**")
                            st.code(text, language=None, wrap_lines=True)
                        for tc_idx, tc in enumerate(tcalls, start=1):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            try:
                                args = json.loads(fn.get("arguments", "") or "{}")
                            except json.JSONDecodeError:
                                args = {"_raw": fn.get("arguments", "")}
                            st.markdown(f"**🔧 tool #{tc_idx}: `{name}`**")
                            st.json(args, expanded=False)
                            # Match tool result.
                            tc_id = tc.get("id") or ""
                            result_text = tool_result_by_id.get(tc_id, "")
                            if result_text:
                                st.markdown("_result:_")
                                rs = result_text.strip()
                                if rs and (
                                    (rs.startswith("{") and rs.endswith("}"))
                                    or (rs.startswith("[") and rs.endswith("]"))
                                ):
                                    try:
                                        st.json(json.loads(rs), expanded=False)
                                    except Exception:  # noqa: BLE001
                                        st.code(result_text, language=None, wrap_lines=True)
                                else:
                                    st.code(result_text, language=None, wrap_lines=True)

                # Per-level system prompt (the one the researcher saw)
                system_msg = next(
                    (m for m in level_msgs if m.get("role") == "system"), None
                )
                if system_msg is not None:
                    with st.expander("📜 Researcher system prompt (what this level saw)"):
                        st.code(system_msg.get("content", ""), language=None, wrap_lines=True)

# --- Tab: Plan versions (llm mode only) -------------------------------
if plan_versions_tab is not None:
    with plan_versions_tab:
        if len(plan_versions) <= 1:
            st.caption("Only one plan version — no plan-updater invocations.")
            if plan_versions:
                st.json(plan_versions[0], expanded=False)
        else:
            st.caption(
                f"{len(plan_versions)} plan versions (initial + "
                f"{len(plan_versions) - 1} revisions)"
            )
            for i, p in enumerate(plan_versions):
                tag = "initial" if i == 0 else f"revision {i}"
                with st.expander(f"Version {i} ({tag})", expanded=(i == 0 or i == len(plan_versions) - 1)):
                    try:
                        order_v, levels_v, topo_error_v = compute_topo_info(p)
                        if topo_error_v:
                            st.warning(f"Topology: {topo_error_v}")
                        else:
                            st.graphviz_chart(
                                _plan_to_dot(p, order=order_v, levels=levels_v),
                                use_container_width=True,
                            )
                    except Exception:  # noqa: BLE001
                        pass
                    st.json(p, expanded=False)


# --- Tab: Executed blanks ---------------------------------------------
with executed_tab:
    eb = _executed_blanks_df(extra)
    if eb.empty:
        st.info("No executed_blanks (adapter didn't emit state, or fallback fired).")
    else:
        st.dataframe(eb, use_container_width=True, hide_index=True)

# --- Tab: Messages -----------------------------------------------------
with messages_tab:
    if not all_messages:
        st.info("No full-message log captured.")
    else:
        # Parse reasoning into (level, turn) → text for inline display.
        reasoning_by_key = _parse_reasoning(traj.get("hidden_reasoning") or "")

        # Group messages by level.
        by_level: dict[Any, list[dict]] = {}
        for m in all_messages:
            by_level.setdefault(m.get("_level", "—"), []).append(m)

        def _render_content(content):
            """Render message content with wrapping; use st.json for dicts."""
            if content is None:
                st.caption("(empty)")
                return
            if isinstance(content, (dict, list)):
                st.json(content, expanded=False)
                return
            s = str(content)
            # If it's JSON-looking, try to pretty-print as tree.
            stripped = s.strip()
            if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            ):
                try:
                    parsed = json.loads(stripped)
                    st.json(parsed, expanded=False)
                    return
                except Exception:  # noqa: BLE001
                    pass
            # Plain string: code block with soft-wrap.
            st.code(s, language=None, wrap_lines=True)

        for lvl in sorted(by_level.keys(), key=lambda x: (x is None, str(x))):
            if lvl == -1:
                header = "### 🎯 Orchestrator (level -1)"
            else:
                header = f"### 🔬 Researcher dispatch / level {lvl}"
            st.markdown(header)

            # Walk messages in order; maintain a per-level turn counter
            # incremented each time we emit an assistant message, so we
            # can look up the matching reasoning chunk.
            assistant_turn = 0
            for i, m in enumerate(by_level[lvl]):
                role = m.get("role", "?")
                content = m.get("content")
                tool_calls = m.get("tool_calls") or []
                preview_len = len(str(content)) if content else 0

                # For assistant messages: increment turn counter; look up reasoning.
                reasoning_text = ""
                if role == "assistant":
                    assistant_turn += 1
                    reasoning_text = reasoning_by_key.get((lvl, assistant_turn), "")

                # Icons per role for quick visual scan
                role_icon = {
                    "system": "⚙️",
                    "user": "👤",
                    "assistant": "🤖",
                    "tool": "🔧",
                }.get(role, "•")

                # Build expander header
                hdr_parts = [f"[{i}] {role_icon} {role} · {preview_len} chars"]
                if role == "assistant":
                    hdr_parts.append(f"turn {assistant_turn}")
                    if tool_calls:
                        tool_names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                        hdr_parts.append(f"tools=[{tool_names}]")
                    if reasoning_text:
                        hdr_parts.append(f"💭 +{len(reasoning_text)} reasoning")
                hdr = " · ".join(hdr_parts)

                # Auto-expand assistant turns (where the action is);
                # collapse tool results + system by default.
                expanded_default = role == "assistant"
                with st.expander(hdr, expanded=expanded_default):
                    if reasoning_text:
                        st.markdown("**💭 Reasoning (hidden_reasoning for this turn)**")
                        st.code(reasoning_text, language=None, wrap_lines=True)
                        st.markdown("**Assistant output**")
                    _render_content(content)
                    if tool_calls:
                        st.markdown("**Tool calls**")
                        for tc_idx, tc in enumerate(tool_calls, start=1):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            try:
                                args = json.loads(fn.get("arguments", "") or "{}")
                            except json.JSONDecodeError:
                                args = {"_raw": fn.get("arguments", "")}
                            st.markdown(f"`{tc_idx}` **{name}**")
                            st.json(args, expanded=False)

            st.markdown("---")


# --- Tab: Reasoning ---------------------------------------------------
with reasoning_tab:
    hidden = traj.get("hidden_reasoning") or ""
    summary = traj.get("reasoning") or ""
    if not (hidden.strip() or summary.strip()):
        st.info("(No reasoning captured.)")
    else:
        if summary.strip():
            st.markdown("**Adapter final-answer**")
            st.code(summary, language=None, wrap_lines=True)
        if hidden.strip():
            st.markdown(
                "**Hidden reasoning (orchestrator + per-researcher sections)**"
            )
            st.caption(
                "Also shown inline per-turn in the Messages tab — this is the "
                "raw concatenated view for quick scanning."
            )
            st.code(hidden, language=None, wrap_lines=True)
