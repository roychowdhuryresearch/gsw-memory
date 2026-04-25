"""Streamlit inspector for plan-dump artefacts.

Points at a ``logs/planner_dump_<ts>/`` directory produced by
``dump_plans.py`` and lets the user browse each question's emitted plan,
inspect the raw LLM response + reasoning + full prompt, and compare
models side-by-side on the same question.

Run with::

    .venv/bin/streamlit run playground/planner_dump_inspector.py

Discovers dump dirs under ``research_agent/logs/planner_dump_*/``.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from _planner_viz import (
    compute_topo_info,
    constraints_df as _constraint_df,
    dangling_badge,
    entities_df as _entity_df,
    plan_to_dot as _plan_to_dot,
    solve_steps_markdown,
    vps_df as _vp_df,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
    _PARENT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
    if _PARENT_ENV.exists():
        load_dotenv(_PARENT_ENV, override=False)
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Discovery + load
# ---------------------------------------------------------------------------


_ROOT = Path(__file__).resolve().parent.parent  # research_agent/
_LOGS = _ROOT / "logs"


@st.cache_data
def _discover_dumps() -> list[Path]:
    """Find all planner_dump_* directories, newest first."""
    if not _LOGS.exists():
        return []
    return sorted(_LOGS.glob("planner_dump_*"), key=lambda p: p.name, reverse=True)


@st.cache_data
def _discover_models(dump_dir_str: str) -> list[str]:
    dump_dir = Path(dump_dir_str)
    if not dump_dir.exists():
        return []
    return sorted(
        [d.name for d in dump_dir.iterdir() if d.is_dir() and list(d.glob("q_*.json"))]
    )


@st.cache_data
def _load_records(dump_dir_str: str, model_slug: str) -> list[dict[str, Any]]:
    model_dir = Path(dump_dir_str) / model_slug
    records: list[dict[str, Any]] = []
    for f in sorted(model_dir.glob("q_*.json")):
        try:
            records.append(json.loads(f.read_text()))
        except Exception:  # noqa: BLE001
            continue
    return records


def _record_by_qid(records: list[dict], qid: str) -> Optional[dict]:
    for r in records:
        if r.get("question_id") == qid:
            return r
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _parse_status_badge(record: dict) -> str:
    if record.get("parse_ok"):
        if record.get("repair_used"):
            return "✓ (after repair)"
        return "✓"
    return f"✗ {record.get('parse_error', '')[:80]}"


def _target_value_type(plan: Optional[dict]) -> str:
    if not plan:
        return "—"
    for e in plan.get("entities", []):
        if e.get("is_target"):
            return e.get("value_type") or "—"
    return "—"


_PLAN_REVIEW_MODEL = "gpt-5"

_PLAN_REVIEW_SYSTEM = """You are auditing a GSW-fragment plan for a \
question-planning system.

Judge ONLY from:
- the question text,
- the emitted plan JSON,
- the provided structural/topology checks.

Do not use outside world knowledge to repair the plan, fill missing gaps,
or excuse unsupported entities.

Architecture rules to enforce:
- Filled entities should be grounded in the question text. Minor surface
  normalization is fine; inventing a new named entity, date anchor,
  location, candidate, or other fact from model knowledge is not.
- Blank entities represent unknown values to be filled later.
- Exactly one blank should be the target.
- Every entity should be usable by at least one verb-phrase or constraint.
- Verb-phrases should encode retrieval signals or blank-to-blank projection.
- Constraints should appear only when arithmetic or argmax/argmin style
  selection is needed.
- The topological sort is computed from blank dependencies induced by
  constraints and blank-entity projection edges.
- A plan can be topologically valid yet still semantically wrong if it
  fails to encode a dependency the executor actually needs.

Important:
- Do NOT penalize benign predicate normalization like `died_in`.
- Do flag hallucination when the plan introduces a grounded entity or
  fact that the question did not provide.
- Evaluate plan quality, grounding, and dependency/topology quality.

Respond in plain text with these short sections:
- Overall
- Grounding / hallucination
- Topology / dependency quality
- Issues
- Suggested fixes

Use concrete references to entity ids / blank ids / verb phrases when helpful."""


def _normalize_surface(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _target_blank_ids(plan: Optional[dict]) -> list[str]:
    if not plan:
        return []
    return [e.get("id", "") for e in plan.get("entities", []) if e.get("is_target")]


def _missing_role_entity_ids(plan: Optional[dict]) -> list[str]:
    if not plan:
        return []
    return [
        e.get("id", "")
        for e in plan.get("entities", [])
        if not (e.get("role") or "").strip()
    ]


_COLLECTIVE_STATE_PREFIX = "expanded_from_collective:"

# Token patterns that trigger the date-anchor rule (Hard rule 2).
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
_DATE_ANCHOR_PHRASE_RE = re.compile(
    r"\b(as of|in the year|the year that|after [0-9]|before [0-9])",
    re.IGNORECASE,
)
_DATE_ANCHOR_ROLES = {"year-anchor", "date-anchor", "as-of-date"}


def _surface_ungrounded_filled_entities(
    question: str,
    plan: Optional[dict],
) -> list[dict[str, Any]]:
    """Heuristic only: filled entity surface not found in question text.

    Excludes entities whose `state` begins with
    ``"expanded_from_collective:<group>"`` — those are legitimate
    planner-time expansions (Hard rule 3) whose provenance is declared
    via the breadcrumb, not grounding. Those entities are surfaced
    separately by ``_expanded_from_collective_entities``.
    """
    if not plan:
        return []
    q_norm = f" {_normalize_surface(question)} "
    out: list[dict[str, Any]] = []
    for e in plan.get("entities", []):
        if e.get("kind") != "filled":
            continue
        name = (e.get("name") or "").strip()
        if not name:
            continue
        state = (e.get("state") or "").strip()
        if state.startswith(_COLLECTIVE_STATE_PREFIX):
            # Legitimate collective expansion — not ungrounded.
            continue
        name_norm = _normalize_surface(name)
        if name_norm and f" {name_norm} " not in q_norm:
            out.append(
                {
                    "id": e.get("id", ""),
                    "name": name,
                    "category": bool(e.get("category")),
                }
            )
    return out


def _expanded_from_collective_entities(
    plan: Optional[dict],
) -> list[dict[str, Any]]:
    """Return filled entities whose `state` declares collective expansion.

    These are legitimate under Hard rule 3 but still worth surfacing so
    the auditor can count / review / challenge the planner's use of
    world-knowledge enumeration.
    """
    if not plan:
        return []
    out: list[dict[str, Any]] = []
    for e in plan.get("entities", []):
        if e.get("kind") != "filled":
            continue
        state = (e.get("state") or "").strip()
        if not state.startswith(_COLLECTIVE_STATE_PREFIX):
            continue
        collective_name = state[len(_COLLECTIVE_STATE_PREFIX):].strip()
        out.append(
            {
                "id": e.get("id", ""),
                "name": (e.get("name") or "").strip(),
                "collective": collective_name,
            }
        )
    return out


def _missing_date_anchors(
    question: str,
    plan: Optional[dict],
) -> list[str]:
    """Date tokens in the question with no matching filled entity role.

    Returns a list of date-anchor snippets pulled from the question
    when the plan has zero filled entities with role in
    ``{year-anchor, date-anchor, as-of-date}``. An empty list means
    either the question has no explicit temporal token, or the plan
    correctly surfaced at least one date anchor.
    """
    if not question:
        return []
    tokens: list[str] = []
    tokens.extend(_YEAR_RE.findall(question))
    tokens.extend(m.group(0) for m in _MONTH_RE.finditer(question))
    tokens.extend(m.group(0) for m in _DATE_ANCHOR_PHRASE_RE.finditer(question))
    if not tokens:
        return []

    if not plan:
        return sorted(set(tokens))

    has_date_anchor = any(
        (e.get("role") or "").strip() in _DATE_ANCHOR_ROLES
        for e in plan.get("entities", [])
        if e.get("kind") == "filled"
    )
    if has_date_anchor:
        return []
    return sorted(set(tokens))


def _topology_levels(order: list[str], levels: dict[str, int]) -> list[dict[str, Any]]:
    if not order:
        return []
    grouped: dict[int, list[str]] = {}
    for blank_id in order:
        lvl = levels.get(blank_id, 0)
        grouped.setdefault(lvl, []).append(blank_id)
    return [
        {"level": lvl + 1, "blank_ids": grouped[lvl]}
        for lvl in sorted(grouped)
    ]


def _format_topology_order(order: list[str], topo_error: Optional[str]) -> str:
    if topo_error:
        return f"— ({topo_error})"
    if not order:
        return "—"
    return " -> ".join(f"`{blank_id}`" for blank_id in order)


def _format_topology_levels(order: list[str], levels: dict[str, int]) -> str:
    grouped = _topology_levels(order, levels)
    if not grouped:
        return "—"
    parts = []
    for row in grouped:
        blanks = ", ".join(f"`{blank_id}`" for blank_id in row["blank_ids"])
        parts.append(f"L{row['level']}: {blanks}")
    return " · ".join(parts)


def _plan_checks(
    question: str,
    plan: Optional[dict],
    order: list[str],
    levels: dict[str, int],
    topo_error: Optional[str],
) -> dict[str, Any]:
    entities = plan.get("entities", []) if plan else []
    return {
        "filled_count": sum(1 for e in entities if e.get("kind") == "filled"),
        "blank_count": sum(1 for e in entities if e.get("kind") == "blank"),
        "target_blank_ids": _target_blank_ids(plan),
        "missing_role_entity_ids": _missing_role_entity_ids(plan),
        "surface_ungrounded_filled_entities": _surface_ungrounded_filled_entities(
            question, plan
        ),
        "expanded_from_collective_entities": _expanded_from_collective_entities(plan),
        "missing_date_anchors": _missing_date_anchors(question, plan),
        "topology_validation_error": topo_error or "",
        "topological_order": order,
        "topology_levels": _topology_levels(order, levels),
    }


@st.cache_data(show_spinner=False)
def _run_gpt5_plan_review(
    *,
    question: str,
    plan_json_text: str,
    checks_json_text: str,
    review_model: str,
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY is not set."}

    try:
        from research_agent.models.llm_client import LLMClient
    except Exception as exc:  # noqa: BLE001
        return {"error": f"imports failed: {exc}"}

    prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Plan JSON:\n"
        f"{plan_json_text}\n\n"
        "Structural / topology checks:\n"
        f"{checks_json_text}\n\n"
        "Evaluate the plan now and return the review in plain text."
    )

    try:
        client = LLMClient(
            model=review_model,
            base_url=None,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            default_temperature=0.0,
            default_max_tokens=3200,
            reasoning_effort="low",
        )
        resp = client.chat(
            messages=[
                {"role": "system", "content": _PLAN_REVIEW_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2400,
            temperature=0.0,
            stream=False,
        )
        raw_text = getattr(resp, "text", "") or ""
        return {
            "raw_text": raw_text,
            "reasoning": getattr(resp, "reasoning_content", "") or "",
            "prompt_tokens": int(getattr(resp, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(resp, "completion_tokens", 0) or 0),
            "model": review_model,
            "finish_reason": getattr(resp, "finish_reason", "") or "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _render_gpt5_plan_review(
    *,
    question: str,
    plan: Optional[dict],
    checks: dict[str, Any],
    enabled: bool,
    review_model: str,
    review_key: str,
) -> None:
    with st.expander(f"GPT-5 plan review ({review_model})", expanded=False):
        if not plan:
            st.info("No parsed plan available for GPT review.")
            return
        if not enabled:
            st.caption("Enable `GPT-5 plan review` in the sidebar to run this check.")
            return

        requested_key = f"planner_dump_review_requested__{review_key}"
        cols = st.columns([1, 1, 4])
        if cols[0].button("Run review", key=f"run_review__{review_key}"):
            st.session_state[requested_key] = True
        if cols[1].button("Clear", key=f"clear_review__{review_key}"):
            st.session_state.pop(requested_key, None)

        if not st.session_state.get(requested_key, False):
            st.caption("Click `Run review` to analyze only this plan.")
            return

        plan_json_text = json.dumps(plan, indent=2, sort_keys=True)
        checks_json_text = json.dumps(checks, indent=2, sort_keys=True)
        with st.spinner(f"Reviewing plan with {review_model}..."):
            result = _run_gpt5_plan_review(
                question=question,
                plan_json_text=plan_json_text,
                checks_json_text=checks_json_text,
                review_model=review_model,
            )

        if result.get("error"):
            st.warning(f"GPT review failed: {result['error']}")
            return

        raw_text = (result.get("raw_text") or "").strip()
        reasoning_text = (result.get("reasoning") or "").strip()
        if raw_text:
            st.markdown(raw_text)
        elif reasoning_text:
            finish_reason = result.get("finish_reason", "") or "unknown"
            st.warning(
                "GPT review did not produce final visible text; "
                f"showing reasoning summary instead. finish_reason={finish_reason}"
            )
            st.markdown(reasoning_text)
        else:
            finish_reason = result.get("finish_reason", "") or "unknown"
            st.warning(f"GPT review returned empty text. finish_reason={finish_reason}")

        st.caption(
            f"{result.get('model', review_model)} tokens: "
            f"{result.get('prompt_tokens', 0)} -> {result.get('completion_tokens', 0)}"
        )
        if (result.get("reasoning") or "").strip():
            with st.expander("Reasoning summary", expanded=False):
                st.code(result.get("reasoning", ""), language="text")
        with st.expander("Raw GPT review text", expanded=False):
            st.code(result.get("raw_text", ""), language="text")


def _render_record_panel(
    record: dict,
    label: str = "",
    *,
    enable_gpt_review: bool = False,
    review_model: str = _PLAN_REVIEW_MODEL,
) -> None:
    """Render a single Q's plan + prompts + reasoning into the current container."""
    st.markdown(
        f"### {label} "
        f"Q{record['question_id']} · {record.get('num_hops','?')}-hop · {record.get('model','')}"
    )
    st.markdown(f"**Question**: {record['question']}")
    st.markdown(f"**Gold answer**: {record['gold_answer']}")
    parse = _parse_status_badge(record)
    tokens = f"{record.get('prompt_tokens',0)} → {record.get('completion_tokens',0)}"
    st.markdown(
        f"**Parse**: {parse} · **Wall**: {record.get('wall_s', 0):.1f}s · "
        f"**Tokens** (in/out): {tokens}"
    )

    plan = record.get("plan")
    order, levels, topo_error = compute_topo_info(plan)
    checks = _plan_checks(record["question"], plan, order, levels, topo_error)
    st.markdown(
        f"**Target value_type**: `{_target_value_type(plan)}`  ·  "
        f"**Graph**: {dangling_badge(topo_error)}"
    )
    st.markdown(f"**Topo order**: {_format_topology_order(order, topo_error)}")
    st.markdown(f"**Topo levels**: {_format_topology_levels(order, levels)}")
    st.markdown(
        f"**Checks**: targets={len(checks['target_blank_ids'])} · "
        f"missing_roles={len(checks['missing_role_entity_ids'])} · "
        f"surface_grounding_flags={len(checks['surface_ungrounded_filled_entities'])}"
    )
    if topo_error and plan:
        st.warning(f"Plan failed schema/topo validation — {topo_error}")
    if len(checks["target_blank_ids"]) != 1:
        st.warning(
            "Expected exactly one target blank, found "
            f"{len(checks['target_blank_ids'])}: {checks['target_blank_ids']}"
        )
    if checks["missing_role_entity_ids"]:
        st.warning(
            "Entities missing `role`: "
            + ", ".join(f"`{eid}`" for eid in checks["missing_role_entity_ids"])
        )
    if checks["surface_ungrounded_filled_entities"]:
        flagged = ", ".join(
            f"`{row['id']}`={row['name']!r}"
            for row in checks["surface_ungrounded_filled_entities"]
        )
        st.warning(
            "Filled entities not surface-grounded in the question "
            f"(heuristic only): {flagged}"
        )

    if plan:
        with st.expander("📊 Visual graph", expanded=True):
            st.graphviz_chart(
                _plan_to_dot(plan, order=order, levels=levels),
                use_container_width=True,
            )
        with st.expander("🪜 Solve steps (topological order)", expanded=False):
            st.markdown(solve_steps_markdown(plan, order, levels))
        _render_gpt5_plan_review(
            question=record["question"],
            plan=plan,
            checks=checks,
            enabled=enable_gpt_review,
            review_model=review_model,
            review_key=f"{record.get('model','')}__{record.get('question_id','')}",
        )

    st.subheader("Entities")
    ent_df = _entity_df(plan)
    if not ent_df.empty:
        st.dataframe(ent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No entities (plan missing).")

    st.subheader("VerbPhrases")
    vp_df = _vp_df(plan)
    if not vp_df.empty:
        st.dataframe(vp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No verb phrases.")

    st.subheader("Constraints")
    con_df = _constraint_df(plan)
    if not con_df.empty:
        st.dataframe(con_df, use_container_width=True, hide_index=True)
    else:
        st.caption("_(none)_")

    # Tabs for raw / reasoning / system / user / repair
    tab_names = ["Raw response", "Reasoning", "System prompt", "User prompt"]
    if record.get("repair_used"):
        tab_names += ["Repair response", "Repair reasoning"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        raw = record.get("raw_response") or "(empty)"
        st.code(raw, language="json" if record.get("parse_ok") else "text")
    with tabs[1]:
        reasoning = record.get("raw_reasoning", "") or ""
        if reasoning.strip():
            st.code(reasoning, language="text")
        else:
            st.info("_Not emitted for this model (no reasoning_content returned)._")
    with tabs[2]:
        st.code(record.get("system_prompt", ""), language="text")
    with tabs[3]:
        st.code(record.get("user_prompt", ""), language="text")
    if record.get("repair_used"):
        with tabs[4]:
            st.code(record.get("repair_response") or "(empty)", language="text")
        with tabs[5]:
            rr = record.get("repair_reasoning", "") or ""
            if rr.strip():
                st.code(rr, language="text")
            else:
                st.info("_Repair reasoning not emitted._")


def _summary_row(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    ok = sum(1 for r in records if r.get("parse_ok"))
    repaired = sum(1 for r in records if r.get("parse_ok") and r.get("repair_used"))
    failed = sum(1 for r in records if not r.get("parse_ok"))
    n_ents = [len(r["plan"].get("entities", [])) for r in records if r.get("plan")]
    n_vps = [len(r["plan"].get("verb_phrases", [])) for r in records if r.get("plan")]
    n_cons = [len(r["plan"].get("constraints", [])) for r in records if r.get("plan")]
    target_vts = Counter()
    for r in records:
        plan = r.get("plan")
        if not plan:
            continue
        for e in plan.get("entities", []):
            if e.get("is_target"):
                target_vts[e.get("value_type") or "unknown"] += 1
                break
    return {
        "n": n,
        "ok": ok,
        "repaired": repaired,
        "failed": failed,
        "validity_pct": 100 * ok / max(n, 1),
        "median_ents": pd.Series(n_ents).median() if n_ents else 0,
        "median_vps": pd.Series(n_vps).median() if n_vps else 0,
        "median_cons": pd.Series(n_cons).median() if n_cons else 0,
        "target_vts": dict(target_vts),
    }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Planner dump inspector", layout="wide")
st.title("🧭 GSW planner — dump inspector")

# Sidebar: dump dir + model(s)
dumps = _discover_dumps()
if not dumps:
    st.error(f"No planner_dump_* directories found under {_LOGS}.")
    st.stop()

dump_choice = st.sidebar.selectbox(
    "Dump directory",
    options=[str(p) for p in dumps],
    index=0,
    format_func=lambda s: Path(s).name,
)

models = _discover_models(dump_choice)
if not models:
    st.error(f"No model subfolders with q_*.json found under {dump_choice}.")
    st.stop()

compare = st.sidebar.toggle("Side-by-side (2 models)", value=False)
if compare and len(models) >= 2:
    model_a = st.sidebar.selectbox("Model A", models, index=0)
    model_b = st.sidebar.selectbox(
        "Model B", models, index=min(1, len(models) - 1)
    )
    picked_models = [model_a, model_b]
else:
    picked_models = [st.sidebar.selectbox("Model", models, index=0)]

st.sidebar.markdown("### Review")
enable_gpt_review = st.sidebar.toggle("GPT-5 plan review", value=False)
review_model = _PLAN_REVIEW_MODEL
st.sidebar.caption(
    "Uses `gpt-5`; runs one cached OpenAI review call per visible plan."
)
if enable_gpt_review and not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.warning("OPENAI_API_KEY is not set.")

# Load records for the picked models
model_records: dict[str, list[dict]] = {}
for m in picked_models:
    model_records[m] = _load_records(dump_choice, m)

# Top-of-page summary stats
st.header("Summary")
cols = st.columns(len(picked_models))
for col, m in zip(cols, picked_models):
    with col:
        summ = _summary_row(model_records[m])
        st.metric(m, f"{summ['validity_pct']:.0f}%", f"{summ['ok']}/{summ['n']} parse_ok")
        st.caption(
            f"after-repair: {summ['repaired']} · failed: {summ['failed']}"
        )
        st.caption(
            f"median ents/vps/cons: {summ['median_ents']:.0f}/"
            f"{summ['median_vps']:.0f}/{summ['median_cons']:.0f}"
        )
        if summ["target_vts"]:
            st.caption(
                "target types: " + ", ".join(f"{k}={v}" for k, v in summ["target_vts"].items())
            )

st.markdown("---")

# Sidebar filters (applied to first-model records for Q selection)
primary_records = model_records[picked_models[0]]

hop_values = sorted({r.get("num_hops", 0) for r in primary_records})
selected_hops = st.sidebar.multiselect(
    "Hops filter", hop_values, default=hop_values
)

parse_filter = st.sidebar.multiselect(
    "Parse status",
    ["ok", "ok_after_repair", "failed"],
    default=["ok", "ok_after_repair", "failed"],
)


def _matches_filter(r: dict) -> bool:
    if r.get("num_hops", 0) not in selected_hops:
        return False
    if r.get("parse_ok") and r.get("repair_used"):
        cat = "ok_after_repair"
    elif r.get("parse_ok"):
        cat = "ok"
    else:
        cat = "failed"
    return cat in parse_filter


filtered = [r for r in primary_records if _matches_filter(r)]
if not filtered:
    st.warning("No questions match the current filters.")
    st.stop()

q_options = [
    f"q{r['question_id']} · {r.get('num_hops','?')}-hop · {r.get('question','')[:80]}"
    for r in filtered
]

# Persist selected index across reruns so Prev/Next buttons work.
_session_key = f"planner_dump_idx__{dump_choice}"
if _session_key not in st.session_state or st.session_state[_session_key] >= len(q_options):
    st.session_state[_session_key] = 0

# Prev/Next buttons on sidebar.
prev_col, next_col = st.sidebar.columns(2)
if prev_col.button("⬅ Prev", use_container_width=True, disabled=st.session_state[_session_key] == 0):
    st.session_state[_session_key] = max(0, st.session_state[_session_key] - 1)
    st.rerun()
if next_col.button(
    "Next ➡",
    use_container_width=True,
    disabled=st.session_state[_session_key] >= len(q_options) - 1,
):
    st.session_state[_session_key] = min(len(q_options) - 1, st.session_state[_session_key] + 1)
    st.rerun()

q_pick = st.sidebar.selectbox(
    "Question",
    options=q_options,
    index=st.session_state[_session_key],
    key=f"{_session_key}_select",
)
pick_idx = q_options.index(q_pick)
# Sync: if the user picks via the dropdown, update the session index.
st.session_state[_session_key] = pick_idx
st.sidebar.caption(f"{pick_idx + 1} / {len(q_options)}")
picked_qid = filtered[pick_idx]["question_id"]

# Optional: quick list of failed-parse Qs as a jump pane.
failed_records = [r for r in primary_records if not r.get("parse_ok")]
if failed_records:
    with st.sidebar.expander(f"⚠ Failed parses ({len(failed_records)})"):
        for r in failed_records[:50]:
            st.caption(f"q{r['question_id']}: {r.get('parse_error','')[:60]}")

# Main detail view
st.header("Question detail")

if compare and len(picked_models) == 2:
    col_a, col_b = st.columns(2)
    rec_a = _record_by_qid(model_records[picked_models[0]], picked_qid)
    rec_b = _record_by_qid(model_records[picked_models[1]], picked_qid)
    with col_a:
        if rec_a:
            _render_record_panel(
                rec_a,
                label=f"[A: {picked_models[0]}]",
                enable_gpt_review=enable_gpt_review,
                review_model=review_model,
            )
        else:
            st.info("No record for this Q in model A.")
    with col_b:
        if rec_b:
            _render_record_panel(
                rec_b,
                label=f"[B: {picked_models[1]}]",
                enable_gpt_review=enable_gpt_review,
                review_model=review_model,
            )
        else:
            st.info("No record for this Q in model B.")
else:
    rec = _record_by_qid(model_records[picked_models[0]], picked_qid)
    if rec:
        _render_record_panel(
            rec,
            enable_gpt_review=enable_gpt_review,
            review_model=review_model,
        )
    else:
        st.error("No record for selected question.")
