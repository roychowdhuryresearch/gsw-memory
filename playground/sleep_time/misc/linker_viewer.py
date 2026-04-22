"""
Streamlit viewer for LinkerPipeline GSW outputs.

Shows the raw source article side-by-side with the GSW entities, verb
phrases / links, and questions. Surfaces linker-specific signals: filler
vs real VPs, entity roles, orphans, per-doc cost (from pipeline.log or
trace files), and optional linker_trace.md rendering.

Usage:
    .venv/bin/streamlit run playground/sleep_time/misc/linker_viewer.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_DIR = REPO_ROOT / "data" / "sleep_time" / "frames"
NETWORKS_OUTPUT_DIR = DATA_DIR / "networks_output"
ARTICLES_CACHE_PATH = DATA_DIR / "articles" / "articles_cache.json"
TITLE_MAPPING_PATH = DATA_DIR / "title_to_doc_idx.json"
ARTICLES_INDIVIDUAL_DIR = DATA_DIR / "articles" / "individual"


st.set_page_config(
    page_title="Linker GSW Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_title_mapping() -> Dict[str, int]:
    if TITLE_MAPPING_PATH.exists():
        return json.loads(TITLE_MAPPING_PATH.read_text())
    return {}


@st.cache_data
def load_articles_cache() -> Dict[str, str]:
    """Load the full articles_cache.json once; the viewer looks up titles here."""
    if ARTICLES_CACHE_PATH.exists():
        return json.loads(ARTICLES_CACHE_PATH.read_text())
    return {}


@st.cache_data
def load_gsw(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _sanitize_filename(title: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", title)
    name = name.strip(". ")
    return name[:200]


def load_article_text(title: str) -> Optional[str]:
    """Try the articles_cache first, then fall back to individual .txt files."""
    cache = load_articles_cache()
    text = cache.get(title)
    if text:
        return text
    individual_path = ARTICLES_INDIVIDUAL_DIR / f"{_sanitize_filename(title)}.txt"
    if individual_path.exists():
        return individual_path.read_text()
    return None


@st.cache_data
def discover_runs() -> List[str]:
    if not NETWORKS_OUTPUT_DIR.exists():
        return []
    runs: List[str] = []
    for entry in NETWORKS_OUTPUT_DIR.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("frames_linker_") or name.startswith("frames_agentic_"):
            runs.append(name)
    runs.sort(reverse=True)  # newest first by timestamp suffix
    return runs


def discover_docs(run_dir: Path) -> List[int]:
    networks = run_dir / "networks"
    if not networks.exists():
        return []
    indices: List[int] = []
    for doc_dir in networks.iterdir():
        if not doc_dir.is_dir() or not doc_dir.name.startswith("doc_"):
            continue
        try:
            indices.append(int(doc_dir.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(indices)


@st.cache_data
def load_run_metadata(run_dir: str) -> Optional[Dict[str, Any]]:
    meta_path = Path(run_dir) / "linker_run_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return None
    # Also try the agentic metadata shape
    agentic_meta = Path(run_dir) / "agentic_run_metadata.json"
    if agentic_meta.exists():
        try:
            return json.loads(agentic_meta.read_text())
        except Exception:
            return None
    return None


@st.cache_data
def load_linker_trace(run_dir: str, doc_idx: int) -> Optional[Dict[str, Any]]:
    trace_path = Path(run_dir) / "traces" / f"doc_{doc_idx}" / "linker_trace.json"
    if trace_path.exists():
        try:
            return json.loads(trace_path.read_text())
        except Exception:
            return None
    return None


@st.cache_data
def load_linker_trace_md(run_dir: str, doc_idx: int) -> Optional[str]:
    md_path = Path(run_dir) / "traces" / f"doc_{doc_idx}" / "linker_trace.md"
    if md_path.exists():
        try:
            return md_path.read_text()
        except Exception:
            return None
    return None


@st.cache_data
def load_run_log_summary(run_dir: str) -> Dict[int, Dict[str, Any]]:
    """Grep the pipeline.log for every doc's pipeline_done line.

    Returns a dict ``{doc_idx: {entities, links, calls, prompt_tokens,
    completion_tokens, cost_usd, duration_s}}``. Older logs won't have
    the cost/token fields; those keys are simply absent.
    """
    log_path = Path(run_dir) / "pipeline.log"
    if not log_path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    pattern = re.compile(
        r"doc=(\d+)\s+pipeline_done\s+total_duration=([\d.]+)s"
        r"(?:\s+entities=(\d+))?"
        r"(?:\s+links=(\d+))?"
        r"(?:\s+calls=(\d+))?"
        r"(?:\s+prompt_tokens=(\d+))?"
        r"(?:\s+completion_tokens=(\d+))?"
        r"(?:\s+cost_usd=\$([\d.]+))?"
    )
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if not match:
                continue
            idx, dur, ents, links, calls, pt, ct, cost = match.groups()
            out[int(idx)] = {
                "duration_s": float(dur),
                "entities": int(ents) if ents else None,
                "links": int(links) if links else None,
                "calls": int(calls) if calls else None,
                "prompt_tokens": int(pt) if pt else None,
                "completion_tokens": int(ct) if ct else None,
                "cost_usd": float(cost) if cost else None,
            }
    except Exception:
        return {}
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_entity_lookup(gsw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {e["id"]: e for e in gsw.get("entity_nodes", [])}


def classify_vps(vps: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    real, filler = [], []
    for vp in vps:
        if vp.get("id", "").startswith("l_filler_"):
            filler.append(vp)
        else:
            real.append(vp)
    return real, filler


def compute_orphan_ids(gsw: Dict[str, Any]) -> List[str]:
    entity_ids = {e["id"] for e in gsw.get("entity_nodes", [])}
    referenced: set[str] = set()
    for vp in gsw.get("verb_phrase_nodes", []):
        for q in vp.get("questions", []):
            for ans in q.get("answers", []):
                if ans in entity_ids:
                    referenced.add(ans)
    return sorted(entity_ids - referenced)


# ---------------------------------------------------------------------------
# Deterministic per-document quality analysis
# ---------------------------------------------------------------------------


_PLACEHOLDER_NAMES = {
    "a", "an", "the", "generic", "unknown", "none", "null", "undefined",
    "mentioned", "listed", "concept",
}
_SLUDGE_PHRASE_PATTERNS = re.compile(
    r"^(is a|was|until|in \d+|on [\w.]+\.com|mentioned in|appears in|"
    r"listed in|included in|referenced in|covered in|discussed in|"
    r"related to|associated with|linked to|part of)\b",
    re.IGNORECASE,
)


def analyze_doc(gsw: Dict[str, Any], article_text: Optional[str]) -> Dict[str, Any]:
    """Run every deterministic quality check we have against a single GSW.

    No LLM calls. Everything is substring matching, set arithmetic, and
    regex. Returns a dict shaped for rendering in the viewer's Analysis
    tab.
    """
    entities = gsw.get("entity_nodes", [])
    vps = gsw.get("verb_phrase_nodes", [])
    entity_ids = {e["id"] for e in entities}
    entity_by_id = {e["id"]: e for e in entities}

    # ---- Structural checks (all should be zero on a healthy GSW)
    structural: Dict[str, Any] = {
        "dangling_refs": [],
        "text_prefix_answers": [],
        "none_answers": [],
        "empty_answer_questions": [],
        "circular_relations": [],
        "placeholder_entity_names": [],
        "empty_role_entities": [],
        "concept_role_entities": [],
    }

    for e in entities:
        name = (e.get("name") or "").strip()
        if (
            not name
            or len(name) <= 1
            or name.casefold() in _PLACEHOLDER_NAMES
        ):
            structural["placeholder_entity_names"].append({"id": e["id"], "name": name})
        roles = e.get("roles", []) or []
        if not roles:
            structural["empty_role_entities"].append({"id": e["id"], "name": name})
        if any((r.get("role") or "").casefold() == "concept" for r in roles):
            structural["concept_role_entities"].append({"id": e["id"], "name": name})

    total_answers = 0
    total_questions = 0
    for vp in vps:
        # Find forward/reverse questions to derive subject + object
        questions = vp.get("questions", [])
        fwd_q = next((q for q in questions if "forward" in q.get("id", "")), questions[0] if questions else None)
        rev_q = next((q for q in questions if "reverse" in q.get("id", "")), questions[1] if len(questions) > 1 else None)
        obj_id = (fwd_q or {}).get("answers", [None])[0] if fwd_q else None
        subj_id = (rev_q or {}).get("answers", [None])[0] if rev_q else None
        if subj_id and obj_id and subj_id == obj_id:
            structural["circular_relations"].append(
                {"vp_id": vp.get("id", ""), "phrase": vp.get("phrase", ""), "entity_id": subj_id}
            )

        for q in questions:
            total_questions += 1
            answers = q.get("answers", [])
            if not answers:
                structural["empty_answer_questions"].append(
                    {"vp_id": vp.get("id", ""), "question_id": q.get("id", "")}
                )
                continue
            for ans in answers:
                total_answers += 1
                if ans == "None":
                    structural["none_answers"].append(
                        {"vp_id": vp.get("id", ""), "question_id": q.get("id", "")}
                    )
                elif isinstance(ans, str) and ans.startswith("TEXT:"):
                    structural["text_prefix_answers"].append(
                        {"vp_id": vp.get("id", ""), "answer": ans}
                    )
                elif isinstance(ans, str) and ans.startswith("e") and ans not in entity_ids:
                    structural["dangling_refs"].append(
                        {"vp_id": vp.get("id", ""), "question_id": q.get("id", ""), "answer": ans}
                    )

    # ---- Entity coverage against source text
    entity_match: Dict[str, Any] = {
        "article_chars": len(article_text or ""),
        "matched": [],
        "unmatched": [],
    }
    if article_text:
        text_casefold = article_text.casefold()
        for e in entities:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            if name.casefold() in text_casefold:
                entity_match["matched"].append(e["id"])
                continue
            # Try aliases
            aliases = e.get("aliases") or []
            if any(a and a.casefold() in text_casefold for a in aliases):
                entity_match["matched"].append(e["id"])
                continue
            entity_match["unmatched"].append({"id": e["id"], "name": name})

    # ---- Role distribution
    role_counts: Dict[str, int] = {}
    for e in entities:
        for r in e.get("roles", []) or []:
            role = (r.get("role") or "").strip() or "(empty)"
            role_counts[role] = role_counts.get(role, 0) + 1
    role_distribution = sorted(role_counts.items(), key=lambda x: -x[1])

    # ---- Filler breakdown
    real_vps, filler_vps = classify_vps(vps)
    sludge_filler = []
    grounded_filler = []
    for vp in filler_vps:
        phrase = (vp.get("phrase") or "").strip()
        if _SLUDGE_PHRASE_PATTERNS.search(phrase):
            sludge_filler.append({"vp_id": vp.get("id", ""), "phrase": phrase})
        else:
            grounded_filler.append(vp)

    # ---- Answer concentration
    answer_counts: Dict[str, int] = {}
    for vp in vps:
        for q in vp.get("questions", []):
            for ans in q.get("answers", []):
                if ans in entity_ids:
                    answer_counts[ans] = answer_counts.get(ans, 0) + 1
    top_answer_share = 0
    top_answer_id = None
    top_answer_name = None
    if answer_counts and total_answers:
        top_answer_id, top_count = max(answer_counts.items(), key=lambda x: x[1])
        top_answer_share = round(100 * top_count / max(1, total_answers), 1)
        top_answer_name = entity_by_id.get(top_answer_id, {}).get("name", top_answer_id)

    # ---- Potential answer-swap heuristic for filler VPs
    # FillerAgent has shown a pattern of wiring the answer to the wrong side
    # when both subject and object are in the text. We flag filler VPs where
    # the forward question text contains the object entity's name as a
    # substring (which should be hidden in a forward question).
    suspected_swaps = []
    for vp in filler_vps:
        questions = vp.get("questions", [])
        fwd_q = next((q for q in questions if "forward" in q.get("id", "")), None)
        if not fwd_q:
            continue
        answers = fwd_q.get("answers", [])
        if not answers:
            continue
        obj_entity = entity_by_id.get(answers[0])
        if not obj_entity:
            continue
        obj_name = (obj_entity.get("name") or "").strip()
        if len(obj_name) < 3:
            continue
        fwd_text = (fwd_q.get("text") or "").casefold()
        if obj_name.casefold() in fwd_text:
            suspected_swaps.append(
                {
                    "vp_id": vp.get("id", ""),
                    "phrase": vp.get("phrase", ""),
                    "forward_q": fwd_q.get("text", ""),
                    "obj_name": obj_name,
                }
            )

    # ---- Verdict counters
    red_flag_count = sum(
        len(structural[k])
        for k in (
            "dangling_refs",
            "text_prefix_answers",
            "none_answers",
            "empty_answer_questions",
            "circular_relations",
            "placeholder_entity_names",
            "empty_role_entities",
            "concept_role_entities",
        )
    )

    return {
        "structural": structural,
        "structural_red_flag_count": red_flag_count,
        "entity_match": entity_match,
        "role_distribution": role_distribution,
        "filler_breakdown": {
            "total_vps": len(vps),
            "real": len(real_vps),
            "filler": len(filler_vps),
            "filler_pct": round(100 * len(filler_vps) / max(1, len(vps)), 1),
            "sludge_filler": sludge_filler,
            "grounded_filler_count": len(grounded_filler),
        },
        "top_answer": {
            "id": top_answer_id,
            "name": top_answer_name,
            "share_pct": top_answer_share,
        },
        "total_questions": total_questions,
        "total_answers": total_answers,
        "suspected_swaps": suspected_swaps,
    }


def title_for_doc(run_name: str, doc_idx: int) -> str:
    mapping = load_title_mapping()
    for title, idx in mapping.items():
        if idx == doc_idx:
            return title
    return f"doc_{doc_idx}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


st.sidebar.title("Linker GSW Viewer")

runs = discover_runs()
if not runs:
    st.error(f"No runs found under {NETWORKS_OUTPUT_DIR}")
    st.stop()

selected_run = st.sidebar.selectbox("Run", runs)
run_path = NETWORKS_OUTPUT_DIR / selected_run

docs = discover_docs(run_path)
if not docs:
    st.sidebar.warning("No documents in this run.")
    st.stop()

# Pair each doc index with its title for the selectbox label
doc_labels = [f"doc_{i} — {title_for_doc(selected_run, i)[:60]}" for i in docs]
selected_label = st.sidebar.selectbox("Document", doc_labels)
selected_doc_idx = docs[doc_labels.index(selected_label)]

gsw_path = run_path / "networks" / f"doc_{selected_doc_idx}" / f"gsw_{selected_doc_idx}_0.json"
gsw = load_gsw(str(gsw_path))
if gsw is None:
    st.sidebar.error(f"GSW not found: {gsw_path}")
    st.stop()

entity_nodes = gsw.get("entity_nodes", [])
vp_nodes = gsw.get("verb_phrase_nodes", [])
entity_lookup = build_entity_lookup(gsw)
real_vps, filler_vps = classify_vps(vp_nodes)
orphan_ids = compute_orphan_ids(gsw)
total_questions = sum(len(vp.get("questions", [])) for vp in vp_nodes)

doc_title = title_for_doc(selected_run, selected_doc_idx)
article_text = load_article_text(doc_title)
run_meta = load_run_metadata(str(run_path))
trace = load_linker_trace(str(run_path), selected_doc_idx)
trace_md = load_linker_trace_md(str(run_path), selected_doc_idx)
log_summary = load_run_log_summary(str(run_path))
doc_summary = log_summary.get(selected_doc_idx, {})

st.sidebar.divider()
st.sidebar.metric("Entities", len(entity_nodes))
st.sidebar.metric("VP nodes", len(vp_nodes))
st.sidebar.metric("Filler VPs", f"{len(filler_vps)} ({100 * len(filler_vps) // max(1, len(vp_nodes))}%)")
st.sidebar.metric("Orphans", len(orphan_ids))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(doc_title)
st.caption(f"`{selected_run}` · doc_{selected_doc_idx} · `{gsw_path.relative_to(NETWORKS_OUTPUT_DIR.parent.parent)}`")

metric_cols = st.columns(6)
metric_cols[0].metric("Entities", len(entity_nodes))
metric_cols[1].metric("Links", len(vp_nodes))
metric_cols[2].metric("Filler %", f"{100 * len(filler_vps) // max(1, len(vp_nodes))}%")
metric_cols[3].metric("Questions", total_questions)
dur = doc_summary.get("duration_s")
metric_cols[4].metric("Wall time", f"{dur:.0f}s" if dur else "—")
cost = doc_summary.get("cost_usd")
metric_cols[5].metric("Cost", f"${cost:.4f}" if cost is not None else "—")

if doc_summary.get("prompt_tokens") is not None:
    st.caption(
        f"Tokens — prompt: {doc_summary['prompt_tokens']:,}"
        f" · completion: {doc_summary.get('completion_tokens', 0):,}"
        f" · calls: {doc_summary.get('calls', 0):,}"
    )

st.divider()

# ---------------------------------------------------------------------------
# Side-by-side body
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Source article")
    if article_text is None:
        st.warning(
            f"Article text not found. Looked up key `{doc_title}` in "
            f"`{ARTICLES_CACHE_PATH.name}` and fallback path "
            f"`{_sanitize_filename(doc_title)}.txt`."
        )
    else:
        st.text_area(
            "Raw article",
            article_text,
            height=780,
            label_visibility="collapsed",
            disabled=True,
        )

with right_col:
    st.subheader("GSW")
    (
        tab_analysis,
        tab_ents,
        tab_links,
        tab_questions,
        tab_orphans,
        tab_cost,
        tab_raw,
    ) = st.tabs(
        ["🔎 Analysis", "Entities", "Links", "Questions", "Orphans", "Cost", "Raw JSON"]
    )

    # --- Analysis tab ---
    with tab_analysis:
        analysis = analyze_doc(gsw, article_text)

        # Top summary row
        col_a, col_b, col_c, col_d = st.columns(4)
        red_count = analysis["structural_red_flag_count"]
        col_a.metric(
            "Structural red flags",
            red_count,
            delta=None,
            delta_color="off",
        )
        matched = len(analysis["entity_match"]["matched"])
        unmatched = len(analysis["entity_match"]["unmatched"])
        total_ents = matched + unmatched
        match_pct = int(100 * matched / max(1, total_ents)) if total_ents else 0
        col_b.metric(
            "Entities in source",
            f"{matched}/{total_ents}",
            f"{match_pct}%" if total_ents else "—",
        )
        col_c.metric(
            "Filler rate",
            f"{analysis['filler_breakdown']['filler_pct']}%",
            f"{analysis['filler_breakdown']['filler']}/{analysis['filler_breakdown']['total_vps']}",
        )
        top = analysis["top_answer"]
        col_d.metric(
            "Top-1 answer share",
            f"{top['share_pct']}%" if top["share_pct"] else "—",
            top["name"] or "",
        )

        st.divider()

        # Structural red flags — should all be 0 on a healthy GSW
        st.markdown("### Structural red flags")
        if red_count == 0:
            st.success(
                "✅ All structural bars pass: no dangling refs, no `TEXT:` or "
                "`None` answers, no empty answer lists, no circular relations, "
                "no placeholder entity names, no empty roles, no `concept` "
                "drift."
            )
        else:
            struct = analysis["structural"]
            bars = [
                ("Dangling answer refs", struct["dangling_refs"]),
                ("`TEXT:` prefixed answers", struct["text_prefix_answers"]),
                ("`None` answers", struct["none_answers"]),
                ("Empty answer lists", struct["empty_answer_questions"]),
                ("Circular relations (subject == object)", struct["circular_relations"]),
                ("Placeholder entity names", struct["placeholder_entity_names"]),
                ("Empty role entities", struct["empty_role_entities"]),
                ("`concept` role drift", struct["concept_role_entities"]),
            ]
            for label, items in bars:
                if not items:
                    continue
                with st.expander(f"❌ {label} — {len(items)}"):
                    st.json(items[:30], expanded=False)
                    if len(items) > 30:
                        st.caption(f"({len(items) - 30} more not shown)")

        st.divider()

        # Entity coverage against the source article
        st.markdown("### Entity coverage against source")
        if not article_text:
            st.info(
                "Article text not available for this doc — can't verify "
                "entity-name substring matches."
            )
        else:
            if unmatched == 0:
                st.success(
                    f"✅ All {matched} entity names (or aliases) appear verbatim "
                    "in the source article."
                )
            else:
                st.warning(
                    f"⚠️ {unmatched}/{total_ents} entity names not found in "
                    "source (neither the canonical name nor any alias is a "
                    "substring of the article). These may be paraphrased "
                    "aggregations minted by the LinkAgent."
                )
                with st.expander("Show unmatched entities"):
                    rows = [
                        {"id": u["id"], "name": u["name"]}
                        for u in analysis["entity_match"]["unmatched"][:50]
                    ]
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                    if len(analysis["entity_match"]["unmatched"]) > 50:
                        st.caption(
                            f"({len(analysis['entity_match']['unmatched']) - 50} more not shown)"
                        )

        st.divider()

        # Role distribution
        st.markdown("### Role distribution")
        role_rows = [
            {"role": role, "count": count, "share": f"{100 * count / max(1, total_ents):.1f}%"}
            for role, count in analysis["role_distribution"][:25]
        ]
        st.dataframe(role_rows, use_container_width=True, hide_index=True)
        if len(analysis["role_distribution"]) > 25:
            st.caption(
                f"({len(analysis['role_distribution']) - 25} more roles not shown — "
                f"total distinct roles: {len(analysis['role_distribution'])})"
            )

        st.divider()

        # Filler breakdown
        st.markdown("### Filler VP breakdown")
        fb = analysis["filler_breakdown"]
        st.markdown(
            f"- **Real VPs** (`l_vp*`): `{fb['real']}`\n"
            f"- **Filler VPs** (`l_filler_*`): `{fb['filler']}` "
            f"({fb['filler_pct']}%)\n"
            f"- Of filler VPs: `{fb['grounded_filler_count']}` grounded, "
            f"`{len(fb['sludge_filler'])}` sludge"
        )
        if fb["filler_pct"] > 50:
            st.warning(
                f"⚠️ Filler is doing {fb['filler_pct']}% of the work here. "
                "That's a signal the `VerbPhraseListAgent` under-extracted on "
                "the first pass and the `FillerAgent` had to patch most of "
                "the graph. Filler can still be correct — this is a cost + "
                "quality efficiency flag, not a correctness bug."
            )
        if fb["sludge_filler"]:
            with st.expander(
                f"❌ {len(fb['sludge_filler'])} sludge-pattern filler labels"
            ):
                st.dataframe(
                    fb["sludge_filler"][:30],
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        # Suspected answer-swap VPs (FillerAgent subject/object heuristic)
        st.markdown("### Suspected answer-swap filler VPs")
        st.caption(
            "Heuristic: a forward question should **hide** its object entity, "
            "so the object's name should NOT appear in the question text. "
            "When it does, the FillerAgent likely wired subject and object "
            "to the wrong sides."
        )
        if not analysis["suspected_swaps"]:
            st.success("✅ No suspected answer-swap VPs detected.")
        else:
            st.warning(
                f"⚠️ {len(analysis['suspected_swaps'])} filler VPs have an "
                "object name appearing in their forward question text."
            )
            st.dataframe(
                analysis["suspected_swaps"][:30],
                use_container_width=True,
                hide_index=True,
            )

    # --- Entities tab ---
    with tab_ents:
        role_counts: Dict[str, int] = {}
        for e in entity_nodes:
            for r in e.get("roles", []):
                role_counts[r.get("role", "")] = role_counts.get(r.get("role", ""), 0) + 1
        st.caption(
            f"{len(entity_nodes)} entities · roles: "
            + ", ".join(f"{k}={v}" for k, v in sorted(role_counts.items(), key=lambda x: -x[1])[:8])
        )

        role_filter = st.multiselect(
            "Filter by role",
            options=sorted(role_counts.keys()),
            default=[],
            key="entity_role_filter",
        )

        rows = []
        for e in entity_nodes:
            roles = [r.get("role", "") for r in e.get("roles", [])]
            if role_filter and not any(r in role_filter for r in roles):
                continue
            rows.append(
                {
                    "id": e["id"],
                    "name": e.get("name", ""),
                    "roles": ", ".join(roles),
                    "aliases": ", ".join(e.get("aliases", []) if isinstance(e.get("aliases"), list) else []),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True, height=600)

    # --- Links tab ---
    with tab_links:
        filler_filter = st.radio(
            "Show",
            ["All", "Real only", "Filler only"],
            index=0,
            horizontal=True,
            key="filler_filter",
        )
        search = st.text_input("Search phrase", "", key="link_search")

        rows = []
        for vp in vp_nodes:
            is_filler = vp.get("id", "").startswith("l_filler_")
            if filler_filter == "Real only" and is_filler:
                continue
            if filler_filter == "Filler only" and not is_filler:
                continue
            phrase = vp.get("phrase", "")
            if search and search.lower() not in phrase.lower():
                continue
            # Resolve subject/object from the forward question's answers and
            # the vp's paired questions
            questions = vp.get("questions", [])
            forward_q = next((q for q in questions if "forward" in q.get("id", "")), questions[0] if questions else None)
            reverse_q = next((q for q in questions if "reverse" in q.get("id", "")), questions[1] if len(questions) > 1 else None)
            obj_id = (forward_q or {}).get("answers", [None])[0] if forward_q else None
            subj_id = (reverse_q or {}).get("answers", [None])[0] if reverse_q else None
            subj_name = entity_lookup.get(subj_id, {}).get("name", subj_id or "—")
            obj_name = entity_lookup.get(obj_id, {}).get("name", obj_id or "—")
            rows.append(
                {
                    "id": vp.get("id", ""),
                    "filler": "✓" if is_filler else "",
                    "subject": subj_name,
                    "phrase": phrase,
                    "object": obj_name,
                    "fwd question": (forward_q or {}).get("text", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True, height=600)

    # --- Questions tab ---
    with tab_questions:
        q_search = st.text_input("Search question text", "", key="question_search")
        shown = 0
        for vp in vp_nodes:
            for q in vp.get("questions", []):
                text = q.get("text", "")
                if q_search and q_search.lower() not in text.lower():
                    continue
                shown += 1
                if shown > 300:
                    break
                answer_names = [
                    entity_lookup.get(a, {}).get("name", a) for a in q.get("answers", [])
                ]
                with st.expander(text or "(empty question)"):
                    st.markdown(f"**VP**: `{vp.get('id', '')}` — {vp.get('phrase', '')}")
                    st.markdown(f"**Answers**: {', '.join(answer_names) or '—'}")
                    st.caption(f"question id: `{q.get('id', '')}`")
            if shown > 300:
                st.caption("(showing first 300 matches — refine the search)")
                break

    # --- Orphans tab ---
    with tab_orphans:
        if not orphan_ids:
            st.success(f"No orphan entities — all {len(entity_nodes)} entities appear in ≥1 link.")
        else:
            st.warning(f"{len(orphan_ids)} orphan entities (no link references them)")
            for oid in orphan_ids:
                entity = entity_lookup.get(oid, {})
                roles = ", ".join(r.get("role", "") for r in entity.get("roles", []))
                st.markdown(f"- `{oid}` **{entity.get('name', '?')}** — roles: {roles or '—'}")

    # --- Cost tab ---
    with tab_cost:
        cost_info: Optional[Dict[str, Any]] = None
        # Prefer the trace file if present, else the log summary
        if trace and isinstance(trace.get("cost"), dict):
            cost_info = trace["cost"]
        elif doc_summary.get("cost_usd") is not None:
            cost_info = {
                "model": run_meta.get("model_name", "?") if run_meta else "?",
                "calls": doc_summary.get("calls"),
                "prompt_tokens": doc_summary.get("prompt_tokens"),
                "completion_tokens": doc_summary.get("completion_tokens"),
                "total_cost_usd": doc_summary.get("cost_usd"),
                "by_stage": {},
            }

        if not cost_info:
            st.info(
                "No cost data available. This run was produced before per-doc "
                "cost tracking landed, or the pipeline.log doesn't contain a "
                "pipeline_done line for this doc."
            )
        else:
            st.metric("Total cost", f"${cost_info.get('total_cost_usd', 0):.4f}")
            st.caption(f"Model: `{cost_info.get('model', '?')}`")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Calls", cost_info.get("calls") or "—")
            col_b.metric("Prompt tokens", f"{cost_info.get('prompt_tokens', 0):,}")
            col_c.metric("Completion tokens", f"{cost_info.get('completion_tokens', 0):,}")

            if cost_info.get("by_stage"):
                st.subheader("By stage")
                stage_rows = []
                for stage, slot in cost_info["by_stage"].items():
                    stage_rows.append(
                        {
                            "stage": stage,
                            "calls": slot.get("calls", 0),
                            "prompt tokens": slot.get("prompt_tokens", 0),
                            "completion tokens": slot.get("completion_tokens", 0),
                            "$ usd": round(slot.get("cost_usd", 0.0), 4),
                        }
                    )
                st.dataframe(stage_rows, use_container_width=True, hide_index=True)

            if trace_md:
                with st.expander("Show full trace markdown"):
                    st.markdown(trace_md)

    # --- Raw JSON tab ---
    with tab_raw:
        st.json(gsw, expanded=False)
