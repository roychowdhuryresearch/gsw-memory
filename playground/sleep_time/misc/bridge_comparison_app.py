"""Streamlit app for comparing bridge QA pairs with original 2WikiMultiHopQA questions."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bridge QA Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRIDGE_TEST_DIR = Path("logs/bridge_test")
WIKI_PATH = Path("playground_data/2wikimultihopqa.json")
CORPUS_PATH = Path("playground_data/2wikimultihopqa_corpus.json")
GSW_BASE_PATH = Path("/mnt/SSD1/shreyas/SM_GSW/2wiki/networks")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_2wiki_questions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def discover_question_runs(bridge_dir: str) -> dict[int, list[dict]]:
    """Discover all q_* dirs and their runs. Returns {question_idx: [run_info, ...]}."""
    result = {}
    bridge_path = Path(bridge_dir)
    for q_dir in sorted(bridge_path.glob("q_*")):
        match = re.match(r"q_(\d+)", q_dir.name)
        if not match:
            continue
        q_idx = int(match.group(1))
        runs = []
        for run_dir in sorted(q_dir.glob("run_*"), reverse=True):
            results_file = run_dir / "results.json"
            if results_file.exists():
                runs.append({"path": str(results_file), "run_id": run_dir.name})
        if runs:
            result[q_idx] = runs
    return result


@st.cache_data
def load_run_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def build_title_to_corpus_idx(corpus_path: str) -> dict[str, int]:
    """Build title -> corpus index mapping from the corpus file."""
    with open(corpus_path) as f:
        corpus = json.load(f)
    return {doc["title"]: i for i, doc in enumerate(corpus)}


@st.cache_data
def load_gsw_dir(dir_path: str) -> dict | None:
    """Load and merge all GSW chunk files in a doc directory."""
    d = Path(dir_path)
    if not d.exists():
        return None
    chunk_files = sorted(d.glob("gsw_*_0.json"))
    if not chunk_files:
        return None

    merged = {"entity_nodes": [], "verb_phrase_nodes": []}
    for cf in chunk_files:
        with open(cf) as f:
            chunk = json.load(f)
        merged["entity_nodes"].extend(chunk.get("entity_nodes", []))
        merged["verb_phrase_nodes"].extend(chunk.get("verb_phrase_nodes", []))
    return merged


def load_question_gsws(
    context: list[list], title_to_idx: dict[str, int],
) -> list[dict]:
    """Load all GSW structures for a question's context documents via title lookup."""
    gsws = []
    for local_idx, (title, paragraphs) in enumerate(context):
        corpus_idx = title_to_idx.get(title)
        gsw = None
        if corpus_idx is not None:
            gsw_dir = GSW_BASE_PATH / f"doc_{corpus_idx}"
            gsw = load_gsw_dir(str(gsw_dir))
        gsws.append(
            {
                "local_idx": local_idx,
                "global_idx": corpus_idx,
                "title": title,
                "paragraphs": paragraphs,
                "gsw": gsw,
            }
        )
    return gsws


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def compute_similarity(evidences: list[list[str]], bridge: dict) -> float:
    """Score bridge relevance to original question using evidence entity/relation overlap."""
    if not evidences:
        return 0.0

    # Flatten evidence triples into tokens
    evidence_tokens = set()
    for triple in evidences:
        for part in triple:
            evidence_tokens |= _tokenize(part)

    # Combine bridge text sources
    bridge_text = " ".join(
        [
            bridge.get("question", ""),
            bridge.get("reverse_question", ""),
            bridge.get("reasoning", ""),
        ]
    )
    bridge_tokens = _tokenize(bridge_text)

    if not evidence_tokens:
        return 0.0

    overlap = evidence_tokens & bridge_tokens
    return len(overlap) / len(evidence_tokens)


def compute_entity_match(evidences: list[list[str]], bridge: dict) -> dict:
    """Detailed entity-level matching between evidence and bridge."""
    evidence_entities = set()
    evidence_relations = set()
    for triple in evidences:
        if len(triple) >= 3:
            evidence_entities.add(triple[0].lower())
            evidence_relations.add(triple[1].lower())
            evidence_entities.add(triple[2].lower())

    bridge_text = " ".join(
        [
            bridge.get("question", ""),
            bridge.get("reverse_question", ""),
            bridge.get("reasoning", ""),
        ]
    ).lower()

    matched_entities = {e for e in evidence_entities if e in bridge_text}
    matched_relations = {r for r in evidence_relations if r in bridge_text}

    return {
        "matched_entities": matched_entities,
        "total_entities": evidence_entities,
        "matched_relations": matched_relations,
        "total_relations": evidence_relations,
        "entity_coverage": len(matched_entities) / max(len(evidence_entities), 1),
        "relation_coverage": len(matched_relations) / max(len(evidence_relations), 1),
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_evidence_chain(evidences: list[list[str]]):
    """Render evidence triples as a visual chain."""
    for i, triple in enumerate(evidences):
        if len(triple) >= 3:
            st.markdown(
                f"**Hop {i + 1}**: `{triple[0]}` --*{triple[1]}*--> `{triple[2]}`"
            )
        else:
            st.markdown(f"**Hop {i + 1}**: {triple}")


def render_bridge_card(bridge: dict, similarity: float, match_info: dict, rank: int):
    """Render a single bridge QA pair."""
    conf = bridge.get("confidence", 0)
    conf_color = "green" if conf >= 0.8 else "orange" if conf >= 0.5 else "red"
    sim_color = "green" if similarity >= 0.5 else "orange" if similarity >= 0.3 else "red"

    header = f"#{rank} | Similarity: {similarity:.2f} | Confidence: {conf:.2f} | Docs: {', '.join(bridge.get('source_docs', []))}"

    with st.expander(header, expanded=rank <= 3):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Forward Question**")
            st.info(bridge.get("question", ""))
            st.caption(f"Answer: {bridge.get('answers', [])}")
        with col2:
            st.markdown("**Reverse Question**")
            st.info(bridge.get("reverse_question", ""))
            st.caption(f"Answer: {bridge.get('reverse_answers', [])}")

        st.markdown("**Reasoning**")
        st.text(bridge.get("reasoning", "N/A"))

        # Entity match details
        if match_info["total_entities"]:
            matched = match_info["matched_entities"]
            total = match_info["total_entities"]
            unmatched = total - matched
            st.markdown(
                f"**Entity coverage**: {len(matched)}/{len(total)} "
                f"({match_info['entity_coverage']:.0%}) | "
                f"**Relation coverage**: {len(match_info['matched_relations'])}/{len(match_info['total_relations'])} "
                f"({match_info['relation_coverage']:.0%})"
            )
            if matched:
                st.success(f"Matched: {', '.join(sorted(matched))}")
            if unmatched:
                st.warning(f"Missing: {', '.join(sorted(unmatched))}")


def render_gsw_doc(doc_info: dict):
    """Render a single document's GSW structure."""
    title = doc_info["title"]
    local_idx = doc_info["local_idx"]
    global_idx = doc_info["global_idx"]
    gsw = doc_info["gsw"]
    paragraphs = doc_info["paragraphs"]

    label = f"doc_{local_idx} (global: doc_{global_idx}) — {title}"
    if gsw is None:
        with st.expander(f"{label} [NO GSW]"):
            st.warning("GSW file not found for this document.")
            st.markdown("**Raw Text**")
            st.text("\n".join(paragraphs))
        return

    entities = gsw.get("entity_nodes", [])
    vps = gsw.get("verb_phrase_nodes", [])
    total_qs = sum(len(vp.get("questions", [])) for vp in vps)

    with st.expander(f"{label} [{len(entities)} entities, {len(vps)} VPs, {total_qs} QAs]"):
        tab_e, tab_vp, tab_raw = st.tabs(["Entities", "Verb Phrases & QAs", "Raw Text"])

        with tab_e:
            rows = []
            for e in entities:
                roles = e.get("roles", [])
                role_str = ", ".join(
                    f"{r['role']} ({', '.join(r.get('states', []))})" for r in roles
                )
                rows.append(
                    {"ID": e["id"], "Name": e["name"], "Roles": role_str}
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tab_vp:
            for vp in vps:
                st.markdown(f"**{vp['id']}**: *{vp['phrase']}*")
                for q in vp.get("questions", []):
                    answers_str = ", ".join(q.get("answers", []))
                    # Resolve answer entity names
                    entity_map = {e["id"]: e["name"] for e in entities}
                    answer_names = [entity_map.get(a, a) for a in q.get("answers", [])]
                    st.markdown(
                        f"  - **{q['id']}**: {q['text']}  \n"
                        f"    Answer: `{', '.join(answer_names)}`"
                    )
                st.markdown("")

        with tab_raw:
            st.text("\n".join(paragraphs))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    st.title("Bridge QA Comparison")

    # Load data
    if not WIKI_PATH.exists():
        st.error(f"2WikiMultiHopQA file not found at {WIKI_PATH}")
        return

    wiki_questions = load_2wiki_questions(str(WIKI_PATH))
    question_runs = discover_question_runs(str(BRIDGE_TEST_DIR))

    if not question_runs:
        st.error(f"No bridge test runs found in {BRIDGE_TEST_DIR}")
        return

    # ---- Sidebar ----
    st.sidebar.header("Controls")

    available_indices = sorted(question_runs.keys())
    q_labels = {
        idx: f"q_{idx}: {wiki_questions[idx]['question'][:60]}..."
        if idx < len(wiki_questions)
        else f"q_{idx}: (unknown)"
        for idx in available_indices
    }

    selected_idx = st.sidebar.selectbox(
        "Select Question",
        available_indices,
        format_func=lambda x: q_labels[x],
    )

    runs = question_runs[selected_idx]
    selected_run = st.sidebar.selectbox(
        "Select Run",
        range(len(runs)),
        format_func=lambda x: runs[x]["run_id"],
    )

    sim_threshold = st.sidebar.slider("Min Similarity", 0.0, 1.0, 0.0, 0.05)

    # Load run data
    run_data = load_run_results(runs[selected_run]["path"])
    bridges = run_data.get("bridges", [])
    summary = run_data.get("summary", {})

    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.header("Run Stats")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Bridges", summary.get("total_bridges", len(bridges)))
    col2.metric("Entities", summary.get("entities_explored", "?"))
    col1.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.2f}")
    col2.metric("Duration", f"{summary.get('', run_data.get('metadata', {}).get('duration_seconds', 0)):.0f}s")

    st.sidebar.metric("Questions Available", len(available_indices))

    # ---- Main content ----
    wiki_q = wiki_questions[selected_idx] if selected_idx < len(wiki_questions) else None

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Question Detail", "Document GSWs", "Statistics", "All Questions Overview"]
    )

    # ================================================================
    # Tab 1: Question Detail
    # ================================================================
    with tab1:
        if wiki_q:
            st.header(f"Original Question (#{selected_idx})")
            st.markdown(f"### {wiki_q['question']}")

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Answer", wiki_q["answer"])
            mcol2.metric("Type", wiki_q.get("type", "unknown"))
            mcol3.metric("Hops", len(wiki_q.get("evidences", [])))

            # Evidence chain
            evidences = wiki_q.get("evidences", [])
            if evidences:
                st.subheader("Evidence Chain")
                render_evidence_chain(evidences)

            # Supporting docs
            supporting = wiki_q.get("supporting_facts", [])
            if supporting:
                doc_titles = list(dict.fromkeys(sf[0] for sf in supporting))
                st.subheader("Supporting Documents")
                st.write(", ".join(doc_titles))

            st.markdown("---")

            # Score and sort bridges
            scored_bridges = []
            for b in bridges:
                sim = compute_similarity(evidences, b)
                match_info = compute_entity_match(evidences, b)
                scored_bridges.append((b, sim, match_info))

            scored_bridges.sort(key=lambda x: x[1], reverse=True)

            # Filter by threshold
            filtered = [(b, s, m) for b, s, m in scored_bridges if s >= sim_threshold]

            st.subheader(f"Bridge QA Pairs ({len(filtered)}/{len(bridges)} shown)")

            if not filtered:
                st.info("No bridges match the similarity threshold. Try lowering it.")
            else:
                for rank, (b, sim, match_info) in enumerate(filtered, 1):
                    render_bridge_card(b, sim, match_info, rank)
        else:
            st.error(f"Question index {selected_idx} not found in 2WikiMultiHopQA")

    # ================================================================
    # Tab 2: Document GSWs
    # ================================================================
    with tab2:
        if wiki_q:
            st.header(f"Document GSWs for Question #{selected_idx}")
            st.markdown(f"*{wiki_q['question']}*")

            context = wiki_q.get("context", [])
            title_to_idx = build_title_to_corpus_idx(str(CORPUS_PATH))
            gsw_docs = load_question_gsws(context, title_to_idx)

            # Summary metrics
            docs_with_gsw = sum(1 for d in gsw_docs if d["gsw"] is not None)
            total_entities = sum(
                len(d["gsw"].get("entity_nodes", [])) for d in gsw_docs if d["gsw"]
            )
            total_vps = sum(
                len(d["gsw"].get("verb_phrase_nodes", [])) for d in gsw_docs if d["gsw"]
            )
            total_qas = sum(
                sum(len(vp.get("questions", [])) for vp in d["gsw"].get("verb_phrase_nodes", []))
                for d in gsw_docs
                if d["gsw"]
            )

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Docs with GSW", f"{docs_with_gsw}/{len(context)}")
            mc2.metric("Total Entities", total_entities)
            mc3.metric("Total Verb Phrases", total_vps)
            mc4.metric("Total QA Pairs", total_qas)

            # Highlight supporting docs
            supporting = wiki_q.get("supporting_facts", [])
            supporting_titles = set(sf[0] for sf in supporting)

            for doc_info in gsw_docs:
                is_supporting = doc_info["title"] in supporting_titles
                if is_supporting:
                    st.markdown(f":star: **Supporting document**")
                render_gsw_doc(doc_info)
        else:
            st.error(f"Question index {selected_idx} not found")

    # ================================================================
    # Tab 3: Statistics
    # ================================================================
    with tab3:
        if not bridges:
            st.info("No bridges in this run.")
        else:
            st.header("Run Statistics")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            confidences = [b.get("confidence", 0) for b in bridges]
            m1.metric("Total Bridges", len(bridges))
            m2.metric("Avg Confidence", f"{sum(confidences) / len(confidences):.2f}")
            m3.metric("Min Confidence", f"{min(confidences):.2f}")
            m4.metric("Max Confidence", f"{max(confidences):.2f}")

            col_left, col_right = st.columns(2)

            with col_left:
                # Confidence distribution
                fig_conf = px.histogram(
                    x=confidences,
                    nbins=20,
                    title="Confidence Distribution",
                    labels={"x": "Confidence", "y": "Count"},
                )
                st.plotly_chart(fig_conf, use_container_width=True)

                # Source docs frequency
                doc_counter = Counter()
                for b in bridges:
                    for doc in b.get("source_docs", []):
                        doc_counter[doc] += 1
                if doc_counter:
                    doc_df = pd.DataFrame(
                        doc_counter.most_common(),
                        columns=["Document", "Bridge Count"],
                    )
                    fig_docs = px.bar(
                        doc_df,
                        x="Document",
                        y="Bridge Count",
                        title="Bridges per Document",
                    )
                    st.plotly_chart(fig_docs, use_container_width=True)

            with col_right:
                # Hop distribution
                hops = [b.get("hop_count", 2) for b in bridges]
                hop_counter = Counter(hops)
                fig_hops = px.pie(
                    names=[f"{k}-hop" for k in hop_counter.keys()],
                    values=list(hop_counter.values()),
                    title="Hop Distribution",
                )
                st.plotly_chart(fig_hops, use_container_width=True)

                # Document pair heatmap
                pair_counter = Counter()
                for b in bridges:
                    docs = sorted(b.get("source_docs", []))
                    if len(docs) == 2:
                        pair_counter[tuple(docs)] += 1
                if pair_counter:
                    pair_df = pd.DataFrame(
                        [(p[0], p[1], c) for p, c in pair_counter.most_common()],
                        columns=["Doc A", "Doc B", "Count"],
                    )
                    fig_pairs = px.bar(
                        pair_df,
                        x=pair_df.apply(
                            lambda r: f"{r['Doc A']} - {r['Doc B']}", axis=1
                        ),
                        y="Count",
                        title="Top Document Pairs",
                    )
                    fig_pairs.update_xaxes(title="Document Pair")
                    st.plotly_chart(fig_pairs, use_container_width=True)

            # RLM metrics
            rlm = summary.get("rlm_metrics", {})
            if rlm:
                st.subheader("RLM Pipeline Metrics")
                rlm_cols = st.columns(4)
                metrics_display = [
                    ("Root Calls", rlm.get("root_calls", 0)),
                    ("Worker Calls", rlm.get("worker_calls", 0)),
                    ("Verifier Calls", rlm.get("verifier_calls", 0)),
                    ("Edges Explored", rlm.get("edges_explored", 0)),
                    ("Edges w/ Bridges", rlm.get("edges_with_bridges", 0)),
                    ("Docs Completed", rlm.get("docs_completed", 0)),
                    ("Parallel Batches", rlm.get("parallel_batches", 0)),
                    ("Guided Retries", f"{rlm.get('guided_retry_succeeded', 0)}/{rlm.get('guided_retry_scheduled', 0)}"),
                ]
                for i, (label, val) in enumerate(metrics_display):
                    rlm_cols[i % 4].metric(label, val)

            # Token usage
            st.subheader("Token Usage")
            tok_cols = st.columns(3)
            tok_cols[0].metric("Input Tokens", f"{summary.get('input_tokens', 0):,}")
            tok_cols[1].metric("Output Tokens", f"{summary.get('output_tokens', 0):,}")
            tok_cols[2].metric("Total Tokens", f"{summary.get('tokens_used', 0):,}")

    # ================================================================
    # Tab 4: All Questions Overview
    # ================================================================
    with tab4:
        st.header("All Questions Overview")

        overview_rows = []
        for q_idx in available_indices:
            run_path = question_runs[q_idx][0]["path"]  # latest run
            try:
                rd = load_run_results(run_path)
                brs = rd.get("bridges", [])
                sm = rd.get("summary", {})
                wiki_entry = wiki_questions[q_idx] if q_idx < len(wiki_questions) else {}

                # Compute best similarity
                evidences = wiki_entry.get("evidences", [])
                best_sim = 0.0
                avg_sim = 0.0
                if brs and evidences:
                    sims = [compute_similarity(evidences, b) for b in brs]
                    best_sim = max(sims)
                    avg_sim = sum(sims) / len(sims)

                overview_rows.append(
                    {
                        "Question Idx": q_idx,
                        "Question": wiki_entry.get("question", "?"),
                        "Type": wiki_entry.get("type", "?"),
                        "Answer": wiki_entry.get("answer", "?"),
                        "Hops": len(evidences),
                        "Bridges": len(brs),
                        "Avg Confidence": sm.get("avg_confidence", 0),
                        "Best Similarity": best_sim,
                        "Avg Similarity": avg_sim,
                        "Entities Explored": sm.get("entities_explored", 0),
                    }
                )
            except Exception:
                continue

        if overview_rows:
            df = pd.DataFrame(overview_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary charts
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    df,
                    x="Question Idx",
                    y="Bridges",
                    color="Best Similarity",
                    title="Bridges per Question (colored by best similarity)",
                    hover_data=["Question", "Type"],
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(
                    df,
                    x="Bridges",
                    y="Best Similarity",
                    size="Avg Confidence",
                    color="Type",
                    hover_data=["Question Idx", "Question"],
                    title="Bridges vs Best Similarity",
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
