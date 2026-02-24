"""
Pipeline Inspector -- Step-by-step diagnostic tool for the personal memory pipeline.

Interactive Streamlit dashboard that runs each pipeline step individually and
visualises intermediate results for debugging and analysis.

Usage:
streamlit run src/gsw_memory/personal_memory/playground/pipeline_inspector.py
"""

from __future__ import annotations

import copy
import glob as _glob
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — make the package importable regardless of CWD
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parents[3]  # gsw-memory/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from openai import OpenAI

from gsw_memory.memory.models import GSWStructure
from gsw_memory.memory.operator_utils.spacetime import apply_spacetime_to_gsw
from gsw_memory.memory.reconciler import Reconciler
from gsw_memory.personal_memory.chunker import TopicBoundaryChunker
from gsw_memory.personal_memory.data_ingestion.locomo import (
    Conversation,
    LoCoMoLoader,
    Session,
)
from gsw_memory.personal_memory.models import ConversationMemory, PersonMemory
from gsw_memory.personal_memory.qa_agent import PersonalMemoryQAAgent
from gsw_memory.personal_memory.reconciler import (
    ConversationReconciler,
    _LAYER3_SYSTEM,
    _LAYER3_USER,
)
from gsw_memory.prompts.operator_prompts import (
    CorefPrompts,
    ConversationalOperatorPrompts,
    SpaceTimePrompts,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_DATA_PATH = str(
    _REPO_ROOT / "data" / "personal_memory" / "locomo" / "data" / "locomo10.json"
)
DEFAULT_MODEL = "gpt-4o-mini"
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
MAX_SESSIONS = 20

STEP_NAMES = [
    "0. Data Loading",
    "1. Coref Resolution",
    "2. Topic Chunking",
    "3. GSW Extraction",
    "4. SpaceTime Linker",
    "5. Layer-1 Reconcile",
    "6. Layer-2 Reconcile",
    "7. Layer-3 Agentic",
]

STEP_DEPS: Dict[int, Set[int]] = {
    0: set(),
    1: {0},
    2: {1},
    3: {2},
    4: {3},
    5: {4},
    6: {5},
    7: {6},
}

TAB_NAMES = [
    "Data",
    "Coref",
    "Chunks",
    "GSW",
    "SpaceTime",
    "Layer-1",
    "Layer-2",
    "Layer-3",
    "QA",
    "Diagnosis",
]

# Pre-defined conversation labels and their indices in locomo10.json
KNOWN_CONVS = {
    "conv-26 (Caroline x Melanie)": 0,
    "conv-30 (Jon x Gina)": 1,
    "conv-41 (John x Maria)": 2,
    "conv-42 (Joanna x Nate)": 3,
    "conv-43 (Tim x John)": 4,
    "conv-44 (Audrey x Andrew)": 5,
    "conv-47 (James x John)": 6,
    "conv-48 (Deborah x Jolene)": 7,
    "conv-49 (Evan x Sam)": 8,
    "conv-50 (Calvin x Dave)": 9,
}

DEFAULT_CONV_LABELS = [
    "conv-26 (Caroline x Melanie)",
    "conv-41 (John x Maria)",
    "conv-43 (Tim x John)",
]


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Pipeline Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# SESSION STATE
# ============================================================================

_DEFAULTS: Dict[str, Any] = {
    "all_conversations": None,  # full list from LoCoMoLoader
    "selected_conv_labels": DEFAULT_CONV_LABELS,
    "completed_steps": set(),
    # per-step results keyed by conv_id
    "raw_texts": {},
    "resolved_texts": {},
    "session_chunks": {},
    "chunk_gsws": {},
    "post_spacetime_gsws": {},
    "session_gsws": {},
    "conversation_memories": {},
    # cross-conversation (per focal person)
    "person_memories": {},      # Dict[str, PersonMemory]
    "layer3_traces": {},        # Dict[str, List[dict]]
    "layer3_merge_pairs_all": {},  # Dict[str, list]
    # QA
    "qa_last_result": None,
    # AI quality check results
    "qa_results": {},
}


def _init_state():
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                default.copy() if isinstance(default, (dict, list, set)) else default
            )


def _can_run(step: int) -> bool:
    return STEP_DEPS[step].issubset(st.session_state.get("completed_steps", set()))


def _mark_done(step: int):
    st.session_state["completed_steps"].add(step)
    if st.session_state.get("_auto_save", False):
        _save_state_to_disk()


def _invalidate_from(step: int):
    """Remove cached results from *step* onwards."""
    keys_per_step = {
        0: ["all_conversations"],
        1: ["raw_texts", "resolved_texts"],
        2: ["session_chunks"],
        3: ["chunk_gsws"],
        4: ["post_spacetime_gsws"],
        5: ["session_gsws"],
        6: ["conversation_memories"],
        7: ["person_memories", "layer3_traces", "layer3_merge_pairs_all", "qa_last_result"],
    }
    completed = st.session_state.get("completed_steps", set())
    for s in range(step, 8):
        completed.discard(s)
        for key in keys_per_step.get(s, []):
            val = st.session_state.get(key)
            if val is not None:
                st.session_state[key] = type(val)() if isinstance(val, (dict, list, set)) else None
    st.session_state["completed_steps"] = completed
    # Clear stale quality check results
    st.session_state["qa_results"] = {}


# ============================================================================
# SAVE / LOAD
# ============================================================================

_SAVE_DIR = _REPO_ROOT / "logs" / "pipeline_inspector"

_STATE_KEYS_TO_SAVE = [
    "completed_steps",
    "raw_texts",
    "resolved_texts",
    "session_chunks",
    "chunk_gsws",
    "post_spacetime_gsws",
    "session_gsws",
    "conversation_memories",
    "person_memories",
    "layer3_traces",
    "layer3_merge_pairs_all",
]


def _serialize_state() -> dict:
    """Serialize saveable session state keys to a JSON-compatible dict."""
    out: Dict[str, Any] = {}
    for key in _STATE_KEYS_TO_SAVE:
        val = st.session_state.get(key)
        if val is None:
            out[key] = None
            continue

        if key == "completed_steps":
            out[key] = sorted(val)

        elif key in ("raw_texts", "resolved_texts", "session_chunks",
                      "layer3_traces", "layer3_merge_pairs_all"):
            # Already JSON-native (dicts of strings/lists)
            out[key] = val

        elif key in ("chunk_gsws", "post_spacetime_gsws", "session_gsws"):
            # Dict[str, List[GSWStructure]]
            out[key] = {
                cid: [gsw.model_dump() for gsw in gsw_list]
                for cid, gsw_list in val.items()
            }

        elif key == "conversation_memories":
            # Dict[str, ConversationMemory]
            out[key] = {
                cid: {
                    "conversation_id": cm.conversation_id,
                    "speaker_a": cm.speaker_a,
                    "speaker_b": cm.speaker_b,
                    "gsw": cm.gsw.model_dump(),
                    "session_gsws": [sg.model_dump() for sg in cm.session_gsws],
                }
                for cid, cm in val.items()
            }

        elif key == "person_memories":
            # Dict[str, PersonMemory]
            out[key] = {}
            for pid, pm in val.items():
                cm_dict = {}
                for cid, cm in pm.conversation_memories.items():
                    cm_dict[cid] = {
                        "conversation_id": cm.conversation_id,
                        "speaker_a": cm.speaker_a,
                        "speaker_b": cm.speaker_b,
                        "gsw": cm.gsw.model_dump(),
                        "session_gsws": [sg.model_dump() for sg in cm.session_gsws],
                    }
                out[key][pid] = {
                    "person_id": pm.person_id,
                    "conversation_memories": cm_dict,
                    "global_gsw": pm.global_gsw.model_dump() if pm.global_gsw else None,
                }
        else:
            out[key] = val

    # Also save which conversations were selected so load can restore context
    out["_selected_conv_labels"] = st.session_state.get("selected_conv_labels", [])
    return out


def _deserialize_state(data: dict) -> None:
    """Restore session state from a serialized dict."""
    for key in _STATE_KEYS_TO_SAVE:
        val = data.get(key)
        if val is None:
            st.session_state[key] = _DEFAULTS.get(key)
            if isinstance(st.session_state[key], (dict, list, set)):
                st.session_state[key] = type(st.session_state[key])()
            continue

        if key == "completed_steps":
            st.session_state[key] = set(val)

        elif key in ("raw_texts", "resolved_texts", "session_chunks",
                      "layer3_traces", "layer3_merge_pairs_all"):
            st.session_state[key] = val

        elif key in ("chunk_gsws", "post_spacetime_gsws", "session_gsws"):
            st.session_state[key] = {
                cid: [GSWStructure.model_validate(d) for d in gsw_list]
                for cid, gsw_list in val.items()
            }

        elif key == "conversation_memories":
            st.session_state[key] = {
                cid: ConversationMemory(
                    conversation_id=cm_d["conversation_id"],
                    speaker_a=cm_d["speaker_a"],
                    speaker_b=cm_d["speaker_b"],
                    gsw=GSWStructure.model_validate(cm_d["gsw"]),
                    session_gsws=[GSWStructure.model_validate(sg) for sg in cm_d["session_gsws"]],
                )
                for cid, cm_d in val.items()
            }

        elif key == "person_memories":
            pm_dict: Dict[str, PersonMemory] = {}
            for pid, pm_d in val.items():
                cm_map: Dict[str, ConversationMemory] = {}
                for cid, cm_d in pm_d["conversation_memories"].items():
                    cm_map[cid] = ConversationMemory(
                        conversation_id=cm_d["conversation_id"],
                        speaker_a=cm_d["speaker_a"],
                        speaker_b=cm_d["speaker_b"],
                        gsw=GSWStructure.model_validate(cm_d["gsw"]),
                        session_gsws=[GSWStructure.model_validate(sg) for sg in cm_d["session_gsws"]],
                    )
                pm = PersonMemory(
                    person_id=pm_d["person_id"],
                    conversation_memories=cm_map,
                )
                pm.global_gsw = (
                    GSWStructure.model_validate(pm_d["global_gsw"])
                    if pm_d.get("global_gsw") else None
                )
                pm_dict[pid] = pm
            st.session_state[key] = pm_dict
        else:
            st.session_state[key] = val

    # Restore conversation selection
    if "_selected_conv_labels" in data:
        st.session_state["selected_conv_labels"] = data["_selected_conv_labels"]


def _save_state_to_disk() -> str:
    """Serialize current state and write to logs/pipeline_inspector/. Returns file path."""
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Build filename from timestamp + conversation ids
    conv_memories = st.session_state.get("conversation_memories", {})
    if conv_memories:
        conv_labels = "_".join(sorted(conv_memories.keys()))
    else:
        labels = st.session_state.get("selected_conv_labels", [])
        # Extract short conv id like "conv-26" from label strings
        parts = []
        for lb in labels:
            part = lb.split(" ")[0] if " " in lb else lb
            parts.append(part)
        conv_labels = "_".join(parts) if parts else "empty"

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{ts}_{conv_labels}.json"
    filepath = _SAVE_DIR / filename

    payload = _serialize_state()
    filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(filepath)


def _load_state_from_disk(path: str) -> None:
    """Read JSON from *path* and restore into session state."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _deserialize_state(data)


# ============================================================================
# HELPERS
# ============================================================================

def _get_selected_conversations(num_sessions: int) -> List[Conversation]:
    """Return selected Conversation objects with truncated sessions."""
    all_convs = st.session_state["all_conversations"]
    if all_convs is None:
        return []
    labels = st.session_state.get("selected_conv_labels", [])
    result = []
    for label in labels:
        idx = KNOWN_CONVS.get(label)
        if idx is not None and idx < len(all_convs):
            conv = all_convs[idx]
            # Truncate sessions
            conv_copy = Conversation(
                sample_id=conv.sample_id,
                speaker_a=conv.speaker_a,
                speaker_b=conv.speaker_b,
                sessions=conv.sessions[:num_sessions],
                qa_pairs=conv.qa_pairs,
                event_summaries=conv.event_summaries,
                session_summaries=conv.session_summaries,
            )
            result.append(conv_copy)
    return result


def _derive_entity_speaker_ids(gsw: GSWStructure) -> None:
    for entity in gsw.entity_nodes:
        role_speakers = {r.speaker_id for r in entity.roles if r.speaker_id is not None}
        if len(role_speakers) == 1:
            entity.speaker_id = role_speakers.pop()


def _gsw_summary(gsw: GSWStructure) -> Dict[str, int]:
    return {
        "Entities": len(gsw.entity_nodes),
        "VPs": len(gsw.verb_phrase_nodes),
        "SpaceN": len(gsw.space_nodes),
        "TimeN": len(gsw.time_nodes),
        "SpaceE": len(gsw.space_edges),
        "TimeE": len(gsw.time_edges),
    }


def _speaker_color(speaker_id: Optional[str], speaker_a: str, speaker_b: str) -> str:
    if speaker_id == speaker_a:
        return "background-color: rgba(30, 100, 200, 0.3)"
    elif speaker_id == speaker_b:
        return "background-color: rgba(30, 180, 80, 0.3)"
    return ""


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def render_gsw_entity_table(
    gsw: GSWStructure,
    speaker_a: str = "",
    speaker_b: str = "",
) -> Optional[pd.DataFrame]:
    rows = []
    for e in gsw.entity_nodes:
        for r in e.roles:
            rows.append({
                "entity_id": e.id,
                "name": e.name,
                "speaker": e.speaker_id or "",
                "role": r.role,
                "states": ", ".join(r.states) if r.states else "",
                "role_speaker": r.speaker_id or "",
                "evidence": ", ".join(r.evidence_turn_ids) if r.evidence_turn_ids else "",
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)

    def _color_row(row):
        c = _speaker_color(row.get("speaker"), speaker_a, speaker_b)
        return [c] * len(row) if c else [""] * len(row)

    return df.style.apply(_color_row, axis=1)


def render_gsw_vp_table(gsw: GSWStructure) -> Optional[pd.DataFrame]:
    rows = []
    for vp in gsw.verb_phrase_nodes:
        for q in vp.questions:
            rows.append({
                "vp_id": vp.id,
                "phrase": vp.phrase,
                "question": q.text,
                "answers": ", ".join(q.answers) if q.answers else "",
                "speaker": q.speaker_id or "",
                "evidence": ", ".join(q.evidence_turn_ids) if q.evidence_turn_ids else "",
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def render_spacetime_tables(gsw: GSWStructure):
    entity_map = {e.id: e.name for e in gsw.entity_nodes}

    if gsw.space_nodes:
        space_rows = []
        for sn in gsw.space_nodes:
            linked = [
                entity_map.get(eid, eid)
                for eid, sid in gsw.space_edges
                if sid == sn.id
            ]
            space_rows.append({
                "id": sn.id,
                "name": sn.current_name,
                "linked_entities": ", ".join(linked),
            })
        st.markdown("**Space Nodes**")
        st.dataframe(pd.DataFrame(space_rows), width="stretch", hide_index=True)

    if gsw.time_nodes:
        time_rows = []
        for tn in gsw.time_nodes:
            linked = [
                entity_map.get(eid, eid)
                for eid, tid in gsw.time_edges
                if tid == tn.id
            ]
            time_rows.append({
                "id": tn.id,
                "name": tn.current_name,
                "linked_entities": ", ".join(linked),
            })
        st.markdown("**Time Nodes**")
        st.dataframe(pd.DataFrame(time_rows), width="stretch", hide_index=True)


def _to_json_str(data) -> str:
    """Convert data to a pretty-printed JSON string."""
    if isinstance(data, GSWStructure):
        return json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
    elif hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
    elif isinstance(data, (dict, list)):
        return json.dumps(data, indent=2, ensure_ascii=False)
    return str(data)


_KEY_COUNTER = 0


def _next_key(prefix: str = "ta") -> str:
    """Generate a unique key for st.text_area widgets."""
    global _KEY_COUNTER
    _KEY_COUNTER += 1
    return f"{prefix}_{_KEY_COUNTER}"


def _wrap_text(content: str, height: int = 300):
    """Render text in a wrapping, read-only text area."""
    st.text_area(" ", content, height=height, disabled=True,
                 key=_next_key(), label_visibility="collapsed")


def render_raw_json(data, label: str = "Raw JSON"):
    """Show raw data in a collapsible pretty-printed text area."""
    with st.expander(label, expanded=False):
        _wrap_text(_to_json_str(data), height=400)


def _ask_ai_quality_check(prompt: str, data: str, model_name: str, key: str):
    """Button that sends step output to LLM for quality assessment."""
    result_key = f"qa_result_{key}"
    if st.button("Ask AI for Quality Check", key=f"qa_btn_{key}"):
        with st.spinner("Analyzing..."):
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": (
                        "You are a quality auditor for a memory extraction pipeline that converts "
                        "conversations into structured semantic graphs (GSW). Be concise and specific. "
                        "Flag issues with bullet points. If quality is good, say so briefly."
                    )},
                    {"role": "user", "content": f"{prompt}\n\n---\n\n{data}"},
                ],
                temperature=0,
                max_tokens=1500,
            )
            st.session_state["qa_results"][result_key] = resp.choices[0].message.content
    if result_key in st.session_state.get("qa_results", {}):
        st.markdown(st.session_state["qa_results"][result_key])


def render_full_gsw(
    gsw: GSWStructure,
    label: str,
    speaker_a: str = "",
    speaker_b: str = "",
    expanded: bool = False,
):
    with st.expander(label, expanded=expanded):
        stats = _gsw_summary(gsw)
        stat_cols = st.columns(6)
        for col, (k, v) in zip(stat_cols, stats.items()):
            col.metric(k, v)

        left, right = st.columns(2)
        with left:
            et = render_gsw_entity_table(gsw, speaker_a, speaker_b)
            if et is not None:
                st.markdown("**Entities**")
                st.dataframe(et, width="stretch", hide_index=True)
            else:
                st.info("No entities")

            vt = render_gsw_vp_table(gsw)
            if vt is not None:
                st.markdown("**Verb Phrases**")
                st.dataframe(vt, width="stretch", hide_index=True)
            else:
                st.info("No verb phrases")

            render_spacetime_tables(gsw)

        with right:
            st.markdown("**Raw JSON**")
            _wrap_text(_to_json_str(gsw), height=500)


# ============================================================================
# STEP FUNCTIONS
# ============================================================================

def step_0_load(data_path: str):
    loader = LoCoMoLoader(data_path)
    convs = loader.load()
    st.session_state["all_conversations"] = convs
    _mark_done(0)


def step_1_coref(conversations: List[Conversation], model_name: str):
    client = OpenAI()
    raw_texts: Dict[str, List[str]] = {}
    resolved_texts: Dict[str, List[str]] = {}

    total = sum(len(c.sessions) for c in conversations)
    progress = st.progress(0, text="Running coref resolution...")
    done = 0

    for conv in conversations:
        raws = []
        resolveds = []
        for sess in conv.sessions:
            raw = sess.to_document()
            raws.append(raw)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": CorefPrompts.SYSTEM_PROMPT},
                    {"role": "user", "content": CorefPrompts.USER_PROMPT_TEMPLATE.format(text=raw)},
                ],
                temperature=0,
                max_tokens=4000,
            )
            resolveds.append(resp.choices[0].message.content)
            done += 1
            progress.progress(done / total, text=f"Coref: session {done}/{total}")
        raw_texts[conv.sample_id] = raws
        resolved_texts[conv.sample_id] = resolveds

    progress.empty()
    st.session_state["raw_texts"] = raw_texts
    st.session_state["resolved_texts"] = resolved_texts
    _mark_done(1)


def step_2_chunking(conversations: List[Conversation], model_name: str):
    chunker = TopicBoundaryChunker(model_name=model_name)
    resolved_texts = st.session_state["resolved_texts"]
    session_chunks: Dict[str, List[List[str]]] = {}

    total = sum(len(c.sessions) for c in conversations)
    progress = st.progress(0, text="Chunking sessions...")
    done = 0

    for conv in conversations:
        chunks_per_session = []
        for sess_idx, sess in enumerate(conv.sessions):
            resolved = resolved_texts[conv.sample_id][sess_idx]
            chunks = chunker.chunk_session_from_text(resolved, sess)
            chunks_per_session.append(chunks)
            done += 1
            progress.progress(done / total, text=f"Chunking: session {done}/{total}")
        session_chunks[conv.sample_id] = chunks_per_session

    progress.empty()
    st.session_state["session_chunks"] = session_chunks
    _mark_done(2)


def step_3_gsw_extraction(conversations: List[Conversation], model_name: str):
    client = OpenAI()
    gsw_schema = GSWStructure.model_json_schema()
    session_chunks = st.session_state["session_chunks"]
    chunk_gsws: Dict[str, List[GSWStructure]] = {}

    # Count total chunks
    total = sum(
        sum(len(cks) for cks in session_chunks[c.sample_id])
        for c in conversations
    )
    progress = st.progress(0, text="Extracting GSWs...")
    done = 0

    for conv in conversations:
        speaker_context = f"Speaker A: {conv.speaker_a}, Speaker B: {conv.speaker_b}"
        all_gsws = []
        for sess_idx, chunks in enumerate(session_chunks[conv.sample_id]):
            for chunk_text in chunks:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": ConversationalOperatorPrompts.SYSTEM_PROMPT},
                        {"role": "user", "content": ConversationalOperatorPrompts.USER_PROMPT_TEMPLATE.format(
                            speaker_context=speaker_context,
                            input_text=chunk_text,
                            background_context="",
                        )},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "GSWStructure", "strict": False, "schema": gsw_schema},
                    },
                    temperature=0,
                    max_tokens=4000,
                )
                gsw = GSWStructure(**json.loads(resp.choices[0].message.content))
                # Clear hallucinated spacetime
                gsw.space_nodes = []
                gsw.time_nodes = []
                gsw.space_edges = []
                gsw.time_edges = []
                gsw.similarity_edges = []
                _derive_entity_speaker_ids(gsw)
                all_gsws.append(gsw)
                done += 1
                progress.progress(done / total, text=f"GSW extraction: chunk {done}/{total}")
        chunk_gsws[conv.sample_id] = all_gsws

    progress.empty()
    st.session_state["chunk_gsws"] = chunk_gsws
    _mark_done(3)


def step_4_spacetime(conversations: List[Conversation], model_name: str):
    client = OpenAI()
    session_chunks = st.session_state["session_chunks"]
    chunk_gsws_orig = st.session_state["chunk_gsws"]

    # Deep copy so we don't mutate the pre-spacetime GSWs
    post_gsws: Dict[str, List[GSWStructure]] = {}
    for conv in conversations:
        post_gsws[conv.sample_id] = [copy.deepcopy(g) for g in chunk_gsws_orig[conv.sample_id]]

    total = sum(len(post_gsws[c.sample_id]) for c in conversations)
    progress = st.progress(0, text="Running SpaceTimeLinker...")
    done = 0

    for conv in conversations:
        flat_idx = 0
        for sess_idx, chunks in enumerate(session_chunks[conv.sample_id]):
            session_date = conv.sessions[sess_idx].date_time if sess_idx < len(conv.sessions) else ""
            session_ctx = f"Session date: {session_date}" if session_date else ""

            for chunk_idx, chunk_text in enumerate(chunks):
                gsw = post_gsws[conv.sample_id][flat_idx]
                if gsw.entity_nodes:
                    operator_output = {"entity_nodes": [e.model_dump() for e in gsw.entity_nodes]}
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SpaceTimePrompts.SYSTEM_PROMPT},
                            {"role": "user", "content": SpaceTimePrompts.USER_PROMPT_TEMPLATE.format(
                                text_chunk_content=chunk_text,
                                operator_output_json=json.dumps(operator_output, indent=2),
                                session_context=session_ctx,
                            )},
                        ],
                        temperature=0,
                        max_tokens=1000,
                    )
                    answer_text = resp.choices[0].message.content.strip()
                    if "```json" in answer_text:
                        json_content = answer_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in answer_text:
                        json_content = answer_text.split("```")[1].split("```")[0].strip()
                    else:
                        json_content = answer_text
                    try:
                        parsed = json.loads(json_content)
                        links = parsed.get("spatio_temporal_links", [])
                        chunk_id = f"s{sess_idx}_c{chunk_idx}"
                        apply_spacetime_to_gsw(gsw, links, chunk_id=chunk_id)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse SpaceTime JSON for %s chunk %d", conv.sample_id, flat_idx)

                flat_idx += 1
                done += 1
                progress.progress(done / total, text=f"SpaceTime: chunk {done}/{total}")

    progress.empty()
    st.session_state["post_spacetime_gsws"] = post_gsws
    _mark_done(4)


def step_5_layer1(conversations: List[Conversation]):
    session_chunks = st.session_state["session_chunks"]
    post_gsws = st.session_state["post_spacetime_gsws"]
    session_gsws: Dict[str, List[GSWStructure]] = {}

    for conv in conversations:
        gsw_cursor = 0
        sess_results = []
        for sess_idx, chunks in enumerate(session_chunks[conv.sample_id]):
            n = len(chunks)
            sess_chunk_gsws = post_gsws[conv.sample_id][gsw_cursor:gsw_cursor + n]
            gsw_cursor += n

            reconciler = Reconciler(matching_approach="exact")
            for chunk_idx, (chunk_text, chunk_gsw) in enumerate(zip(chunks, sess_chunk_gsws)):
                chunk_id = f"session{sess_idx}_chunk{chunk_idx}"
                gsw_copy = copy.deepcopy(chunk_gsw)
                reconciler.reconcile(new_gsw=gsw_copy, chunk_id=chunk_id, new_chunk_text=chunk_text)

            session_gsw = reconciler.global_memory
            if session_gsw is None:
                session_gsw = GSWStructure(entity_nodes=[], verb_phrase_nodes=[])

            # Stamp conversation_id
            for entity in session_gsw.entity_nodes:
                entity.conversation_id = conv.sample_id
                for role in entity.roles:
                    role.conversation_id = conv.sample_id
            for vp in session_gsw.verb_phrase_nodes:
                for q in vp.questions:
                    if q.conversation_id is None:
                        q.conversation_id = conv.sample_id

            sess_results.append(session_gsw)
        session_gsws[conv.sample_id] = sess_results

    st.session_state["session_gsws"] = session_gsws
    _mark_done(5)


def step_6_layer2(conversations: List[Conversation]):
    session_gsws = st.session_state["session_gsws"]
    conversation_memories: Dict[str, ConversationMemory] = {}

    for conv in conversations:
        conv_reconciler = ConversationReconciler()
        conversation_gsw = conv_reconciler.reconcile_sessions(
            session_gsws=session_gsws[conv.sample_id],
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
        )
        conversation_memories[conv.sample_id] = ConversationMemory(
            conversation_id=conv.sample_id,
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
            gsw=conversation_gsw,
            session_gsws=session_gsws[conv.sample_id],
        )

    st.session_state["conversation_memories"] = conversation_memories
    _mark_done(6)


def _run_layer3_for_person(
    focal_person: str,
    conversation_memories: Dict[str, ConversationMemory],
    model_name: str,
) -> Tuple[GSWStructure, list, list]:
    """Run Layer-3 agentic reconciliation for a single focal person.

    Returns (global_gsw, trace, merge_pairs).
    """
    # Collect all entities + VPs
    all_entities = []
    for cid, cm in conversation_memories.items():
        for entity in cm.gsw.entity_nodes:
            e_copy = entity.model_copy(deep=True)
            if e_copy.conversation_id is None:
                e_copy.conversation_id = cid
            all_entities.append(e_copy)

    all_vps = []
    seen_vp_ids: set = set()
    for cm in conversation_memories.values():
        for vp in cm.gsw.verb_phrase_nodes:
            if vp.id not in seen_vp_ids:
                seen_vp_ids.add(vp.id)
                all_vps.append(vp.model_copy(deep=True))

    if not all_entities:
        return GSWStructure(entity_nodes=[], verb_phrase_nodes=[]), [], []

    # Build entity summary and tool definitions
    reconciler_l3 = ConversationReconciler()
    entity_summary = reconciler_l3._build_entity_summary(all_entities)
    reconciler_l3._all_vps = all_vps

    tool_definitions = _build_tool_definitions()

    messages = [
        {"role": "system", "content": _LAYER3_SYSTEM},
        {"role": "user", "content": _LAYER3_USER.format(
            person_id=focal_person,
            entity_summary=entity_summary,
        )},
    ]

    client = OpenAI()
    trace = []
    merge_pairs = []
    finished = False
    max_iterations = 20

    status = st.status(f"Layer-3 Agent running for {focal_person}...", expanded=True)
    for iteration in range(max_iterations):
        status.write(f"**Iteration {iteration + 1}**")
        response = client.responses.create(
            model=model_name,
            input=messages,
            tools=tool_definitions,
            tool_choice="auto",
            temperature=0,
        )

        tool_calls_in_turn = []
        agent_text = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        agent_text.append(part.text)
            elif item_type == "function_call":
                tool_calls_in_turn.append(item)

        trace_entry: Dict[str, Any] = {
            "iteration": iteration + 1,
            "agent_text": "\n".join(agent_text),
            "tool_calls": [],
        }

        if agent_text:
            status.write("\n".join(agent_text))

        if not tool_calls_in_turn:
            trace_entry["finished_no_tools"] = True
            trace.append(trace_entry)
            break

        messages.extend(response.output)

        tool_results = []
        for tc in tool_calls_in_turn:
            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            result = reconciler_l3._dispatch_tool(tc.name, args, all_entities)
            status.write(f"`{tc.name}({json.dumps(args)})`")

            if tc.name == "finish_reconciliation":
                merge_pairs = args.get("merge_pairs", [])
                finished = True

            trace_entry["tool_calls"].append({
                "name": tc.name,
                "args": args,
                "result": result,
            })
            tool_results.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": json.dumps(result),
            })

        messages.extend(tool_results)
        trace.append(trace_entry)
        if finished:
            break

    status.update(label=f"Layer-3 for {focal_person} ({iteration + 1} iterations)", state="complete")

    # Apply merges
    merged_entities = reconciler_l3._apply_merges(all_entities, merge_pairs, all_vps)

    # Collect spacetime
    all_space_nodes, all_time_nodes = [], []
    all_space_edges, all_time_edges = [], []
    for cm in conversation_memories.values():
        all_space_nodes.extend(cm.gsw.space_nodes)
        all_time_nodes.extend(cm.gsw.time_nodes)
        all_space_edges.extend(cm.gsw.space_edges)
        all_time_edges.extend(cm.gsw.time_edges)

    entity_id_remap = getattr(reconciler_l3, "_entity_id_remap", {})
    if entity_id_remap:
        all_space_edges = [[entity_id_remap.get(eid, eid), sid] for eid, sid in all_space_edges]
        all_time_edges = [[entity_id_remap.get(eid, eid), tid] for eid, tid in all_time_edges]

    global_gsw = GSWStructure(
        entity_nodes=merged_entities,
        verb_phrase_nodes=reconciler_l3._all_vps,
        space_nodes=all_space_nodes,
        time_nodes=all_time_nodes,
        space_edges=all_space_edges,
        time_edges=all_time_edges,
    )

    return global_gsw, trace, merge_pairs


def step_7_layer3(model_name: str):
    """Run Layer-3 agentic reconciliation for ALL unique speakers."""
    conversation_memories = st.session_state["conversation_memories"]
    if not conversation_memories:
        return

    # Collect unique speakers
    speakers: set = set()
    for cm in conversation_memories.values():
        speakers.add(cm.speaker_a)
        speakers.add(cm.speaker_b)

    person_memories: Dict[str, PersonMemory] = {}
    all_traces: Dict[str, list] = {}
    all_merge_pairs: Dict[str, list] = {}

    for speaker in sorted(speakers):
        global_gsw, trace, merge_pairs = _run_layer3_for_person(
            speaker, conversation_memories, model_name,
        )
        pm = PersonMemory(person_id=speaker, conversation_memories=dict(conversation_memories))
        pm.global_gsw = global_gsw
        person_memories[speaker] = pm
        all_traces[speaker] = trace
        all_merge_pairs[speaker] = merge_pairs

    st.session_state["person_memories"] = person_memories
    st.session_state["layer3_traces"] = all_traces
    st.session_state["layer3_merge_pairs_all"] = all_merge_pairs
    _mark_done(7)


def _build_tool_definitions() -> list:
    """Return the 9 tool definitions for Layer-3 agent (Responses API format)."""
    return [
        {
            "type": "function",
            "name": "get_entity_timeline",
            "description": "Get all roles, states, and associated verb phrase questions for a specific entity by index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_index": {"type": "integer", "description": "0-based entity index from the summary"},
                },
                "required": ["entity_index"],
            },
        },
        {
            "type": "function",
            "name": "compare_entities",
            "description": "Compare two entities by index to assess if they refer to the same real-world entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index_a": {"type": "integer"},
                    "index_b": {"type": "integer"},
                },
                "required": ["index_a", "index_b"],
            },
        },
        {
            "type": "function",
            "name": "detect_contradictions",
            "description": "Detect contradictory states between two entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index_a": {"type": "integer"},
                    "index_b": {"type": "integer"},
                },
                "required": ["index_a", "index_b"],
            },
        },
        {
            "type": "function",
            "name": "get_unanswered_questions",
            "description": "List all unanswered (None) questions for an entity, with cross-references from other VPs with same phrase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_index": {"type": "integer"},
                },
                "required": ["entity_index"],
            },
        },
        {
            "type": "function",
            "name": "get_duplicate_verb_phrases",
            "description": "List groups of VPs with the same phrase text, showing each VP's questions and answers side-by-side.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": "get_reconciliation_summary",
            "description": "Get current reconciliation state: entity count, unanswered questions, duplicate VPs.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": "resolve_question",
            "description": "Set the answer for a specific question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vp_id": {"type": "string"},
                    "question_id": {"type": "string"},
                    "new_answers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["vp_id", "question_id", "new_answers"],
            },
        },
        {
            "type": "function",
            "name": "merge_verb_phrases",
            "description": "Merge two verb phrases that represent the same event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keep_vp_id": {"type": "string"},
                    "discard_vp_id": {"type": "string"},
                },
                "required": ["keep_vp_id", "discard_vp_id"],
            },
        },
        {
            "type": "function",
            "name": "finish_reconciliation",
            "description": "Complete reconciliation with a list of entity merge pairs. Call this as the final step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merge_pairs": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    },
                },
                "required": ["merge_pairs"],
            },
        },
    ]


# ============================================================================
# TAB RENDERERS
# ============================================================================

def render_data_tab(data_path: str, num_sessions: int):
    st.header("Data Loading")
    if st.button("Load Data", disabled=False, type="primary"):
        with st.spinner("Loading LoCoMo data..."):
            step_0_load(data_path)
        st.rerun()

    all_convs = st.session_state.get("all_conversations")
    if all_convs is None:
        st.info("Click 'Load Data' to start.")
        return

    rows = []
    for i, conv in enumerate(all_convs):
        rows.append({
            "Index": i,
            "ID": conv.sample_id,
            "Speaker A": conv.speaker_a,
            "Speaker B": conv.speaker_b,
            "Sessions": len(conv.sessions),
            "QA Pairs": len(conv.qa_pairs),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Show selected conversations detail
    selected = _get_selected_conversations(num_sessions)
    if selected:
        st.subheader(f"Selected: {len(selected)} conversations, {num_sessions} sessions each")
        for conv in selected:
            with st.expander(f"{conv.sample_id}: {conv.speaker_a} x {conv.speaker_b}"):
                for sess in conv.sessions:
                    left, right = st.columns([2, 1])
                    with left:
                        st.markdown(f"**Session {sess.session_id}** ({sess.date_time}) -- {len(sess.turns)} turns")
                    with right:
                        with st.popover("Raw text"):
                            _wrap_text(sess.to_document(), height=400)


def render_coref_tab(conversations: List[Conversation], model_name: str):
    st.header("Step 1: Coreference Resolution")
    can = _can_run(1)
    if st.button("Run Coref", disabled=not can, type="primary"):
        _invalidate_from(1)
        step_1_coref(conversations, model_name)
        st.rerun()

    if 1 not in st.session_state.get("completed_steps", set()):
        st.info("Run Step 0 (Data) first." if not can else "Click 'Run Coref' to start.")
        return

    raw_texts = st.session_state["raw_texts"]
    resolved_texts = st.session_state["resolved_texts"]

    conv_ids = list(raw_texts.keys())
    selected_conv = st.selectbox("Conversation", conv_ids, key="coref_conv")
    if not selected_conv:
        return

    raws = raw_texts[selected_conv]
    resolveds = resolved_texts[selected_conv]

    for sess_idx, (raw, resolved) in enumerate(zip(raws, resolveds)):
        with st.expander(f"Session {sess_idx + 1}", expanded=sess_idx == 0):
            orig_lines = raw.splitlines()
            res_lines = resolved.splitlines()
            changed = sum(1 for o, r in zip(orig_lines, res_lines) if o.strip() != r.strip())
            changed += abs(len(orig_lines) - len(res_lines))
            st.caption(f"{changed}/{max(len(orig_lines), len(res_lines))} lines changed")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                _wrap_text(raw, height=400)
            with col2:
                st.markdown("**Resolved**")
                _wrap_text(resolved, height=400)

    # Quality check
    all_pairs = "\n\n".join(
        f"--- Session {i+1} ---\nORIGINAL:\n{r}\n\nRESOLVED:\n{d}"
        for i, (r, d) in enumerate(zip(raws, resolveds))
    )
    _ask_ai_quality_check(
        "Check the coreference resolution quality. Are pronouns resolved correctly? "
        "Any missed or incorrect resolutions? Is text fluency preserved?",
        all_pairs, model_name, f"coref_{selected_conv}",
    )


def render_chunks_tab(conversations: List[Conversation], model_name: str):
    st.header("Step 2: Topic Chunking")
    can = _can_run(2)
    if st.button("Run Chunking", disabled=not can, type="primary"):
        _invalidate_from(2)
        step_2_chunking(conversations, model_name)
        st.rerun()

    if 2 not in st.session_state.get("completed_steps", set()):
        st.info("Run previous steps first." if not can else "Click 'Run Chunking' to start.")
        return

    session_chunks = st.session_state["session_chunks"]
    conv_ids = list(session_chunks.keys())
    selected_conv = st.selectbox("Conversation", conv_ids, key="chunks_conv")
    if not selected_conv:
        return

    for sess_idx, chunks in enumerate(session_chunks[selected_conv]):
        st.subheader(f"Session {sess_idx + 1}: {len(chunks)} chunk(s)")
        for ci, chunk in enumerate(chunks):
            n_lines = len(chunk.splitlines())
            with st.expander(f"Chunk {ci} ({n_lines} lines, {len(chunk)} chars)"):
                left, right = st.columns(2)
                with left:
                    _wrap_text(chunk, height=300)
                with right:
                    st.markdown("**Raw**")
                    _wrap_text(json.dumps({"lines": n_lines, "chars": len(chunk), "text": chunk}, indent=2, ensure_ascii=False), height=300)

    # Quality check
    all_chunks = "\n\n".join(
        f"--- Session {si+1}, Chunk {ci} ---\n{c}"
        for si, chunks in enumerate(session_chunks[selected_conv])
        for ci, c in enumerate(chunks)
    )
    _ask_ai_quality_check(
        "Check chunk boundary quality. Are boundaries at semantically meaningful points? "
        "Does each chunk cover a coherent single topic? Are turn boundaries respected?",
        all_chunks, model_name, f"chunks_{selected_conv}",
    )


def render_gsw_tab(conversations: List[Conversation], model_name: str):
    st.header("Step 3: GSW Extraction")
    can = _can_run(3)
    if st.button("Run GSW Extraction", disabled=not can, type="primary"):
        _invalidate_from(3)
        step_3_gsw_extraction(conversations, model_name)
        st.rerun()

    if 3 not in st.session_state.get("completed_steps", set()):
        st.info("Run previous steps first." if not can else "Click 'Run GSW Extraction' to start.")
        return

    chunk_gsws = st.session_state["chunk_gsws"]
    session_chunks = st.session_state["session_chunks"]
    conv_ids = list(chunk_gsws.keys())
    selected_conv = st.selectbox("Conversation", conv_ids, key="gsw_conv")
    if not selected_conv:
        return

    conv = next((c for c in conversations if c.sample_id == selected_conv), None)
    speaker_a = conv.speaker_a if conv else ""
    speaker_b = conv.speaker_b if conv else ""

    gsws = chunk_gsws[selected_conv]
    flat_idx = 0
    for sess_idx, chunks in enumerate(session_chunks[selected_conv]):
        for ci in range(len(chunks)):
            if flat_idx < len(gsws):
                gsw = gsws[flat_idx]
                chunk_text = chunks[ci]
                label = (
                    f"S{sess_idx + 1} C{ci}: "
                    f"{len(gsw.entity_nodes)} entities, {len(gsw.verb_phrase_nodes)} VPs"
                )
                with st.expander(label, expanded=False):
                    stats = _gsw_summary(gsw)
                    stat_cols = st.columns(6)
                    for col, (k, v) in zip(stat_cols, stats.items()):
                        col.metric(k, v)

                    col_chunk, col_tables, col_json = st.columns(3)
                    with col_chunk:
                        st.markdown("**Source Chunk**")
                        _wrap_text(chunk_text, height=400)
                    with col_tables:
                        et = render_gsw_entity_table(gsw, speaker_a, speaker_b)
                        if et is not None:
                            st.markdown("**Entities**")
                            st.dataframe(et, width="stretch", hide_index=True)
                        else:
                            st.info("No entities")

                        vt = render_gsw_vp_table(gsw)
                        if vt is not None:
                            st.markdown("**Verb Phrases**")
                            st.dataframe(vt, width="stretch", hide_index=True)
                        else:
                            st.info("No verb phrases")

                        render_spacetime_tables(gsw)
                    with col_json:
                        st.markdown("**Raw JSON**")
                        _wrap_text(_to_json_str(gsw), height=500)
            flat_idx += 1

    # Quality check — send all chunk+GSW pairs for selected conversation
    qa_pairs = []
    flat_idx = 0
    for sess_idx, chunks in enumerate(session_chunks[selected_conv]):
        for ci in range(len(chunks)):
            if flat_idx < len(gsws):
                qa_pairs.append(
                    f"--- S{sess_idx+1} C{ci} ---\nCHUNK:\n{chunks[ci]}\n\nGSW:\n{_to_json_str(gsws[flat_idx])}"
                )
            flat_idx += 1
    _ask_ai_quality_check(
        "Check GSW extraction quality. Does each GSW capture all entities and relationships "
        "from its source chunk? Any hallucinated entities or missing information?",
        "\n\n".join(qa_pairs), model_name, f"gsw_{selected_conv}",
    )


def render_spacetime_tab(conversations: List[Conversation], model_name: str):
    st.header("Step 4: SpaceTime Linker")
    can = _can_run(4)
    if st.button("Run SpaceTime", disabled=not can, type="primary"):
        _invalidate_from(4)
        step_4_spacetime(conversations, model_name)
        st.rerun()

    if 4 not in st.session_state.get("completed_steps", set()):
        st.info("Run previous steps first." if not can else "Click 'Run SpaceTime' to start.")
        return

    post_gsws = st.session_state["post_spacetime_gsws"]
    session_chunks = st.session_state["session_chunks"]
    conv_ids = list(post_gsws.keys())
    selected_conv = st.selectbox("Conversation", conv_ids, key="st_conv")
    if not selected_conv:
        return

    conv = next((c for c in conversations if c.sample_id == selected_conv), None)
    speaker_a = conv.speaker_a if conv else ""
    speaker_b = conv.speaker_b if conv else ""

    gsws = post_gsws[selected_conv]
    flat_idx = 0
    for sess_idx, chunks in enumerate(session_chunks[selected_conv]):
        for ci in range(len(chunks)):
            if flat_idx < len(gsws):
                gsw = gsws[flat_idx]
                label = (
                    f"S{sess_idx + 1} C{ci}: "
                    f"{len(gsw.entity_nodes)} ent, "
                    f"{len(gsw.space_nodes)} space, {len(gsw.time_nodes)} time"
                )
                render_full_gsw(gsw, label, speaker_a, speaker_b)
            flat_idx += 1

    # Quality check
    st_summaries = "\n\n".join(
        f"--- S{si+1} C{ci} ---\n{_to_json_str(gsws[fi])}"
        for si, cks in enumerate(session_chunks[selected_conv])
        for ci, fi in enumerate(range(
            sum(len(session_chunks[selected_conv][s]) for s in range(si)),
            sum(len(session_chunks[selected_conv][s]) for s in range(si)) + len(cks),
        ))
        if fi < len(gsws)
    )
    _ask_ai_quality_check(
        "Check space/time extraction quality. Are locations and dates extracted accurately? "
        "Are entity-to-space and entity-to-time linkages correct? Any false links?",
        st_summaries, model_name, f"spacetime_{selected_conv}",
    )


def render_layer1_tab(conversations: List[Conversation], model_name: str):
    st.header("Step 5: Layer-1 Reconciliation (per session)")
    can = _can_run(5)
    if st.button("Run Layer-1", disabled=not can, type="primary"):
        _invalidate_from(5)
        step_5_layer1(conversations)
        st.rerun()

    if 5 not in st.session_state.get("completed_steps", set()):
        st.info("Run previous steps first." if not can else "Click 'Run Layer-1' to start.")
        return

    session_gsws = st.session_state["session_gsws"]
    conv_ids = list(session_gsws.keys())
    selected_conv = st.selectbox("Conversation", conv_ids, key="l1_conv")
    if not selected_conv:
        return

    conv = next((c for c in conversations if c.sample_id == selected_conv), None)
    speaker_a = conv.speaker_a if conv else ""
    speaker_b = conv.speaker_b if conv else ""

    # Summary table
    rows = []
    for sess_idx, gsw in enumerate(session_gsws[selected_conv]):
        row = {"Session": sess_idx + 1}
        row.update(_gsw_summary(gsw))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Detailed views
    for sess_idx, gsw in enumerate(session_gsws[selected_conv]):
        render_full_gsw(gsw, f"Session {sess_idx + 1}", speaker_a, speaker_b)

    # Quality check
    all_session_gsws = "\n\n".join(
        f"--- Session {i+1} ---\n{_to_json_str(g)}"
        for i, g in enumerate(session_gsws[selected_conv])
    )
    _ask_ai_quality_check(
        "Check Layer-1 within-session reconciliation quality. Are duplicate entities properly merged? "
        "Are roles consolidated correctly? Any incorrect merges or missed duplicates?",
        all_session_gsws, model_name, f"layer1_{selected_conv}",
    )


def render_layer2_tab(conversations: List[Conversation], model_name: str):
    st.header("Step 6: Layer-2 Reconciliation (cross-session)")
    can = _can_run(6)
    if st.button("Run Layer-2", disabled=not can, type="primary"):
        _invalidate_from(6)
        step_6_layer2(conversations)
        st.rerun()

    if 6 not in st.session_state.get("completed_steps", set()):
        st.info("Run previous steps first." if not can else "Click 'Run Layer-2' to start.")
        return

    conv_memories = st.session_state["conversation_memories"]
    session_gsws = st.session_state["session_gsws"]
    conv_ids = list(conv_memories.keys())
    selected_conv = st.selectbox("Conversation", conv_ids, key="l2_conv")
    if not selected_conv:
        return

    cm = conv_memories[selected_conv]
    gsw = cm.gsw

    # Before/after comparison
    l1_entities = sum(len(sg.entity_nodes) for sg in session_gsws[selected_conv])
    l1_vps = sum(len(sg.verb_phrase_nodes) for sg in session_gsws[selected_conv])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("L1 Entities", l1_entities)
    col2.metric("L2 Entities", len(gsw.entity_nodes), delta=len(gsw.entity_nodes) - l1_entities)
    col3.metric("L1 VPs", l1_vps)
    col4.metric("L2 VPs", len(gsw.verb_phrase_nodes), delta=len(gsw.verb_phrase_nodes) - l1_vps)

    render_full_gsw(gsw, f"{selected_conv} Layer-2", cm.speaker_a, cm.speaker_b, expanded=True)

    # Quality check
    l2_data = (
        f"L1 entities: {l1_entities}, L2 entities: {len(gsw.entity_nodes)}, "
        f"L1 VPs: {l1_vps}, L2 VPs: {len(gsw.verb_phrase_nodes)}\n\n"
        f"Layer-2 GSW:\n{_to_json_str(gsw)}"
    )
    _ask_ai_quality_check(
        "Check Layer-2 cross-session reconciliation quality. Are entities correctly matched "
        "across sessions? Is speaker filtering working properly? Any incorrect merges?",
        l2_data, model_name, f"layer2_{selected_conv}",
    )


def render_layer3_tab(model_name: str):
    st.header("Step 7: Layer-3 Agentic Reconciliation")
    can = _can_run(7)
    if st.button("Run Layer-3 (all speakers)", disabled=not can, type="primary"):
        _invalidate_from(7)
        step_7_layer3(model_name)
        st.rerun()

    if 7 not in st.session_state.get("completed_steps", set()):
        st.info("Run previous steps first." if not can else "Click 'Run Layer-3' to start.")
        return

    person_memories = st.session_state.get("person_memories", {})
    all_traces = st.session_state.get("layer3_traces", {})
    all_merge_pairs = st.session_state.get("layer3_merge_pairs_all", {})
    conv_memories = st.session_state["conversation_memories"]

    if not person_memories:
        return

    total_l2 = sum(len(cm.gsw.entity_nodes) for cm in conv_memories.values())

    # Per-speaker results
    selected_speaker = st.selectbox(
        "Focal Person", list(person_memories.keys()), key="l3_speaker",
    )
    if not selected_speaker:
        return

    pm = person_memories[selected_speaker]
    global_gsw = pm.global_gsw
    trace = all_traces.get(selected_speaker, [])
    merge_pairs = all_merge_pairs.get(selected_speaker, [])

    if global_gsw is None:
        st.info("No GSW for this speaker.")
        return

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("L2 Entities", total_l2)
    col2.metric("L3 Entities", len(global_gsw.entity_nodes), delta=len(global_gsw.entity_nodes) - total_l2)
    col3.metric("Iterations", len(trace))
    col4.metric("Merge Pairs", len(merge_pairs))

    if merge_pairs:
        st.markdown(f"**Merge pairs:** {merge_pairs}")

    # Agent trace
    st.subheader("Agent Trace")
    for entry in trace:
        with st.expander(f"Iteration {entry['iteration']} ({len(entry.get('tool_calls', []))} tool calls)"):
            if entry.get("agent_text"):
                st.markdown(entry["agent_text"])
            for tc in entry.get("tool_calls", []):
                _wrap_text(f"{tc['name']}({json.dumps(tc['args'])})", height=68)
                with st.expander("Result", expanded=False):
                    _wrap_text(json.dumps(tc["result"], indent=2, ensure_ascii=False), height=200)

    # Full GSW view
    render_full_gsw(global_gsw, f"Layer-3 GSW — {selected_speaker}", expanded=True)

    # Quality check
    trace_summary = json.dumps(merge_pairs, indent=2) if merge_pairs else "No merges"
    l3_data = (
        f"Focal: {selected_speaker}\n"
        f"L2 total entities: {total_l2}, L3 entities: {len(global_gsw.entity_nodes)}\n"
        f"Merge pairs: {trace_summary}\n\n"
        f"Layer-3 GSW:\n{_to_json_str(global_gsw)}"
    )
    _ask_ai_quality_check(
        "Check Layer-3 agentic cross-conversation reconciliation quality. "
        "Are the agent's merge decisions sound? Any contradictions missed or incorrect merges?",
        l3_data, model_name, f"layer3_{selected_speaker}",
    )


def render_qa_tab(model_name: str):
    st.header("Question Answering")

    person_memories = st.session_state.get("person_memories", {})
    if not person_memories:
        st.info("Run Layer-3 first to build person memories.")
        return

    focal = st.selectbox("Focal Person", list(person_memories.keys()), key="qa_focal")
    if not focal:
        return

    person_memory = person_memories[focal]

    question = st.text_input("Ask a question about this person's memory", key="qa_question")
    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            agent = PersonalMemoryQAAgent(model_name=model_name)
            result = agent.answer(question, person_memory)
            st.session_state["qa_last_result"] = result

    result = st.session_state.get("qa_last_result")
    if result:
        st.divider()
        if result.get("abstain"):
            st.warning("Speaker mismatch — answer withheld (adversarial question detected)")
        st.markdown(f"**Answer:** {result.get('answer', '')}")
        if result.get("reasoning"):
            with st.expander("Reasoning", expanded=False):
                st.markdown(result["reasoning"])
        cols = st.columns(3)
        cols[0].metric("Confidence", result.get("confidence", ""))
        cols[1].metric("Speaker", result.get("speaker_id") or "N/A")
        evidence = result.get("evidence_turn_ids", [])
        cols[2].metric("Evidence Turns", ", ".join(evidence) if evidence else "N/A")

        # Agent trace
        trace = result.get("trace", [])
        if trace:
            st.subheader("Agent Trace")
            tcols = st.columns(2)
            tcols[0].metric("Iterations", len(trace))
            tcols[1].metric("Tool Calls", sum(len(e.get("tool_calls", [])) for e in trace))

            for entry in trace:
                with st.expander(f"Iteration {entry['iteration']} ({len(entry.get('tool_calls', []))} tool calls)"):
                    if entry.get("agent_text"):
                        st.markdown(entry["agent_text"])
                    for tc in entry.get("tool_calls", []):
                        _wrap_text(f"{tc['name']}({json.dumps(tc['args'])})", height=68)
                        with st.expander("Result", expanded=False):
                            _wrap_text(json.dumps(tc["result"], indent=2, ensure_ascii=False), height=200)


def render_diagnosis_tab(conversations: List[Conversation], model_name: str):
    st.header("Diagnosis Summary")
    completed = st.session_state.get("completed_steps", set())

    if not ({4, 5, 6}.issubset(completed)):
        st.info("Run at least through Layer-2 to see the diagnosis summary.")
        return

    session_chunks = st.session_state["session_chunks"]
    post_gsws = st.session_state["post_spacetime_gsws"]
    session_gsws = st.session_state["session_gsws"]
    conv_memories = st.session_state["conversation_memories"]
    person_memories = st.session_state.get("person_memories", {})

    for conv in conversations:
        cid = conv.sample_id
        if cid not in conv_memories:
            continue
        cm = conv_memories[cid]

        st.subheader(f"{cid} ({conv.speaker_a} x {conv.speaker_b})")

        rows = []
        # Per-chunk (post-SpaceTime)
        flat_idx = 0
        for sess_idx, chunks in enumerate(session_chunks.get(cid, [])):
            for ci in range(len(chunks)):
                if flat_idx < len(post_gsws.get(cid, [])):
                    g = post_gsws[cid][flat_idx]
                    row = {"Stage": f"S{sess_idx + 1} C{ci} (post-ST)"}
                    row.update(_gsw_summary(g))
                    rows.append(row)
                flat_idx += 1

        # Per-session (Layer-1)
        for sess_idx, sg in enumerate(session_gsws.get(cid, [])):
            row = {"Stage": f"Session {sess_idx + 1} (Layer-1)"}
            row.update(_gsw_summary(sg))
            rows.append(row)

        # Conversation (Layer-2)
        row = {"Stage": f"{cid} (Layer-2)"}
        row.update(_gsw_summary(cm.gsw))
        rows.append(row)

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Cross-conversation summary (per focal person)
    if 7 in completed and person_memories:
        st.subheader("Cross-Conversation (Layer-3)")
        total_l2_entities = sum(len(cm.gsw.entity_nodes) for cm in conv_memories.values())
        total_l2_vps = sum(len(cm.gsw.verb_phrase_nodes) for cm in conv_memories.values())
        rows = [{"Stage": "Layer-2 total", "Entities": total_l2_entities, "VPs": total_l2_vps}]
        for speaker, pm in person_memories.items():
            if pm.global_gsw is not None:
                rows.append({"Stage": f"L3 — {speaker}", **_gsw_summary(pm.global_gsw)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Dangling edge detection
    st.subheader("Bug Detection")
    dangling = 0
    gsws_to_check = [cm.gsw for cm in conv_memories.values()]
    for pm in person_memories.values():
        if pm.global_gsw is not None:
            gsws_to_check.append(pm.global_gsw)

    for gsw in gsws_to_check:
        entity_ids = {e.id for e in gsw.entity_nodes}
        space_ids = {s.id for s in gsw.space_nodes}
        time_ids = {t.id for t in gsw.time_nodes}
        for eid, sid in gsw.space_edges:
            if eid not in entity_ids or sid not in space_ids:
                dangling += 1
        for eid, tid in gsw.time_edges:
            if eid not in entity_ids or tid not in time_ids:
                dangling += 1

    if dangling == 0:
        st.success(f"Dangling spacetime edges: {dangling}")
    else:
        st.error(f"Dangling spacetime edges: {dangling}")

    # Raw data dump of all GSWs
    all_raw = {}
    for cid, cm in conv_memories.items():
        all_raw[cid] = cm.gsw.model_dump()
    for speaker, pm in person_memories.items():
        if pm.global_gsw is not None:
            all_raw[f"global_layer3_{speaker}"] = pm.global_gsw.model_dump()
    render_raw_json(all_raw, "Raw All GSWs (Layer-2 + Layer-3)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    _init_state()

    st.title("Personal Memory Pipeline Inspector")
    st.caption("Step-by-step diagnostic tool for the LoCoMo personal memory pipeline")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Configuration")
        data_path = st.text_input("LoCoMo data path", value=DEFAULT_DATA_PATH)
        model_name = st.selectbox("Model", MODELS, index=0)
        num_sessions = st.slider("Sessions per conversation", 1, MAX_SESSIONS, 2)

        # Conversation selection (always visible so user picks before loading)
        selected = st.multiselect(
            "Conversations",
            list(KNOWN_CONVS.keys()),
            default=st.session_state.get("selected_conv_labels", DEFAULT_CONV_LABELS),
            key="conv_selector",
        )
        st.session_state["selected_conv_labels"] = selected

        st.divider()
        st.header("Progress")
        completed = st.session_state.get("completed_steps", set())
        for i, name in enumerate(STEP_NAMES):
            if i in completed:
                st.markdown(f"~~{name}~~")
            elif _can_run(i):
                st.markdown(f"**{name}** (ready)")
            else:
                st.markdown(f"{name}")

        # ---- Save / Load ----
        st.divider()
        st.header("Save / Load")
        auto_save = st.checkbox("Auto-save after each step", value=st.session_state.get("_auto_save", False))
        st.session_state["_auto_save"] = auto_save

        if st.button("Save State"):
            path = _save_state_to_disk()
            st.success(f"Saved to {Path(path).name}")

        saved_files = sorted(
            _glob.glob(str(_SAVE_DIR / "*.json")), reverse=True,
        )
        if saved_files:
            display_names = [Path(f).name for f in saved_files]
            selected_idx = st.selectbox(
                "Load saved state",
                range(len(display_names)),
                format_func=lambda i: display_names[i],
                key="load_selector",
            )
            if st.button("Load"):
                _load_state_from_disk(saved_files[selected_idx])
                st.rerun()
        else:
            st.caption("No saved states yet.")

    # ---- Build conversation list ----
    conversations = _get_selected_conversations(num_sessions)

    # ---- Tabs ----
    tabs = st.tabs(TAB_NAMES)

    with tabs[0]:
        render_data_tab(data_path, num_sessions)
    with tabs[1]:
        render_coref_tab(conversations, model_name)
    with tabs[2]:
        render_chunks_tab(conversations, model_name)
    with tabs[3]:
        render_gsw_tab(conversations, model_name)
    with tabs[4]:
        render_spacetime_tab(conversations, model_name)
    with tabs[5]:
        render_layer1_tab(conversations, model_name)
    with tabs[6]:
        render_layer2_tab(conversations, model_name)
    with tabs[7]:
        render_layer3_tab(model_name)
    with tabs[8]:
        render_qa_tab(model_name)
    with tabs[9]:
        render_diagnosis_tab(conversations, model_name)


if __name__ == "__main__":
    main()
