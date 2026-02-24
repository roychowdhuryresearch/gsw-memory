#!/usr/bin/env python3
"""
Personal Memory CLI — interactive terminal tool for the personal memory pipeline.

Combines pipeline processing, FAISS indexing, interactive QA, batch evaluation,
and memory inspection in a single rich TUI application.

Usage:
    python src/gsw_memory/personal_memory/playground/personal_memory_cli.py
    python src/gsw_memory/personal_memory/playground/personal_memory_cli.py --load path/to/state.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parents[3]  # gsw-memory/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

import faiss
from openai import OpenAI

# Lazy vLLM import
try:
    from vllm import LLM as _VllmLLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Rich
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.tree import Tree

# prompt_toolkit for QA REPL
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# gsw_memory imports
from gsw_memory.memory.models import GSWStructure
from gsw_memory.memory.operator_utils.spacetime import apply_spacetime_to_gsw
from gsw_memory.memory.reconciler import Reconciler
from gsw_memory.personal_memory.chunker import TopicBoundaryChunker
from gsw_memory.personal_memory.data_ingestion.locomo import (
    Conversation,
    LoCoMoLoader,
)
from gsw_memory.personal_memory.models import ConversationMemory
from gsw_memory.prompts.operator_prompts import (
    CorefPrompts,
    ConversationalOperatorPrompts,
    SpaceTimePrompts,
)
from gsw_memory.evaluation.hipporag_eval import calculate_exact_match, calculate_f1_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WORK_DIR = Path("/mnt/SSD1/shreyas/GSW_PM")
DEFAULT_CACHE_DIR = DEFAULT_WORK_DIR / ".exploration_cache"
DEFAULT_DATA_PATH = str(
    _REPO_ROOT / "data" / "personal_memory" / "locomo" / "data" / "locomo10.json"
)
DEFAULT_STATE_PATH = DEFAULT_WORK_DIR / "gsw_output.json"
DEFAULT_MODEL = "gpt-4o-mini"
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]

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

CAT_LABELS = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-ended",
    5: "Adversarial",
}

STEP_NAMES = [
    "0. Data Loading",
    "1. Coref Resolution",
    "2. Topic Chunking",
    "3. GSW Extraction",
    "4. SpaceTime Linker",
]

STEP_DEPS: Dict[int, Set[int]] = {
    0: set(),
    1: {0},
    2: {1},
    3: {2},
    4: {3},
}

ENTITY_TASK = (
    "Given an entity with its roles and states from a personal conversation, "
    "create an embedding for semantic retrieval."
)
QA_TASK = (
    "Given a question-answer pair from a personal conversation, "
    "create an embedding for similarity search."
)
SEARCH_TASK = (
    "Given a user question about a person's memory, "
    "create an embedding for searching relevant entities and facts."
)
CHUNK_TASK = "Given a conversation excerpt, create an embedding for semantic retrieval."

_ABSTENTION_PHRASES = [
    "i don't know", "i do not know", "no answer", "unanswerable",
    "cannot answer", "can't answer", "not mentioned", "no information",
    "not enough information", "cannot be determined", "unknown",
    "not available", "n/a", "none",
]


def is_abstention(text: str) -> bool:
    normalized = text.strip().lower().rstrip(".")
    return any(phrase in normalized for phrase in _ABSTENTION_PHRASES)


# ============================================================================
# Pipeline State
# ============================================================================

@dataclass
class PipelineState:
    data_path: str = DEFAULT_DATA_PATH
    work_dir: Path = field(default_factory=lambda: DEFAULT_WORK_DIR)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    all_conversations: List[Conversation] = field(default_factory=list)
    selected_conv_labels: List[str] = field(default_factory=lambda: ["conv-26 (Caroline x Melanie)"])
    completed_steps: Set[int] = field(default_factory=set)
    model_name: str = DEFAULT_MODEL
    embedding_backend: str = "openai"
    num_sessions: int = 20
    # per-step results keyed by conv_id
    raw_texts: Dict[str, List[str]] = field(default_factory=dict)
    resolved_texts: Dict[str, List[str]] = field(default_factory=dict)
    session_chunks: Dict[str, List[List[str]]] = field(default_factory=dict)
    chunk_gsws: Dict[str, List[GSWStructure]] = field(default_factory=dict)
    post_spacetime_gsws: Dict[str, List[Any]] = field(default_factory=dict)
    session_gsws: Dict[str, List[GSWStructure]] = field(default_factory=dict)
    # vector index
    builder: Optional["VectorStoreBuilder"] = field(default=None, repr=False)
    baseline_retriever: Optional["BaselineRetriever"] = field(default=None, repr=False)
    # embed function cache
    _embed_fn: Optional[Any] = field(default=None, repr=False)


# ============================================================================
# Embedding Backends
# ============================================================================

EmbedFn = Callable[[List[str], str], np.ndarray]


def make_openai_embed_fn() -> EmbedFn:
    client = OpenAI()

    def embed(texts: List[str], task: str, batch_size: int = 100) -> np.ndarray:
        all_embs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=batch,
            )
            all_embs.extend([d.embedding for d in resp.data])
        embs = np.array(all_embs, dtype=np.float32)
        faiss.normalize_L2(embs)
        return embs

    return embed


def make_vllm_embed_fn() -> EmbedFn:
    if not VLLM_AVAILABLE:
        raise ImportError("vllm is not installed. Install with: pip install vllm>=0.8.5")
    model = _VllmLLM(model="Qwen/Qwen3-Embedding-8B", task="embed")

    def _get_detailed_instruct(task: str, query: str) -> str:
        return f"Instruct: {task}\nQuery: {query}"

    def embed(texts: List[str], task: str, batch_size: int = 64) -> np.ndarray:
        instructed = [_get_detailed_instruct(task, t) for t in texts]
        all_embs: List[Any] = []
        for i in range(0, len(instructed), batch_size):
            batch = instructed[i : i + batch_size]
            outputs = model.embed(batch)
            all_embs.extend([o.outputs.embedding for o in outputs])
        embs = np.array(all_embs, dtype=np.float32)
        faiss.normalize_L2(embs)
        return embs

    return embed


def get_embed_fn(state: PipelineState) -> EmbedFn:
    if state._embed_fn is None:
        if state.embedding_backend == "vllm":
            state._embed_fn = make_vllm_embed_fn()
        else:
            state._embed_fn = make_openai_embed_fn()
    return state._embed_fn


# ============================================================================
# VectorStoreBuilder
# ============================================================================


class VectorStoreBuilder:
    def __init__(self, embed_fn: EmbedFn):
        self._embed_fn = embed_fn
        self.entity_texts: List[str] = []
        self.entity_metadata: List[Dict[str, Any]] = []
        self.entity_index: Optional[faiss.IndexFlatIP] = None
        self.qa_texts: List[str] = []
        self.qa_metadata: List[Dict[str, Any]] = []
        self.qa_index: Optional[faiss.IndexFlatIP] = None

    def build(
        self,
        gsw_dicts: Dict[str, List[dict]],
        session_chunks: Dict[str, List[List[str]]],
    ) -> None:
        self.entity_texts = []
        self.entity_metadata = []
        self.qa_texts = []
        self.qa_metadata = []

        for conv_id, gsw_list in gsw_dicts.items():
            chunks_flat = session_chunks.get(conv_id, [])
            flat_to_session: Dict[int, Tuple[int, int]] = {}
            flat_idx = 0
            for sess_idx, sess_chunks in enumerate(chunks_flat):
                for chunk_idx in range(len(sess_chunks)):
                    flat_to_session[flat_idx] = (sess_idx, chunk_idx)
                    flat_idx += 1

            for gi, gsw_dict in enumerate(gsw_list):
                gsw = GSWStructure.model_validate(gsw_dict)
                sess_idx, chunk_idx = flat_to_session.get(gi, (gi, 0))
                chunk_id = f"{conv_id}::s{sess_idx}_c{chunk_idx}"

                id_to_name = {e.id: e.name for e in gsw.entity_nodes}

                for entity in gsw.entity_nodes:
                    role_parts = []
                    for r in entity.roles:
                        if r.states:
                            role_parts.append(f"{r.role}: {', '.join(r.states)}")
                        else:
                            role_parts.append(r.role)
                    roles_text = " | ".join(role_parts) if role_parts else "no roles"
                    search_text = f"{entity.name} — {roles_text}"

                    linked_spaces = []
                    for eid, sid in gsw.space_edges:
                        if eid == entity.id:
                            sn = next((s for s in gsw.space_nodes if s.id == sid), None)
                            if sn:
                                linked_spaces.append(sn.current_name or sn.id)
                    linked_times = []
                    for eid, tid in gsw.time_edges:
                        if eid == entity.id:
                            tn = next((t for t in gsw.time_nodes if t.id == tid), None)
                            if tn:
                                linked_times.append(tn.current_name or tn.id)

                    self.entity_texts.append(search_text)
                    self.entity_metadata.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "chunk_id": chunk_id,
                        "session_idx": sess_idx,
                        "conv_id": conv_id,
                        "speaker_id": entity.speaker_id,
                        "roles": [
                            {
                                "role": r.role,
                                "states": r.states,
                                "speaker_id": r.speaker_id,
                                "evidence_turn_ids": r.evidence_turn_ids,
                            }
                            for r in entity.roles
                        ],
                        "linked_spaces": linked_spaces,
                        "linked_times": linked_times,
                    })

                for vp in gsw.verb_phrase_nodes:
                    for q in vp.questions:
                        answer_names = [id_to_name.get(a, a) for a in q.answers]
                        qa_text = f"{q.text} {', '.join(answer_names)}"
                        self.qa_texts.append(qa_text)
                        self.qa_metadata.append({
                            "question_id": q.id,
                            "question_text": q.text,
                            "answer_names": answer_names,
                            "answer_ids": q.answers,
                            "vp_phrase": vp.phrase,
                            "chunk_id": chunk_id,
                            "session_idx": sess_idx,
                            "conv_id": conv_id,
                            "speaker_id": q.speaker_id,
                            "evidence_turn_ids": q.evidence_turn_ids,
                        })

        if self.entity_texts:
            entity_embs = self._embed_fn(self.entity_texts, ENTITY_TASK)
            self.entity_index = faiss.IndexFlatIP(entity_embs.shape[1])
            self.entity_index.add(entity_embs)

        if self.qa_texts:
            qa_embs = self._embed_fn(self.qa_texts, QA_TASK)
            self.qa_index = faiss.IndexFlatIP(qa_embs.shape[1])
            self.qa_index.add(qa_embs)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_fn([query], SEARCH_TASK)

    def search_entities(
        self, query_emb: np.ndarray, top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.entity_index is None or self.entity_index.ntotal == 0:
            return []
        scores, indices = self.entity_index.search(query_emb, min(top_k, self.entity_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.entity_metadata[idx], float(score)))
        return results

    def search_qa(
        self, query_emb: np.ndarray, top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.qa_index is None or self.qa_index.ntotal == 0:
            return []
        scores, indices = self.qa_index.search(query_emb, min(top_k, self.qa_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.qa_metadata[idx], float(score)))
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self.entity_index:
            faiss.write_index(self.entity_index, str(directory / "entity_index.faiss"))
            (directory / "entity_metadata.json").write_text(
                json.dumps(self.entity_metadata, ensure_ascii=False), encoding="utf-8"
            )
        if self.qa_index:
            faiss.write_index(self.qa_index, str(directory / "qa_index.faiss"))
            (directory / "qa_metadata.json").write_text(
                json.dumps(self.qa_metadata, ensure_ascii=False), encoding="utf-8"
            )

    def load(self, directory: Path) -> bool:
        entity_faiss = directory / "entity_index.faiss"
        qa_faiss = directory / "qa_index.faiss"
        if not entity_faiss.exists():
            return False
        self.entity_index = faiss.read_index(str(entity_faiss))
        self.entity_metadata = json.loads(
            (directory / "entity_metadata.json").read_text(encoding="utf-8")
        )
        self.entity_texts = [
            f"{m['entity_name']} — " + " | ".join(
                f"{r['role']}: {', '.join(r['states'])}" if r['states'] else r['role']
                for r in m['roles']
            )
            for m in self.entity_metadata
        ]
        if qa_faiss.exists():
            self.qa_index = faiss.read_index(str(qa_faiss))
            self.qa_metadata = json.loads(
                (directory / "qa_metadata.json").read_text(encoding="utf-8")
            )
            self.qa_texts = [
                f"{m['question_text']} {', '.join(m['answer_names'])}"
                for m in self.qa_metadata
            ]
        return True


# ============================================================================
# QA Agent
# ============================================================================

_VECTOR_QA_SYSTEM = """\
You answer questions about people using a vector-indexed personal memory.

## How the memory works
Conversational memories are indexed per-chunk (no entity deduplication). The same \
entity may appear in multiple chunks with different roles/states.

Each entity has:
- **roles**: what the entity does or is (e.g. "researcher", "friend of Maria")
- **states**: how the entity changes within that role (e.g. "submitted paper", "moved to NYC")
- **speaker_id**: which conversation participant asserted this role
- **evidence_turn_ids**: dialogue turn IDs proving this role (e.g. "D1:3", "D4:7")

Entities can also be linked to **space nodes** (locations) and **time nodes** (dates/times). \
These are returned by get_entity_detail and search_spacetime.

IMPORTANT: A single entity can have roles from DIFFERENT speakers. Always check \
role.speaker_id to know who provided the information.

## Tools available
- **search_entities(query)**: Semantic search on entity index. Returns names, speakers, chunk_ids, scores.
- **search_questions(query)**: Semantic search on QA pair index. Returns questions, answers, evidence.
- **get_entity_detail(entity_name)**: ALL occurrences of an entity across chunks — full roles, \
states, spacetime, plus QA pairs where the entity is an answer.
- **search_spacetime(entity_names)**: Matrix view of time/space connections for a list of entities. \
Shows which entities share the same times or locations, grouped by time/space node.

## Tool usage strategy
1. Decompose the question into atomic sub-questions.
2. Use search_entities(query) to find relevant entities — returns basic info (name, speaker, score).
3. ALWAYS call get_entity_detail(entity_name) next to see detailed roles, states, speaker_id, \
evidence_turn_ids, and linked spacetime. This is where the real information is.
4. Use search_questions(query) when looking for specific facts — QA pairs may directly contain \
the answer with evidence_turn_ids.
5. For "when"/"where" questions: search relevant entities first, then call \
search_spacetime(entity_names=[...]) with the entity names. The result groups time/space \
nodes showing which entities share the same temporal/spatial context.
6. For multi-hop: search entity A → get_entity_detail(A) → find linked entity B in roles/states → \
get_entity_detail(B). Chain lookups to follow relationships.
7. If search misses, rephrase with synonyms or related terms.

## Relationship navigation
- Relationships may be stored in one direction only (e.g. "son of" but not "parent of").
- If you don't find a direct match, search for the related entity and look at ALL its \
roles for the inverse relationship.
- The same entity in different chunks has complementary info — get_entity_detail aggregates it all.

## Speaker attribution
- When the question targets a specific person (e.g. "What does Caroline do?"), verify \
that the role's speaker_id matches that person.
- If evidence comes from a different speaker than expected, still report accurately.

## Evidence chain
- Your final evidence_turn_ids MUST come from the role's evidence_turn_ids returned by \
get_entity_detail or search_questions. Do not invent turn IDs.
- Collect evidence_turn_ids from every role you used to build the answer.

## Output format
When you have the answer, respond with ONLY a JSON object:
{
    "answer": "Concise answer, no filler",
    "reasoning": "Step-by-step explanation of how you found it",
    "speaker_id": "Person this answer is about (null if unclear)",
    "evidence_turn_ids": ["D1:3", "D4:7"]
}

Do NOT include phrases like "The answer is" or "Based on my search" in the answer field. \
If you cannot find sufficient evidence, set answer to "I don't know" and explain why in reasoning.
"""


class _VectorTools:
    def __init__(self, builder: VectorStoreBuilder):
        self._builder = builder

    def search_entities(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        query_emb = self._builder.embed_query(query)
        results = self._builder.search_entities(query_emb, top_k)
        return {"entities": [
            {
                "name": m["entity_name"],
                "speaker_id": m.get("speaker_id"),
                "chunk_id": m["chunk_id"],
                "role_count": len(m.get("roles", [])),
                "score": round(s, 4),
            }
            for m, s in results
        ]}

    def search_questions(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        query_emb = self._builder.embed_query(query)
        results = self._builder.search_qa(query_emb, top_k)
        return {"questions": [
            {
                "question": m["question_text"],
                "answers": m["answer_names"],
                "vp_phrase": m["vp_phrase"],
                "speaker_id": m.get("speaker_id"),
                "evidence_turn_ids": m.get("evidence_turn_ids", []),
                "chunk_id": m["chunk_id"],
                "score": round(s, 4),
            }
            for m, s in results
        ]}

    def get_entity_detail(self, entity_name: str) -> Dict[str, Any]:
        name_lower = entity_name.lower()
        occurrences = []
        entity_chunks: Dict[str, set] = {}

        for meta in self._builder.entity_metadata:
            if name_lower in meta["entity_name"].lower():
                chunk_id = meta["chunk_id"]
                entity_chunks.setdefault(chunk_id, set()).add(meta["entity_id"])
                occurrences.append({
                    "entity_name": meta["entity_name"],
                    "chunk_id": chunk_id,
                    "session_idx": meta["session_idx"],
                    "speaker_id": meta.get("speaker_id"),
                    "roles": meta.get("roles", []),
                    "linked_spaces": meta.get("linked_spaces", []),
                    "linked_times": meta.get("linked_times", []),
                })

        qa_pairs = []
        for qa_meta in self._builder.qa_metadata:
            chunk_id = qa_meta["chunk_id"]
            if chunk_id in entity_chunks:
                chunk_eids = entity_chunks[chunk_id]
                if chunk_eids & set(qa_meta.get("answer_ids", [])):
                    qa_pairs.append({
                        "question": qa_meta["question_text"],
                        "answers": qa_meta["answer_names"],
                        "vp_phrase": qa_meta["vp_phrase"],
                        "speaker_id": qa_meta.get("speaker_id"),
                        "evidence_turn_ids": qa_meta.get("evidence_turn_ids", []),
                        "chunk_id": chunk_id,
                    })

        return {
            "entity_name": entity_name,
            "occurrences": occurrences,
            "qa_pairs": qa_pairs,
            "count": len(occurrences),
        }

    def search_spacetime(self, entity_names: List[str]) -> Dict[str, Any]:
        name_to_entries: Dict[str, List[Dict]] = {}
        for ename in entity_names:
            el = ename.lower()
            name_to_entries[ename] = [
                m for m in self._builder.entity_metadata
                if el in m["entity_name"].lower()
            ]

        time_groups: Dict[Tuple[str, str], set] = {}
        space_groups: Dict[Tuple[str, str], set] = {}

        for ename, entries in name_to_entries.items():
            for meta in entries:
                chunk_id = meta["chunk_id"]
                for t in meta.get("linked_times", []):
                    time_groups.setdefault((t, chunk_id), set()).add(ename)
                for s in meta.get("linked_spaces", []):
                    space_groups.setdefault((s, chunk_id), set()).add(ename)

        time_connections = [
            {"time": tv, "chunk_id": cid, "linked_entities": sorted(names)}
            for (tv, cid), names in sorted(time_groups.items())
        ]
        space_connections = [
            {"location": sv, "chunk_id": cid, "linked_entities": sorted(names)}
            for (sv, cid), names in sorted(space_groups.items())
        ]

        return {
            "time_connections": time_connections,
            "space_connections": space_connections,
        }

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "search_entities",
                "description": (
                    "Semantic search on the entity index. Returns entity name, "
                    "speaker_id, chunk_id, role_count, and similarity score."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results (default 10)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "type": "function",
                "name": "search_questions",
                "description": (
                    "Semantic search on the QA pair index. Returns question text, "
                    "answers, verb phrase, speaker_id, evidence_turn_ids, and score."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results (default 10)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "type": "function",
                "name": "get_entity_detail",
                "description": (
                    "Get ALL occurrences of an entity by name across all chunks. "
                    "Returns full roles, states, speaker_id, evidence_turn_ids, "
                    "linked_spaces, linked_times per occurrence, plus QA pairs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Entity name (case-insensitive substring match)",
                        },
                    },
                    "required": ["entity_name"],
                },
            },
            {
                "type": "function",
                "name": "search_spacetime",
                "description": (
                    "Find time and space connections for a list of entities. "
                    "Returns matrix of shared times/locations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of entity names",
                        },
                    },
                    "required": ["entity_names"],
                },
            },
        ]

    def dispatch(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "search_entities":
            return self.search_entities(**args)
        elif name == "search_questions":
            return self.search_questions(**args)
        elif name == "get_entity_detail":
            return self.get_entity_detail(**args)
        elif name == "search_spacetime":
            return self.search_spacetime(**args)
        return {"error": f"Unknown tool: {name}"}


class VectorQAAgent:
    def __init__(
        self,
        builder: VectorStoreBuilder,
        model_name: str = "gpt-4o",
        max_iterations: int = 10,
    ):
        self.builder = builder
        self.model_name = model_name
        self.max_iterations = max_iterations
        self._client = OpenAI()

    def answer(self, question: str, console: Optional[Console] = None) -> Dict[str, Any]:
        """Answer a question. If console is provided, streams the trace live."""
        tools = _VectorTools(self.builder)
        tool_defs = tools.tool_definitions()
        trace: List[Dict[str, Any]] = []

        messages: List[Any] = [
            {"role": "user", "content": f"Question: {question}"},
        ]

        for iteration in range(self.max_iterations):
            if console:
                console.print(f"  [dim]Iteration {iteration + 1}...[/dim]")

            response = self._client.responses.create(
                model=self.model_name,
                input=messages,
                tools=tool_defs,
                instructions=_VECTOR_QA_SYSTEM,
                temperature=0,
            )

            messages += response.output

            agent_text_parts: List[str] = []
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for part in getattr(item, "content", []):
                        if getattr(part, "type", None) == "output_text":
                            agent_text_parts.append(part.text)

            trace_entry: Dict[str, Any] = {
                "iteration": iteration + 1,
                "agent_text": "\n".join(agent_text_parts),
                "tool_calls": [],
            }

            if console and agent_text_parts:
                text = "\n".join(agent_text_parts)
                console.print(f"  [italic]{text}[/italic]")

            function_calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            if function_calls:
                for fc in function_calls:
                    name = getattr(fc, "name", "")
                    raw_args = getattr(fc, "arguments", "{}")
                    call_id = getattr(fc, "call_id", None)
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}

                    if console:
                        args_str = json.dumps(args)
                        console.print(f"  [cyan]{name}[/cyan]({args_str})")

                    result = tools.dispatch(name, args)

                    if console:
                        # Show compact result summary
                        if name == "search_entities":
                            entities = result.get("entities", [])
                            names = [e["name"] for e in entities[:5]]
                            console.print(f"    [dim]→ {len(entities)} results: {', '.join(names)}{'...' if len(entities) > 5 else ''}[/dim]")
                        elif name == "search_questions":
                            qs = result.get("questions", [])
                            console.print(f"    [dim]→ {len(qs)} QA pairs found[/dim]")
                        elif name == "get_entity_detail":
                            count = result.get("count", 0)
                            qa_count = len(result.get("qa_pairs", []))
                            console.print(f"    [dim]→ {count} occurrences, {qa_count} QA pairs[/dim]")
                        elif name == "search_spacetime":
                            tc = len(result.get("time_connections", []))
                            sc = len(result.get("space_connections", []))
                            console.print(f"    [dim]→ {tc} time, {sc} space connections[/dim]")

                    trace_entry["tool_calls"].append({
                        "name": name,
                        "args": args,
                        "result": result,
                    })
                    messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    })
                trace.append(trace_entry)
                continue

            trace.append(trace_entry)
            content = getattr(response, "output_text", "") or ""
            if not content:
                try:
                    msg_items = [
                        it for it in response.output
                        if getattr(it, "type", None) == "message"
                    ]
                    if msg_items:
                        content = getattr(msg_items[-1], "content", "") or ""
                except Exception:
                    content = ""

            raw = self._parse_answer(content)
            return {
                "answer": raw.get("answer", ""),
                "reasoning": raw.get("reasoning", ""),
                "speaker_id": raw.get("speaker_id"),
                "evidence_turn_ids": raw.get("evidence_turn_ids", []),
                "trace": trace,
            }

        return {
            "answer": "Unable to answer within iteration limit.",
            "reasoning": "",
            "speaker_id": None,
            "evidence_turn_ids": [],
            "trace": trace,
        }

    @staticmethod
    def _parse_answer(content: str) -> Dict[str, Any]:
        try:
            j_start = content.find("{")
            j_end = content.rfind("}") + 1
            if j_start != -1 and j_end > j_start:
                return json.loads(content[j_start:j_end])
        except Exception:
            pass
        return {"answer": content, "reasoning": ""}


# ============================================================================
# Baseline Retriever
# ============================================================================


class BaselineRetriever:
    def __init__(self, embed_fn: EmbedFn):
        self._embed_fn = embed_fn
        self.chunk_texts: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None

    def build(self, session_chunks: Dict[str, List[List[str]]]) -> None:
        for conv_id, sessions in session_chunks.items():
            for sess_idx, chunks in enumerate(sessions):
                for chunk_idx, text in enumerate(chunks):
                    chunk_id = f"{conv_id}::s{sess_idx}_c{chunk_idx}"
                    self.chunk_texts.append(text)
                    self.chunk_metadata.append({
                        "chunk_id": chunk_id,
                        "session_idx": sess_idx,
                        "conv_id": conv_id,
                    })

        embs = self._embed_fn(self.chunk_texts, CHUNK_TASK)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int = 5) -> str:
        query_emb = self._embed_fn([query], CHUNK_TASK)
        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))
        parts: List[str] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                meta = self.chunk_metadata[idx]
                parts.append(
                    f"[{meta['chunk_id']} | score={score:.3f}]\n{self.chunk_texts[idx]}"
                )
        return "\n\n---\n\n".join(parts)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "baseline_chunk_index.faiss"))
        (directory / "baseline_chunk_metadata.json").write_text(
            json.dumps({"metadata": self.chunk_metadata, "texts": self.chunk_texts}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> bool:
        path = directory / "baseline_chunk_index.faiss"
        if not path.exists():
            return False
        self.index = faiss.read_index(str(path))
        data = json.loads((directory / "baseline_chunk_metadata.json").read_text(encoding="utf-8"))
        self.chunk_metadata = data["metadata"]
        self.chunk_texts = data["texts"]
        return True


# ============================================================================
# LLM Judge
# ============================================================================

try:
    from bespokelabs import curator

    class LLMJudge(curator.LLM):
        return_completions_object = True

        def prompt(self, input_data):
            system = (
                "You are an expert judge evaluating the accuracy of AI-generated answers "
                "against a ground truth answer. You must respond with a JSON object only."
            )
            cat = int(input_data["category"])
            gold = input_data["gold"]
            predicted = input_data["predicted"]

            if cat == 5 and not gold:
                user = (
                    f"Question: {input_data['question']}\n"
                    f"Ground truth: This is an adversarial/trick question. The correct "
                    f"response is to refuse to answer or say 'I don't know'.\n"
                    f"AI answer: {predicted}\n\n"
                    "Score the AI answer:\n"
                    "- 1 if the AI correctly refused to answer or said it doesn't know\n"
                    "- 0.5 if the AI expressed uncertainty but still attempted an answer\n"
                    "- 0 if the AI confidently gave a specific (wrong) answer\n\n"
                    'Respond with JSON: {"score": <0|0.5|1>, "reason": "<brief explanation>"}'
                )
            else:
                user = (
                    f"Question: {input_data['question']}\n"
                    f"Ground truth: {gold}\n"
                    f"AI answer: {predicted}\n\n"
                    "Score the AI answer:\n"
                    "- 1 if correct (same meaning, phrasing differences OK)\n"
                    "- 0.5 if partially correct\n"
                    "- 0 if wrong, irrelevant, or says 'I don't know' when ground truth has an answer\n\n"
                    'Respond with JSON: {"score": <0|0.5|1>, "reason": "<brief explanation>"}'
                )

            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        def parse(self, input_data, response):
            raw = response["choices"][0]["message"]["content"].strip()
            score = 0.0
            reason = ""
            try:
                parsed = json.loads(raw)
                score = float(parsed.get("score", 0))
                reason = parsed.get("reason", "")
            except (json.JSONDecodeError, ValueError):
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        score = float(parsed.get("score", 0))
                        reason = parsed.get("reason", "")
                    except (json.JSONDecodeError, ValueError):
                        reason = f"parse_error: {raw[:200]}"
                else:
                    reason = f"no_json: {raw[:200]}"

            if score not in (0.0, 0.5, 1.0):
                score = round(score * 2) / 2

            return [{
                "question": input_data["question"],
                "predicted": input_data["predicted"],
                "gold": input_data["gold"],
                "category": input_data["category"],
                "llm_score": score,
                "llm_reason": reason,
            }]

    CURATOR_AVAILABLE = True
except ImportError:
    CURATOR_AVAILABLE = False


class BaselineChunkQA:
    """Simple baseline QA without curator — uses direct OpenAI calls."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self._client = OpenAI()

    def answer(self, question: str, context: str) -> str:
        system = (
            "You answer questions about a person's conversational memories using "
            "retrieved conversation excerpts.\n\n"
            "Your response starts after 'Thought: ', where you reason step-by-step. "
            "Conclude with 'Answer: ' to present a concise, definitive answer.\n\n"
            "If you cannot answer from the excerpts, say 'Answer: I don't know'."
        )
        user = f"{context}\n\nQuestion: {question}\nThought: "
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        answer_text = resp.choices[0].message.content
        if "Answer: " in answer_text:
            return answer_text.split("Answer: ")[-1].strip()
        return answer_text.strip()


# ============================================================================
# Pipeline Step Functions
# ============================================================================


def _derive_entity_speaker_ids(gsw: GSWStructure) -> None:
    for entity in gsw.entity_nodes:
        role_speakers = {r.speaker_id for r in entity.roles if r.speaker_id is not None}
        if len(role_speakers) == 1:
            entity.speaker_id = role_speakers.pop()


def _get_selected_conversations(state: PipelineState) -> List[Conversation]:
    if not state.all_conversations:
        return []
    result = []
    for label in state.selected_conv_labels:
        idx = KNOWN_CONVS.get(label)
        if idx is not None and idx < len(state.all_conversations):
            conv = state.all_conversations[idx]
            conv_copy = Conversation(
                sample_id=conv.sample_id,
                speaker_a=conv.speaker_a,
                speaker_b=conv.speaker_b,
                sessions=conv.sessions[:state.num_sessions],
                qa_pairs=conv.qa_pairs,
                event_summaries=conv.event_summaries,
                session_summaries=conv.session_summaries,
            )
            result.append(conv_copy)
    return result


def step_load_data(state: PipelineState, console: Console) -> None:
    console.print(f"[cyan]Loading LoCoMo data from {state.data_path}...[/cyan]")
    loader = LoCoMoLoader(state.data_path)
    state.all_conversations = loader.load()
    state.completed_steps.add(0)
    console.print(f"[green]Loaded {len(state.all_conversations)} conversations[/green]")


def step_coref(state: PipelineState, console: Console) -> None:
    client = OpenAI()
    conversations = _get_selected_conversations(state)
    raw_texts: Dict[str, List[str]] = {}
    resolved_texts: Dict[str, List[str]] = {}

    total = sum(len(c.sessions) for c in conversations)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Coref resolution", total=total)
        for conv in conversations:
            raws, resolveds = [], []
            for sess in conv.sessions:
                raw = sess.to_document()
                raws.append(raw)
                resp = client.chat.completions.create(
                    model=state.model_name,
                    messages=[
                        {"role": "system", "content": CorefPrompts.SYSTEM_PROMPT},
                        {"role": "user", "content": CorefPrompts.USER_PROMPT_TEMPLATE.format(text=raw)},
                    ],
                    temperature=0,
                    max_tokens=4000,
                )
                resolveds.append(resp.choices[0].message.content)
                progress.advance(task)
            raw_texts[conv.sample_id] = raws
            resolved_texts[conv.sample_id] = resolveds

    state.raw_texts = raw_texts
    state.resolved_texts = resolved_texts
    state.completed_steps.add(1)
    console.print(f"[green]Coref done: {total} sessions processed[/green]")


def step_chunking(state: PipelineState, console: Console) -> None:
    chunker = TopicBoundaryChunker(model_name=state.model_name)
    conversations = _get_selected_conversations(state)
    session_chunks: Dict[str, List[List[str]]] = {}

    total = sum(len(c.sessions) for c in conversations)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Topic chunking", total=total)
        for conv in conversations:
            chunks_per_session = []
            for sess_idx, sess in enumerate(conv.sessions):
                resolved = state.resolved_texts[conv.sample_id][sess_idx]
                chunks = chunker.chunk_session_from_text(resolved, sess)
                chunks_per_session.append(chunks)
                progress.advance(task)
            session_chunks[conv.sample_id] = chunks_per_session

    state.session_chunks = session_chunks
    state.completed_steps.add(2)
    total_chunks = sum(len(c) for cs in session_chunks.values() for c in cs)
    console.print(f"[green]Chunking done: {total_chunks} chunks created[/green]")


def step_gsw_extraction(state: PipelineState, console: Console) -> None:
    client = OpenAI()
    gsw_schema = GSWStructure.model_json_schema()
    conversations = _get_selected_conversations(state)
    chunk_gsws: Dict[str, List[GSWStructure]] = {}

    total = sum(
        sum(len(cks) for cks in state.session_chunks[c.sample_id])
        for c in conversations
    )
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("GSW extraction", total=total)
        for conv in conversations:
            speaker_context = f"Speaker A: {conv.speaker_a}, Speaker B: {conv.speaker_b}"
            all_gsws = []
            for sess_idx, chunks in enumerate(state.session_chunks[conv.sample_id]):
                for chunk_text in chunks:
                    resp = client.chat.completions.create(
                        model=state.model_name,
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
                    gsw.space_nodes = []
                    gsw.time_nodes = []
                    gsw.space_edges = []
                    gsw.time_edges = []
                    gsw.similarity_edges = []
                    _derive_entity_speaker_ids(gsw)
                    all_gsws.append(gsw)
                    progress.advance(task)
            chunk_gsws[conv.sample_id] = all_gsws

    state.chunk_gsws = chunk_gsws
    state.completed_steps.add(3)
    total_entities = sum(len(g.entity_nodes) for gs in chunk_gsws.values() for g in gs)
    console.print(f"[green]GSW extraction done: {total_entities} entities extracted[/green]")


def step_spacetime(state: PipelineState, console: Console) -> None:
    client = OpenAI()
    conversations = _get_selected_conversations(state)
    post_gsws: Dict[str, List[GSWStructure]] = {}
    for conv in conversations:
        post_gsws[conv.sample_id] = [copy.deepcopy(g) for g in state.chunk_gsws[conv.sample_id]]

    total = sum(len(post_gsws[c.sample_id]) for c in conversations)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("SpaceTime linking", total=total)
        for conv in conversations:
            flat_idx = 0
            for sess_idx, chunks in enumerate(state.session_chunks[conv.sample_id]):
                session_date = conv.sessions[sess_idx].date_time if sess_idx < len(conv.sessions) else ""
                session_ctx = f"Session date: {session_date}" if session_date else ""

                for chunk_idx, chunk_text in enumerate(chunks):
                    gsw = post_gsws[conv.sample_id][flat_idx]
                    if gsw.entity_nodes:
                        operator_output = {"entity_nodes": [e.model_dump() for e in gsw.entity_nodes]}
                        resp = client.chat.completions.create(
                            model=state.model_name,
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
                            logger.warning("Failed to parse SpaceTime JSON for %s chunk %d",
                                           conv.sample_id, flat_idx)

                    flat_idx += 1
                    progress.advance(task)

    state.post_spacetime_gsws = post_gsws
    state.completed_steps.add(4)
    total_space = sum(len(g.space_nodes) for gs in post_gsws.values() for g in gs)
    total_time = sum(len(g.time_nodes) for gs in post_gsws.values() for g in gs)
    console.print(f"[green]SpaceTime done: {total_space} space nodes, {total_time} time nodes[/green]")


# ============================================================================
# Serialization (compatible with pipeline_inspector)
# ============================================================================

_STATE_KEYS_TO_SAVE = [
    "completed_steps",
    "raw_texts",
    "resolved_texts",
    "session_chunks",
    "chunk_gsws",
    "post_spacetime_gsws",
    "session_gsws",
]


def serialize_state(state: PipelineState) -> dict:
    out: Dict[str, Any] = {}
    for key in _STATE_KEYS_TO_SAVE:
        val = getattr(state, key, None)
        if val is None:
            out[key] = None
            continue

        if key == "completed_steps":
            out[key] = sorted(val)
        elif key in ("raw_texts", "resolved_texts", "session_chunks"):
            out[key] = val
        elif key in ("chunk_gsws", "post_spacetime_gsws", "session_gsws"):
            out[key] = {
                cid: [gsw.model_dump() if isinstance(gsw, GSWStructure) else gsw for gsw in gsw_list]
                for cid, gsw_list in val.items()
            }
        else:
            out[key] = val

    out["_selected_conv_labels"] = state.selected_conv_labels
    return out


def deserialize_state(data: dict, state: PipelineState) -> None:
    for key in _STATE_KEYS_TO_SAVE:
        val = data.get(key)
        if val is None:
            continue

        if key == "completed_steps":
            state.completed_steps = set(val)
        elif key in ("raw_texts", "resolved_texts", "session_chunks"):
            setattr(state, key, val)
        elif key in ("chunk_gsws", "session_gsws"):
            setattr(state, key, {
                cid: [GSWStructure.model_validate(d) for d in gsw_list]
                for cid, gsw_list in val.items()
            })
        elif key == "post_spacetime_gsws":
            # Keep as dicts for VectorStoreBuilder compatibility
            setattr(state, key, val)
        else:
            setattr(state, key, val)

    if "_selected_conv_labels" in data:
        state.selected_conv_labels = data["_selected_conv_labels"]


def save_state(state: PipelineState, console: Console) -> str:
    state.cache_dir.mkdir(parents=True, exist_ok=True)
    filepath = state.cache_dir / "gsw_output.json"
    payload = serialize_state(state)
    filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]State saved to {filepath}[/green]")
    return str(filepath)


def load_state(path: str, state: PipelineState, console: Console) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    deserialize_state(data, state)
    console.print(f"[green]State loaded from {Path(path).name}[/green]")
    console.print(f"  Completed steps: {sorted(state.completed_steps)}")
    if state.post_spacetime_gsws:
        conv_ids = list(state.post_spacetime_gsws.keys())
        total_gsws = sum(len(v) for v in state.post_spacetime_gsws.values())
        console.print(f"  Conversations: {conv_ids}, {total_gsws} GSWs")


# ============================================================================
# Menu Handlers
# ============================================================================

def _show_status(state: PipelineState, console: Console) -> None:
    """Show compact status bar."""
    parts = []
    parts.append(f"Model: [cyan]{state.model_name}[/cyan]")
    parts.append(f"Embed: [cyan]{state.embedding_backend}[/cyan]")
    if state.completed_steps:
        parts.append(f"Steps: [green]{sorted(state.completed_steps)}[/green]")
    if state.post_spacetime_gsws:
        n = sum(len(v) for v in state.post_spacetime_gsws.values())
        parts.append(f"GSWs: [green]{n}[/green]")
    if state.builder and state.builder.entity_index:
        parts.append(
            f"Index: [green]{state.builder.entity_index.ntotal}E "
            f"{state.builder.qa_index.ntotal if state.builder.qa_index else 0}Q[/green]"
        )
    console.print(" | ".join(parts))


def menu_pipeline(state: PipelineState, console: Console) -> None:
    while True:
        console.print()
        console.rule("[bold]Pipeline[/bold]")

        # Show step status
        for i, name in enumerate(STEP_NAMES):
            if i in state.completed_steps:
                console.print(f"  [green][done][/green] {name}")
            elif STEP_DEPS[i].issubset(state.completed_steps):
                console.print(f"  [yellow][ready][/yellow] {name}")
            else:
                console.print(f"  [dim][locked][/dim] {name}")

        choice = Prompt.ask(
            "\nAction",
            choices=["next", "all", "back"],
            default="next",
        )

        if choice == "back":
            return

        if choice == "next":
            # Find next runnable step
            for i in range(5):
                if i not in state.completed_steps and STEP_DEPS[i].issubset(state.completed_steps):
                    _run_step(i, state, console)
                    break
            else:
                console.print("[green]All steps complete![/green]")

        elif choice == "all":
            for i in range(5):
                if i not in state.completed_steps and STEP_DEPS[i].issubset(state.completed_steps):
                    _run_step(i, state, console)


def _run_step(step: int, state: PipelineState, console: Console) -> None:
    console.print(f"\n[bold cyan]Running {STEP_NAMES[step]}...[/bold cyan]")
    start = time.time()
    if step == 0:
        step_load_data(state, console)
    elif step == 1:
        step_coref(state, console)
    elif step == 2:
        step_chunking(state, console)
    elif step == 3:
        step_gsw_extraction(state, console)
    elif step == 4:
        step_spacetime(state, console)
    elapsed = time.time() - start
    console.print(f"[dim]Step completed in {elapsed:.1f}s[/dim]")


def menu_index(state: PipelineState, console: Console) -> None:
    while True:
        console.print()
        console.rule("[bold]Vector Index[/bold]")

        if state.builder and state.builder.entity_index:
            console.print(
                f"  [green]Index loaded:[/green] "
                f"{state.builder.entity_index.ntotal} entities, "
                f"{state.builder.qa_index.ntotal if state.builder.qa_index else 0} QA pairs"
            )
        else:
            console.print("  [yellow]No index loaded[/yellow]")

        if state.baseline_retriever and state.baseline_retriever.index:
            console.print(f"  [green]Baseline index:[/green] {state.baseline_retriever.index.ntotal} chunks")

        choice = Prompt.ask(
            "\nAction",
            choices=["build", "load", "stats", "back"],
            default="load" if not (state.builder and state.builder.entity_index) else "stats",
        )

        if choice == "back":
            return

        if choice == "build":
            if not state.post_spacetime_gsws:
                console.print("[red]No GSWs available. Run pipeline first or load a saved state.[/red]")
                continue
            console.print("[cyan]Building FAISS index...[/cyan]")
            embed_fn = get_embed_fn(state)
            builder = VectorStoreBuilder(embed_fn)
            # Convert GSWStructure objects to dicts if needed
            gsw_dicts = {}
            for cid, gsw_list in state.post_spacetime_gsws.items():
                gsw_dicts[cid] = [
                    g.model_dump() if isinstance(g, GSWStructure) else g
                    for g in gsw_list
                ]
            builder.build(gsw_dicts, state.session_chunks)
            builder.save(state.cache_dir)
            state.builder = builder
            console.print(
                f"[green]Index built and saved: "
                f"{builder.entity_index.ntotal} entities, "
                f"{builder.qa_index.ntotal if builder.qa_index else 0} QA pairs[/green]"
            )

            # Also build baseline
            console.print("[cyan]Building baseline chunk index...[/cyan]")
            retriever = BaselineRetriever(embed_fn)
            retriever.build(state.session_chunks)
            retriever.save(state.cache_dir)
            state.baseline_retriever = retriever
            console.print(f"[green]Baseline index: {retriever.index.ntotal} chunks[/green]")

        elif choice == "load":
            embed_fn = get_embed_fn(state)
            if (state.cache_dir / "entity_index.faiss").exists():
                builder = VectorStoreBuilder(embed_fn)
                if builder.load(state.cache_dir):
                    state.builder = builder
                    console.print(
                        f"[green]Index loaded: "
                        f"{builder.entity_index.ntotal} entities, "
                        f"{builder.qa_index.ntotal if builder.qa_index else 0} QA pairs[/green]"
                    )
                else:
                    console.print("[red]Failed to load index[/red]")
            else:
                console.print(f"[yellow]No cached index found at {state.cache_dir}[/yellow]")

            # Load baseline if available
            if (state.cache_dir / "baseline_chunk_index.faiss").exists():
                retriever = BaselineRetriever(embed_fn)
                if retriever.load(state.cache_dir):
                    state.baseline_retriever = retriever
                    console.print(f"[green]Baseline index loaded: {retriever.index.ntotal} chunks[/green]")

        elif choice == "stats":
            if state.builder and state.builder.entity_index:
                console.print(f"  Entity entries: {state.builder.entity_index.ntotal}")
                console.print(f"  QA entries: {state.builder.qa_index.ntotal if state.builder.qa_index else 0}")
                # Show unique entity names
                unique_entities = set(m["entity_name"] for m in state.builder.entity_metadata)
                console.print(f"  Unique entity names: {len(unique_entities)}")
                # Show conversation breakdown
                conv_counts: Dict[str, int] = {}
                for m in state.builder.entity_metadata:
                    conv_counts[m["conv_id"]] = conv_counts.get(m["conv_id"], 0) + 1
                for cid, cnt in sorted(conv_counts.items()):
                    console.print(f"    {cid}: {cnt} entity entries")
            else:
                console.print("[yellow]No index loaded[/yellow]")


def menu_ask(state: PipelineState, console: Console) -> None:
    if not (state.builder and state.builder.entity_index):
        console.print("[red]No vector index loaded. Use Index menu first.[/red]")
        return

    console.print()
    console.rule("[bold]Interactive QA[/bold]")
    console.print("[dim]Type a question, /trace for last trace, quit to exit[/dim]")

    session = PromptSession(history=InMemoryHistory())
    last_result: Optional[Dict[str, Any]] = None
    qa_model = Prompt.ask("QA model", default="gpt-4o", choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"])

    while True:
        try:
            question = session.prompt("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "/quit", "q", "back"):
            break

        if question == "/trace":
            if last_result and last_result.get("trace"):
                _show_trace(last_result["trace"], console)
            else:
                console.print("[yellow]No trace available[/yellow]")
            continue

        if question.startswith("/baseline "):
            q = question[len("/baseline "):].strip()
            if state.baseline_retriever and state.baseline_retriever.index:
                console.print("[cyan]Running baseline QA...[/cyan]")
                baseline_qa = BaselineChunkQA(model_name=qa_model)
                ctx = state.baseline_retriever.retrieve(q, top_k=20)
                answer = baseline_qa.answer(q, ctx)
                console.print(Panel(answer, title="Baseline Answer", border_style="blue"))
            else:
                console.print("[yellow]No baseline index loaded[/yellow]")
            continue

        # Agentic QA
        console.print("[cyan]Running agentic QA...[/cyan]")
        agent = VectorQAAgent(state.builder, model_name=qa_model)
        result = agent.answer(question, console=console)
        last_result = result

        # Display answer
        console.print()
        console.print(Panel(
            result["answer"],
            title="Answer",
            border_style="green" if result["answer"] != "I don't know" else "yellow",
        ))

        if result.get("reasoning"):
            console.print(f"[dim]Reasoning: {result['reasoning']}[/dim]")

        if result.get("speaker_id"):
            console.print(f"  Speaker: {result['speaker_id']}")
        if result.get("evidence_turn_ids"):
            console.print(f"  Evidence: {', '.join(result['evidence_turn_ids'])}")

        trace = result.get("trace", [])
        if trace:
            total_calls = sum(len(e.get("tool_calls", [])) for e in trace)
            console.print(f"  [dim]{len(trace)} iterations, {total_calls} tool calls (/trace for full details)[/dim]")


def _show_trace(trace: List[Dict[str, Any]], console: Console) -> None:
    for entry in trace:
        console.print(f"\n[bold]Iteration {entry['iteration']}[/bold]")
        if entry.get("agent_text"):
            console.print(f"  [dim]{entry['agent_text']}[/dim]")
        for tc in entry.get("tool_calls", []):
            console.print(f"  [cyan]{tc['name']}[/cyan]({json.dumps(tc['args'])})")
            result_str = json.dumps(tc["result"], indent=2, ensure_ascii=False)
            console.print(f"    [dim]{result_str}[/dim]")


def menu_eval(state: PipelineState, console: Console) -> None:
    if not (state.builder and state.builder.entity_index):
        console.print("[red]No vector index loaded. Use Index menu first.[/red]")
        return

    console.print()
    console.rule("[bold]Batch Evaluation[/bold]")

    # Load LoCoMo QA pairs
    data_path = state.data_path
    if not Path(data_path).exists():
        console.print(f"[red]LoCoMo file not found: {data_path}[/red]")
        return

    conversations = LoCoMoLoader(data_path).load()
    conv_ids = set(state.post_spacetime_gsws.keys()) if state.post_spacetime_gsws else set()

    if not conv_ids:
        console.print("[red]No conversation IDs in state[/red]")
        return

    console.print(f"Conversations in state: {sorted(conv_ids)}")

    # Mode
    mode = Prompt.ask("Mode", choices=["baseline", "agentic", "both"], default="both")

    # Categories
    cat_input = Prompt.ask("Categories (comma-separated)", default="1,2,3,4,5")
    categories = [int(c.strip()) for c in cat_input.split(",")]

    # Max questions
    max_q_input = Prompt.ask("Max questions (empty=all)", default="")
    max_questions = int(max_q_input) if max_q_input else None

    # QA model
    qa_model = Prompt.ask("QA model", default="gpt-4o", choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"])

    # Judge model
    judge_model = Prompt.ask("Judge model (empty=skip)", default="")
    judge_model = judge_model if judge_model else None

    # Match QA pairs
    qa_items: List[Tuple[str, Any]] = []
    for conv in conversations:
        if conv.sample_id in conv_ids:
            for qa in conv.qa_pairs:
                if qa.category in categories:
                    qa_items.append((conv.sample_id, qa))
    if max_questions:
        qa_items = qa_items[:max_questions]

    console.print(f"\n[cyan]{len(qa_items)} questions matched[/cyan]")
    if not qa_items:
        return

    output: Dict[str, Any] = {
        "metadata": {
            "conv_ids": sorted(conv_ids),
            "qa_model": qa_model,
            "categories": categories,
            "max_questions": max_questions,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Run baseline
    if mode in ("baseline", "both") and state.baseline_retriever and state.baseline_retriever.index:
        console.print("\n[bold magenta]Running Baseline[/bold magenta]")
        baseline_qa = BaselineChunkQA(model_name=qa_model)
        baseline_results = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Baseline QA", total=len(qa_items))
            for conv_id, qa in qa_items:
                ctx = state.baseline_retriever.retrieve(qa.question, top_k=20)
                predicted = baseline_qa.answer(qa.question, ctx)

                gold = [str(qa.answer)] if qa.answer is not None else []
                cat = qa.category
                if not gold and cat == 5:
                    em = 1.0 if is_abstention(predicted) else 0.0
                    f1 = em
                else:
                    em = calculate_exact_match(gold, predicted) if gold else 0.0
                    f1 = calculate_f1_score(gold, predicted) if gold else 0.0

                baseline_results.append({
                    "question": qa.question,
                    "predicted": predicted,
                    "gold": str(qa.answer) if qa.answer is not None else "",
                    "category": cat,
                    "conv_id": conv_id,
                    "EM": em,
                    "F1": f1,
                })
                progress.advance(task)

        output["baseline"] = _compute_metrics(baseline_results)

    # Run agentic
    if mode in ("agentic", "both"):
        console.print("\n[bold magenta]Running Agentic[/bold magenta]")
        agent = VectorQAAgent(state.builder, model_name=qa_model)
        agentic_results = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Agentic QA", total=len(qa_items))
            for conv_id, qa in qa_items:
                try:
                    result = agent.answer(qa.question)
                    predicted = result["answer"]
                except Exception as e:
                    logger.warning("Agentic QA failed: %s", e)
                    predicted = "Error"
                    result = {"reasoning": str(e), "speaker_id": None, "evidence_turn_ids": [], "trace": []}

                gold = [str(qa.answer)] if qa.answer is not None else []
                cat = qa.category
                if not gold and cat == 5:
                    em = 1.0 if is_abstention(predicted) else 0.0
                    f1 = em
                else:
                    em = calculate_exact_match(gold, predicted) if gold else 0.0
                    f1 = calculate_f1_score(gold, predicted) if gold else 0.0

                agentic_results.append({
                    "question": qa.question,
                    "predicted": predicted,
                    "gold": str(qa.answer) if qa.answer is not None else "",
                    "category": cat,
                    "conv_id": conv_id,
                    "EM": em,
                    "F1": f1,
                    "reasoning": result.get("reasoning", ""),
                    "speaker_id": result.get("speaker_id"),
                })
                progress.advance(task)

        output["agentic"] = _compute_metrics(agentic_results)

    # LLM Judge
    if judge_model and CURATOR_AVAILABLE:
        for mode_key in ("baseline", "agentic"):
            if mode_key in output:
                per_q = output[mode_key].get("per_question", [])
                if per_q:
                    console.print(f"\n[cyan]Running LLM judge ({judge_model}) on {mode_key}...[/cyan]")
                    _llm_judge_rescore(per_q, judge_model, console)
                    output[mode_key] = _compute_metrics(per_q)

    # Save results
    eval_dir = state.work_dir / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Strip traces
    save_output = json.loads(json.dumps(output, default=str))
    for mk in ("baseline", "agentic"):
        if mk in save_output:
            for q in save_output[mk].get("per_question", []):
                q.pop("trace", None)

    output_path.write_text(json.dumps(save_output, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Results saved to {output_path}[/green]")

    # Print summary
    _print_summary(output, console)


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"overall": {"EM": 0, "F1": 0, "count": 0}, "per_category": {}, "per_question": []}

    has_llm = "llm_score" in results[0]

    overall_em = np.mean([r["EM"] for r in results])
    overall_f1 = np.mean([r["F1"] for r in results])
    overall: Dict[str, Any] = {"EM": float(overall_em), "F1": float(overall_f1), "count": len(results)}
    if has_llm:
        overall["LLM"] = float(np.mean([r["llm_score"] for r in results]))

    per_cat: Dict[int, List[Dict]] = {}
    for r in results:
        per_cat.setdefault(r["category"], []).append(r)

    per_category = {}
    for cat, items in sorted(per_cat.items()):
        cat_data: Dict[str, Any] = {
            "name": CAT_LABELS.get(cat, f"Cat-{cat}"),
            "EM": float(np.mean([r["EM"] for r in items])),
            "F1": float(np.mean([r["F1"] for r in items])),
            "count": len(items),
        }
        if has_llm:
            cat_data["LLM"] = float(np.mean([r["llm_score"] for r in items]))
        per_category[str(cat)] = cat_data

    return {"overall": overall, "per_category": per_category, "per_question": results}


def _print_summary(output: Dict[str, Any], console: Console) -> None:
    for mode in ("baseline", "agentic"):
        data = output.get(mode)
        if not data:
            continue

        has_llm = "LLM" in data.get("overall", {})

        table = Table(title=f"{mode.upper()} Results")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("EM", justify="right")
        table.add_column("F1", justify="right")
        if has_llm:
            table.add_column("LLM", justify="right")

        for cat_key, cat_data in sorted(data["per_category"].items()):
            row = [cat_data["name"], str(cat_data["count"]),
                   f"{cat_data['EM']:.4f}", f"{cat_data['F1']:.4f}"]
            if has_llm:
                row.append(f"{cat_data.get('LLM', 0):.4f}")
            table.add_row(*row)

        ov = data["overall"]
        table.add_section()
        ov_row = ["Overall", str(ov["count"]), f"{ov['EM']:.4f}", f"{ov['F1']:.4f}"]
        if has_llm:
            ov_row.append(f"{ov.get('LLM', 0):.4f}")
        table.add_row(*ov_row, style="bold")

        console.print(table)
        console.print()


def _llm_judge_rescore(
    per_question: List[Dict[str, Any]],
    judge_model: str,
    console: Console,
) -> None:
    if not CURATOR_AVAILABLE:
        console.print("[red]bespokelabs.curator not installed[/red]")
        return

    inputs = [{
        "question": q["question"],
        "predicted": q["predicted"],
        "gold": str(q["gold"]),
        "category": str(q["category"]),
    } for q in per_question]

    start = time.time()
    judge = LLMJudge(
        model_name=judge_model,
        generation_params={"temperature": 0},
    )
    results = judge(inputs)
    elapsed = time.time() - start
    console.print(f"[green]LLM judge completed in {elapsed:.1f}s[/green]")

    for q, judged in zip(per_question, results.dataset):
        q["llm_score"] = judged["llm_score"]
        q["llm_reason"] = judged["llm_reason"]


def menu_inspect(state: PipelineState, console: Console) -> None:
    while True:
        console.print()
        console.rule("[bold]Inspect Memory[/bold]")

        choice = Prompt.ask(
            "View",
            choices=["entities", "qa", "chunks", "stats", "back"],
            default="entities",
        )

        if choice == "back":
            return

        if choice == "entities":
            if not (state.builder and state.builder.entity_metadata):
                console.print("[yellow]No index loaded[/yellow]")
                continue

            filter_text = Prompt.ask("Filter by entity name (empty=all)", default="").strip()
            filtered = [
                m for m in state.builder.entity_metadata
                if filter_text.lower() in m["entity_name"].lower()
            ] if filter_text else state.builder.entity_metadata[:50]

            table = Table(title=f"Entities ({len(filtered)} shown)")
            table.add_column("Name", style="cyan")
            table.add_column("Speaker")
            table.add_column("Chunk")
            table.add_column("Roles", justify="right")
            table.add_column("Spaces")
            table.add_column("Times")

            for m in filtered[:50]:
                table.add_row(
                    m["entity_name"],
                    m.get("speaker_id", ""),
                    m["chunk_id"],
                    str(len(m.get("roles", []))),
                    ", ".join(m.get("linked_spaces", []))[:30],
                    ", ".join(m.get("linked_times", []))[:30],
                )
            console.print(table)

        elif choice == "qa":
            if not (state.builder and state.builder.qa_metadata):
                console.print("[yellow]No index loaded[/yellow]")
                continue

            filter_text = Prompt.ask("Filter by question text (empty=all)", default="").strip()
            filtered = [
                m for m in state.builder.qa_metadata
                if filter_text.lower() in m["question_text"].lower()
            ] if filter_text else state.builder.qa_metadata[:50]

            table = Table(title=f"QA Pairs ({len(filtered)} shown)")
            table.add_column("Question", style="cyan", max_width=50)
            table.add_column("Answers")
            table.add_column("VP")
            table.add_column("Speaker")
            table.add_column("Chunk")

            for m in filtered[:50]:
                table.add_row(
                    m["question_text"][:50],
                    ", ".join(m["answer_names"])[:30],
                    m["vp_phrase"][:25],
                    m.get("speaker_id", ""),
                    m["chunk_id"],
                )
            console.print(table)

        elif choice == "chunks":
            if not state.session_chunks:
                console.print("[yellow]No chunks in state[/yellow]")
                continue

            for conv_id, sessions in state.session_chunks.items():
                console.print(f"\n[bold]{conv_id}[/bold]")
                for sess_idx, chunks in enumerate(sessions):
                    console.print(f"  Session {sess_idx}: {len(chunks)} chunks")
                    for ci, chunk in enumerate(chunks):
                        lines = len(chunk.splitlines())
                        console.print(f"    Chunk {ci}: {lines} lines, {len(chunk)} chars")

            view_chunk = Confirm.ask("View a specific chunk?", default=False)
            if view_chunk:
                conv_id = Prompt.ask("Conversation ID", default=list(state.session_chunks.keys())[0])
                sess_idx = IntPrompt.ask("Session index", default=0)
                chunk_idx = IntPrompt.ask("Chunk index", default=0)
                try:
                    chunk_text = state.session_chunks[conv_id][sess_idx][chunk_idx]
                    console.print(Panel(chunk_text, title=f"{conv_id} S{sess_idx} C{chunk_idx}"))
                except (KeyError, IndexError):
                    console.print("[red]Invalid indices[/red]")

        elif choice == "stats":
            if not state.post_spacetime_gsws:
                console.print("[yellow]No GSWs in state[/yellow]")
                continue

            tree = Tree("[bold]GSW Statistics[/bold]")
            for conv_id, gsw_list in state.post_spacetime_gsws.items():
                conv_branch = tree.add(f"[cyan]{conv_id}[/cyan] ({len(gsw_list)} chunks)")
                total_e = total_vp = total_s = total_t = 0
                for gi, gsw_data in enumerate(gsw_list):
                    if isinstance(gsw_data, GSWStructure):
                        ne = len(gsw_data.entity_nodes)
                        nvp = len(gsw_data.verb_phrase_nodes)
                        ns = len(gsw_data.space_nodes)
                        nt = len(gsw_data.time_nodes)
                    else:
                        ne = len(gsw_data.get("entity_nodes", []))
                        nvp = len(gsw_data.get("verb_phrase_nodes", []))
                        ns = len(gsw_data.get("space_nodes", []))
                        nt = len(gsw_data.get("time_nodes", []))
                    total_e += ne
                    total_vp += nvp
                    total_s += ns
                    total_t += nt
                conv_branch.add(f"Entities: {total_e}, VPs: {total_vp}, Space: {total_s}, Time: {total_t}")

            console.print(tree)


def menu_save_load(state: PipelineState, console: Console) -> None:
    while True:
        console.print()
        console.rule("[bold]Save / Load[/bold]")

        choice = Prompt.ask(
            "Action",
            choices=["save", "load", "back"],
            default="back",
        )

        if choice == "back":
            return

        if choice == "save":
            save_state(state, console)

        elif choice == "load":
            # List available files
            cache_dir = state.cache_dir
            files = sorted(cache_dir.glob("*.json"), reverse=True) if cache_dir.exists() else []

            # Also check pipeline_inspector dir
            pi_dir = _REPO_ROOT / "logs" / "pipeline_inspector"
            if pi_dir.exists():
                files.extend(sorted(pi_dir.glob("*.json"), reverse=True))

            if not files:
                console.print("[yellow]No saved states found[/yellow]")
                continue

            console.print("\nAvailable states:")
            for i, f in enumerate(files[:20]):
                console.print(f"  [{i}] {f.name} ({f.parent.name}/)")

            idx_str = Prompt.ask("Select file index", default="0")
            try:
                idx = int(idx_str)
                load_state(str(files[idx]), state, console)
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")


def menu_settings(state: PipelineState, console: Console) -> None:
    while True:
        console.print()
        console.rule("[bold]Settings[/bold]")
        console.print(f"  Model: [cyan]{state.model_name}[/cyan]")
        console.print(f"  Embedding: [cyan]{state.embedding_backend}[/cyan]")
        console.print(f"  Sessions: [cyan]{state.num_sessions}[/cyan]")
        console.print(f"  Work dir: [cyan]{state.work_dir}[/cyan]")
        console.print(f"  Selected: [cyan]{state.selected_conv_labels}[/cyan]")

        choice = Prompt.ask(
            "Change",
            choices=["model", "embedding", "sessions", "conversations", "back"],
            default="back",
        )

        if choice == "back":
            return

        if choice == "model":
            state.model_name = Prompt.ask("Model", choices=MODELS, default=state.model_name)

        elif choice == "embedding":
            backends = ["openai"]
            if VLLM_AVAILABLE:
                backends.append("vllm")
            state.embedding_backend = Prompt.ask("Backend", choices=backends, default=state.embedding_backend)
            state._embed_fn = None  # Reset cached embed function

        elif choice == "sessions":
            state.num_sessions = IntPrompt.ask("Sessions per conversation", default=state.num_sessions)

        elif choice == "conversations":
            console.print("\nAvailable conversations:")
            labels = list(KNOWN_CONVS.keys())
            for i, label in enumerate(labels):
                selected = "[green]*[/green]" if label in state.selected_conv_labels else " "
                console.print(f"  {selected} [{i}] {label}")

            indices_str = Prompt.ask("Select indices (comma-separated)", default="0")
            try:
                indices = [int(x.strip()) for x in indices_str.split(",")]
                state.selected_conv_labels = [labels[i] for i in indices if i < len(labels)]
                console.print(f"[green]Selected: {state.selected_conv_labels}[/green]")
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal Memory CLI")
    parser.add_argument("--load", default=DEFAULT_STATE_PATH, help="Path to saved state JSON to load on startup")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default LLM model")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to LoCoMo JSON")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Working directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    console = Console()
    console.print(Panel(
        "[bold]Personal Memory CLI[/bold]\n"
        "Interactive pipeline, indexing, QA, and evaluation tool",
        border_style="blue",
    ))

    state = PipelineState(
        data_path=args.data_path,
        model_name=args.model,
        work_dir=Path(args.work_dir),
        cache_dir=Path(args.work_dir) / ".exploration_cache",
    )

    # Auto-load state
    load_path = args.load
    if not load_path:
        default_state = state.cache_dir / "gsw_output.json"
        if default_state.exists():
            load_path = str(default_state)

    if load_path and Path(load_path).exists():
        load_state(load_path, state, console)

        # Auto-load cached FAISS index
        if (state.cache_dir / "entity_index.faiss").exists():
            embed_fn = get_embed_fn(state)
            builder = VectorStoreBuilder(embed_fn)
            if builder.load(state.cache_dir):
                state.builder = builder
                console.print(
                    f"[green]Auto-loaded index: "
                    f"{builder.entity_index.ntotal} entities, "
                    f"{builder.qa_index.ntotal if builder.qa_index else 0} QA[/green]"
                )

            # Auto-load baseline
            if (state.cache_dir / "baseline_chunk_index.faiss").exists():
                retriever = BaselineRetriever(embed_fn)
                if retriever.load(state.cache_dir):
                    state.baseline_retriever = retriever
                    console.print(f"[green]Auto-loaded baseline index: {retriever.index.ntotal} chunks[/green]")

    while True:
        console.print()
        _show_status(state, console)
        console.print()
        menu_table = Table(show_header=False, box=None, padding=(0, 2))
        menu_table.add_column("Key", style="bold cyan", width=5)
        menu_table.add_column("Name", style="bold", width=12)
        menu_table.add_column("Description", style="dim")
        menu_table.add_row("[1]", "Pipeline", "Process conversations (coref → chunk → GSW → spacetime)")
        menu_table.add_row("[2]", "Index", "Build or load FAISS vector index from GSWs")
        menu_table.add_row("[3]", "Ask", "Interactive QA — ask questions about the memory")
        menu_table.add_row("[4]", "Eval", "Batch evaluation on LoCoMo QA pairs (EM/F1/LLM)")
        menu_table.add_row("[5]", "Inspect", "Browse entities, QA pairs, chunks, and stats")
        menu_table.add_row("[6]", "Save/Load", "Save or load pipeline state (JSON)")
        menu_table.add_row("[7]", "Settings", "Model, embedding backend, conversations")
        menu_table.add_row("[q]", "Quit", "Exit the CLI")
        console.print(menu_table)

        choice = Prompt.ask("\nSelect", choices=["1", "2", "3", "4", "5", "6", "7", "q"], default="3")

        if choice == "q":
            console.print("[dim]Goodbye![/dim]")
            break
        elif choice == "1":
            menu_pipeline(state, console)
        elif choice == "2":
            menu_index(state, console)
        elif choice == "3":
            menu_ask(state, console)
        elif choice == "4":
            menu_eval(state, console)
        elif choice == "5":
            menu_inspect(state, console)
        elif choice == "6":
            menu_save_load(state, console)
        elif choice == "7":
            menu_settings(state, console)


if __name__ == "__main__":
    main()
