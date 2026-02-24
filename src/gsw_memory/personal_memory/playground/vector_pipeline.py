"""
Vector-Store Personal Memory Pipeline — FAISS-based retrieval without reconciliation.

Loads pre-computed per-chunk GSWs (from pipeline_inspector saved state), indexes
entities and QA pairs into FAISS, and answers questions via an agentic
tool-calling QA agent.

Embedding backends:
  - OpenAI text-embedding-3-small (default, no GPU needed)
  - Qwen3-Embedding-8B via vLLM (requires vllm + GPU)

Usage:
    streamlit run src/gsw_memory/personal_memory/playground/vector_pipeline.py
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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

from gsw_memory.evaluation.hipporag_eval import (
    calculate_exact_match,
    calculate_f1_score,
)
from gsw_memory.memory.models import GSWStructure
from gsw_memory.personal_memory.data_ingestion.locomo import LoCoMoLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SAVE_DIR = _REPO_ROOT / "logs" / "pipeline_inspector"
_INDEX_DIR = _REPO_ROOT / "logs" / "vector_pipeline"
_KEY_COUNTER = 0

TAB_NAMES = ["Load", "Index", "QA", "Eval"]

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

# Embedding backend names
_BACKEND_OPENAI = "OpenAI text-embedding-3-small"
_BACKEND_VLLM = "Qwen3-Embedding-8B (vLLM)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_key(prefix: str = "ta") -> str:
    global _KEY_COUNTER
    _KEY_COUNTER += 1
    return f"{prefix}_{_KEY_COUNTER}"


def _wrap_text(content: str, height: int = 300):
    st.text_area(" ", content, height=height, disabled=True,
                 key=_next_key(), label_visibility="collapsed")


def _get_detailed_instruct(task: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f"Instruct: {task}\nQuery: {query}"


# ---------------------------------------------------------------------------
# Embedding backend factories
# ---------------------------------------------------------------------------

# Type: (texts: List[str], task: str) -> np.ndarray  (L2-normalized)
EmbedFn = Callable[[List[str], str], np.ndarray]


def make_openai_embed_fn() -> EmbedFn:
    """Create an embed function using OpenAI text-embedding-3-small."""
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
    """Create an embed function using Qwen3-Embedding-8B via vLLM."""
    if not VLLM_AVAILABLE:
        raise ImportError("vllm is not installed. Install with: pip install vllm>=0.8.5")
    model = _VllmLLM(model="Qwen/Qwen3-Embedding-8B", task="embed")

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


# ============================================================================
# VectorStoreBuilder — extract, embed, index
# ============================================================================


class VectorStoreBuilder:
    """Build FAISS indices from per-chunk GSWs.

    Extracts entities and QA pairs, embeds them with the provided embed_fn,
    and builds two FAISS IndexFlatIP indices (cosine via L2-norm + IP).
    """

    def __init__(self, embed_fn: EmbedFn):
        self._embed_fn = embed_fn

        # Entity index
        self.entity_texts: List[str] = []
        self.entity_metadata: List[Dict[str, Any]] = []
        self.entity_index: Optional[faiss.IndexFlatIP] = None

        # QA index
        self.qa_texts: List[str] = []
        self.qa_metadata: List[Dict[str, Any]] = []
        self.qa_index: Optional[faiss.IndexFlatIP] = None

    def build(
        self,
        gsw_dicts: Dict[str, List[dict]],
        session_chunks: Dict[str, List[List[str]]],
    ) -> None:
        """Build indices from post-spacetime GSW dicts.

        Args:
            gsw_dicts: ``{conv_id: [gsw_dict, ...]}`` — raw dicts from saved JSON.
            session_chunks: ``{conv_id: [[chunk_str, ...], ...]}`` per session.
        """
        self.entity_texts = []
        self.entity_metadata = []
        self.qa_texts = []
        self.qa_metadata = []

        for conv_id, gsw_list in gsw_dicts.items():
            chunks_flat = session_chunks.get(conv_id, [])
            # Build a mapping from flat GSW index → (session_idx, chunk_idx)
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

                # Build entity id→name map for resolving answer IDs
                id_to_name = {e.id: e.name for e in gsw.entity_nodes}

                # -- Entities --
                for entity in gsw.entity_nodes:
                    role_parts = []
                    for r in entity.roles:
                        if r.states:
                            role_parts.append(f"{r.role}: {', '.join(r.states)}")
                        else:
                            role_parts.append(r.role)
                    roles_text = " | ".join(role_parts) if role_parts else "no roles"
                    search_text = f"{entity.name} — {roles_text}"

                    # Collect linked space/time from edges
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

                # -- QA pairs --
                for vp in gsw.verb_phrase_nodes:
                    for q in vp.questions:
                        answer_names = [
                            id_to_name.get(a, a) for a in q.answers
                        ]
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

        # -- Embed and build FAISS --
        if self.entity_texts:
            entity_embs = self._embed_fn(self.entity_texts, ENTITY_TASK)
            self.entity_index = faiss.IndexFlatIP(entity_embs.shape[1])
            self.entity_index.add(entity_embs)

        if self.qa_texts:
            qa_embs = self._embed_fn(self.qa_texts, QA_TASK)
            self.qa_index = faiss.IndexFlatIP(qa_embs.shape[1])
            self.qa_index.add(qa_embs)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query for retrieval."""
        return self._embed_fn([query], SEARCH_TASK)

    def search_entities(
        self, query_emb: np.ndarray, top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search entity index. Returns list of (metadata, score)."""
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
        """Search QA index. Returns list of (metadata, score)."""
        if self.qa_index is None or self.qa_index.ntotal == 0:
            return []
        scores, indices = self.qa_index.search(query_emb, min(top_k, self.qa_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.qa_metadata[idx], float(score)))
        return results

    def save(self, directory: Path) -> None:
        """Save FAISS indices + metadata to disk."""
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
        """Load saved FAISS indices + metadata. Returns True on success."""
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
# VectorQAAgent — agentic tool-calling QA over FAISS indices
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

Example:
Question: "What job does Caroline have?"
{
    "answer": "Counselor",
    "reasoning": "search_entities('Caroline') found entity in multiple chunks. \
get_entity_detail('Caroline') showed role 'counselor' with states \
['pursuing counseling certification'], speaker_id Caroline, evidence D1:9 and D1:11.",
    "speaker_id": "Caroline",
    "evidence_turn_ids": ["D1:9", "D1:11"]
}

Do NOT include phrases like "The answer is" or "Based on my search" in the answer field. \
If you cannot find sufficient evidence, set answer to "I don't know" and explain why in reasoning.
"""


class _VectorTools:
    """Tools wrapping FAISS search + metadata for the QA agent."""

    def __init__(self, builder: VectorStoreBuilder):
        self._builder = builder

    # -- Tool 1: search_entities --
    def search_entities(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """FAISS semantic search on the entity index."""
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

    # -- Tool 2: search_questions --
    def search_questions(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """FAISS semantic search on the QA pair index."""
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

    # -- Tool 3: get_entity_detail --
    def get_entity_detail(self, entity_name: str) -> Dict[str, Any]:
        """Find ALL occurrences of an entity by name across all chunks.

        Returns full roles/states/spacetime per occurrence, plus QA pairs
        where this entity is an answer.
        """
        name_lower = entity_name.lower()
        occurrences = []
        entity_chunks: Dict[str, set] = {}  # chunk_id -> set of entity IDs

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

        # QA pairs from same chunks where entity is an answer
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

    # -- Tool 4: search_spacetime --
    def search_spacetime(self, entity_names: List[str]) -> Dict[str, Any]:
        """Matrix/connection view of spacetime links for given entities.

        Groups time/space nodes by value+chunk, showing which of the queried
        entity names are linked to each node.
        """
        # For each entity name, collect matching metadata entries
        name_to_entries: Dict[str, List[Dict]] = {}
        for ename in entity_names:
            el = ename.lower()
            name_to_entries[ename] = [
                m for m in self._builder.entity_metadata
                if el in m["entity_name"].lower()
            ]

        # Group time connections: (time_value, chunk_id) -> set of entity names
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
                    "speaker_id, chunk_id, role_count, and similarity score. "
                    "Use get_entity_detail(entity_name) for full details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default 10)",
                        },
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
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default 10)",
                        },
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
                    "linked_spaces, linked_times per occurrence, plus QA pairs "
                    "where the entity is an answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Entity name to look up (case-insensitive substring match)",
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
                    "Returns a matrix showing which entities share the same "
                    "times or locations. Use for 'when'/'where' questions: "
                    "pass relevant entity names to see their temporal and "
                    "spatial connections grouped by time/space node."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of entity names to find spacetime connections for",
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
    """Agentic QA agent over FAISS-indexed personal memory."""

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

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a question using FAISS retrieval + agentic reasoning."""
        tools = _VectorTools(self.builder)
        tool_defs = tools.tool_definitions()
        trace: List[Dict[str, Any]] = []

        messages: List[Any] = [
            {"role": "user", "content": f"Question: {question}"},
        ]

        for iteration in range(self.max_iterations):
            response = self._client.responses.create(
                model=self.model_name,
                input=messages,
                tools=tool_defs,
                instructions=_VECTOR_QA_SYSTEM,
                temperature=0,
            )

            messages += response.output

            # Extract agent text
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

            # Collect function calls
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
                    result = tools.dispatch(name, args)
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

            # No more tool calls → parse final answer
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
# Streamlit UI
# ============================================================================


def _get_embed_fn(backend: str) -> EmbedFn:
    """Get or create the embed function for the selected backend."""
    cache_key = f"embed_fn_{backend}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if backend == _BACKEND_VLLM:
        fn = make_vllm_embed_fn()
    else:
        fn = make_openai_embed_fn()

    st.session_state[cache_key] = fn
    return fn


def render_load_tab():
    """Tab 0: Load saved pipeline state JSON."""
    st.header("Load Saved Pipeline State")

    saved_files = sorted(_glob.glob(str(_SAVE_DIR / "*.json")), reverse=True)
    if not saved_files:
        st.warning("No saved pipeline states found in logs/pipeline_inspector/")
        return

    display_names = [Path(f).name for f in saved_files]
    selected_idx = st.selectbox(
        "Select saved state",
        range(len(display_names)),
        format_func=lambda i: display_names[i],
        key="load_selector",
    )

    if st.button("Load"):
        data = json.loads(Path(saved_files[selected_idx]).read_text(encoding="utf-8"))

        gsw_dicts = data.get("post_spacetime_gsws", {})
        session_chunks = data.get("session_chunks", {})

        if not gsw_dicts:
            st.error("No post_spacetime_gsws found in saved state. Run pipeline through step 4 first.")
            return

        st.session_state["gsw_dicts"] = gsw_dicts
        st.session_state["session_chunks"] = session_chunks
        st.session_state["loaded_file"] = display_names[selected_idx]
        st.session_state["conv_labels"] = data.get("_selected_conv_labels", [])
        st.rerun()

    # Show loaded state info
    if st.session_state.get("gsw_dicts"):
        st.success(f"Loaded: {st.session_state.get('loaded_file', 'unknown')}")
        gsw_dicts = st.session_state["gsw_dicts"]

        total_gsws = sum(len(v) for v in gsw_dicts.values())
        total_entities = 0
        total_qas = 0
        for conv_id, gsw_list in gsw_dicts.items():
            for gsw_dict in gsw_list:
                total_entities += len(gsw_dict.get("entity_nodes", []))
                for vp in gsw_dict.get("verb_phrase_nodes", []):
                    total_qas += len(vp.get("questions", []))

        cols = st.columns(4)
        cols[0].metric("Conversations", len(gsw_dicts))
        cols[1].metric("Chunks (GSWs)", total_gsws)
        cols[2].metric("Total Entities", total_entities)
        cols[3].metric("Total QA Pairs", total_qas)


def render_index_tab(emb_backend: str):
    """Tab 1: Build FAISS index and browse entries."""
    st.header("Vector Index")

    gsw_dicts = st.session_state.get("gsw_dicts")
    if not gsw_dicts:
        st.info("Load a saved pipeline state first (Load tab).")
        return

    builder: Optional[VectorStoreBuilder] = st.session_state.get("builder")

    if st.button("Build Index"):
        with st.spinner(f"Initializing {emb_backend}..."):
            embed_fn = _get_embed_fn(emb_backend)

        builder = VectorStoreBuilder(embed_fn)
        with st.spinner("Building FAISS indices..."):
            builder.build(gsw_dicts, st.session_state.get("session_chunks", {}))
            st.session_state["builder"] = builder
            # Save indices
            _INDEX_DIR.mkdir(parents=True, exist_ok=True)
            builder.save(_INDEX_DIR)
        st.success("Index built!")
        st.rerun()

    if builder is None:
        # Try loading from disk
        if (_INDEX_DIR / "entity_index.faiss").exists():
            st.info("Index files found on disk. Click 'Load Cached Index' or 'Build Index'.")
            if st.button("Load Cached Index"):
                with st.spinner(f"Initializing {emb_backend}..."):
                    embed_fn = _get_embed_fn(emb_backend)
                builder = VectorStoreBuilder(embed_fn)
                builder.load(_INDEX_DIR)
                st.session_state["builder"] = builder
                st.rerun()
            return
        else:
            st.info("Click 'Build Index' to create FAISS indices from loaded GSWs.")
            return

    # Show index stats
    st.subheader("Index Statistics")
    cols = st.columns(2)
    cols[0].metric("Entity Entries", builder.entity_index.ntotal if builder.entity_index else 0)
    cols[1].metric("QA Entries", builder.qa_index.ntotal if builder.qa_index else 0)

    # Browse entities
    st.subheader("Browse Entities")
    search_term = st.text_input("Filter entities by name", key="entity_filter")
    filtered = [
        m for m in builder.entity_metadata
        if search_term.lower() in m["entity_name"].lower()
    ] if search_term else builder.entity_metadata[:50]

    if filtered:
        df = pd.DataFrame([
            {
                "Name": m["entity_name"],
                "Speaker": m.get("speaker_id", ""),
                "Chunk": m["chunk_id"],
                "Roles": len(m.get("roles", [])),
                "Spaces": ", ".join(m.get("linked_spaces", [])),
                "Times": ", ".join(m.get("linked_times", [])),
            }
            for m in filtered[:100]
        ])
        st.dataframe(df, use_container_width=True)

    # Browse QA pairs
    st.subheader("Browse QA Pairs")
    qa_filter = st.text_input("Filter QA by question text", key="qa_filter")
    filtered_qa = [
        m for m in builder.qa_metadata
        if qa_filter.lower() in m["question_text"].lower()
    ] if qa_filter else builder.qa_metadata[:50]

    if filtered_qa:
        df_qa = pd.DataFrame([
            {
                "Question": m["question_text"],
                "Answers": ", ".join(m["answer_names"]),
                "VP": m["vp_phrase"],
                "Speaker": m.get("speaker_id", ""),
                "Chunk": m["chunk_id"],
            }
            for m in filtered_qa[:100]
        ])
        st.dataframe(df_qa, use_container_width=True)


def render_qa_tab(model_name: str):
    """Tab 2: Single question → agentic answer + trace."""
    st.header("Question Answering")

    builder = st.session_state.get("builder")
    if builder is None:
        st.info("Build the vector index first (Index tab).")
        return

    question = st.text_input("Ask a question", key="qa_question")
    if st.button("Answer") and question:
        agent = VectorQAAgent(builder, model_name=model_name)
        with st.spinner("Running agentic QA..."):
            result = agent.answer(question)
        st.session_state["last_qa_result"] = result

    result = st.session_state.get("last_qa_result")
    if result:
        st.subheader("Answer")
        st.markdown(f"**{result['answer']}**")
        if result.get("reasoning"):
            with st.expander("Reasoning", expanded=False):
                st.markdown(result["reasoning"])
        cols = st.columns(2)
        cols[0].metric("Speaker", result.get("speaker_id") or "N/A")
        evidence = result.get("evidence_turn_ids", [])
        cols[1].metric("Evidence Turns", ", ".join(evidence) if evidence else "N/A")

        # Agent trace
        trace = result.get("trace", [])
        if trace:
            st.subheader("Agent Trace")
            tcols = st.columns(2)
            tcols[0].metric("Iterations", len(trace))
            tcols[1].metric("Tool Calls", sum(len(e.get("tool_calls", [])) for e in trace))

            for entry in trace:
                with st.expander(
                    f"Iteration {entry['iteration']} ({len(entry.get('tool_calls', []))} tool calls)"
                ):
                    if entry.get("agent_text"):
                        st.markdown(entry["agent_text"])
                    for tc in entry.get("tool_calls", []):
                        _wrap_text(f"{tc['name']}({json.dumps(tc['args'])})", height=68)
                        with st.expander("Result", expanded=False):
                            _wrap_text(
                                json.dumps(tc["result"], indent=2, ensure_ascii=False),
                                height=200,
                            )


def render_eval_tab(model_name: str):
    """Tab 3: Batch LoCoMo QA evaluation with EM/F1 metrics."""
    st.header("Evaluation")

    builder = st.session_state.get("builder")
    if builder is None:
        st.info("Build the vector index first (Index tab).")
        return

    # Data path for LoCoMo
    data_path = st.text_input(
        "LoCoMo data path",
        value=str(_REPO_ROOT / "playground_data" / "locomo10.json"),
        key="eval_data_path",
    )

    if not Path(data_path).exists():
        st.warning(f"File not found: {data_path}")
        return

    # Load QA pairs
    loader = LoCoMoLoader(data_path)
    conversations = loader.load()

    # Select conversation
    conv_labels = [f"{c.sample_id} ({c.speaker_a} x {c.speaker_b})" for c in conversations]
    selected_label = st.selectbox("Conversation", conv_labels, key="eval_conv")
    selected_conv = conversations[conv_labels.index(selected_label)]

    # Category filter
    categories = sorted(set(q.category for q in selected_conv.qa_pairs))
    cat_labels = {
        1: "Single-hop",
        2: "Temporal",
        3: "Multi-hop",
        4: "Open-ended",
        5: "Adversarial",
    }
    selected_cats = st.multiselect(
        "Categories",
        categories,
        default=categories,
        format_func=lambda c: f"Cat {c}: {cat_labels.get(c, 'Unknown')}",
        key="eval_cats",
    )

    qa_pairs = [q for q in selected_conv.qa_pairs if q.category in selected_cats]
    st.write(f"{len(qa_pairs)} questions selected")

    if st.button("Run Evaluation") and qa_pairs:
        agent = VectorQAAgent(builder, model_name=model_name)
        results = []
        progress = st.progress(0)

        for i, qa in enumerate(qa_pairs):
            with st.spinner(f"Question {i + 1}/{len(qa_pairs)}..."):
                result = agent.answer(qa.question)
                predicted = result["answer"]
                gold = [qa.answer] if qa.answer else []

                em = calculate_exact_match(gold, predicted) if gold else 0.0
                f1 = calculate_f1_score(gold, predicted) if gold else 0.0

                results.append({
                    "question": qa.question,
                    "predicted": predicted,
                    "gold": qa.answer,
                    "category": qa.category,
                    "EM": em,
                    "F1": f1,
                    "speaker_id": result.get("speaker_id", ""),
                    "reasoning": result.get("reasoning", ""),
                })
            progress.progress((i + 1) / len(qa_pairs))

        st.session_state["eval_results"] = results

    results = st.session_state.get("eval_results")
    if results:
        df = pd.DataFrame(results)

        # Overall metrics
        st.subheader("Overall Metrics")
        cols = st.columns(2)
        cols[0].metric("Exact Match", f"{df['EM'].mean():.4f}")
        cols[1].metric("F1 Score", f"{df['F1'].mean():.4f}")

        # Per-category metrics
        st.subheader("Per-Category Metrics")
        cat_df = df.groupby("category").agg(
            Count=("EM", "count"),
            EM=("EM", "mean"),
            F1=("F1", "mean"),
        ).reset_index()
        cat_df["category_name"] = cat_df["category"].map(cat_labels)
        st.dataframe(cat_df[["category", "category_name", "Count", "EM", "F1"]], use_container_width=True)

        # Per-question results
        st.subheader("Per-Question Results")
        display_df = df[["question", "predicted", "gold", "category", "EM", "F1"]].copy()
        display_df["EM"] = display_df["EM"].apply(lambda x: "Y" if x == 1.0 else "")
        display_df["F1"] = display_df["F1"].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, use_container_width=True)


# ============================================================================
# Main
# ============================================================================


def main():
    st.set_page_config(page_title="Vector Pipeline", layout="wide")
    st.title("Vector-Store Personal Memory Pipeline")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "QA Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
            key="qa_model",
        )

        # Embedding backend selector
        backends = [_BACKEND_OPENAI]
        if VLLM_AVAILABLE:
            backends.insert(0, _BACKEND_VLLM)
        emb_backend = st.selectbox("Embedding Backend", backends, key="emb_backend")

        st.divider()

        # Status
        embed_fn_key = f"embed_fn_{emb_backend}"
        if embed_fn_key in st.session_state:
            st.success(f"Embedding: {emb_backend}")
        else:
            st.info(f"Embedding: {emb_backend} (not yet initialized)")

        builder = st.session_state.get("builder")
        if builder and builder.entity_index:
            st.success(
                f"Index: {builder.entity_index.ntotal} entities, "
                f"{builder.qa_index.ntotal if builder.qa_index else 0} QA"
            )

    # Tabs
    tabs = st.tabs(TAB_NAMES)

    with tabs[0]:
        render_load_tab()
    with tabs[1]:
        render_index_tab(emb_backend)
    with tabs[2]:
        render_qa_tab(model_name)
    with tabs[3]:
        render_eval_tab(model_name)


if __name__ == "__main__":
    main()
