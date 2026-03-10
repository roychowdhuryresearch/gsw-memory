"""
Tool implementations for agentic sleep-time GSW exploration.

Provides tools that enable an agent to freely explore GSW structures,
find implicit multi-hop connections, and create bridge QA pairs.

Active tools (14, registered in AgenticReconciler):
- Discovery (3): get_document_entities, get_entity_documents, search_entities
- Graph Walk (1): find_entity_neighbors
- Context (1): get_entity_context
- Bridges (2): create_bridge_qa (batch), get_bridge_statistics
- Planning (6): plan_document_exploration, begin_neighbor_focus, plan_neighbor_doc_coverage, plan_corpus_exploration, mark_neighbor_explored, get_doc_exploration_status
- Tracking (1): mark_document_explored
"""

from typing import List, Dict, Any, Optional, Set, Union
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import json
import hashlib
import logging
import re
from datetime import datetime

from ..memory.models import GSWStructure

logger = logging.getLogger(__name__)


TEXT_SIMILARITY_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "for",
    "to",
    "from",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
}


def normalize_similarity_text(value: Any) -> str:
    """Normalize free-form text for semantic dedupe/matching."""
    text = str(value or "").strip().strip("\"'`")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize_similarity_text(value: Any) -> Set[str]:
    """Tokenize normalized text with lightweight stopword filtering."""
    normalized = normalize_similarity_text(value)
    if not normalized:
        return set()
    return {
        token
        for token in normalized.split()
        if token and token not in TEXT_SIMILARITY_STOPWORDS
    }


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity in [0,1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class GSWTools:
    """Collection of tools for agentic GSW exploration."""

    def __init__(self, entity_searcher):
        """
        Initialize tools with existing EntitySearcher infrastructure.

        Args:
            entity_searcher: Initialized EntitySearcher instance with loaded GSWs
        """
        self.entity_searcher = entity_searcher
        self.explored_entities: Set[str] = set()
        self.explored_documents: Set[str] = set()
        self.bridges_created: List[Dict[str, Any]] = []

        # Exploration tracking for systematic relationship coverage
        self.exploration_plans: Dict[str, Dict[str, Any]] = {}  # entity -> plan
        self.doc_exploration_plans: Dict[str, Dict[str, Any]] = {}  # doc_id -> plan
        self.active_neighbor_focus: Optional[Dict[str, str]] = None  # one active edge focus lock
        # (doc_id, entity_name, neighbor_name, relationship) -> coverage state
        self.neighbor_doc_coverage_plans: Dict[tuple, Dict[str, Any]] = {}

        # High-level exploration queue: which entities to explore and why
        self.exploration_targets: List[Dict[str, Any]] = []

        # Build entity_to_docs mapping from entity_searcher.entities
        # entity_searcher.entities is a flat list where each entry has 'name' and 'doc_id'
        # Normalize entity names to lowercase for consistent lookups
        self.entity_to_docs = {}
        for entity in self.entity_searcher.entities:
            entity_name = entity["name"].lower()
            doc_id = entity["doc_id"]
            if entity_name not in self.entity_to_docs:
                self.entity_to_docs[entity_name] = set()
            self.entity_to_docs[entity_name].add(doc_id)

        # Build entity degree mapping
        self.entity_degrees = {
            entity: len(docs)
            for entity, docs in self.entity_to_docs.items()
        }

    def _edge_key(
        self,
        doc_id: str,
        entity_name: str,
        neighbor_name: str,
        relationship: str
    ) -> tuple:
        """Create canonical key for one source edge."""
        return (
            str(doc_id).strip().lower(),
            str(entity_name).strip().lower(),
            str(neighbor_name).strip().lower(),
            str(relationship).strip().lower(),
        )

    def _active_focus_edge_key(self) -> Optional[tuple]:
        """Return canonical edge key for the active focus lock."""
        if not self.active_neighbor_focus:
            return None
        f = self.active_neighbor_focus
        return self._edge_key(
            f["doc_id"],
            f["entity_name"],
            f["neighbor_name"],
            f["relationship"],
        )

    def _build_neighbor_doc_coverage_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal coverage state to stable response payload."""
        mandatory_docs = sorted(set(state.get("mandatory_docs", [])))
        explored_mandatory_docs = sorted(set(state.get("explored_mandatory_docs", [])))
        pending_mandatory_docs = [d for d in mandatory_docs if d not in set(explored_mandatory_docs)]
        optional_fuzzy_docs = state.get("optional_fuzzy_docs", [])

        return {
            "doc_id": state["doc_id"],
            "entity_name": state["entity_name"],
            "neighbor_name": state["neighbor_name"],
            "relationship": state["relationship"],
            "mandatory_docs": mandatory_docs,
            "optional_fuzzy_docs": optional_fuzzy_docs,
            "explored_mandatory_docs": explored_mandatory_docs,
            "pending_mandatory_docs": pending_mandatory_docs,
            "ready_to_close": len(pending_mandatory_docs) == 0,
        }

    def get_neighbor_doc_coverage_status(
        self,
        doc_id: str,
        entity_name: str,
        neighbor_name: str,
        relationship: str
    ) -> Optional[Dict[str, Any]]:
        """Get current per-edge doc coverage snapshot, if it exists."""
        key = self._edge_key(doc_id, entity_name, neighbor_name, relationship)
        state = self.neighbor_doc_coverage_plans.get(key)
        if state is None:
            return None
        return self._build_neighbor_doc_coverage_snapshot(state)

    def _mark_focus_mandatory_doc_explored(
        self,
        entity_name: str,
        doc_id: Optional[str]
    ) -> None:
        """
        Track mandatory doc coverage progress when context is read under active lock.
        """
        if not self.active_neighbor_focus or not doc_id or not isinstance(doc_id, str):
            return

        focus = self.active_neighbor_focus
        if entity_name.strip().lower() != focus["neighbor_name"].strip().lower():
            return

        key = self._active_focus_edge_key()
        if key is None:
            return
        state = self.neighbor_doc_coverage_plans.get(key)
        if state is None:
            return

        doc_value = doc_id.strip()
        if doc_value in set(state.get("mandatory_docs", [])):
            explored = set(state.get("explored_mandatory_docs", []))
            if doc_value not in explored:
                explored.add(doc_value)
                state["explored_mandatory_docs"] = sorted(explored)

    @staticmethod
    def _is_empty_context_payload(context: Any) -> bool:
        """Return True when context has no usable facts."""
        if not isinstance(context, dict):
            return True
        return not (
            context.get("qa_pairs")
            or context.get("roles")
            or context.get("states")
            or context.get("relationships")
        )

    def _resolve_optional_doc_alias_for_active_focus(
        self,
        entity_name: str,
        doc_id: Optional[str],
    ) -> Optional[str]:
        """
        Resolve a doc-scoped alias from optional fuzzy coverage for the active edge.

        This is intentionally strict: only active focus edge + optional docs + same
        requested neighbor name are eligible.
        """
        if not isinstance(doc_id, str) or not doc_id.strip():
            return None
        if not self.active_neighbor_focus:
            return None

        focus = self.active_neighbor_focus
        requested_name = str(entity_name).strip().lower()
        focus_neighbor = str(focus.get("neighbor_name", "")).strip().lower()
        if requested_name != focus_neighbor:
            return None

        key = self._active_focus_edge_key()
        if key is None:
            return None

        state = self.neighbor_doc_coverage_plans.get(key)
        if not isinstance(state, dict):
            return None

        doc_value = doc_id.strip()
        mandatory_docs = set(state.get("mandatory_docs", []) or [])
        if doc_value in mandatory_docs:
            return None

        for item in state.get("optional_fuzzy_docs", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("doc_id", "")).strip() != doc_value:
                continue
            alias_name = str(item.get("name", "")).strip()
            if alias_name and alias_name.lower() != requested_name:
                return alias_name
            return None

        return None

    # ========== DISCOVERY TOOLS (4) ==========

    def browse_entities(
        self,
        sort_by: str = "degree",
        min_docs: int = 2,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get list of entities sorted by frequency/importance.

        Args:
            sort_by: "degree" (frequency), "qa_density" (# of QA pairs), or "alphabetical"
            min_docs: Only include entities appearing in at least this many docs
            limit: Maximum number of entities to return

        Returns:
            List of entity dicts with name, num_docs, num_qa_pairs

        Example:
            >>> tools.browse_entities(sort_by="degree", min_docs=2, limit=10)
            [
                {"name": "Lothair II", "num_docs": 5, "num_qa_pairs": 23},
                {"name": "Ermengarde of Tours", "num_docs": 3, "num_qa_pairs": 12},
                ...
            ]
        """
        entities_data = []

        for entity, docs in self.entity_to_docs.items():
            if len(docs) < min_docs:
                continue

            # Count QA pairs mentioning this entity (rough estimate)
            num_qa_pairs = 0
            for doc_id in list(docs)[:5]:  # Sample first 5 docs for efficiency
                if doc_id in self.entity_searcher.gsw_by_doc_id:
                    gsw = self.entity_searcher.gsw_by_doc_id[doc_id]
                    for vp in gsw.verb_phrase_nodes:
                        for q in vp.questions:
                            if entity in q.text.lower():
                                num_qa_pairs += 1

            entities_data.append({
                "name": entity,
                "num_docs": len(docs),
                "num_qa_pairs": num_qa_pairs
            })

        # Sort
        if sort_by == "degree":
            entities_data.sort(key=lambda x: x["num_docs"], reverse=True)
        elif sort_by == "qa_density":
            entities_data.sort(key=lambda x: x["num_qa_pairs"], reverse=True)
        elif sort_by == "alphabetical":
            entities_data.sort(key=lambda x: x["name"])

        return entities_data[:limit]

    def get_entity_documents(self, entity_name: str) -> List[str]:
        """
        Get list of document IDs that mention this entity.

        Args:
            entity_name: Entity to search for (case-insensitive)

        Returns:
            List of document IDs

        Example:
            >>> tools.get_entity_documents("Lothair II")
            ["doc_3", "doc_47", "doc_89"]
        """
        entity_normalized = entity_name.strip().lower()
        docs = self.entity_to_docs.get(entity_normalized, set())
        return sorted(list(docs))

    def search_entities(
        self,
        query: str,
        top_k: int = 10,
        exclude_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entities across all documents using BM25/embedding similarity.

        Use when exact name matching (get_entity_documents) finds nothing — discovers
        fuzzy matches like "Rare" → "Rare Ltd" in other documents.

        Args:
            query: Entity name or description to search for
            top_k: Number of results to return
            exclude_doc_id: Optional doc_id to exclude from results (current document)

        Returns:
            List of dicts with name, doc_id, roles, states, score
        """
        results = self.entity_searcher.search(query, top_k=top_k * 2, verbose=False)

        # Deduplicate by (name_lower, doc_id) keeping highest score
        seen = {}
        for entity_dict, score in results:
            doc_id = entity_dict.get("doc_id", "")
            if exclude_doc_id and doc_id == exclude_doc_id:
                continue
            key = (entity_dict["name"].lower(), doc_id)
            if key not in seen or score > seen[key]["score"]:
                # Extract roles/states from entity_dict
                roles = []
                states = entity_dict.get("all_states", [])
                for role_info in entity_dict.get("roles", []):
                    if isinstance(role_info, dict):
                        roles.append(role_info.get("role", ""))
                    elif isinstance(role_info, str):
                        roles.append(role_info)
                seen[key] = {
                    "name": entity_dict["name"],
                    "doc_id": doc_id,
                    "roles": sorted(set(r for r in roles if r)),
                    "states": sorted(set(s for s in states if s)),
                    "score": round(float(score), 4)
                }

        # Sort by score descending and limit to top_k
        results_list = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return results_list[:top_k]

    def get_document_entities(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get entities in a document with their relationships and neighbor info.

        Returns entities sorted by total neighbors (most connected first).

        Args:
            doc_id: Document ID (e.g., "doc_3")

        Returns:
            List of dicts with name, roles, states, neighbors, and counts.
        """
        if doc_id not in self.entity_searcher.gsw_by_doc_id:
            return []

        gsw = self.entity_searcher.gsw_by_doc_id[doc_id]
        result = []

        for entity_node in gsw.entity_nodes:
            # Extract roles and states
            roles = []
            states = []
            for role in entity_node.roles:
                roles.append(role.role)
                states.extend(role.states)

            # Build neighbor map with QA pairs (deduplicated by entity+vp)
            neighbor_map = {}  # key: (entity_name_lower, vp_phrase_lower)

            for vp in gsw.verb_phrase_nodes:
                for q in vp.questions:
                    if entity_node.id in q.answers:
                        for e in gsw.entity_nodes:
                            if e.name.lower() in q.text.lower() and e.id != entity_node.id:
                                key = (e.name.lower(), vp.phrase.lower())
                                answer_refs = [f"{doc_id}::{aid}" for aid in q.answers]
                                qa_entry = {"question": q.text, "answer_refs": answer_refs}

                                if key not in neighbor_map:
                                    other_docs = [d for d in self.get_entity_documents(e.name) if d != doc_id]
                                    neighbor_map[key] = {
                                        "entity": e.name,
                                        "relationship": vp.phrase,
                                        "cross_doc": len(other_docs) > 0,
                                        "other_docs": other_docs,
                                        "qa_pairs": [qa_entry]
                                    }
                                else:
                                    neighbor_map[key]["qa_pairs"].append(qa_entry)

            neighbors = list(neighbor_map.values())
            cross_doc_count = len([n for n in neighbors if n["cross_doc"]])
            result.append({
                "name": entity_node.name,
                "roles": sorted(set(roles)),
                "states": sorted(set(states)),
                "neighbors": neighbors,
                "cross_doc_neighbors": cross_doc_count,
                "total_neighbors": len(neighbors)
            })

        # Sort by total_neighbors descending so most connected entities are first
        result.sort(key=lambda x: x["total_neighbors"], reverse=True)
        return result

    def find_entity_neighbors(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities connected to this entity via verb phrases, with QA pairs.

        Args:
            entity_name: Entity to find neighbors for
            relationship_types: Filter by relationship types (e.g., ["mother", "father"])
                               If None, returns all relationships

        Returns:
            List of neighbor dicts with entity, relationship, doc_id, qa_pairs

        Example:
            >>> tools.find_entity_neighbors("Master Aldric")
            [
                {
                    "entity": "Lady Margaux",
                    "relationship": "commissioned by",
                    "doc_id": "doc_5",
                    "qa_pairs": [
                        {"question": "Who commissioned Master Aldric?", "answer_refs": ["doc_5::e2"]}
                    ]
                }
            ]
        """
        # Use dict to deduplicate and merge qa_pairs for same (entity, vp, doc)
        neighbor_map = {}  # key: (entity_name_lower, vp_phrase_lower, doc_id)
        entity_normalized = entity_name.strip().lower()
        docs = self.get_entity_documents(entity_name)

        for doc_id in docs:
            if doc_id not in self.entity_searcher.gsw_by_doc_id:
                continue

            gsw = self.entity_searcher.gsw_by_doc_id[doc_id]

            # Find this entity's ID
            entity_id = None
            for e in gsw.entity_nodes:
                if e.name.lower() == entity_normalized:
                    entity_id = e.id
                    break

            if not entity_id:
                continue

            # Find verb phrases involving this entity
            for vp in gsw.verb_phrase_nodes:
                for q in vp.questions:
                    if entity_id in q.answers:
                        for e in gsw.entity_nodes:
                            if e.name.lower() in q.text.lower() and e.name.lower() != entity_normalized:
                                relationship = vp.phrase

                                if relationship_types:
                                    if not any(rt in relationship.lower() for rt in relationship_types):
                                        continue

                                key = (e.name.lower(), vp.phrase.lower(), doc_id)
                                answer_refs = [f"{doc_id}::{aid}" for aid in q.answers]
                                qa_entry = {"question": q.text, "answer_refs": answer_refs}

                                if key not in neighbor_map:
                                    neighbor_map[key] = {
                                        "entity": e.name,
                                        "relationship": relationship,
                                        "doc_id": doc_id,
                                        "qa_pairs": [qa_entry]
                                    }
                                else:
                                    neighbor_map[key]["qa_pairs"].append(qa_entry)

        return list(neighbor_map.values())

    def get_document_overlap_matrix(self, min_shared: int = 1) -> Dict[str, Any]:
        """
        Show entity overlap between document pairs.

        Returns doc pairs that share entities, sorted by overlap count descending.
        Gives the agent structural awareness of how documents relate.

        Args:
            min_shared: Minimum shared entities to include a pair (default 1)

        Returns:
            Dict with num_docs, doc_pairs (sorted by shared_count desc),
            and isolated_docs (docs sharing no entities with others).
        """
        # Build doc -> entity set mapping
        doc_entity_sets: Dict[str, Set[str]] = {}
        for entity_name, doc_ids in self.entity_to_docs.items():
            for doc_id in doc_ids:
                if doc_id not in doc_entity_sets:
                    doc_entity_sets[doc_id] = set()
                doc_entity_sets[doc_id].add(entity_name)

        doc_ids = sorted(doc_entity_sets.keys())
        connected_docs: Set[str] = set()
        doc_pairs = []

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                da, db = doc_ids[i], doc_ids[j]
                shared = doc_entity_sets[da] & doc_entity_sets[db]
                if len(shared) >= min_shared:
                    doc_pairs.append({
                        "doc_a": da,
                        "doc_b": db,
                        "shared_count": len(shared),
                        "shared_entities": sorted(shared),
                    })
                    connected_docs.add(da)
                    connected_docs.add(db)

        doc_pairs.sort(key=lambda x: x["shared_count"], reverse=True)

        all_docs = set(self.entity_searcher.gsw_by_doc_id.keys())
        isolated = sorted(all_docs - connected_docs)

        return {
            "num_docs": len(all_docs),
            "num_pairs": len(doc_pairs),
            "doc_pairs": doc_pairs,
            "isolated_docs": isolated,
        }

    def preview_cross_doc_connections(self, entity_name: str) -> Dict[str, Any]:
        """
        Quick preview of an entity's role in each document it appears in.

        Much cheaper than reconcile_entity_across_docs. Shows key verb-phrase
        relationships per doc so the agent can decide if deep exploration is worthwhile.

        Args:
            entity_name: Entity to preview

        Returns:
            Dict with per-doc relationship summaries and bridge_potential rating.
        """
        entity_normalized = entity_name.strip().lower()
        docs = self.get_entity_documents(entity_name)

        if not docs:
            return {
                "entity": entity_name,
                "num_docs": 0,
                "per_doc_preview": [],
                "unique_relationships_across_docs": 0,
                "bridge_potential": "none",
            }

        per_doc = []
        all_relationships: Set[str] = set()
        relationship_docs: Dict[str, Set[str]] = {}  # relationship -> set of doc_ids

        for doc_id in docs:
            if doc_id not in self.entity_searcher.gsw_by_doc_id:
                continue

            gsw = self.entity_searcher.gsw_by_doc_id[doc_id]

            # Find entity node
            entity_node = None
            for e in gsw.entity_nodes:
                if e.name.lower() == entity_normalized:
                    entity_node = e
                    break

            if not entity_node:
                continue

            # Extract verb phrase relationships involving this entity
            doc_relationships = []
            num_qa = 0
            for vp in gsw.verb_phrase_nodes:
                for q in vp.questions:
                    if entity_normalized in q.text.lower() or entity_node.id in q.answers:
                        num_qa += 1
                        # Build readable relationship string
                        other_entities = []
                        for ans_id in q.answers:
                            for e in gsw.entity_nodes:
                                if e.id == ans_id and e.name.lower() != entity_normalized:
                                    other_entities.append(e.name)
                        if other_entities:
                            rel = f"{vp.phrase} {', '.join(other_entities)}"
                            doc_relationships.append(rel)
                            all_relationships.add(rel)
                            if rel not in relationship_docs:
                                relationship_docs[rel] = set()
                            relationship_docs[rel].add(doc_id)

            # Deduplicate and limit
            unique_rels = list(dict.fromkeys(doc_relationships))[:5]
            per_doc.append({
                "doc_id": doc_id,
                "key_relationships": unique_rels,
                "num_qa_pairs": num_qa,
            })

        # Assess bridge potential
        # High: diverse relationships that appear in different docs
        unique_across = len(all_relationships)
        docs_with_unique_rels = sum(
            1 for rel, doc_set in relationship_docs.items()
            if len(doc_set) == 1
        )
        if len(docs) >= 2 and docs_with_unique_rels >= 2:
            potential = "high"
        elif len(docs) >= 2:
            potential = "medium"
        else:
            potential = "low"

        return {
            "entity": entity_name,
            "num_docs": len(docs),
            "per_doc_preview": per_doc,
            "unique_relationships_across_docs": unique_across,
            "bridge_potential": potential,
        }

    # ========== PLANNING TOOLS (2) ==========

    def set_exploration_targets(self, targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Register entities to explore after discovery phase.

        Call after get_document_overlap_matrix + preview_cross_doc_connections
        to lock in which entities to pursue. If called again, replaces the queue.

        Args:
            targets: List of dicts, each with:
                - entity_name (str, required): Entity to explore
                - reason (str, required): Why this entity is promising
                - priority (str, optional): "high", "medium", "low" (default: "medium")

        Returns:
            Dict with targets_set count and the queue

        Example:
            >>> tools.set_exploration_targets([
            ...     {"entity_name": "Diego Maradona", "reason": "high bridge_potential, diverse roles across 3 docs", "priority": "high"},
            ...     {"entity_name": "FC Barcelona", "reason": "shared across doc_0 and doc_5, different relationships", "priority": "medium"},
            ... ])
            {
                "targets_set": 2,
                "queue": [
                    {"entity_name": "Diego Maradona", "reason": "...", "priority": "high", "status": "pending"},
                    {"entity_name": "FC Barcelona", "reason": "...", "priority": "medium", "status": "pending"}
                ]
            }
        """
        validated = []
        for t in targets:
            if "entity_name" not in t or "reason" not in t:
                return {"error": "Each target must have 'entity_name' and 'reason' fields."}
            validated.append({
                "entity_name": t["entity_name"],
                "reason": t["reason"],
                "priority": t.get("priority", "medium"),
                "status": "pending",
                "bridges_created": 0,
            })

        self.exploration_targets = validated

        return {
            "targets_set": len(validated),
            "queue": [
                {"entity_name": t["entity_name"], "reason": t["reason"],
                 "priority": t["priority"], "status": t["status"]}
                for t in validated
            ],
        }

    def get_exploration_targets(self) -> Dict[str, Any]:
        """
        Get current exploration queue with progress.

        Call after mark_entity_explored to see what's next, or anytime
        to check remaining targets.

        Returns:
            Dict with completed/in_progress/pending lists and next_target

        Example:
            >>> tools.get_exploration_targets()
            {
                "completed": [{"entity_name": "X", "reason": "...", "bridges_created": 3}],
                "in_progress": [{"entity_name": "Y", "reason": "..."}],
                "pending": [{"entity_name": "Z", "reason": "...", "priority": "high"}],
                "total": 3,
                "completed_count": 1,
                "next_target": {"entity_name": "Z", "reason": "...", "priority": "high"}
            }
        """
        if not self.exploration_targets:
            return {
                "error": "No exploration targets set. Call set_exploration_targets first.",
                "total": 0,
            }

        completed = []
        in_progress = []
        pending = []

        for t in self.exploration_targets:
            entry = {
                "entity_name": t["entity_name"],
                "reason": t["reason"],
                "priority": t["priority"],
            }
            if t["status"] == "completed":
                entry["bridges_created"] = t["bridges_created"]
                completed.append(entry)
            elif t["status"] == "in_progress":
                in_progress.append(entry)
            else:
                pending.append(entry)

        # Next target: first pending entry
        next_target = pending[0] if pending else None

        return {
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "total": len(self.exploration_targets),
            "completed_count": len(completed),
            "next_target": next_target,
        }

    def _update_target_status(self, entity_name: str, status: str, bridges_created: int = 0) -> None:
        """Internal helper to update a target's status."""
        entity_key = entity_name.lower()
        for t in self.exploration_targets:
            if t["entity_name"].lower() == entity_key:
                t["status"] = status
                if bridges_created > 0:
                    t["bridges_created"] = bridges_created
                break

    # ========== CONTEXT TOOLS (3) ==========

    def get_entity_context(
        self,
        entity_name: str,
        doc_id: Optional[Union[str, List[str]]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get all QA pairs, roles, states, relationships for an entity.

        Args:
            entity_name: Entity to get context for
            doc_id: Single doc ID (str), list of doc IDs (List[str]), or None
                   - If str: Returns context from this doc only
                   - If List[str]: Returns list of contexts, one per doc (BATCH MODE)
                   - If None: Returns merged context across all docs

        Returns:
            - If doc_id is str: Single dict with context
            - If doc_id is List[str]: List of dicts, one per doc
            - If doc_id is None: Single merged dict across all docs

        Examples:
            >>> # Single doc
            >>> tools.get_entity_context("Lothair II", doc_id="doc_4")
            {"entity": "Lothair II", "doc_id": "doc_4", ...}

            >>> # Batch mode (multiple docs)
            >>> tools.get_entity_context("Lothair II", doc_id=["doc_0", "doc_4", "doc_6"])
            [
                {"entity": "Lothair II", "doc_id": "doc_0", ...},
                {"entity": "Lothair II", "doc_id": "doc_4", ...},
                {"entity": "Lothair II", "doc_id": "doc_6", ...}
            ]

            >>> # All docs (merged)
            >>> tools.get_entity_context("Lothair II")
            {"entity": "Lothair II", "doc_id": "merged", ...}
        """
        # BATCH MODE: Multiple docs
        if isinstance(doc_id, list):
            results = []
            for d in doc_id:
                self._mark_focus_mandatory_doc_explored(entity_name, d)
                result = self._get_single_doc_context(entity_name, d)
                results.append(result)
            return results

        # SINGLE DOC MODE or MERGED MODE
        self._mark_focus_mandatory_doc_explored(entity_name, doc_id)
        context = self._get_single_doc_context(entity_name, doc_id)
        if not isinstance(doc_id, str) or not doc_id.strip():
            return context
        if not self._is_empty_context_payload(context):
            return context

        alias_name = self._resolve_optional_doc_alias_for_active_focus(entity_name, doc_id)
        if not alias_name:
            return context

        alias_context = self._get_single_doc_context(alias_name, doc_id)
        if self._is_empty_context_payload(alias_context):
            return context

        logger.info(
            "Optional-doc alias fallback used [edge=%s doc=%s requested=%s alias=%s]",
            self._active_focus_edge_key(),
            doc_id,
            entity_name,
            alias_name,
        )
        return alias_context

    def _get_single_doc_context(
        self,
        entity_name: str,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Helper method to get context for entity in single doc or merged across all docs.

        Args:
            entity_name: Entity to get context for
            doc_id: If specified, returns context from this doc only.
                   If None, returns merged context across all docs.

        Returns:
            Dict with qa_pairs, roles, states, relationships
        """
        entity_normalized = entity_name.strip().lower()
        docs = self.get_entity_documents(entity_name)

        if not docs:
            return {
                "entity": entity_name,
                "doc_id": doc_id,
                "qa_pairs": [],
                "roles": [],
                "states": [],
                "relationships": {}
            }

        # If doc_id specified, only use that doc
        if doc_id:
            docs = [doc_id] if doc_id in docs else []

        qa_pairs = []
        roles = set()
        states = set()
        relationships = defaultdict(list)

        for doc in docs:
            if doc not in self.entity_searcher.gsw_by_doc_id:
                continue

            gsw = self.entity_searcher.gsw_by_doc_id[doc]

            # Find entity node
            entity_node = None
            for e in gsw.entity_nodes:
                if e.name.lower() == entity_normalized:
                    entity_node = e
                    break

            if not entity_node:
                continue

            # Extract roles and states
            for role in entity_node.roles:
                roles.add(role.role)
                states.update(role.states)

            # Extract QA pairs
            for vp in gsw.verb_phrase_nodes:
                for q in vp.questions:
                    # Check if entity is in question or answer
                    if entity_normalized in q.text.lower() or entity_node.id in q.answers:
                        # Get answer text
                        answer_names = []
                        for ans_id in q.answers:
                            for e in gsw.entity_nodes:
                                if e.id == ans_id:
                                    answer_names.append(e.name)

                                    # Extract relationships
                                    if e.name.lower() != entity_normalized:
                                        relationships[vp.phrase].append(e.name)

                        answer_refs = []
                        for ans_id in q.answers:
                            for e in gsw.entity_nodes:
                                if e.id == ans_id:
                                    answer_refs.append(f"{doc}::{ans_id}")

                        qa_pairs.append({
                            "question": q.text,
                            "answer": ", ".join(answer_names) if answer_names else "Unknown",
                            "answer_refs": answer_refs,
                            "doc_id": doc
                        })

        return {
            "entity": entity_name,
            "doc_id": doc_id if doc_id else "merged",
            "qa_pairs": qa_pairs,
            "roles": sorted(list(roles)),
            "states": sorted(list(states)),
            "relationships": {k: list(set(v)) for k, v in relationships.items()}
        }

    def reconcile_entity_across_docs(self, entity_name: str) -> Dict[str, Any]:
        """
        Merge all information about entity from all documents.

        Args:
            entity_name: Entity to reconcile

        Returns:
            Dict with merged view showing which docs contribute which facts

        Example:
            >>> tools.reconcile_entity_across_docs("Ermengarde of Tours")
            {
                "entity": "Ermengarde of Tours",
                "total_docs": 3,
                "docs": ["doc_3", "doc_47", "doc_89"],
                "merged_qa_pairs": [
                    {"qa": "Who is Lothair II's mother?", "answer": "Ermengarde", "source": "doc_3"},
                    {"qa": "When did Ermengarde die?", "answer": "20 March 851", "source": "doc_47"}
                ],
                "merged_roles": ["person", "nobility"],
                "merged_relationships": {
                    "mother_of": ["Lothair II"],
                    "died_on": ["20 March 851"]
                }
            }
        """
        docs = self.get_entity_documents(entity_name)

        if not docs:
            return {
                "entity": entity_name,
                "total_docs": 0,
                "docs": [],
                "merged_qa_pairs": [],
                "merged_roles": [],
                "merged_relationships": {}
            }

        merged_qa_pairs = []
        merged_roles = set()
        merged_relationships = defaultdict(set)

        for doc_id in docs:
            context = self.get_entity_context(entity_name, doc_id=doc_id)

            # Merge QA pairs
            for qa in context["qa_pairs"]:
                merged_qa_pairs.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "answer_refs": qa.get("answer_refs", []),
                    "source": doc_id
                })

            # Merge roles
            merged_roles.update(context["roles"])

            # Merge relationships
            for rel_type, entities in context["relationships"].items():
                merged_relationships[rel_type].update(entities)

        return {
            "entity": entity_name,
            "total_docs": len(docs),
            "docs": docs,
            "merged_qa_pairs": merged_qa_pairs,
            "merged_roles": sorted(list(merged_roles)),
            "merged_relationships": {k: sorted(list(v)) for k, v in merged_relationships.items()}
        }

    def search_qa_pairs(
        self,
        query: str,
        entity_filter: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search QA pairs using existing FAISS/BM25 index.

        Args:
            query: Search query
            entity_filter: Optional entity name to filter results
            top_k: Number of results to return

        Returns:
            List of QA dicts with question, answer, doc_id, score

        Example:
            >>> tools.search_qa_pairs("death date", entity_filter="Ermengarde", top_k=5)
            [
                {"question": "When did Ermengarde die?", "answer": "20 March 851",
                 "doc_id": "doc_47", "score": 0.92}
            ]
        """
        # Use existing search functionality
        results = self.entity_searcher.search_qa_pairs_direct(query, top_k=top_k)

        # Filter by entity if specified
        if entity_filter:
            entity_normalized = entity_filter.strip().lower()
            filtered_results = []
            for r in results:
                # Check question text and answer_names (list of entity names)
                question_match = entity_normalized in r.get("question", "").lower()
                answer_names = r.get("answer_names", [])
                answer_match = any(entity_normalized in name.lower() for name in answer_names)

                if question_match or answer_match:
                    filtered_results.append(r)
            results = filtered_results[:top_k]

        return results

    # ========== RELATIONSHIPS TOOL (1) ==========

    def trace_relationship_chain(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 4
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities across documents.

        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_hops: Maximum path length

        Returns:
            List of paths (each path is list of relationship steps)

        Example:
            >>> tools.trace_relationship_chain("Lothair II", "Tours", max_hops=3)
            [
                [
                    {"from": "Lothair II", "relation": "mother", "to": "Ermengarde", "doc": "doc_3"},
                    {"from": "Ermengarde", "relation": "born_in", "to": "Tours", "doc": "doc_89"}
                ]
            ]
        """
        # Simple BFS traversal
        start_normalized = start_entity.strip().lower()
        end_normalized = end_entity.strip().lower()

        # Queue: (current_entity, path, visited)
        queue = [(start_entity, [], {start_normalized})]
        paths_found = []

        while queue and len(paths_found) < 10:  # Limit to 10 paths
            current, path, visited = queue.pop(0)

            if len(path) >= max_hops:
                continue

            # Find neighbors
            neighbors = self.find_entity_neighbors(current)

            for neighbor in neighbors:
                neighbor_entity = neighbor["entity"]
                neighbor_normalized = neighbor_entity.strip().lower()

                if neighbor_normalized in visited:
                    continue

                new_path = path + [{
                    "from": current,
                    "relation": neighbor["relationship"],
                    "to": neighbor_entity,
                    "doc": neighbor["doc_id"]
                }]

                # Check if we reached target
                if neighbor_normalized == end_normalized:
                    paths_found.append(new_path)
                    continue

                # Add to queue for further exploration
                new_visited = visited | {neighbor_normalized}
                queue.append((neighbor_entity, new_path, new_visited))

        return paths_found

    # ========== BRIDGE TOOLS (3) ==========

    def _validate_entity_refs(self, refs: List[str], source_docs: set) -> tuple:
        """Validate doc_x::ey entity references. Returns (valid, resolved, errors)."""
        resolved = []
        errors = []
        for ref in refs:
            if "::" not in ref:
                errors.append(f"'{ref}' is not in doc_x::ey format")
                continue
            doc_id, entity_id = ref.split("::", 1)
            if doc_id not in source_docs:
                errors.append(f"'{ref}': doc '{doc_id}' not in source_docs")
                continue
            gsw = self.entity_searcher.gsw_by_doc_id.get(doc_id)
            if not gsw:
                errors.append(f"'{ref}': doc '{doc_id}' not found")
                continue
            entity = gsw.get_entity_by_id(entity_id)
            if not entity:
                valid_ids = [f"{doc_id}::{e.id} ({e.name})" for e in gsw.entity_nodes[:10]]
                errors.append(f"'{ref}': entity '{entity_id}' not found in {doc_id}. Valid: {valid_ids}")
                continue
            resolved.append({"ref": ref, "name": entity.name, "doc_id": doc_id})
        return len(errors) == 0, resolved, errors

    def create_bridge_qa(
        self,
        question: Optional[str] = None,
        answers: Optional[List[str]] = None,
        reverse_question: Optional[str] = None,
        reverse_answers: Optional[List[str]] = None,
        source_docs: Optional[List[str]] = None,
        reasoning: Optional[str] = None,
        confidence: float = 0.9,
        entities_involved: Optional[List[str]] = None,
        bridges: Optional[List[Dict[str, Any]]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Create one or more two-ended bridge QA pairs with automatic validation.

        Each bridge has a forward question and a reverse question (QA inversion),
        mirroring how GSW verb phrases generate bidirectional QA pairs.

        Can be used in two modes:
        1. Single bridge mode: Pass question, answers, reverse_question, reverse_answers, source_docs, reasoning
        2. Multiple bridge mode: Pass bridges list

        Args:
            question: Forward bridge question (single mode)
            answers: Entity names that answer the forward question (single mode)
            reverse_question: Reverse question viewing relationship from opposite side (single mode)
            reverse_answers: Entity names that answer the reverse question (single mode)
            source_docs: List of source document IDs (single mode)
            reasoning: How this bridge was derived (single mode)
            confidence: Confidence score (0-1) (single mode)
            entities_involved: Entities mentioned in bridge (single mode)
            bridges: List of bridge specifications (multiple mode). Each must contain:
                - question (str): Forward question
                - answers (List[str]): Entity names answering forward question
                - reverse_question (str): Reverse question
                - reverse_answers (List[str]): Entity names answering reverse question
                - source_docs (List[str]): Source document IDs
                - reasoning (str): Explanation
                - confidence (float, optional): Confidence score (default 0.9)
                - entities_involved (List[str], optional): Entities mentioned

        Returns:
            Single mode: Dict with validation results and bridge_id if successful
            Multiple mode: List of Dicts, one per bridge

        Examples:
            Single bridge:
            >>> result = tools.create_bridge_qa(
            ...     question="Where did Robert of Alsace's father die?",
            ...     answers=["Strasbourg"],
            ...     reverse_question="Whose father died in Strasbourg?",
            ...     reverse_answers=["Robert of Alsace"],
            ...     source_docs=["doc_12", "doc_28"],
            ...     reasoning="Robert's father was Duke William (doc_12). Duke William died in Strasbourg (doc_28)."
            ... )
            {"success": True, "bridge_id": "bridge_12345", "validation": {...}}

            Multiple bridges:
            >>> results = tools.create_bridge_qa(bridges=[
            ...     {
            ...         "question": "Where did Robert of Alsace's father die?",
            ...         "answers": ["Strasbourg"],
            ...         "reverse_question": "Whose father died in Strasbourg?",
            ...         "reverse_answers": ["Robert of Alsace"],
            ...         "source_docs": ["doc_12", "doc_28"],
            ...         "reasoning": "Robert's father was Duke William (doc_12). Duke William died in Strasbourg (doc_28)."
            ...     }
            ... ])
            [{"success": True, "bridge_id": "bridge_12345", ...}]
        """
        # Multiple bridge mode
        if bridges is not None:
            if not bridges:
                return [{
                    "success": False,
                    "error": "No bridges provided. Must provide at least 1 bridge.",
                    "validation": {"valid": False, "confidence": 0.0}
                }]

            if len(bridges) > 20:
                return [{
                    "success": False,
                    "error": f"Too many bridges ({len(bridges)}). Maximum is 20 bridges per call.",
                    "validation": {"valid": False, "confidence": 0.0}
                }]

            results = []
            for i, bridge_spec in enumerate(bridges):
                # Validate required fields
                required_fields = ["question", "answers", "reverse_question", "reverse_answers", "source_docs", "reasoning"]
                missing_fields = [field for field in required_fields if field not in bridge_spec]

                if missing_fields:
                    results.append({
                        "success": False,
                        "error": f"Bridge {i+1}: Missing required fields: {missing_fields}",
                        "validation": {"valid": False, "confidence": 0.0}
                    })
                    continue

                # Recursively call single bridge mode
                result = self.create_bridge_qa(
                    question=bridge_spec["question"],
                    answers=bridge_spec["answers"],
                    reverse_question=bridge_spec["reverse_question"],
                    reverse_answers=bridge_spec["reverse_answers"],
                    source_docs=bridge_spec["source_docs"],
                    reasoning=bridge_spec["reasoning"],
                    confidence=bridge_spec.get("confidence", 0.9),
                    entities_involved=bridge_spec.get("entities_involved")
                )
                results.append(result)

            return results

        # Single bridge mode - validate required parameters
        if question is None or answers is None or reverse_question is None or reverse_answers is None or source_docs is None or reasoning is None:
            return {
                "success": False,
                "error": "Single bridge mode requires question, answers, reverse_question, reverse_answers, source_docs, and reasoning parameters.",
                "validation": {"valid": False, "confidence": 0.0}
            }

        if not answers or not reverse_answers:
            return {
                "success": False,
                "error": "answers and reverse_answers must be non-empty lists.",
                "validation": {"valid": False, "confidence": 0.0}
            }

        # Continue with single bridge creation logic...
        # STEP 1: Validate that bridge spans multiple documents
        if len(source_docs) < 2:
            return {
                "success": False,
                "error": "A bridge MUST involve at least 2 documents. Single-document QA pairs are not bridges.",
                "hint": "Bridges combine information across multiple documents. Ensure your question requires facts from at least 2 different source documents.",
                "validation": {
                    "valid": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "reasoning": "Bridge must span multiple documents"
                }
            }

        # STEP 2: Check for duplicate documents
        if len(source_docs) != len(set(source_docs)):
            return {
                "success": False,
                "error": "Duplicate documents in source_docs list. Each document should appear only once.",
                "hint": "Remove duplicate document IDs from your source_docs list.",
                "validation": {
                    "valid": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "reasoning": "Duplicate documents detected"
                }
            }

        # STEP 3: Check if all source documents exist
        invalid_docs = [doc_id for doc_id in source_docs
                        if doc_id not in self.entity_searcher.gsw_by_doc_id]

        if invalid_docs:
            valid_docs = sorted(list(self.entity_searcher.gsw_by_doc_id.keys()))
            max_doc_num = max([int(d.split('_')[1]) for d in valid_docs if d.startswith('doc_')])

            return {
                "success": False,
                "error": f"Invalid source documents: {invalid_docs}. These documents do not exist in the corpus.",
                "invalid_docs": invalid_docs,
                "valid_range": f"doc_0 to doc_{max_doc_num}",
                "hint": "Use get_entity_documents(entity_name) to find which documents actually contain the entities.",
                "validation": {
                    "valid": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "reasoning": f"Invalid documents: {invalid_docs}"
                }
            }

        # STEP 4: Validate entity references (doc_x::ey format)
        source_docs_set = set(source_docs)
        ans_valid, ans_resolved, ans_errors = self._validate_entity_refs(answers, source_docs_set)
        rev_valid, rev_resolved, rev_errors = self._validate_entity_refs(reverse_answers, source_docs_set)

        all_errors = ans_errors + rev_errors
        if all_errors:
            return {
                "success": False,
                "error": f"Invalid entity references: {all_errors}",
                "hint": "Use doc_x::ey format (e.g. 'doc_21::e5'). Check get_entity_context output for entity IDs.",
                "validation": {
                    "valid": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "reasoning": f"Entity reference validation failed: {all_errors}"
                }
            }

        # STEP 5: Hard cross-doc ref gate (answers must actually touch >=2 docs)
        docs_from_refs = {
            item["doc_id"]
            for item in (ans_resolved + rev_resolved)
            if isinstance(item, dict) and item.get("doc_id")
        }
        if len(docs_from_refs) < 2:
            return {
                "success": False,
                "error": (
                    "Bridge refs are not cross-document. "
                    "answers + reverse_answers must resolve to at least 2 distinct docs."
                ),
                "hint": (
                    "Use refs grounded in at least two source docs. "
                    "Do not rely on source_docs list alone."
                ),
                "validation": {
                    "valid": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "reasoning": f"Resolved refs only touch docs={sorted(docs_from_refs)}",
                    "stage": "tool_validation",
                    "failure_codes": ["DOC_MISMATCH", "NOT_CHAIN"],
                }
            }

        # Canonicalize source_docs to only documents actually used by grounded refs.
        canonical_source_docs: List[str] = []
        seen_docs: Set[str] = set()
        for doc_id in source_docs:
            normalized_doc = str(doc_id).strip()
            if normalized_doc in docs_from_refs and normalized_doc not in seen_docs:
                canonical_source_docs.append(normalized_doc)
                seen_docs.add(normalized_doc)
        # Defensive fallback: include any resolved ref docs omitted by caller ordering.
        for doc_id in sorted(docs_from_refs):
            if doc_id not in seen_docs:
                canonical_source_docs.append(doc_id)
                seen_docs.add(doc_id)
        source_docs = canonical_source_docs

        # STEP 6: Check for duplicate question
        bridge_id = f"bridge_{hashlib.md5((question + '||' + reverse_question).encode()).hexdigest()[:8]}"
        if any(b["bridge_id"] == bridge_id for b in self.bridges_created):
            return {
                "success": False,
                "error": f"Duplicate bridge — a bridge with this question already exists (id: {bridge_id}).",
                "hint": "This exact question was already created. Try a different angle or move on.",
                "validation": {"valid": False, "confidence": 0.0}
            }

        # STEP 7: Semantic similarity signal (non-blocking, verifier decides)
        similarity_hits: List[Dict[str, Any]] = []
        source_doc_set = set(source_docs)
        forward_tokens = tokenize_similarity_text(question)
        reverse_tokens = tokenize_similarity_text(reverse_question)
        for existing in self.bridges_created:
            existing_docs = set(existing.get("source_docs", []))
            shared_docs = sorted(source_doc_set & existing_docs)
            if len(shared_docs) < 2:
                continue

            existing_forward_tokens = tokenize_similarity_text(existing.get("question", ""))
            existing_reverse_tokens = tokenize_similarity_text(existing.get("reverse_question", ""))
            sim_forward = jaccard_similarity(forward_tokens, existing_forward_tokens)
            sim_reverse = jaccard_similarity(reverse_tokens, existing_reverse_tokens)
            max_similarity = max(sim_forward, sim_reverse)
            if max_similarity >= 0.70:
                similarity_hits.append(
                    {
                        "bridge_id": existing.get("bridge_id"),
                        "max_similarity": round(max_similarity, 4),
                        "forward_similarity": round(sim_forward, 4),
                        "reverse_similarity": round(sim_reverse, 4),
                        "shared_docs": shared_docs,
                    }
                )

        similarity_hits.sort(key=lambda item: item.get("max_similarity", 0.0), reverse=True)

        # STEP 8: Validation passed - create the bridge
        all_resolved = ans_resolved + rev_resolved
        final_confidence = min(0.9, 0.7 + 0.1 * len(all_resolved))

        bridge_data = {
            "question": question,
            "answers": answers,
            "reverse_question": reverse_question,
            "reverse_answers": reverse_answers,
            "source": "sleep_time_agent",
            "source_docs": source_docs,
            "reasoning": reasoning,
            "confidence": final_confidence,
            "entities_involved": entities_involved or [],
            "bridge_id": bridge_id,
            "timestamp": datetime.now().isoformat(),
            "hop_count": len(source_docs)
        }
        if similarity_hits:
            bridge_data["similarity_signal"] = similarity_hits[:3]

        # Store bridge
        self.bridges_created.append(bridge_data)

        result = {
            "success": True,
            "bridge_id": bridge_id,
            "message": f"Bridge created successfully",
            "resolved_entities": {
                "answers": [{"ref": r["ref"], "name": r["name"]} for r in ans_resolved],
                "reverse_answers": [{"ref": r["ref"], "name": r["name"]} for r in rev_resolved],
            },
            "validation": {
                "valid": True,
                "confidence": final_confidence,
            }
        }
        if similarity_hits:
            result["validation"]["similarity_signal"] = similarity_hits[:3]
        return result

    def validate_bridge(
        self,
        question: str,
        answer: str,
        source_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Check if bridge is grounded in source QA pairs.

        Args:
            question: Bridge question
            answer: Proposed answer
            source_docs: Source documents

        Returns:
            Validation result with valid (bool), confidence, evidence

        Example:
            >>> result = tools.validate_bridge(
            ...     "When did Lothair II's mother die?",
            ...     "20 March 851",
            ...     ["doc_3", "doc_47"]
            ... )
            {"valid": True, "confidence": 0.92, "evidence": [...]}
        """
        # CRITICAL: Bridge must span multiple documents
        if len(source_docs) < 2:
            return {
                "valid": False,
                "confidence": 0.0,
                "evidence": [],
                "reasoning": "A bridge MUST involve at least 2 documents. Single-document QA pairs are not bridges.",
                "answer_found_in_source": False,
                "hint": "Bridges combine information across multiple documents. Ensure your question requires facts from at least 2 different source documents."
            }

        # Check for duplicate documents
        if len(source_docs) != len(set(source_docs)):
            return {
                "valid": False,
                "confidence": 0.0,
                "evidence": [],
                "reasoning": "Duplicate documents in source_docs list. Each document should appear only once.",
                "answer_found_in_source": False,
                "hint": "Remove duplicate document IDs from your source_docs list."
            }

        # Check if all source documents exist
        invalid_docs = [doc_id for doc_id in source_docs
                        if doc_id not in self.entity_searcher.gsw_by_doc_id]

        if invalid_docs:
            valid_docs = sorted(list(self.entity_searcher.gsw_by_doc_id.keys()))
            max_doc_num = max([int(d.split('_')[1]) for d in valid_docs if d.startswith('doc_')])

            return {
                "valid": False,
                "confidence": 0.0,
                "evidence": [],
                "reasoning": f"Invalid source documents: {invalid_docs}. These documents do not exist in the corpus.",
                "answer_found_in_source": False,
                "invalid_docs": invalid_docs,
                "valid_range": f"doc_0 to doc_{max_doc_num}",
                "hint": "Use get_entity_documents(entity_name) to find which documents actually contain the entities."
            }

        # Collect all QA pairs from source docs (now we know all docs exist)
        source_qa_pairs = []
        for doc_id in source_docs:
            gsw = self.entity_searcher.gsw_by_doc_id[doc_id]
            for vp in gsw.verb_phrase_nodes:
                for q in vp.questions:
                    answer_names = []
                    for ans_id in q.answers:
                        for e in gsw.entity_nodes:
                            if e.id == ans_id:
                                answer_names.append(e.name)

                    source_qa_pairs.append({
                        "doc": doc_id,
                        "question": q.text,
                        "answer": ", ".join(answer_names) if answer_names else "Unknown"
                    })

        # Check if answer appears in source QAs
        answer_normalized = answer.strip().lower()
        answer_found = False
        supporting_qa = []

        for qa in source_qa_pairs:
            if answer_normalized in qa["answer"].lower():
                answer_found = True
                supporting_qa.append(qa)

        # Simple heuristic validation (can be enhanced with LLM later)
        if answer_found and len(supporting_qa) > 0:
            return {
                "valid": True,
                "confidence": min(0.9, 0.7 + 0.1 * len(supporting_qa)),
                "evidence": supporting_qa,
                "reasoning": f"Answer '{answer}' found in source QA pairs",
                "answer_found_in_source": True
            }
        else:
            return {
                "valid": False,
                "confidence": 0.3,
                "evidence": [],
                "reasoning": f"Answer '{answer}' not found in source documents",
                "answer_found_in_source": False
            }

    def get_bridge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on bridges created so far.

        Returns:
            Stats dict with total_bridges, avg_confidence, coverage, etc.

        Example:
            >>> tools.get_bridge_statistics()
            {
                "total_bridges": 127,
                "avg_confidence": 0.88,
                "docs_coverage": 0.73,
                "hop_distribution": {"2-hop": 89, "3-hop": 31}
            }
        """
        if not self.bridges_created:
            return {
                "total_bridges": 0,
                "avg_confidence": 0.0,
                "docs_coverage": 0.0,
                "hop_distribution": {}
            }

        avg_confidence = np.mean([b["confidence"] for b in self.bridges_created])

        # Hop distribution
        hop_counts = Counter([b["hop_count"] for b in self.bridges_created])
        hop_distribution = {f"{k}-hop": v for k, v in hop_counts.items()}

        # Docs coverage (unique docs involved)
        docs_involved = set()
        for bridge in self.bridges_created:
            docs_involved.update(bridge["source_docs"])

        total_docs = len(self.entity_searcher.gsw_by_doc_id)
        docs_coverage = len(docs_involved) / total_docs if total_docs > 0 else 0

        return {
            "total_bridges": len(self.bridges_created),
            "avg_confidence": float(avg_confidence),
            "docs_coverage": float(docs_coverage),
            "docs_involved": len(docs_involved),
            "hop_distribution": hop_distribution,
            "entities_explored": len(self.explored_entities)
        }

    # ========== STRATEGY TOOLS (3) ==========

    def suggest_next_entity(
        self,
        strategy: str = "high_degree",
        exclude_explored: bool = True
    ) -> Optional[str]:
        """
        Suggest which entity to explore next.

        Args:
            strategy: "high_degree", "unexplored", or "high_qa_density"
            exclude_explored: Skip entities already explored

        Returns:
            Entity name to explore, or None if no entities left

        Example:
            >>> tools.suggest_next_entity(strategy="high_degree")
            "Lothair II"
        """
        entities = self.browse_entities(sort_by="degree" if strategy == "high_degree" else "qa_density", limit=1000)

        for entity_data in entities:
            entity_name = entity_data["name"]
            if exclude_explored and entity_name in self.explored_entities:
                continue
            return entity_name

        return None

    def mark_entity_explored(self, entity_name: str, num_bridges_created: int = 0) -> None:
        """
        Mark entity as explored.

        Args:
            entity_name: Entity that was explored
            num_bridges_created: Number of bridges created for this entity
        """
        self.explored_entities.add(entity_name)
        # Auto-update exploration targets queue
        self._update_target_status(entity_name, "completed", num_bridges_created)

    def mark_document_explored(self, doc_id: str, num_bridges_created: int = 0) -> Dict[str, Any]:
        """
        Mark a document as fully explored.

        Args:
            doc_id: Document ID (e.g., "doc_3")
            num_bridges_created: Number of bridges created from this document

        Returns:
            Dict with exploration progress
        """
        self.explored_documents.add(doc_id)
        all_docs = set(self.entity_searcher.gsw_by_doc_id.keys())
        remaining = sorted(all_docs - self.explored_documents)
        return {
            "doc_id": doc_id,
            "bridges_created": num_bridges_created,
            "total_explored": len(self.explored_documents),
            "total_docs": len(all_docs),
            "remaining_docs": remaining
        }

    # ========== DOCUMENT-LEVEL PLANNING TOOLS ==========

    def plan_document_exploration(self, doc_id: str) -> Dict[str, Any]:
        """
        Auto-build an exploration checklist for a document's entities and ALL their neighbors.

        Call this after get_document_entities to create a structured TODO list.
        Includes ALL entities with neighbors — use search_entities to find fuzzy cross-doc matches.

        Args:
            doc_id: Document ID (e.g., "doc_5")

        Returns:
            Dict with entities_to_explore checklist, each with pending neighbors.
        """
        entities_data = self.get_document_entities(doc_id)
        if not entities_data:
            return {"error": f"No entities found in {doc_id} or document does not exist."}

        entities_to_explore = []
        total_neighbors = 0

        for entity in entities_data:
            neighbors = []
            for neighbor in entity.get("neighbors", []):
                neighbors.append({
                    "entity": neighbor["entity"],
                    "relationship": neighbor["relationship"],
                    "other_docs": neighbor.get("other_docs", []),
                    "status": "pending"
                })

            if neighbors:
                entities_to_explore.append({
                    "name": entity["name"],
                    "roles": entity.get("roles", []),
                    "neighbors": neighbors,
                    "status": "pending"
                })
                total_neighbors += len(neighbors)

        plan = {
            "doc_id": doc_id,
            "entities_to_explore": entities_to_explore,
            "total_neighbors_to_explore": total_neighbors,
            "explored_count": 0,
            "pending_count": total_neighbors
        }

        # Store the plan keyed by doc_id
        self.doc_exploration_plans[doc_id] = plan

        return plan

    def begin_neighbor_focus(
        self,
        doc_id: str,
        entity_name: str,
        neighbor_name: str,
        relationship: str
    ) -> Dict[str, Any]:
        """
        Begin strict focus on one source-entity -> neighbor edge.

        Only one focus lock can be active at a time.
        """
        if doc_id not in self.doc_exploration_plans:
            return {"error": f"No exploration plan for '{doc_id}'. Call plan_document_exploration first."}

        if self.active_neighbor_focus is not None:
            return {
                "error": "Another neighbor focus is already active.",
                "active_focus": self.active_neighbor_focus,
                "hint": "Finish current edge and call mark_neighbor_explored to release the lock."
            }

        plan = self.doc_exploration_plans[doc_id]
        entity_key = entity_name.strip().lower()
        neighbor_key = neighbor_name.strip().lower()
        relationship_key = relationship.strip().lower()

        entity_plan = None
        for ep in plan["entities_to_explore"]:
            if ep["name"].lower() == entity_key:
                entity_plan = ep
                break
        if entity_plan is None:
            return {"error": f"Entity '{entity_name}' not found in plan for {doc_id}."}

        target_neighbor = None
        for nb in entity_plan["neighbors"]:
            if (
                nb["entity"].lower() == neighbor_key
                and nb["relationship"].lower() == relationship_key
                and nb["status"] == "pending"
            ):
                target_neighbor = nb
                break

        if target_neighbor is None:
            return {
                "error": (
                    f"Pending edge not found for entity='{entity_name}', "
                    f"neighbor='{neighbor_name}', relationship='{relationship}'."
                ),
                "hint": "Use get_doc_exploration_status to inspect pending edges exactly."
            }

        self.active_neighbor_focus = {
            "doc_id": doc_id,
            "entity_name": entity_plan["name"],
            "neighbor_name": target_neighbor["entity"],
            "relationship": target_neighbor["relationship"],
        }
        logger.info(
            "Neighbor focus lock acquired [doc=%s entity=%s neighbor=%s relationship=%s]",
            doc_id,
            entity_plan["name"],
            target_neighbor["entity"],
            target_neighbor["relationship"],
        )

        return {
            "success": True,
            "active_focus": dict(self.active_neighbor_focus),
            "message": "Neighbor focus lock acquired."
        }

    def plan_neighbor_doc_coverage(
        self,
        doc_id: str,
        entity_name: str,
        neighbor_name: str,
        relationship: str,
        include_fuzzy: bool = True,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Build per-edge doc coverage plan for one active source edge.

        Mandatory docs come from exact entity-name matches. Fuzzy docs are optional hints.
        """
        if self.active_neighbor_focus is None:
            return {
                "error": "No active neighbor focus lock.",
                "stage": "focus_guard",
                "hint": "Call begin_neighbor_focus first."
            }

        focus = self.active_neighbor_focus
        focus_key = self._edge_key(
            focus["doc_id"],
            focus["entity_name"],
            focus["neighbor_name"],
            focus["relationship"],
        )
        requested_key = self._edge_key(doc_id, entity_name, neighbor_name, relationship)
        if requested_key != focus_key:
            return {
                "error": "Coverage plan can only be created for the active focused edge.",
                "stage": "focus_guard",
                "active_focus": dict(focus),
                "hint": "Use active focus tuple exactly."
            }

        if doc_id not in self.doc_exploration_plans:
            return {"error": f"No exploration plan for '{doc_id}'. Call plan_document_exploration first."}

        if top_k < 1:
            top_k = 1

        mandatory_docs = [
            d for d in self.get_entity_documents(neighbor_name)
            if d != doc_id
        ]
        mandatory_docs = sorted(set(mandatory_docs))

        optional_fuzzy_docs: List[Dict[str, Any]] = []
        if include_fuzzy:
            fuzzy_results = self.search_entities(
                query=neighbor_name,
                top_k=top_k,
                exclude_doc_id=doc_id,
            )
            best_by_doc: Dict[str, Dict[str, Any]] = {}
            for item in fuzzy_results:
                fuzzy_doc_id = item.get("doc_id")
                if not fuzzy_doc_id or fuzzy_doc_id in mandatory_docs:
                    continue
                prev = best_by_doc.get(fuzzy_doc_id)
                if prev is None or float(item.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    best_by_doc[fuzzy_doc_id] = {
                        "doc_id": fuzzy_doc_id,
                        "name": item.get("name", ""),
                        "score": float(item.get("score", 0.0)),
                    }
            optional_fuzzy_docs = sorted(
                best_by_doc.values(),
                key=lambda x: (-x["score"], x["doc_id"])
            )

        existing = self.neighbor_doc_coverage_plans.get(requested_key, {})
        explored_existing = set(existing.get("explored_mandatory_docs", []))
        explored_mandatory_docs = sorted(
            d for d in explored_existing
            if d in set(mandatory_docs)
        )

        state = {
            "doc_id": doc_id,
            "entity_name": focus["entity_name"],
            "neighbor_name": focus["neighbor_name"],
            "relationship": focus["relationship"],
            "mandatory_docs": mandatory_docs,
            "optional_fuzzy_docs": optional_fuzzy_docs,
            "explored_mandatory_docs": explored_mandatory_docs,
            "include_fuzzy": bool(include_fuzzy),
            "top_k": int(top_k),
        }
        self.neighbor_doc_coverage_plans[requested_key] = state
        return self._build_neighbor_doc_coverage_snapshot(state)

    def plan_corpus_exploration(
        self,
        strategy: str = "max_pending_neighbors",
        limit: int = 20,
        include_completed: bool = False
    ) -> Dict[str, Any]:
        """
        Build global document queue from existing per-doc plans and completion state.
        """
        def _doc_sort_key(doc_value: str) -> tuple:
            if "_" in doc_value:
                parts = doc_value.split("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    return (0, int(parts[1]), doc_value)
            return (1, 0, doc_value)

        all_docs = sorted(self.entity_searcher.gsw_by_doc_id.keys(), key=_doc_sort_key)
        if limit < 1:
            limit = 1

        queue = []
        for doc_id in all_docs:
            plan = self.doc_exploration_plans.get(doc_id)
            is_completed = doc_id in self.explored_documents

            if is_completed:
                status = "completed"
            elif plan is None:
                status = "unplanned"
            else:
                status = "in_progress"

            pending_neighbors = int(plan.get("pending_count", 0)) if plan else 0
            pending_entities = (
                sum(1 for ep in plan.get("entities_to_explore", []) if ep.get("status") != "completed")
                if plan else 0
            )
            suggested_next_entity = None
            if plan:
                for ep in plan.get("entities_to_explore", []):
                    if ep.get("status") != "completed":
                        suggested_next_entity = ep.get("name")
                        break

            entry = {
                "doc_id": doc_id,
                "status": status,
                "pending_neighbors": pending_neighbors,
                "pending_entities": pending_entities,
                "suggested_next_entity": suggested_next_entity,
                "ready_to_complete": bool(plan and pending_neighbors == 0 and not is_completed),
            }
            if include_completed or status != "completed":
                queue.append(entry)

        if strategy != "max_pending_neighbors":
            strategy = "max_pending_neighbors"

        next_doc = None
        candidates = [q for q in queue if q["status"] != "completed"]
        if candidates:
            if any(c["status"] == "in_progress" and c["pending_neighbors"] > 0 for c in candidates):
                in_progress = [c for c in candidates if c["status"] == "in_progress"]
                next_doc = sorted(
                    in_progress,
                    key=lambda x: (-x["pending_neighbors"], x["doc_id"])
                )[0]
            elif any(c["status"] == "unplanned" for c in candidates):
                next_doc = sorted([c for c in candidates if c["status"] == "unplanned"], key=lambda x: x["doc_id"])[0]
            else:
                next_doc = sorted(candidates, key=lambda x: x["doc_id"])[0]

        queue_sorted = sorted(
            queue,
            key=lambda x: (
                0 if x["status"] == "in_progress" else (1 if x["status"] == "unplanned" else 2),
                -x["pending_neighbors"],
                x["doc_id"],
            ),
        )

        return {
            "strategy": strategy,
            "total_docs": len(all_docs),
            "returned_docs": min(limit, len(queue_sorted)),
            "queue": queue_sorted[:limit],
            "next_doc": next_doc,
            "active_focus": dict(self.active_neighbor_focus) if self.active_neighbor_focus else None,
        }

    def mark_neighbor_explored(
        self,
        doc_id: str,
        entity_name: str,
        neighbor_name: Union[str, List[str]],
        relationship: Optional[Union[str, List[str]]] = None,
        bridges_created: Union[int, List[int]] = 0
    ) -> Dict[str, Any]:
        """
        Mark one or more cross-doc neighbors as explored in a document's plan.

        Args:
            doc_id: Document ID the plan belongs to
            entity_name: Entity in the document whose neighbor was explored
            neighbor_name: Single neighbor name (str) or list of names (batch mode)
            relationship: Optional relationship for disambiguating repeated neighbor names.
                         Required when the same neighbor appears multiple times for an entity.
            bridges_created: Bridge count(s) — int for single/uniform, list for individual counts

        Returns:
            Updated status with remaining/explored counts.
        """
        if doc_id not in self.doc_exploration_plans:
            return {"error": f"No exploration plan for '{doc_id}'. Call plan_document_exploration first."}

        plan = self.doc_exploration_plans[doc_id]
        entity_key = entity_name.strip().lower()

        # Find the entity in the plan
        entity_plan = None
        for ep in plan["entities_to_explore"]:
            if ep["name"].lower() == entity_key:
                entity_plan = ep
                break

        if entity_plan is None:
            return {"error": f"Entity '{entity_name}' not found in plan for {doc_id}."}

        # Normalize to list for uniform handling
        names = neighbor_name if isinstance(neighbor_name, list) else [neighbor_name]
        rels = relationship if isinstance(relationship, list) else [relationship] * len(names)

        # Under an active lock, relationship is required to disambiguate the edge tuple.
        if self.active_neighbor_focus is not None:
            if relationship is None:
                return {
                    "error": (
                        "relationship is required while a neighbor focus lock is active."
                    ),
                    "hint": "Pass relationship exactly as returned by begin_neighbor_focus."
                }
            if isinstance(neighbor_name, list):
                return {
                    "error": "Batch mark is not allowed while a neighbor focus lock is active.",
                    "hint": "Mark the active edge with a single neighbor_name and relationship."
                }
            if not isinstance(relationship, str) or not relationship.strip():
                return {
                    "error": "relationship must be a non-empty string while focus lock is active."
                }

        # Relationship disambiguation for batch mode
        if isinstance(relationship, list) and len(relationship) != len(names):
            return {"error": f"relationship length ({len(relationship)}) must match neighbor_name length ({len(names)})"}

        if isinstance(bridges_created, list):
            if len(bridges_created) != len(names):
                return {"error": f"bridges_created length ({len(bridges_created)}) must match neighbor_name length ({len(names)})"}
            counts = bridges_created
        else:
            counts = [bridges_created] * len(names)

        marked = []
        not_found = []
        for name, rel, count in zip(names, rels, counts):
            name_key = name.strip().lower()
            rel_key = rel.strip().lower() if isinstance(rel, str) and rel.strip() else None
            found = False
            candidates = [
                nb for nb in entity_plan["neighbors"]
                if nb["entity"].lower() == name_key and nb["status"] == "pending"
            ]

            if rel_key is None and len(candidates) > 1:
                not_found.append(name)
                continue

            for nb in entity_plan["neighbors"]:
                if nb["entity"].lower() != name_key or nb["status"] != "pending":
                    continue
                if rel_key is not None and nb["relationship"].lower() != rel_key:
                    continue

                # If focus lock is active for this doc/entity, only matching edge can be marked.
                if self.active_neighbor_focus is not None:
                    f = self.active_neighbor_focus
                    if (
                        f["doc_id"] != doc_id
                        or f["entity_name"].lower() != entity_key
                        or f["neighbor_name"].lower() != nb["entity"].lower()
                        or f["relationship"].lower() != nb["relationship"].lower()
                    ):
                        continue

                    # Coverage gate: focused edge can close only after all mandatory docs are explored.
                    coverage = self.get_neighbor_doc_coverage_status(
                        doc_id=f["doc_id"],
                        entity_name=f["entity_name"],
                        neighbor_name=f["neighbor_name"],
                        relationship=f["relationship"],
                    )
                    if coverage is None:
                        return {
                            "error": (
                                "Neighbor doc coverage plan missing for active edge. "
                                "Call plan_neighbor_doc_coverage first."
                            ),
                            "stage": "coverage_guard",
                            "active_focus": dict(f),
                        }
                    if coverage["pending_mandatory_docs"]:
                        return {
                            "error": (
                                "Cannot mark active edge explored before mandatory neighbor docs are covered."
                            ),
                            "stage": "coverage_guard",
                            "pending_mandatory_docs": coverage["pending_mandatory_docs"],
                            "active_focus": dict(f),
                            "hint": "Read each pending mandatory doc with get_entity_context(neighbor, 'doc_x').",
                        }

                nb["status"] = "explored"
                nb["bridges_created"] = count
                marked.append(name)
                found = True

                # Successful mark of focused edge releases the lock.
                if self.active_neighbor_focus is not None:
                    logger.info(
                        "Neighbor focus lock released [doc=%s entity=%s neighbor=%s relationship=%s]",
                        doc_id,
                        entity_name,
                        nb["entity"],
                        nb["relationship"],
                    )
                    self.active_neighbor_focus = None
                break

            if not found:
                not_found.append(name)

        # Update entity status if all neighbors explored
        all_explored = all(nb["status"] == "explored" for nb in entity_plan["neighbors"])
        if all_explored:
            entity_plan["status"] = "completed"

        # Recount plan-level totals
        explored = sum(
            1 for ep in plan["entities_to_explore"]
            for nb in ep["neighbors"]
            if nb["status"] == "explored"
        )
        pending = plan["total_neighbors_to_explore"] - explored
        plan["explored_count"] = explored
        plan["pending_count"] = pending

        result = {
            "doc_id": doc_id,
            "entity": entity_name,
            "marked": marked,
            "explored_count": explored,
            "pending_count": pending,
        }
        if not_found:
            result["not_found"] = not_found
            if relationship is None:
                result["hint"] = "Add relationship to disambiguate repeated neighbor names."
        return result

    def get_doc_exploration_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get current exploration status for a document's plan.

        Shows which cross-doc neighbors have been explored vs still pending.

        Args:
            doc_id: Document ID to check

        Returns:
            Dict with per-entity explored/pending breakdown and overall counts.
        """
        if doc_id not in self.doc_exploration_plans:
            return {"error": f"No exploration plan for '{doc_id}'. Call plan_document_exploration first."}

        plan = self.doc_exploration_plans[doc_id]

        entities_status = []
        for ep in plan["entities_to_explore"]:
            explored = [
                {"entity": nb["entity"], "relationship": nb["relationship"], "bridges": nb.get("bridges_created", 0)}
                for nb in ep["neighbors"] if nb["status"] == "explored"
            ]
            pending = [
                {"entity": nb["entity"], "relationship": nb["relationship"], "other_docs": nb["other_docs"]}
                for nb in ep["neighbors"] if nb["status"] == "pending"
            ]
            entities_status.append({
                "name": ep["name"],
                "status": ep["status"],
                "explored": explored,
                "pending": pending
            })

        return {
            "doc_id": doc_id,
            "entities": entities_status,
            "explored_count": plan["explored_count"],
            "pending_count": plan["pending_count"],
            "total_neighbors_to_explore": plan["total_neighbors_to_explore"],
            "ready_to_complete": plan["pending_count"] == 0,
            "active_focus": dict(self.active_neighbor_focus) if self.active_neighbor_focus else None,
            "active_focus_coverage": (
                self.get_neighbor_doc_coverage_status(
                    self.active_neighbor_focus["doc_id"],
                    self.active_neighbor_focus["entity_name"],
                    self.active_neighbor_focus["neighbor_name"],
                    self.active_neighbor_focus["relationship"],
                )
                if self.active_neighbor_focus and self.active_neighbor_focus.get("doc_id") == doc_id
                else None
            ),
        }

    def plan_entity_exploration(
        self,
        entity_name: str,
        relationships: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Create exploration plan for an entity based on its relationships.

        Call this ONCE after reconcile_entity_across_docs to create TODO list.

        Args:
            entity_name: Entity being explored
            relationships: merged_relationships dict from reconcile_entity_across_docs

        Returns:
            Dict with exploration plan showing all relationships to check

        Example:
            >>> plan = tools.plan_entity_exploration(
            ...     "Lothair II",
            ...     {"married to": ["Teutberga"], "son of": ["Emperor Lothair I", "Ermengarde of Tours"]}
            ... )
            >>> plan
            {
                "entity": "Lothair II",
                "relationships_to_explore": [
                    {"name": "Teutberga", "type": "married to", "status": "pending"},
                    {"name": "Emperor Lothair I", "type": "son of", "status": "pending"},
                    {"name": "Ermengarde of Tours", "type": "son of", "status": "pending"}
                ],
                "total_relationships": 3,
                "explored_count": 0
            }
        """
        entity_key = entity_name.lower()

        # Build relationship list (deduplicated by entity name)
        relationships_list = []
        seen_names = set()
        for rel_type, entities in relationships.items():
            for related_entity in entities:
                name_key = related_entity.lower()
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    relationships_list.append({
                        "name": related_entity,
                        "type": rel_type,
                        "status": "pending",
                        "bridges_created": 0
                    })

        plan = {
            "entity": entity_name,
            "relationships_to_explore": relationships_list,
            "total_relationships": len(relationships_list),
            "explored_count": 0,
            "pending_count": len(relationships_list)
        }

        # Store plan
        self.exploration_plans[entity_key] = plan

        # Auto-update exploration targets queue
        self._update_target_status(entity_name, "in_progress")

        return plan

    def mark_relationship_explored(
        self,
        entity_name: str,
        relationship_name: Union[str, List[str]],
        bridges_created: Union[int, List[int]] = 0
    ) -> Dict[str, Any]:
        """
        Mark one or more relationships as explored in the TODO list.

        Call this after fully exploring related entity/entities across all their documents.

        Args:
            entity_name: Main entity being explored
            relationship_name: Single entity name (str) or list of entity names (List[str])
            bridges_created: Single count (int), list of counts (List[int]), or 0
                            - If relationship_name is str and bridges_created is int: single mode
                            - If relationship_name is list and bridges_created is int: same count for all
                            - If relationship_name is list and bridges_created is list: individual counts

        Returns:
            Updated exploration status

        Examples:
            >>> # Single relationship
            >>> tools.mark_relationship_explored("Lothair II", "Teutberga", bridges_created=2)
            {
                "relationship_marked": "Teutberga",
                "bridges_from_this": 2,
                "remaining": ["Emperor Lothair I", "Ermengarde of Tours"],
                "explored_count": 1,
                "pending_count": 2
            }

            >>> # Multiple relationships with individual bridge counts (BATCH MODE)
            >>> tools.mark_relationship_explored(
            ...     "Lotharingia",
            ...     ["Teutberga", "Lothair II", "Adolf I"],
            ...     bridges_created=[2, 1, 0]
            ... )
            {
                "relationships_marked": ["Teutberga", "Lothair II", "Adolf I"],
                "bridges_from_each": [2, 1, 0],
                "total_bridges": 3,
                "remaining": ["Keldachgau", "Deutz"],
                "explored_count": 3,
                "pending_count": 2,
                "completion_percentage": 60.0
            }
        """
        entity_key = entity_name.lower()

        if entity_key not in self.exploration_plans:
            return {
                "error": f"No exploration plan found for '{entity_name}'. Call plan_entity_exploration first."
            }

        plan = self.exploration_plans[entity_key]

        # BATCH MODE: Multiple relationships
        if isinstance(relationship_name, list):
            # Handle bridges_created parameter
            if isinstance(bridges_created, list):
                if len(bridges_created) != len(relationship_name):
                    return {
                        "error": f"bridges_created list length ({len(bridges_created)}) must match relationship_name list length ({len(relationship_name)})"
                    }
                bridge_counts = bridges_created
            else:
                # Same count for all relationships
                bridge_counts = [bridges_created] * len(relationship_name)

            # Mark each relationship
            marked = []
            not_found = []
            for rel_name, bridge_count in zip(relationship_name, bridge_counts):
                rel_key = rel_name.lower()
                found = False
                for rel in plan["relationships_to_explore"]:
                    if rel["name"].lower() == rel_key:
                        rel["status"] = "explored"
                        rel["bridges_created"] = bridge_count
                        marked.append(rel_name)
                        found = True
                        break
                if not found:
                    not_found.append(rel_name)

            if not_found:
                return {
                    "error": f"Relationships not found in plan: {not_found}"
                }

            # Update counts
            plan["explored_count"] = sum(1 for r in plan["relationships_to_explore"] if r["status"] == "explored")
            plan["pending_count"] = sum(1 for r in plan["relationships_to_explore"] if r["status"] == "pending")

            # Get remaining relationships
            remaining = [r["name"] for r in plan["relationships_to_explore"] if r["status"] == "pending"]

            total_bridges = sum(bridge_counts)

            return {
                "entity": entity_name,
                "relationships_marked": marked,
                "bridges_from_each": bridge_counts,
                "total_bridges": total_bridges,
                "remaining": remaining,
                "explored_count": plan["explored_count"],
                "pending_count": plan["pending_count"],
                "completion_percentage": round(100 * plan["explored_count"] / plan["total_relationships"], 1) if plan["total_relationships"] > 0 else 100.0
            }

        # SINGLE MODE: One relationship (existing behavior)
        else:
            relationship_key = relationship_name.lower()

            # Find and mark relationship
            found = False
            for rel in plan["relationships_to_explore"]:
                if rel["name"].lower() == relationship_key:
                    rel["status"] = "explored"
                    rel["bridges_created"] = bridges_created if isinstance(bridges_created, int) else bridges_created[0]
                    found = True
                    break

            if not found:
                return {
                    "error": f"Relationship '{relationship_name}' not found in plan for '{entity_name}'."
                }

            # Update counts
            plan["explored_count"] = sum(1 for r in plan["relationships_to_explore"] if r["status"] == "explored")
            plan["pending_count"] = sum(1 for r in plan["relationships_to_explore"] if r["status"] == "pending")

            # Get remaining relationships
            remaining = [r["name"] for r in plan["relationships_to_explore"] if r["status"] == "pending"]

            return {
                "entity": entity_name,
                "relationship_marked": relationship_name,
                "bridges_from_this": bridges_created if isinstance(bridges_created, int) else bridges_created[0],
                "remaining": remaining,
                "explored_count": plan["explored_count"],
                "pending_count": plan["pending_count"],
                "completion_percentage": round(100 * plan["explored_count"] / plan["total_relationships"], 1) if plan["total_relationships"] > 0 else 100.0
            }

    def get_exploration_status(self, entity_name: str) -> Dict[str, Any]:
        """
        Get current exploration status showing which relationships checked vs pending.

        Call this before mark_entity_explored to verify all relationships checked.

        Args:
            entity_name: Entity to check status for

        Returns:
            Dict with checklist showing explored vs pending relationships

        Example:
            >>> status = tools.get_exploration_status("Lothair II")
            >>> status
            {
                "entity": "Lothair II",
                "explored": [
                    {"name": "Teutberga", "type": "married to", "bridges": 2}
                ],
                "pending": [
                    {"name": "Emperor Lothair I", "type": "son of"},
                    {"name": "Ermengarde of Tours", "type": "son of"}
                ],
                "total_relationships": 3,
                "explored_count": 1,
                "pending_count": 2,
                "ready_to_complete": False
            }
        """
        entity_key = entity_name.lower()

        if entity_key not in self.exploration_plans:
            return {
                "error": f"No exploration plan found for '{entity_name}'. Call plan_entity_exploration first.",
                "entity": entity_name
            }

        plan = self.exploration_plans[entity_key]

        explored = [
            {"name": r["name"], "type": r["type"], "bridges": r["bridges_created"]}
            for r in plan["relationships_to_explore"]
            if r["status"] == "explored"
        ]

        pending = [
            {"name": r["name"], "type": r["type"]}
            for r in plan["relationships_to_explore"]
            if r["status"] == "pending"
        ]

        return {
            "entity": entity_name,
            "explored": explored,
            "pending": pending,
            "total_relationships": plan["total_relationships"],
            "explored_count": plan["explored_count"],
            "pending_count": plan["pending_count"],
            "ready_to_complete": plan["pending_count"] == 0
        }

    def get_exploration_progress(self) -> Dict[str, Any]:
        """
        Get exploration progress stats.

        Returns:
            Progress dict with entities explored, remaining, bridges created

        Example:
            >>> tools.get_exploration_progress()
            {
                "entities_explored": 342,
                "total_multi_doc_entities": 658,
                "bridges_created": 127,
                "estimated_completion": 0.52
            }
        """
        # Count multi-doc entities (entities in 2+ docs)
        multi_doc_entities = sum(1 for deg in self.entity_degrees.values() if deg >= 2)

        completion = len(self.explored_entities) / multi_doc_entities if multi_doc_entities > 0 else 0

        return {
            "entities_explored": len(self.explored_entities),
            "total_multi_doc_entities": multi_doc_entities,
            "bridges_created": len(self.bridges_created),
            "estimated_completion": float(completion)
        }
