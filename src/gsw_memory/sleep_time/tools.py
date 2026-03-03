"""
Tool implementations for agentic sleep-time GSW exploration.

Provides 11 tools that enable an agent to freely explore GSW structures,
find implicit multi-hop connections, and create bridge QA pairs.

Tools:
- Discovery (4): get_entity_documents, get_document_entities, get_document_overlap_matrix, preview_cross_doc_connections
- Context (2): get_entity_context, reconcile_entity_across_docs
- Bridges (2): create_bridge_qa (supports batch creation), get_bridge_statistics
- Planning (2): set_exploration_targets, get_exploration_targets
- Strategy (1): mark_entity_explored
"""

from typing import List, Dict, Any, Optional, Set, Union
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import json
import hashlib
from datetime import datetime

from ..memory.models import GSWStructure


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
        self.bridges_created: List[Dict[str, Any]] = []

        # Exploration tracking for systematic relationship coverage
        self.exploration_plans: Dict[str, Dict[str, Any]] = {}  # entity -> plan

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

    def get_document_entities(self, doc_id: str) -> List[str]:
        """
        Get list of entities mentioned in this document.

        Args:
            doc_id: Document ID (e.g., "doc_3")

        Returns:
            List of entity names

        Example:
            >>> tools.get_document_entities("doc_3")
            ["Lothair II", "Ermengarde of Tours", "Lotharingia"]
        """
        if doc_id not in self.entity_searcher.gsw_by_doc_id:
            return []

        gsw = self.entity_searcher.gsw_by_doc_id[doc_id]
        return [entity.name for entity in gsw.entity_nodes]

    def find_entity_neighbors(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities connected to this entity via verb phrases.

        Args:
            entity_name: Entity to find neighbors for
            relationship_types: Filter by relationship types (e.g., ["mother", "father"])
                               If None, returns all relationships

        Returns:
            List of neighbor dicts with entity, relationship, doc_id

        Example:
            >>> tools.find_entity_neighbors("Lothair II", relationship_types=["mother", "spouse"])
            [
                {"entity": "Ermengarde of Tours", "relationship": "mother of", "doc_id": "doc_3"},
                {"entity": "Teutberga", "relationship": "married to", "doc_id": "doc_3"}
            ]
        """
        neighbors = []
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
                    # Check if this question involves our entity
                    if entity_id in q.answers:
                        # Find the other entity in the question
                        for e in gsw.entity_nodes:
                            if e.name.lower() in q.text.lower() and e.name.lower() != entity_normalized:
                                relationship = vp.phrase

                                # Filter by relationship type if specified
                                if relationship_types:
                                    if not any(rt in relationship.lower() for rt in relationship_types):
                                        continue

                                neighbors.append({
                                    "entity": e.name,
                                    "relationship": relationship,
                                    "doc_id": doc_id
                                })

        return neighbors

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
                result = self._get_single_doc_context(entity_name, d)
                results.append(result)
            return results

        # SINGLE DOC MODE or MERGED MODE
        return self._get_single_doc_context(entity_name, doc_id)

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

        # STEP 5: Check for duplicate question
        bridge_id = f"bridge_{hashlib.md5(question.encode()).hexdigest()[:8]}"
        if any(b["bridge_id"] == bridge_id for b in self.bridges_created):
            return {
                "success": False,
                "error": f"Duplicate bridge — a bridge with this question already exists (id: {bridge_id}).",
                "hint": "This exact question was already created. Try a different angle or move on.",
                "validation": {"valid": False, "confidence": 0.0}
            }

        # STEP 6: Validation passed - create the bridge
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
            "confidence": confidence,
            "entities_involved": entities_involved or [],
            "bridge_id": bridge_id,
            "timestamp": datetime.now().isoformat(),
            "hop_count": len(source_docs)
        }

        # Store bridge
        self.bridges_created.append(bridge_data)

        return {
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
