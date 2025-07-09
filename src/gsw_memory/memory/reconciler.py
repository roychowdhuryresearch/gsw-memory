"""
GSW Memory Reconciler.

This module contains the main Reconciler class that orchestrates the reconciliation
of new GSW structures with existing global memory across multiple chunks.
"""

import json
import os
import uuid
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from .models import GSWStructure
from .reconciler_utils.matching import (
    EntityIndex,
    MatchingStrategy,
    create_matching_components,
)


class Reconciler:
    """
    Reconciles a new GSW representation with an existing global memory.

    The reconciler handles two main processes:
    1. Entity Reconciliation - Merging new entities with existing ones
    2. Question Reconciliation - Resolving unanswered questions with new information

    Formula: GSW_new = Reconciler(GSW_old, new_context)
    """

    def __init__(self, matching_approach: str = "exact", **matching_kwargs):
        """
        Initialize the reconciler with a matching approach.

        Args:
            matching_approach: Type of matching to use ("exact" or "embedding")
            **matching_kwargs: Additional parameters for the matching strategy/index
        """
        # Create strategy and index using factory
        self.matching_strategy, self.entity_index = create_matching_components(
            approach=matching_approach, **matching_kwargs
        )

        self.global_memory = None

    @classmethod
    def with_strategy(
        cls, matching_strategy: MatchingStrategy, entity_index: EntityIndex
    ) -> "Reconciler":
        """
        Create a reconciler with custom strategy and index instances.

        Args:
            matching_strategy: Custom matching strategy instance
            entity_index: Custom entity index instance

        Returns:
            Reconciler instance
        """
        reconciler = cls.__new__(cls)
        reconciler.matching_strategy = matching_strategy
        reconciler.entity_index = entity_index
        reconciler.global_memory = None
        return reconciler

    def reconcile(
        self,
        new_gsw: GSWStructure,
        chunk_id: Optional[str] = None,
        new_chunk_text: Optional[str] = None,
    ) -> GSWStructure:
        """
        Reconcile a new GSW representation with the existing global memory.

        Args:
            new_gsw: New GSW representation to be integrated
            chunk_id: Identifier for this chunk
            new_chunk_text: Optional raw text of the new chunk for question resolution

        Returns:
            Updated global memory GSW structure
        """
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())

        # Step 1: Prepare new GSW with global IDs and chunk info
        self._prepare_new_gsw(new_gsw, chunk_id)

        # Step 2: Initialize global memory if first chunk
        if self.global_memory is None:
            self.entity_index.add_entities(new_gsw.entity_nodes)
            self.global_memory = new_gsw
            return self.global_memory

        # Step 3: Entity reconciliation
        verified_entity_pairs = self.matching_strategy.reconcile_entities(
            new_gsw.entity_nodes, self.entity_index
        )

        # Step 4: Merge entities and track mappings
        entity_merge_map, space_merge_map, time_merge_map = self._merge_entities(
            verified_entity_pairs, new_gsw
        )

        # Step 5: Add unmerged entities to global memory
        self._add_unmerged_entities(new_gsw, entity_merge_map)

        # Step 6: Handle space/time nodes and edges
        self._reconcile_spacetime_nodes(
            new_gsw, space_merge_map, time_merge_map, chunk_id
        )
        self._update_spacetime_edges(
            new_gsw, entity_merge_map, space_merge_map, time_merge_map
        )

        # Step 7: Update answers in new GSW to point to merged entities
        self._update_verb_phrase_answers(new_gsw, entity_merge_map)

        # Step 8: Verb phrase reconciliation
        matched_new_verb_ids = self.matching_strategy.reconcile_verb_phrases(
            new_gsw, self.global_memory
        )

        # Step 9: Add unmatched verb phrases
        self._add_unmatched_verb_phrases(new_gsw, matched_new_verb_ids)

        # Step 10: Question reconciliation
        if verified_entity_pairs and new_chunk_text:
            matched_old_entity_ids = {
                old_entity.id for (_, old_entity) in verified_entity_pairs
            }
            self._resolve_questions(
                new_chunk_text, matched_old_entity_ids, new_gsw.entity_nodes, chunk_id
            )

        # Step 11: Handle conversation nodes and edges
        self._reconcile_conversation_nodes_and_edges(
            new_gsw, entity_merge_map, space_merge_map, time_merge_map
        )

        return self.global_memory

    def _prepare_new_gsw(self, new_gsw: GSWStructure, chunk_id: str) -> None:
        """Assign global IDs and chunk info to new GSW components."""
        id_mapping = {}

        # Process entities
        for entity in new_gsw.entity_nodes:
            original_id = entity.id
            if not entity.chunk_id:
                entity.chunk_id = chunk_id

            global_id = f"{chunk_id}::{original_id}"
            id_mapping[original_id] = global_id
            entity.id = global_id

            for role in entity.roles:
                if not role.chunk_id:
                    role.chunk_id = chunk_id

        # Process space and time nodes
        for space_node in new_gsw.space_nodes:
            original_id = space_node.id
            if not space_node.chunk_id:
                space_node.chunk_id = chunk_id
            global_id = f"{chunk_id}::{original_id}"
            id_mapping[original_id] = global_id
            space_node.id = global_id

        for time_node in new_gsw.time_nodes:
            original_id = time_node.id
            if not time_node.chunk_id:
                time_node.chunk_id = chunk_id
            global_id = f"{chunk_id}::{original_id}"
            id_mapping[original_id] = global_id
            time_node.id = global_id

        # Update space and time edges
        new_space_edges = []
        for entity_id, space_id in new_gsw.space_edges:
            global_entity_id = id_mapping.get(entity_id, entity_id)
            global_space_id = id_mapping.get(space_id, space_id)
            new_space_edges.append((global_entity_id, global_space_id))
        new_gsw.space_edges = new_space_edges

        new_time_edges = []
        for entity_id, time_id in new_gsw.time_edges:
            global_entity_id = id_mapping.get(entity_id, entity_id)
            global_time_id = id_mapping.get(time_id, time_id)
            new_time_edges.append((global_entity_id, global_time_id))
        new_gsw.time_edges = new_time_edges

        # Process verb phrases and questions
        for verb in new_gsw.verb_phrase_nodes:
            original_verb_id = verb.id
            if not verb.chunk_id:
                verb.chunk_id = chunk_id
            verb.id = f"{chunk_id}::{original_verb_id}"

            for question in verb.questions:
                original_question_id = question.id
                if not question.chunk_id:
                    question.chunk_id = chunk_id
                question.id = f"{chunk_id}::{original_question_id}"

                # Update answer references
                updated_answers = []
                for answer in question.answers:
                    global_entity_id = id_mapping.get(answer)
                    if global_entity_id:
                        updated_answers.append(global_entity_id)
                    elif (
                        "::" in answer
                        and self.global_memory
                        and self.global_memory.get_entity_by_id(answer)
                    ):
                        updated_answers.append(answer)
                    else:
                        updated_answers.append(answer)
                question.answers = updated_answers

        # Process conversation nodes and create ID mapping
        conv_id_mapping = {}
        for conv_node in new_gsw.conversation_nodes:
            original_conv_id = conv_node.get("id", "")
            if original_conv_id:
                global_conv_id = f"{chunk_id}::{original_conv_id}"
                conv_id_mapping[original_conv_id] = global_conv_id
                conv_node["id"] = global_conv_id
                if not conv_node.get("chunk_id"):
                    conv_node["chunk_id"] = chunk_id
                
                # Update participant entity IDs to global IDs
                if "participants" in conv_node:
                    updated_participants = []
                    for participant_id in conv_node["participants"]:
                        global_participant_id = id_mapping.get(participant_id, participant_id)
                        updated_participants.append(global_participant_id)
                    conv_node["participants"] = updated_participants
                
                # Update topics_entity IDs to global IDs
                if "topics_entity" in conv_node:
                    updated_topics_entity = []
                    for topic_entity_id in conv_node["topics_entity"]:
                        global_topic_entity_id = id_mapping.get(topic_entity_id, topic_entity_id)
                        updated_topics_entity.append(global_topic_entity_id)
                    conv_node["topics_entity"] = updated_topics_entity
                
                # Update participant_summaries keys to global IDs
                if "participant_summaries" in conv_node and conv_node["participant_summaries"]:
                    updated_participant_summaries = {}
                    for original_entity_id, summary in conv_node["participant_summaries"].items():
                        # Skip special keys like "TEXT:me"
                        if original_entity_id.startswith("TEXT:"):
                            updated_participant_summaries[original_entity_id] = summary
                        else:
                            global_entity_id = id_mapping.get(original_entity_id, original_entity_id)
                            updated_participant_summaries[global_entity_id] = summary
                    conv_node["participant_summaries"] = updated_participant_summaries
                
                # Update location_id and time_id to global IDs if they exist
                if "location_id" in conv_node and conv_node["location_id"]:
                    global_location_id = id_mapping.get(conv_node["location_id"], conv_node["location_id"])
                    conv_node["location_id"] = global_location_id
                
                if "time_id" in conv_node and conv_node["time_id"]:
                    global_time_id = id_mapping.get(conv_node["time_id"], conv_node["time_id"])
                    conv_node["time_id"] = global_time_id

        # Update conversation edges with global IDs
        new_conv_participant_edges = []
        for edge in new_gsw.conversation_participant_edges:
            if len(edge) >= 2:
                entity_id = id_mapping.get(edge[0], edge[0])
                conv_id = conv_id_mapping.get(edge[1], edge[1])
                new_conv_participant_edges.append([entity_id, conv_id])
        new_gsw.conversation_participant_edges = new_conv_participant_edges

        new_conv_topic_edges = []
        for edge in new_gsw.conversation_topic_edges:
            if len(edge) >= 2:
                entity_id = id_mapping.get(edge[0], edge[0])
                conv_id = conv_id_mapping.get(edge[1], edge[1])
                new_conv_topic_edges.append([entity_id, conv_id])
        new_gsw.conversation_topic_edges = new_conv_topic_edges

        new_conv_space_edges = []
        for edge in new_gsw.conversation_space_edges:
            if len(edge) >= 2:
                conv_id = conv_id_mapping.get(edge[0], edge[0])
                space_id = id_mapping.get(edge[1], edge[1])
                new_conv_space_edges.append([conv_id, space_id])
        new_gsw.conversation_space_edges = new_conv_space_edges

        new_conv_time_edges = []
        for edge in new_gsw.conversation_time_edges:
            if len(edge) >= 2:
                conv_id = conv_id_mapping.get(edge[0], edge[0])
                time_id = id_mapping.get(edge[1], edge[1])
                new_conv_time_edges.append([conv_id, time_id])
        new_gsw.conversation_time_edges = new_conv_time_edges

    def _merge_entities(self, verified_entity_pairs, new_gsw):
        """Merge entities and return mapping dictionaries."""
        merged_new_entity_ids = set()
        entity_merge_map = {}
        space_merge_map = {}
        time_merge_map = {}

        for new_entity, old_entity in verified_entity_pairs:
            if (
                new_entity.id != old_entity.id
                and new_entity.id not in merged_new_entity_ids
            ):
                self.global_memory.merge_external_entity(old_entity.id, new_entity)
                merged_new_entity_ids.add(new_entity.id)
                entity_merge_map[new_entity.id] = old_entity.id

                # Handle space/time node associations
                if self.global_memory.space_edges and new_gsw.space_edges:
                    old_space_id = None
                    new_space_id = None

                    for entity_id, space_id in self.global_memory.space_edges:
                        if entity_id == old_entity.id:
                            old_space_id = space_id

                    for entity_id, space_id in new_gsw.space_edges:
                        if entity_id == new_entity.id:
                            new_space_id = space_id

                    if old_space_id and new_space_id:
                        space_merge_map[new_space_id] = old_space_id

                if self.global_memory.time_edges and new_gsw.time_edges:
                    old_time_id = None
                    new_time_id = None

                    for entity_id, time_id in self.global_memory.time_edges:
                        if entity_id == old_entity.id:
                            old_time_id = time_id

                    for entity_id, time_id in new_gsw.time_edges:
                        if entity_id == new_entity.id:
                            new_time_id = time_id

                    if old_time_id and new_time_id:
                        time_merge_map[new_time_id] = old_time_id

        return entity_merge_map, space_merge_map, time_merge_map

    def _add_unmerged_entities(self, new_gsw, entity_merge_map):
        """Add entities that weren't merged to global memory."""
        merged_new_entity_ids = set(entity_merge_map.keys())
        unmerged_new_entities = [
            e for e in new_gsw.entity_nodes if e.id not in merged_new_entity_ids
        ]

        for entity in unmerged_new_entities:
            self.global_memory.add_entity(entity)

        self.entity_index.add_entities(unmerged_new_entities)

    def _reconcile_spacetime_nodes(
        self, new_gsw, space_merge_map, time_merge_map, chunk_id
    ):
        """Handle space and time node reconciliation."""
        # Add space/time nodes
        for space_node in new_gsw.space_nodes:
            self.global_memory.add_space_node(space_node)

        for time_node in new_gsw.time_nodes:
            self.global_memory.add_time_node(time_node)

        # Merge space/time nodes
        if space_merge_map:
            for new_space_id, old_space_id in space_merge_map.items():
                self.global_memory.merge_space_nodes(
                    target_id=old_space_id,
                    source_id=new_space_id,
                    chunk_id=chunk_id,
                )

        if time_merge_map:
            for new_time_id, old_time_id in time_merge_map.items():
                self.global_memory.merge_time_nodes(
                    target_id=old_time_id,
                    source_id=new_time_id,
                    chunk_id=chunk_id,
                )

    def _update_spacetime_edges(
        self, new_gsw, entity_merge_map, space_merge_map, time_merge_map
    ):
        """Update space and time edges with final IDs."""
        # Update space edges
        existing_global_space_edges = set(self.global_memory.space_edges)
        for original_entity_id, original_space_id in new_gsw.space_edges:
            final_entity_id = entity_merge_map.get(
                original_entity_id, original_entity_id
            )
            final_space_id = space_merge_map.get(original_space_id, original_space_id)
            final_space_edge = (final_entity_id, final_space_id)

            if final_space_edge not in existing_global_space_edges:
                self.global_memory.add_space_edge(final_entity_id, final_space_id)
                existing_global_space_edges.add(final_space_edge)

        # Update time edges
        existing_global_time_edges = set(self.global_memory.time_edges)
        for original_entity_id, original_time_id in new_gsw.time_edges:
            final_entity_id = entity_merge_map.get(
                original_entity_id, original_entity_id
            )
            final_time_id = time_merge_map.get(original_time_id, original_time_id)
            final_time_edge = (final_entity_id, final_time_id)

            if final_time_edge not in existing_global_time_edges:
                self.global_memory.add_time_edge(final_entity_id, final_time_id)
                existing_global_time_edges.add(final_time_edge)

    def _update_verb_phrase_answers(self, new_gsw, entity_merge_map):
        """Update answers in new GSW to point to merged entities."""
        for vp in new_gsw.verb_phrase_nodes:
            for q in vp.questions:
                updated_q_answers = []
                for answer_id in q.answers:
                    final_id = entity_merge_map.get(answer_id, answer_id)
                    updated_q_answers.append(final_id)
                q.answers = updated_q_answers

    def _add_unmatched_verb_phrases(self, new_gsw, matched_new_verb_ids):
        """Add verb phrases that didn't match existing ones."""
        existing_vp_ids = {vp.id for vp in self.global_memory.verb_phrase_nodes}
        for new_vp in new_gsw.verb_phrase_nodes:
            if (
                new_vp.id not in matched_new_verb_ids
                and new_vp.id not in existing_vp_ids
            ):
                self.global_memory.add_verb_phrase(new_vp)
                existing_vp_ids.add(new_vp.id)

    def _resolve_questions(
        self, new_chunk_text, matched_old_entity_ids, new_gsw_entities, chunk_id
    ):
        """Delegate question resolution to the matching strategy."""
        try:
            self.matching_strategy.resolve_questions(
                new_chunk_text=new_chunk_text,
                matched_old_entity_ids=matched_old_entity_ids,
                global_memory=self.global_memory,
                new_gsw_entities=new_gsw_entities,
                chunk_id=f"chunk_{chunk_id}",
            )
        except NotImplementedError:
            print(
                "Warning: resolve_questions not implemented for the current strategy."
            )
        except Exception as e:
            print(f"Error during question resolution: {e}")

    def _reconcile_conversation_nodes_and_edges(
        self, new_gsw, entity_merge_map, space_merge_map, time_merge_map
    ):
        """Handle conversation nodes and edges reconciliation."""
        # Add conversation nodes to global memory with updated entity references
        for conv_node in new_gsw.conversation_nodes:
            conv_id = conv_node.get("id")
            if conv_id and not self.global_memory.get_conversation_by_id(conv_id):
                # Create a copy of the conversation node to avoid modifying the original
                updated_conv_node = conv_node.copy()
                
                # Update participant entity IDs to final merged IDs
                if "participants" in updated_conv_node:
                    final_participants = []
                    for participant_id in updated_conv_node["participants"]:
                        final_participant_id = entity_merge_map.get(participant_id, participant_id)
                        final_participants.append(final_participant_id)
                    updated_conv_node["participants"] = final_participants
                
                # Update topics_entity IDs to final merged IDs
                if "topics_entity" in updated_conv_node:
                    final_topics_entity = []
                    for topic_entity_id in updated_conv_node["topics_entity"]:
                        final_topic_entity_id = entity_merge_map.get(topic_entity_id, topic_entity_id)
                        final_topics_entity.append(final_topic_entity_id)
                    updated_conv_node["topics_entity"] = final_topics_entity
                
                # Update participant_summaries keys to final merged IDs
                if "participant_summaries" in updated_conv_node and updated_conv_node["participant_summaries"]:
                    final_participant_summaries = {}
                    for entity_id, summary in updated_conv_node["participant_summaries"].items():
                        # Skip special keys like "TEXT:me"
                        if entity_id.startswith("TEXT:"):
                            final_participant_summaries[entity_id] = summary
                        else:
                            final_entity_id = entity_merge_map.get(entity_id, entity_id)
                            final_participant_summaries[final_entity_id] = summary
                    updated_conv_node["participant_summaries"] = final_participant_summaries
                
                # Update location_id and time_id to final merged IDs
                if "location_id" in updated_conv_node and updated_conv_node["location_id"]:
                    final_location_id = space_merge_map.get(updated_conv_node["location_id"], updated_conv_node["location_id"])
                    updated_conv_node["location_id"] = final_location_id
                
                if "time_id" in updated_conv_node and updated_conv_node["time_id"]:
                    final_time_id = time_merge_map.get(updated_conv_node["time_id"], updated_conv_node["time_id"])
                    updated_conv_node["time_id"] = final_time_id
                
                self.global_memory.add_conversation_node(updated_conv_node)

        # Update conversation participant edges with final entity IDs
        existing_global_conv_participant_edges = set(
            tuple(edge) for edge in self.global_memory.conversation_participant_edges
        )
        for edge in new_gsw.conversation_participant_edges:
            if len(edge) >= 2:
                final_entity_id = entity_merge_map.get(edge[0], edge[0])
                conv_id = edge[1]
                final_edge = [final_entity_id, conv_id]
                
                if tuple(final_edge) not in existing_global_conv_participant_edges:
                    self.global_memory.add_conversation_participant_edge(
                        final_entity_id, conv_id
                    )
                    existing_global_conv_participant_edges.add(tuple(final_edge))

        # Update conversation topic edges with final entity IDs
        existing_global_conv_topic_edges = set(
            tuple(edge) for edge in self.global_memory.conversation_topic_edges
        )
        for edge in new_gsw.conversation_topic_edges:
            if len(edge) >= 2:
                final_entity_id = entity_merge_map.get(edge[0], edge[0])
                conv_id = edge[1]
                final_edge = [final_entity_id, conv_id]
                
                if tuple(final_edge) not in existing_global_conv_topic_edges:
                    self.global_memory.add_conversation_topic_edge(
                        final_entity_id, conv_id
                    )
                    existing_global_conv_topic_edges.add(tuple(final_edge))

        # Update conversation space edges with final space IDs
        existing_global_conv_space_edges = set(
            tuple(edge) for edge in self.global_memory.conversation_space_edges
        )
        for edge in new_gsw.conversation_space_edges:
            if len(edge) >= 2:
                conv_id = edge[0]
                final_space_id = space_merge_map.get(edge[1], edge[1])
                final_edge = [conv_id, final_space_id]
                
                if tuple(final_edge) not in existing_global_conv_space_edges:
                    self.global_memory.add_conversation_space_edge(
                        conv_id, final_space_id
                    )
                    existing_global_conv_space_edges.add(tuple(final_edge))

        # Update conversation time edges with final time IDs
        existing_global_conv_time_edges = set(
            tuple(edge) for edge in self.global_memory.conversation_time_edges
        )
        for edge in new_gsw.conversation_time_edges:
            if len(edge) >= 2:
                conv_id = edge[0]
                final_time_id = time_merge_map.get(edge[1], edge[1])
                final_edge = [conv_id, final_time_id]
                
                if tuple(final_edge) not in existing_global_conv_time_edges:
                    self.global_memory.add_conversation_time_edge(
                        conv_id, final_time_id
                    )
                    existing_global_conv_time_edges.add(tuple(final_edge))

    def get_statistics(self) -> dict:
        """Get statistics about the current global memory."""
        if not self.global_memory:
            return {"entities": 0, "verb_phrases": 0, "questions": 0}

        total_questions = sum(
            len(vp.questions) for vp in self.global_memory.verb_phrase_nodes
        )

        # Count entities with temporal evolution
        entities_with_evolution = 0
        total_roles = 0

        for entity in self.global_memory.entity_nodes:
            chunk_ids = set(role.chunk_id for role in entity.roles if role.chunk_id)
            if len(chunk_ids) > 1:
                entities_with_evolution += 1
            total_roles += len(entity.roles)

        return {
            "entities": len(self.global_memory.entity_nodes),
            "verb_phrases": len(self.global_memory.verb_phrase_nodes),
            "questions": total_questions,
            "entities_with_evolution": entities_with_evolution,
            "total_roles": total_roles,
            "avg_roles_per_entity": total_roles / len(self.global_memory.entity_nodes)
            if self.global_memory.entity_nodes
            else 0,
            "space_nodes": len(self.global_memory.space_nodes),
            "time_nodes": len(self.global_memory.time_nodes),
            "space_edges": len(self.global_memory.space_edges),
            "time_edges": len(self.global_memory.time_edges),
            "conversation_nodes": len(self.global_memory.conversation_nodes),
            "conversation_participant_edges": len(self.global_memory.conversation_participant_edges),
            "conversation_topic_edges": len(self.global_memory.conversation_topic_edges),
            "conversation_space_edges": len(self.global_memory.conversation_space_edges),
            "conversation_time_edges": len(self.global_memory.conversation_time_edges),
        }


# Integration Functions


def _extract_chunk_data(processor_outputs: List[Dict[str, Dict]]) -> List[Dict]:
    """
    Extract chunk data from GSWProcessor output format.

    Args:
        processor_outputs: List of document outputs from GSWProcessor.process_documents()

    Returns:
        List of chunk data dictionaries with keys: chunk_id, gsw, text, doc_idx, chunk_idx
    """
    all_chunks = []

    for doc_idx, doc_chunks in enumerate(processor_outputs):
        for chunk_id, chunk_data in doc_chunks.items():
            if chunk_data.get("gsw") is not None:  # Only include chunks with valid GSW
                extracted_chunk = {
                    "chunk_id": chunk_id,
                    "gsw": chunk_data["gsw"],
                    "text": chunk_data.get("text", ""),
                    "doc_idx": chunk_data.get("doc_idx", doc_idx),
                    "chunk_idx": chunk_data.get("chunk_idx", 0),
                }
                all_chunks.append(extracted_chunk)

    return all_chunks


def _create_reconciler(
    matching_approach: str = "exact", **reconciler_kwargs
) -> Reconciler:
    """
    Create a new Reconciler instance with specified matching approach.

    Args:
        matching_approach: "exact" or "embedding"
        **reconciler_kwargs: Additional arguments for the reconciler

    Returns:
        Configured Reconciler instance
    """
    return Reconciler(matching_approach=matching_approach, **reconciler_kwargs)


def _reconcile_document_chunks(
    reconciler: Reconciler, doc_chunks: List[Dict]
) -> GSWStructure:
    """
    Reconcile all chunks from a single document through the provided reconciler.

    Args:
        reconciler: Reconciler instance to use
        doc_chunks: List of chunk data for a single document

    Returns:
        Reconciled GSWStructure for the document
    """
    for chunk_data in tqdm(doc_chunks, desc="Reconciling chunks"):
        reconciler.reconcile(
            new_gsw=chunk_data["gsw"],
            chunk_id=chunk_data["chunk_id"],
            new_chunk_text=chunk_data["text"],
        )

    return reconciler.global_memory


def _reconcile_all_chunks(
    reconciler: Reconciler, all_chunks: List[Dict]
) -> GSWStructure:
    """
    Reconcile all chunks from all documents through the provided reconciler.

    Args:
        reconciler: Reconciler instance to use
        all_chunks: List of all chunk data from all documents

    Returns:
        Reconciled GSWStructure across all documents
    """
    for chunk_data in all_chunks:
        reconciler.reconcile(
            new_gsw=chunk_data["gsw"],
            chunk_id=chunk_data["chunk_id"],
            new_chunk_text=chunk_data["text"],
        )

    return reconciler.global_memory


def _save_reconciled_outputs(
    reconciled_results: Union[List[GSWStructure], GSWStructure],
    strategy: str,
    matching_approach: str,
    output_dir: str,
    processor_outputs: List[Dict[str, Dict]],
    all_chunks: List[Dict],
    save_statistics: bool,
    enable_visualization: bool,
) -> None:
    """
    Save reconciled GSW outputs with comprehensive metadata and statistics.

    Args:
        reconciled_results: Reconciled GSW structure(s)
        strategy: Reconciliation strategy used
        matching_approach: Matching approach used
        output_dir: Directory to save outputs
        processor_outputs: Original processor outputs for statistics
        all_chunks: All chunk data for statistics
        save_statistics: Whether to save statistics
        enable_visualization: Whether to create visualizations
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    reconciled_dir = os.path.join(output_dir, "reconciled")
    os.makedirs(reconciled_dir, exist_ok=True)

    if save_statistics:
        stats_dir = os.path.join(output_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

    if enable_visualization:
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

    # Create metadata
    metadata = {
        "reconciliation_metadata": {
            "strategy": strategy,
            "matching_approach": matching_approach,
            "processed_at": datetime.now().isoformat(),
            "total_input_chunks": len(all_chunks),
            "total_documents": len(processor_outputs),
        }
    }

    # Save metadata
    with open(os.path.join(reconciled_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if strategy == "local":
        # Save individual reconciled GSWs
        assert isinstance(reconciled_results, list)

        per_doc_stats = {}
        for doc_idx, reconciled_gsw in enumerate(reconciled_results):
            # Save reconciled GSW
            gsw_file = os.path.join(reconciled_dir, f"doc_{doc_idx}_reconciled.json")
            with open(gsw_file, "w") as f:
                json.dump(reconciled_gsw.model_dump(mode="json"), f, indent=2)

            # Calculate statistics for this document
            if save_statistics:
                doc_chunks = [
                    chunk for chunk in all_chunks if chunk["doc_idx"] == doc_idx
                ]
                input_entities = sum(
                    len(chunk["gsw"].entity_nodes) for chunk in doc_chunks
                )

                # Count entities with evolution (roles from multiple chunks)
                entities_with_evolution = 0
                for entity in reconciled_gsw.entity_nodes:
                    chunk_ids = set(
                        role.chunk_id for role in entity.roles if role.chunk_id
                    )
                    if len(chunk_ids) > 1:
                        entities_with_evolution += 1

                per_doc_stats[f"doc_{doc_idx}"] = {
                    "input_chunks": len(doc_chunks),
                    "input_entities": input_entities,
                    "reconciled_entities": len(reconciled_gsw.entity_nodes),
                    "compression_ratio": input_entities
                    / len(reconciled_gsw.entity_nodes)
                    if reconciled_gsw.entity_nodes
                    else 0,
                    "entities_with_evolution": entities_with_evolution,
                    "verb_phrases": len(reconciled_gsw.verb_phrase_nodes),
                    "questions": sum(
                        len(vp.questions) for vp in reconciled_gsw.verb_phrase_nodes
                    ),
                    "total_roles": sum(
                        len(entity.roles) for entity in reconciled_gsw.entity_nodes
                    ),
                    "avg_roles_per_entity": sum(
                        len(entity.roles) for entity in reconciled_gsw.entity_nodes
                    )
                    / len(reconciled_gsw.entity_nodes)
                    if reconciled_gsw.entity_nodes
                    else 0,
                }

                # Save per-document statistics
                doc_stats_file = os.path.join(stats_dir, f"doc_{doc_idx}_stats.json")
                with open(doc_stats_file, "w") as f:
                    json.dump(per_doc_stats[f"doc_{doc_idx}"], f, indent=2)

            # Create visualization if requested
            if enable_visualization:
                try:
                    from ..utils.visualization import create_and_save_gsw_visualization

                    viz_file = os.path.join(viz_dir, f"doc_{doc_idx}_reconciled.cyjs")
                    create_and_save_gsw_visualization(reconciled_gsw, viz_file)
                except ImportError:
                    print(
                        "Warning: Visualization not available. Install NetworkX for visualizations."
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to create visualization for doc_{doc_idx}: {e}"
                    )

        # Save overall statistics
        if save_statistics:
            total_input_entities = sum(
                len(chunk["gsw"].entity_nodes) for chunk in all_chunks
            )
            total_reconciled_entities = sum(
                len(gsw.entity_nodes) for gsw in reconciled_results
            )
            total_entities_with_evolution = sum(
                per_doc_stats[f"doc_{i}"]["entities_with_evolution"]
                for i in range(len(reconciled_results))
            )

            overall_stats = {
                "reconciliation_metadata": metadata["reconciliation_metadata"],
                "per_document_stats": per_doc_stats,
                "overall_stats": {
                    "total_input_entities": total_input_entities,
                    "total_reconciled_entities": total_reconciled_entities,
                    "overall_compression_ratio": total_input_entities
                    / total_reconciled_entities
                    if total_reconciled_entities
                    else 0,
                    "cross_chunk_evolution_entities": total_entities_with_evolution,
                    "total_verb_phrases": sum(
                        len(gsw.verb_phrase_nodes) for gsw in reconciled_results
                    ),
                    "total_questions": sum(
                        sum(len(vp.questions) for vp in gsw.verb_phrase_nodes)
                        for gsw in reconciled_results
                    ),
                },
            }

            summary_file = os.path.join(stats_dir, "reconciliation_summary.json")
            with open(summary_file, "w") as f:
                json.dump(overall_stats, f, indent=2)

    elif strategy == "global":
        # Save single global reconciled GSW
        assert isinstance(reconciled_results, GSWStructure)

        gsw_file = os.path.join(reconciled_dir, "global_reconciled.json")
        with open(gsw_file, "w") as f:
            json.dump(reconciled_results.model_dump(mode="json"), f, indent=2)

        # Calculate global statistics
        if save_statistics:
            total_input_entities = sum(
                len(chunk["gsw"].entity_nodes) for chunk in all_chunks
            )

            # Count entities with evolution
            entities_with_evolution = 0
            for entity in reconciled_results.entity_nodes:
                chunk_ids = set(role.chunk_id for role in entity.roles if role.chunk_id)
                if len(chunk_ids) > 1:
                    entities_with_evolution += 1

            global_stats = {
                "reconciliation_metadata": metadata["reconciliation_metadata"],
                "global_stats": {
                    "total_input_chunks": len(all_chunks),
                    "total_input_entities": total_input_entities,
                    "reconciled_entities": len(reconciled_results.entity_nodes),
                    "compression_ratio": total_input_entities
                    / len(reconciled_results.entity_nodes)
                    if reconciled_results.entity_nodes
                    else 0,
                    "entities_with_evolution": entities_with_evolution,
                    "verb_phrases": len(reconciled_results.verb_phrase_nodes),
                    "questions": sum(
                        len(vp.questions) for vp in reconciled_results.verb_phrase_nodes
                    ),
                    "total_roles": sum(
                        len(entity.roles) for entity in reconciled_results.entity_nodes
                    ),
                    "avg_roles_per_entity": sum(
                        len(entity.roles) for entity in reconciled_results.entity_nodes
                    )
                    / len(reconciled_results.entity_nodes)
                    if reconciled_results.entity_nodes
                    else 0,
                },
            }

            summary_file = os.path.join(stats_dir, "reconciliation_summary.json")
            with open(summary_file, "w") as f:
                json.dump(global_stats, f, indent=2)

        # Create visualization if requested
        if enable_visualization:
            try:
                from ..utils.visualization import create_and_save_gsw_visualization

                viz_file = os.path.join(viz_dir, "global_reconciled.cyjs")
                create_and_save_gsw_visualization(reconciled_results, viz_file)
            except ImportError:
                print(
                    "Warning: Visualization not available. Install NetworkX for visualizations."
                )
            except Exception as e:
                print(f"Warning: Failed to create global visualization: {e}")

    print(f"Reconciled outputs saved to: {output_dir}")
    if save_statistics:
        print(f"  - Statistics: {stats_dir}")
    if enable_visualization:
        print(f"  - Visualizations: {viz_dir}")
    print(f"  - Reconciled GSWs: {reconciled_dir}")


def reconcile_gsw_outputs(
    processor_outputs: List[Dict[str, Dict]],
    strategy: str = "local",
    matching_approach: str = "exact",
    output_dir: Optional[str] = None,
    save_statistics: bool = True,
    enable_visualization: bool = False,
    **reconciler_kwargs,
) -> Union[List[GSWStructure], GSWStructure]:
    """
    Reconcile GSWProcessor outputs using different reconciliation strategies.

    Args:
        processor_outputs: Output from GSWProcessor.process_documents()
        strategy: Reconciliation strategy - "local" or "global"
        matching_approach: Entity matching approach - "exact" or "embedding"
        output_dir: Directory to save reconciled outputs (if None, no saving)
        save_statistics: Whether to save reconciliation statistics
        enable_visualization: Whether to create Cytoscape visualizations
        **reconciler_kwargs: Additional arguments passed to Reconciler initialization

    Returns:
        - For "local" strategy: List[GSWStructure] - one reconciled GSW per document
        - For "global" strategy: GSWStructure - single unified GSW across all documents

    Raises:
        ValueError: If strategy is not "local" or "global"
    """
    if strategy not in ["local", "global"]:
        raise ValueError(f"Strategy must be 'local' or 'global', got '{strategy}'")

    # Extract all chunk data from processor outputs
    all_chunks = _extract_chunk_data(processor_outputs)

    if not all_chunks:
        print("Warning: No valid chunks found in processor outputs")
        return [] if strategy == "local" else None

    print(
        f"Found {len(all_chunks)} valid chunks across {len(processor_outputs)} documents"
    )

    # Initialize result variables
    reconciled_documents = None
    reconciled_gsw = None

    if strategy == "local":
        # Document-level reconciliation: one reconciler per document
        reconciled_documents = []

        for doc_idx in range(len(processor_outputs)):
            # Get chunks for this document
            doc_chunks = [chunk for chunk in all_chunks if chunk["doc_idx"] == doc_idx]

            if not doc_chunks:
                print(f"Warning: No valid chunks found for document {doc_idx}")
                continue

            print(f"Reconciling document {doc_idx} with {len(doc_chunks)} chunks")

            # Create fresh reconciler for this document
            doc_reconciler = _create_reconciler(matching_approach, **reconciler_kwargs)

            # Reconcile all chunks for this document
            try:
                doc_reconciled_gsw = _reconcile_document_chunks(
                    doc_reconciler, doc_chunks
                )
                reconciled_documents.append(doc_reconciled_gsw)

                # Print statistics for this document
                stats = doc_reconciler.get_statistics()
                print(
                    f"Document {doc_idx} reconciled: {stats['entities']} entities, "
                    f"{stats['entities_with_evolution']} with evolution"
                )

            except Exception as e:
                print(f"Error reconciling document {doc_idx}: {e}")
                continue

    elif strategy == "global":
        # Global reconciliation: one reconciler for all documents
        print(f"Reconciling all {len(all_chunks)} chunks globally")

        # Create single reconciler for all documents
        global_reconciler = _create_reconciler(matching_approach, **reconciler_kwargs)

        # Sort chunks to ensure consistent processing order
        all_chunks.sort(key=lambda x: (x["doc_idx"], x["chunk_idx"]))

        try:
            # Reconcile all chunks sequentially
            reconciled_gsw = _reconcile_all_chunks(global_reconciler, all_chunks)

            # Print final statistics
            stats = global_reconciler.get_statistics()
            print(
                f"Global reconciliation complete: {stats['entities']} entities, "
                f"{stats['entities_with_evolution']} with evolution"
            )

        except Exception as e:
            print(f"Error in global reconciliation: {e}")
            return None

    # Determine final results
    final_results = reconciled_documents if strategy == "local" else reconciled_gsw

    # Save outputs if output_dir is provided
    if output_dir is not None:
        _save_reconciled_outputs(
            reconciled_results=final_results,
            strategy=strategy,
            matching_approach=matching_approach,
            output_dir=output_dir,
            processor_outputs=processor_outputs,
            all_chunks=all_chunks,
            save_statistics=save_statistics,
            enable_visualization=enable_visualization,
        )
    else:
        print("No output directory specified - results not saved")

    return final_results
