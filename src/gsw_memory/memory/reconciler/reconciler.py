"""
Main GSW Reconciler class for orchestrating entity and question reconciliation.

This module provides the core Reconciler class that coordinates the reconciliation
of new GSW structures with existing global memory.
"""

import uuid
from typing import Optional, Set

from .matching import MatchingStrategy, EntityIndex, create_matching_components
from ..models import GSWStructure, EntityNode


class Reconciler:
    """
    Reconciles a new GSW representation with an existing global memory.
    
    The reconciler handles two main processes:
    1. Entity Reconciliation - Merging new entities with existing ones
    2. Question Reconciliation - Resolving unanswered questions with new information
    """

    def __init__(
        self,
        matching_approach: str = "exact",
        **matching_kwargs
    ):
        """
        Initialize the reconciler with a matching approach.

        Args:
            matching_approach: Type of matching to use ("exact" or "embedding")
            **matching_kwargs: Additional parameters for the matching strategy/index
        """
        # Create strategy and index using factory
        self.matching_strategy, self.entity_index = create_matching_components(
            approach=matching_approach,
            **matching_kwargs
        )
        
        self.global_memory = None

    @classmethod
    def with_strategy(
        cls,
        matching_strategy: MatchingStrategy,
        entity_index: EntityIndex
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
        self._reconcile_spacetime_nodes(new_gsw, space_merge_map, time_merge_map, chunk_id)
        self._update_spacetime_edges(new_gsw, entity_merge_map, space_merge_map, time_merge_map)

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

    def _reconcile_spacetime_nodes(self, new_gsw, space_merge_map, time_merge_map, chunk_id):
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

    def _update_spacetime_edges(self, new_gsw, entity_merge_map, space_merge_map, time_merge_map):
        """Update space and time edges with final IDs."""
        # Update space edges
        existing_global_space_edges = set(self.global_memory.space_edges)
        for original_entity_id, original_space_id in new_gsw.space_edges:
            final_entity_id = entity_merge_map.get(original_entity_id, original_entity_id)
            final_space_id = space_merge_map.get(original_space_id, original_space_id)
            final_space_edge = (final_entity_id, final_space_id)
            
            if final_space_edge not in existing_global_space_edges:
                self.global_memory.add_space_edge(final_entity_id, final_space_id)
                existing_global_space_edges.add(final_space_edge)

        # Update time edges
        existing_global_time_edges = set(self.global_memory.time_edges)
        for original_entity_id, original_time_id in new_gsw.time_edges:
            final_entity_id = entity_merge_map.get(original_entity_id, original_entity_id)
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
            if new_vp.id not in matched_new_verb_ids and new_vp.id not in existing_vp_ids:
                self.global_memory.add_verb_phrase(new_vp)
                existing_vp_ids.add(new_vp.id)

    def _resolve_questions(self, new_chunk_text, matched_old_entity_ids, new_gsw_entities, chunk_id):
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
            print("Warning: resolve_questions not implemented for the current strategy.")
        except Exception as e:
            print(f"Error during question resolution: {e}")

    def get_statistics(self) -> dict:
        """Get statistics about the current global memory."""
        if not self.global_memory:
            return {"entities": 0, "verb_phrases": 0, "questions": 0}
        
        total_questions = sum(len(vp.questions) for vp in self.global_memory.verb_phrase_nodes)
        
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
            "avg_roles_per_entity": total_roles / len(self.global_memory.entity_nodes) if self.global_memory.entity_nodes else 0,
            "space_nodes": len(self.global_memory.space_nodes),
            "time_nodes": len(self.global_memory.time_nodes),
            "space_edges": len(self.global_memory.space_edges),
            "time_edges": len(self.global_memory.time_edges),
        }