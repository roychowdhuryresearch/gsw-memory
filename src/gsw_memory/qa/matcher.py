"""
Entity matching for finding GSW entities that correspond to extracted question entities.

This module implements the second step of the Q&A pipeline:
matching extracted entities from questions to actual entity nodes in the GSW.
"""

from typing import List, Set
from ..memory.models import GSWStructure, EntityNode

class EntityMatcher:
    """
    Matches extracted entities to GSW entity nodes using approximate matching.
    
    Implements the exact substring + word-level overlap matching approach
    from the original gsw_qa.py find_approximate_matching_entities function.
    """
    
    def __init__(self):
        """Initialize the entity matcher."""
        pass
        
    def find_matching_entities(
        self, 
        entity_names: List[str], 
        gsw: GSWStructure,
        include_connected: bool = False
    ) -> List[EntityNode]:
        """
        Find entities in the GSW that match the given names using approximate matching.
        
        This is an exact replication of find_approximate_matching_entities
        from the original gsw_qa.py (lines 410-435):
        - First pass: substring matching 
        - Second pass: word-level overlap matching
        - Does NOT filter stop words (matches original behavior)
        
        Args:
            entity_names: List of entity names extracted from question
            gsw: GSW structure to search in
            include_connected: If True, also include entities connected via verb phrases
            
        Returns:
            List of matched EntityNode objects
        """
        matched_entities = []
        
        # First pass: try the current substring matching approach
        for entity in gsw.entity_nodes:
            # Clean entity name by removing quotes if present
            clean_entity_name = entity.name.strip('"').strip("'")
            
            # Check if entity name (with or without quotes) matches
            if any(entity.name.lower() in name.lower() or 
                   clean_entity_name.lower() in name.lower() or
                   name.lower() in entity.name.lower() or
                   name.lower() in clean_entity_name.lower() 
                   for name in entity_names):
                matched_entities.append(entity)
                continue

            # Second pass: word-level matching
            entity_words = set(clean_entity_name.lower().split())
            for name in entity_names:
                name_words = set(name.lower().split())
                # Check if there's significant word overlap
                common_words = entity_words.intersection(name_words)

                # If there's at least one meaningful word in common (not stop words)
                if common_words:
                    matched_entities.append(entity)
                    break

        # Optionally include connected entities for multi-hop reasoning
        if include_connected:
            # Use entity IDs to avoid duplicates (EntityNode objects aren't hashable)
            all_entity_ids = {entity.id for entity in matched_entities}
            all_entities_dict = {entity.id: entity for entity in matched_entities}
            
            for entity in matched_entities:
                connected = self._find_connected_entities(entity, gsw)
                for connected_entity in connected:
                    if connected_entity.id not in all_entity_ids:
                        all_entity_ids.add(connected_entity.id)
                        all_entities_dict[connected_entity.id] = connected_entity
            
            return list(all_entities_dict.values())

        return matched_entities

    def _find_connected_entities(self, entity: EntityNode, gsw: GSWStructure) -> List[EntityNode]:
        """
        Find entities connected to the given entity through GSW relationships.
        
        This method traverses verb phrase questions to find other entities that appear
        in the same semantic context, enabling multi-hop reasoning for questions that
        require bridging entities (e.g., "When did X's mother die?" needs both X and mother).
        
        Args:
            entity: The entity to find connections for
            gsw: GSW structure to search in
            
        Returns:
            List of connected EntityNode objects
        """
        connected_entity_ids: Set[str] = set()
        
        # Find connections through verb phrase questions
        # Collect ALL entities from ALL verb phrases where this entity appears
        for vp in gsw.verb_phrase_nodes:
            # Check if this entity appears in ANY question of this verb phrase
            entity_found_in_vp = False
            for question in vp.questions:
                if entity.id in question.answers:
                    entity_found_in_vp = True
                    break
            
            # If entity is found in this verb phrase, collect ALL entities from ALL questions in this VP
            if entity_found_in_vp:
                for question in vp.questions:
                    for answer_id in question.answers:
                        if (answer_id != entity.id and 
                            answer_id != "None" and 
                            not (isinstance(answer_id, str) and answer_id.startswith("TEXT:"))):
                            connected_entity_ids.add(answer_id)
        
        # Convert entity IDs to EntityNode objects
        connected_entities = []
        for entity_id in connected_entity_ids:
            connected_entity = gsw.get_entity_by_id(entity_id)
            if connected_entity:
                connected_entities.append(connected_entity)
        
        return connected_entities