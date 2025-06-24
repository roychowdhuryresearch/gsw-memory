"""
Entity matching for finding GSW entities that correspond to extracted question entities.

This module implements the second step of the Q&A pipeline:
matching extracted entities from questions to actual entity nodes in the GSW.
"""

from typing import List
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
        gsw: GSWStructure
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
            
        Returns:
            List of matched EntityNode objects
        """
        matched_entities = []
        
        # First pass: try the current substring matching approach
        for entity in gsw.entity_nodes:
            if any(entity.name.lower() in name.lower() for name in entity_names):
                matched_entities.append(entity)
                continue

            # Second pass: word-level matching
            entity_words = set(entity.name.lower().split())
            for name in entity_names:
                name_words = set(name.lower().split())
                # Check if there's significant word overlap
                common_words = entity_words.intersection(name_words)

                # If there's at least one meaningful word in common (not stop words)
                if common_words:
                    matched_entities.append(entity)
                    break

        return matched_entities