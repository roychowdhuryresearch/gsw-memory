"""
GSW Tools for Agentic Question Answering.

This module provides tool functions that give agentic answering agents
direct access to query and explore the GSW structure dynamically.
"""

from typing import Dict, List, Any, Optional
from ..memory.models import GSWStructure


class GSWTools:
    """
    Tool collection for agentic GSW exploration.
    
    Provides direct access to GSW structure for dynamic querying
    and multi-hop reasoning with minimal, focused tools.
    """
    
    def __init__(self, gsw: GSWStructure):
        """Initialize GSW tools with a GSW structure."""
        self.gsw = gsw
    
    def search_gsw(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across GSW questions and entities using multi-token matching.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching items sorted by relevance score
        """
        results = []
        query_tokens = query.lower().split()
        
        # Search in question texts
        # for vp in self.gsw.verb_phrase_nodes:
        #     for question in vp.questions:
        #         # Count matching tokens
        #         question_text_lower = question.text.lower()
        #         matching_tokens = sum(1 for token in query_tokens 
        #                             if token in question_text_lower)
                
        #         # Require at least 2 tokens to match for questions
        #         if matching_tokens >= 2:
        #             # Get entity names for answers
        #             answer_entities = []
        #             for answer_id in question.answers:
        #                 if answer_id != "None":
        #                     entity = self.gsw.get_entity_by_id(answer_id)
        #                     if entity:
        #                         answer_entities.append({
        #                             "entity_id": answer_id,
        #                             "entity_name": entity.name
        #                         })
        #                     else:
        #                         # Keep raw answer if not an entity ID
        #                         answer_entities.append({
        #                             "entity_id": answer_id,
        #                             "entity_name": answer_id
        #                         })
                    
        #             results.append({
        #                 "type": "question",
        #                 "question_id": question.id,
        #                 "question_text": question.text,
        #                 "verb_phrase": vp.phrase,
        #                 "answers": answer_entities,
        #                 "match_score": matching_tokens
        #             })
        
        # Search in entity names
        for entity in self.gsw.entity_nodes:
            entity_name_lower = entity.name.lower()
            matching_tokens = sum(1 for token in query_tokens 
                                if token in entity_name_lower)
            
            # Even 1 token match for entities can be meaningful
            if matching_tokens >= 1:
                results.append({
                    "type": "entity",
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "roles": [{"role": r.role, "states": r.states} 
                             for r in entity.roles],
                    "match_score": matching_tokens
                })
        
        # Sort by relevance score and return top results
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:limit]
    
    def get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """
        Get the context of an entity - all questions it participates in
        and other entities in those questions.
        
        Args:
            entity_id: ID of the entity to get context for
            
        Returns:
            Dict containing entity info and all questions it participates in
        """
        entity = self.gsw.get_entity_by_id(entity_id)
        if not entity:
            return {"error": f"Entity {entity_id} not found"}
        
        context = {
            "entity_id": entity_id,
            "entity_name": entity.name,
            "roles": [{"role": r.role, "states": r.states} 
                     for r in entity.roles],
            "questions": []
        }
        
        # Find all questions this entity answers
        for vp in self.gsw.verb_phrase_nodes:
            for question in vp.questions:
                if entity_id in question.answers:
                    # Get all other entities in this same question
                    other_entities = []
                    for answer_id in question.answers:
                        if answer_id != entity_id and answer_id != "None":
                            other_entity = self.gsw.get_entity_by_id(answer_id)
                            if other_entity:
                                other_entities.append({
                                    "entity_id": answer_id,
                                    "entity_name": other_entity.name
                                })
                            else:
                                # Handle non-entity answers
                                other_entities.append({
                                    "entity_id": answer_id,
                                    "entity_name": answer_id
                                })
                    
                    context["questions"].append({
                        "question_id": question.id,
                        "question_text": question.text,
                        "verb_phrase": vp.phrase,
                        "other_entities": other_entities
                    })
        
        return context