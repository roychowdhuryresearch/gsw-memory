"""
Conversation linking utilities for GSW operators.

This module contains conversation detection and analysis functionality
for identifying and linking conversations in GSW structures.
"""

import json
from typing import Dict, List, Optional

from bespokelabs import curator
from pydantic import BaseModel

from ..models import GSWStructure, EntityNode, Role
from ...prompts import ConversationAnalysisPrompts, ConversationDetectionPrompts


class ConversationDetectionOutput(BaseModel):
    has_conversation: bool
    confidence: float
    reasoning: str


class NewEntityRole(BaseModel):
    role: str
    states: List[str]


class NewEntity(BaseModel):
    name: str
    roles: List[NewEntityRole]


class ConversationNode(BaseModel):
    participants: List[str]
    topics_entity: List[str]
    topics_general: List[str]
    location_id: Optional[str]
    time_id: Optional[str]
    motivation: str
    summary: str
    participant_summaries: Dict[str, str]


class ConversationAnalysisOutput(BaseModel):
    conversation_node: ConversationNode
    new_entities: List[NewEntity]


# Note: GSWStructure now natively supports conversation nodes and edges
# No need for a separate ConversationGSWStructure class


class ConversationDetector(curator.LLM):
    """
    Curator Class for detecting if a chunk contains dialogue/conversation
    """

    response_format = ConversationDetectionOutput

    def prompt(self, input_data):
        """Create a prompt to detect if a chunk contains conversation/dialogue."""

        system_prompt = ConversationDetectionPrompts.SYSTEM_PROMPT

        user_prompt = ConversationDetectionPrompts.USER_PROMPT_TEMPLATE.replace(
            "{input_data['text_chunk_content']}", input_data["text_chunk_content"]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data, response):
        """Parse the LLM response to extract conversation detection result."""
        return {
            "has_conversation": response.has_conversation,
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "chunk_id": input_data.get("chunk_id", "unknown_chunk"),
            "idx": input_data.get("idx", 0),
        }


class ConversationAnalyzer(curator.LLM):
    """
    Curator Class for analyzing conversation details when dialogue is detected
    """

    response_format = ConversationAnalysisOutput

    def prompt(self, input_data):
        """Create a prompt to analyze conversation details."""

        system_prompt = ConversationAnalysisPrompts.SYSTEM_PROMPT

        user_prompt = ConversationAnalysisPrompts.USER_PROMPT_TEMPLATE.replace(
            "{input_data['text_chunk_content']}", input_data["text_chunk_content"]
        ).replace(
            "{json.dumps(input_data['gsw_structure'], indent=2)}", json.dumps(input_data["gsw_structure"], indent=2)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data, response):
        """Parse the LLM response to extract conversation analysis."""
        return {
            "conversation_node": response.conversation_node.model_dump(),
            "new_entities": [entity.model_dump() for entity in response.new_entities],
            "chunk_id": input_data.get("chunk_id", "unknown_chunk"),
            "idx": input_data.get("idx", 0),
        }


def process_conversation_batch(
    all_documents_data: List[Dict[str, Dict]],
    model_name: str = "gpt-4o",
    generation_params: Optional[Dict] = None
) -> List[Dict[str, Dict]]:
    """
    Process conversation detection and analysis for all chunks across all documents.
    
    Args:
        all_documents_data: List of document chunk data from the main pipeline
        model_name: LLM model to use
        generation_params: Generation parameters for the LLM
        
    Returns:
        Updated all_documents_data with conversation information
    """
    if generation_params is None:
        generation_params = {"temperature": 0.0, "max_tokens": 1500}
    
    print("--- Processing Conversation Detection and Analysis ---")
    
    # Stage 1: Prepare detection prompts for all chunks
    detection_prompts = []
    chunk_mapping = {}  # Map (doc_idx, chunk_id) to chunk data
    
    for doc_idx, doc_chunks in enumerate(all_documents_data):
        for chunk_id, chunk_data in doc_chunks.items():
            if chunk_data.get("gsw") is not None:  # Only process chunks with valid GSWs
                detection_prompts.append({
                    "text_chunk_content": chunk_data["text"],
                    "chunk_id": chunk_id,
                    "idx": f"{doc_idx}_{chunk_id}",
                })
                chunk_mapping[f"{doc_idx}_{chunk_id}"] = (doc_idx, chunk_id)
    
    if not detection_prompts:
        print("No chunks with valid GSWs found for conversation processing")
        return all_documents_data
    
    print(f"Processing conversation detection for {len(detection_prompts)} chunks...")
    
    # Stage 2: Batch conversation detection
    conversation_detector = ConversationDetector(
        model_name=model_name,
        generation_params=generation_params,
    )
    
    detection_results = conversation_detector(detection_prompts)
    
    # Stage 3: Filter chunks with conversations and prepare analysis
    conversation_chunks = []
    analysis_prompts = []
    
    for result in detection_results.dataset:
        idx = result.get("idx", "")
        has_conversation = result.get("has_conversation", False)
        confidence = result.get("confidence", 0.0)
        
        if has_conversation and idx in chunk_mapping:
            doc_idx, chunk_id = chunk_mapping[idx]
            chunk_data = all_documents_data[doc_idx][chunk_id]
            
            print(f"Chunk {chunk_id}: Conversation detected (confidence: {confidence:.2f})")
            conversation_chunks.append((doc_idx, chunk_id))
            
            # Use the native GSWStructure for analysis
            gsw = chunk_data["gsw"]
            
            # Prepare analysis prompt
            analysis_prompts.append({
                "text_chunk_content": chunk_data["text"],
                "gsw_structure": gsw.model_dump(),
                "chunk_id": chunk_id,
                "idx": idx,
            })
    
    print(f"Found {len(conversation_chunks)} chunks with conversations")
    
    # Stage 4: Batch conversation analysis
    if analysis_prompts:
        print(f"Running conversation analysis for {len(analysis_prompts)} chunks...")
        conversation_analyzer = ConversationAnalyzer(
            model_name=model_name,
            generation_params=generation_params,
        )
        
        analysis_results = conversation_analyzer(analysis_prompts)
        
        # Stage 5: Apply results to chunks
        analysis_by_idx = {result.get("idx", ""): result for result in analysis_results.dataset}
        
        for doc_idx, chunk_id in conversation_chunks:
            idx = f"{doc_idx}_{chunk_id}"
            if idx in analysis_by_idx:
                print(f"Applying conversation analysis to chunk {chunk_id}...")
                
                # Get the chunk data
                chunk_data = all_documents_data[doc_idx][chunk_id]
                gsw = chunk_data["gsw"]
                
                # Apply conversation analysis directly to the GSW structure
                updated_gsw = _apply_conversation_analysis(
                    gsw, 
                    analysis_by_idx[idx], 
                    chunk_id
                )
                
                # Update the chunk data with conversation information
                chunk_data["gsw"] = updated_gsw
                chunk_data["has_conversation"] = True
                chunk_data["conversation_count"] = len(updated_gsw.conversation_nodes)
    
    return all_documents_data


def _apply_conversation_analysis(
    gsw: GSWStructure, 
    analysis_result: Dict, 
    chunk_id: str
) -> GSWStructure:
    """
    Apply conversation analysis results to a ConversationGSWStructure.
    
    Args:
        conv_gsw: The ConversationGSWStructure to update
        analysis_result: The analysis result from the conversation analyzer
        chunk_id: The chunk ID
        
    Returns:
        Updated ConversationGSWStructure
    """
    conversation_node_data = analysis_result.get("conversation_node", {})
    new_entities_data = analysis_result.get("new_entities", [])

    if not conversation_node_data:
        print("No conversation node data returned, skipping")
        return gsw

    # Create new entities for speakers not in existing GSW
    entity_name_to_id = {}
    for new_entity_data in new_entities_data:
        entity_name = new_entity_data.get("name", "")
        if not entity_name:
            continue
            
        # Generate new entity ID
        existing_ids = [entity.id for entity in gsw.entity_nodes]
        entity_counter = len(existing_ids)
        while f"e{entity_counter}" in existing_ids:
            entity_counter += 1
        new_entity_id = f"e{entity_counter}"
        
        # Create entity with inferred roles
        roles = []
        for role_data in new_entity_data.get("roles", []):
            role = Role(
                role=role_data.get("role", "speaker"),
                states=role_data.get("states", []),
                chunk_id=chunk_id
            )
            roles.append(role)
        
        new_entity = EntityNode(
            id=new_entity_id,
            name=entity_name,
            roles=roles,
            chunk_id=chunk_id,
        )
        
        gsw.entity_nodes.append(new_entity)
        entity_name_to_id[entity_name] = new_entity_id
        print(f"Created new entity: {new_entity_id} ({entity_name})")

    # Process conversation node
    conv_id = conversation_node_data.get("id", "cv_0")
    participants = conversation_node_data.get("participants", [])
    topics_entity = conversation_node_data.get("topics_entity", [])
    topics_general = conversation_node_data.get("topics_general", [])
    location_id = conversation_node_data.get("location_id")
    time_id = conversation_node_data.get("time_id")
    
    # Convert participant names to entity IDs
    participant_ids = []
    for participant in participants:
        if participant in entity_name_to_id:
            participant_ids.append(entity_name_to_id[participant])
        else:
            participant_ids.append(participant)  # Assume it's already an entity ID

    # Create conversation node
    conversation_node = {
        "id": conv_id,
        "chunk_id": chunk_id,
        "participants": participant_ids,
        "topics_entity": topics_entity,
        "topics_general": topics_general,
        "location_id": location_id,
        "time_id": time_id,
        "motivation": conversation_node_data.get("motivation", ""),
        "summary": conversation_node_data.get("summary", ""),
        "participant_summaries": conversation_node_data.get("participant_summaries", {}),
        "type": "conversation"
    }

    # Add conversation node to GSW structure
    gsw.add_conversation_node(conversation_node)

    # Add participant edges
    for participant_id in participant_ids:
        gsw.add_conversation_participant_edge(participant_id, conv_id)

    # Add topic edges
    for topic_entity_id in topics_entity:
        gsw.add_conversation_topic_edge(topic_entity_id, conv_id)

    # Add space/time edges if applicable
    if location_id:
        gsw.add_conversation_space_edge(conv_id, location_id)
    
    if time_id:
        gsw.add_conversation_time_edge(conv_id, time_id)

    print(f"Created conversation node: {conv_id}")
    print(f"Participants: {participant_ids}")
    print(f"Topic entities: {topics_entity}")
    print(f"General topics: {topics_general}")
    print(f"Location: {location_id}, Time: {time_id}")

    return gsw
