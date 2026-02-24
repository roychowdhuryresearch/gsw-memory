"""
Space-Time Linking Operator.

This module contains the SpaceTimeLinker class for identifying temporal and spatial
relationships between entities in GSW structures using LLM-based processing.
"""

import json
from typing import Dict, List

from bespokelabs import curator

from ...prompts.operator_prompts import SpaceTimePrompts


class SpaceTimeLinker(curator.LLM):
    """
    Curator Class for Space-Time linking of GSW instances.
    
    Identifies groups of entity IDs that share the same location (spatial context) 
    or the same time/date (temporal context) based on the events described.
    """

    return_completions_object = True

    def prompt(self, input_data):
        """Create a prompt for the LLM to identify spatio-temporal relationships."""
        return [
            {"role": "system", "content": SpaceTimePrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SpaceTimePrompts.USER_PROMPT_TEMPLATE.format(
                    text_chunk_content=input_data['text_chunk_content'],
                    operator_output_json=input_data['operator_output_json'],
                    session_context=input_data.get('session_context', ''),
                )
            },
        ]

    def parse(self, input_data, response):
        """Parse the LLM response to extract the JSON with spatio-temporal links."""
        answer_text = response["choices"][0]["message"]["content"].strip()

        # Extract the JSON content from the response
        try:
            if "```json" in answer_text:
                json_content = answer_text.split("```json")[1].split("```")[0].strip()
            elif "```" in answer_text:
                json_content = answer_text.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON-like content directly
                json_content = answer_text

            # Parse the JSON content
            parsed_links = json.loads(json_content)

            # Return the parsed content and original answer
            return [
                {
                    "spatio_temporal_links": parsed_links.get("spatio_temporal_links", []),
                    "full_response": answer_text,
                    "chunk_id": input_data.get("chunk_id", "unknown_chunk"),
                    "idx": input_data.get("idx", 0),
                    "doc_idx": input_data.get("doc_idx", 0),
                    "global_id": input_data.get("global_id", "unknown"),
                }
            ]
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing LLM response: {e}")
            return [
                {
                    "spatio_temporal_links": [],
                    "error": str(e),
                    "full_response": answer_text,
                    "chunk_id": input_data.get("chunk_id", "unknown_chunk"),
                    "idx": input_data.get("idx", 0),
                    "doc_idx": input_data.get("doc_idx", 0),
                    "global_id": input_data.get("global_id", "unknown"),
                }
            ]


def apply_spacetime_to_gsw(
    gsw,  # GSWStructure — avoid circular import by using duck typing
    spatio_temporal_links: List[Dict],
    chunk_id: str,
) -> None:
    """Convert SpaceTimeLinker output into SpaceNode/TimeNode objects and attach to gsw in-place.

    Entity IDs in spatio_temporal_links are chunk-local (e.g. "e1") matching the
    entity_nodes in the gsw before reconciliation globalises them.  The reconciler's
    _prepare_new_gsw() will later convert "s0" → "chunk_id::s0", same as entities.

    Args:
        gsw: GSWStructure to modify in place.
        spatio_temporal_links: List of link dicts from SpaceTimeLinker output, each with
            keys: linked_entities (list of entity id strings), tag_type ("spatial" or
            "temporal"), tag_value (str or None).
        chunk_id: Identifier for this chunk, used as the key in name_history.
    """
    from ..models import SpaceNode, TimeNode  # local import to avoid circular dependency

    # Build a set of entity IDs that exist in the GSW for fast membership checks
    existing_entity_ids = {e.id for e in gsw.entity_nodes}

    space_counter = 0
    time_counter = 0

    for link in spatio_temporal_links:
        tag_type = link.get("tag_type", "")
        tag_value = link.get("tag_value") or "unknown"
        linked_entities: List[str] = link.get("linked_entities", [])

        # Filter to only entity IDs that actually exist in this GSW chunk
        valid_entity_ids = [eid for eid in linked_entities if eid in existing_entity_ids]
        if not valid_entity_ids:
            continue

        if tag_type == "spatial":
            node_id = f"s{space_counter}"
            space_counter += 1
            node = SpaceNode(
                id=node_id,
                name_history={chunk_id: tag_value},
                current_name=tag_value,
                chunk_id=chunk_id,
            )
            gsw.add_space_node(node)
            for entity_id in valid_entity_ids:
                gsw.add_space_edge(entity_id, node_id)

        elif tag_type == "temporal":
            node_id = f"t{time_counter}"
            time_counter += 1
            node = TimeNode(
                id=node_id,
                name_history={chunk_id: tag_value},
                current_name=tag_value,
                chunk_id=chunk_id,
            )
            gsw.add_time_node(node)
            for entity_id in valid_entity_ids:
                gsw.add_time_edge(entity_id, node_id)