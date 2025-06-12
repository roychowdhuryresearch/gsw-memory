"""
Space-Time Linking Operator.

This module contains the SpaceTimeLinker class for identifying temporal and spatial
relationships between entities in GSW structures using LLM-based processing.
"""

import json
from typing import Dict

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
                    operator_output_json=input_data['operator_output_json']
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