"""
Utility functions for GSW operators.

This module contains utility functions used across different operators,
including text chunking, JSON extraction, and GSW parsing.
"""

import json
from typing import Dict, List

from ..models import EntityNode, GSWStructure, Question, Role, VerbPhraseNode


def extract_json_from_output(text: str) -> dict:
    """Extract JSON part from LLM output."""
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Error extracting JSON: {e}")


def parse_gsw(text: str) -> GSWStructure:
    """Parse LLM output text into a GSWStructure object."""
    # Clean up the text to extract JSON
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    elif "</semantic_construction>" in text:
        text = text.split("</semantic_construction>")[0]
    else:
        # Try to find JSON structure
        text = text.rsplit("}", 1)[0] + "}"
    
    try:
        # Extract JSON part
        data = extract_json_from_output(text.strip())

        # Parse entity nodes
        entities = []
        if "entity_nodes" in data:
            for e in data["entity_nodes"]:
                roles = [Role(**r) for r in e.get("roles", [])]
                entities.append(
                    EntityNode(
                        id=e["id"], 
                        name=e["name"], 
                        roles=roles,
                        chunk_id=e.get("chunk_id"),
                        summary=e.get("summary")
                    )
                )

        # Parse verb phrase nodes
        verb_phrases = []
        if "verb_phrase_nodes" in data:
            for v in data["verb_phrase_nodes"]:
                questions = [Question(**q) for q in v.get("questions", [])]
                verb_phrases.append(
                    VerbPhraseNode(
                        id=v["id"],
                        phrase=v["phrase"],
                        questions=questions,
                        chunk_id=v.get("chunk_id")
                    )
                )

        return GSWStructure(entity_nodes=entities, verb_phrase_nodes=verb_phrases)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required field: {e}")


def chunk_text(text: str, chunk_size: int = 3, overlap: int = 1) -> List[Dict]:
    """Split text into overlapping chunks.

    Args:
        text: The input text to chunk
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks

    Returns:
        List of dictionaries containing chunked text and indices
    """
    # Split into sentences - basic split on ., ! and ?
    sentences = [
        s.strip()
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]

    chunks = []
    i = 0
    chunk_id = 0

    while i < len(sentences):
        # Get chunk_size sentences starting from i
        chunk_sentences = sentences[i : i + chunk_size]
        if chunk_sentences:  # Only add if we have sentences
            chunks.append(
                {
                    "text": ". ".join(chunk_sentences) + ".",
                    "idx": chunk_id,
                    "start_sentence": i,
                    "end_sentence": i + len(chunk_sentences),
                }
            )
            chunk_id += 1

        # Move forward by chunk_size - overlap sentences
        i += chunk_size - overlap

    return chunks