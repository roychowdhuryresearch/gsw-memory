"""
GSW Operator - Core functionality for generating Generative Semantic Workspaces.

This module contains the main operator classes for processing text and generating
semantic workspaces, including coreference resolution, context generation, 
and GSW structure creation.
"""

import json
from typing import Dict, List, Optional

from bespokelabs import curator

from .models import EntityNode, GSWStructure, Question, Role, VerbPhraseNode
from ..prompts.operator_prompts import CorefPrompts, ContextPrompts, OperatorPrompts


class CorefOperator(curator.LLM):
    """Curator class for performing coreference resolution."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to perform coreference resolution."""
        return [
            {"role": "system", "content": CorefPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CorefPrompts.USER_PROMPT_TEMPLATE.format(text=input['text']),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract resolved text."""
        return [
            {
                "text": response["choices"][0]["message"]["content"],
                "idx": input["idx"],
            }
        ]


class ContextGenerator(curator.LLM):
    """Curator class for generating chunk-specific context."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to generate context."""
        return [
            {"role": "system", "content": ContextPrompts.SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": ContextPrompts.USER_PROMPT_TEMPLATE.format(
                    doc_text=input['doc_text'],
                    chunk_text=input['chunk_text']
                )
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract the generated context."""
        return [
            {
                "context": response["choices"][0]["message"]["content"].strip(),
                "doc_idx": input["doc_idx"],
                "chunk_idx": input["chunk_idx"],
            }
        ]


class GSWOperator(curator.LLM):
    """Curator class for generating GSWs using sophisticated semantic role extraction."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to generate a GSW."""
        return [
            {"role": "system", "content": OperatorPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": OperatorPrompts.USER_PROMPT_TEMPLATE.replace(
                    "{{input_text}}", input["text"]
                ).replace(
                    "{{background_context}}", input.get("context", "")
                ),
            },
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract text and graph."""
        parsed_response = {
            "text": input["text"],
            "idx": input["idx"],
            "graph": response["choices"][0]["message"]["content"],
            "context": input.get("context", ""),
            "doc_idx": input.get("doc_idx", input["idx"]),
        }
        
        # Include sentence indices if available
        if "start_sentence" in input:
            parsed_response["start_sentence"] = input["start_sentence"]
        if "end_sentence" in input:
            parsed_response["end_sentence"] = input["end_sentence"]

        return [parsed_response]


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


def process_long_text(
    text: str, 
    coref_chunk_size: int = 20,
    model_name: str = "gpt-4o",
    generation_params: Optional[Dict] = None
) -> str:
    """Process long text by first performing coreference resolution on larger chunks."""
    # If coref_chunk_size is -1, process the entire text at once
    if coref_chunk_size == -1:
        coref_chunks = [{"text": text, "idx": 0}]
    else:
        # First chunk the text into larger pieces for coref
        coref_chunks = chunk_text(text, chunk_size=coref_chunk_size, overlap=2)

    # Initialize coref model
    coref_model = CorefOperator(
        model_name=model_name,
        generation_params=generation_params or {"temperature": 0.0, "max_tokens": 4000},
    )

    # Process each chunk with coref
    coref_responses = coref_model(coref_chunks)

    # Combine the coref resolved chunks
    resolved_text = " ".join([resp["text"] for resp in coref_responses])

    return resolved_text


class GSWProcessor:
    """Main processor class that orchestrates the complete GSW generation pipeline."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict] = None,
        enable_coref: bool = True,
        enable_context: bool = True,
        chunking_enabled: bool = True,
        chunk_size: int = 3,
        overlap: int = 1,
        coref_chunk_size: int = 20,
        enable_visualization: bool = False
    ):
        """Initialize the GSW processor with configuration options."""
        self.model_name = model_name
        self.generation_params = generation_params or {"temperature": 0.0}
        self.enable_coref = enable_coref
        self.enable_context = enable_context
        self.chunking_enabled = chunking_enabled
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.coref_chunk_size = coref_chunk_size
        self.enable_visualization = enable_visualization
        
        # Check if visualization is requested but NetworkX is not available
        if self.enable_visualization:
            try:
                from ..utils.visualization import NETWORKX_AVAILABLE
                if not NETWORKX_AVAILABLE:
                    print("Warning: NetworkX not available. Visualization disabled.")
                    self.enable_visualization = False
            except ImportError:
                print("Warning: Visualization module not available. Visualization disabled.")
                self.enable_visualization = False
        
    def process_text(
        self, 
        text: str, 
        doc_idx: int = 0, 
        save_visualizations: bool = None,
        visualization_dir: str = None
    ) -> List[GSWStructure]:
        """Process a text through the complete GSW pipeline.
        
        Args:
            text: Input text to process
            doc_idx: Document index for this text
            save_visualizations: Whether to save visualizations (overrides class setting)
            visualization_dir: Directory to save visualizations
            
        Returns:
            List of GSWStructure objects
        """
        # Override class setting if parameter provided
        save_viz = save_visualizations if save_visualizations is not None else self.enable_visualization
        
        # Step 1: Coreference resolution
        if self.enable_coref:
            resolved_text = process_long_text(
                text, 
                coref_chunk_size=self.coref_chunk_size,
                model_name=self.model_name,
                generation_params=self.generation_params
            )
        else:
            resolved_text = text
            
        # Step 2: Chunking
        if self.chunking_enabled:
            chunks = chunk_text(resolved_text, chunk_size=self.chunk_size, overlap=self.overlap)
            for chunk in chunks:
                chunk["doc_idx"] = doc_idx
        else:
            chunks = [{"text": resolved_text, "idx": 0, "doc_idx": doc_idx}]
            
        # Step 3: Context generation
        contexts = {}
        if self.enable_context:
            context_model = ContextGenerator(
                model_name=self.model_name,
                generation_params={"temperature": 0.1, "max_tokens": 150}
            )
            context_inputs = [
                {
                    "doc_text": resolved_text,
                    "chunk_text": chunk["text"],
                    "doc_idx": chunk["doc_idx"],
                    "chunk_idx": chunk["idx"],
                }
                for chunk in chunks
            ]
            context_responses = context_model(context_inputs)
            for resp in context_responses:
                key = (resp["doc_idx"], resp["chunk_idx"])
                contexts[key] = resp["context"]
                
        # Step 4: GSW generation
        gsw_model = GSWOperator(
            model_name=self.model_name,
            generation_params=self.generation_params
        )
        
        # Prepare data for GSW generation
        gsw_inputs = []
        for chunk in chunks:
            chunk_data = {
                "text": chunk["text"],
                "idx": chunk["idx"],
                "doc_idx": chunk["doc_idx"],
            }
            if self.enable_context:
                key = (chunk["doc_idx"], chunk["idx"])
                chunk_data["context"] = contexts.get(key, "")
            gsw_inputs.append(chunk_data)
            
        # Generate GSWs
        gsw_responses = gsw_model(gsw_inputs)
        
        # Parse responses into GSWStructure objects
        gsw_structures = []
        for response in gsw_responses:
            try:
                gsw = parse_gsw(response["graph"])
                gsw_structures.append(gsw)
                
                # Create visualization if enabled
                if save_viz and visualization_dir:
                    try:
                        from ..utils.visualization import create_and_save_gsw_visualization
                        import os
                        
                        os.makedirs(visualization_dir, exist_ok=True)
                        chunk_idx = response.get('idx', 'unknown')
                        viz_path = os.path.join(
                            visualization_dir, 
                            f"gsw_doc_{doc_idx}_chunk_{chunk_idx}.cyjs"
                        )
                        create_and_save_gsw_visualization(gsw, viz_path)
                        print(f"Visualization saved: {viz_path}")
                        
                    except Exception as viz_error:
                        print(f"Warning: Failed to create visualization: {viz_error}")
                        
            except Exception as e:
                print(f"Error parsing GSW for chunk {response.get('idx', 'unknown')}: {e}")
                continue
                
        return gsw_structures
    
    def create_visualization(self, gsw: GSWStructure, output_path: str) -> None:
        """Create and save a visualization for a GSW structure.
        
        Args:
            gsw: GSWStructure to visualize
            output_path: Path to save the visualization file
        """
        if not self.enable_visualization:
            print("Visualization not enabled. Set enable_visualization=True or install NetworkX.")
            return
            
        try:
            from ..utils.visualization import create_and_save_gsw_visualization
            create_and_save_gsw_visualization(gsw, output_path)
            print(f"Visualization saved: {output_path}")
        except ImportError as e:
            print(f"Visualization not available: {e}")
        except Exception as e:
            print(f"Error creating visualization: {e}")