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


class GSWProcessor:
    """Main processor class that orchestrates the complete GSW generation pipeline."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict] = None,
        enable_coref: bool = True,
        enable_chunking: bool = True,
        enable_context: bool = True,
        chunk_size: int = 3,
        overlap: int = 1,
        coref_chunk_size: int = 20,
        enable_visualization: bool = False
    ):
        """Initialize the GSW processor with configuration options."""
        self.model_name = model_name
        self.generation_params = generation_params or {"temperature": 0.0}
        self.enable_coref = enable_coref
        self.enable_chunking = enable_chunking
        self.enable_context = enable_context
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
        
    def process_documents(
        self, 
        documents: List[str],
        output_dir: Optional[str] = None,
        save_intermediates: bool = False,
        enable_coref: Optional[bool] = None,
        enable_chunking: Optional[bool] = None,
        enable_context: Optional[bool] = None,
        enable_visualization: Optional[bool] = None
    ) -> List[GSWStructure]:
        """Process multiple documents through the complete GSW pipeline with full parallelization.
        
        Args:
            documents: List of document texts to process
            output_dir: Directory to save outputs (if None, no saving)
            save_intermediates: Whether to save intermediate steps (coref, chunks, context)
            enable_coref: Override class setting for coreference resolution
            enable_chunking: Override class setting for chunking
            enable_context: Override class setting for context generation
            enable_visualization: Override class setting for visualization
            
        Returns:
            List of GSWStructure objects (one per chunk across all documents)
        """
        import os
        import json
        
        # Override class settings if provided
        do_coref = enable_coref if enable_coref is not None else self.enable_coref
        do_chunking = enable_chunking if enable_chunking is not None else self.enable_chunking
        do_context = enable_context if enable_context is not None else self.enable_context
        do_visualization = enable_visualization if enable_visualization is not None else self.enable_visualization
        
        # Step 1: Coreference Resolution (parallel across all documents)
        if do_coref:
            print(f"--- Processing Coreference for {len(documents)} documents ---")
            coref_model = CorefOperator(
                model_name=self.model_name,
                generation_params={"temperature": 0.0, "max_tokens": 4000}
            )
            
            # Prepare coref inputs
            coref_inputs = []
            for doc_idx, document in enumerate(documents):
                if self.coref_chunk_size == -1:
                    coref_inputs.append({"text": document, "idx": doc_idx})
                else:
                    # Chunk document for coref
                    coref_chunks = chunk_text(document, chunk_size=self.coref_chunk_size, overlap=2)
                    for chunk in coref_chunks:
                        chunk["doc_idx"] = doc_idx
                        coref_inputs.append(chunk)
            
            # Process all coref chunks in parallel
            coref_responses = coref_model(coref_inputs)
            
            # Reconstruct resolved documents
            if self.coref_chunk_size == -1:
                resolved_documents = {resp["idx"]: resp["text"] for resp in coref_responses.dataset}
            else:
                # Group responses by document and reconstruct
                doc_coref_texts = {}
                for resp in coref_responses.dataset:
                    doc_idx = resp.get("doc_idx", resp["idx"])
                    if doc_idx not in doc_coref_texts:
                        doc_coref_texts[doc_idx] = []
                    doc_coref_texts[doc_idx].append(resp["text"])
                
                resolved_documents = {
                    doc_idx: " ".join(texts) for doc_idx, texts in doc_coref_texts.items()
                }
        else:
            resolved_documents = {idx: doc for idx, doc in enumerate(documents)}
        
        # Step 2: Chunking (create all chunks from all documents)
        all_chunks = []
        if do_chunking:
            print("--- Chunking Documents ---")
            for doc_idx, resolved_text in resolved_documents.items():
                chunks = chunk_text(resolved_text, chunk_size=self.chunk_size, overlap=self.overlap)
                for chunk in chunks:
                    chunk["doc_idx"] = doc_idx
                    # Create global ID as doc_idx_chunk_idx
                    chunk["global_id"] = f"{doc_idx}_{chunk['idx']}"
                all_chunks.extend(chunks)
        else:
            # No chunking: each document becomes one chunk
            for doc_idx, resolved_text in resolved_documents.items():
                chunk = {
                    "text": resolved_text,
                    "idx": 0,
                    "doc_idx": doc_idx,
                    "global_id": f"{doc_idx}_0"
                }
                all_chunks.append(chunk)
        
        # Step 3: Context Generation (parallel across all chunks, but only if multiple chunks exist)
        contexts = {}
        should_generate_context = do_context and len(all_chunks) > len(documents)  # More chunks than docs
        
        if should_generate_context:
            print(f"--- Generating Context for {len(all_chunks)} chunks ---")
            context_model = ContextGenerator(
                model_name=self.model_name,
                generation_params={"temperature": 0.1, "max_tokens": 150}
            )
            
            context_inputs = [
                {
                    "doc_text": resolved_documents[chunk["doc_idx"]],
                    "chunk_text": chunk["text"],
                    "doc_idx": chunk["doc_idx"],
                    "chunk_idx": chunk["idx"],
                }
                for chunk in all_chunks
            ]
            
            context_responses = context_model(context_inputs)
            for resp in context_responses.dataset:
                key = (resp["doc_idx"], resp["chunk_idx"])
                contexts[key] = resp["context"]
        
        # Step 4: GSW Generation (parallel across all chunks)
        print(f"--- Generating GSWs for {len(all_chunks)} chunks ---")
        gsw_model = GSWOperator(
            model_name=self.model_name,
            generation_params=self.generation_params
        )
        
        # Prepare GSW inputs
        gsw_inputs = []
        for chunk in all_chunks:
            chunk_data = {
                "text": chunk["text"],
                "idx": chunk["idx"],
                "doc_idx": chunk["doc_idx"],
                "global_id": chunk["global_id"]
            }
            
            # Add sentence indices if available
            if "start_sentence" in chunk:
                chunk_data["start_sentence"] = chunk["start_sentence"]
            if "end_sentence" in chunk:
                chunk_data["end_sentence"] = chunk["end_sentence"]
            
            # Add context if available
            if should_generate_context:
                key = (chunk["doc_idx"], chunk["idx"])
                chunk_data["context"] = contexts.get(key, "")
            else:
                chunk_data["context"] = ""
                
            gsw_inputs.append(chunk_data)
        
        # Generate all GSWs in parallel
        gsw_responses = gsw_model(gsw_inputs)
        
        # Step 5: Parse responses into GSWStructure objects
        gsw_structures = []
        for response in gsw_responses.dataset:
            try:
                gsw = parse_gsw(response["graph"])
                gsw_structures.append(gsw)
            except Exception as e:
                print(f"Error parsing GSW for chunk {response.get('global_id', 'unknown')}: {e}")
                continue
        
        # Step 6: Save outputs if requested
        if output_dir:
            self._save_outputs(
                output_dir=output_dir,
                save_intermediates=save_intermediates,
                resolved_documents=resolved_documents,
                all_chunks=all_chunks,
                contexts=contexts,
                gsw_responses=gsw_responses,
                gsw_structures=gsw_structures,
                do_visualization=do_visualization
            )
        
        return gsw_structures
    
    def _save_outputs(
        self,
        output_dir: str,
        save_intermediates: bool,
        resolved_documents: Dict[int, str],
        all_chunks: List[Dict],
        contexts: Dict,
        gsw_responses: List[Dict],
        gsw_structures: List[GSWStructure],
        do_visualization: bool
    ):
        """Save all outputs according to the saving strategy."""
        import os
        import json
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Always save final results
        networks_dir = os.path.join(output_dir, "networks")
        networks_raw_dir = os.path.join(output_dir, "networks_raw")
        os.makedirs(networks_dir, exist_ok=True)
        os.makedirs(networks_raw_dir, exist_ok=True)
        
        if do_visualization:
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
        
        # Save intermediates if requested
        if save_intermediates:
            coref_dir = os.path.join(output_dir, "coref")
            chunks_dir = os.path.join(output_dir, "chunks")
            context_dir = os.path.join(output_dir, "context")
            os.makedirs(coref_dir, exist_ok=True)
            os.makedirs(chunks_dir, exist_ok=True)
            os.makedirs(context_dir, exist_ok=True)
            
            # Save coreference results
            for doc_idx, resolved_text in resolved_documents.items():
                with open(os.path.join(coref_dir, f"coref_{doc_idx}.txt"), "w") as f:
                    f.write(resolved_text)
            
            # Save individual chunks
            for chunk in all_chunks:
                doc_chunks_dir = os.path.join(chunks_dir, f"doc_{chunk['doc_idx']}")
                os.makedirs(doc_chunks_dir, exist_ok=True)
                chunk_file = os.path.join(doc_chunks_dir, f"chunk_{chunk['idx']}.txt")
                with open(chunk_file, "w") as f:
                    f.write(chunk["text"])
            
            # Save context files
            for (doc_idx, chunk_idx), context in contexts.items():
                doc_context_dir = os.path.join(context_dir, f"doc_{doc_idx}")
                os.makedirs(doc_context_dir, exist_ok=True)
                context_file = os.path.join(doc_context_dir, f"context_{chunk_idx}.txt")
                with open(context_file, "w") as f:
                    f.write(context)
        
        # Save combined results (all raw responses in one JSONL)
        combined_file = os.path.join(output_dir, "gsw_results_combined.jsonl")
        with open(combined_file, "w") as f:
            for resp in gsw_responses.dataset:
                f.write(json.dumps(resp) + "\n")
        
        # Save individual results and visualizations
        print("--- Saving Networks and Visualizations ---")
        for response, gsw_structure in zip(gsw_responses.dataset, gsw_structures):
            try:
                doc_idx = response["doc_idx"]
                chunk_idx = response["idx"]
                file_stem = f"gsw_{doc_idx}_{chunk_idx}"
                
                # Create document-specific subdirectories
                doc_networks_dir = os.path.join(networks_dir, f"doc_{doc_idx}")
                doc_networks_raw_dir = os.path.join(networks_raw_dir, f"doc_{doc_idx}")
                os.makedirs(doc_networks_dir, exist_ok=True)
                os.makedirs(doc_networks_raw_dir, exist_ok=True)
                
                # Save raw response
                raw_file = os.path.join(doc_networks_raw_dir, f"{file_stem}.json")
                with open(raw_file, "w") as f:
                    json.dump(response, f, indent=4)
                
                # Save parsed network
                network_file = os.path.join(doc_networks_dir, f"{file_stem}.json")
                with open(network_file, "w") as f:
                    json.dump(gsw_structure.model_dump(mode="json"), f, indent=4)
                
                # Save visualization if enabled
                if do_visualization:
                    try:
                        from ..utils.visualization import create_and_save_gsw_visualization
                        
                        doc_viz_dir = os.path.join(viz_dir, f"doc_{doc_idx}")
                        os.makedirs(doc_viz_dir, exist_ok=True)
                        viz_file = os.path.join(doc_viz_dir, f"{file_stem}.cyjs")
                        create_and_save_gsw_visualization(gsw_structure, viz_file)
                        
                    except Exception as viz_error:
                        print(f"Warning: Failed to create visualization for {file_stem}: {viz_error}")
                        
            except Exception as e:
                print(f"Error saving outputs for chunk {response.get('global_id', 'unknown')}: {e}")
                continue
    
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


if __name__ == "__main__":
    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=False,
        enable_chunking=False,
        enable_context=False,
        chunk_size=3,
        overlap=1,
        coref_chunk_size=20
    )

    test_document = """
    John walked into the coffee shop. He ordered a large latte from the barista. 
    The barista, whose name was Sarah, smiled at him. She prepared the drink carefully. 
    John paid for his coffee and sat down at a table near the window. 
    He opened his laptop and began working on his presentation.
    """

    print("Processing single document...")
    gsw_structures = processor.process_documents([test_document])

    print(gsw_structures)