"""
GSW Memory Main Operator.

This module contains the main GSWProcessor class that orchestrates the complete
GSW generation pipeline using modular operator components.
"""

import json
import os
from typing import Dict, List, Optional

from .models import GSWStructure, SpaceNode, TimeNode
from .operator_utils import (
    ContextGenerator,
    CorefOperator,
    GSWOperator,
    SpaceTimeLinker,
    parse_gsw,
)
from .operator_utils.utils import estimate_tokens
from .operator_utils.coref import process_single_coref_batch, process_chunked_coref_batch
from .operator_utils.chunk import chunk_text_unified
from .operator_utils.conversation import process_conversation_batch

class GSWProcessor:
    """Main processor class that orchestrates the complete GSW generation pipeline."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict] = None,
        enable_coref: bool = True,
        enable_chunking: bool = True,
        enable_context: bool = True,
        enable_spacetime: bool = True,
        enable_conversation: bool = True,
        chunk_size: int = 3,
        overlap: int = 1,
        chunking_method: str = "sentence",  # "sentence" or "event_boundary"
        event_chunking_params: Optional[Dict] = None,  # Parameters for event boundary chunking
        enable_visualization: bool = False,
    ):
        """Initialize the GSW processor with configuration options."""
        self.model_name = model_name
        self.generation_params = generation_params or {"temperature": 0.0}
        self.enable_coref = enable_coref
        self.enable_chunking = enable_chunking
        self.enable_context = enable_context
        self.enable_spacetime = enable_spacetime
        self.enable_conversation = enable_conversation
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunking_method = chunking_method
        self.event_chunking_params = event_chunking_params or {
            "window_size": 5000,
            "overlap": 1000
        }
        self.enable_visualization = enable_visualization

        # Check if visualization is requested but NetworkX is not available
        if self.enable_visualization:
            try:
                from ..utils.visualization import NETWORKX_AVAILABLE

                if not NETWORKX_AVAILABLE:
                    print("Warning: NetworkX not available. Visualization disabled.")
                    self.enable_visualization = False
            except ImportError:
                print(
                    "Warning: Visualization module not available. Visualization disabled."
                )
                self.enable_visualization = False

    def process_documents(
        self,
        documents: List[str],
        output_dir: Optional[str] = None,
        save_intermediates: bool = False,
        enable_coref: Optional[bool] = None,
        enable_chunking: Optional[bool] = None,
        enable_context: Optional[bool] = None,
        enable_spacetime: Optional[bool] = None,
        enable_conversation: Optional[bool] = None,
        enable_visualization: Optional[bool] = None,
    ) -> List[Dict[str, Dict]]:
        """Process multiple documents through the complete GSW pipeline with full parallelization.

        Args:
            documents: List of document texts to process
            output_dir: Directory to save outputs (if None, no saving)
            save_intermediates: Whether to save intermediate steps (coref, chunks, context)
            enable_coref: Override class setting for coreference resolution
            enable_chunking: Override class setting for chunking
            enable_context: Override class setting for context generation
            enable_spacetime: Override class setting for spacetime linking
            enable_visualization: Override class setting for visualization

        Returns:
            List of dictionaries, one per document. Each dict contains chunk_id -> chunk_data mappings.
            Structure: [{"0_0": {gsw, text, spacetime, context, ...}, "0_1": {...}}, {"1_0": {...}}]
        """

        # Override class settings if provided
        do_coref = enable_coref if enable_coref is not None else self.enable_coref
        do_chunking = (
            enable_chunking if enable_chunking is not None else self.enable_chunking
        )
        do_context = (
            enable_context if enable_context is not None else self.enable_context
        )
        do_spacetime = (
            enable_spacetime if enable_spacetime is not None else self.enable_spacetime
        )
        do_conversation = (
            enable_conversation if enable_conversation is not None else self.enable_conversation
        )
        do_visualization = (
            enable_visualization
            if enable_visualization is not None
            else self.enable_visualization
        )

        # Initialize unified chunk data structure: List[Dict[str, Dict]]
        # Each list item represents a document, each dict key is chunk_id, value is chunk data
        all_documents_data = []

        # Step 1: Coreference Resolution (parallel across all documents)
        if do_coref:
            print(f"--- Processing Coreference for {len(documents)} documents ---")
            
            # Separate documents into short and long based on token count
            short_docs = []
            long_docs = []
            
            for doc_idx, document in enumerate(documents):
                token_count = estimate_tokens(document, self.model_name)
                print(f"Document {doc_idx}: {token_count} tokens")
                
                doc_info = {"text": document, "idx": doc_idx}
                if token_count <= 3000:
                    short_docs.append(doc_info)
                else:
                    long_docs.append(doc_info)
            
            # Process short documents in parallel
            resolved_documents = {}
            if short_docs:
                print(f"Processing {len(short_docs)} short documents with single-pass coreference")
                short_resolved = process_single_coref_batch(short_docs, self.model_name)
                resolved_documents.update(short_resolved)
            
            # Process long documents with chunking in parallel
            if long_docs:
                print(f"Processing {len(long_docs)} long documents with chunked coreference")
                long_resolved = process_chunked_coref_batch(long_docs, self.model_name)
                resolved_documents.update(long_resolved)
            
            # Ensure documents are in the correct order
            resolved_documents = {idx: resolved_documents[idx] for idx in range(len(documents))}
        else:
            resolved_documents = {idx: doc for idx, doc in enumerate(documents)}

        # Step 2: Chunking and Initialize Chunk Data Structure
        print("--- Chunking Documents and Initializing Data Structure ---")
        for doc_idx, resolved_text in resolved_documents.items():
            document_chunks = {}

            if do_chunking:
                if self.chunking_method == "sentence":
                    # Use sentence-based chunking
                    chunks = chunk_text_unified(
                        resolved_text,
                        method="sentence",
                        chunk_size=self.chunk_size,
                        overlap=self.overlap
                    )
                elif self.chunking_method == "event_boundary":
                    # Use event boundary-based chunking
                    chunks = chunk_text_unified(
                        resolved_text,
                        method="event_boundary",
                        model_name=self.model_name,
                        window_size=self.event_chunking_params["window_size"],
                        overlap=self.event_chunking_params["overlap"]
                    )
                    print(f"DEBUG: Chunking method: {self.chunking_method}")
                    print(f"DEBUG: Number of chunks: {len(chunks)}")
                else:
                    raise ValueError(f"Unknown chunking method: {self.chunking_method}")
            else:
                # No chunking: each document becomes one chunk
                chunks = [
                    {
                        "text": resolved_text,
                        "idx": 0,
                        "start_sentence": 0,
                        "end_sentence": 0,
                    }
                ]

            # Initialize chunk data for this document
            for chunk in chunks:
                global_id = f"{doc_idx}_{chunk['idx']}"
                document_chunks[global_id] = {
                    "text": chunk["text"],
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk["idx"],
                    "global_id": global_id,
                    "start_sentence": chunk.get("start_sentence", 0),
                    "end_sentence": chunk.get("end_sentence", 0),
                    # Placeholders for pipeline results
                    "gsw": None,
                    "context": "",
                    "spacetime": {},
                }

            all_documents_data.append(document_chunks)

        # Step 3: Context Generation (parallel across all chunks, but only if multiple chunks exist)
        total_chunks = sum(len(doc_chunks) for doc_chunks in all_documents_data)
        should_generate_context = do_context and total_chunks > len(
            documents
        )  # More chunks than docs

        if should_generate_context:
            print(f"--- Generating Context for {total_chunks} chunks ---")
            context_model = ContextGenerator(
                model_name=self.model_name,
                generation_params={"temperature": 0.1, "max_tokens": 150},
            )

            # Prepare context inputs from all chunks across all documents
            context_inputs = []
            for doc_idx, doc_chunks in enumerate(all_documents_data):
                for chunk_id, chunk_data in doc_chunks.items():
                    context_inputs.append(
                        {
                            "doc_text": resolved_documents[doc_idx],
                            "chunk_text": chunk_data["text"],
                            "doc_idx": doc_idx,
                            "chunk_idx": chunk_data["chunk_idx"],
                            "global_id": chunk_id,
                        }
                    )

            # Process all context generation in parallel
            context_responses = context_model(context_inputs)

            # Update chunk data with context results
            for resp in context_responses.dataset:
                doc_idx = resp["doc_idx"]
                global_id = resp["global_id"]
                all_documents_data[doc_idx][global_id]["context"] = resp["context"]

        # Step 4: GSW Generation (parallel across all chunks)
        print(f"--- Generating GSWs for {total_chunks} chunks ---")
        gsw_model = GSWOperator(
            model_name=self.model_name, generation_params=self.generation_params
        )

        # Prepare GSW inputs from all chunks across all documents
        gsw_inputs = []
        for doc_idx, doc_chunks in enumerate(all_documents_data):
            for chunk_id, chunk_data in doc_chunks.items():
                gsw_input = {
                    "text": chunk_data["text"],
                    "idx": chunk_data["chunk_idx"],
                    "doc_idx": doc_idx,
                    "global_id": chunk_id,
                    "start_sentence": chunk_data["start_sentence"],
                    "end_sentence": chunk_data["end_sentence"],
                    "context": chunk_data["context"],
                }
                gsw_inputs.append(gsw_input)

        # Generate all GSWs in parallel
        gsw_responses = gsw_model(gsw_inputs)

        # Step 5: Parse responses and update chunk data with GSW structures
        for response in gsw_responses.dataset:
            try:
                gsw = parse_gsw(response["graph"])
                doc_idx = response["doc_idx"]
                global_id = response["global_id"]
                all_documents_data[doc_idx][global_id]["gsw"] = gsw
            except Exception as e:
                print(
                    f"Error parsing GSW for chunk {response.get('global_id', 'unknown')}: {e}"
                )
                continue

        # Step 6: Spacetime Linking (parallel across all chunks)
        if do_spacetime:
            print(f"--- Processing Spacetime Links for {total_chunks} chunks ---")
            spacetime_model = SpaceTimeLinker(
                model_name=self.model_name,
                generation_params={"temperature": 0.0, "max_tokens": 1000},
            )

            # Prepare spacetime inputs from all chunks that have GSW structures
            spacetime_inputs = []
            for doc_idx, doc_chunks in enumerate(all_documents_data):
                for chunk_id, chunk_data in doc_chunks.items():
                    if (
                        chunk_data["gsw"] is not None
                    ):  # Only process chunks with valid GSWs
                        # Format the GSW data for the LLM
                        operator_output = {
                            "entity_nodes": [
                                {
                                    "id": entity.id,
                                    "name": entity.name,
                                    "roles": [
                                        {"role": role.role, "states": role.states}
                                        for role in entity.roles
                                    ],
                                }
                                for entity in chunk_data["gsw"].entity_nodes
                            ]
                        }

                        spacetime_inputs.append(
                            {
                                "text_chunk_content": chunk_data["text"],
                                "operator_output_json": json.dumps(
                                    operator_output, indent=2
                                ),
                                "chunk_id": chunk_id,
                                "doc_idx": doc_idx,
                                "global_id": chunk_id,
                            }
                        )

            if spacetime_inputs:
                # Process all spacetime linking in parallel
                spacetime_responses = spacetime_model(spacetime_inputs)

                # Update chunk data with spacetime results
                for resp in spacetime_responses.dataset:
                    doc_idx = resp["doc_idx"]
                    global_id = resp["global_id"]
                    all_documents_data[doc_idx][global_id]["spacetime"] = {
                        "spatio_temporal_links": resp.get("spatio_temporal_links", []),
                        "full_response": resp.get("full_response", ""),
                    }

        # Step 6.1: Integrate spacetime data into GSW structures
        if do_spacetime:
            print(f"--- Integrating Spacetime Data into GSW Structures ---")
            self._integrate_spacetime_into_gsw_structures(all_documents_data)

        # Step 6.5: Conversation Linking (parallel across all chunks)
        if do_conversation:
            print(f"--- Processing Conversation Links for {total_chunks} chunks ---")
            all_documents_data = process_conversation_batch(
                all_documents_data,
                model_name=self.model_name,
                generation_params={"temperature": 0.0, "max_tokens": 1500}
            )

        # Step 7: Save outputs if requested
        if output_dir:
            self._save_outputs_unified(
                output_dir=output_dir,
                save_intermediates=save_intermediates,
                resolved_documents=resolved_documents,
                all_documents_data=all_documents_data,
                do_visualization=do_visualization,
                do_conversation=do_conversation,
            )

        return all_documents_data

    def _integrate_spacetime_into_gsw_structures(self, all_documents_data: List[Dict[str, Dict]]) -> None:
        """
        Integrate spacetime data into GSW structures by creating SpaceNode and TimeNode objects.
        
        This converts the separate spacetime link data generated by SpaceTimeLinker
        into actual nodes and edges in the GSW structure.
        """
        total_space_nodes = 0
        total_time_nodes = 0
        total_space_edges = 0
        total_time_edges = 0
        
        for doc_idx, doc_chunks in enumerate(all_documents_data):
            for chunk_id, chunk_data in doc_chunks.items():
                gsw = chunk_data.get("gsw")
                spacetime_data = chunk_data.get("spacetime", {})
                
                if not gsw or not spacetime_data:
                    continue
                
                spatio_temporal_links = spacetime_data.get("spatio_temporal_links", [])
                
                if not spatio_temporal_links:
                    continue
                
                # Track space and time nodes by tag_value to avoid duplicates
                space_nodes_created = {}
                time_nodes_created = {}
                
                for link in spatio_temporal_links:
                    linked_entities = link.get("linked_entities", [])
                    tag_type = link.get("tag_type", "")  # "spatial" or "temporal"
                    tag_value = link.get("tag_value")  # actual location/time name
                    
                    if not linked_entities or not tag_type or tag_value is None:
                        continue
                    
                    if tag_type == "spatial":
                        # Create or reuse space node
                        if tag_value in space_nodes_created:
                            space_node_id = space_nodes_created[tag_value]
                        else:
                            space_node_id = f"sp_{len(gsw.space_nodes)}_{doc_idx}_{chunk_id}"
                            space_node = SpaceNode(
                                id=space_node_id,
                                current_name=tag_value,
                                name_history={chunk_id: tag_value},
                                chunk_id=chunk_id,
                                type="space"
                            )
                            gsw.add_space_node(space_node)
                            space_nodes_created[tag_value] = space_node_id
                            total_space_nodes += 1
                        
                        # Add edges for each linked entity
                        for entity_id in linked_entities:
                            # Check if entity exists in GSW
                            if any(e.id == entity_id for e in gsw.entity_nodes):
                                gsw.add_space_edge(entity_id, space_node_id)
                                total_space_edges += 1
                            else:
                                print(f"Warning: Entity {entity_id} not found in GSW for space linking")
                    
                    elif tag_type == "temporal":
                        # Create or reuse time node
                        if tag_value in time_nodes_created:
                            time_node_id = time_nodes_created[tag_value]
                        else:
                            time_node_id = f"tm_{len(gsw.time_nodes)}_{doc_idx}_{chunk_id}"
                            time_node = TimeNode(
                                id=time_node_id,
                                current_name=tag_value,
                                name_history={chunk_id: tag_value},
                                chunk_id=chunk_id,
                                type="time"
                            )
                            gsw.add_time_node(time_node)
                            time_nodes_created[tag_value] = time_node_id
                            total_time_nodes += 1
                        
                        # Add edges for each linked entity
                        for entity_id in linked_entities:
                            # Check if entity exists in GSW
                            if any(e.id == entity_id for e in gsw.entity_nodes):
                                gsw.add_time_edge(entity_id, time_node_id)
                                total_time_edges += 1
                            else:
                                print(f"Warning: Entity {entity_id} not found in GSW for time linking")
        
        print(f"   ✅ Spacetime integration complete!")
        print(f"      Space nodes created: {total_space_nodes}")
        print(f"      Time nodes created: {total_time_nodes}")
        print(f"      Space edges created: {total_space_edges}")
        print(f"      Time edges created: {total_time_edges}")

    def _save_outputs_unified(
        self,
        output_dir: str,
        save_intermediates: bool,
        resolved_documents: Dict[int, str],
        all_documents_data: List[Dict[str, Dict]],
        do_visualization: bool,
        do_conversation: bool,
    ):
        """Save all outputs according to the unified chunk data structure."""

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
            spacetime_dir = os.path.join(output_dir, "spacetime")
            if do_conversation:
                conversation_dir = os.path.join(output_dir, "conversation")
            os.makedirs(coref_dir, exist_ok=True)
            os.makedirs(chunks_dir, exist_ok=True)
            os.makedirs(context_dir, exist_ok=True)
            os.makedirs(spacetime_dir, exist_ok=True)
            if do_conversation:
                os.makedirs(conversation_dir, exist_ok=True)

            # Save coreference results
            for doc_idx, resolved_text in resolved_documents.items():
                with open(os.path.join(coref_dir, f"coref_{doc_idx}.txt"), "w") as f:
                    f.write(resolved_text)

            # Save individual chunks, contexts, and spacetime results
            for doc_idx, doc_chunks in enumerate(all_documents_data):
                for chunk_id, chunk_data in doc_chunks.items():
                    doc_idx_val = chunk_data["doc_idx"]
                    chunk_idx_val = chunk_data["chunk_idx"]

                    # Save chunks
                    doc_chunks_dir = os.path.join(chunks_dir, f"doc_{doc_idx_val}")
                    os.makedirs(doc_chunks_dir, exist_ok=True)
                    chunk_file = os.path.join(
                        doc_chunks_dir, f"chunk_{chunk_idx_val}.txt"
                    )
                    with open(chunk_file, "w") as f:
                        f.write(chunk_data["text"])

                    # Save contexts
                    if chunk_data["context"]:
                        doc_context_dir = os.path.join(
                            context_dir, f"doc_{doc_idx_val}"
                        )
                        os.makedirs(doc_context_dir, exist_ok=True)
                        context_file = os.path.join(
                            doc_context_dir, f"context_{chunk_idx_val}.txt"
                        )
                        with open(context_file, "w") as f:
                            f.write(chunk_data["context"])

                    # Save spacetime results
                    if chunk_data["spacetime"]:
                        doc_spacetime_dir = os.path.join(
                            spacetime_dir, f"doc_{doc_idx_val}"
                        )
                        os.makedirs(doc_spacetime_dir, exist_ok=True)
                        spacetime_file = os.path.join(
                            doc_spacetime_dir, f"spacetime_{chunk_idx_val}.json"
                        )
                        with open(spacetime_file, "w") as f:
                            json.dump(chunk_data["spacetime"], f, indent=4)

                    # Save conversation results (now part of main GSW structure)
                    if do_conversation and chunk_data.get("has_conversation", False):
                        doc_conversation_dir = os.path.join(
                            conversation_dir, f"doc_{doc_idx_val}"
                        )
                        os.makedirs(doc_conversation_dir, exist_ok=True)
                        conversation_file = os.path.join(
                            doc_conversation_dir, f"conversation_{chunk_idx_val}.json"
                        )
                        # Extract conversation data from the main GSW structure
                        gsw_data = chunk_data["gsw"].model_dump(mode="json")
                        conversation_data = {
                            "conversation_nodes": gsw_data.get("conversation_nodes", []),
                            "conversation_participant_edges": gsw_data.get("conversation_participant_edges", []),
                            "conversation_topic_edges": gsw_data.get("conversation_topic_edges", []),
                            "conversation_space_edges": gsw_data.get("conversation_space_edges", []),
                            "conversation_time_edges": gsw_data.get("conversation_time_edges", [])
                        }
                        with open(conversation_file, "w") as f:
                            json.dump(conversation_data, f, indent=4)

        # Save combined results (all unified chunk data in one JSON)
        combined_file = os.path.join(output_dir, "gsw_results_combined.json")
        combined_data = {
            "metadata": {
                "total_documents": len(all_documents_data),
                "total_chunks": sum(
                    len(doc_chunks) for doc_chunks in all_documents_data
                ),
                "processed_at": __import__("datetime").datetime.now().isoformat(),
            },
            "documents": {},
        }

        # Convert all_documents_data to JSON-serializable format
        for doc_idx, doc_chunks in enumerate(all_documents_data):
            combined_data["documents"][f"doc_{doc_idx}"] = {}
            for chunk_id, chunk_data in doc_chunks.items():
                # Convert GSW structure to dict if it exists
                chunk_export = chunk_data.copy()
                if chunk_export["gsw"] is not None:
                    chunk_export["gsw"] = chunk_export["gsw"].model_dump(mode="json")
                # Note: Conversation data is now part of the main GSW structure
                combined_data["documents"][f"doc_{doc_idx}"][chunk_id] = chunk_export

        with open(combined_file, "w") as f:
            json.dump(combined_data, f, indent=2)

        # Save individual results and visualizations
        print("--- Saving Networks and Visualizations ---")
        for doc_idx, doc_chunks in enumerate(all_documents_data):
            for chunk_id, chunk_data in doc_chunks.items():
                try:
                    if chunk_data["gsw"] is None:
                        print(
                            f"Warning: No GSW structure for chunk {chunk_id}, skipping..."
                        )
                        continue

                    doc_idx_val = chunk_data["doc_idx"]
                    chunk_idx_val = chunk_data["chunk_idx"]
                    file_stem = f"gsw_{doc_idx_val}_{chunk_idx_val}"

                    # Create document-specific subdirectories
                    doc_networks_dir = os.path.join(networks_dir, f"doc_{doc_idx_val}")
                    doc_networks_raw_dir = os.path.join(
                        networks_raw_dir, f"doc_{doc_idx_val}"
                    )
                    os.makedirs(doc_networks_dir, exist_ok=True)
                    os.makedirs(doc_networks_raw_dir, exist_ok=True)

                    # Save raw chunk data (includes text, spacetime, context, etc.)
                    raw_chunk_data = chunk_data.copy()
                    if raw_chunk_data["gsw"] is not None:
                        raw_chunk_data["gsw"] = raw_chunk_data["gsw"].model_dump(
                            mode="json"
                        )
                    # Note: Conversation data is now part of the main GSW structure

                    raw_file = os.path.join(doc_networks_raw_dir, f"{file_stem}.json")
                    with open(raw_file, "w") as f:
                        json.dump(raw_chunk_data, f, indent=4)

                    # Save parsed network (GSW structure only)
                    network_file = os.path.join(doc_networks_dir, f"{file_stem}.json")
                    with open(network_file, "w") as f:
                        json.dump(
                            chunk_data["gsw"].model_dump(mode="json"), f, indent=4
                        )

                    # Save visualization if enabled
                    if do_visualization:
                        try:
                            from ..utils.visualization import (
                                create_and_save_gsw_visualization,
                            )

                            doc_viz_dir = os.path.join(viz_dir, f"doc_{doc_idx_val}")
                            os.makedirs(doc_viz_dir, exist_ok=True)
                            viz_file = os.path.join(doc_viz_dir, f"{file_stem}.cyjs")
                            create_and_save_gsw_visualization(
                                chunk_data["gsw"], viz_file
                            )

                        except Exception as viz_error:
                            print(
                                f"Warning: Failed to create visualization for {file_stem}: {viz_error}"
                            )

                except Exception as e:
                    print(f"Error saving outputs for chunk {chunk_id}: {e}")
                    continue

    def create_visualization(self, gsw: GSWStructure, output_path: str) -> None:
        """Create and save a visualization for a GSW structure.

        Args:
            gsw: GSWStructure to visualize
            output_path: Path to save the visualization file
        """
        if not self.enable_visualization:
            print(
                "Visualization not enabled. Set enable_visualization=True or install NetworkX."
            )
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
        enable_coref=True,
        enable_chunking=True,
        enable_context=True,
        enable_conversation=True,
        chunking_method="event_boundary",
        event_chunking_params={
            "window_size": 5000,
            "overlap": 1000
        },
        chunk_size=3,
        overlap=1,
    )

    # test_document = """
    # John walked into the coffee shop. He ordered a large latte from the barista. 
    # The barista, whose name was Sarah, smiled at him. She prepared the drink carefully. 
    # John paid for his coffee and sat down at a table near the window. 
    # He opened his laptop and began working on his presentation.
    # """
    
    with open(
        "/mnt/SSD1/nlp/gsw/new_data/corpus/narrativeqa_docs_10_raw_docs.json", "r"
    ) as f:
        chapters = json.load(f)
    test_document = chapters['5']

    print("Processing single document...")
    gsw_structures = processor.process_documents([test_document], output_dir="test_output", save_intermediates=True)

    print("Processing complete! Results saved to test_output/ directory.")
