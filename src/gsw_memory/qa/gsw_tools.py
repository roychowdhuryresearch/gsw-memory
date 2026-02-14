"""
GSW Tools for Agentic Question Answering.

This module provides tool functions that give agentic answering agents
direct access to query and explore the GSW structure dynamically.
"""

import json
import os
import hashlib
from typing import Dict, List, Any, Union, Optional
from rank_bm25 import BM25Okapi
import numpy as np
try:
    import faiss
    from langchain_voyageai import VoyageAIEmbeddings
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    faiss = None
    VoyageAIEmbeddings = None
from ..memory.models import GSWStructure

class GSWTools:
    """
    Tool collection for agentic GSW exploration.
    
    Provides direct access to GSW structure for dynamic querying
    and multi-hop reasoning with minimal, focused tools.
    """
    
    def __init__(self, gsw_file_paths: Union[str, List[str]], gpu_device: int = 0):
        """Initialize GSW tools with GSW file path(s).

        Args:
            gsw_file_paths: Path or list of paths to GSW files
            gpu_device: GPU device ID for FAISS GPU acceleration (default: 0)
        """
        # Normalize to list
        if isinstance(gsw_file_paths, str):
            self.gsw_files = [gsw_file_paths]
        else:
            self.gsw_files = gsw_file_paths

        self.bm25 = None
        self.entity_corpus = []
        self.entity_metadata = []
        self._index_built = False
        self.gpu_device = gpu_device

        # Embedding search components
        self.embedding_model = None
        self.faiss_index = None
        self.entity_embeddings = []
        self.embedding_metadata = []  # Same structure as entity_metadata

        # Initialize GPU resources for FAISS
        self.gpu_resources = None
        if EMBEDDING_AVAILABLE and faiss is not None:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize GPU resources: {e}")

        # Cache directory for embeddings
        self.cache_dir = ".gsw_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def build_index(self):
        """
        Explicitly build the search index. Call this before using search methods 
        for better performance and progress tracking.
        """
        if self._index_built:
            print("Search index already built.")
            return
            
        print(f"Building search index from {len(self.gsw_files)} GSW files...")
        
        self.entity_corpus = []
        self.entity_metadata = []
        
        # Load all GSW files and combine entities
        processed_files = 0
        total_entities = 0
        
        for i, file_path in enumerate(self.gsw_files):
            try:
                with open(file_path, 'r') as f:
                    gsw_data = json.load(f)
                gsw = GSWStructure.from_json(gsw_data)
                
                file_entities = 0
                for entity in gsw.entity_nodes:
                    # Tokenize entity name (could extend with additional context)
                    tokens = entity.name.lower().split()
                    self.entity_corpus.append(tokens)
                    self.entity_metadata.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "source_file": file_path,
                        "global_id": f"{file_path}::{entity.id}",  # Prevent ID collisions
                        # "roles": [{"role": r.role, "states": r.states} for r in entity.roles]
                    })
                    file_entities += 1
                    total_entities += 1
                
                processed_files += 1
                
                # Show progress every 100 files or for small datasets every 10 files
                progress_interval = 100 if len(self.gsw_files) > 100 else max(1, len(self.gsw_files) // 10)
                if processed_files % progress_interval == 0 or processed_files == len(self.gsw_files):
                    print(f"  Processed {processed_files}/{len(self.gsw_files)} files, {total_entities} entities loaded")
                    
            except Exception as e:
                print(f"Warning: Failed to load GSW file {file_path}: {e}")
                continue
        
        # Build BM25 index
        print(f"Building BM25 index from {total_entities} entities...")
        if self.entity_corpus:
            self.bm25 = BM25Okapi(self.entity_corpus)
        else:
            self.bm25 = None
        
        # Build embedding index if available
        if EMBEDDING_AVAILABLE and total_entities > 0:
            print(f"Building embedding index from {total_entities} entities...")
            self._build_embedding_index()
        elif not EMBEDDING_AVAILABLE:
            print("⚠️  Embedding search not available (missing dependencies)")
            
        self._index_built = True
        print(f"✅ Search index built successfully! {total_entities} entities indexed from {processed_files} files.")
    
    def _build_search_index(self):
        """Build BM25 search index for entities (called lazily). Use build_index() for explicit building."""
        if self._index_built:
            return  # Already built
            
        print("Building search index lazily...")
        self.build_index()
    
    def _build_embedding_index(self):
        """Build FAISS GPU embedding index for semantic search."""
        if not EMBEDDING_AVAILABLE:
            return

        # Initialize embedding model
        self.embedding_model = VoyageAIEmbeddings(model="voyage-3")

        # Copy metadata for embeddings (same as BM25)
        self.embedding_metadata = self.entity_metadata.copy()

        # Extract entity names for embedding
        entity_names = [meta["entity_name"] for meta in self.entity_metadata]

        # Cache file names - GPU format
        faiss_index_cache = "gsw_embeddings_gpu.faiss"
        metadata_cache = "gsw_metadata.json"
        # Legacy HNSW cache path for backward compatibility
        faiss_hnsw_cache = "gsw_embeddings_hnsw.faiss"

        # Try to load from FAISS GPU cache
        if self.gpu_resources is not None and os.path.exists(faiss_index_cache) and os.path.exists(metadata_cache):
            try:
                print(f"Loading cached FAISS GPU index from {faiss_index_cache}")

                with open(metadata_cache, 'r') as f:
                    cached_metadata = json.load(f)

                # Check if cache is valid
                cached_count = cached_metadata.get('num_entities', 0)
                if cached_count == len(entity_names):
                    # Load CPU index from disk
                    cpu_index = faiss.read_index(faiss_index_cache)

                    # Transfer to GPU
                    self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

                    # Reconstruct embeddings from CPU index for compatibility
                    num_vectors = cpu_index.ntotal
                    self.entity_embeddings = faiss.vector_to_array(cpu_index.reconstruct_n(0, num_vectors))
                    self.entity_embeddings = self.entity_embeddings.reshape(num_vectors, -1)

                    print(f"✅ Loaded FAISS GPU index with {num_vectors} cached embeddings")
                    return
                else:
                    print("⚠️ Cache size mismatch, rebuilding...")

            except Exception as e:
                print(f"⚠️ Failed to load GPU cache: {e}, trying HNSW fallback...")

        # Try to load from legacy HNSW cache and convert to GPU
        if self.gpu_resources is not None and os.path.exists(faiss_hnsw_cache) and os.path.exists(metadata_cache):
            try:
                print(f"Loading cached FAISS HNSW index from {faiss_hnsw_cache}")

                with open(metadata_cache, 'r') as f:
                    cached_metadata = json.load(f)

                # Check if cache is valid
                cached_count = cached_metadata.get('num_entities', 0)
                if cached_count == len(entity_names):
                    # Load HNSW index
                    hnsw_index = faiss.read_index(faiss_hnsw_cache)

                    # Reconstruct embeddings from HNSW index
                    num_vectors = hnsw_index.ntotal
                    self.entity_embeddings = faiss.vector_to_array(hnsw_index.reconstruct_n(0, num_vectors))
                    self.entity_embeddings = self.entity_embeddings.reshape(num_vectors, -1)

                    # Build GPU flat index from embeddings
                    embedding_dim = self.entity_embeddings.shape[1]
                    cpu_index = faiss.IndexFlatIP(embedding_dim)
                    embeddings_normalized = self.entity_embeddings.copy()
                    faiss.normalize_L2(embeddings_normalized)
                    cpu_index.add(embeddings_normalized)

                    # Transfer to GPU
                    self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

                    print(f"✅ Converted HNSW cache to GPU index with {num_vectors} embeddings")
                    return
                else:
                    print("⚠️ Cache size mismatch, rebuilding...")

            except Exception as e:
                print(f"⚠️ Failed to load HNSW cache: {e}, rebuilding...")

        # Build embeddings from scratch
        print("Building embeddings from scratch...")
        batch_size = 500  # Increased from 100 to reduce API calls
        all_embeddings = []

        for i in range(0, len(entity_names), batch_size):
            batch = entity_names[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

            # Progress update
            processed = min(i + batch_size, len(entity_names))
            if processed % 1000 == 0 or processed == len(entity_names):
                print(f"  Embedded {processed}/{len(entity_names)} entities")

        # Convert to numpy array
        self.entity_embeddings = np.array(all_embeddings, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.entity_embeddings)

        # Create FAISS GPU index
        embedding_dim = self.entity_embeddings.shape[1]
        if self.gpu_resources is not None:
            # Create CPU flat index for exact nearest neighbor search
            cpu_index = faiss.IndexFlatIP(embedding_dim)
            cpu_index.add(self.entity_embeddings)

            # Transfer to GPU
            self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)

            # Save to cache (GPU format)
            try:
                # Transfer from GPU to CPU for saving
                cpu_index_for_save = faiss.index_gpu_to_cpu(self.faiss_index)

                # Save CPU FAISS index
                faiss.write_index(cpu_index_for_save, faiss_index_cache)
                print(f"💾 Saved FAISS GPU index to {faiss_index_cache}")

                # Save metadata with entity count
                metadata_with_count = {
                    "num_entities": len(entity_names),
                    "embedding_dim": embedding_dim
                }
                with open(metadata_cache, 'w') as f:
                    json.dump(metadata_with_count, f)
                print(f"💾 Saved metadata")
            except Exception as e:
                print(f"⚠️ Failed to save cache: {e}")
        else:
            # Fallback to CPU index if GPU not available
            cpu_index = faiss.IndexFlatIP(embedding_dim)
            cpu_index.add(self.entity_embeddings)
            self.faiss_index = cpu_index
            print("⚠️ GPU not available, using CPU index")

        print(f"✅ Embedding index built with {len(entity_names)} entities")
    
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
        
        # Search in entity names across all GSW files
        for file_path in self.gsw_files:
            try:
                with open(file_path, 'r') as f:
                    gsw_data = json.load(f)
                gsw = GSWStructure.from_json(gsw_data)
                for entity in gsw.entity_nodes:
                    entity_name_lower = entity.name.lower()
                    matching_tokens = sum(1 for token in query_tokens 
                                        if token in entity_name_lower)
                    
                    # Even 1 token match for entities can be meaningful
                    if matching_tokens >= 1:
                        results.append({
                            "type": "entity",
                            "entity_id": entity.id,
                            "entity_name": entity.name,
                            "source_file": file_path,
                            "global_id": f"{file_path}::{entity.id}",
                            "roles": [{"role": r.role, "states": r.states} 
                                     for r in entity.roles],
                            "match_score": matching_tokens
                        })
            except Exception as e:
                print(f"Warning: Failed to load GSW file {file_path}: {e}")
                continue
        
        # Sort by relevance score and return top results
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:limit]
    
    def search_gsw_bm25(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across GSW entities using BM25 ranking.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching items sorted by BM25 relevance score
        """
        # Build index if not already built
        self._build_search_index()
        
        if not self.bm25 or not self.entity_corpus:
            return []
            
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top results
        top_indices = scores.argsort()[-limit:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches with positive scores
                result = self.entity_metadata[idx].copy()
                result["match_score"] = float(scores[idx])
                result["type"] = "entity"
                results.append(result)
        
        return results
    
    def search_gsw_entity_embeddings(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across GSW entities using semantic embeddings.
        
        Better for handling name variations, titles, and partial matches.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities sorted by semantic similarity
        """
        # Build index if not already built
        self._build_search_index()
        
        if not EMBEDDING_AVAILABLE or self.faiss_index is None:
            print("Warning: Embedding search not available, falling back to BM25")
            return self.search_gsw_bm25(query, limit)
        
        # Embed the query
        query_embedding = np.array(self.embedding_model.embed_query(query), dtype=np.float32)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        similarities, indices = self.faiss_index.search(query_embedding, limit)
        
        # Convert to result format
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx >= 0 and similarity > 0:  # Valid index with positive similarity
                result = self.embedding_metadata[idx].copy()
                result["match_score"] = float(similarity)
                result["type"] = "entity"
                results.append(result)
        
        return results
    
    def get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """
        Get the context of an entity - all questions it participates in
        and other entities in those questions.
        
        Args:
            entity_id: ID of the entity to get context for (can be global_id or original entity_id)
            
        Returns:
            Dict containing entity info and all questions it participates in
        """
        # Handle both global IDs and original entity IDs
        if "::" in entity_id:
            # Global ID format: "file_path::entity_id"
            source_file, actual_entity_id = entity_id.split("::", 1)
        else:
            # Original entity ID - need to find it in all files
            source_file = None
            actual_entity_id = entity_id
            
            # Search through all files to find this entity
            for file_path in self.gsw_files:
                try:
                    with open(file_path, 'r') as f:
                        gsw_data = json.load(f)
                    gsw = GSWStructure.from_json(gsw_data)
                    if gsw.get_entity_by_id(actual_entity_id):
                        source_file = file_path
                        break
                except Exception:
                    continue
            
            if not source_file:
                return {"error": f"Entity {entity_id} not found in any GSW file"}
        
        # Load the specific GSW file
        try:
            with open(source_file, 'r') as f:
                gsw_data = json.load(f)
            gsw = GSWStructure.from_json(gsw_data)
        except Exception as e:
            return {"error": f"Failed to load GSW file {source_file}: {e}"}
        
        entity = gsw.get_entity_by_id(actual_entity_id)
        if not entity:
            return {"error": f"Entity {actual_entity_id} not found in {source_file}"}
        
        context = {
            "entity_id": actual_entity_id,
            "global_id": f"{source_file}::{actual_entity_id}",
            "entity_name": entity.name,
            "source_file": source_file,
            "roles": [{"role": r.role, "states": r.states} 
                     for r in entity.roles],
            "questions": []
        }
        
        # Find all questions this entity answers
        for vp in gsw.verb_phrase_nodes:
            for question in vp.questions:
                if actual_entity_id in question.answers:
                    # Get all other entities in this same question
                    other_entities = []
                    for answer_id in question.answers:
                        if answer_id != actual_entity_id and answer_id != "None":
                            other_entity = gsw.get_entity_by_id(answer_id)
                            if other_entity:
                                other_entities.append({
                                    "entity_id": answer_id,
                                    "global_id": f"{source_file}::{answer_id}",
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
    
    
    def get_multiple_entity_contexts(self, entity_ids: List[str]) -> Dict[str, Any]:
        """
        Get the context of multiple entities - all questions they participate in
        and other entities in those questions.
        
        Args:
            entity_ids: IDs of the entities to get context for (can be global_id or original entity_id)
            
        Returns:
            Dict containing entity info and all questions it participates in
        """
        # Handle both global IDs and original entity IDs
        contexts = []
        for entity_id in entity_ids:
            if "::" in entity_id:
                # Global ID format: "file_path::entity_id"
                source_file, actual_entity_id = entity_id.split("::", 1)
            else:
                # Original entity ID - need to find it in all files
                source_file = None
                actual_entity_id = entity_id
                
                # Search through all files to find this entity
                for file_path in self.gsw_files:
                    try:
                        with open(file_path, 'r') as f:
                            gsw_data = json.load(f)
                        gsw = GSWStructure.from_json(gsw_data)
                        if gsw.get_entity_by_id(actual_entity_id):
                            source_file = file_path
                            break
                    except Exception:
                        continue
                
                if not source_file:
                    return {"error": f"Entity {entity_id} not found in any GSW file"}
            
            # Load the specific GSW file
            try:
                with open(source_file, 'r') as f:
                    gsw_data = json.load(f)
                gsw = GSWStructure.from_json(gsw_data)
            except Exception as e:
                return {"error": f"Failed to load GSW file {source_file}: {e}"}
            
            entity = gsw.get_entity_by_id(actual_entity_id)
            if not entity:
                return {"error": f"Entity {actual_entity_id} not found in {source_file}"}
            
            context = {
                "entity_id": actual_entity_id,
                "global_id": f"{source_file}::{actual_entity_id}",
                "entity_name": entity.name,
                "source_file": source_file,
                "roles": [{"role": r.role, "states": r.states} 
                        for r in entity.roles],
                "questions": []
            }
            
            # Find all questions this entity answers
            for vp in gsw.verb_phrase_nodes:
                for question in vp.questions:
                    if actual_entity_id in question.answers:
                        # Get all other entities in this same question
                        other_entities = []
                        for answer_id in question.answers:
                            if answer_id != actual_entity_id and answer_id != "None":
                                other_entity = gsw.get_entity_by_id(answer_id)
                                if other_entity:
                                    other_entities.append({
                                        "entity_id": answer_id,
                                        "global_id": f"{source_file}::{answer_id}",
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
            
            contexts.append(context)
        
        return contexts