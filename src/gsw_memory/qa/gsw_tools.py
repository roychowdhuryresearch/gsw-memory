"""
GSW Tools for Agentic Question Answering.

This module provides tool functions that give agentic answering agents
direct access to query and explore the GSW structure dynamically.
"""

import json
from typing import Dict, List, Any, Union
from rank_bm25 import BM25Okapi
from ..memory.models import GSWStructure
from gsw_memory import EntitySummaryAggregator
import os
from transformers import AutoModel
from torch.nn import functional as F
import torch
from openai import OpenAI

class GSWTools:
    """
    Tool collection for agentic GSW exploration.
    
    Provides direct access to GSW structure for dynamic querying
    and multi-hop reasoning with minimal, focused tools.
    """
    
    def __init__(self, gsw_file_paths: Union[str, List[str]]):
        """Initialize GSW tools with GSW file path(s)."""
        # Normalize to list
        if isinstance(gsw_file_paths, str):
            self.gsw_files = [gsw_file_paths]
        else:
            self.gsw_files = gsw_file_paths
            
        self.bm25_entity_name = None
        self.bm25_entity_summary = None
        self.entity_corpus = []
        self.entity_summary_corpus = []
        self.entity_metadata = []
        self.entity_summary_metadata = []
        self.summary_embeddings = None
        self._index_built_entity_name = False
        self._index_built_entity_summary = False
        self.doc_idx_to_summaries = {}
        
        self.embedding_model_name = "nvidia/NV-Embed-v2"
        self.generation_params = {"temperature": 0.0}
        self.client = OpenAI()
        self.cache_dir = "/home/yigit/codebase/gsw-memory/logs/entity_summaries_embeddings"
        self.device = "cuda:1"
        
        # Initialize embedding model using transformers
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = AutoModel.from_pretrained(
            self.embedding_model_name, 
            trust_remote_code=True
        ).to(self.device)
        self.embedding_model.eval()
        
        # Set up max sequence length
        self.max_length = 32768
        
        # Define task instruction for nvembed-v2
        self.task_instruct = "Given a question, retrieve relevant documents that best answer the question"
        self.query_prefix = f"Instruct: {self.task_instruct}\nQuery: "
        self.passage_prefix = ""  # No instruction needed for passages
        
    def generate_entity_summaries(self):
        """Generate entity summaries for factual Q&A."""
        print("\n=== Generating Entity Summaries ===")
        if os.path.exists(os.path.join("/home/yigit/codebase/gsw-memory/logs/entity_summaries", "doc_idx_to_summaries_all.json")):
            with open(os.path.join("/home/yigit/codebase/gsw-memory/logs/entity_summaries", "doc_idx_to_summaries_all.json"), "r") as f:
                self.doc_idx_to_summaries = json.load(f)
            print("Loaded existing entity summaries")
            return
        
        llm_config = {
            "model_name": "gpt-4o",
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 500,
            },  # Shorter for facts
        }
        for i, file_path in enumerate(self.gsw_files):
            try:
                with open(file_path, 'r') as f:
                    gsw_data = json.load(f)
                gsw = GSWStructure.from_json(gsw_data)
                aggregator = EntitySummaryAggregator(gsw, llm_config)
                summaries = aggregator.precompute_summaries(include_space_time=False)
                self.doc_idx_to_summaries[file_path.split("/")[-1].split(".")[0]] = summaries
            except Exception as e:
                print(f"Warning: Failed to load GSW file {file_path}: {e}")
                continue
            
        # save doc_idx_to_summaries to a json file in the log_dir
        with open(os.path.join("/home/yigit/codebase/gsw-memory/logs/entity_summaries", "doc_idx_to_summaries.json"), "w") as f:
            json.dump(self.doc_idx_to_summaries, f)

        print(f"Generated summaries for {len(summaries)} entities")   
    
    def build_entity_index(self):
        """
        Explicitly build the search index. Call this before using search methods 
        for better performance and progress tracking.
        """
        if self._index_built_entity_name:
            # print("Search index already built.")
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
                        "summary": self.doc_idx_to_summaries[file_path.split("/")[-1].split(".")[0]][entity.id]["summary"],
                        "source_file": file_path,
                        "global_id": f"{file_path}::{entity.id}",  # Prevent ID collisions
                        "roles": [{"role": r.role, "states": r.states} for r in entity.roles]
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
            self.bm25_entity_name = BM25Okapi(self.entity_corpus)
        else:
            self.bm25_entity_name = None
            
        self._index_built_entity_name = True
        print(f"✅ Search index built successfully! {total_entities} entities indexed from {processed_files} files.")
        
    def build_entity_summary_index(self):
        """
        Explicitly build the search index. Call this before using search methods 
        for better performance and progress tracking.
        """
        if self._index_built_entity_summary:
            # print("Search index already built.")
            return
            
        print(f"Building search index from {len(self.gsw_files)} GSW files...")
        
        self.entity_summary_corpus = []
        self.entity_summary_metadata = []
        
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
                    tokens = self.doc_idx_to_summaries[file_path.split("/")[-1].split(".")[0]][entity.id]["summary"].lower().split()
                    self.entity_summary_corpus.append(tokens)
                    self.entity_summary_metadata.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "summary": self.doc_idx_to_summaries[file_path.split("/")[-1].split(".")[0]][entity.id]["summary"],
                        "source_file": file_path,
                        "global_id": f"{file_path}::{entity.id}",  # Prevent ID collisions
                        "roles": [{"role": r.role, "states": r.states} for r in entity.roles]
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
        print(f"Building Summary BM25 index from {total_entities} entities...")
        if self.entity_summary_corpus:
            self.bm25_entity_summary = BM25Okapi(self.entity_summary_corpus)
        else:
            self.bm25_entity_summary = None
            
        self._index_built_entity_summary = True
        print(f"✅ Summary Search index built successfully! {total_entities} entities indexed from {processed_files} files.")
        
    def _build_embeddings(self):
        """Build or load document embeddings using transformers."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        embeddings_path = os.path.join(self.cache_dir, "nvembed_v2_transformers_entity_summaries_embeddings.pt")
        
        if os.path.exists(embeddings_path):
            print("Loading cached embeddings...")
            # Load directly to the target device
            self.summary_embeddings = torch.load(embeddings_path, map_location=self.device)
            print(f"Loaded embeddings with shape {self.summary_embeddings.shape}")
        else:
            print("Building embeddings for corpus...")
            
            # Ensure we have the metadata built first
            if not self._index_built_entity_summary:
                print("Building entity summary index first...")
                self.build_entity_summary_index()
            
            # Prepare documents for embedding using metadata
            # Combine title and text for better representation
            documents = []
            for doc_metadata in self.entity_summary_metadata:
                # Format: "Title: [title]\n[text]"
                doc_text = f"Title: {doc_metadata['entity_name']}\n{doc_metadata['summary']}"
                documents.append(doc_text)
            
            # Encode documents in batches
            batch_size = 4
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    print(f"Encoding batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                    
                    # Encode with nvembed-v2 using the direct encode method
                    embeddings = self.embedding_model.encode(
                        batch,
                        instruction=self.passage_prefix,  # No instruction for passages
                        max_length=self.max_length
                    )
                    
                    # Normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu())
                    
                    # Clear cache to save memory
                    torch.cuda.empty_cache()
            
            # Stack all embeddings
            self.summary_embeddings = torch.cat(all_embeddings, dim=0)
            
            print(f"Generated embeddings with shape {self.summary_embeddings.shape}")
            
            # Save embeddings
            print("Saving embeddings to cache...")
            torch.save(self.summary_embeddings, embeddings_path)
            print("Embeddings saved successfully")
            
    def search_gsw_embeddings_of_entity_summaries(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most relevant documents using embedding similarity.
        
        Args:
            query: The search query
            top_k: Number of top results to return (default 5)
            
        Returns:
            List of relevant documents with rich metadata and scores
        """
        # Ensure embeddings and metadata are built
        if self.summary_embeddings is None:
            print("Building embeddings...")
            self._build_embeddings()
        
        # Ensure metadata index is built
        if not self._index_built_entity_summary:
            print("Building entity summary index...")
            self.build_entity_summary_index()
            
        with torch.no_grad():
            # Encode query with instruction prefix
            query_embedding = self.embedding_model.encode(
                [query],
                instruction=self.query_prefix,
                max_length=self.max_length
            )
            
            # Normalize query embedding
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            # Ensure embeddings are on the same device
            if self.summary_embeddings.device != query_embedding.device:
                self.summary_embeddings = self.summary_embeddings.to(query_embedding.device)
            
            # Compute cosine similarity using matrix multiplication
            # Both embeddings are already normalized, so dot product = cosine similarity
            # Multiply by 100 to match the original nvembed-v2 scoring scale
            similarities = (query_embedding @ self.summary_embeddings.T).squeeze(0) * 100
            
            # Get top-k most similar documents
            top_k = min(top_k, len(similarities))
            scores, indices = torch.topk(similarities, k=top_k, largest=True)
            
            # Prepare results with rich metadata
            results = []
            for idx, score in zip(indices.tolist(), scores.tolist()):
                if idx < len(self.entity_summary_metadata):
                    doc_metadata = self.entity_summary_metadata[idx]
                    result = doc_metadata.copy()
                    result["match_score"] = float(score)
                    result["type"] = "entity"
                    results.append(result)
        
        return results
    
    def _build_search_index(self):
        """Build BM25 search index for entities (called lazily). Use build_index() for explicit building."""
        if self._index_built_entity_name:
            return  # Already built
            
        print("Building search index lazily...")
        self.build_entity_index()
        self.build_entity_summary_index()
    
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
    
    def search_gsw_bm25_entity_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across GSW entities using BM25 ranking with entity name.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching items sorted by BM25 relevance score
        """
        # Build index if not already built
        self.build_entity_index()
        
        if not self.bm25_entity_name or not self.entity_corpus:
            return []
            
        query_tokens = query.lower().split()
        scores = self.bm25_entity_name.get_scores(query_tokens)
        
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
    
    def search_gsw_bm25_entity_with_entity_features(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search across GSW entities using BM25 ranking with entity features.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching items sorted by BM25 relevance score
        """
        # Build index if not already built
        self.build_entity_summary_index()
        
        if not self.bm25_entity_summary or not self.entity_summary_corpus:
            return []
            
        query_tokens = query.lower().split()
        scores = self.bm25_entity_summary.get_scores(query_tokens)
        
        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches with positive scores
                result = self.entity_summary_metadata[idx].copy()
                result["match_score"] = float(scores[idx])
                result["type"] = "entity"
                results.append(result)
        
        return results
    
    def search_gsw_bm25_hybrid(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid search that combines results from both entity name and entity features search.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return from each method (total results <= limit*2)
            
        Returns:
            List of unique matching items sorted by BM25 relevance score, combining both search methods
        """
        # Get results from both search methods
        name_results = self.search_gsw_bm25_entity_name(query, limit)
        features_results = self.search_gsw_bm25_entity_with_entity_features(query, limit)
        
        # Create a dictionary to track unique entities and keep the best score
        unique_entities = {}
        
        # Process name search results
        for result in name_results:
            entity_key = result.get("global_id", "") + "::" + result.get("entity_id", "")
            if entity_key:
                result_copy = result.copy()
                result_copy["search_source"] = "entity_name"
                unique_entities[entity_key] = result_copy
        
        # Process features search results - merge or add new ones
        for result in features_results:
            entity_key = result.get("global_id", "") + "::" + result.get("entity_id", "")
            if entity_key:
                result_copy = result.copy()
                result_copy["search_source"] = "entity_features"
                
                if entity_key in unique_entities:
                    # Entity found in both searches - keep the higher score and mark as hybrid
                    existing = unique_entities[entity_key]
                    if result_copy["match_score"] > existing["match_score"]:
                        result_copy["search_source"] = "hybrid_features_better"
                        unique_entities[entity_key] = result_copy
                    else:
                        existing["search_source"] = "hybrid_name_better"
                else:
                    # New entity from features search
                    unique_entities[entity_key] = result_copy
        
        # Convert back to list and sort by score
        combined_results = list(unique_entities.values())
        # combined_results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return combined_results
    
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
    
    
    def get_multiple_relevant_entity_contexts(self, entity_ids: List[str]) -> Dict[str, Any]:
        """
        Get the context of multiple entities with the same name - all questions it participates in
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