"""
Verb phrase summary aggregator for GSW structures.

This module implements an aggregator that creates chronological summaries
of verb phrases within a GSW, focusing on semantic role patterns and usage
contexts across different chunks.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_voyageai import VoyageAIEmbeddings
from bespokelabs import curator

from ...prompts import VerbPhraseSummaryPrompts
from ..models import VerbPhraseNode, GSWStructure
from .base import AggregatedView, BaseAggregator


class GSWVerbPhraseSummarizer(curator.LLM):
    """Curator class for generating verb phrase summaries from GSW data."""

    return_completions_object = True
    require_all_responses = False  # Allow some requests to fail

    def prompt(self, input_data):
        """Create the prompt for the LLM summarizer."""
        system_prompt = VerbPhraseSummaryPrompts.get_system_prompt()
        user_prompt = VerbPhraseSummaryPrompts.get_user_prompt(
            verb_phrase=input_data["verb_phrase"],
            formatted_data=input_data["formatted_data"]
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data, response):
        """Parse the LLM response to extract the summary."""
        summary_text = response["choices"][0]["message"]["content"].strip()
        return {
            "verb_phrase_id": input_data["verb_phrase_id"],
            "verb_phrase": input_data["verb_phrase"],
            "summary": summary_text,
            "_raw_response": response,  # For debugging
        }


class VerbSummaryAggregator(BaseAggregator):
    """
    Aggregator that creates chronological summaries of verb phrases in a GSW.

    Supports both static batch processing and dynamic query-driven generation
    with efficient batching. Uses curator for LLM integration.
    """

    def __init__(self, gsw: GSWStructure, llm_config: Optional[Dict] = None, embedding_model: str = "voyage-3"):
        """
        Initialize the verb phrase summary aggregator.

        Args:
            gsw: The GSW structure to aggregate
            llm_config: Configuration for the LLM (model name, generation params)
            embedding_model: Name of the embedding model for similarity matching
        """
        super().__init__(gsw)
        self.llm_config = llm_config or {
            "model_name": "gpt-4o",
            "generation_params": {"temperature": 0.0, "max_tokens": 500},
        }
        self._verb_phrase_map = {vp.id: vp.phrase for vp in gsw.verb_phrase_nodes}
        self._entity_map = {entity.id: entity.name for entity in gsw.entity_nodes}
        self._precomputed_summaries = None  # Cache for precomputed summaries
        self.embedding_model = VoyageAIEmbeddings(model=embedding_model)

    def aggregate(self, query: str, **kwargs) -> AggregatedView:
        """
        Generate aggregated summaries for verb phrases relevant to the query using similarity ranking.

        Args:
            query: Query string to find similar verb phrases
            **kwargs: Additional parameters:
                - verb_phrase_ids: Explicit list of verb phrase IDs to aggregate
                - max_verb_phrases: Maximum number of verb phrases to return (default: 5)

        Returns:
            AggregatedView containing summaries for top-k most relevant verb phrases
        """
        max_verb_phrases = kwargs.get("max_verb_phrases", 5)
        
        # If explicit verb phrase IDs provided, use those
        verb_phrase_ids = kwargs.get("verb_phrase_ids")
        if verb_phrase_ids:
            verb_phrase_summaries = {}
            if self._precomputed_summaries:
                for vid in verb_phrase_ids:
                    if vid in self._precomputed_summaries:
                        verb_phrase_summaries[vid] = self._precomputed_summaries[vid]
            
            if not verb_phrase_summaries:
                verb_phrase_summaries = self._generate_summaries(verb_phrase_ids)
        else:
            # Use similarity-based ranking to find most relevant verb phrases
            if not self._precomputed_summaries:
                # Need to precompute summaries first for similarity matching
                self._precomputed_summaries = self.precompute_summaries()
            
            ranked_verb_phrases = self._rank_verb_phrases_by_similarity(query, max_verb_phrases)
            verb_phrase_summaries = {vid: self._precomputed_summaries[vid] for vid in ranked_verb_phrases}
            verb_phrase_ids = ranked_verb_phrases

        # Create aggregated view
        aggregated_view = AggregatedView(
            view_type="verb_phrase_summary",
            content={
                "verb_phrase_summaries": verb_phrase_summaries,
                "verb_phrase_ids": verb_phrase_ids,
            },
            metadata={
                "query": query, 
                "num_verb_phrases": len(verb_phrase_summaries),
                "max_requested": max_verb_phrases
            },
        )

        return aggregated_view

    def get_context(self, aggregated_view: AggregatedView) -> str:
        """
        Format an aggregated view into context string for downstream consumption.

        Args:
            aggregated_view: The aggregated view to format

        Returns:
            Formatted context string with all verb phrase summaries
        """
        content = aggregated_view.content
        verb_phrase_summaries = content.get("verb_phrase_summaries", {})

        if not verb_phrase_summaries:
            return ""

        # Format multiple verb phrase summaries
        context_parts = []
        for vp_id, summary_data in verb_phrase_summaries.items():
            verb_phrase = summary_data["verb_phrase"]
            summary = summary_data["summary"]
            context_parts.append(f"**{verb_phrase}**: {summary}")

        return "\n\n".join(context_parts)

    def precompute_summaries(
        self, verb_phrase_ids: Optional[List[str]] = None,
        cache_file: Optional[str] = None, force_recompute: bool = False
    ) -> Dict[str, Dict]:
        """
        Pre-compute summaries for multiple verb phrases (static generation).

        Args:
            verb_phrase_ids: List of verb phrase IDs to process (None for all verb phrases)
            cache_file: Optional file path to save/load summaries cache
            force_recompute: If True, ignore existing cache and recompute

        Returns:
            Dictionary mapping verb phrase IDs to their summary data
        """
        # Try to load from file cache first (if not forcing recompute)
        if cache_file and not force_recompute and os.path.exists(cache_file):
            try:
                loaded_summaries = self.load_summaries_from_file(cache_file)
                if loaded_summaries:
                    self._precomputed_summaries = loaded_summaries
                    print(f"Loaded {len(loaded_summaries)} verb phrase summaries from cache file: {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to load cache file {cache_file}: {e}")

        # If we already have precomputed summaries, return those instead of regenerating
        if self._precomputed_summaries is not None and not force_recompute:
            if verb_phrase_ids is None:
                return self._precomputed_summaries
            else:
                # Return subset for specific verb phrase IDs
                return {vid: self._precomputed_summaries[vid] 
                       for vid in verb_phrase_ids 
                       if vid in self._precomputed_summaries}
        
        # Otherwise generate new summaries
        if verb_phrase_ids is None:
            verb_phrase_ids = [vp.id for vp in self.gsw.verb_phrase_nodes]

        # Generate and cache summaries
        generated_summaries = self._generate_summaries(verb_phrase_ids)
        
        # Cache the summaries for future use (only if generating all verb phrases)
        all_vp_ids = [vp.id for vp in self.gsw.verb_phrase_nodes]
        if verb_phrase_ids == all_vp_ids:
            self._precomputed_summaries = generated_summaries
            
            # Save to file cache if specified
            if cache_file:
                try:
                    self.save_summaries_to_file(generated_summaries, cache_file)
                    print(f"Saved {len(generated_summaries)} verb phrase summaries to cache file: {cache_file}")
                except Exception as e:
                    print(f"Warning: Failed to save cache file {cache_file}: {e}")
        
        return generated_summaries

    def _extract_verb_phrases_from_query(self, query: str) -> List[str]:
        """
        Extract verb phrase IDs from query string using phrase matching.

        Args:
            query: Query string that may contain verb phrases

        Returns:
            List of verb phrase IDs found in the query
        """
        query_lower = query.lower()
        found_verb_phrases = []

        # Try exact phrase matching first
        for vp_id, verb_phrase in self._verb_phrase_map.items():
            if verb_phrase.lower() in query_lower:
                found_verb_phrases.append(vp_id)

        # If no exact matches, try partial matching for multi-word phrases
        if not found_verb_phrases:
            for vp_id, verb_phrase in self._verb_phrase_map.items():
                phrase_words = verb_phrase.lower().split()
                if len(phrase_words) > 1:
                    # Check if all words appear in query
                    if all(word in query_lower for word in phrase_words):
                        found_verb_phrases.append(vp_id)

        return found_verb_phrases

    def _rank_verb_phrases_by_similarity(self, query: str, max_verb_phrases: int = 5) -> List[str]:
        """
        Rank verb phrases by similarity to query using embeddings.

        Args:
            query: Query string to match against
            max_verb_phrases: Maximum number of verb phrases to return

        Returns:
            List of verb phrase IDs ranked by similarity
        """
        if not self._precomputed_summaries:
            return []

        # Prepare summaries for embedding
        verb_phrase_data = []
        verb_phrase_ids = []
        
        for vp_id, summary_data in self._precomputed_summaries.items():
            # Use verb phrase + summary as the text to embed
            verb_phrase = summary_data.get("verb_phrase", "")
            summary = summary_data.get("summary", "")
            combined_text = f"{verb_phrase}: {summary}"
            
            verb_phrase_data.append(combined_text)
            verb_phrase_ids.append(vp_id)

        if not verb_phrase_data:
            return []

        try:
            # Get embeddings
            query_embedding = self.embedding_model.embed_query(query)
            document_embeddings = self.embedding_model.embed_documents(verb_phrase_data)

            # Calculate similarities
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            document_embeddings_np = np.array(document_embeddings)

            similarities = cosine_similarity(query_embedding_np, document_embeddings_np).flatten()

            # Rank by similarity
            similarity_pairs = list(zip(verb_phrase_ids, similarities))
            similarity_pairs.sort(key=lambda x: x[1], reverse=True)

            # Return top k verb phrase IDs
            return [vp_id for vp_id, _ in similarity_pairs[:max_verb_phrases]]

        except Exception as e:
            print(f"Warning: Similarity ranking failed: {e}")
            # Fallback to first max_verb_phrases
            return verb_phrase_ids[:max_verb_phrases]

    def _generate_summaries(self, verb_phrase_ids: List[str]) -> Dict[str, Dict]:
        """
        Generate summaries for multiple verb phrases using curator batching.

        Args:
            verb_phrase_ids: List of verb phrase IDs to generate summaries for

        Returns:
            Dictionary mapping verb phrase IDs to summary data
        """
        # Prepare inputs for curator
        summarizer_inputs = []

        for vp_id in verb_phrase_ids:
            verb_phrase = self._get_verb_phrase_by_id(vp_id)
            if not verb_phrase:
                print(f"Verb phrase {vp_id} not found, skipping")
                continue

            # Aggregate chronological semantic role data
            chronological_data = self._aggregate_verb_phrase_data(verb_phrase)

            # Skip verb phrases with no meaningful data
            if not chronological_data:
                print(f"No data found for verb phrase {vp_id} ({verb_phrase.phrase}), skipping")
                continue

            # Format data for prompt
            formatted_data = self._format_data_for_prompt(
                verb_phrase.phrase, vp_id, chronological_data
            )

            summarizer_inputs.append(
                {
                    "verb_phrase_id": vp_id,
                    "verb_phrase": verb_phrase.phrase,
                    "formatted_data": formatted_data,
                }
            )

        if not summarizer_inputs:
            print("No verb phrases found suitable for summarization")
            return {}

        # Initialize and run curator summarizer
        summarizer = GSWVerbPhraseSummarizer(
            model_name=self.llm_config["model_name"],
            generation_params=dict(self.llm_config["generation_params"]),
        )

        print(
            f"Generating summaries for {len(summarizer_inputs)} verb phrases using {self.llm_config['model_name']}..."
        )

        # Batch process all verb phrases
        summarization_results = summarizer(summarizer_inputs)

        # Convert results to dictionary format
        summary_map = {}
        failed_verb_phrases = []
        
        for result in summarization_results.dataset:
            vp_id = result["verb_phrase_id"]
            
            # Check if we got a valid summary
            if "summary" in result and result["summary"]:
                summary_map[vp_id] = {
                    "verb_phrase": result["verb_phrase"],
                    "summary": result["summary"],
                    "verb_phrase_id": vp_id,
                }
            else:
                failed_verb_phrases.append(vp_id)
                # Create a fallback summary for failed verb phrases
                verb_phrase = result.get("verb_phrase", vp_id)
                summary_map[vp_id] = {
                    "verb_phrase": verb_phrase,
                    "summary": f"Verb phrase: {verb_phrase}. Summary generation failed.",
                    "verb_phrase_id": vp_id,
                }

        if failed_verb_phrases:
            print(f"Warning: Failed to generate summaries for {len(failed_verb_phrases)} verb phrases")
            
        print(f"Generated {len(summary_map)} summaries ({len(summary_map) - len(failed_verb_phrases)} successful)")
        return summary_map

    def _get_verb_phrase_by_id(self, vp_id: str) -> Optional[VerbPhraseNode]:
        """Get verb phrase node by ID."""
        for vp in self.gsw.verb_phrase_nodes:
            if vp.id == vp_id:
                return vp
        return None

    def _aggregate_verb_phrase_data(self, verb_phrase: VerbPhraseNode) -> Dict[str, Any]:
        """
        Aggregate chronological semantic role data for a single verb phrase.

        Args:
            verb_phrase: The verb phrase node to aggregate data for

        Returns:
            Dictionary mapping chunk IDs to semantic role information
        """
        chronological_data = defaultdict(
            lambda: {"questions_answers": [], "participating_entities": set()}
        )

        # Process each question in the verb phrase
        for question in verb_phrase.questions:
            if not question.chunk_id:
                continue
                
            chunk_id = question.chunk_id
            
            # Extract answers and resolve entity names
            resolved_answers = []
            for answer in question.answers:
                if answer == "None" or not answer:
                    resolved_answers.append("None")
                elif answer.startswith("TEXT:"):
                    resolved_answers.append(answer[5:].strip())
                elif answer in self._entity_map:
                    entity_name = self._entity_map[answer]
                    resolved_answers.append(entity_name)
                    chronological_data[chunk_id]["participating_entities"].add(entity_name)
                else:
                    resolved_answers.append(answer)

            # Store question-answer pair
            qa_info = {
                "question": question.text,
                "answers": resolved_answers,
                "question_id": question.id
            }
            chronological_data[chunk_id]["questions_answers"].append(qa_info)

        # Convert sets to lists for JSON serialization
        for chunk_data in chronological_data.values():
            chunk_data["participating_entities"] = list(chunk_data["participating_entities"])

        # Sort by chunk_id
        sorted_chunk_ids = sorted(chronological_data.keys(), key=self._sort_chunk_key)
        sorted_data = {
            chunk_id: chronological_data[chunk_id] for chunk_id in sorted_chunk_ids
        }

        return sorted_data

    def _sort_chunk_key(self, chunk_id_str: str) -> List[int]:
        """
        Robust key function for sorting chunk IDs like 'chunk_X_Y'.
        
        Ported from entity_summary.py's sort_key function.
        """
        if not isinstance(chunk_id_str, str):
            return [-1]

        # Clean up potential duplicate prefix
        cleaned_id = chunk_id_str
        if cleaned_id.startswith("chunk_chunk_"):
            cleaned_id = cleaned_id.replace("chunk_chunk_", "chunk_", 1)

        # Proceed with splitting and conversion
        if "_" not in cleaned_id:
            return [-1]

        parts = cleaned_id.split("_")
        numeric_parts = []
        # Iterate through parts after the first underscore
        for part in parts[1:]:
            try:
                numeric_parts.append(int(part))
            except ValueError:
                pass

        return numeric_parts if numeric_parts else [-1]

    def _format_data_for_prompt(
        self,
        verb_phrase: str,
        vp_id: str,
        chronological_data: Dict[str, Any],
    ) -> str:
        """
        Format chronological semantic role data into a structured timeline string.

        Args:
            verb_phrase: The verb phrase text
            vp_id: The verb phrase ID
            chronological_data: Chronological data organized by chunk

        Returns:
            Formatted string for LLM consumption
        """
        prompt_text = [
            f"VERB PHRASE: {verb_phrase}",
            f"VERB PHRASE ID: {vp_id}",
            "",
            "EVENT INFORMATION (by Chunk ID):",
            "",
        ]

        for chunk_id, data in chronological_data.items():
            prompt_text.append(f"--- Chunk: {chunk_id} ---")

            # Add participating entities
            if data.get("participating_entities"):
                entities_list = ", ".join(data["participating_entities"])
                prompt_text.append(f"PARTICIPATING ENTITIES: {entities_list}")

            # Add question-answer patterns
            if data.get("questions_answers"):
                prompt_text.append("EVENT DETAILS:")
                for qa in data["questions_answers"]:
                    question = qa["question"]
                    answers = qa["answers"]
                    answers_str = ", ".join(answers) if answers else "None"
                    prompt_text.append(f"  - {question} → {answers_str}")

            prompt_text.append("")  # Add space between chunks

        return "\n".join(prompt_text)

    def save_summaries_to_file(self, summaries: Dict[str, Dict], file_path: str) -> None:
        """
        Save precomputed summaries to a JSON file.

        Args:
            summaries: Dictionary mapping verb phrase IDs to summary data
            file_path: Path to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Add metadata to the saved data
        cache_data = {
            "metadata": {
                "aggregator_type": "verb_summary",
                "total_summaries": len(summaries),
                "llm_config": self.llm_config
            },
            "summaries": summaries
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    def load_summaries_from_file(self, file_path: str) -> Optional[Dict[str, Dict]]:
        """
        Load precomputed summaries from a JSON file.

        Args:
            file_path: Path to load the file from

        Returns:
            Dictionary mapping verb phrase IDs to summary data, or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate the cache data structure
            if "summaries" not in cache_data:
                print(f"Warning: Invalid cache file format in {file_path}")
                return None
                
            metadata = cache_data.get("metadata", {})
            if metadata.get("aggregator_type") != "verb_summary":
                print(f"Warning: Cache file {file_path} is not for verb phrase summaries")
                return None
                
            return cache_data["summaries"]
            
        except Exception as e:
            print(f"Error loading cache file {file_path}: {e}")
            return None 