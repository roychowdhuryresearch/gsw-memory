"""
Conversation summary aggregator for GSW structures.

This module implements an aggregator that extracts and formats existing
conversation summaries from GSW conversation nodes. Unlike other aggregators,
this one directly uses pre-existing summary fields rather than generating
new summaries via LLM.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_voyageai import VoyageAIEmbeddings

from ..models import GSWStructure
from .base import AggregatedView, BaseAggregator


class ConversationSummaryAggregator(BaseAggregator):
    """
    Aggregator that extracts existing conversation summaries from a GSW.

    This aggregator focuses on extracting and formatting conversation summaries
    that are already present in conversation nodes, without additional LLM processing.
    """

    def __init__(self, gsw: GSWStructure, llm_config: Optional[Dict] = None, embedding_model: str = "voyage-3"):
        """
        Initialize the conversation summary aggregator.

        Args:
            gsw: The GSW structure to aggregate
            llm_config: Configuration for LLM (not used, kept for interface consistency)
            embedding_model: Name of the embedding model for similarity matching
        """
        super().__init__(gsw)
        self.llm_config = llm_config  # Kept for interface consistency
        self._conversation_map = {
            conv.get("id", ""): conv for conv in gsw.conversation_nodes
        }
        self._entity_map = {entity.id: entity.name for entity in gsw.entity_nodes}
        self._precomputed_summaries = None  # Cache for precomputed summaries
        self.embedding_model = VoyageAIEmbeddings(model=embedding_model)

    def aggregate(self, query: str, **kwargs) -> AggregatedView:
        """
        Generate aggregated summaries for conversations relevant to the query using similarity ranking.

        Args:
            query: Query string to find similar conversations
            **kwargs: Additional parameters:
                - conversation_ids: Explicit list of conversation IDs to aggregate
                - max_conversations: Maximum number of conversations to return (default: 5)

        Returns:
            AggregatedView containing summaries for top-k most relevant conversations
        """
        max_conversations = kwargs.get("max_conversations", 5)
        
        # If explicit conversation IDs provided, use those
        conversation_ids = kwargs.get("conversation_ids")
        if conversation_ids:
            conversation_summaries = {}
            if self._precomputed_summaries:
                for cid in conversation_ids:
                    if cid in self._precomputed_summaries:
                        conversation_summaries[cid] = self._precomputed_summaries[cid]
            
            if not conversation_summaries:
                conversation_summaries = self._extract_summaries(conversation_ids)
        else:
            # Use similarity-based ranking to find most relevant conversations
            if not self._precomputed_summaries:
                # Need to precompute summaries first for similarity matching
                self._precomputed_summaries = self.precompute_summaries()
            
            ranked_conversations = self._rank_conversations_by_similarity(query, max_conversations)
            conversation_summaries = {cid: self._precomputed_summaries[cid] for cid in ranked_conversations}
            conversation_ids = ranked_conversations

        # Create aggregated view
        aggregated_view = AggregatedView(
            view_type="conversation_summary",
            content={
                "conversation_summaries": conversation_summaries,
                "conversation_ids": conversation_ids,
            },
            metadata={
                "query": query, 
                "num_conversations": len(conversation_summaries),
                "max_requested": max_conversations
            },
        )

        return aggregated_view

    def get_context(self, aggregated_view: AggregatedView) -> str:
        """
        Format an aggregated view into context string for downstream consumption.

        Args:
            aggregated_view: The aggregated view to format

        Returns:
            Formatted context string with all conversation summaries
        """
        content = aggregated_view.content
        conversation_summaries = content.get("conversation_summaries", {})

        if not conversation_summaries:
            return ""

        # Format multiple conversation summaries
        context_parts = []
        for conv_id, summary_data in conversation_summaries.items():
            participants = ", ".join(summary_data.get("participants", []))
            summary = summary_data.get("summary", "No summary available.")
            context_parts.append(f"**Conversation ({participants})**: {summary}")

        return "\n\n".join(context_parts)

    def precompute_summaries(
        self, conversation_ids: Optional[List[str]] = None,
        cache_file: Optional[str] = None, force_recompute: bool = False
    ) -> Dict[str, Dict]:
        """
        Pre-extract summaries for multiple conversations (static extraction).

        Args:
            conversation_ids: List of conversation IDs to process (None for all conversations)
            cache_file: Optional file path to save/load summaries cache
            force_recompute: If True, ignore existing cache and recompute

        Returns:
            Dictionary mapping conversation IDs to their summary data
        """
        # Try to load from file cache first (if not forcing recompute)
        if cache_file and not force_recompute and os.path.exists(cache_file):
            try:
                loaded_summaries = self.load_summaries_from_file(cache_file)
                if loaded_summaries:
                    self._precomputed_summaries = loaded_summaries
                    print(f"Loaded {len(loaded_summaries)} conversation summaries from cache file: {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to load cache file {cache_file}: {e}")

        # If we already have precomputed summaries, return those instead of regenerating
        if self._precomputed_summaries is not None and not force_recompute:
            if conversation_ids is None:
                return self._precomputed_summaries
            else:
                # Return subset for specific conversation IDs
                return {cid: self._precomputed_summaries[cid] 
                       for cid in conversation_ids 
                       if cid in self._precomputed_summaries}
        
        # Otherwise extract new summaries
        if conversation_ids is None:
            conversation_ids = [conv.get("id", "") for conv in self.gsw.conversation_nodes if conv.get("id")]

        # Extract and cache summaries
        extracted_summaries = self._extract_summaries(conversation_ids)
        
        # Cache the summaries for future use (only if extracting all conversations)
        all_conv_ids = [conv.get("id", "") for conv in self.gsw.conversation_nodes if conv.get("id")]
        if conversation_ids == all_conv_ids:
            self._precomputed_summaries = extracted_summaries
            
            # Save to file cache if specified
            if cache_file:
                try:
                    self.save_summaries_to_file(extracted_summaries, cache_file)
                    print(f"Saved {len(extracted_summaries)} conversation summaries to cache file: {cache_file}")
                except Exception as e:
                    print(f"Warning: Failed to save cache file {cache_file}: {e}")
        
        return extracted_summaries

    def _extract_conversations_from_query(self, query: str) -> List[str]:
        """
        Extract conversation IDs from query string using participant or topic matching.

        Args:
            query: Query string that may reference conversations

        Returns:
            List of conversation IDs found relevant to the query
        """
        query_lower = query.lower()
        found_conversations = []

        # Check each conversation for relevance
        for conv_id, conv_data in self._conversation_map.items():
            if not conv_id:
                continue
                
            # Check if query mentions participants
            participants = conv_data.get("participants", [])
            for participant_id in participants:
                if participant_id in self._entity_map:
                    participant_name = self._entity_map[participant_id].lower()
                    if participant_name in query_lower:
                        found_conversations.append(conv_id)
                        break

            # Check if query mentions general topics
            topics_general = conv_data.get("topics_general", [])
            for topic in topics_general:
                if isinstance(topic, str) and topic.lower() in query_lower:
                    found_conversations.append(conv_id)
                    break

            # Check if query mentions topic entities
            topics_entity = conv_data.get("topics_entity", [])
            for topic_entity_id in topics_entity:
                if topic_entity_id in self._entity_map:
                    topic_entity_name = self._entity_map[topic_entity_id].lower()
                    if topic_entity_name in query_lower:
                        found_conversations.append(conv_id)
                        break

        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_conversations))

    def _rank_conversations_by_similarity(self, query: str, max_conversations: int = 5) -> List[str]:
        """
        Rank conversations by similarity to query using embeddings.

        Args:
            query: Query string to match against
            max_conversations: Maximum number of conversations to return

        Returns:
            List of conversation IDs ranked by similarity
        """
        if not self._precomputed_summaries:
            return []

        # Prepare summaries for embedding
        conversation_data = []
        conversation_ids = []
        
        for conv_id, summary_data in self._precomputed_summaries.items():
            # Use participants, topics + summary as the text to embed
            participants = ", ".join(summary_data.get("participants", []))
            topics_general = ", ".join(summary_data.get("topics_general", []))
            topics_entity = ", ".join(summary_data.get("topics_entity", []))
            summary = summary_data.get("summary", "")
            
            combined_text = f"Participants: {participants}. Topics: {topics_general} {topics_entity}. Summary: {summary}"
            
            conversation_data.append(combined_text)
            conversation_ids.append(conv_id)

        if not conversation_data:
            return []

        try:
            # Get embeddings
            query_embedding = self.embedding_model.embed_query(query)
            document_embeddings = self.embedding_model.embed_documents(conversation_data)

            # Calculate similarities
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            document_embeddings_np = np.array(document_embeddings)

            similarities = cosine_similarity(query_embedding_np, document_embeddings_np).flatten()

            # Rank by similarity
            similarity_pairs = list(zip(conversation_ids, similarities))
            similarity_pairs.sort(key=lambda x: x[1], reverse=True)

            # Return top k conversation IDs
            return [conv_id for conv_id, _ in similarity_pairs[:max_conversations]]

        except Exception as e:
            print(f"Warning: Similarity ranking failed: {e}")
            # Fallback to first max_conversations
            return conversation_ids[:max_conversations]

    def _extract_summaries(self, conversation_ids: List[str]) -> Dict[str, Dict]:
        """
        Extract summaries for multiple conversations from existing conversation data.

        Args:
            conversation_ids: List of conversation IDs to extract summaries for

        Returns:
            Dictionary mapping conversation IDs to summary data
        """
        summary_map = {}

        for conv_id in conversation_ids:
            if conv_id not in self._conversation_map:
                print(f"Conversation {conv_id} not found, skipping")
                continue

            conv_data = self._conversation_map[conv_id]
            
            # Resolve participant names
            participant_names = []
            for participant_id in conv_data.get("participants", []):
                if participant_id in self._entity_map:
                    participant_names.append(self._entity_map[participant_id])
                else:
                    participant_names.append(participant_id)

            # Resolve topic entity names
            topic_entity_names = []
            for topic_entity_id in conv_data.get("topics_entity", []):
                if topic_entity_id in self._entity_map:
                    topic_entity_names.append(self._entity_map[topic_entity_id])
                else:
                    topic_entity_names.append(topic_entity_id)

            # Create summary data structure
            summary_map[conv_id] = {
                "conversation_id": conv_id,
                "participants": participant_names,
                "topics_entity": topic_entity_names,
                "topics_general": conv_data.get("topics_general", []),
                "summary": conv_data.get("summary", "No summary available."),
                "motivation": conv_data.get("motivation", ""),
                "chunk_id": conv_data.get("chunk_id", ""),
                "participant_summaries": self._resolve_participant_summaries(
                    conv_data.get("participant_summaries", {})
                ),
            }

        print(f"Extracted {len(summary_map)} conversation summaries")
        return summary_map

    def _resolve_participant_summaries(self, participant_summaries: Dict[str, str]) -> Dict[str, str]:
        """
        Resolve participant IDs to names in participant summaries.

        Args:
            participant_summaries: Dict mapping participant IDs to their summaries

        Returns:
            Dict mapping participant names to their summaries
        """
        resolved_summaries = {}
        
        for participant_id, summary in participant_summaries.items():
            # Skip special keys like "TEXT:me"
            if participant_id.startswith("TEXT:"):
                resolved_summaries[participant_id] = summary
            elif participant_id in self._entity_map:
                participant_name = self._entity_map[participant_id]
                resolved_summaries[participant_name] = summary
            else:
                # Use the ID as-is if we can't resolve it
                resolved_summaries[participant_id] = summary
                
        return resolved_summaries

    def save_summaries_to_file(self, summaries: Dict[str, Dict], file_path: str) -> None:
        """
        Save precomputed summaries to a JSON file.

        Args:
            summaries: Dictionary mapping conversation IDs to summary data
            file_path: Path to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Add metadata to the saved data
        cache_data = {
            "metadata": {
                "aggregator_type": "conversation_summary",
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
            Dictionary mapping conversation IDs to summary data, or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate the cache data structure
            if "summaries" not in cache_data:
                print(f"Warning: Invalid cache file format in {file_path}")
                return None
                
            metadata = cache_data.get("metadata", {})
            if metadata.get("aggregator_type") != "conversation_summary":
                print(f"Warning: Cache file {file_path} is not for conversation summaries")
                return None
                
            return cache_data["summaries"]
            
        except Exception as e:
            print(f"Error loading cache file {file_path}: {e}")
            return None 