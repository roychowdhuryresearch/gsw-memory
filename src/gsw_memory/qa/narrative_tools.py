"""
Tools for the narrative Q&A system.

This module contains all tool implementations and building logic
used by the GSWQuestionAnswerer_Narrative class.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import EntityNode
from .entity_extractor import QuestionEntityExtractor
from .matcher import EntityMatcher 
from .reranker import SummaryReranker
from ..prompts.narrative_qa_prompts import NarrativeQAPrompts


class NarrativeQATools:
    """Tool implementations for the narrative Q&A agent."""
    
    def __init__(
        self, 
        gsws: List,
        entity_aggregators: List,
        verb_aggregators: List,
        conversation_aggregators: List,
        entity_extractor: QuestionEntityExtractor,
        entity_matcher: EntityMatcher,
        summary_reranker: SummaryReranker,
        chunks_folder_path: Optional[str] = None,
        include_verb_phrases: bool = False,
        include_conversations: bool = False,
        tool_result_limit: int = 5,
        context_retrieval_size: int = 20,
        verbose: bool = True,
    ):
        """
        Initialize narrative Q&A tools.
        
        Args:
            gsws: List of GSW structures
            entity_aggregators: List of entity summary aggregators
            verb_aggregators: List of verb phrase aggregators  
            conversation_aggregators: List of conversation aggregators
            entity_extractor: Entity extractor for processing queries
            entity_matcher: Entity matcher for finding entities
            summary_reranker: Summary reranker for ranking results
            chunks_folder_path: Path to folder containing chunk text files
            include_verb_phrases: Whether VP tools are available
            include_conversations: Whether conversation tools are available
            tool_result_limit: Default number of results to return
            context_retrieval_size: Number of summaries to retrieve before reranking
            verbose: Whether to show verbose output
        """
        self.gsws = gsws
        self.entity_aggregators = entity_aggregators
        self.verb_aggregators = verb_aggregators
        self.conversation_aggregators = conversation_aggregators
        self.entity_extractor = entity_extractor
        self.entity_matcher = entity_matcher
        self.summary_reranker = summary_reranker
        self.chunks_folder_path = chunks_folder_path
        self.include_verb_phrases = include_verb_phrases
        self.include_conversations = include_conversations
        self.tool_result_limit = tool_result_limit
        self.context_retrieval_size = context_retrieval_size
        self.verbose = verbose
        
        # Caches
        self._entity_summaries_cache: Optional[Dict[str, Dict[str, str]]] = None
        self._verb_summaries_cache: Optional[Dict[str, Dict[str, str]]] = None
        self._conversation_summaries_cache: Optional[Dict[str, Dict[str, str]]] = None
        self._chunk_text_cache: Optional[Dict[str, str]] = None
        
        # Activity logging callback
        self.log_activity_callback: Optional[callable] = None

    def set_activity_logger(self, callback: callable) -> None:
        """Set callback for activity logging."""
        self.log_activity_callback = callback
    
    def _log_activity(self, tool: str, reasoning: str, args: Dict[str, Any], result: Any) -> None:
        """Log activity if callback is set."""
        if self.log_activity_callback:
            self.log_activity_callback(tool, reasoning, args, result)

    def build_agent_tools(self) -> List[Dict[str, Any]]:
        """Build the list of tools available to the agent."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_entity_by_name",
                    "description": "Search for entities by name/description and get their summaries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Entity name or description to search for"},
                            "k": {"type": "integer", "description": "Number of results to return", "default": self.tool_result_limit},
                            "reasoning": {"type": "string", "description": "Why you are calling this tool"}
                        },
                        "required": ["query", "reasoning"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "search_entity_summaries",
                    "description": "Search entity summaries directly by semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query to search in entity summaries"},
                            "k": {"type": "integer", "description": "Number of results to return", "default": self.tool_result_limit},
                            "reasoning": {"type": "string", "description": "Why you are calling this tool"}
                        },
                        "required": ["query", "reasoning"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_chunks",
                    "description": "Retrieve original text chunks by their IDs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "List of chunk IDs to retrieve"},
                            "reasoning": {"type": "string", "description": "Why you are calling this tool"}
                        },
                        "required": ["chunk_ids", "reasoning"]
                    }
                }
            }
        ]

        # Add verb phrase search if enabled
        if self.include_verb_phrases:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_vp_summaries",
                    "description": "Search verb phrase summaries by semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query to search in verb phrase summaries"},
                            "k": {"type": "integer", "description": "Number of results to return", "default": self.tool_result_limit},
                            "reasoning": {"type": "string", "description": "Why you are calling this tool"}
                        },
                        "required": ["query", "reasoning"]
                    }
                }
            })

        # Add conversation search if enabled
        if self.include_conversations:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_conversation_summaries",
                    "description": "Search conversation summaries by semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query to search in conversation summaries"},
                            "k": {"type": "integer", "description": "Number of results to return", "default": self.tool_result_limit},
                            "reasoning": {"type": "string", "description": "Why you are calling this tool"}
                        },
                        "required": ["query", "reasoning"]
                    }
                }
            })
            
            # Add detailed conversation lookup tool
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_detailed_conversation",
                    "description": "Get detailed conversation information by conversation ID, including resolved participant names, topics, and participant summaries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "conversation_id": {"type": "string", "description": "The conversation ID to get details for"},
                            "reasoning": {"type": "string", "description": "Why you are calling this tool"}
                        },
                        "required": ["conversation_id", "reasoning"]
                    }
                }
            })

        return tools

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a tool and return results."""
        reasoning = tool_args.get("reasoning", "No reasoning provided")
        
        if tool_name == "search_entity_by_name":
            return self.search_entity_by_name(tool_args["query"], tool_args.get("k", self.tool_result_limit), reasoning)
        elif tool_name == "search_entity_summaries":
            return self.search_entity_summaries(tool_args["query"], tool_args.get("k", self.tool_result_limit), reasoning)
        elif tool_name == "search_vp_summaries":
            return self.search_vp_summaries(tool_args["query"], tool_args.get("k", self.tool_result_limit), reasoning)
        elif tool_name == "search_conversation_summaries":
            return self.search_conversation_summaries(tool_args["query"], tool_args.get("k", self.tool_result_limit), reasoning)
        elif tool_name == "get_detailed_conversation":
            return self.get_detailed_conversation(tool_args["conversation_id"], reasoning)
        elif tool_name == "search_chunks":
            return self.search_chunks(tool_args["chunk_ids"], reasoning)
        else:
            self._log_activity("unknown_tool", f"Unknown tool: {tool_name}", {"tool_name": tool_name, "args": tool_args}, None)
            return []

    # Tool implementation methods
    def search_entity_by_name(self, query: str, k: int, reasoning: str) -> List[Dict[str, Any]]:
        """Tool: Search entities by name and return their summaries."""
        self._log_activity("search_entity_by_name", reasoning, {"query": query, "k": k}, None)
        
        # Extract entities from query
        extracted_entities = self.entity_extractor.extract_entities(query)
        if not extracted_entities:
            # Fallback: use the query itself as entity name
            extracted_entities = [query]
        
        # Find matching entities
        entity_matches_with_source = self._find_matching_entities(extracted_entities)
        entity_summaries = self._get_entity_summaries(entity_matches_with_source)
        
        # Get more results then rerank to final size
        initial_k = min(len(entity_summaries), self.context_retrieval_size)
        ranked_summaries = self.summary_reranker.rerank_summaries(entity_summaries, query, initial_k)
        final_summaries = ranked_summaries[:k]  # Take top k after reranking
        
        # Format results
        results = []
        for entity_name, summary, score in final_summaries:
            entity_id = self._find_entity_id_for_summary(entity_name, summary)
            chunk_ids = self._extract_chunk_ids_from_id(entity_id)
            
            results.append({
                "entity_name": entity_name,
                "entity_id": entity_id,
                "summary": summary,
                "score": score,
                "chunk_ids": chunk_ids
            })
        
        self._log_activity("search_entity_by_name", "Found entities by name", {"query": query, "count": len(results)}, results)
        return results

    def search_entity_summaries(self, query: str, k: int, reasoning: str) -> List[Dict[str, Any]]:
        """Tool: Search entity summaries directly by similarity."""
        self._log_activity("search_entity_summaries", reasoning, {"query": query, "k": k}, None)
        
        # Get all entity summaries
        all_summaries = []
        entity_summaries_cache = self._get_entity_summaries_cache()
        
        for cache_key, summary_data in entity_summaries_cache.items():
            gsw_index, entity_id = cache_key.split("_", 1)
            entity_name = summary_data.get("entity_name", "Unknown")
            summary = summary_data["summary"]
            
            all_summaries.append((entity_name, summary))
        
        # Get more results then rerank to final size
        initial_k = min(len(all_summaries), self.context_retrieval_size)
        ranked_summaries = self.summary_reranker.rerank_summaries(all_summaries, query, initial_k)
        final_summaries = ranked_summaries[:k]  # Take top k after reranking
        
        # Format results
        results = []
        for entity_name, summary, score in final_summaries:
            entity_id = self._find_entity_id_for_summary(entity_name, summary)
            chunk_ids = self._extract_chunk_ids_from_id(entity_id)
            
            results.append({
                "entity_name": entity_name,
                "entity_id": entity_id,
                "summary": summary,
                "score": score,
                "chunk_ids": chunk_ids
            })
        
        self._log_activity("search_entity_summaries", "Found entities by summary similarity", {"query": query, "count": len(results)}, results)
        return results

    def search_vp_summaries(self, query: str, k: int, reasoning: str) -> List[Dict[str, Any]]:
        """Tool: Search verb phrase summaries by similarity."""
        if not self.include_verb_phrases:
            return []
            
        self._log_activity("search_vp_summaries", reasoning, {"query": query, "k": k}, None)
        
        # Get all VP summaries
        all_summaries = []
        vp_summaries_cache = self._get_vp_summaries_cache()
        
        for cache_key, summary_data in vp_summaries_cache.items():
            gsw_index, vp_id = cache_key.split("_", 1)
            summary = summary_data["summary"]
            all_summaries.append((vp_id, summary))
        
        # Get more results then rerank to final size
        initial_k = min(len(all_summaries), self.context_retrieval_size)
        ranked_summaries = self.summary_reranker.rerank_summaries(all_summaries, query, initial_k)
        final_summaries = ranked_summaries[:k]  # Take top k after reranking
        
        # Format results
        results = []
        for vp_id, summary, score in final_summaries:
            chunk_ids = self._extract_chunk_ids_from_id(vp_id)
            
            results.append({
                "vp_id": vp_id,
                "summary": summary,
                "score": score,
                "chunk_ids": chunk_ids
            })
        
        self._log_activity("search_vp_summaries", "Found VPs by summary similarity", {"query": query, "count": len(results)}, results)
        return results

    def search_conversation_summaries(self, query: str, k: int, reasoning: str) -> List[Dict[str, Any]]:
        """Tool: Search conversation summaries by similarity."""
        if not self.include_conversations:
            return []
            
        self._log_activity("search_conversation_summaries", reasoning, {"query": query, "k": k}, None)
        
        # Get all conversation summaries
        all_summaries = []
        conv_summaries_cache = self._get_conversation_summaries_cache()
        
        for cache_key, summary_data in conv_summaries_cache.items():
            gsw_index, conv_id = cache_key.split("_", 1)
            summary = summary_data["summary"]
            all_summaries.append((conv_id, summary))
        
        # Get more results then rerank to final size
        initial_k = min(len(all_summaries), self.context_retrieval_size)
        ranked_summaries = self.summary_reranker.rerank_summaries(all_summaries, query, initial_k)
        final_summaries = ranked_summaries[:k]  # Take top k after reranking
        
        # Format results
        results = []
        for conv_id, summary, score in final_summaries:
            chunk_ids = self._extract_chunk_ids_from_id(conv_id)
            
            results.append({
                "conversation_id": conv_id,
                "summary": summary,
                "score": score,
                "chunk_ids": chunk_ids
            })
        
        self._log_activity("search_conversation_summaries", "Found conversations by summary similarity", {"query": query, "count": len(results)}, results)
        return results

    def search_chunks(self, chunk_ids: List[str], reasoning: str) -> List[Dict[str, Any]]:
        """Tool: Retrieve original text chunks by IDs."""
        self._log_activity("search_chunks", reasoning, {"chunk_ids": chunk_ids}, None)
        
        chunk_text_cache = self._get_chunk_text_cache()
        
        results = []
        for chunk_id in chunk_ids:
            text = chunk_text_cache.get(chunk_id, "")
            results.append({
                "chunk_id": chunk_id,
                "text": text
            })
        
        self._log_activity("search_chunks", "Retrieved chunk texts", {"count": len(results)}, results)
        return results

    def get_detailed_conversation(self, conversation_id: str, reasoning: str) -> List[Dict[str, Any]]:
        """Tool: Get detailed conversation information with resolved entity references."""
        if not self.include_conversations:
            return []
            
        self._log_activity("get_detailed_conversation", reasoning, {"conversation_id": conversation_id}, None)
        
        # Find the conversation across all GSWs
        conversation_data = None
        source_gsw_index = None
        
        for gsw_index, gsw in enumerate(self.gsws):
            for conv_node in gsw.conversation_nodes:
                if conv_node.get("id") == conversation_id:
                    conversation_data = conv_node
                    source_gsw_index = gsw_index
                    break
            if conversation_data:
                break
        
        if not conversation_data:
            self._log_activity("get_detailed_conversation", f"Conversation {conversation_id} not found", {"conversation_id": conversation_id}, [])
            return []
        
        # Get entity mapping from the source GSW for resolving IDs
        entity_map = {entity.id: entity for entity in self.gsws[source_gsw_index].entity_nodes}
        
        # Resolve participant IDs to entity information
        resolved_participants = []
        for participant_id in conversation_data.get("participants", []):
            if participant_id in entity_map:
                entity = entity_map[participant_id]
                resolved_participants.append({
                    "entity_id": participant_id,
                    "entity_name": entity.name,
                    "entity_type": getattr(entity, 'type', 'unknown')
                })
            else:
                resolved_participants.append({
                    "entity_id": participant_id,
                    "entity_name": participant_id,  # fallback to ID
                    "entity_type": "unresolved"
                })
        
        # Resolve topic entity IDs
        resolved_topic_entities = []
        for topic_entity_id in conversation_data.get("topics_entity", []):
            if topic_entity_id in entity_map:
                entity = entity_map[topic_entity_id]
                resolved_topic_entities.append({
                    "entity_id": topic_entity_id,
                    "entity_name": entity.name,
                    "entity_type": getattr(entity, 'type', 'unknown')
                })
            else:
                resolved_topic_entities.append({
                    "entity_id": topic_entity_id,
                    "entity_name": topic_entity_id,  # fallback to ID
                    "entity_type": "unresolved"
                })
        
        # Resolve participant summaries (keys are entity IDs)
        resolved_participant_summaries = {}
        for participant_id, summary in conversation_data.get("participant_summaries", {}).items():
            if participant_id.startswith("TEXT:"):
                # Keep special TEXT: keys as-is
                resolved_participant_summaries[participant_id] = summary
            elif participant_id in entity_map:
                entity_name = entity_map[participant_id].name
                resolved_participant_summaries[f"{entity_name} ({participant_id})"] = summary
            else:
                # Use ID if can't resolve
                resolved_participant_summaries[participant_id] = summary
        
        # Get location and time information if available
        location_info = None
        if conversation_data.get("location_id"):
            # Try to find location node
            for space_node in self.gsws[source_gsw_index].space_nodes:
                if space_node.get("id") == conversation_data["location_id"]:
                    location_info = {
                        "location_id": space_node.get("id"),
                        "location_name": space_node.get("name", "Unknown location"),
                        "location_type": space_node.get("type", "unknown")
                    }
                    break
        
        time_info = None
        if conversation_data.get("time_id"):
            # Try to find time node
            for time_node in self.gsws[source_gsw_index].time_nodes:
                if time_node.get("id") == conversation_data["time_id"]:
                    time_info = {
                        "time_id": time_node.get("id"),
                        "time_name": time_node.get("name", "Unknown time"),
                        "time_type": time_node.get("type", "unknown")
                    }
                    break
        
        # Extract chunk IDs for reference
        chunk_ids = self._extract_chunk_ids_from_id(conversation_id)
        if not chunk_ids and conversation_data.get("chunk_id"):
            chunk_ids = [conversation_data["chunk_id"]]
        
        # Build detailed conversation result
        detailed_conversation = {
            "conversation_id": conversation_id,
            "chunk_id": conversation_data.get("chunk_id", ""),
            "chunk_ids": chunk_ids,
            "summary": conversation_data.get("summary", "No summary available"),
            "motivation": conversation_data.get("motivation", ""),
            "participants": resolved_participants,
            "topics_general": conversation_data.get("topics_general", []),
            "topics_entity": resolved_topic_entities,
            "participant_summaries": resolved_participant_summaries,
            "location": location_info,
            "time": time_info,
            "raw_conversation_data": conversation_data  # Include raw data for debugging
        }
        
        result = [detailed_conversation]
        self._log_activity("get_detailed_conversation", f"Retrieved detailed conversation {conversation_id}", {"conversation_id": conversation_id, "participants_count": len(resolved_participants)}, result)
        return result

    # Helper methods for initial context building
    def get_initial_vp_summaries(self, question: str, initial_context_size: int) -> List[Tuple[str, str, float]]:
        """Get initial verb phrase summaries ranked by relevance to question."""
        if not self.include_verb_phrases:
            return []
            
        all_summaries = []
        vp_summaries_cache = self._get_vp_summaries_cache()
        
        for cache_key, summary_data in vp_summaries_cache.items():
            gsw_index, vp_id = cache_key.split("_", 1)
            summary = summary_data["summary"]
            all_summaries.append((vp_id, summary))
        
        # Get more results then rerank to final size
        initial_k = min(len(all_summaries), self.context_retrieval_size)
        ranked_summaries = self.summary_reranker.rerank_summaries(all_summaries, question, initial_k)
        return ranked_summaries[:initial_context_size]  # Take top results for initial context

    def get_initial_conversation_summaries(self, question: str, initial_context_size: int) -> List[Tuple[str, str, float]]:
        """Get initial conversation summaries ranked by relevance to question."""
        if not self.include_conversations:
            return []
            
        all_summaries = []
        conv_summaries_cache = self._get_conversation_summaries_cache()
        
        for cache_key, summary_data in conv_summaries_cache.items():
            gsw_index, conv_id = cache_key.split("_", 1)
            summary = summary_data["summary"]
            all_summaries.append((conv_id, summary))
        
        # Get more results then rerank to final size
        initial_k = min(len(all_summaries), self.context_retrieval_size)
        ranked_summaries = self.summary_reranker.rerank_summaries(all_summaries, question, initial_k)
        return ranked_summaries[:initial_context_size]  # Take top results for initial context

    def find_matching_entities(self, entity_names: List[str]) -> List[Tuple[EntityNode, int]]:
        """Find matching entities across all GSWs."""
        return self._find_matching_entities(entity_names)

    def get_entity_summaries(self, entities_with_source: List[Tuple[EntityNode, int]]) -> List[Tuple[str, str]]:
        """Get summaries for matched entities from appropriate aggregators."""
        return self._get_entity_summaries(entities_with_source)

    # Cache and helper methods
    def _get_entity_summaries_cache(self) -> Dict[str, Dict[str, str]]:
        """Get or build entity summaries cache."""
        if self._entity_summaries_cache is None:
            self._entity_summaries_cache = {}
            for gsw_index, aggregator in enumerate(self.entity_aggregators):
                aggregator_summaries = aggregator.precompute_summaries()
                for entity_id, summary_data in aggregator_summaries.items():
                    cache_key = f"{gsw_index}_{entity_id}"
                    self._entity_summaries_cache[cache_key] = summary_data
        return self._entity_summaries_cache

    def _get_vp_summaries_cache(self) -> Dict[str, Dict[str, str]]:
        """Get or build verb phrase summaries cache."""
        if self._verb_summaries_cache is None:
            self._verb_summaries_cache = {}
            for gsw_index, aggregator in enumerate(self.verb_aggregators):
                aggregator_summaries = aggregator.precompute_summaries()
                for vp_id, summary_data in aggregator_summaries.items():
                    cache_key = f"{gsw_index}_{vp_id}"
                    self._verb_summaries_cache[cache_key] = summary_data
        return self._verb_summaries_cache

    def _get_conversation_summaries_cache(self) -> Dict[str, Dict[str, str]]:
        """Get or build conversation summaries cache."""
        if self._conversation_summaries_cache is None:
            self._conversation_summaries_cache = {}
            for gsw_index, aggregator in enumerate(self.conversation_aggregators):
                aggregator_summaries = aggregator.precompute_summaries()
                for conv_id, summary_data in aggregator_summaries.items():
                    cache_key = f"{gsw_index}_{conv_id}"
                    self._conversation_summaries_cache[cache_key] = summary_data
        return self._conversation_summaries_cache

    def _get_chunk_text_cache(self) -> Dict[str, str]:
        """Get or build chunk text cache from folder structure."""
        if self._chunk_text_cache is None:
            self._chunk_text_cache = {}
            
            if not self.chunks_folder_path:
                if self.verbose:
                    print("Warning: No chunks folder path provided. Chunk text retrieval unavailable.")
                return self._chunk_text_cache
                
            if not os.path.exists(self.chunks_folder_path):
                if self.verbose:
                    print(f"Warning: Chunks folder path does not exist: {self.chunks_folder_path}")
                return self._chunk_text_cache
            
            total_chunks_loaded = 0
            
            # Scan for doc_X/chunk_Y.txt files
            for doc_dir in os.listdir(self.chunks_folder_path):
                doc_path = os.path.join(self.chunks_folder_path, doc_dir)
                if os.path.isdir(doc_path) and doc_dir.startswith("doc_"):
                    # Extract doc number
                    try:
                        doc_num = doc_dir.replace("doc_", "")
                        doc_chunks_loaded = 0
                        
                        for chunk_file in os.listdir(doc_path):
                            if chunk_file.endswith(".txt") and chunk_file.startswith("chunk_"):
                                chunk_path = os.path.join(doc_path, chunk_file)
                                
                                # Extract chunk number
                                chunk_num = chunk_file.replace("chunk_", "").replace(".txt", "")
                                chunk_id = f"chunk_{doc_num}_{chunk_num}"
                                
                                # Alternative chunk ID formats
                                alt_chunk_id = f"{doc_num}_{chunk_num}"
                                simple_chunk_id = f"chunk_{chunk_num}"
                                
                                try:
                                    with open(chunk_path, 'r', encoding='utf-8') as f:
                                        text = f.read()
                                        if text.strip():  # Only add non-empty chunks
                                            self._chunk_text_cache[chunk_id] = text
                                            self._chunk_text_cache[alt_chunk_id] = text
                                            self._chunk_text_cache[simple_chunk_id] = text
                                            doc_chunks_loaded += 1
                                            total_chunks_loaded += 1
                                        elif self.verbose:
                                            print(f"Warning: Empty chunk file: {chunk_path}")
                                except Exception as e:
                                    if self.verbose:
                                        print(f"Warning: Could not read chunk file {chunk_path}: {e}")
                        
                        if self.verbose and doc_chunks_loaded > 0:
                            print(f"📄 Loaded {doc_chunks_loaded} chunks from {doc_dir}")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Could not process doc directory {doc_dir}: {e}")
            
            if self.verbose:
                if total_chunks_loaded > 0:
                    print(f"✅ Total chunks loaded: {total_chunks_loaded} with {len(self._chunk_text_cache)} cache entries")
                else:
                    print(f"❌ No chunks loaded from {self.chunks_folder_path}")
                            
        return self._chunk_text_cache

    def _find_matching_entities(self, entity_names: List[str]) -> List[Tuple[EntityNode, int]]:
        """Find matching entities across all GSWs."""
        matched_entities_with_source = []
        for gsw_index, gsw in enumerate(self.gsws):
            matched_entities = self.entity_matcher.find_matching_entities(entity_names, gsw)
            for entity in matched_entities:
                matched_entities_with_source.append((entity, gsw_index))
        return matched_entities_with_source

    def _get_entity_summaries(self, entities_with_source: List[Tuple[EntityNode, int]]) -> List[Tuple[str, str]]:
        """Get summaries for matched entities from appropriate aggregators."""
        entity_summaries_cache = self._get_entity_summaries_cache()
        summaries = []
        for entity, gsw_index in entities_with_source:
            cache_key = f"{gsw_index}_{entity.id}"
            if cache_key in entity_summaries_cache:
                summary_data = entity_summaries_cache[cache_key]
                summaries.append((entity.name, summary_data["summary"]))
        return summaries

    def _find_entity_id_for_summary(self, entity_name: str, summary: str) -> str:
        """Find entity ID that matches the given name and summary."""
        entity_summaries_cache = self._get_entity_summaries_cache()
        for cache_key, summary_data in entity_summaries_cache.items():
            if (summary_data["summary"] == summary and 
                summary_data.get("entity_name", "") == entity_name):
                gsw_index, entity_id = cache_key.split("_", 1)
                return entity_id
        return "unknown"

    def _extract_chunk_ids_from_id(self, full_id: str) -> List[str]:
        """Extract chunk IDs from entity/VP/conversation ID."""
        # Assumes ID format like "chunk_0::e1" or "chunk_0_1::vp2" or "0_1"
        if "::" in full_id:
            chunk_part = full_id.split("::")[0]
            return [chunk_part]
        elif full_id.startswith("chunk_"):
            # Find the chunk ID pattern
            parts = full_id.split("_")
            if len(parts) >= 2:
                try:
                    # Extract chunk_X pattern
                    chunk_num = parts[1].split("::")[0] if "::" in parts[1] else parts[1]
                    return [f"chunk_{chunk_num}"]
                except:
                    pass
        elif "_" in full_id and not full_id.startswith("chunk_"):
            # Format like "0_1" -> convert to "chunk_0_1"
            return [f"chunk_{full_id}"]
        return []

    @staticmethod
    def summarize_tool_result(tool_result: List[Dict[str, Any]]) -> str:
        """Summarize tool results to reduce conversation size."""
        if not tool_result:
            return "No results found."
        
        result_count = len(tool_result)
        if result_count == 0:
            return "No results found."
        
        # Show summary of results
        summaries = []
        for i, result in enumerate(tool_result[:3]):  # Show top 3
            if "entity_name" in result:
                summaries.append(f"Entity '{result['entity_name']}' ({result.get('entity_id', 'unknown')})")
            elif "vp_id" in result:
                summaries.append(f"VP {result['vp_id']}")
            elif "conversation_id" in result and "participants" in result:
                # Detailed conversation result
                participant_count = len(result.get("participants", []))
                summaries.append(f"Detailed conversation {result['conversation_id']} ({participant_count} participants)")
            elif "conversation_id" in result:
                summaries.append(f"Conversation {result['conversation_id']}")
            elif "chunk_id" in result:
                text_preview = result.get("text", "")[:100]
                summaries.append(f"Chunk {result['chunk_id']}: {text_preview}...")
        
        summary_text = " | ".join(summaries)
        more_text = f" (+{result_count-3} more)" if result_count > 3 else ""
        return f"Found {result_count} result(s): {summary_text}{more_text}" 