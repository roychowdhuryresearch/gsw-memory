"""
Entity summary aggregator for GSW structures.

This module implements an aggregator that creates chronological summaries
of entities within a GSW, supporting both static pre-computation and
dynamic query-driven generation with efficient batching.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from bespokelabs import curator

from ...prompts import EntitySummaryPrompts
from ..models import EntityNode, GSWStructure
from .base import AggregatedView, BaseAggregator


class GSWEntitySummarizer(curator.LLM):
    """Curator class for generating entity summaries from GSW data."""

    return_completions_object = True

    def prompt(self, input_data):
        """Create the prompt for the LLM summarizer."""
        include_space_time = input_data.get("include_space_time", False)
        
        system_prompt = EntitySummaryPrompts.get_system_prompt(include_space_time)
        user_prompt = EntitySummaryPrompts.get_user_prompt(
            entity_name=input_data["entity_name"],
            formatted_data=input_data["formatted_data"],
            include_space_time=include_space_time
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data, response):
        """Parse the LLM response to extract the summary."""
        summary_text = response["choices"][0]["message"]["content"].strip()
        return {
            "entity_id": input_data["entity_id"],
            "entity_name": input_data["entity_name"],
            "summary": summary_text,
            "_raw_response": response,  # For debugging
        }


class EntitySummaryAggregator(BaseAggregator):
    """
    Aggregator that creates chronological summaries of entities in a GSW.

    Supports both static batch processing and dynamic query-driven generation
    with efficient batching. Uses curator for LLM integration.
    """

    def __init__(self, gsw: GSWStructure, llm_config: Optional[Dict] = None):
        """
        Initialize the entity summary aggregator.

        Args:
            gsw: The GSW structure to aggregate
            llm_config: Configuration for the LLM (model name, generation params)
        """
        super().__init__(gsw)
        self.llm_config = llm_config or {
            "model_name": "gpt-4o",
            "generation_params": {"temperature": 0.0, "max_tokens": 500},
        }
        self._entity_map = {entity.id: entity.name for entity in gsw.entity_nodes}

    def aggregate(self, query: str, **kwargs) -> AggregatedView:
        """
        Generate aggregated summaries for entities relevant to the query.

        Args:
            query: Query string to extract target entities from
            **kwargs: Additional parameters:
                - entity_ids: Explicit list of entity IDs to aggregate
                - include_space_time: Whether to include spatial/temporal info

        Returns:
            AggregatedView containing summaries for all relevant entities
        """
        # Extract entities from query or kwargs
        entity_ids = kwargs.get("entity_ids")
        if not entity_ids:
            entity_ids = self._extract_entities_from_query(query)

        if not entity_ids:
            raise ValueError(f"Could not identify entities from query: {query}")

        include_space_time = kwargs.get("include_space_time", False)

        # Generate summaries for all relevant entities
        entity_summaries = self._generate_summaries(entity_ids, include_space_time)

        # Create aggregated view
        aggregated_view = AggregatedView(
            view_type="entity_summary",
            content={
                "entity_summaries": entity_summaries,
                "entity_ids": entity_ids,
                "include_space_time": include_space_time,
            },
            metadata={"query": query, "num_entities": len(entity_summaries)},
        )

        return aggregated_view

    def get_context(self, aggregated_view: AggregatedView) -> str:
        """
        Format an aggregated view into context string for downstream consumption.

        Args:
            aggregated_view: The aggregated view to format

        Returns:
            Formatted context string with all entity summaries
        """
        content = aggregated_view.content
        entity_summaries = content.get("entity_summaries", {})

        if not entity_summaries:
            return ""

        # Format multiple entity summaries
        context_parts = []
        for entity_id, summary_data in entity_summaries.items():
            entity_name = summary_data["entity_name"]
            summary = summary_data["summary"]
            context_parts.append(f"**{entity_name}**: {summary}")

        return "\n\n".join(context_parts)

    def precompute_summaries(
        self, entity_ids: Optional[List[str]] = None, include_space_time: bool = False
    ) -> Dict[str, Dict]:
        """
        Pre-compute summaries for multiple entities (static generation).

        Args:
            entity_ids: List of entity IDs to process (None for all entities)
            include_space_time: Whether to include spatial/temporal information

        Returns:
            Dictionary mapping entity IDs to their summary data
        """
        if entity_ids is None:
            entity_ids = [entity.id for entity in self.gsw.entity_nodes]

        return self._generate_summaries(entity_ids, include_space_time)

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract entity IDs from query string using name matching.

        Args:
            query: Query string that may contain entity names

        Returns:
            List of entity IDs found in the query
        """
        query_lower = query.lower()
        found_entities = []

        # Try exact name matching first
        for entity_id, entity_name in self._entity_map.items():
            if entity_name.lower() in query_lower:
                found_entities.append(entity_id)

        # If no exact matches, try partial matching for multi-word names
        if not found_entities:
            for entity_id, entity_name in self._entity_map.items():
                name_words = entity_name.lower().split()
                if len(name_words) > 1:
                    # Check if all words appear in query
                    if all(word in query_lower for word in name_words):
                        found_entities.append(entity_id)

        return found_entities

    def _generate_summaries(
        self, entity_ids: List[str], include_space_time: bool = False
    ) -> Dict[str, Dict]:
        """
        Generate summaries for multiple entities using curator batching.

        Args:
            entity_ids: List of entity IDs to generate summaries for
            include_space_time: Whether to include spatial/temporal information

        Returns:
            Dictionary mapping entity IDs to summary data
        """
        # Prepare inputs for curator
        summarizer_inputs = []

        for entity_id in entity_ids:
            entity = self._get_entity_by_id(entity_id)
            if not entity:
                print(f"Entity {entity_id} not found, skipping")
                continue

            # Aggregate chronological data
            chronological_data = self._aggregate_entity_data(entity, include_space_time)

            # Skip entities with no data
            if not chronological_data:
                print(f"No data found for entity {entity_id} ({entity.name}), skipping")
                continue

            # Format data for prompt
            formatted_data = self._format_data_for_prompt(
                entity.name, entity_id, chronological_data, include_space_time
            )

            summarizer_inputs.append(
                {
                    "entity_id": entity_id,
                    "entity_name": entity.name,
                    "formatted_data": formatted_data,
                    "include_space_time": include_space_time,
                }
            )

        if not summarizer_inputs:
            print("No entities found suitable for summarization")
            return {}

        # Initialize and run curator summarizer
        summarizer = GSWEntitySummarizer(
            model_name=self.llm_config["model_name"],
            generation_params=dict(self.llm_config["generation_params"]),
        )

        print(
            f"Generating summaries for {len(summarizer_inputs)} entities using {self.llm_config['model_name']}..."
        )

        # Batch process all entities
        summarization_results = summarizer(summarizer_inputs)

        # Convert results to dictionary format
        summary_map = {}
        for result in summarization_results.dataset:
            entity_id = result["entity_id"]
            summary_map[entity_id] = {
                "entity_name": result["entity_name"],
                "summary": result["summary"],
                "entity_id": entity_id,
            }

        print(f"Generated {len(summary_map)} summaries")
        return summary_map

    def _get_entity_by_id(self, entity_id: str) -> Optional[EntityNode]:
        """Get entity node by ID."""
        for entity in self.gsw.entity_nodes:
            if entity.id == entity_id:
                return entity
        return None

    def _aggregate_entity_data(
        self, entity: EntityNode, include_space_time: bool = False
    ) -> Dict[str, Any]:
        """
        Aggregate chronological data for a single entity.

        This method ports the core logic from gsw_summarizer.py's _aggregate_entity_data.
        """
        chronological_data = defaultdict(
            lambda: {"roles_states": [], "actions": [], "space_time": []}
        )

        # 1. Gather Roles/States
        for role_info in entity.roles:
            if role_info.chunk_id:
                chronological_data[role_info.chunk_id]["roles_states"].append(
                    {"role": role_info.role, "states": role_info.states}
                )

        # 2. Gather VP Involvement (Actions)
        for vp in self.gsw.verb_phrase_nodes:
            for q in vp.questions:
                if entity.id in q.answers:
                    if q.chunk_id:
                        action_context = {
                            "vp_phrase": vp.phrase,
                            "answered_question": q.text,
                            "context": [],
                        }

                        # Gather context from other questions in same VP
                        for other_q in vp.questions:
                            if (
                                other_q.id != q.id
                                and other_q.answers
                                and other_q.answers != ["None"]
                            ):
                                context_answers = []
                                for ans_id in other_q.answers:
                                    if ans_id in self._entity_map:
                                        context_answers.append(self._entity_map[ans_id])
                                    elif ans_id.startswith("TEXT:"):
                                        context_answers.append(ans_id[5:].strip())
                                    elif ans_id not in ["None", "none", "NA"]:
                                        context_answers.append(ans_id)

                                if context_answers:
                                    action_context["context"].append(
                                        f"{other_q.text} -> {', '.join(context_answers)}"
                                    )

                        chronological_data[q.chunk_id]["actions"].append(action_context)

        # 3. Gather Space/Time information if requested
        if include_space_time:
            self._add_space_time_data(entity, chronological_data)

        # 4. Sort by chunk_id
        sorted_chunk_ids = sorted(chronological_data.keys(), key=self._sort_chunk_key)
        sorted_data = {
            chunk_id: chronological_data[chunk_id] for chunk_id in sorted_chunk_ids
        }

        return sorted_data

    def _add_space_time_data(self, entity: EntityNode, chronological_data: defaultdict):
        """Add spatial and temporal information to chronological data."""
        # Create mapping of space/time node IDs for quick lookup
        space_node_map = {node.id: node for node in self.gsw.space_nodes}
        time_node_map = {node.id: node for node in self.gsw.time_nodes}

        # Find space connections through space_edges
        for edge in self.gsw.space_edges:
            entity_id, space_node_id = edge
            if entity_id == entity.id and space_node_id in space_node_map:
                space_node = space_node_map[space_node_id]
                chunk_id = space_node.chunk_id
                if chunk_id:
                    space_info = {
                        "type": "space",
                        "node_id": space_node.id,
                        "current_name": space_node.current_name,
                        "name_history": space_node.name_history,
                    }
                    chronological_data[chunk_id]["space_time"].append(space_info)

        # Find time connections through time_edges
        for edge in self.gsw.time_edges:
            entity_id, time_node_id = edge
            if entity_id == entity.id and time_node_id in time_node_map:
                time_node = time_node_map[time_node_id]
                chunk_id = time_node.chunk_id
                if chunk_id:
                    time_info = {
                        "type": "time",
                        "node_id": time_node.id,
                        "current_name": time_node.current_name,
                        "name_history": time_node.name_history,
                    }
                    chronological_data[chunk_id]["space_time"].append(time_info)

    def _sort_chunk_key(self, chunk_id_str: str) -> List[int]:
        """
        Robust key function for sorting chunk IDs like 'chunk_X_Y'.

        Ported from gsw_summarizer.py's sort_key function.
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
        entity_name: str,
        entity_id: str,
        chronological_data: Dict[str, Any],
        include_space_time: bool = False,
    ) -> str:
        """
        Format chronological data into a structured timeline string.

        Ported from gsw_summarizer.py's _format_data_for_prompt function.
        """
        prompt_text = [
            f"ENTITY NAME: {entity_name}",
            f"ENTITY ID: {entity_id}",
            "",
            "INFORMATION TIMELINE (by Chunk ID):",
            "",
        ]

        for chunk_id, data in chronological_data.items():
            prompt_text.append(f"--- Chunk: {chunk_id} ---")

            # Add Roles/States
            if data.get("roles_states"):
                prompt_text.append("ROLES/STATES:")
                for rs in data["roles_states"]:
                    prompt_text.append(f"  - Role: {rs['role']}")
                    if rs["states"]:
                        prompt_text.append(f"    States: {json.dumps(rs['states'])}")

            # Add Actions
            if data.get("actions"):
                prompt_text.append("ACTIONS (Verb Phrases Involved In):")
                for action in data["actions"]:
                    prompt_text.append(f"  - Verb Phrase: {action['vp_phrase']}")
                    prompt_text.append(
                        f'    - Role in action: Answered "{action["answered_question"]}"'
                    )
                    if action["context"]:
                        prompt_text.append("    - Context:")
                        for ctx in action["context"]:
                            prompt_text.append(f"      - {ctx}")

            # Add Space/Time information when flag is enabled
            if include_space_time and data.get("space_time"):
                prompt_text.append("SPACE/TIME CONTEXT:")
                for st_info in data["space_time"]:
                    if st_info["type"] == "space":
                        prompt_text.append(
                            f"  - Location Node ID: {st_info['node_id']}"
                        )
                        if st_info.get("current_name"):
                            prompt_text.append(
                                f"    Current Name: {st_info['current_name']}"
                            )
                        if st_info.get("name_history"):
                            prompt_text.append("    Name History:")
                            sorted_chunks = sorted(st_info["name_history"].keys())
                            for chunk in sorted_chunks:
                                prompt_text.append(
                                    f"      - {chunk}: {st_info['name_history'][chunk]}"
                                )
                    elif st_info["type"] == "time":
                        prompt_text.append(f"  - Time Node ID: {st_info['node_id']}")
                        if st_info.get("current_name"):
                            prompt_text.append(
                                f"    Current Name: {st_info['current_name']}"
                            )
                        if st_info.get("name_history"):
                            prompt_text.append("    Name History:")
                            sorted_chunks = sorted(st_info["name_history"].keys())
                            for chunk in sorted_chunks:
                                prompt_text.append(
                                    f"      - {chunk}: {st_info['name_history'][chunk]}"
                                )

            prompt_text.append("")  # Add space between chunks

        return "\n".join(prompt_text)
