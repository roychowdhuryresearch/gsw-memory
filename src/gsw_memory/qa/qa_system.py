"""
Main Q&A system that orchestrates the complete pipeline.

This module provides the main GSWQuestionAnswerer class that coordinates
all steps of the Q&A process and provides both simple and agentic interfaces.
Supports both individual questions and batch processing for efficiency.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..memory.aggregators import EntitySummaryAggregator
from ..memory.models import EntityNode, GSWStructure
from .answering_agent import AnsweringAgent
from .entity_extractor import QuestionEntityExtractor
from .matcher import EntityMatcher
from .reranker import SummaryReranker


class GSWQuestionAnswerer:
    """
    Main Q&A system for GSW Memory.

    Orchestrates the 4-step pipeline:
    1. Extract entities from question (NER)
    2. Match entities to GSW nodes
    3. Retrieve entity summaries
    4. Rerank summaries by relevance

    Designed to support both simple usage and future agentic frameworks.
    Supports efficient batch processing for multiple questions.
    Supports multiple GSW structures for cross-document question answering.
    """

    def __init__(
        self,
        gsw: Union[GSWStructure, List[GSWStructure]],
        entity_aggregator: Union[
            EntitySummaryAggregator, List[EntitySummaryAggregator]
        ],
        llm_config: Dict[str, Any],
        embedding_model: str = "voyage-3",
    ):
        """
        Initialize the Q&A system.

        Args:
            gsw: GSW structure(s) to query - single GSWStructure or list of GSWStructures
            entity_aggregator: Pre-computed entity summaries - single aggregator or list matching GSWs
            llm_config: LLM configuration for entity extraction
            embedding_model: Embedding model name for reranking
        """
        # Handle backward compatibility: convert single inputs to lists
        if isinstance(gsw, GSWStructure):
            self.gsws = [gsw]
        else:
            self.gsws = gsw

        if isinstance(entity_aggregator, EntitySummaryAggregator):
            self.entity_aggregators = [entity_aggregator]
        else:
            self.entity_aggregators = entity_aggregator

        # Validate that GSWs and aggregators lists match
        if len(self.gsws) != len(self.entity_aggregators):
            raise ValueError(
                f"Number of GSWs ({len(self.gsws)}) must match number of aggregators ({len(self.entity_aggregators)})"
            )

        self.llm_config = llm_config

        # Initialize components
        self.entity_extractor = QuestionEntityExtractor(
            model_name=llm_config["model_name"],
            generation_params=llm_config.get("generation_params", {"temperature": 0.0}),
        )
        self.entity_matcher = EntityMatcher()
        self.summary_reranker = SummaryReranker(embedding_model)
        self.answering_agent = AnsweringAgent(
            model_name=llm_config["model_name"],
            generation_params=llm_config.get("generation_params", {"temperature": 0.0}),
        )

        # Cache for precomputed summaries from all aggregators
        self._summaries_cache: Optional[Dict[str, Dict[str, str]]] = None

    def ask(self, question: str, max_summaries: int = 17) -> Dict[str, Any]:
        """
        Simple interface: ask a question and get an answer.

        Args:
            question: The user's question
            max_summaries: Maximum summaries to use for context

        Returns:
            Dict with answer, sources, and metadata
        """
        results = self.ask_batch([question], max_summaries)
        return results[0]

    def ask_batch(
        self, questions: List[str], max_summaries: int = 17
    ) -> List[Dict[str, Any]]:
        """
        Batch interface: ask multiple questions efficiently using curator batching.

        Args:
            questions: List of questions to ask
            max_summaries: Maximum summaries to use for context per question

        Returns:
            List of result dicts, one per question
        """
        # Step 1: Extract entities from all questions (batched)
        all_entities = self.extract_entities_batch(questions)

        results = []
        questions_with_context = []
        for i, question in enumerate(questions):
            # Steps 2-4: Process each question individually
            # (these steps are fast and don't need batching)
            entities = all_entities[i]
            matches_with_source = self.find_matching_entities(entities)
            summaries = self.get_entity_summaries(matches_with_source)
            ranked_summaries = self.rerank_summaries(summaries, question, max_summaries)
            context_to_answering_agent = []
            for summary in ranked_summaries:
                context_to_answering_agent.append(
                    f"Entity: {summary[0]}\nSummary: {summary[1]}"
                )
            context_to_answering_agent = "\n\n".join(context_to_answering_agent)

            results.append(
                {
                    "question": question,
                    "extracted_entities": entities,
                    "matched_entities": [
                        entity.name for entity, _ in matches_with_source
                    ],
                    "ranked_summaries": ranked_summaries,
                    "num_summaries_used": len(ranked_summaries),
                    "context_to_answering_agent": context_to_answering_agent,
                }
            )
            questions_with_context.append(
                {
                    "question": question,
                    "context": context_to_answering_agent,
                }
            )

        # Step 5: Generate answers for all questions (batched)
        answers = self.generate_answer(questions_with_context).dataset
        for i, result in enumerate(results):
            result["answer"] = answers[i]["answer"]
            result["sources"] = answers[i]["context_used"]

        return results

    # Individual methods for agentic use
    def extract_entities(self, question: str) -> List[str]:
        """Step 1: Extract entities from question."""
        return self.entity_extractor.extract_entities(question)

    def extract_entities_batch(self, questions: List[str]) -> List[List[str]]:
        """Step 1 (batch): Extract entities from multiple questions efficiently."""
        return self.entity_extractor.extract_entities_batch(questions)

    def find_matching_entities(
        self, entity_names: List[str]
    ) -> List[Tuple[EntityNode, int]]:
        """Step 2: Find matching entities across all GSWs.

        Args:
            entity_names: List of entity names to match

        Returns:
            List of (EntityNode, gsw_index) tuples indicating which GSW each entity came from
        """
        matched_entities_with_source = []

        # Search across all GSWs
        for gsw_index, gsw in enumerate(self.gsws):
            matched_entities = self.entity_matcher.find_matching_entities(
                entity_names, gsw
            )
            # Add source GSW index to each matched entity
            for entity in matched_entities:
                matched_entities_with_source.append((entity, gsw_index))

        return matched_entities_with_source

    def get_entity_summaries(
        self, entities_with_source: List[Tuple[EntityNode, int]]
    ) -> List[Tuple[str, str]]:
        """Step 3: Get summaries for matched entities from appropriate aggregators.

        Args:
            entities_with_source: List of (EntityNode, gsw_index) tuples

        Returns:
            List of (entity_name, summary) tuples
        """
        if self._summaries_cache is None:
            # Build unified cache from all aggregators
            self._summaries_cache = {}
            for gsw_index, aggregator in enumerate(self.entity_aggregators):
                aggregator_summaries = aggregator.precompute_summaries()
                # Prefix entity IDs with GSW index to avoid conflicts
                for entity_id, summary_data in aggregator_summaries.items():
                    cache_key = f"{gsw_index}_{entity_id}"
                    self._summaries_cache[cache_key] = summary_data

        summaries = []
        for entity, gsw_index in entities_with_source:
            cache_key = f"{gsw_index}_{entity.id}"
            if cache_key in self._summaries_cache:
                summary_data = self._summaries_cache[cache_key]
                summaries.append((entity.name, summary_data["summary"]))

        return summaries

    def rerank_summaries(
        self, summaries: List[Tuple[str, str]], question: str, max_summaries: int = 17
    ) -> List[Tuple[str, str, float]]:
        """Step 4: Rerank summaries by relevance."""
        return self.summary_reranker.rerank_summaries(
            summaries, question, max_summaries
        )

    def generate_answer(
        self, questions_with_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Step 5: Generate final answer from ranked summaries."""
        return self.answering_agent(questions_with_context)
