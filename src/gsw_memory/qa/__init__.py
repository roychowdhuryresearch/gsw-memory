"""
Q&A module for GSW Memory System.

This module provides question-answering capabilities for the GSW framework,
implementing the paper's approach:
1. Extract entities from questions using LLM-based NER
2. Match entities to GSW nodes using approximate matching
3. Retrieve entity summaries from EntitySummaryAggregator
4. Rerank summaries using VoyageAI embeddings
"""

from .entity_extractor import QuestionEntityExtractor
from .matcher import EntityMatcher
from .reranker import SummaryReranker
from .qa_system import GSWQuestionAnswerer

__all__ = [
    "QuestionEntityExtractor",
    "EntityMatcher", 
    "SummaryReranker",
    "GSWQuestionAnswerer",
]