"""
Q&A module for GSW Memory System.

This module provides question-answering capabilities for the GSW framework:

1. **Summary-based QA** (original pipeline):
   Extract entities → match → summarize → rerank → answer.

2. **Chain-following QA** (multi-hop pipeline):
   Decompose question → identify chains → beam-search retrieval → answer.
"""

from .answering_agent import AnsweringAgent
from .chain_answer_generator import ChainAnswerGenerator
from .chain_following_qa import ChainFollowingQA
from .chain_models import (
    ChainFollowingResult,
    DecomposedQuestion,
    DecomposedQuestionList,
)
from .entity_extractor import QuestionEntityExtractor
from .gsw_tools import GSWTools
from .matcher import EntityMatcher
from .qa_system import GSWQuestionAnswerer
from .question_decomposer import QuestionDecomposer
from .reranker import SummaryReranker

__all__ = [
    # Original pipeline
    "QuestionEntityExtractor",
    "EntityMatcher",
    "SummaryReranker",
    "GSWQuestionAnswerer",
    "AnsweringAgent",
    # Chain-following pipeline
    "ChainFollowingQA",
    "ChainFollowingResult",
    "ChainAnswerGenerator",
    "QuestionDecomposer",
    "DecomposedQuestion",
    "DecomposedQuestionList",
    # Tools
    "GSWTools",
]
