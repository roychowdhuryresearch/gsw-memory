"""
Data models for the chain-following multi-hop QA pipeline.

Provides Pydantic models for question decomposition, QA pair results,
and chain-following results used across the pipeline.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DecomposedQuestion(BaseModel):
    """A single decomposed sub-question."""

    question: str
    requires_retrieval: bool


class DecomposedQuestionList(BaseModel):
    """Wrapper for structured LLM output of decomposed questions."""

    questions: List[DecomposedQuestion]


class QAPairResult(BaseModel):
    """A single QA pair retrieved from the GSW index."""

    question: str
    answer_names: List[str] = []
    answer_ids: List[str] = []
    answer_rolestates: List[str] = []
    verb_phrase: str = ""
    source_file: str = ""
    similarity_score: float = 0.0
    entity_score: float = 0.0
    source_method: str = ""


class ChainFollowingResult(BaseModel):
    """Final result from the chain-following QA pipeline."""

    question: str
    answer: str
    evidence: List[str] = []
    evidence_count: int = 0
    decomposed_questions: List[Dict[str, Any]] = []
    chains_info: Dict[str, Any] = {}
    time_taken: float = 0.0
