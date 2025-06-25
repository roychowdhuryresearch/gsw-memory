"""
Abstract base class for evaluation judges.

This module provides the interface that all judges (LLM-based or otherwise)
should implement for evaluating Q&A responses. Each judge is responsible for
both evaluation and metrics calculation specific to their benchmark.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseJudge(ABC):
    """Abstract base class for evaluation judges."""

    @abstractmethod
    def evaluate_answers(
        self,
        questions: List[str],
        correct_answers: List[List[str]],
        answers_to_evaluate: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of Q&A responses.

        Args:
            questions: List of questions
            correct_answers: List of ground truth answers (each can have multiple items)
            answers_to_evaluate: List of AI-generated answers to evaluate
            **kwargs: Additional arguments specific to the judge implementation

        Returns:
            List of evaluation results with computed metrics for each response
        """
        pass

    @abstractmethod
    def evaluate_single(
        self,
        question: str,
        correct_answer: List[str],
        answer_to_evaluate: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A response.

        Args:
            question: The question
            correct_answer: Ground truth answer (can have multiple items)
            answer_to_evaluate: AI-generated answer to evaluate
            **kwargs: Additional arguments specific to the judge implementation

        Returns:
            Evaluation result with computed metrics
        """
        pass

    @abstractmethod
    def calculate_aggregate_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate metrics across multiple evaluation results.

        Args:
            evaluation_results: List of individual evaluation results from evaluate_answers()

        Returns:
            Dictionary with aggregate metrics (e.g., average precision, recall, F1)
        """
        pass