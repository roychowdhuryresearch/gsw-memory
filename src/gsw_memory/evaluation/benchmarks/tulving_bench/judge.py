"""
Tulving Bench specific LLM judge implementation.

This module provides the LLM judge for evaluating Q&A responses using the
Tulving Bench methodology, including prompt structure, response parsing,
and metrics calculation specific to Tulving Bench.
"""

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from bespokelabs import curator

from ...judges.base_judge import BaseJudge


class TulvingBenchJudge(BaseJudge, curator.LLM):
    """LLM-as-a-judge for Tulving Bench evaluation with integrated metrics calculation."""

    return_completions_object = True

    def prompt(self, input_data):
        """Create a prompt for the judge LLM to evaluate an answer using Tulving Bench methodology."""
        system_prompt = "You are an expert judge evaluating the accuracy of AI-generated answers against known ground truth."

        # Format the expected evaluation structure
        d = [{x: "score_between_0_and_1"} for x in input_data["correct_answer"]]

        user_prompt = f"""
You are an expert judge evaluating the accuracy of an AI-generated answer against a known groundtruth. Questions can probe for different types or aspects, like what actions or events took place, what people were involved, what were the dates, or what were the locations or spaces.

Question type: {input_data['retrieval_type']}
Groundtruth: {input_data['correct_answer']}
AI-generated answer: {input_data['answer_to_evaluate']}

Your task:
- Identify all unique items in the AI-generated answer that are relevant to the question type. Answer an empty list [] for this field in case of at least one negative information (e.g., when the answer begins by telling there is no information, or cannot answer)
- Determine a matching score between 0 and 1 for each ground truth item. Give 1 if the item has been found in the relevant items of the AI-generated answer, considering synonyms, paraphrases, or close meanings. Give 0.5 if the item could be considered related to any AI-generated item but without being explicitly stated as such. Give 0 if the item missed mentioning a specific AI-generated item.
- Provide a brief explanation of the evaluation

Provide your evaluation in the following JSON format:
{{
    "identified_items_in_AI_answer": ["AI_answer_item_1", "AI_answer_item_2", ...],
    "matching_score": {json.dumps(d)},
    "explanation": "Brief explanation of your evaluation"
}}
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input_data, response):
        """Parse the LLM response and extract the evaluation with metrics."""
        answer_text = response["choices"][0]["message"]["content"].strip()

        # Try to parse JSON from the response
        try:
            # First check if the response is a clean JSON object
            evaluation = json.loads(answer_text)
        except json.JSONDecodeError:
            # If not, try to extract JSON from the text
            json_match = re.search(r"\{.*\}", answer_text, re.DOTALL)
            if json_match:
                try:
                    evaluation = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # If JSON extraction fails, return a structured error
                    evaluation = {
                        "error": "Could not parse JSON",
                        "raw_response": answer_text,
                        "identified_items_in_AI_answer": [],
                        "matching_score": [],
                        "explanation": "Failed to parse judge response",
                    }
            else:
                # If no JSON-like text is found
                evaluation = {
                    "error": "No JSON found in response",
                    "raw_response": answer_text,
                    "identified_items_in_AI_answer": [],
                    "matching_score": [],
                    "explanation": "Failed to parse judge response",
                }

        # Calculate Tulving Bench specific metrics
        metrics = self._calculate_tulving_metrics(
            evaluation.get("matching_score", []),
            len(input_data["correct_answer"]),
            len(evaluation.get("identified_items_in_AI_answer", [])),
        )

        # Return the full evaluation with metrics
        return {
            "question": input_data["question"],
            "answer_type": input_data.get("answer_type", "unknown"),
            "correct_answer": input_data["correct_answer"],
            "answer_evaluated": input_data["answer_to_evaluate"],
            "retrieval_type": input_data.get("retrieval_type", "unknown"),
            "identified_items": evaluation.get("identified_items_in_AI_answer", []),
            "matching_score": evaluation.get("matching_score", []),
            "explanation": evaluation.get("explanation", ""),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "raw_response": answer_text if "error" in evaluation else None,
        }

    def _calculate_tulving_metrics(
        self, matching_score: List[Dict[str, str]], num_gt_items: int, num_pred_items: int
    ) -> Dict[str, Optional[float]]:
        """Calculate Tulving Bench specific precision, recall, and F1 score."""
        
        # Case 1: No ground truth items
        if num_gt_items == 0:
            if num_pred_items == 0:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            else:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Case 2a: No predicted items, but ground truth items exist
        if num_pred_items == 0:
            return {"precision": None, "recall": 0.0, "f1": 0.0}

        # Case 2b: Both ground truth items and predicted items exist
        try:
            sum_scores = 0.0
            if matching_score:
                valid_scores = []
                for score_dict in matching_score:
                    if (
                        score_dict
                        and isinstance(score_dict, dict)
                        and len(score_dict.values()) == 1
                    ):
                        try:
                            valid_scores.append(float(list(score_dict.values())[0]))
                        except (ValueError, TypeError):
                            print(
                                f"Warning: Invalid score format in matching_score item: {score_dict}. Skipping."
                            )
                    elif score_dict:
                        print(
                            f"Warning: Malformed score_dict in matching_score: {score_dict}. Skipping."
                        )
                sum_scores = sum(valid_scores)
            elif num_gt_items > 0:
                print(
                    f"Warning: matching_score is empty but num_gt_items ({num_gt_items}) > 0. Assuming all scores for GT items are 0."
                )

            # Tulving Bench specific calculations
            precision = sum_scores / num_pred_items
            recall = sum_scores / num_gt_items

            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            return {"precision": precision, "recall": recall, "f1": f1}

        except Exception as e:
            print(
                f"Error during Tulving Bench metric calculation (num_gt={num_gt_items}, num_pred={num_pred_items}): {e}"
            )
            return {"precision": None, "recall": None, "f1": 0.0}

    def evaluate_answers(
        self,
        questions: List[str],
        correct_answers: List[List[str]],
        answers_to_evaluate: List[str],
        retrieval_types: List[str] = None,
        answer_types: List[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of Q&A responses using Tulving Bench methodology."""
        if retrieval_types is None:
            retrieval_types = ["unknown"] * len(questions)
        if answer_types is None:
            answer_types = ["unknown"] * len(questions)

        # Prepare input data for curator
        evaluation_inputs = []
        for i, (question, correct_answer, answer_to_eval, retrieval_type, answer_type) in enumerate(
            zip(questions, correct_answers, answers_to_evaluate, retrieval_types, answer_types)
        ):
            evaluation_inputs.append({
                "question": question,
                "correct_answer": correct_answer,
                "answer_to_evaluate": answer_to_eval,
                "retrieval_type": retrieval_type,
                "answer_type": answer_type,
            })

        # Run evaluation
        results = self(evaluation_inputs)
        return results.dataset

    def evaluate_single(
        self,
        question: str,
        correct_answer: List[str],
        answer_to_evaluate: str,
        retrieval_type: str = "unknown",
        answer_type: str = "unknown",
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a single Q&A response using Tulving Bench methodology."""
        results = self.evaluate_answers(
            [question], [correct_answer], [answer_to_evaluate], [retrieval_type], [answer_type]
        )
        return results[0]

    def calculate_aggregate_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate Tulving Bench metrics across multiple evaluation results."""
        if not evaluation_results:
            return {"precision": 0, "recall": 0, "f1": 0}

        # Filter out None values before averaging
        precision_values = [
            e["precision"] for e in evaluation_results if e["precision"] is not None
        ]
        recall_values = [e["recall"] for e in evaluation_results if e["recall"] is not None]
        f1_values = [e["f1"] for e in evaluation_results if e["f1"] is not None]

        # Calculate averages
        avg_precision = np.mean(precision_values) if precision_values else 0
        avg_recall = np.mean(recall_values) if recall_values else 0
        avg_f1 = np.mean(f1_values) if f1_values else 0

        return {
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1),
        }

    def calculate_chapter_metrics(
        self, chapters_hit: List[int], correct_chapters: List[int]
    ) -> Dict[str, float]:
        """Calculate chapter retrieval metrics specific to Tulving Bench."""
        # Convert to sets for easier intersection calculation
        hit_set = set(chapters_hit)
        correct_set = set(correct_chapters)

        # Calculate intersection
        intersection = hit_set.intersection(correct_set)

        # Calculate metrics
        precision = len(intersection) / len(hit_set) if hit_set else 0
        recall = len(intersection) / len(correct_set) if correct_set else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}