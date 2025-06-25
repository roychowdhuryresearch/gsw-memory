"""
Tulving Bench evaluator with flexible interface.

This module provides the main evaluator for Tulving Bench that supports:
- Single system evaluation (GSW only or baseline only)
- Comparative evaluation (GSW vs baseline)
- Both original and new result formats
- Comprehensive metrics and reporting
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .judge import TulvingBenchJudge


class TulvingBenchEvaluator:
    """
    Main evaluator for Tulving Bench with flexible interface.
    
    Supports single-system or comparative evaluation with automatic
    format detection and comprehensive metrics reporting.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Tulving Bench evaluator.

        Args:
            model_name: LLM model to use for judge evaluation
            generation_params: Parameters for LLM generation
        """
        if generation_params is None:
            generation_params = {"temperature": 0.0}

        self.judge = TulvingBenchJudge(
            model_name=model_name,
            generation_params=generation_params,
        )

    def evaluate(
        self,
        qa_results: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[List[Dict[str, Any]]] = None,
        gsw_results: Optional[List[Dict[str, Any]]] = None,
        baseline_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Flexible evaluation interface supporting multiple input formats.

        Args:
            qa_results: Single system Q&A results (new format)
            ground_truth: Ground truth data (when using qa_results)
            gsw_results: GSW system results (comparative mode)
            baseline_results: Baseline system results (comparative mode)

        Returns:
            Dictionary with evaluation results and metrics
        """
        if qa_results is not None and ground_truth is not None:
            # Single system evaluation (new format)
            return self._evaluate_single_system(qa_results, ground_truth)
        
        elif gsw_results is not None or baseline_results is not None:
            # Comparative evaluation or single system (original format)
            return self._evaluate_comparative(gsw_results, baseline_results)
        
        else:
            raise ValueError(
                "Must provide either (qa_results + ground_truth) or (gsw_results/baseline_results)"
            )

    def _evaluate_single_system(
        self, qa_results: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single Q&A system against ground truth."""
        # Merge qa_results with ground_truth
        merged_data = []
        
        # Create a lookup for ground truth by question
        gt_lookup = {item["question"]: item for item in ground_truth}
        
        for result in qa_results:
            question = result["question"]
            if question in gt_lookup:
                gt_item = gt_lookup[question]
                merged_data.append({
                    "question": question,
                    "answer": result["answer"],
                    "correct_answer": gt_item["correct_answer"],
                    "retrieval_type": gt_item.get("retrieval_type", "unknown"),
                    "chapters_hit": result.get("matched_entities", []),  # Adapt field names
                    "correct_chapters": gt_item.get("correct_answer_chapters", []),
                })

        # Evaluate using judge
        questions = [item["question"] for item in merged_data]
        correct_answers = [item["correct_answer"] for item in merged_data]
        answers_to_evaluate = [item["answer"] for item in merged_data]
        retrieval_types = [item["retrieval_type"] for item in merged_data]

        evaluations = self.judge.evaluate_answers(
            questions, correct_answers, answers_to_evaluate, retrieval_types
        )

        # Calculate aggregate metrics
        aggregate_metrics = self.judge.calculate_aggregate_metrics(evaluations)

        # Calculate chapter metrics if available
        chapter_metrics = self._calculate_chapter_metrics(merged_data)

        return {
            "num_questions": len(evaluations),
            "system_metrics": aggregate_metrics,
            "chapter_metrics": chapter_metrics,
            "detailed_evaluations": evaluations,
        }

    def _evaluate_comparative(
        self,
        gsw_results: Optional[List[Dict[str, Any]]] = None,
        baseline_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate and compare GSW and/or baseline systems (original format)."""
        results = {}

        # Determine input format and extract data
        if gsw_results is not None and baseline_results is not None:
            # Both systems provided - merge them
            data = self._merge_comparative_results(gsw_results, baseline_results)
        elif gsw_results is not None:
            # GSW only
            data = gsw_results
        elif baseline_results is not None:
            # Baseline only
            data = baseline_results
        else:
            raise ValueError("Must provide at least one of gsw_results or baseline_results")

        # Auto-detect format and extract evaluation data
        evaluation_data = self._extract_evaluation_data(data)

        # Evaluate GSW if present
        if "gsw" in evaluation_data:
            gsw_eval = self.judge.evaluate_answers(**evaluation_data["gsw"])
            gsw_metrics = self.judge.calculate_aggregate_metrics(gsw_eval)
            results["gsw_metrics"] = gsw_metrics
            results["gsw_evaluations"] = gsw_eval

        # Evaluate baseline if present
        if "baseline" in evaluation_data:
            baseline_eval = self.judge.evaluate_answers(**evaluation_data["baseline"])
            baseline_metrics = self.judge.calculate_aggregate_metrics(baseline_eval)
            results["baseline_metrics"] = baseline_metrics
            results["baseline_evaluations"] = baseline_eval

        # Calculate chapter metrics
        chapter_metrics = self._calculate_chapter_metrics(data)
        results["chapter_metrics"] = chapter_metrics

        # Add comparison metrics if both systems evaluated
        if "gsw_metrics" in results and "baseline_metrics" in results:
            results["comparison"] = self._calculate_comparison_metrics(
                results["gsw_metrics"], results["baseline_metrics"]
            )

        results["num_questions"] = len(data)
        return results

    def _merge_comparative_results(
        self, gsw_results: List[Dict[str, Any]], baseline_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge GSW and baseline results for comparative evaluation."""
        merged = []
        
        # Create lookup for baseline results
        baseline_lookup = {item["question"]: item for item in baseline_results}
        
        for gsw_item in gsw_results:
            question = gsw_item["question"]
            if question in baseline_lookup:
                baseline_item = baseline_lookup[question]
                merged.append({
                    "question": question,
                    "correct_answer": gsw_item["correct_answer"],
                    "GSW_answer": gsw_item.get("answer", ""),
                    "baseline_answer": baseline_item.get("answer", ""),
                    "retrieval_type": gsw_item.get("retrieval_type", "unknown"),
                    "chapters_hit": gsw_item.get("chapters_hit", []),
                    "correct_chapters": gsw_item.get("correct_chapters", []),
                })
        
        return merged

    def _extract_evaluation_data(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        """Extract and organize data for judge evaluation."""
        evaluation_data = {}

        # Check if data contains both GSW and baseline answers (original format)
        has_gsw = any("GSW_answer" in item for item in data)
        has_baseline = any("baseline_answer" in item for item in data)
        has_single_answer = any("answer" in item for item in data)

        questions = [item["question"] for item in data]
        correct_answers = [item["correct_answer"] for item in data]
        retrieval_types = [item.get("retrieval_type", "unknown") for item in data]

        if has_gsw:
            evaluation_data["gsw"] = {
                "questions": questions,
                "correct_answers": correct_answers,
                "answers_to_evaluate": [item["GSW_answer"] for item in data],
                "retrieval_types": retrieval_types,
                "answer_types": ["gsw"] * len(data),
            }

        if has_baseline:
            evaluation_data["baseline"] = {
                "questions": questions,
                "correct_answers": correct_answers,
                "answers_to_evaluate": [item["baseline_answer"] for item in data],
                "retrieval_types": retrieval_types,
                "answer_types": ["baseline"] * len(data),
            }

        if has_single_answer and not has_gsw and not has_baseline:
            # Single system with "answer" field
            evaluation_data["single"] = {
                "questions": questions,
                "correct_answers": correct_answers,
                "answers_to_evaluate": [item["answer"] for item in data],
                "retrieval_types": retrieval_types,
                "answer_types": ["single"] * len(data),
            }

        return evaluation_data

    def _calculate_chapter_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate chapter retrieval metrics."""
        chapter_metrics_list = []
        non_empty_chapter_metrics = []

        for item in data:
            if "chapters_hit" in item and "correct_chapters" in item:
                metrics = self.judge.calculate_chapter_metrics(
                    item["chapters_hit"], item["correct_chapters"]
                )
                chapter_metrics_list.append(metrics)

                if item["correct_chapters"]:
                    non_empty_chapter_metrics.append(metrics)

        # Calculate averages
        avg_metrics = {
            "precision": np.mean([m["precision"] for m in chapter_metrics_list]) if chapter_metrics_list else 0,
            "recall": np.mean([m["recall"] for m in chapter_metrics_list]) if chapter_metrics_list else 0,
            "f1": np.mean([m["f1"] for m in chapter_metrics_list]) if chapter_metrics_list else 0,
        }

        avg_non_empty_metrics = {
            "precision": np.mean([m["precision"] for m in non_empty_chapter_metrics]) if non_empty_chapter_metrics else 0,
            "recall": np.mean([m["recall"] for m in non_empty_chapter_metrics]) if non_empty_chapter_metrics else 0,
            "f1": np.mean([m["f1"] for m in non_empty_chapter_metrics]) if non_empty_chapter_metrics else 0,
        }

        return {
            "all_questions": avg_metrics,
            "questions_with_chapters": avg_non_empty_metrics,
            "num_questions_with_chapters": len(non_empty_chapter_metrics),
        }

    def _calculate_comparison_metrics(
        self, gsw_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate comparison metrics between GSW and baseline."""
        f1_improvement = (
            (gsw_metrics["f1"] - baseline_metrics["f1"]) / baseline_metrics["f1"] * 100
            if baseline_metrics["f1"] > 0
            else float("inf")
        )

        return {
            "f1_difference": gsw_metrics["f1"] - baseline_metrics["f1"],
            "f1_improvement_percent": f1_improvement if f1_improvement != float("inf") else None,
            "precision_difference": gsw_metrics["precision"] - baseline_metrics["precision"],
            "recall_difference": gsw_metrics["recall"] - baseline_metrics["recall"],
        }

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save evaluation results to a JSON file."""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved evaluation results to {output_file}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}")

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the evaluation results."""
        print("\n" + "=" * 50)
        print(f"TULVING BENCH EVALUATION SUMMARY ({results.get('num_questions', 0)} questions)")
        print("=" * 50)

        # Print single system metrics
        if "system_metrics" in results:
            metrics = results["system_metrics"]
            print("\nSYSTEM PERFORMANCE:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        # Print comparative metrics
        if "gsw_metrics" in results:
            gsw = results["gsw_metrics"]
            print("\nGSW MODEL PERFORMANCE:")
            print(f"  Precision: {gsw['precision']:.4f}")
            print(f"  Recall:    {gsw['recall']:.4f}")
            print(f"  F1 Score:  {gsw['f1']:.4f}")

        if "baseline_metrics" in results:
            baseline = results["baseline_metrics"]
            print("\nBASELINE MODEL PERFORMANCE:")
            print(f"  Precision: {baseline['precision']:.4f}")
            print(f"  Recall:    {baseline['recall']:.4f}")
            print(f"  F1 Score:  {baseline['f1']:.4f}")

        # Print comparison
        if "comparison" in results:
            comp = results["comparison"]
            print("\nCOMPARISON:")
            print(f"  F1 Score Difference: {comp['f1_difference']:.4f}")
            if comp["f1_improvement_percent"] is not None:
                print(f"  GSW F1 Improvement:  {comp['f1_improvement_percent']:.2f}%")
            else:
                print("  GSW F1 Improvement:  N/A (baseline F1 is 0)")

        # Print chapter metrics
        if "chapter_metrics" in results:
            chapter = results["chapter_metrics"]["all_questions"]
            print("\nCHAPTER RETRIEVAL PERFORMANCE:")
            print(f"  Precision: {chapter['precision']:.4f}")
            print(f"  Recall:    {chapter['recall']:.4f}")
            print(f"  F1 Score:  {chapter['f1']:.4f}")

            if results["chapter_metrics"]["num_questions_with_chapters"] > 0:
                non_empty = results["chapter_metrics"]["questions_with_chapters"]
                print(f"\nCHAPTER RETRIEVAL (QUESTIONS WITH CHAPTERS, n={results['chapter_metrics']['num_questions_with_chapters']}):")
                print(f"  Precision: {non_empty['precision']:.4f}")
                print(f"  Recall:    {non_empty['recall']:.4f}")
                print(f"  F1 Score:  {non_empty['f1']:.4f}")

        print("=" * 50 + "\n")