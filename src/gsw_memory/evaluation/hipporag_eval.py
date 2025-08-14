"""
HippoRAG-compatible evaluation utilities for fair comparison.

This module implements the same evaluation methodology used by HippoRAG:
- Answer normalization (lowercase, remove punctuation, articles, whitespace)
- Exact Match (EM) scoring
- F1 scoring with token-level precision/recall
- Support for multiple gold answers per question

Based on HippoRAG's evaluation code for direct comparison.
"""

import re
import string
from typing import List, Dict, Tuple, Callable
from collections import Counter
import numpy as np


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string using HippoRAG's methodology.
    
    Applies the following transformations:
    1. Convert to lowercase
    2. Remove punctuation characters
    3. Remove articles "a", "an", "the"
    4. Normalize whitespace (collapse multiple spaces)
    
    Args:
        answer: Input answer string
        
    Returns:
        Normalized answer string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def calculate_exact_match(gold_answers: List[str], predicted_answer: str, 
                         aggregation_fn: Callable = np.max) -> float:
    """
    Calculate Exact Match score between prediction and gold answers.
    
    Args:
        gold_answers: List of gold standard answers
        predicted_answer: Predicted answer string
        aggregation_fn: Function to aggregate scores across multiple gold answers
        
    Returns:
        Exact match score (1.0 if any gold answer matches, 0.0 otherwise)
    """
    em_scores = [
        1.0 if normalize_answer(gold) == normalize_answer(predicted_answer) else 0.0 
        for gold in gold_answers
    ]
    return float(aggregation_fn(em_scores))


def calculate_f1_score(gold_answers: List[str], predicted_answer: str,
                      aggregation_fn: Callable = np.max) -> float:
    """
    Calculate F1 score between prediction and gold answers.
    
    Args:
        gold_answers: List of gold standard answers
        predicted_answer: Predicted answer string  
        aggregation_fn: Function to aggregate scores across multiple gold answers
        
    Returns:
        F1 score (best score across all gold answers)
    """
    def compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(predicted_tokens) if predicted_tokens else 0.0
        recall = 1.0 * num_same / len(gold_tokens) if gold_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)

    f1_scores = [compute_f1(gold, predicted_answer) for gold in gold_answers]
    return float(aggregation_fn(f1_scores))


def evaluate_qa_batch(gold_answers_list: List[List[str]], 
                     predicted_answers: List[str]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Evaluate a batch of Q&A predictions using HippoRAG methodology.
    
    Args:
        gold_answers_list: List of lists, each containing gold answers for a question
        predicted_answers: List of predicted answers
        
    Returns:
        Tuple of (overall_metrics, per_example_metrics)
        - overall_metrics: Dict with average EM and F1 scores
        - per_example_metrics: List of dicts with per-question EM and F1 scores
    """
    assert len(gold_answers_list) == len(predicted_answers), \
        "Length of gold answers and predicted answers should be the same."
    
    example_results = []
    total_em = 0.0
    total_f1 = 0.0
    
    for gold_answers, predicted in zip(gold_answers_list, predicted_answers):
        em_score = calculate_exact_match(gold_answers, predicted)
        f1_score = calculate_f1_score(gold_answers, predicted)
        
        example_results.append({
            "ExactMatch": em_score,
            "F1": f1_score,
            "predicted_answer": predicted,
            "gold_answers": gold_answers
        })
        
        total_em += em_score
        total_f1 += f1_score
    
    # Calculate averages
    avg_em = total_em / len(gold_answers_list) if gold_answers_list else 0.0
    avg_f1 = total_f1 / len(gold_answers_list) if gold_answers_list else 0.0
    
    overall_results = {
        "ExactMatch": round(avg_em, 4),
        "F1": round(avg_f1, 4)
    }
    
    return overall_results, example_results

def evaluate_qa_batch_w_types(question_types: List[str], gold_answers_list: List[List[str]], 
                     predicted_answers: List[str]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Evaluate a batch of Q&A predictions using HippoRAG methodology.
    
    Args:
        gold_answers_list: List of lists, each containing gold answers for a question
        predicted_answers: List of predicted answers
        
    Returns:
        Tuple of (overall_metrics, per_example_metrics)
        - overall_metrics: Dict with average EM and F1 scores
        - per_example_metrics: List of dicts with per-question EM and F1 scores
    """
    unique_question_types = list(set(question_types))
    assert len(gold_answers_list) == len(predicted_answers), \
        "Length of gold answers and predicted answers should be the same."
    assert len(question_types) == len(gold_answers_list), \
        "Length of question types and gold answers should be the same."
    
    example_results = []
    total_em = 0.0
    total_f1 = 0.0
    
    # Initialize per-type metrics
    type_metrics = {qtype: {"total_em": 0.0, "total_f1": 0.0, "count": 0} for qtype in unique_question_types}
    
    for i, (q_type, gold_answers, predicted) in enumerate(zip(question_types, gold_answers_list, predicted_answers)):
        em_score = calculate_exact_match(gold_answers, predicted)
        f1_score = calculate_f1_score(gold_answers, predicted)
        
        example_results.append({
            "ExactMatch": em_score,
            "F1": f1_score,
            "predicted_answer": predicted,
            "gold_answers": gold_answers,
            "question_type": q_type
        })
        
        # Update overall metrics
        total_em += em_score
        total_f1 += f1_score
        
        # Update per-type metrics
        type_metrics[q_type]["total_em"] += em_score
        type_metrics[q_type]["total_f1"] += f1_score
        type_metrics[q_type]["count"] += 1
    
    # Calculate overall averages
    avg_em = total_em / len(gold_answers_list) if gold_answers_list else 0.0
    avg_f1 = total_f1 / len(gold_answers_list) if gold_answers_list else 0.0
    
    # Calculate per-type averages
    type_results = {}
    for q_type, metrics in type_metrics.items():
        if metrics["count"] > 0:
            type_results[q_type] = {
                "ExactMatch": round(metrics["total_em"] / metrics["count"], 4),
                "F1": round(metrics["total_f1"] / metrics["count"], 4),
                "count": metrics["count"]
            }
    
    overall_results = {
        "ExactMatch": round(avg_em, 4),
        "F1": round(avg_f1, 4),
        "by_type": type_results
    }
    
    return overall_results, example_results


def format_evaluation_report(overall_results: Dict[str, float], 
                           example_results: List[Dict[str, float]],
                           show_examples: int = 3) -> str:
    """
    Format evaluation results into a readable report.
    
    Args:
        overall_results: Overall metrics dict
        example_results: Per-example results list
        show_examples: Number of examples to show in detail
        
    Returns:
        Formatted evaluation report string
    """
    report_lines = []
    
    # Overall results
    report_lines.append("=== HippoRAG-Style Evaluation Results ===")
    report_lines.append(f"Overall Metrics:")
    report_lines.append(f"  Exact Match: {overall_results['ExactMatch']:.4f}")
    report_lines.append(f"  F1 Score: {overall_results['F1']:.4f}")
    report_lines.append(f"  Total Questions: {len(example_results)}")
    
    # Detailed examples
    if show_examples > 0:
        report_lines.append(f"\n=== Sample Results (first {show_examples}) ===")
        for i, result in enumerate(example_results[:show_examples]):
            em_status = "✅" if result["ExactMatch"] == 1.0 else "❌"
            f1_status = "✅" if result["F1"] > 0.5 else "❌"
            
            report_lines.append(f"\nExample {i+1}:")
            report_lines.append(f"  Predicted: {result['predicted_answer']}")
            report_lines.append(f"  Gold: {result['gold_answers']}")
            report_lines.append(f"  EM: {result['ExactMatch']:.3f} {em_status}")
            report_lines.append(f"  F1: {result['F1']:.3f} {f1_status}")
    
    return "\n".join(report_lines)