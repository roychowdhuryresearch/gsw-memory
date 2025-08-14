#!/usr/bin/env python3
"""
Test script for baseline embedding agent experiments.

This script runs the baseline embedding agent on Musique questions and compares
performance with the original GSW agent.
"""

import json
import argparse
from datetime import datetime
import os
import sys
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gsw_memory.qa.baseline_embedding_agent import BaselineEmbeddingAgent


def load_musique_questions(musique_path: str, num_questions: int = None) -> List[Dict[str, Any]]:
    """Load Musique questions from JSON file."""
    with open(musique_path, 'r') as f:
        data = json.load(f)
    
    if num_questions:
        data = data[:num_questions]
    
    return data


def evaluate_baseline_agent(
    agent: BaselineEmbeddingAgent,
    questions: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """
    Evaluate the baseline agent on Musique questions.
    
    Args:
        agent: The baseline embedding agent
        questions: List of Musique question dictionaries
        output_dir: Directory to save results
        
    Returns:
        Evaluation results dictionary
    """
    results = []
    correct = 0
    total = len(questions)
    
    print(f"\n{'='*80}")
    print(f"Starting evaluation on {total} questions")
    print(f"{'='*80}\n")
    
    for i, q_data in enumerate(questions):
        question = q_data['question']
        gold_answer = q_data['answer']
        
        print(f"\n[{i+1}/{total}] Processing: {question[:100]}...")
        print(f"Gold answer: {gold_answer}")
        
        try:
            # Get agent's response
            response = agent.answer_question(question)
            
            # Check if answer matches (case-insensitive)
            predicted = response.answer.strip().lower()
            gold = gold_answer.strip().lower()
            is_correct = predicted == gold or gold in predicted
            
            if is_correct:
                correct += 1
                print(f"✓ Correct! Agent answer: {response.answer}")
            else:
                print(f"✗ Incorrect. Agent answer: {response.answer}")
            
            # Store result
            result = {
                "question_id": q_data.get('id', f"q_{i}"),
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": response.answer,
                "is_correct": is_correct,
                "reasoning": response.reasoning,
                "num_tool_calls": len(response.tool_calls_made),
                "tool_calls": response.tool_calls_made
            }
            results.append(result)
            
            # Save intermediate results
            if (i + 1) % 5 == 0:
                save_results(results, output_dir, "intermediate")
                print(f"\nProgress: {correct}/{i+1} correct ({100*correct/(i+1):.1f}%)")
            
        except Exception as e:
            print(f"Error processing question: {e}")
            result = {
                "question_id": q_data.get('id', f"q_{i}"),
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "",
                "is_correct": False,
                "reasoning": f"Error: {str(e)}",
                "num_tool_calls": 0,
                "tool_calls": []
            }
            results.append(result)
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    avg_tool_calls = sum(r['num_tool_calls'] for r in results) / total if total > 0 else 0
    
    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_tool_calls": avg_tool_calls,
        "results": results
    }
    
    # Save final results
    save_results(evaluation, output_dir, "final")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Accuracy: {correct}/{total} ({100*accuracy:.1f}%)")
    print(f"Average tool calls per question: {avg_tool_calls:.1f}")
    print(f"Results saved to: {output_dir}")
    
    return evaluation


def save_results(data: Any, output_dir: str, suffix: str = ""):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_results_{suffix}_{timestamp}.json" if suffix else f"baseline_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved results to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Test baseline embedding agent on Musique questions")
    parser.add_argument(
        "--corpus",
        type=str,
        default="/home/yigit/codebase/gsw-memory/musique_corpus_50_q.json",
        help="Path to the corpus JSON file"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="/home/yigit/codebase/gsw-memory/musique_50_q.json",
        help="Path to the Musique questions JSON file"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of questions to evaluate (default: 10, use -1 for all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum iterations per question (default: 15)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nvidia/NV-Embed-v2",
        help="Embedding model to use (default: nvidia/NV-Embed-v2)"
    )
    parser.add_argument(
        "--test-search",
        action="store_true",
        help="Test the search functionality with a sample query"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/home/yigit/codebase/gsw-memory/logs/baseline_embedding_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"BASELINE EMBEDDING AGENT EXPERIMENT")
    print(f"{'='*80}")
    print(f"Corpus: {args.corpus}")
    print(f"Questions: {args.questions}")
    print(f"Num questions: {args.num_questions if args.num_questions > 0 else 'all'}")
    print(f"Model: {args.model}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Initialize agent
    print("Initializing baseline embedding agent...")
    agent = BaselineEmbeddingAgent(
        corpus_path=args.corpus,
        model_name=args.model,
        generation_params={"temperature": 0.0},
        max_iterations=args.max_iterations,
        embedding_model=args.embedding_model
    )
    
    # Test search if requested
    if args.test_search:
        print("\n" + "="*80)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*80)
        
        test_queries = [
            "first President of Namibia",
            "McDonald's history",
            "Malakoff capture",
            "Christopher Nolan films",
            "capital of France"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = agent.search_embeddings(query, top_k=3)
            for i, result in enumerate(results):
                print(f"\n  [{i+1}] {result['title']} (score: {result['score']:.4f})")
                print(f"      {result['text'][:200]}...")
        
        print("\n" + "="*80)
        print("Search test complete. Exiting.")
        return
    
    # Load questions
    print(f"\nLoading questions from {args.questions}...")
    questions = load_musique_questions(
        args.questions,
        None if args.num_questions == -1 else args.num_questions
    )
    print(f"Loaded {len(questions)} questions")
    
    # Run evaluation
    evaluation = evaluate_baseline_agent(agent, questions, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Corpus: {args.corpus}")
    print(f"  - Model: {args.model}")
    print(f"  - Embedding model: {args.embedding_model}")
    print(f"  - Max iterations: {args.max_iterations}")
    print(f"\nResults:")
    print(f"  - Questions evaluated: {evaluation['num_questions']}")
    print(f"  - Correct answers: {evaluation['correct']}")
    print(f"  - Accuracy: {evaluation['accuracy']*100:.1f}%")
    print(f"  - Avg tool calls: {evaluation['avg_tool_calls']:.1f}")
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()