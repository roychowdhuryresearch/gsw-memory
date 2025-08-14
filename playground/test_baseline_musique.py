#!/usr/bin/env python3
"""
Test Baseline Embedding Agent on Musique Questions

This script tests the baseline embedding approach using nvembed-v2 model
for dense retrieval, comparing it to the GSW agentic approach.
"""

import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gsw_memory.qa.baseline_embedding_agent import BaselineEmbeddingAgent
from gsw_memory import hipporag_eval

# Load environment variables
load_dotenv()

# Configuration
USE_SUBSET = True  # Test on subset of questions
SUBSET_SIZE = 50  # Number of questions for subset
CORPUS_PATH = "/home/yigit/codebase/gsw-memory/musique_corpus_50_q.json"
QUESTIONS_PATH = "/home/yigit/codebase/gsw-memory/musique.json"


def setup_logging():
    """Create timestamped log directory for baseline results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__), "logs", f"baseline_musique_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 Created log directory: {log_dir}")
    return log_dir, timestamp


def load_questions_data():
    """Load Musique questions from JSON file."""
    print(f"\n=== Loading Questions from {QUESTIONS_PATH} ===")
    
    with open(QUESTIONS_PATH, "r") as f:
        questions_data = json.load(f)
    
    if USE_SUBSET:
        questions_data = questions_data[:SUBSET_SIZE]
        print(f"Using subset: {len(questions_data)} questions")
    
    print(f"Total questions loaded: {len(questions_data)}")
    
    return questions_data


def test_baseline_search(baseline_agent):
    """Quick test of baseline search functionality."""
    print("\n=== Testing Baseline Search ===")
    
    test_queries = [
        "first President of Namibia",
        "McDonald's history",
        "Academy Award winner",
        "capital of France",
        "Christopher Nolan films"
    ]
    
    for query in test_queries:
        print(f"\nTest query: '{query}'")
        try:
            results = baseline_agent.search_embeddings(query, top_k=3)
            print(f"Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"  [{i+1}] {result['title']} (score: {result['score']:.4f})")
                print(f"      {result['text'][:100]}...")
        except Exception as e:
            print(f"  Error: {e}")


def run_baseline_evaluation(questions_data, log_dir, corpus_path):
    """Run baseline embedding agent evaluation."""
    print(f"\n=== Running Baseline Embedding Evaluation ===")
    print(f"Corpus: {corpus_path}")
    
    # Initialize baseline agent
    print("Initializing baseline embedding agent...")
    baseline_agent = BaselineEmbeddingAgent(
        corpus_path=corpus_path,
        model_name="gpt-4o",
        generation_params={"temperature": 0.0},
        max_iterations=15,
        embedding_model="nvidia/NV-Embed-v2"
    )
    
    # Test search functionality first
    test_baseline_search(baseline_agent)
    
    # Extract questions and metadata
    questions = [item["question"] for item in questions_data]
    question_types = [item["id"].split("__")[0] for item in questions_data]
    question_ids = [item["id"] for item in questions_data]
    supporting_doc_ids = [item["id"].split("__")[1].split("_") for item in questions_data]
    question_decomposition = [item.get("question_decomposition", []) for item in questions_data]
    
    # Extract gold answers
    gold_answers_list = []
    for item in questions_data:
        if isinstance(item["answer"], str):
            gold_answers = [item["answer"]]
        else:
            gold_answers = item["answer"] if isinstance(item["answer"], list) else [str(item["answer"])]
        
        if "answer_aliases" in item:
            gold_answers.extend(item["answer_aliases"])
        
        gold_answers_list.append(gold_answers)
    
    print(f"\nProcessing {len(questions)} questions...")
    
    # Run baseline agent Q&A
    responses = baseline_agent.answer_batch(questions)
    predicted_answers = [r.answer for r in responses]
    
    # Save detailed results
    detailed_results = []
    for i, (response, question) in enumerate(zip(responses, questions)):
        detailed_results.append({
            "question_id": question_ids[i],
            "question_index": i,
            "question": question,
            "question_type": question_types[i],
            "predicted_answer": response.answer,
            "gold_answers": gold_answers_list[i],
            "supporting_doc_ids": supporting_doc_ids[i],
            "question_decomposition_gold": question_decomposition[i],
            "reasoning": response.reasoning,
            "tool_calls": response.tool_calls_made,
            "num_tool_calls": len(response.tool_calls_made),
            "approach": "baseline_embedding"
        })
    
    # Save detailed results to file
    results_file = os.path.join(log_dir, "baseline_embedding_results.json")
    with open(results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"💾 Saved detailed results to {results_file}")
    
    # Evaluate using HippoRAG methodology
    print("\nEvaluating with HippoRAG metrics...")
    overall_metrics, example_results = hipporag_eval.evaluate_qa_batch(
        gold_answers_list, predicted_answers
    )
    
    # Evaluate by question type
    overall_metrics_by_type, example_results_by_type = hipporag_eval.evaluate_qa_batch_w_types(
        question_types, gold_answers_list, predicted_answers
    )
    
    # Print results
    print("\n=== Baseline Embedding Evaluation Results ===")
    print(f"Total questions: {len(questions)}")
    print(f"Exact Match: {overall_metrics['ExactMatch']:.4f}")
    print(f"F1 Score: {overall_metrics['F1']:.4f}")
    print(f"Corpus documents: {len(baseline_agent.corpus)}")
    print("-" * 100)
    
    # Print results by question type
    for qtype, metrics in overall_metrics_by_type["by_type"].items():
        print(f"Question type: {qtype}")
        print(f"  Exact Match: {metrics['ExactMatch']:.4f}")
        print(f"  F1 Score: {metrics['F1']:.4f}")
        print(f"  Count: {metrics['count']}")
        print("-" * 50)
    
    # Show sample results
    print("\n=== Sample Results ===")
    for i in range(min(5, len(responses))):
        result = detailed_results[i]
        eval_result = example_results[i]
        
        em_status = "✅" if eval_result["ExactMatch"] == 1.0 else "❌"
        f1_status = "✅" if eval_result["F1"] > 0.5 else "⚠️" if eval_result["F1"] > 0.0 else "❌"
        
        print(f"\nQuestion {i + 1} ({result['question_type']}): {result['question']}")
        print(f"  Predicted: {result['predicted_answer']}")
        print(f"  Gold: {result['gold_answers']}")
        print(f"  EM: {eval_result['ExactMatch']:.3f} {em_status}")
        print(f"  F1: {eval_result['F1']:.3f} {f1_status}")
        print(f"  Tool calls: {result['num_tool_calls']}")
        
        # Show search queries used
        if result['tool_calls']:
            search_queries = [call['arguments']['query'] for call in result['tool_calls'] 
                            if call['tool'] == 'search_embeddings']
            if search_queries:
                quoted_queries = [f'"{q}"' for q in search_queries]
                ellipsis = "..." if len(search_queries) > 3 else ""
                print(f"  Searches: {', '.join(quoted_queries)}{ellipsis}")
    
    # Calculate additional metrics
    avg_tool_calls = sum(len(r.tool_calls_made) for r in responses) / len(responses)
    successful_questions = sum(1 for r in example_results if r['ExactMatch'] == 1.0)
    
    print(f"\n=== Additional Statistics ===")
    print(f"Average tool calls per question: {avg_tool_calls:.2f}")
    print(f"Questions with exact match: {successful_questions}/{len(questions)} ({100*successful_questions/len(questions):.1f}%)")
    
    # Analyze tool call patterns
    tool_call_counts = [len(r.tool_calls_made) for r in responses]
    max_calls = max(tool_call_counts) if tool_call_counts else 0
    min_calls = min(tool_call_counts) if tool_call_counts else 0
    print(f"Tool calls range: {min_calls} - {max_calls}")
    
    # Save summary metrics
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "approach": "baseline_embedding",
        "corpus_path": corpus_path,
        "embedding_model": "nvidia/NV-Embed-v2",
        "llm_model": "gpt-4o",
        "max_iterations": 15,
        "num_questions": len(questions),
        "corpus_size": len(baseline_agent.corpus),
        "overall_metrics": overall_metrics,
        "metrics_by_type": overall_metrics_by_type["by_type"],
        "statistics": {
            "avg_tool_calls": round(avg_tool_calls, 2),
            "successful_questions": successful_questions,
            "success_rate": round(successful_questions / len(questions), 4),
            "tool_call_range": [min_calls, max_calls]
        }
    }
    
    summary_file = os.path.join(log_dir, "summary_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"💾 Saved summary metrics to {summary_file}")
    
    return overall_metrics, detailed_results, summary_data


def analyze_failure_cases(detailed_results, log_dir):
    """Analyze questions where the baseline agent failed."""
    print("\n=== Analyzing Failure Cases ===")
    
    failure_cases = []
    partial_success = []
    
    for result in detailed_results:
        # Check if we have evaluation results
        if 'gold_answers' in result and 'predicted_answer' in result:
            # Simple check for exact match (case-insensitive)
            predicted = result['predicted_answer'].lower().strip()
            gold_matches = [ans.lower().strip() for ans in result['gold_answers']]
            
            is_exact_match = any(predicted == gold for gold in gold_matches)
            is_partial_match = any(gold in predicted or predicted in gold for gold in gold_matches)
            
            if not is_exact_match:
                if is_partial_match:
                    partial_success.append(result)
                else:
                    failure_cases.append(result)
    
    print(f"Complete failures: {len(failure_cases)}")
    print(f"Partial matches: {len(partial_success)}")
    
    # Analyze common patterns in failures
    if failure_cases:
        print(f"\n=== Sample Failure Cases ===")
        for i, case in enumerate(failure_cases[:5]):
            print(f"\nFailure {i+1}:")
            print(f"  Question: {case['question']}")
            print(f"  Type: {case['question_type']}")
            print(f"  Gold: {case['gold_answers']}")
            print(f"  Predicted: {case['predicted_answer']}")
            print(f"  Tool calls: {case['num_tool_calls']}")
            
            # Show what was searched for
            if case['tool_calls']:
                searches = [call['arguments']['query'] for call in case['tool_calls'] 
                          if call['tool'] == 'search_embeddings']
                if searches:
                    print(f"  Searches: {searches}")
    
    # Save failure analysis
    failure_analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(detailed_results),
        "complete_failures": len(failure_cases),
        "partial_matches": len(partial_success),
        "failure_cases": failure_cases[:10],  # Save top 10 for analysis
        "partial_cases": partial_success[:5]   # Save top 5 for analysis
    }
    
    failure_file = os.path.join(log_dir, "failure_analysis.json")
    with open(failure_file, "w") as f:
        json.dump(failure_analysis, f, indent=2)
    print(f"💾 Saved failure analysis to {failure_file}")


def main():
    """Run complete baseline evaluation pipeline."""
    print("🚀 Starting Baseline Embedding Musique Q&A Evaluation")
    print("Testing nvembed-v2 dense retrieval approach\n")
    
    try:
        # Setup logging
        log_dir, timestamp = setup_logging()
        
        # Load questions data
        questions_data = load_questions_data()
        
        # Determine corpus path
        corpus_path = CORPUS_PATH
        if not os.path.exists(corpus_path):
            # Try alternative paths
            alt_paths = [
                "/home/yigit/codebase/gsw-memory/musique_corpus_50_q.json",
                "musique_corpus.json",
                "musique_corpus_50_q.json"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    corpus_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Could not find corpus file. Tried: {CORPUS_PATH}, {alt_paths}")
        
        print(f"Using corpus: {corpus_path}")
        
        # Run baseline evaluation
        overall_metrics, detailed_results, summary_data = run_baseline_evaluation(
            questions_data, log_dir, corpus_path
        )
        
        # Analyze failure cases
        analyze_failure_cases(detailed_results, log_dir)
        
        # Create experiment configuration file
        config_data = {
            "experiment_type": "baseline_embedding",
            "timestamp": timestamp,
            "configuration": {
                "corpus_path": corpus_path,
                "questions_path": QUESTIONS_PATH,
                "embedding_model": "nvidia/NV-Embed-v2",
                "llm_model": "gpt-4o",
                "max_iterations": 15,
                "use_subset": USE_SUBSET,
                "subset_size": SUBSET_SIZE if USE_SUBSET else None,
                "top_k_retrieval": 5
            },
            "results": {
                "num_questions": len(questions_data),
                "exact_match": overall_metrics["ExactMatch"],
                "f1_score": overall_metrics["F1"]
            }
        }
        
        config_file = os.path.join(log_dir, "experiment_config.json")
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n✅ Baseline evaluation complete!")
        print(f"📊 Final Results:")
        print(f"   Questions: {len(questions_data)}")
        print(f"   Exact Match: {overall_metrics['ExactMatch']:.4f}")
        print(f"   F1 Score: {overall_metrics['F1']:.4f}")
        print(f"📁 Results saved to: {log_dir}")
        
        return {
            "metrics": overall_metrics,
            "results": detailed_results,
            "log_dir": log_dir,
            "config": config_data
        }
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()