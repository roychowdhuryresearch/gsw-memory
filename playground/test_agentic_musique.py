#!/usr/bin/env python3
"""
Test Agentic GSW Q&A on 2wiki MultiHopQA

This script tests the new agentic approach where the LLM can dynamically
explore the GSW structure using tools, comparing it to the traditional
pre-computed summary approach.
"""

import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from gsw_memory.qa.gsw_tools import GSWTools
from gsw_memory.qa.agentic_agent import AgenticAnsweringAgent
from gsw_memory import hipporag_eval
from gsw_memory.utils.loaders import load_from_logs
from gsw_memory import EntitySummaryAggregator

# Load environment variables
load_dotenv()

# Configuration
LOAD_BASE_LOGS = "/home/yigit/codebase/gsw-memory/logs/full_2wiki_corpus_20250716_165147"  # Existing logs with reconciled GSW
USE_SUBSET = True # Test on subset of questions
TEST_MODE = "multi_file"  # Options: "reconciled", "multi_file", "both"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def setup_logging():
    """Create timestamped log directory for agentic results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__), "..", "logs", f"agentic_2wiki_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 Created log directory: {log_dir}")
    return log_dir, timestamp


def load_existing_data():
    """Load reconciled GSW and questions from existing logs."""
    print(f"\n=== Loading Data from {LOAD_BASE_LOGS} ===")
    
    # Load reconciled GSW
    loaded_data = load_from_logs(LOAD_BASE_LOGS)
    reconciled_gsw = loaded_data["reconciled_gsw"]
    
    if reconciled_gsw:
        print(f"Loaded GSW with:")
        print(f"  Entities: {len(reconciled_gsw.entity_nodes)}")
        print(f"  Verb phrases: {len(reconciled_gsw.verb_phrase_nodes)}")
    
    # Load questions
    questions_path = "/home/yigit/codebase/gsw-memory/musique.json"
    with open(questions_path, "r") as f:
        questions_data = json.load(f)
    
    if USE_SUBSET:
        questions_data = questions_data[:50]
        print(f"Using subset: {len(questions_data)} questions")
    
    return reconciled_gsw, questions_data

def generate_entity_summaries(reconciled_gsw, log_dirs):
    """Generate entity summaries for factual Q&A."""
    print("\n=== Generating Entity Summaries ===")

    llm_config = {
        "model_name": "gpt-4o",
        "generation_params": {
            "temperature": 0.0,
            "max_tokens": 500,
        },  # Shorter for facts
    }

    aggregator = EntitySummaryAggregator(reconciled_gsw, llm_config)
    summaries = aggregator.precompute_summaries(include_space_time=True)

    print(f"Generated summaries for {len(summaries)} entities")

    # Save all generated summaries for debugging
    summaries_file = os.path.join(log_dirs["results_dir"], "all_entity_summaries.json")
    with open(summaries_file, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"💾 Saved all entity summaries to {summaries_file}")

    return aggregator


def test_gsw_tools(gsw_tools):
    """Quick test of GSW tools functionality."""
    print("\n=== Testing GSW Tools ===")
    
    # Test search
    print("\nTest 1: Search for 'Academy Award'")
    results = gsw_tools.search_gsw("Academy Award", limit=3)
    print(f"Found {len(results)} results")
    for r in results[:2]:
        if r["type"] == "question":
            print(f"  Q: {r['question_text'][:80]}...")
        else:
            print(f"  E: {r['entity_name']}")
    
    # Test entity context (if we found any entities)
    entity_results = [r for r in results if r["type"] == "entity"]
    if entity_results:
        entity_id = entity_results[0]["entity_id"]
        print(f"\nTest 2: Get context for entity '{entity_results[0]['entity_name']}'")
        context = gsw_tools.get_entity_context(entity_id)
        if "error" not in context:
            print(f"  Participates in {len(context['questions'])} questions")
            if context['questions']:
                print(f"  Example: {context['questions'][0]['question_text'][:80]}...")
                
    # Test get_entity_context_same_name
    print("\nTest 3: Get context for entities with the same name")
    entity_ids = [entity_results[0]["entity_id"], entity_results[1]["entity_id"]]
    context = gsw_tools.get_multiple_relevant_entity_contexts(entity_ids)
    if "error" not in context:
        print(f"  Context for {len(context)} entities")
        for c in context:
            print(c)
            
    # Test search_gsw_embeddings_of_entity_summaries
    print("\nTest 4: Search for 'Academy Award' using embeddings")
    results = gsw_tools.search_gsw_embeddings_of_entity_summaries("Academy Award", top_k=3)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  E: {r}")


def run_agentic_evaluation_reconciled(questions_data, log_dir):
    """Run agentic Q&A evaluation using reconciled GSW."""
    print("\n=== Running Agentic Q&A Evaluation (Reconciled) ===")
    
    # Initialize tools and agent with reconciled file
    reconciled_file_path = os.path.join(LOAD_BASE_LOGS, "reconciled_output", "reconciled", "global_reconciled.json")
    gsw_tools = GSWTools(reconciled_file_path)
    
    # Build search index explicitly for better performance
    gsw_tools.build_index()
    agent = AgenticAnsweringAgent(
        model_name="gpt-4o-mini",
        generation_params={"temperature": 0.0},
        max_iterations=10
    )
    
    # Create tool dictionary for agent
    tools = {
        # "search_gsw": gsw_tools.search_gsw,
        # "search_gsw_bm25_entity_name": gsw_tools.search_gsw_bm25_entity_name,
        "search_gsw_embeddings_of_entity_summaries": gsw_tools.search_gsw_embeddings_of_entity_summaries,
        # "get_entity_context": gsw_tools.get_entity_context
    }
    
    # Test tools first
    test_gsw_tools(gsw_tools)
    
    # Extract questions and gold answers
    questions = [item["question"] for item in questions_data]
    question_types = [item["type"] for item in questions_data]
    
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
    
    # Run agentic Q&A
    responses = agent.answer_batch(questions, tools)
    predicted_answers = [r.answer for r in responses]
    
    # Save detailed results
    detailed_results = []
    for i, (response, question) in enumerate(zip(responses, questions)):
        detailed_results.append({
            "question_id": i,
            "question": question,
            "question_type": question_types[i],
            "predicted_answer": response.answer,
            "gold_answers": gold_answers_list[i],
            "reasoning": response.reasoning,
            "tool_calls": response.tool_calls_made,
            "num_tool_calls": len(response.tool_calls_made)
        })
    
    # Save results
    results_file = os.path.join(log_dir, "agentic_results.json")
    with open(results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"💾 Saved detailed results to {results_file}")
    
    # Evaluate using HippoRAG methodology
    overall_metrics, example_results = hipporag_eval.evaluate_qa_batch(
        gold_answers_list, predicted_answers
    )
    
    print("\n=== Agentic Evaluation Results ===")
    print(f"Total questions: {len(questions)}")
    print(f"Exact Match: {overall_metrics['ExactMatch']:.4f}")
    print(f"F1 Score: {overall_metrics['F1']:.4f}")
    
    # Break down by question type
    type_metrics = {}
    for i, result in enumerate(example_results):
        qtype = question_types[i]
        if qtype not in type_metrics:
            type_metrics[qtype] = {"em_scores": [], "f1_scores": []}
        type_metrics[qtype]["em_scores"].append(result["ExactMatch"])
        type_metrics[qtype]["f1_scores"].append(result["F1"])
    
    print("\nMetrics by question type:")
    for qtype, metrics in type_metrics.items():
        avg_em = sum(metrics["em_scores"]) / len(metrics["em_scores"])
        avg_f1 = sum(metrics["f1_scores"]) / len(metrics["f1_scores"])
        count = len(metrics["em_scores"])
        print(f"  {qtype} ({count} questions):")
        print(f"    EM: {avg_em:.4f}")
        print(f"    F1: {avg_f1:.4f}")
    
    # Show sample results
    print("\n=== Sample Results ===")
    for i in range(min(3, len(responses))):
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
        
        # Show first tool call as example
        if result['tool_calls']:
            first_call = result['tool_calls'][0]
            print(f"  First tool: {first_call['tool']}({first_call['arguments']})")
    
    # Save summary
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "approach": "agentic",
        "num_questions": len(questions),
        "overall_metrics": overall_metrics,
        "question_type_breakdown": {}
    }
    
    for qtype, metrics in type_metrics.items():
        avg_em = sum(metrics["em_scores"]) / len(metrics["em_scores"])
        avg_f1 = sum(metrics["f1_scores"]) / len(metrics["f1_scores"])
        summary_data["question_type_breakdown"][qtype] = {
            "count": len(metrics["em_scores"]),
            "exact_match": round(avg_em, 4),
            "f1_score": round(avg_f1, 4)
        }
    
    summary_file = os.path.join(log_dir, "summary_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    return overall_metrics, detailed_results


def run_agentic_evaluation_multi_file(questions_data, log_dir):
    """Run agentic Q&A evaluation using multiple individual GSW files (no reconciliation)."""
    print("\n=== Running Agentic Q&A Evaluation (Multi-File) ===")
    
    # Get all individual GSW files with global IDs
    import glob
    # gsw_files_pattern = os.path.join(LOAD_BASE_LOGS, "gsw_output_global_ids", "networks", "doc_*", "gsw_*_0.json")
    gsw_files_pattern = os.path.join(LOAD_BASE_LOGS, "gsw_output", "batch_*", "networks","doc_*" ,"gsw_*_0.json")
    # Sort files naturally so that we get 0, 1, 2 instead of 0, 1, 10
    gsw_files = sorted(glob.glob(gsw_files_pattern), key=natural_sort_key)[:932]   #[:932] or 198
    
    # if USE_SUBSET:
        # Use only first 10 files for testing
    gsw_files = gsw_files
    print(f"Using subset: {len(gsw_files)} GSW files")
    
    print(f"Loading {len(gsw_files)} individual GSW files...")
    
    # Initialize tools and agent with multiple files
    gsw_tools = GSWTools(gsw_files)
    gsw_tools.generate_entity_summaries()
    
    # Build search index explicitly (this may take time with many files)
    gsw_tools.build_entity_index()
    gsw_tools.build_entity_summary_index()
    gsw_tools._build_embeddings()
    agent = AgenticAnsweringAgent(
        model_name="gpt-4o",
        generation_params={"temperature": 0.0},
        max_iterations=15
    )
    
    # Create tool dictionary for agent
    tools = {
        # "search_gsw": gsw_tools.search_gsw,
        "search_gsw_bm25_entity_name": gsw_tools.search_gsw_bm25_entity_name,
        "search_gsw_embeddings_of_entity_summaries": gsw_tools.search_gsw_embeddings_of_entity_summaries,
        "get_multiple_relevant_entity_contexts": gsw_tools.get_multiple_relevant_entity_contexts
    }
    
    # Test tools first
    test_gsw_tools(gsw_tools)
    
    # Extract questions and gold answers
    questions = [item["question"] for item in questions_data]
    question_types = [item["id"].split("__")[0] for item in questions_data]
    supporting_doc_ids = [item["id"].split("__")[1].split("_") for item in questions_data]
    question_decomposition = [item["question_decomposition"] for item in questions_data]
    
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
    
    # Run agentic Q&A
    responses = agent.answer_batch(questions, tools)
    predicted_answers = [r.answer for r in responses]
    
    # Save detailed results
    detailed_results = []
    for i, (response, question) in enumerate(zip(responses, questions)):
        detailed_results.append({
            "question_id": i,
            "question": question,
            "question_type": question_types[i],
            "predicted_answer": response.answer,
            "gold_answers": gold_answers_list[i],
            "supporting_doc_ids": supporting_doc_ids[i],
            "question_decomposition_gold": question_decomposition[i],
            "reasoning": response.reasoning,
            "tool_calls": response.tool_calls_made,
            "num_tool_calls": len(response.tool_calls_made),
            "approach": "multi_file"
        })
    
    # Save results
    results_file = os.path.join(log_dir, "agentic_multi_file_results.json")
    with open(results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"💾 Saved detailed results to {results_file}")
    
    # Evaluate using HippoRAG methodology
    from gsw_memory import hipporag_eval
    overall_metrics, example_results = hipporag_eval.evaluate_qa_batch(
        gold_answers_list, predicted_answers
    )
    overall_metrics_by_type, example_results_by_type = hipporag_eval.evaluate_qa_batch_w_types(question_types,
    gold_answers_list, predicted_answers
    )
    
    print("\n=== Multi-File Agentic Evaluation Results ===")
    print(f"Total questions: {len(questions)}")
    print(f"Exact Match: {overall_metrics['ExactMatch']:.4f}")
    print(f"F1 Score: {overall_metrics['F1']:.4f}")
    print(f"GSW files used: {len(gsw_files)}")
    print("-"*100)
    for qtype, metrics in overall_metrics_by_type["by_type"].items():
        print(f"Question type: {qtype}")
        print(f"Exact Match: {metrics['ExactMatch']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        print(f"Count: {metrics['count']}")
        print("-"*100)
    
    return overall_metrics, detailed_results


def compare_with_traditional(agentic_metrics, log_dir):
    """Compare agentic results with traditional approach."""
    print("\n=== Comparison with Traditional Approach ===")
    
    # Load traditional results if available
    traditional_results_path = os.path.join(LOAD_BASE_LOGS, "results", "summary_metrics.json")
    if os.path.exists(traditional_results_path):
        with open(traditional_results_path, "r") as f:
            traditional_data = json.load(f)
        
        trad_metrics = traditional_data["overall_metrics"]
        
        print(f"\nTraditional Approach:")
        print(f"  Exact Match: {trad_metrics['ExactMatch']:.4f}")
        print(f"  F1 Score: {trad_metrics['F1']:.4f}")
        
        print(f"\nAgentic Approach:")
        print(f"  Exact Match: {agentic_metrics['ExactMatch']:.4f}")
        print(f"  F1 Score: {agentic_metrics['F1']:.4f}")
        
        print(f"\nDifference:")
        em_diff = agentic_metrics['ExactMatch'] - trad_metrics['ExactMatch']
        f1_diff = agentic_metrics['F1'] - trad_metrics['F1']
        print(f"  EM: {em_diff:+.4f} ({'better' if em_diff > 0 else 'worse'})")
        print(f"  F1: {f1_diff:+.4f} ({'better' if f1_diff > 0 else 'worse'})")
        
        # Save comparison
        comparison = {
            "traditional": trad_metrics,
            "agentic": agentic_metrics,
            "improvement": {
                "exact_match": round(em_diff, 4),
                "f1_score": round(f1_diff, 4)
            }
        }
        
        comparison_file = os.path.join(log_dir, "comparison.json")
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
    else:
        print("Traditional results not found for comparison")


def main():
    """Run complete agentic evaluation pipeline."""
    print("🚀 Starting Agentic 2wiki Q&A Evaluation")
    print(f"Testing dynamic GSW exploration - Mode: {TEST_MODE}\n")
    
    try:
        # Setup logging
        log_dir, timestamp = setup_logging()
        
        # Load questions data
        reconciled_gsw, questions_data = load_existing_data()
        
        results = {}
        
        if TEST_MODE == "reconciled" or TEST_MODE == "both":
            # Run reconciled evaluation
            agentic_metrics, detailed_results = run_agentic_evaluation_reconciled(
                questions_data, log_dir
            )
            results["reconciled"] = {
                "metrics": agentic_metrics,
                "results": detailed_results
            }
            
            # Compare with traditional approach
            compare_with_traditional(agentic_metrics, log_dir)
        
        if TEST_MODE == "multi_file" or TEST_MODE == "both":
            # Run multi-file evaluation
            multi_metrics, multi_results = run_agentic_evaluation_multi_file(
                questions_data, log_dir
            )
            results["multi_file"] = {
                "metrics": multi_metrics,
                "results": multi_results
            }
        
        if TEST_MODE == "both":
            # Compare reconciled vs multi-file
            print("\n=== Reconciled vs Multi-File Comparison ===")
            reconciled_em = results["reconciled"]["metrics"]["ExactMatch"]
            multi_em = results["multi_file"]["metrics"]["ExactMatch"]
            reconciled_f1 = results["reconciled"]["metrics"]["F1"]
            multi_f1 = results["multi_file"]["metrics"]["F1"]
            
            print(f"Reconciled - EM: {reconciled_em:.4f}, F1: {reconciled_f1:.4f}")
            print(f"Multi-File  - EM: {multi_em:.4f}, F1: {multi_f1:.4f}")
            print(f"Difference  - EM: {multi_em - reconciled_em:+.4f}, F1: {multi_f1 - reconciled_f1:+.4f}")
        
        print(f"\n✅ Evaluation complete! Results saved to: {log_dir}")
        
        return {
            "results": results,
            "log_dir": log_dir
        }
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
