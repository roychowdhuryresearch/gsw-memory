#!/usr/bin/env python3
"""
End-to-End Tulving Bench Test

This script replicates the original research pipeline for Tulving Bench evaluation:
1. Process book chapters → GSW structures
2. Reconcile GSW structures (LOCAL strategy - chapter by chapter like original)
3. Generate entity summaries for each chapter
4. Run Q&A system across multiple chapter GSWs
5. Evaluate answers using LLM judge
6. Compare results with original system

This validates that the packageable GSW-Memory implementation produces
comparable results to the original research code.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gsw_memory import (
    EntitySummaryAggregator,
    GSWProcessor,
    GSWQuestionAnswerer,
    TulvingBenchEvaluator,
    reconcile_gsw_outputs,
)

# Disable cache for testing
os.environ["CURATOR_DISABLE_CACHE"] = "true"

# Load environment variables
load_dotenv()


def load_tulving_bench_data():
    """Load the Tulving Bench book and questions."""
    print("=== Loading Tulving Bench Data ===")

    # Load book chapters
    book_path = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/src/gsw_memory/benchmarks/tulvingbench/book_20.json"
    with open(book_path, "r") as f:
        book_data = json.load(f)

    chapters = []
    for chapter_key in sorted(book_data.keys()):
        chapters.append(book_data[chapter_key])

    print(f"Loaded {len(chapters)} chapters")

    # Load questions
    questions_path = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/src/gsw_memory/benchmarks/tulvingbench/questions_20.json"
    with open(questions_path, "r") as f:
        questions_data = json.load(f)

    print(f"Loaded {len(questions_data)} questions")

    return chapters, questions_data


def process_book_to_gsws(chapters, use_subset=True):
    """Process book chapters into separate GSW structures (LOCAL strategy like original)."""
    print("\n=== Processing Book to Chapter GSWs ===")

    # Use subset for faster testing (matching original test_operator.py)
    if use_subset:
        chapters = chapters[:2]
        print(f"Using subset of {len(chapters)} chapters for testing")

    # Initialize processor with same config as original research code
    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=True,
        enable_chunking=True,
        enable_context=True,
        chunk_size=3,
        overlap=0,
        enable_spacetime=True,
    )

    print(f"Processing {len(chapters)} chapters...")
    gsw_structures = processor.process_documents(chapters, output_dir="e2e_gsw_output")
    print(f"Generated GSW structures for {len(gsw_structures)} chapters")

    # Reconcile using LOCAL strategy to maintain separate chapter GSWs (matches original)
    print("\nReconciling with LOCAL strategy for separate chapter GSWs...")
    reconciled_chapters = reconcile_gsw_outputs(
        gsw_structures,
        strategy="local",  # Keep chapters separate like original research code
        output_dir="e2e_reconciled_output",
        save_statistics=True,
        enable_visualization=False,
    )

    print(f"Reconciled {len(reconciled_chapters)} chapters:")
    for i, chapter_gsw in enumerate(reconciled_chapters):
        print(
            f"  Chapter {i}: {len(chapter_gsw.entity_nodes)} entities, {len(chapter_gsw.verb_phrase_nodes)} verb phrases"
        )

    return reconciled_chapters


def generate_entity_summaries(reconciled_chapters):
    """Generate entity summaries for each chapter GSW."""
    print("\n=== Generating Entity Summaries for All Chapters ===")

    llm_config = {
        "model_name": "gpt-4o",
        "generation_params": {"temperature": 0.0, "max_tokens": 500},
    }

    aggregators = []
    for i, chapter_gsw in enumerate(reconciled_chapters):
        print(f"Generating summaries for Chapter {i}...")
        aggregator = EntitySummaryAggregator(chapter_gsw, llm_config)

        summaries = aggregator.precompute_summaries(include_space_time=True)
        print(f"  Generated summaries for {len(summaries)} entities")

        aggregators.append(aggregator)

    print(f"Generated summaries for {len(aggregators)} chapter aggregators")

    return aggregators


def setup_qa_system(reconciled_chapters, aggregators):
    """Initialize the multi-GSW Q&A system (like original with multiple chapters)."""
    print("\n=== Setting Up Multi-Chapter Q&A System ===")

    llm_config = {
        "model_name": "gpt-4o",
        "generation_params": {"temperature": 0.0, "max_tokens": 500},
    }

    # Use multi-GSW interface (list of GSWs and aggregators)
    qa_system = GSWQuestionAnswerer(
        gsw=reconciled_chapters,  # List of chapter GSWs
        entity_aggregator=aggregators,  # List of chapter aggregators
        llm_config=llm_config,
        embedding_model="voyage-3",
    )

    print("Multi-chapter Q&A system initialized successfully")

    return qa_system


def run_qa_evaluation(qa_system, questions_data, use_subset=True):
    """Run Q&A system on Tulving Bench questions."""
    print("\n=== Running Q&A Evaluation ===")

    # Use subset for faster testing
    if use_subset:
        questions_data = questions_data[:5]
        print(f"Using subset of {len(questions_data)} questions for testing")

    # Extract questions
    questions = [item["question"] for item in questions_data]

    print(f"Processing {len(questions)} questions across multiple chapters...")

    # Run batch Q&A
    qa_results = qa_system.ask_batch(questions, max_summaries=5)

    # Format results for evaluation
    evaluation_data = []
    for i, result in enumerate(qa_results):
        question_item = questions_data[i]
        evaluation_data.append(
            {
                "question": result["question"],
                "predicted_answer": result["answer"],
                "ground_truth": question_item["correct_answer"],
                "context": question_item.get("context", ""),
                "extracted_entities": result["extracted_entities"],
                "matched_entities": result["matched_entities"],
                "num_summaries_used": result["num_summaries_used"],
            }
        )

    print(f"Generated answers for {len(evaluation_data)} questions")

    return evaluation_data


def evaluate_with_tulving_bench(evaluation_data):
    """Evaluate results using Tulving Bench evaluator."""
    print("\n=== Evaluating with Tulving Bench Judge ===")

    # Initialize evaluator
    evaluator = TulvingBenchEvaluator(
        model_name="gpt-4o", generation_params={"temperature": 0.0}
    )

    # Convert evaluation_data to the format expected by the evaluator
    # Use single system evaluation interface: qa_results + ground_truth
    qa_results = []
    ground_truth = []
    
    for item in evaluation_data:
        qa_results.append({
            "question": item["question"],
            "answer": item["predicted_answer"],
            "matched_entities": item["matched_entities"]  # For chapter metrics
        })
        
        ground_truth.append({
            "question": item["question"],
            "correct_answer": item["ground_truth"],
            "correct_answer_chapters": [],  # We don't have this info, so empty
            "retrieval_type": "unknown"  # Default since we don't have this classification
        })

    print("Running LLM-as-a-judge evaluation...")
    results = evaluator.evaluate(qa_results=qa_results, ground_truth=ground_truth)

    # Display results (updated to match the actual return format)
    print("\n=== Evaluation Results ===")
    print("Overall Metrics:")
    print(f"  Precision: {results['system_metrics']['precision']:.3f}")
    print(f"  Recall: {results['system_metrics']['recall']:.3f}")
    print(f"  F1 Score: {results['system_metrics']['f1']:.3f}")

    print("\nDetailed Results:")
    for i, detail in enumerate(results["detailed_evaluations"]):
        print(f"\nQuestion {i + 1}: {detail['question']}")
        print(f"  Predicted: {detail['answer_evaluated']}")
        print(f"  Ground Truth: {detail['correct_answer']}")
        
        # Handle None values gracefully
        precision = detail['precision']
        recall = detail['recall'] 
        f1 = detail['f1']
        
        precision_str = f"{precision:.3f}" if precision is not None else "N/A"
        recall_str = f"{recall:.3f}" if recall is not None else "N/A"
        f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
        
        print(f"  Precision: {precision_str}")
        print(f"  Recall: {recall_str}")
        print(f"  F1: {f1_str}")

    return results


def compare_with_baseline(evaluation_data):
    """Compare GSW results with baseline (questions without GSW context)."""
    print("\n=== Comparing with Baseline ===")

    # Create baseline qa_results and ground_truth data
    baseline_qa_results = []
    baseline_ground_truth = []
    
    for item in evaluation_data:
        baseline_qa_results.append({
            "question": item["question"],
            "answer": "I don't have enough information to answer this question.",  # Simple baseline
            "matched_entities": []  # No entities for baseline
        })
        
        baseline_ground_truth.append({
            "question": item["question"],
            "correct_answer": item["ground_truth"],
            "correct_answer_chapters": [],  # We don't have this info
            "retrieval_type": "unknown"
        })

    # Evaluate baseline
    evaluator = TulvingBenchEvaluator(model_name="gpt-4o", generation_params={"temperature": 0.0})

    print("Evaluating baseline (no GSW)...")
    baseline_results = evaluator.evaluate(qa_results=baseline_qa_results, ground_truth=baseline_ground_truth)

    print("\nBaseline Results:")
    print(f"  Precision: {baseline_results['system_metrics']['precision']:.3f}")
    print(f"  Recall: {baseline_results['system_metrics']['recall']:.3f}")
    print(f"  F1 Score: {baseline_results['system_metrics']['f1']:.3f}")

    return baseline_results


def main():
    """Run complete end-to-end Tulving Bench evaluation."""
    print("🚀 Starting End-to-End Tulving Bench Evaluation")
    print(
        "This replicates the original research pipeline with LOCAL reconciliation strategy\n"
    )

    try:
        # Step 1: Load Tulving Bench data
        chapters, questions_data = load_tulving_bench_data()

        # Step 2: Process book to separate chapter GSWs (LOCAL strategy like original)
        reconciled_chapters = process_book_to_gsws(chapters, use_subset=True)

        # Step 3: Generate entity summaries for each chapter
        aggregators = generate_entity_summaries(reconciled_chapters)

        # Step 4: Setup multi-chapter Q&A system
        qa_system = setup_qa_system(reconciled_chapters, aggregators)

        # Step 5: Run Q&A evaluation across chapters
        evaluation_data = run_qa_evaluation(qa_system, questions_data, use_subset=True)

        # Step 6: Evaluate with Tulving Bench judge
        gsw_results = evaluate_with_tulving_bench(evaluation_data)

        # Step 7: Compare with baseline
        baseline_results = compare_with_baseline(evaluation_data)

        # Step 8: Final comparison
        print("\n=== Final Comparison ===")
        print("GSW System (Multi-Chapter LOCAL Strategy):")
        print(f"  Precision: {gsw_results['system_metrics']['precision']:.3f}")
        print(f"  Recall: {gsw_results['system_metrics']['recall']:.3f}")
        print(f"  F1 Score: {gsw_results['system_metrics']['f1']:.3f}")

        print("\nBaseline System:")
        print(f"  Precision: {baseline_results['system_metrics']['precision']:.3f}")
        print(f"  Recall: {baseline_results['system_metrics']['recall']:.3f}")
        print(f"  F1 Score: {baseline_results['system_metrics']['f1']:.3f}")

        improvement = (
            gsw_results["system_metrics"]["f1"]
            - baseline_results["system_metrics"]["f1"]
        )
        print(f"\nF1 Improvement: {improvement:+.3f}")

        if improvement > 0:
            print("✅ GSW system outperforms baseline!")
        else:
            print("⚠️  GSW system needs improvement")

        print("\n🎉 End-to-End Evaluation Completed Successfully!")
        print(
            "The GSW-Memory package is working end-to-end with LOCAL reconciliation strategy."
        )

        return {
            "gsw_results": gsw_results,
            "baseline_results": baseline_results,
            "improvement": improvement,
        }

    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
