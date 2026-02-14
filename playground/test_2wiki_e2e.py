#!/usr/bin/env python3
"""
End-to-End 2wiki MultiHopQA Test

This script adapts the GSW-Memory pipeline for 2wiki MultiHopQA evaluation:
1. Load 2wiki corpus documents and questions
2. Process corpus → GSW structures
3. Reconcile GSW structures (GLOBAL strategy for cross-document reasoning)
4. Generate entity summaries
5. Run Q&A system on 2wiki questions
6. Evaluate accuracy and reasoning quality

This validates GSW-Memory performance on fact-based multi-hop reasoning.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gsw_memory import (
    EntitySummaryAggregator,
    GSWProcessor,
    GSWQuestionAnswerer,
    hipporag_eval,
    reconcile_gsw_outputs,
)
from gsw_memory.prompts.operator_prompts import PromptType
from gsw_memory.utils.loaders import load_from_logs

# Disable cache for testing
os.environ["CURATOR_DISABLE_CACHE"] = "true"

# Load environment variables
load_dotenv()

# ===== CONFIGURATION: Granular Stage Control =====
# Control which stages to regenerate vs load from existing logs
# Dependencies: operator → reconciler → aggregator
# If you regenerate an earlier stage, all downstream stages will also regenerate

# LOAD_BASE_LOGS = None  # Base directory for loading existing data
LOAD_BASE_LOGS = "../logs/2wiki_eval_20250707_170220"  # Uncomment and set path

REGENERATE_FROM = "aggregator"  # Which stage to start regenerating from
# Options:
#   None - Load all stages (operator, reconciler, aggregator), just run Q&A
#   "operator" - Regenerate operator + reconciler + aggregator
#   "reconciler" - Load operator, regenerate reconciler + aggregator
#   "aggregator" - Load operator + reconciler, regenerate aggregator

# Examples:
# REGENERATE_FROM = "aggregator"  # Test new summary logic, load everything else
# REGENERATE_FROM = "reconciler"  # Test new reconciler logic, load operator only
# REGENERATE_FROM = None          # Just test Q&A changes, load all preprocessing


def setup_timestamped_logging():
    """Create timestamped logs directory for all outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__), "..", "logs", f"2wiki_eval_{timestamp}"
    )

    # Create logs directory structure
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "gsw_output"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "reconciled_output"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "results"), exist_ok=True)

    print(f"📁 Created timestamped log directory: {log_dir}")

    return {
        "base_dir": log_dir,
        "gsw_output_dir": os.path.join(log_dir, "gsw_output"),
        "reconciled_output_dir": os.path.join(log_dir, "reconciled_output"),
        "results_dir": os.path.join(log_dir, "results"),
        "timestamp": timestamp,
    }


def load_2wiki_data(use_subset=True):
    """Load 2wiki questions and extract relevant context documents."""
    print("=== Loading 2wiki Data ===")

    # Load full corpus to create title->text mapping
    corpus_path = "/home/shreyas/NLP/SM/gensemworkspaces/HippoRAG/reproduce/dataset/2wikimultihopqa_corpus.json"
    with open(corpus_path, "r") as f:
        corpus_data = json.load(f)

    # Create mapping from document title to text
    corpus_map = {}
    for doc in corpus_data:
        corpus_map[doc["title"]] = doc["text"]

    print(f"Loaded {len(corpus_map)} total corpus documents")

    # Load questions
    questions_path = "/home/shreyas/NLP/SM/gensemworkspaces/HippoRAG/reproduce/dataset/2wikimultihopqa.json"
    with open(questions_path, "r") as f:
        questions_data = json.load(f)

    # Use subset for faster testing
    if use_subset:
        questions_data = questions_data[:10]  # Few questions for initial test
        print(f"Using subset: {len(questions_data)} questions")

    # Extract relevant documents for our questions
    relevant_docs = set()
    for question in questions_data:
        # Each question has a "context" field with relevant documents
        for doc_title, doc_content in question["context"]:
            relevant_docs.add(doc_title)

    # Get texts for relevant documents
    documents = []
    document_titles = []
    for doc_title in relevant_docs:
        if doc_title in corpus_map:
            documents.append(corpus_map[doc_title])
            document_titles.append(doc_title)
        else:
            print(f"Warning: Document '{doc_title}' not found in corpus")

    print(
        f"Extracted {len(documents)} relevant documents for {len(questions_data)} questions"
    )

    return documents, questions_data, document_titles


def process_corpus_to_gsws(documents, log_dirs, use_subset=True):
    """Process 2wiki corpus into GSW structures with GLOBAL reconciliation."""
    print("\n=== Processing 2wiki Corpus to GSWs ===")

    # Initialize processor optimized for factual content
    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=False,
        enable_chunking=False,
        chunk_size=1,  # Smaller chunks for factual content
        overlap=0,
        enable_context=False,
        enable_spacetime=True,
        prompt_type=PromptType.FACTUAL,  # Use factual extraction prompts for 2wiki
    )

    print(f"Processing {len(documents)} documents...")
    gsw_structures = processor.process_documents(
        documents, output_dir=log_dirs["gsw_output_dir"]
    )
    print(f"Generated GSW structures for {len(gsw_structures)} document chunks")

    # Reconcile using GLOBAL strategy for cross-document entity linking
    print("\nReconciling with GLOBAL strategy for cross-document reasoning...")
    reconciled_gsw = reconcile_gsw_outputs(
        gsw_structures,
        strategy="global",  # Cross-document linking for factual QA
        output_dir=log_dirs["reconciled_output_dir"],
        save_statistics=True,
        enable_visualization=False,
    )

    print("Reconciled into unified GSW:")
    print(f"  Entities: {len(reconciled_gsw.entity_nodes)}")
    print(f"  Verb phrases: {len(reconciled_gsw.verb_phrase_nodes)}")

    return reconciled_gsw


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


def setup_qa_system(reconciled_gsw, aggregator):
    """Initialize Q&A system for factual multi-hop reasoning."""
    print("\n=== Setting Up Q&A System ===")

    llm_config = {
        "model_name": "gpt-4o",
        "generation_params": {
            "temperature": 0.0,
            "max_tokens": 200,
        },  # Concise factual answers
    }

    qa_system = GSWQuestionAnswerer(
        gsw=reconciled_gsw,
        entity_aggregator=aggregator,
        llm_config=llm_config,
        embedding_model="voyage-3",
    )

    print("Q&A system initialized for 2wiki evaluation")

    return qa_system


def run_2wiki_evaluation(qa_system, questions_data, log_dirs, use_subset=True):
    """Run Q&A evaluation on 2wiki questions using HippoRAG methodology."""
    print("\n=== Running 2wiki Q&A Evaluation ===")

    # Extract questions and answers
    questions = [item["question"] for item in questions_data]
    question_types = [item["type"] for item in questions_data]

    # Prepare gold answers (2wiki has single answers, but we format as list for compatibility)
    gold_answers_list = []
    for item in questions_data:
        # Handle both string answers and potential aliases
        if isinstance(item["answer"], str):
            gold_answers = [item["answer"]]
        else:
            gold_answers = (
                item["answer"]
                if isinstance(item["answer"], list)
                else [str(item["answer"])]
            )

        # Add aliases if present (some datasets have these)
        if "answer_aliases" in item:
            gold_answers.extend(item["answer_aliases"])

        gold_answers_list.append(gold_answers)

    print(f"Processing {len(questions)} questions...")

    # Run batch Q&A
    qa_results = qa_system.ask_batch(questions, max_summaries=5, include_connected=True)
    predicted_answers = [result["answer"].strip() for result in qa_results]

    # Save Q&A debugging information
    qa_debug_info = []
    for i, result in enumerate(qa_results):
        debug_item = {
            "question_id": i,
            "question": result["question"],
            "predicted_answer": result["answer"],
            "gold_answers": gold_answers_list[i],
            "question_type": question_types[i],
            "extracted_entities": result["extracted_entities"],
            "matched_entities": result["matched_entities"],
            "num_summaries_used": result["num_summaries_used"],
            "ranked_summaries": result.get(
                "ranked_summaries", []
            ),  # The ranked entity summaries
            "context_to_answering_agent": result.get(
                "context_to_answering_agent", ""
            ),  # The context passed to LLM
            "reasoning": result.get(
                "reasoning", "No reasoning captured"
            ),  # The LLM's reasoning for the answer
        }
        qa_debug_info.append(debug_item)

    # Save Q&A debug information
    qa_debug_file = os.path.join(log_dirs["results_dir"], "qa_debug_info.json")
    with open(qa_debug_file, "w") as f:
        json.dump(qa_debug_info, f, indent=2)
    print(f"💾 Saved Q&A debug info to {qa_debug_file}")

    # Evaluate using HippoRAG methodology
    overall_metrics, example_results = hipporag_eval.evaluate_qa_batch(
        gold_answers_list, predicted_answers
    )

    print("\n=== 2wiki Evaluation Results (HippoRAG-Style) ===")
    print(f"Total questions: {len(questions)}")
    print(f"Exact Match: {overall_metrics['ExactMatch']:.4f}")
    print(f"F1 Score: {overall_metrics['F1']:.4f}")

    # Break down by question type using EM score
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

    # Show detailed examples
    print("\n=== Sample Results ===")
    for i, (qa_result, eval_result) in enumerate(
        zip(qa_results[:3], example_results[:3])
    ):
        em_status = "✅" if eval_result["ExactMatch"] == 1.0 else "❌"
        f1_status = (
            "✅"
            if eval_result["F1"] > 0.5
            else "⚠️"
            if eval_result["F1"] > 0.0
            else "❌"
        )

        print(f"\nQuestion {i + 1} ({question_types[i]}): {qa_result['question']}")
        print(f"  Predicted: {eval_result['predicted_answer']}")
        print(f"  Gold: {eval_result['gold_answers']}")
        print(f"  EM: {eval_result['ExactMatch']:.3f} {em_status}")
        print(f"  F1: {eval_result['F1']:.3f} {f1_status}")
        print(f"  Entities used: {len(qa_result['matched_entities'])}")

    # Combine results for return
    detailed_results = []
    for i, (qa_result, eval_result) in enumerate(zip(qa_results, example_results)):
        detailed_results.append(
            {
                "question": qa_result["question"],
                "predicted_answer": eval_result["predicted_answer"],
                "gold_answers": eval_result["gold_answers"],
                "question_type": question_types[i],
                "exact_match": eval_result["ExactMatch"],
                "f1_score": eval_result["F1"],
                "extracted_entities": qa_result["extracted_entities"],
                "matched_entities": qa_result["matched_entities"],
                "num_summaries_used": qa_result["num_summaries_used"],
            }
        )

    return detailed_results, overall_metrics


def save_evaluation_results(
    evaluation_results, overall_metrics, log_dirs, num_questions
):
    """Save evaluation results to timestamped log directory."""
    print(f"\n💾 Saving results to {log_dirs['results_dir']}")

    # Save detailed results
    detailed_results_file = os.path.join(
        log_dirs["results_dir"], "detailed_results.json"
    )
    with open(detailed_results_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    # Save summary metrics
    summary_data = {
        "timestamp": log_dirs["timestamp"],
        "evaluation_type": "2wiki_multihop_qa",
        "num_questions": num_questions,
        "overall_metrics": overall_metrics,
        "question_type_breakdown": {},
    }

    # Calculate question type breakdown
    type_metrics = {}
    for result in evaluation_results:
        qtype = result["question_type"]
        if qtype not in type_metrics:
            type_metrics[qtype] = {"em_scores": [], "f1_scores": []}
        type_metrics[qtype]["em_scores"].append(result["exact_match"])
        type_metrics[qtype]["f1_scores"].append(result["f1_score"])

    for qtype, metrics in type_metrics.items():
        avg_em = sum(metrics["em_scores"]) / len(metrics["em_scores"])
        avg_f1 = sum(metrics["f1_scores"]) / len(metrics["f1_scores"])
        summary_data["question_type_breakdown"][qtype] = {
            "count": len(metrics["em_scores"]),
            "exact_match": round(avg_em, 4),
            "f1_score": round(avg_f1, 4),
        }

    summary_file = os.path.join(log_dirs["results_dir"], "summary_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Save a quick README
    readme_content = f"""# 2wiki Evaluation Results - {log_dirs["timestamp"]}

## Summary
- **Total Questions**: {num_questions}
- **Exact Match**: {overall_metrics["ExactMatch"]:.4f}
- **F1 Score**: {overall_metrics["F1"]:.4f}

## Files
- `detailed_results.json`: Per-question results with predictions and scores
- `summary_metrics.json`: Overall metrics and question type breakdown
- `all_entity_summaries.json`: All entity summaries generated by the aggregator
- `qa_debug_info.json`: Detailed Q&A process info including summaries used per question

## Directories
- `gsw_output/`: Raw GSW structures from document processing
- `reconciled_output/`: Reconciled GSW structures and statistics

## Debugging
The debug files contain:
- **Entity Summaries**: See what summaries were generated for each entity
- **Q&A Process**: See which entities were extracted, matched, and which summaries were used for each question
"""

    readme_file = os.path.join(log_dirs["base_dir"], "README.md")
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print("✅ Results saved:")
    print(f"   Detailed: {detailed_results_file}")
    print(f"   Summary: {summary_file}")
    print(f"   README: {readme_file}")


def main():
    """Run complete 2wiki evaluation pipeline with granular stage control."""
    print("🚀 Starting 2wiki MultiHopQA Evaluation")
    print("Testing GSW-Memory on factual multi-hop reasoning\n")

    try:
        # Determine what to load vs regenerate
        load_operator = LOAD_BASE_LOGS and REGENERATE_FROM not in ["operator"]
        load_reconciler = LOAD_BASE_LOGS and REGENERATE_FROM not in [
            "operator",
            "reconciler",
        ]
        load_aggregator = LOAD_BASE_LOGS and REGENERATE_FROM not in [
            "operator",
            "reconciler",
            "aggregator",
        ]

        # Print execution plan
        if LOAD_BASE_LOGS:
            print(f"📂 Base logs: {LOAD_BASE_LOGS}")
            print(f"🔄 Regenerate from: {REGENERATE_FROM or 'None (load all)'}")
            print("📋 Execution plan:")
            print(f"   Operator: {'Load' if load_operator else 'Regenerate'}")
            print(f"   Reconciler: {'Load' if load_reconciler else 'Regenerate'}")
            print(f"   Aggregator: {'Load' if load_aggregator else 'Regenerate'}")
            print()
        else:
            print("🔄 Running full pipeline from scratch\n")

        # Setup logging directory
        if LOAD_BASE_LOGS and not REGENERATE_FROM:
            # Use existing directory for Q&A only runs
            log_dirs = {
                "base_dir": LOAD_BASE_LOGS,
                "gsw_output_dir": os.path.join(LOAD_BASE_LOGS, "gsw_output"),
                "reconciled_output_dir": os.path.join(
                    LOAD_BASE_LOGS, "reconciled_output"
                ),
                "results_dir": os.path.join(LOAD_BASE_LOGS, "results"),
                "timestamp": os.path.basename(LOAD_BASE_LOGS).split("_")[-2]
                + "_"
                + os.path.basename(LOAD_BASE_LOGS).split("_")[-1],
            }
        else:
            # Create new timestamped directory for any regeneration
            log_dirs = setup_timestamped_logging()

        # Step 1: Load 2wiki data (always needed for questions)
        documents, questions_data, document_titles = load_2wiki_data(use_subset=True)

        # Load existing data once if needed
        loaded_data = None
        if LOAD_BASE_LOGS and (load_operator or load_reconciler or load_aggregator):
            print(f"📂 Loading data from {LOAD_BASE_LOGS}")
            loaded_data = load_from_logs(LOAD_BASE_LOGS)

        # Step 2: Operator + Reconciler stages (since current function combines them)
        if load_reconciler:
            print("📂 Using loaded reconciled GSW")
            reconciled_gsw = loaded_data["reconciled_gsw"]
        else:
            if load_operator:
                print("📂 Using loaded operator outputs, then running reconciler")
                operator_outputs = loaded_data["operator_outputs"]

                # Convert operator outputs to GSW structures for reconciler
                gsw_structures = []
                for doc_chunks in operator_outputs:
                    for chunk_data in doc_chunks.values():
                        if chunk_data["gsw"] is not None:
                            gsw_structures.append(chunk_data["gsw"])

                print("🔄 Running reconciler on loaded GSW structures")
                reconciled_gsw = reconcile_gsw_outputs(
                    gsw_structures,
                    strategy="global",
                    output_dir=log_dirs["reconciled_output_dir"],
                    save_statistics=True,
                    enable_visualization=False,
                )
            else:
                print("🔄 Processing documents through operator + reconciler")
                reconciled_gsw = process_corpus_to_gsws(
                    documents, log_dirs, use_subset=True
                )

        # Step 3: Aggregator stage - load or regenerate
        if load_aggregator:
            print("📂 Using loaded entity summaries")

            # Create aggregator with preloaded summaries
            llm_config = {
                "model_name": "gpt-4o",
                "generation_params": {"temperature": 0.0, "max_tokens": 500},
            }
            aggregator = EntitySummaryAggregator(reconciled_gsw, llm_config)
            aggregator._precomputed_summaries = loaded_data["entity_summaries"]
            print(
                f"✅ Loaded {len(loaded_data['entity_summaries'])} precomputed entity summaries"
            )
        else:
            print("🔄 Generating entity summaries")
            aggregator = generate_entity_summaries(reconciled_gsw, log_dirs)

        # Step 4: Setup Q&A system (always run)
        qa_system = setup_qa_system(reconciled_gsw, aggregator)

        # Step 5: Run evaluation (always run)
        evaluation_results, overall_metrics = run_2wiki_evaluation(
            qa_system, questions_data, log_dirs, use_subset=True
        )

        # Step 6: Save results (always run)
        save_evaluation_results(
            evaluation_results, overall_metrics, log_dirs, len(questions_data)
        )

        # Step 7: Final summary
        print("\n🎉 2wiki Evaluation Completed!")
        print("GSW-Memory Performance on 2wiki subset:")
        print(f"  Exact Match: {overall_metrics['ExactMatch']:.1%}")
        print(f"  F1 Score: {overall_metrics['F1']:.1%}")

        if overall_metrics["ExactMatch"] > 0.3:  # Reasonable baseline for first attempt
            print("✅ Promising initial results! Ready for optimization.")
        else:
            print("⚠️  Low accuracy - needs debugging and improvement.")

        print(f"\n📁 All outputs saved to: {log_dirs['base_dir']}")

        return {
            "overall_metrics": overall_metrics,
            "evaluation_results": evaluation_results,
            "num_questions": len(questions_data),
            "reconciled_gsw": reconciled_gsw,
            "log_dirs": log_dirs,
        }

    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
