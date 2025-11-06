#!/usr/bin/env python3
"""
Process Full 2wiki Corpus for Agentic Q&A

This script processes the entire 2wiki corpus (6,119 documents) through the GSW pipeline
to create a unified semantic workspace that can be used with the agentic Q&A system.

Pipeline: Documents → GSWProcessor → GSW chunks → Reconciler → Unified GSW

Key optimizations:
- Skips aggregator step (using agentic tools instead)
- Optimized for factual content processing
- Handles large-scale corpus processing efficiently
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gsw_memory import GSWProcessor, reconcile_gsw_outputs
from gsw_memory.prompts.operator_prompts import PromptType

# Disable cache for clean processing
# os.environ["CURATOR_DISABLE_CACHE"] = "true"

import importlib
print(importlib.metadata.version("bespokelabs-curator"))


# Load environment variables
load_dotenv()

# Configuration
CORPUS_PATH = "/mnt/SSD1/nlp/gsw/lvevaldata/all_errors.jsonl"
BATCH_SIZE = 1000  # Process documents in batches to manage memory


def setup_logging():
    """Create timestamped log directory for full corpus processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__), "..", "logs", f"full_2wiki_corpus_{timestamp}"
    )
    
    # Create directory structure
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "gsw_output"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "reconciled_output"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "processing_logs"), exist_ok=True)
    
    print(f"📁 Created log directory: {log_dir}")
    
    return {
        "base_dir": log_dir,
        "gsw_output_dir": os.path.join(log_dir, "gsw_output"),
        "reconciled_output_dir": os.path.join(log_dir, "reconciled_output"),
        "processing_logs_dir": os.path.join(log_dir, "processing_logs"),
        "timestamp": timestamp,
    }


def load_full_corpus(start_idx: int = 0, end_idx: int = None) -> tuple[List[str], List[str]]:
    """Load the complete 2wiki corpus or a subset.

    Args:
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive). If None, load until the end.

    Returns:
        Tuple of (documents, document_titles)
    """
    print("=== Loading Full 2wiki Corpus ===")
    
    # if file is a jsonl file, convert it to a json file
    if CORPUS_PATH.endswith(".jsonl"):
        with open(CORPUS_PATH, "r") as f:
            corpus_data = [json.loads(line) for line in f]
    else:
        with open(CORPUS_PATH, "r") as f:
            corpus_data = json.load(f)

    print(f"Loaded {len(corpus_data)} documents from corpus")

    # Extract document texts and titles
    documents = []
    document_titles = []

    for doc in corpus_data[start_idx:end_idx]:
        documents.append(doc["text"])
        document_titles.append(doc.get("title", ""))

    print(f"Prepared {len(documents)} documents for processing (range: {start_idx} to {end_idx or len(corpus_data)})")

    return documents, document_titles


def initialize_gsw_processor():
    """Initialize GSW processor optimized for large-scale factual content."""
    print("=== Initializing GSW Processor ===")
    
    processor = GSWProcessor(
        model_name="gpt-4.1-mini",
        enable_coref=False,          # Disable for speed and factual content
        enable_chunking=False,       # Factual documents are typically short
        chunk_size=1,               # Single chunk per document
        overlap=0,                  # No overlap needed
        enable_context=False,       # Disable context for efficiency
        enable_spacetime=False,      # Disable spacetime for small factual content
        prompt_type=PromptType.FACTUAL,  # Optimized for factual extraction
        batched=False,
        batch_size=BATCH_SIZE,
    )
    
    print("✅ GSW processor initialized with optimizations:")
    print("   - Factual prompt type for structured extraction")
    print("   - Chunking disabled for single-document processing")
    print("   - Coref disabled for performance")
    print("   - Spacetime enabled for temporal relationships")
    
    return processor


def process_corpus_in_batches(
    processor: GSWProcessor, 
    documents: List[str], 
    document_titles: List[str],
    log_dirs: Dict[str, str]
) -> List[Any]:
    """Process documents in batches to manage memory and monitor progress."""
    print(f"\n=== Processing {len(documents)} Documents in Batches ===")
    
    all_gsw_structures = []
    total_docs = len(documents)
    processed_docs = 0
    
    # Process in batches
                # Process batch
    start_time = datetime.now()
    
    # try:
    documents = [f"{title}\n{doc}" for title, doc in zip(document_titles, documents)] # Add title to each document

    gsw_structures = processor.process_documents(
        documents,
        output_dir=os.path.join(log_dirs["gsw_output_dir"], f"full_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    
    # Count successful GSW structures
    valid_structures = [gsw for gsw in gsw_structures if gsw is not None]
    all_gsw_structures.extend(valid_structures)
    
    end_time = datetime.now()
    run_duration = (end_time - start_time).total_seconds()
    
    print(f"   ✅ Processing completed in {run_duration:.1f}s")
    print(f"   Generated {len(valid_structures)} GSW structures")
    
    processed_docs += len(documents)
    progress = (processed_docs / total_docs) * 100
    print(f"   Progress: {processed_docs}/{total_docs} docs ({progress:.1f}%)")
    
    # Save batch processing log
    batch_log = {
        "batch_num": "full_corpus",
        "batch_start": "full_corpus",
        "batch_end": "full_corpus",
        "documents_processed": len(documents),
        "gsw_structures_generated": len(valid_structures),
        "processing_time_seconds": run_duration,
        "timestamp": end_time.isoformat(),
        "document_titles": document_titles
    }
    
    gsw_log_file = os.path.join(
        log_dirs["processing_logs_dir"], 
        f"full_corpus_log.json"
    )
    with open(gsw_log_file, "w") as f:
        json.dump(batch_log, f, indent=2)
    
    # except Exception as e:
    #     print(f"   ❌ Error processing full corpus: {str(e)}")
    #     print(f"   Continuing with next batch...")
        
    #     # Log error
    #     error_log = {
    #         "batch_num": "full_corpus",
    #         "error": str(e),
    #         "timestamp": datetime.now().isoformat()
    #     }
    #     error_log_file = os.path.join(
    #         log_dirs["processing_logs_dir"], 
    #         f"full_corpus_error.json"
    #     )
    #     with open(error_log_file, "w") as f:
    #         json.dump(error_log, f, indent=2)
    
    print(f"\n✅ Corpus processing completed!")
    print(f"   Total documents processed: {processed_docs}")
    print(f"   Total GSW structures generated: {len(all_gsw_structures)}")
    
    return all_gsw_structures


def reconcile_full_corpus(gsw_structures: List[Any], log_dirs: Dict[str, str]):
    """Reconcile GSW structures using global strategy for cross-document linking."""
    print(f"\n=== Reconciling {len(gsw_structures)} GSW Structures ===")
    
    reconciliation_start_time = datetime.now()
    
    # Use global strategy for cross-document entity linking
    reconciled_gsw = reconcile_gsw_outputs(
        gsw_structures,
        strategy="global",  # Critical for cross-document reasoning
        output_dir=log_dirs["reconciled_output_dir"],
        save_statistics=True,
        enable_visualization=False,  # Disable for large corpus
    )
    
    reconciliation_end_time = datetime.now()
    reconciliation_duration = (reconciliation_end_time - reconciliation_start_time).total_seconds()
    
    print("✅ Reconciliation completed!")
    print(f"   Processing time: {reconciliation_duration:.1f}s")
    print(f"   Final unified GSW:")
    print(f"     Entities: {len(reconciled_gsw.entity_nodes)}")
    print(f"     Verb phrases: {len(reconciled_gsw.verb_phrase_nodes)}")
    
    # Calculate reconciliation statistics
    total_questions = sum(len(vp.questions) for vp in reconciled_gsw.verb_phrase_nodes)
    
    reconciliation_stats = {
        "input_gsw_structures": len(gsw_structures),
        "reconciliation_time_seconds": reconciliation_duration,
        "final_entity_count": len(reconciled_gsw.entity_nodes),
        "final_verb_phrase_count": len(reconciled_gsw.verb_phrase_nodes),
        "total_questions": total_questions,
        "timestamp": reconciliation_end_time.isoformat(),
        "strategy": "global"
    }
    
    stats_file = os.path.join(log_dirs["processing_logs_dir"], "reconciliation_stats.json")
    with open(stats_file, "w") as f:
        json.dump(reconciliation_stats, f, indent=2)
    
    return reconciled_gsw


def save_corpus_processing_summary(
    log_dirs: Dict[str, str],
    total_docs: int,
    gsw_structures_count: int,
    reconciled_gsw,
    processing_start_time: datetime
):
    """Save a comprehensive summary of the corpus processing."""
    processing_end_time = datetime.now()
    total_duration = (processing_end_time - processing_start_time).total_seconds()

    summary = {
        "corpus_processing_summary": {
            "timestamp": processing_end_time.isoformat(),
            "total_processing_time_seconds": total_duration,
            "total_processing_time_hours": round(total_duration / 3600, 2),
            "corpus_info": {
                "total_documents": total_docs,
                "gsw_structures_generated": gsw_structures_count,
                "success_rate": round((gsw_structures_count / total_docs) * 100, 2)
            },
            "final_gsw_stats": {
                "entity_count": len(reconciled_gsw.entity_nodes),
                "verb_phrase_count": len(reconciled_gsw.verb_phrase_nodes),
                "total_questions": sum(len(vp.questions) for vp in reconciled_gsw.verb_phrase_nodes)
            },
            "processing_config": {
                "model_name": "gpt-4.1-mini",
                "prompt_type": "FACTUAL",
                "reconciliation_strategy": "global",
                "chunking_enabled": False,
                "coref_enabled": False,
                "spacetime_enabled": True
            }
        }
    }

    summary_file = os.path.join(log_dirs["base_dir"], "corpus_processing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Create README
    readme_content = f"""# Full 2wiki Corpus Processing - {log_dirs['timestamp']}

## Summary
- **Total Documents**: {total_docs:,}
- **GSW Structures Generated**: {gsw_structures_count:,}
- **Success Rate**: {round((gsw_structures_count / total_docs) * 100, 2)}%
- **Processing Time**: {round(total_duration / 3600, 2)} hours

## Final GSW Statistics
- **Entities**: {len(reconciled_gsw.entity_nodes):,}
- **Verb Phrases**: {len(reconciled_gsw.verb_phrase_nodes):,}
- **Total Questions**: {sum(len(vp.questions) for vp in reconciled_gsw.verb_phrase_nodes):,}

## Output Files
- `corpus_processing_summary.json`: Complete processing statistics
- `reconciled_output/`: Reconciled GSW structure (ready for agentic Q&A)
- `gsw_output/`: Raw GSW structures organized by batch
- `processing_logs/`: Detailed batch processing logs and error reports

## Next Steps
1. Use the reconciled GSW with the agentic Q&A system
2. Test on the full 1,000 question 2wiki dataset
3. Compare performance with subset results

## Configuration Used
- Model: GPT-4.1-mini
- Prompt Type: FACTUAL
- Reconciliation: Global strategy
- Chunking: Disabled
- Coreference: Disabled
- Spacetime: Enabled
"""

    readme_file = os.path.join(log_dirs["base_dir"], "README.md")
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print(f"\n💾 Processing summary saved:")
    print(f"   Summary: {summary_file}")
    print(f"   README: {readme_file}")


def find_missing_documents(output_dir: str, corpus_offset: int = 0) -> Set[int]:
    """
    Find missing document indices by comparing existing network directories to expected range.

    Args:
        output_dir: Path to GSW output directory (e.g., .../full_corpus_20251031_144213/)
        corpus_offset: The offset used when loading the corpus (e.g., documents[2000:] means offset=2000)

    Returns:
        Set of missing document indices relative to the corpus offset
    """
    output_path = Path(output_dir)

    # Read metadata from gsw_results_combined.json to get expected document count
    combined_json = output_path / "gsw_results_combined.json"
    if not combined_json.exists():
        raise FileNotFoundError(f"gsw_results_combined.json not found at {combined_json}")

    with open(combined_json) as f:
        data = json.load(f)
        expected_total = data["metadata"]["total_documents"]
        # expected_total = 9811 # TODO: remove this

    print(f"Expected {expected_total} documents based on metadata")

    # Find existing doc_N directories
    networks_dir = output_path / "networks"
    if not networks_dir.exists():
        raise FileNotFoundError(f"Networks directory not found at {networks_dir}")

    existing_docs = set()
    for doc_dir in networks_dir.iterdir():
        if doc_dir.is_dir() and doc_dir.name.startswith("doc_"):
            doc_id = int(doc_dir.name.split("_")[1])
            existing_docs.add(doc_id)

    print(f"Found {len(existing_docs)} existing document directories")

    # Find missing documents
    # The expected range is from corpus_offset to corpus_offset + expected_total
    expected_docs = set(range(corpus_offset, corpus_offset + expected_total))
    missing_docs = expected_docs - existing_docs

    if missing_docs:
        missing_sorted = sorted(missing_docs)
        print(f"\n⚠️  Found {len(missing_docs)} missing documents:")
        print(f"   First 10: {missing_sorted[:10]}")
        if len(missing_docs) > 10:
            print(f"   Last 10: {missing_sorted[-10:]}")
    else:
        print("✅ No missing documents found!")

    return missing_docs


def process_missing_documents(
    missing_doc_indices: Set[int],
    output_dir: str,
    corpus_offset: int = 0
) -> int:
    """
    Process missing documents in batch and merge into existing output directory.

    Since missing_doc_indices are already sorted and in order, we can batch process them.
    After processing, we rename the output folders and merge into gsw_results_combined.json.

    Args:
        missing_doc_indices: Set of missing document indices
        output_dir: Path to existing GSW output directory
        corpus_offset: The corpus offset used in original processing

    Returns:
        Number of documents successfully processed
    """
    if not missing_doc_indices:
        print("No missing documents to process!")
        return 0

    print(f"\n=== Processing {len(missing_doc_indices)} Missing Documents ===")

    # Load full corpus
    # if file is a jsonl file, convert it to a json file
    if CORPUS_PATH.endswith(".jsonl"):
        with open(CORPUS_PATH, "r") as f:
            corpus_data = [json.loads(line) for line in f]
    else:
        with open(CORPUS_PATH, "r") as f:
            corpus_data = json.load(f)

    # Extract missing documents in sorted order
    missing_docs_sorted = sorted(missing_doc_indices)
    docs_to_process = []

    print(f"Loading missing documents from corpus...")
    for doc_idx in missing_docs_sorted:
        if doc_idx >= len(corpus_data):
            print(f"⚠️  Warning: Document index {doc_idx} out of range")
            continue

        # Get document text and title
        doc_text = corpus_data[doc_idx]["text"]
        doc_title = corpus_data[doc_idx].get("title", "")
        doc_with_title = f"{doc_title}\n{doc_text}"
        docs_to_process.append(doc_with_title)

    print(f"Loaded {len(docs_to_process)} documents for processing")

    # Initialize processor with same settings as original
    processor = initialize_gsw_processor()

    # Create temporary output directory for processing
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="gsw_recovery_")
    print(f"Using temporary directory: {temp_dir}")

    start_time = datetime.now()

    try:
        # Process all missing documents in batch
        print(f"Processing {len(docs_to_process)} documents in batch...")
        gsw_results = processor.process_documents(
            docs_to_process,
            output_dir=temp_dir,
            batch_idx=1,
            batch_size=len(docs_to_process)
        )

        successful_count = sum(1 for r in gsw_results if r is not None)
        print(f"Successfully processed {successful_count}/{len(docs_to_process)} documents")

        # Now we need to:
        # 1. Rename/move the network folders from temp_dir to output_dir with correct doc_N indices
        # 2. Merge the gsw_results_combined.json data with correct doc_N keys

        print(f"\n--- Moving network folders to correct doc_N locations ---")
        temp_networks_dir = Path(temp_dir) / "networks"
        temp_networks_raw_dir = Path(temp_dir) / "networks_raw"
        output_networks_dir = Path(output_dir) / "networks"
        output_networks_raw_dir = Path(output_dir) / "networks_raw"

        # The processor created doc_0, doc_1, doc_2, ... in temp_dir
        # We need to rename them to doc_398, doc_559, doc_951, ... based on missing_docs_sorted
        for i, actual_doc_idx in enumerate(missing_docs_sorted[:successful_count]):
            temp_doc_folder = f"doc_{i}"
            actual_doc_folder = f"doc_{actual_doc_idx}"

            # Move networks folder
            if (temp_networks_dir / temp_doc_folder).exists():
                import shutil
                shutil.move(
                    str(temp_networks_dir / temp_doc_folder),
                    str(output_networks_dir / actual_doc_folder)
                )

            # Move networks_raw folder
            if (temp_networks_raw_dir / temp_doc_folder).exists():
                import shutil
                shutil.move(
                    str(temp_networks_raw_dir / temp_doc_folder),
                    str(output_networks_raw_dir / actual_doc_folder)
                )

            # Also rename the GSW files inside to match doc_idx
            # gsw_0_0.json -> gsw_398_0.json, etc.
            for gsw_file in (output_networks_dir / actual_doc_folder).glob("gsw_*.json"):
                old_name = gsw_file.name
                # Parse: gsw_0_0.json -> gsw_398_0.json
                parts = old_name.replace("gsw_", "").replace(".json", "").split("_")
                if len(parts) >= 2:
                    chunk_idx = parts[1]
                    new_name = f"gsw_{actual_doc_idx}_{chunk_idx}.json"
                    gsw_file.rename(gsw_file.parent / new_name)

            for gsw_file in (output_networks_raw_dir / actual_doc_folder).glob("gsw_*.json"):
                old_name = gsw_file.name
                parts = old_name.replace("gsw_", "").replace(".json", "").split("_")
                if len(parts) >= 2:
                    chunk_idx = parts[1]
                    new_name = f"gsw_{actual_doc_idx}_{chunk_idx}.json"
                    gsw_file.rename(gsw_file.parent / new_name)

        print(f"✅ Moved {successful_count} network folders to correct locations")

        # Load temp gsw_results_combined.json and merge into existing
        print(f"\n--- Merging into gsw_results_combined.json ---")
        temp_combined_path = Path(temp_dir) / "gsw_results_combined.json"
        output_combined_path = Path(output_dir) / "gsw_results_combined.json"

        with open(temp_combined_path) as f:
            temp_combined = json.load(f)

        if output_combined_path.exists():
            with open(output_combined_path) as f:
                existing_combined = json.load(f)
        else:
            existing_combined = {
                "metadata": {"total_documents": 0, "total_chunks": 0},
                "documents": {}
            }

        # Merge documents with correct doc_N keys
        for i, actual_doc_idx in enumerate(missing_docs_sorted[:successful_count]):
            temp_doc_key = f"doc_{i}"
            actual_doc_key = f"doc_{actual_doc_idx}"

            if temp_doc_key in temp_combined["documents"]:
                # Update the global_id and doc_idx in the chunk data
                doc_data = temp_combined["documents"][temp_doc_key]
                updated_doc_data = {}

                for chunk_key, chunk_data in doc_data.items():
                    # Update doc_idx in chunk_data
                    chunk_data["doc_idx"] = actual_doc_idx

                    # Update global_id: 0_0 -> 398_0
                    parts = chunk_key.split("_")
                    if len(parts) >= 2:
                        new_chunk_key = f"{actual_doc_idx}_{parts[1]}"
                        chunk_data["global_id"] = new_chunk_key
                        updated_doc_data[new_chunk_key] = chunk_data
                    else:
                        updated_doc_data[chunk_key] = chunk_data

                existing_combined["documents"][actual_doc_key] = updated_doc_data

        # Update metadata
        existing_combined["metadata"]["total_chunks"] = sum(
            len(doc_chunks) for doc_chunks in existing_combined["documents"].values()
        )
        existing_combined["metadata"]["processed_at"] = datetime.now().isoformat()

        # Save updated combined JSON
        with open(output_combined_path, "w") as f:
            json.dump(existing_combined, f, indent=2)

        print(f"✅ Updated gsw_results_combined.json with {successful_count} recovered documents")

        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory")

    except Exception as e:
        print(f"  ❌ Error processing documents: {str(e)}")
        import traceback
        traceback.print_exc()
        successful_count = 0

    end_time = datetime.now()
    run_duration = (end_time - start_time).total_seconds()

    failed_count = len(docs_to_process) - successful_count
    print(f"\n✅ Missing documents processing completed!")
    print(f"   Total attempted: {len(docs_to_process)} documents")
    print(f"   Successful: {successful_count} GSW structures")
    print(f"   Failed: {failed_count}")
    print(f"   Processing time: {run_duration:.1f}s")

    # Log recovery results
    output_path = Path(output_dir)
    recovery_log = {
        "recovery_timestamp": end_time.isoformat(),
        "missing_documents_count": len(missing_doc_indices),
        "attempted_count": len(docs_to_process),
        "successful_count": successful_count,
        "failed_count": failed_count,
        "processing_time_seconds": run_duration,
        "missing_doc_ids": missing_docs_sorted,
        "corpus_offset": corpus_offset
    }

    recovery_log_file = output_path.parent.parent / "processing_logs" / "recovery_log.json"
    recovery_log_file.parent.mkdir(exist_ok=True)
    with open(recovery_log_file, "w") as f:
        json.dump(recovery_log, f, indent=2)

    print(f"📝 Recovery log saved to: {recovery_log_file}")

    return successful_count


def main():
    """Run the complete full corpus processing pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process 2wiki corpus through GSW pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full corpus
  python process_full_2wiki_corpus.py

  # Recover missing documents from a previous run
  python process_full_2wiki_corpus.py --recover-missing \\
    /home/yigit/codebase/gsw-memory/logs/full_2wiki_corpus_20251031_144213/gsw_output/full_corpus_20251031_144213 \\
    --corpus-offset 0

  # Recover from second batch (offset 2000)
  python process_full_2wiki_corpus.py --recover-missing \\
    /home/yigit/codebase/gsw-memory/logs/full_2wiki_corpus_20251031_145231/gsw_output/full_corpus_20251031_145231 \\
    --corpus-offset 2000
        """
    )

    parser.add_argument(
        "--recover-missing",
        type=str,
        metavar="OUTPUT_DIR",
        help="Path to existing GSW output directory to recover missing documents"
    )

    parser.add_argument(
        "--corpus-offset",
        type=int,
        default=0,
        help="Corpus offset used in original processing (e.g., 0 for documents[0:], 2000 for documents[2000:])"
    )

    args = parser.parse_args()

    # Recovery mode
    if args.recover_missing:
        print("🔧 Starting Missing Documents Recovery Mode")
        print(f"Output directory: {args.recover_missing}")
        print(f"Corpus offset: {args.corpus_offset}\n")

        try:
            # Find missing documents
            missing_docs = find_missing_documents(args.recover_missing, args.corpus_offset)

            if not missing_docs:
                print("\n✅ No missing documents found! Nothing to recover.")
                return {"status": "complete", "missing_count": 0}

            # Process missing documents
            successful_count = process_missing_documents(
                missing_docs,
                args.recover_missing,
                args.corpus_offset
            )

            # Verify recovery
            print("\n🔍 Verifying recovery...")
            remaining_missing = find_missing_documents(args.recover_missing, args.corpus_offset)

            if remaining_missing:
                print(f"\n⚠️  Warning: {len(remaining_missing)} documents still missing after recovery")
                print(f"   These documents may have failed processing")
            else:
                print("\n✅ All missing documents recovered successfully!")

            return {
                "status": "recovery_complete",
                "original_missing_count": len(missing_docs),
                "successful_count": successful_count,
                "remaining_missing_count": len(remaining_missing)
            }

        except Exception as e:
            print(f"\n❌ Error during recovery: {str(e)}")
            raise

    # Normal processing mode
    print("🚀 Starting Full 2wiki Corpus Processing")
    print("Processing 6,119 documents through GSW pipeline for agentic Q&A\n")

    processing_start_time = datetime.now()

    try:
        # Setup logging
        log_dirs = setup_logging()

        # Load corpus
        documents, document_titles = load_full_corpus()

        # Initialize processor
        processor = initialize_gsw_processor()

        # Process corpus in batches
        gsw_structures = process_corpus_in_batches(
            processor, documents, document_titles, log_dirs
        )

        # Reconcile into unified GSW
        # reconciled_gsw = reconcile_full_corpus(gsw_structures, log_dirs)

        # Save processing summary
        # save_corpus_processing_summary(
        #     log_dirs, len(documents), len(gsw_structures),
        #     reconciled_gsw, processing_start_time
        # )

        print(f"\n🎉 Full corpus processing completed successfully!")
        print(f"📁 All outputs saved to: {log_dirs['base_dir']}")
        # print(f"🔍 Ready for agentic Q&A evaluation with {len(reconciled_gsw.entity_nodes):,} entities")
        # print(f"🔍 Ready for agentic Q&A evaluation with {len(reconciled_gsw.entity_nodes):,} entities")

        return {
            # "reconciled_gsw": reconciled_gsw,
            "log_dirs": log_dirs,
            "processing_stats": {
                "total_documents": len(documents),
                "gsw_structures": len(gsw_structures),
                # "final_entities": len(reconciled_gsw.entity_nodes),
                # "final_verb_phrases": len(reconciled_gsw.verb_phrase_nodes)
                # "final_entities": len(reconciled_gsw.entity_nodes),
                # "final_verb_phrases": len(reconciled_gsw.verb_phrase_nodes)
            }
        }

    except Exception as e:
        print(f"\n❌ Error during corpus processing: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()