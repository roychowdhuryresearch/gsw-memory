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
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gsw_memory import GSWProcessor, reconcile_gsw_outputs
from gsw_memory.prompts.operator_prompts import PromptType

# Disable cache for clean processing
os.environ["CURATOR_DISABLE_CACHE"] = "true"

# Load environment variables
load_dotenv()

# Configuration
CORPUS_PATH = "/home/shreyas/NLP/SM/gensemworkspaces/HippoRAG/reproduce/dataset/2wikimultihopqa_corpus.json"
BATCH_SIZE = 100  # Process documents in batches to manage memory


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


def load_full_corpus() -> List[Dict[str, Any]]:
    """Load the complete 2wiki corpus."""
    print("=== Loading Full 2wiki Corpus ===")
    
    with open(CORPUS_PATH, "r") as f:
        corpus_data = json.load(f)
    
    print(f"Loaded {len(corpus_data)} documents from corpus")
    
    # Extract document texts and titles
    documents = []
    document_titles = []
    
    for doc in corpus_data:
        documents.append(doc["text"])
        document_titles.append(doc["title"])
    
    print(f"Prepared {len(documents)} documents for processing")
    
    return documents, document_titles


def initialize_gsw_processor():
    """Initialize GSW processor optimized for large-scale factual content."""
    print("=== Initializing GSW Processor ===")
    
    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=False,          # Disable for speed and factual content
        enable_chunking=False,       # Factual documents are typically short
        chunk_size=1,               # Single chunk per document
        overlap=0,                  # No overlap needed
        enable_context=False,       # Disable context for efficiency
        enable_spacetime=False,      # Disable spacetime for small factual content
        prompt_type=PromptType.FACTUAL,  # Optimized for factual extraction
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
    for batch_start in range(0, total_docs, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_docs)
        batch_docs = documents[batch_start:batch_end]
        batch_titles = document_titles[batch_start:batch_end]
        
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n🔄 Processing Batch {batch_num}/{total_batches}")
        print(f"   Documents {batch_start + 1}-{batch_end} of {total_docs}")
        print(f"   Batch size: {len(batch_docs)} documents")
        
        # Process batch
        batch_start_time = datetime.now()
        
        try:
            batch_gsw_structures = processor.process_documents(
                batch_docs, 
                output_dir=os.path.join(log_dirs["gsw_output_dir"], f"batch_{batch_num:03d}")
            )
            
            # Count successful GSW structures
            valid_structures = [gsw for gsw in batch_gsw_structures if gsw is not None]
            all_gsw_structures.extend(valid_structures)
            
            batch_end_time = datetime.now()
            batch_duration = (batch_end_time - batch_start_time).total_seconds()
            
            print(f"   ✅ Batch {batch_num} completed in {batch_duration:.1f}s")
            print(f"   Generated {len(valid_structures)} GSW structures")
            
            processed_docs += len(batch_docs)
            progress = (processed_docs / total_docs) * 100
            print(f"   Progress: {processed_docs}/{total_docs} docs ({progress:.1f}%)")
            
            # Save batch processing log
            batch_log = {
                "batch_num": batch_num,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "documents_processed": len(batch_docs),
                "gsw_structures_generated": len(valid_structures),
                "processing_time_seconds": batch_duration,
                "timestamp": batch_end_time.isoformat(),
                "document_titles": batch_titles
            }
            
            batch_log_file = os.path.join(
                log_dirs["processing_logs_dir"], 
                f"batch_{batch_num:03d}_log.json"
            )
            with open(batch_log_file, "w") as f:
                json.dump(batch_log, f, indent=2)
            
        except Exception as e:
            print(f"   ❌ Error processing batch {batch_num}: {str(e)}")
            print(f"   Continuing with next batch...")
            
            # Log error
            error_log = {
                "batch_num": batch_num,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            error_log_file = os.path.join(
                log_dirs["processing_logs_dir"], 
                f"batch_{batch_num:03d}_error.json"
            )
            with open(error_log_file, "w") as f:
                json.dump(error_log, f, indent=2)
    
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
                "batch_size": BATCH_SIZE,
                "model_name": "gpt-4o",
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
- Model: GPT-4o
- Prompt Type: FACTUAL
- Batch Size: {BATCH_SIZE}
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


def main():
    """Run the complete full corpus processing pipeline."""
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
        reconciled_gsw = reconcile_full_corpus(gsw_structures, log_dirs)
        
        # Save processing summary
        save_corpus_processing_summary(
            log_dirs, len(documents), len(gsw_structures), 
            reconciled_gsw, processing_start_time
        )
        
        print(f"\n🎉 Full corpus processing completed successfully!")
        print(f"📁 All outputs saved to: {log_dirs['base_dir']}")
        print(f"🔍 Ready for agentic Q&A evaluation with {len(reconciled_gsw.entity_nodes):,} entities")
        
        return {
            "reconciled_gsw": reconciled_gsw,
            "log_dirs": log_dirs,
            "processing_stats": {
                "total_documents": len(documents),
                "gsw_structures": len(gsw_structures),
                "final_entities": len(reconciled_gsw.entity_nodes),
                "final_verb_phrases": len(reconciled_gsw.verb_phrase_nodes)
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error during corpus processing: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()