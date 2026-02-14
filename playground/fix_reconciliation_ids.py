#!/usr/bin/env python3
"""
Run Corrected Reconciliation Only

This script runs the reconciler on the GSW files with corrected global IDs
to fix the entity collision bug without re-running the operator.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gsw_memory import reconcile_gsw_outputs
from gsw_memory.memory.models import GSWStructure

# Load environment variables
load_dotenv()

# Configuration
GLOBAL_GSW_FILE = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/logs/full_2wiki_corpus_20250710_202211/gsw_output_global_ids/all_gsw_global_ids.json"


def setup_corrected_logging():
    """Create timestamped log directory for corrected reconciliation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__), "..", "logs", f"full_2wiki_corpus_corrected_{timestamp}"
    )
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "reconciled_output"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "processing_logs"), exist_ok=True)
    
    print(f"📁 Created corrected log directory: {log_dir}")
    return log_dir


def load_processor_outputs():
    """Load the GSW data in processor output format."""
    print(f"🔄 Loading GSW data from: {GLOBAL_GSW_FILE}")
    
    with open(GLOBAL_GSW_FILE, "r") as f:
        all_gsw_data = json.load(f)
    
    # Convert GSW dictionaries to GSWStructure objects
    processor_outputs = []
    
    for doc_output in all_gsw_data:
        corrected_doc_output = {}
        
        for chunk_id, chunk_data in doc_output.items():
            # Convert GSW dict to GSWStructure object
            gsw_structure = GSWStructure.model_validate(chunk_data["gsw"])
            
            corrected_doc_output[chunk_id] = {
                "gsw": gsw_structure,
                "text": chunk_data.get("text", ""),
                "doc_idx": chunk_data.get("doc_idx"),
                "chunk_idx": chunk_data.get("chunk_idx", 0)
            }
        
        processor_outputs.append(corrected_doc_output)
    
    print(f"✅ Loaded {len(processor_outputs)} documents with corrected global IDs")
    return processor_outputs


def run_reconciliation(processor_outputs, log_dir):
    """Run global reconciliation with corrected IDs."""
    print(f"\n=== Running Corrected Global Reconciliation ===")
    print(f"Processing {len(processor_outputs)} documents")
    print(f"⚠️  WARNING: This may take a long time due to O(n²) entity matching")
    print(f"   Estimated entities: ~61,000 (vs 3,141 in buggy version)")
    print(f"   Consider processing smaller batches if this hangs\n")
    
    reconciliation_start_time = datetime.now()
    
    # Run global reconciliation
    reconciled_gsw = reconcile_gsw_outputs(
        processor_outputs,
        strategy="global",
        output_dir=os.path.join(log_dir, "reconciled_output"),
        save_statistics=True,
        enable_visualization=False,
    )
    
    reconciliation_end_time = datetime.now()
    reconciliation_duration = (reconciliation_end_time - reconciliation_start_time).total_seconds()
    
    print("\n✅ Corrected reconciliation completed!")
    print(f"   Processing time: {reconciliation_duration/60:.1f} minutes")
    print(f"   Final unified GSW:")
    print(f"     Entities: {len(reconciled_gsw.entity_nodes):,}")
    print(f"     Verb phrases: {len(reconciled_gsw.verb_phrase_nodes):,}")
    
    total_questions = sum(len(vp.questions) for vp in reconciled_gsw.verb_phrase_nodes)
    print(f"     Total questions: {total_questions:,}")
    
    # Save processing summary
    summary = {
        "corrected_reconciliation_summary": {
            "timestamp": reconciliation_end_time.isoformat(),
            "processing_time_minutes": round(reconciliation_duration / 60, 1),
            "bug_fixed": "Entity ID collisions from batch-relative indices",
            "documents_processed": len(processor_outputs),
            "final_stats": {
                "entities": len(reconciled_gsw.entity_nodes),
                "verb_phrases": len(reconciled_gsw.verb_phrase_nodes),
                "questions": total_questions,
            }
        }
    }
    
    summary_file = os.path.join(log_dir, "corrected_reconciliation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Results saved to: {log_dir}")
    return reconciled_gsw, log_dir


def main():
    """Run the corrected reconciliation pipeline."""
    print("🔧 Starting Corrected 2wiki Reconciliation")
    print("Running reconciler with fixed global document IDs\n")
    
    try:
        # Setup logging
        log_dir = setup_corrected_logging()
        
        # Load GSW data
        processor_outputs = load_processor_outputs()
        
        # Run reconciliation
        reconciled_gsw, result_dir = run_reconciliation(processor_outputs, log_dir)
        
        print(f"\n🎯 SUCCESS! Corrected reconciliation complete.")
        print(f"Entity count went from 3,141 to {len(reconciled_gsw.entity_nodes):,}")
        print(f"Results: {result_dir}")
        
        return reconciled_gsw, result_dir
        
    except Exception as e:
        print(f"\n❌ Error during reconciliation: {str(e)}")
        raise


if __name__ == "__main__":
    reconciled_gsw, log_dir = main()