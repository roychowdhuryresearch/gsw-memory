#!/usr/bin/env python3
"""
Copy GSW Files with Global Document IDs

This script copies all individual GSW files from batch directories and renames them
with global document indices instead of batch-relative indices.

Input:  batch_001/networks/doc_22/gsw_22_0.json  (batch-relative)
Output: global_gsw/doc_5022/gsw_5022_0.json     (global index)
"""

import json
import os
import shutil
from pathlib import Path

# Configuration
ORIGINAL_LOGS_DIR = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/logs/full_2wiki_corpus_20250710_202211"
OUTPUT_DIR = os.path.join(ORIGINAL_LOGS_DIR, "gsw_output_global_ids")
BATCH_SIZE = 100


def main():
    """Copy all GSW files with corrected global document IDs."""
    print("🔄 Copying GSW files with global document IDs")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    networks_dir = os.path.join(OUTPUT_DIR, "networks")
    os.makedirs(networks_dir, exist_ok=True)
    
    total_files_copied = 0
    
    # Process each batch
    batch_num = 1
    while True:
        batch_dir = os.path.join(ORIGINAL_LOGS_DIR, "gsw_output", f"batch_{batch_num:03d}")
        
        if not os.path.exists(batch_dir):
            break
            
        print(f"Processing batch {batch_num}...")
        
        # Load batch log to get global document indices
        batch_log_file = os.path.join(ORIGINAL_LOGS_DIR, "processing_logs", f"batch_{batch_num:03d}_log.json")
        
        try:
            with open(batch_log_file, "r") as f:
                batch_log = json.load(f)
            
            batch_start_global = batch_log["batch_start"]
            documents_in_batch = batch_log["documents_processed"]
            
            # Copy each document's GSW file
            for doc_idx_in_batch in range(documents_in_batch):
                global_doc_idx = batch_start_global + doc_idx_in_batch
                
                # Source file (batch-relative)
                source_file = os.path.join(
                    batch_dir, "networks", f"doc_{doc_idx_in_batch}", f"gsw_{doc_idx_in_batch}_0.json"
                )
                
                if os.path.exists(source_file):
                    # Destination with global index
                    dest_dir = os.path.join(networks_dir, f"doc_{global_doc_idx}")
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    dest_file = os.path.join(dest_dir, f"gsw_{global_doc_idx}_0.json")
                    
                    # Copy the file
                    shutil.copy2(source_file, dest_file)
                    total_files_copied += 1
                    
                    if total_files_copied % 500 == 0:
                        print(f"  Copied {total_files_copied} files...")
            
            print(f"  Batch {batch_num}: Copied {documents_in_batch} files (global docs {batch_start_global}-{batch_start_global + documents_in_batch - 1})")
            
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            break
            
        batch_num += 1
    
    print(f"\n✅ Copied {total_files_copied} GSW files to {OUTPUT_DIR}")
    print(f"Files are now named with global document indices (0-{total_files_copied-1})")
    
    # Create a simple combined file for the reconciler
    print("\n🔄 Creating combined GSW file for reconciler...")
    
    all_gsw_data = []
    
    for global_doc_idx in range(total_files_copied):
        gsw_file = os.path.join(networks_dir, f"doc_{global_doc_idx}", f"gsw_{global_doc_idx}_0.json")
        
        if os.path.exists(gsw_file):
            with open(gsw_file, "r") as f:
                gsw_data = json.load(f)
            
            # Create processor output format
            doc_output = {
                f"{global_doc_idx}_0": {  # Global chunk ID
                    "gsw": gsw_data,
                    "text": "",
                    "doc_idx": global_doc_idx,
                    "chunk_idx": 0
                }
            }
            all_gsw_data.append(doc_output)
    
    # Save combined file
    combined_file = os.path.join(OUTPUT_DIR, "all_gsw_global_ids.json")
    with open(combined_file, "w") as f:
        json.dump(all_gsw_data, f, indent=2)
    
    print(f"✅ Created combined file: {combined_file}")
    print(f"Ready for reconciliation with {len(all_gsw_data)} documents")
    
    return OUTPUT_DIR, len(all_gsw_data)


if __name__ == "__main__":
    output_dir, num_docs = main()
    print(f"\n🎯 Next step: Run reconciler on {output_dir}/all_gsw_global_ids.json")