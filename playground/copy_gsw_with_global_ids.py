#!/usr/bin/env python3
"""
Copy GSW Files with Global Document IDs

Supports two input layouts:
  1. Flat:  ORIGINAL_LOGS_DIR/networks/doc_i/gsw_i_0.json
  2. Batch: ORIGINAL_LOGS_DIR/gsw_output/batch_001/networks/doc_N/gsw_N_0.json

Output: OUTPUT_DIR/networks/doc_<global>/gsw_<global>_0.json
"""

import json
import os
import shutil
from pathlib import Path

# Configuration
ORIGINAL_LOGS_DIR = "/home/yigit/codebase/gsw-memory/data/sleep_time/frames/networks_output/frames_20260228_222500"
OUTPUT_DIR = os.path.join(ORIGINAL_LOGS_DIR, "gsw_output_global_ids")
BATCH_SIZE = 100


def _process_flat(source_networks_dir: str, dest_networks_dir: str) -> int:
    """Copy GSW files from a flat networks/doc_i/ layout."""
    total = 0
    doc_dirs = sorted(
        [d for d in Path(source_networks_dir).iterdir() if d.is_dir() and d.name.startswith("doc_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for doc_dir in doc_dirs:
        doc_idx = int(doc_dir.name.split("_")[1])
        # Find the GSW file (gsw_<idx>_0.json)
        gsw_files = list(doc_dir.glob("gsw_*_0.json"))
        if not gsw_files:
            continue
        source_file = gsw_files[0]

        dest_dir = os.path.join(dest_networks_dir, f"doc_{doc_idx}")
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(dest_dir, f"gsw_{doc_idx}_0.json")

        shutil.copy2(str(source_file), dest_file)
        total += 1

        if total % 500 == 0:
            print(f"  Copied {total} files...")

    return total


def _process_batches(logs_dir: str, dest_networks_dir: str) -> int:
    """Copy GSW files from batch_NNN/ layout with batch logs."""
    total = 0
    batch_num = 1
    while True:
        batch_dir = os.path.join(logs_dir, "gsw_output", f"batch_{batch_num:03d}")
        if not os.path.exists(batch_dir):
            break

        print(f"Processing batch {batch_num}...")
        batch_log_file = os.path.join(logs_dir, "processing_logs", f"batch_{batch_num:03d}_log.json")

        try:
            with open(batch_log_file, "r") as f:
                batch_log = json.load(f)

            batch_start_global = batch_log["batch_start"]
            documents_in_batch = batch_log["documents_processed"]

            for doc_idx_in_batch in range(documents_in_batch):
                global_doc_idx = batch_start_global + doc_idx_in_batch
                source_file = os.path.join(
                    batch_dir, "networks", f"doc_{doc_idx_in_batch}", f"gsw_{doc_idx_in_batch}_0.json"
                )
                if os.path.exists(source_file):
                    dest_dir = os.path.join(dest_networks_dir, f"doc_{global_doc_idx}")
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_file = os.path.join(dest_dir, f"gsw_{global_doc_idx}_0.json")
                    shutil.copy2(source_file, dest_file)
                    total += 1

                    if total % 500 == 0:
                        print(f"  Copied {total} files...")

            print(f"  Batch {batch_num}: Copied {documents_in_batch} files (global docs {batch_start_global}-{batch_start_global + documents_in_batch - 1})")
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            break

        batch_num += 1
    return total


def main():
    """Copy all GSW files with corrected global document IDs."""
    print("Copying GSW files with global document IDs")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    networks_dir = os.path.join(OUTPUT_DIR, "networks")
    os.makedirs(networks_dir, exist_ok=True)

    # Detect layout: flat (networks/doc_*) or batch (gsw_output/batch_*)
    flat_networks = os.path.join(ORIGINAL_LOGS_DIR, "networks")
    if os.path.isdir(flat_networks) and any(
        d.startswith("doc_") for d in os.listdir(flat_networks)
    ):
        print("Detected flat layout (networks/doc_*)")
        total_files_copied = _process_flat(flat_networks, networks_dir)
    else:
        print("Detected batch layout (gsw_output/batch_*)")
        total_files_copied = _process_batches(ORIGINAL_LOGS_DIR, networks_dir)

    print(f"\nCopied {total_files_copied} GSW files to {OUTPUT_DIR}")
    
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