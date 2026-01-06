#!/usr/bin/env python3
"""
Flatten Batch GSW Output

This script takes a batched GSW output folder and flattens it into a single
folder with globally unique document IDs.

Input structure:
  corpus_TIMESTAMP/
    batch_0000/
      networks/
        doc_0/gsw_0_0.json
        doc_1/gsw_1_0.json
        ...
      networks_raw/
        doc_0/gsw_0_0.json
        ...
    batch_0001/
      networks/
        doc_0/gsw_0_0.json  (this is actually global doc 10 if batch_size=10)
        ...

Output structure:
  output_dir/
    networks/
      doc_0/gsw_0_0.json
      doc_1/gsw_1_0.json
      ...
      doc_10/gsw_10_0.json  (from batch_0001/doc_0)
      doc_11/gsw_11_0.json  (from batch_0001/doc_1)
      ...
    networks_raw/
      doc_0/gsw_0_0.json
      ...
    gsw_results_combined.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime


def get_batch_size_from_batches(input_dir: Path) -> int:
    """Infer batch size by counting docs in first complete batch."""
    batch_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])

    if not batch_dirs:
        raise ValueError(f"No batch directories found in {input_dir}")

    # Use the first batch to determine batch size
    first_batch = batch_dirs[0]
    networks_dir = first_batch / "networks"

    if not networks_dir.exists():
        raise ValueError(f"No networks directory found in {first_batch}")

    doc_dirs = [d for d in networks_dir.iterdir() if d.is_dir() and d.name.startswith("doc_")]
    return len(doc_dirs)


def flatten_batch_output(input_dir: str, output_dir: str, batch_size: int = None):
    """
    Flatten batched GSW output into a single folder with global doc IDs.

    Args:
        input_dir: Path to corpus output directory (e.g., .../corpus_20260102_145007/)
        output_dir: Path to output directory for flattened structure
        batch_size: Number of documents per batch. If None, will be inferred.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    output_networks = output_path / "networks"
    output_networks_raw = output_path / "networks_raw"
    output_networks.mkdir(parents=True, exist_ok=True)
    output_networks_raw.mkdir(parents=True, exist_ok=True)

    # Find all batch directories
    batch_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("batch_")])

    if not batch_dirs:
        raise ValueError(f"No batch directories found in {input_dir}")

    print(f"Found {len(batch_dirs)} batch directories")

    # Infer batch size if not provided
    if batch_size is None:
        batch_size = get_batch_size_from_batches(input_path)
        print(f"Inferred batch size: {batch_size}")
    else:
        print(f"Using provided batch size: {batch_size}")

    # Combined results for gsw_results_combined.json
    combined_results = {
        "metadata": {
            "source_dir": str(input_path),
            "flattened_at": datetime.now().isoformat(),
            "batch_size": batch_size,
            "total_batches": len(batch_dirs),
            "total_documents": 0,
            "total_chunks": 0,
        },
        "documents": {}
    }

    total_docs_processed = 0
    total_chunks = 0

    for batch_dir in batch_dirs:
        # Extract batch index from directory name (batch_0000 -> 0)
        batch_idx = int(batch_dir.name.split("_")[1])
        global_doc_offset = batch_idx * batch_size

        print(f"\nProcessing {batch_dir.name} (global offset: {global_doc_offset})")

        # Process networks directory
        networks_dir = batch_dir / "networks"
        if networks_dir.exists():
            for doc_dir in sorted(networks_dir.iterdir()):
                if not doc_dir.is_dir() or not doc_dir.name.startswith("doc_"):
                    continue

                # Extract local doc index
                local_doc_idx = int(doc_dir.name.split("_")[1])
                global_doc_idx = global_doc_offset + local_doc_idx

                # Create output doc directory
                output_doc_dir = output_networks / f"doc_{global_doc_idx}"
                output_doc_dir.mkdir(exist_ok=True)

                # Process GSW files in this doc directory
                doc_data = {}
                for gsw_file in doc_dir.glob("gsw_*.json"):
                    # Parse filename: gsw_0_0.json -> gsw_{global_doc_idx}_0.json
                    parts = gsw_file.stem.replace("gsw_", "").split("_")
                    if len(parts) >= 2:
                        chunk_idx = parts[1]
                        new_filename = f"gsw_{global_doc_idx}_{chunk_idx}.json"

                        # Read and update the JSON content
                        with open(gsw_file, "r") as f:
                            gsw_data = json.load(f)

                        # Update IDs in the data if they exist
                        if "doc_idx" in gsw_data:
                            gsw_data["doc_idx"] = global_doc_idx
                        if "global_id" in gsw_data:
                            gsw_data["global_id"] = f"{global_doc_idx}_{chunk_idx}"

                        # Write to output
                        output_file = output_doc_dir / new_filename
                        with open(output_file, "w") as f:
                            json.dump(gsw_data, f, indent=2)

                        # Add to combined results
                        chunk_key = f"{global_doc_idx}_{chunk_idx}"
                        doc_data[chunk_key] = gsw_data
                        total_chunks += 1

                if doc_data:
                    combined_results["documents"][f"doc_{global_doc_idx}"] = doc_data
                    total_docs_processed += 1

        # Process networks_raw directory (same logic)
        networks_raw_dir = batch_dir / "networks_raw"
        if networks_raw_dir.exists():
            for doc_dir in sorted(networks_raw_dir.iterdir()):
                if not doc_dir.is_dir() or not doc_dir.name.startswith("doc_"):
                    continue

                local_doc_idx = int(doc_dir.name.split("_")[1])
                global_doc_idx = global_doc_offset + local_doc_idx

                output_doc_dir = output_networks_raw / f"doc_{global_doc_idx}"
                output_doc_dir.mkdir(exist_ok=True)

                for gsw_file in doc_dir.glob("gsw_*.json"):
                    parts = gsw_file.stem.replace("gsw_", "").split("_")
                    if len(parts) >= 2:
                        chunk_idx = parts[1]
                        new_filename = f"gsw_{global_doc_idx}_{chunk_idx}.json"

                        with open(gsw_file, "r") as f:
                            gsw_data = json.load(f)

                        if "doc_idx" in gsw_data:
                            gsw_data["doc_idx"] = global_doc_idx
                        if "global_id" in gsw_data:
                            gsw_data["global_id"] = f"{global_doc_idx}_{chunk_idx}"

                        output_file = output_doc_dir / new_filename
                        with open(output_file, "w") as f:
                            json.dump(gsw_data, f, indent=2)

        print(f"   Processed docs {global_doc_offset} - {global_doc_offset + batch_size - 1}")

    # Update metadata
    combined_results["metadata"]["total_documents"] = total_docs_processed
    combined_results["metadata"]["total_chunks"] = total_chunks

    # Save combined results
    combined_file = output_path / "gsw_results_combined.json"
    with open(combined_file, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Flattening completed!")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Total documents: {total_docs_processed}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Combined JSON: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten batched GSW output into a single folder with global doc IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Flatten with auto-detected batch size
  python flatten_batch_output.py \\
    /path/to/logs/full_2wiki_corpus_TIMESTAMP/gsw_output/corpus_TIMESTAMP \\
    /path/to/output/flattened_corpus

  # Flatten with explicit batch size
  python flatten_batch_output.py \\
    /path/to/corpus_TIMESTAMP \\
    /path/to/output \\
    --batch-size 10
        """
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to batched corpus output directory (contains batch_0000, batch_0001, ...)"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory for flattened structure"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of documents per batch. If not provided, will be inferred from first batch."
    )

    args = parser.parse_args()

    flatten_batch_output(args.input_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
