#!/usr/bin/env python3
"""
Create Training Data from Golden GSWs + Musique Raw Text

This script combines raw text paragraphs from the Musique dataset with their
corresponding golden GSW structures to create training data for fine-tuning.

Usage:
    python create_training_data.py \
        --musique-path playground_data/musique_full_v1.0_train.jsonl \
        --golden-gsw-dir /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini/ \
        --output-path gsw_training_data.json \
        --use-train-set \
        --num-samples 1000
"""

import argparse
import json
import glob
import os
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def sort_natural_key(text):
    """
    Natural sorting key function for sorting strings with numbers.
    E.g., doc_1, doc_2, doc_10 instead of doc_1, doc_10, doc_2
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', str(text))]


def load_musique_corpus(musique_path: str, is_train: bool = False):
    """
    Load Musique corpus data and create a dictionary mapping global_id to text.

    Args:
        musique_path: Path to musique JSON or JSONL file
        is_train: Whether this is training data (JSONL) or test data (JSON)

    Returns:
        Dictionary mapping global_id to document data
    """
    print(f"Loading Musique corpus from {musique_path}...")
    print(f"  Format: {'JSONL (training)' if is_train else 'JSON (test)'}")

    # Load data based on format
    if is_train:
        # Load JSONL format for training data
        musique_data = []
        with open(musique_path) as f:
            for line in f:
                musique_data.append(json.loads(line))
    else:
        # Load JSON format for test data
        with open(musique_path) as f:
            musique_data = json.load(f)

    # Build corpus as ordered list (to match sequential doc_* indexing)
    corpus = []
    for data in musique_data:
        paragraphs = data["paragraphs"]
        for paragraph in paragraphs:
            global_id = f"{data['id']}_{paragraph['idx']}"
            corpus.append({
                "global_id": global_id,
                "title": paragraph["title"],
                "text": paragraph["title"] + "\n" + paragraph["paragraph_text"],
                "question_id": data["id"],
                "paragraph_idx": paragraph["idx"]
            })

    print(f"✓ Loaded {len(corpus)} paragraphs from Musique corpus (ordered)")
    return corpus


def load_golden_gsws(golden_gsw_dir: str, num_samples: int = None):
    """
    Load golden GSW structures from the directory.

    Args:
        golden_gsw_dir: Path to directory containing doc_* subdirectories
        num_samples: Maximum number of documents to load (None = all)

    Returns:
        List of tuples: (doc_identifier, gsw_data)
    """
    print(f"Loading golden GSWs from {golden_gsw_dir}...")

    # Find all doc_* directories
    doc_dirs = sorted(
        glob.glob(f"{golden_gsw_dir}/doc_*"),
        key=sort_natural_key
    )

    if num_samples:
        doc_dirs = doc_dirs[:num_samples]

    print(f"  Found {len(doc_dirs)} doc directories")

    golden_gsws = []
    for doc_dir in tqdm(doc_dirs, desc="Loading golden GSWs"):
        if not os.path.isdir(doc_dir):
            continue

        # Get doc identifier (e.g., "doc_0" -> "0")
        doc_id = os.path.basename(doc_dir).replace("doc_", "")

        # Load all JSON files in this directory
        json_files = sorted(glob.glob(os.path.join(doc_dir, "*.json")))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    gsw_data = json.load(f)

                # Extract the gsw filename pattern (e.g., "gsw_0_0.json" -> "0_0")
                gsw_filename = os.path.basename(json_file)
                # Remove "gsw_" prefix and ".json" suffix
                gsw_id = gsw_filename.replace("gsw_", "").replace(".json", "")

                # Parse the sequential index from gsw_id (e.g., "0_0" -> 0 for first part)
                # This matches the doc_* directory index
                seq_idx = int(gsw_id.split("_")[0])

                golden_gsws.append({
                    "doc_id": doc_id,
                    "gsw_id": gsw_id,
                    "seq_idx": seq_idx,  # Sequential index for matching with corpus
                    "gsw": gsw_data
                })

            except Exception as e:
                print(f"  ✗ Error loading {json_file}: {e}")

    print(f"✓ Loaded {len(golden_gsws)} golden GSW structures")
    return golden_gsws


def match_text_with_gsws(corpus: List[Dict], golden_gsws: List[Dict]):
    """
    Match raw text paragraphs with their corresponding golden GSW structures.
    Uses sequential indexing: corpus[i] matches golden_gsws with seq_idx=i.

    Args:
        corpus: Ordered list of document data from Musique
        golden_gsws: List of golden GSW structures with seq_idx

    Returns:
        List of training examples with format: {global_id, text, gsw}
    """
    print("\nMatching text paragraphs with golden GSWs using sequential indexing...")

    training_data = []
    matched_count = 0
    unmatched_gsws = []

    for gsw_entry in tqdm(golden_gsws, desc="Matching GSWs with text"):
        seq_idx = gsw_entry["seq_idx"]

        # Check if index is within corpus bounds
        if seq_idx < len(corpus):
            # Match found!
            corpus_entry = corpus[seq_idx]
            training_data.append({
                "global_id": corpus_entry["global_id"],
                "text": corpus_entry["text"],
                "title": corpus_entry["title"],
                "gsw": gsw_entry["gsw"]
            })
            matched_count += 1
        else:
            unmatched_gsws.append(f"seq_idx={seq_idx} (out of bounds, corpus size={len(corpus)})")

    print(f"\n{'='*60}")
    print(f"Matching Results:")
    print(f"{'='*60}")
    print(f"  Corpus size: {len(corpus)}")
    print(f"  Total GSWs: {len(golden_gsws)}")
    print(f"  Matched: {matched_count} ({matched_count/len(golden_gsws)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched_gsws)}")

    if unmatched_gsws and len(unmatched_gsws) <= 10:
        print(f"\nUnmatched GSWs (sample):")
        for uid in unmatched_gsws[:10]:
            print(f"  - {uid}")

    return training_data


def save_training_data(training_data: List[Dict], output_path: str):
    """
    Save training data to JSON file.

    Args:
        training_data: List of training examples
        output_path: Path to output JSON file
    """
    print(f"\nSaving training data to {output_path}...")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    # Calculate file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

    print(f"\n{'='*60}")
    print(f"Training Data Saved Successfully!")
    print(f"{'='*60}")
    print(f"  Output file: {output_path}")
    print(f"  Number of examples: {len(training_data)}")
    print(f"  File size: {file_size:.2f} MB")

    # Show sample stats
    if training_data:
        sample = training_data[0]
        print(f"\nSample entry:")
        print(f"  Global ID: {sample['global_id']}")
        print(f"  Text length: {len(sample['text'])} chars")
        print(f"  Entity nodes: {len(sample['gsw'].get('entity_nodes', []))}")
        print(f"  Verb phrase nodes: {len(sample['gsw'].get('verb_phrase_nodes', []))}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Create training data from golden GSWs and Musique raw text"
    )

    # Input arguments
    parser.add_argument(
        "--musique-path",
        type=str,
        default="/home/yigit/codebase/gsw-memory/playground_data/musique_full_v1.0_train.jsonl",
        help="Path to Musique JSON or JSONL file"
    )
    parser.add_argument(
        "--golden-gsw-dir",
        type=str,
        default="/mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini/",
        help="Directory containing golden GSW structures (doc_* subdirectories)"
    )

    # Output arguments
    parser.add_argument(
        "--output-path",
        type=str,
        default="playground/gsw_creation_local/gsw_training_data.json",
        help="Output path for training data JSON file"
    )

    # Processing arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--use-train-set",
        action="store_true",
        help="Use training set (JSONL) instead of test set (JSON)"
    )

    args = parser.parse_args()

    print("="*60)
    print("GSW TRAINING DATA CREATION")
    print("="*60)
    print(f"Musique path: {args.musique_path}")
    print(f"Golden GSW dir: {args.golden_gsw_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Data source: {'Training set (JSONL)' if args.use_train_set else 'Test set (JSON)'}")
    if args.num_samples:
        print(f"Max samples: {args.num_samples}")
    print("="*60 + "\n")

    # Step 1: Load Musique corpus (raw text)
    corpus = load_musique_corpus(args.musique_path, is_train=args.use_train_set)

    # Step 2: Load golden GSWs
    golden_gsws = load_golden_gsws(args.golden_gsw_dir, num_samples=args.num_samples)

    # Step 3: Match text with GSWs
    training_data = match_text_with_gsws(corpus, golden_gsws)

    # Step 4: Save training data
    if training_data:
        save_training_data(training_data, args.output_path)
    else:
        print("\n✗ No training data created (no matches found)")
        print("  Check that the Musique corpus and golden GSWs are properly aligned")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
