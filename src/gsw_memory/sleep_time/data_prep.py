"""
Data preparation for RLVR training.

Assumes GSWs are already generated on disk (via playground/generate_gsws_multihop.py
+ playground/copy_gsw_with_global_ids.py). This script pairs each MuSiQue questions
example with the GSW directories for its supporting paragraphs and writes a
training index.

Mapping:
    paragraph["idx"]  →  doc_{idx}  →  gsw_path/doc_{idx}/gsw_*.json

Input files:
    --musique:  musique.json  (questions file — has question/answer/decomposition)
                Accepts both plain JSON array and JSONL formats.
    --gsw_path: /path/to/networks_4_1_mini  (contains doc_0/, doc_1/, ...)

Usage:
    python -m gsw_memory.sleep_time.data_prep \\
        --musique  playground_data/musique.json \\
        --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \\
        --output   data/rl_training/index.json \\
        --limit    5000
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_musique_questions(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load answerable MuSiQue examples from either:
      - A plain JSON array file (musique.json, musique_50_q.json, ...)
      - A JSONL file (musique_ans_v1.0_train.jsonl)
    """
    with open(path) as f:
        first_char = f.read(1)

    if first_char == "[":
        # Plain JSON array
        with open(path) as f:
            all_examples = json.load(f)
        if limit is not None:
            all_examples = all_examples[:limit]
        return [d for d in all_examples if d.get("answerable", True)]
    else:
        # JSONL — one JSON object per line
        examples = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if limit is not None and i >= limit:
                    break
                d = json.loads(line)
                if d.get("answerable", True):
                    examples.append(d)
        return examples


def supporting_doc_indices(example: Dict[str, Any]) -> List[int]:
    """
    Return corpus indices of supporting paragraphs.
    paragraph["idx"] == corpus position == doc_{idx} in GSW directory.
    """
    return [p["idx"] for p in example["paragraphs"] if p["is_supporting"]]


def gsw_dir_for_doc(gsw_path: str, doc_idx: int) -> str:
    """Return the GSW directory path for a given document index."""
    return os.path.join(gsw_path, f"doc_{doc_idx}")


def gsw_exists(gsw_dir: str) -> bool:
    """Check whether at least one GSW JSON file exists in the directory."""
    d = Path(gsw_dir)
    if not d.exists():
        return False
    return any(d.glob("gsw_*.json"))


def get_hop_count(example: Dict[str, Any]) -> int:
    """Return the number of hops (decomposition steps) in a MuSiQue example."""
    return len(example.get("question_decomposition", []))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build RLVR training index from MuSiQue questions + pre-generated GSWs"
    )
    parser.add_argument(
        "--musique",
        default="playground_data/musique.json",
        help="Path to MuSiQue questions file (.json array or .jsonl)",
    )
    parser.add_argument(
        "--gsw_path",
        required=True,
        help="Directory containing doc_i/ GSW subdirectories (e.g. networks_4_1_mini/)",
    )
    parser.add_argument(
        "--output",
        default="data/rl_training/index.json",
        help="Output path for the training index JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of examples to read from questions file (None = all)",
    )
    parser.add_argument(
        "--min_hops",
        type=int,
        default=2,
        help="Only include examples with at least this many hops (default: 2)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MuSiQue questions from: {args.musique}")
    examples = load_musique_questions(args.musique, limit=args.limit)
    print(f"Loaded {len(examples)} answerable examples")

    index: List[Dict[str, Any]] = []
    missing_gsw = 0
    skipped_hops = 0

    for ex in examples:
        hops = get_hop_count(ex)
        if hops < args.min_hops:
            skipped_hops += 1
            continue

        # Each supporting paragraph maps to a doc_i GSW directory
        support_indices = supporting_doc_indices(ex)
        gsw_dirs = [gsw_dir_for_doc(args.gsw_path, idx) for idx in support_indices]

        # Only include example if ALL supporting GSW dirs exist
        missing = [d for d in gsw_dirs if not gsw_exists(d)]
        if missing:
            missing_gsw += 1
            continue

        # Supporting paragraph metadata for reference
        supporting = [
            {"title": p["title"], "text": p["paragraph_text"], "doc_idx": p["idx"]}
            for p in ex["paragraphs"]
            if p["is_supporting"]
        ]

        entry: Dict[str, Any] = {
            "id": ex["id"],
            # List of GSW dirs — one per supporting paragraph doc
            # EntitySearcher should be initialized with all of these
            "gsw_dirs": gsw_dirs,
            "support_doc_indices": support_indices,
            "question": ex["question"],
            "answer": ex["answer"],
            "answer_aliases": ex.get("answer_aliases", []),
            "decomposition": ex.get("question_decomposition", []),
            "num_hops": hops,
            "supporting_docs": supporting,
        }
        index.append(entry)

    print(f"\nIndex built:")
    print(f"  Included:          {len(index)}")
    print(f"  Missing GSW:       {missing_gsw}")
    print(f"  Skipped (<{args.min_hops} hops): {skipped_hops}")

    hop_counts = Counter(e["num_hops"] for e in index)
    for hops, count in sorted(hop_counts.items()):
        print(f"  {hops}-hop examples: {count}")

    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved index to: {out_path}")


if __name__ == "__main__":
    main()
