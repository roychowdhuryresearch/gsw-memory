"""
Convert RLVR training index to parquet files for veRL.

veRL requires .parquet files with these columns:
    prompt       — formatted chat prompt (list of message dicts)
    ground_truth — gold answer string
    data_source  — dataset name string
    extra_info   — JSON string with answer_aliases + decomposition

Usage:
    python -m gsw_memory.sleep_time.make_parquet \\
        --index   data/rl_training/index.json \\
        --output  data/rl_training/ \\
        --val_split 0.05
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: pip install pandas pyarrow")

from .prompts import SLEEP_TIME_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(question: str, gsw_dirs: List[str]) -> List[Dict[str, str]]:
    """
    Format a training example as a chat prompt for veRL.

    The GSW directory paths are embedded in the user message so the
    reward function (via extra_info) knows where to load the GSW from.
    The agent is not shown the paths — they're only used by the environment.
    """
    user_content = (
        f"Question: {question}\n\n"
        f"Explore the GSW corpus to find multi-hop bridge connections "
        f"that help answer this question. Use the available tools systematically."
    )
    return [
        {"role": "system", "content": SLEEP_TIME_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert RLVR training index to veRL parquet format"
    )
    parser.add_argument(
        "--index",
        default="data/rl_training/index.json",
        help="Path to training index JSON (from data_prep.py)",
    )
    parser.add_argument(
        "--output",
        default="data/rl_training/",
        help="Output directory for train.parquet and val.parquet",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of examples to use for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading index from: {args.index}")
    with open(args.index) as f:
        index: List[Dict[str, Any]] = json.load(f)
    print(f"Loaded {len(index)} examples")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(index)
    n_val = max(1, int(len(index) * args.val_split))
    val_examples = index[:n_val]
    train_examples = index[n_val:]
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    def to_rows(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for ex in examples:
            prompt = format_prompt(ex["question"], ex.get("gsw_dirs", []))
            gsw_dirs = ex.get("gsw_dirs", [])
            answer_aliases = ex.get("answer_aliases", [])
            decomposition = ex.get("decomposition", [])
            extra_info = json.dumps({
                # veRL tool_agent_loop reads interaction_kwargs from here and
                # passes them verbatim to GSWInteraction.start_interaction()
                "interaction_kwargs": {
                    "name": "gsw",              # must match name in gsw_interaction.yaml
                    "ground_truth": ex["answer"],
                    "gsw_dirs": gsw_dirs,
                    "answer_aliases": answer_aliases,
                    "decomposition": decomposition,
                },
                # Legacy fields (used by verl_reward.py if custom_reward_function is re-enabled)
                "answer_aliases": answer_aliases,
                "decomposition": decomposition,
                "gsw_dirs": gsw_dirs,
                "support_doc_indices": ex.get("support_doc_indices", []),
            })
            rows.append({
                "prompt": json.dumps(prompt),   # serialized chat messages
                "ground_truth": ex["answer"],
                "data_source": "musique",
                "extra_info": extra_info,
                # Metadata (not used by veRL directly, useful for analysis)
                "id": ex["id"],
                "num_hops": ex.get("num_hops", 2),
            })
        return rows

    train_df = pd.DataFrame(to_rows(train_examples))
    val_df = pd.DataFrame(to_rows(val_examples))

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    train_df.to_parquet(str(train_path), index=False)
    val_df.to_parquet(str(val_path), index=False)

    print(f"Saved: {train_path}  ({len(train_df)} rows)")
    print(f"Saved: {val_path}  ({len(val_df)} rows)")
    print(f"\nColumns: {list(train_df.columns)}")


if __name__ == "__main__":
    main()
