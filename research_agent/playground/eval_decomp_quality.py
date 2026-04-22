"""CLI: score an emitted decomposer module against gold decompositions.

Loads the module through the zero-LLM sandbox, runs the grader on the
dev split, and prints a small summary + a failure sample.

Usage:

    python playground/eval_decomp_quality.py \
        --module src/research_agent/decomposers/musique_specialized.py \
        --dataset musique \
        --data ../playground_data/musique_10_q.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from research_agent.decomposers import grade_decomposer, load_decomposer
from research_agent.decomposers._datasets import load_monaco, load_musique, split_train_dev


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True, help="Path to the emitted decomposer .py")
    p.add_argument("--dataset", choices=["musique", "twowiki", "frames", "monaco"], default="musique")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev_fraction", type=float, default=0.25)
    p.add_argument("--split", choices=["dev", "train", "all"], default="dev")
    p.add_argument("--per_call_timeout_s", type=float, default=2.0)
    return p.parse_args()


def _load_dataset(dataset: str, path: str, limit: int | None) -> list[dict]:
    if dataset == "musique":
        return load_musique(path, limit=limit)
    if dataset == "monaco":
        return load_monaco(path, limit=limit)
    raise NotImplementedError(f"Loader for {dataset!r} not implemented yet.")


def main() -> None:
    args = _parse_args()
    examples = _load_dataset(args.dataset, args.data, args.limit)
    train, dev = split_train_dev(examples, dev_fraction=args.dev_fraction, seed=args.seed)
    if args.split == "train":
        subset = train
    elif args.split == "all":
        subset = examples
    else:
        subset = dev

    fn = load_decomposer(args.module, per_call_timeout_s=args.per_call_timeout_s)
    score = grade_decomposer(fn, subset)

    print(f"[eval] module={args.module}")
    print(f"[eval] dataset={args.dataset} split={args.split} n={score['n']}")
    headline = {k: v for k, v in score.items() if k not in ("failures", "by_hop")}
    print(json.dumps(headline, indent=2))
    by_hop = score.get("by_hop") or {}
    if by_hop:
        print("[eval] per-hop breakdown:")
        for bucket in sorted(by_hop):
            row = by_hop[bucket]
            print(
                f"  {bucket:>7}  n={row['n']:>4}  "
                f"coverage={row['coverage']:.3f}  "
                f"count_match={row['count_match']:.3f}  "
                f"soft_match={row['soft_match']:.3f}"
            )
    if score.get("failures"):
        print("[eval] first failures:")
        for q, err in score["failures"]:
            print(f"  - {q!r}: {err}")


if __name__ == "__main__":
    main()
