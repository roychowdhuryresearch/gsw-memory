"""CLI: run the Script-Writer offline to emit a deterministic decomposer.

Usage:

    # MuSiQue specialized — uses inline question_decomposition as supervision
    python playground/synthesize_decomposer.py \
        --mode specialized --dataset musique \
        --data ../playground_data/musique_10_q.json \
        --out src/research_agent/decomposers/musique_specialized.py \
        --model gpt-4.1-mini

    # Cross-dataset generalized — pools examples from multiple datasets
    python playground/synthesize_decomposer.py \
        --mode generalized \
        --data ../playground_data/musique_10_q.json \
        --out src/research_agent/decomposers/generalized.py \
        --model gpt-4.1-mini

The emitted module is loaded through the zero-LLM sandbox (no openai /
anthropic / requests imports allowed) and graded on a dev split before
being persisted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the in-repo package importable without install
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from research_agent.decomposers._datasets import (
    load_monaco,
    load_monaco_stratified,
    load_musique,
    load_musique_stratified,
    split_train_dev,
)
from research_agent.decomposers._synthesis import synthesize_decomposer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["specialized"],
        default="specialized",
        help="Always specialized — cross-dataset generalized mode was dropped.",
    )
    p.add_argument(
        "--dataset",
        choices=["musique", "monaco"],
        default="musique",
        help="Target dataset. Selects per-dataset worked examples in the preamble.",
    )
    p.add_argument("--data", required=True, help="Path to dataset JSON (shape depends on dataset).")
    p.add_argument("--out", required=True, help="Path to write the emitted decomposer module.")
    p.add_argument("--model", required=True, help="Model identifier for the script-writer LLM calls.")
    p.add_argument("--base_url", default="", help="Optional OpenAI-compatible base URL (vLLM, etc.).")
    p.add_argument("--reasoning_effort", default="medium")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of examples loaded.")
    p.add_argument(
        "--stratified_per_bucket",
        type=int,
        default=0,
        help="If >0, sample this many questions from EACH hop bucket (2hop/3hop/4hop). "
        "Overrides --limit. Only honoured for MuSiQue so far.",
    )
    p.add_argument("--max_iterations", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument(
        "--seed_module",
        default="",
        help="Path to an existing decomposer .py to seed the loop. "
        "Iter 0 takes the repair branch against this module's dev-failures.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev_fraction", type=float, default=0.25)
    p.add_argument(
        "--work_dir",
        default="",
        help="Directory for per-iteration draft files (kept for auditing). Temp dir if empty.",
    )
    return p.parse_args()


def _load_dataset(
    dataset: str,
    path: str,
    limit: int | None,
    *,
    stratified_per_bucket: int = 0,
    seed: int = 0,
) -> list[dict]:
    if dataset == "musique":
        if stratified_per_bucket > 0:
            return load_musique_stratified(
                path, per_bucket=stratified_per_bucket, seed=seed
            )
        return load_musique(path, limit=limit)
    if dataset == "monaco":
        if stratified_per_bucket > 0:
            return load_monaco_stratified(
                path, per_bucket=stratified_per_bucket, seed=seed
            )
        return load_monaco(path, limit=limit)
    raise NotImplementedError(
        f"Loader for dataset {dataset!r} not yet implemented. "
        "Supported: musique, monaco."
    )


def main() -> None:
    args = _parse_args()

    mode_label = f"specialized/{args.dataset}"

    examples = _load_dataset(
        args.dataset,
        args.data,
        args.limit,
        stratified_per_bucket=args.stratified_per_bucket,
        seed=args.seed,
    )
    train, dev = split_train_dev(examples, dev_fraction=args.dev_fraction, seed=args.seed)
    print(f"[synth] loaded {len(examples)} examples — train={len(train)} dev={len(dev)}")

    result = synthesize_decomposer(
        mode_label=mode_label,
        train_examples=train,
        dev_examples=dev,
        out_path=args.out,
        model_name=args.model,
        base_url=args.base_url or None,
        max_iterations=args.max_iterations,
        patience=args.patience,
        reasoning_effort=args.reasoning_effort,
        work_dir=args.work_dir or None,
        seed_module_path=args.seed_module or None,
    )

    print("\n=== iteration table ===")
    print(result.as_markdown_table())
    print()
    print("=== sample decompositions per iteration ===")
    for rec in result.iterations:
        print(f"--- iter {rec.iteration} (error: {rec.error or '(none)'}) ---")
        for smp in rec.samples:
            print(f"[{smp.get('hop', '?')}] {smp.get('question', '')}")
            for g in smp.get("gold", []):
                print(f"   gold : {g.get('question', '')}  -> {g.get('answer', '')}")
            for p in smp.get("pred", []):
                refs = p.get("references") or []
                tag = f" [refs={','.join(refs)}]" if refs else ""
                if "error" in p:
                    print(f"   pred : ERROR {p['error']}")
                else:
                    print(f"   pred : {p.get('sub_id', '?')}: {p.get('question', '')!r}{tag}")
            print()
    print(f"best_iteration: {result.best_iteration}")
    print(f"best_score: {result.best_score}")
    print(f"emitted: {result.best_module_path}")


if __name__ == "__main__":
    main()
