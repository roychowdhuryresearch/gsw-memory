"""Build a curriculum manifest from 2wiki_platinum for the continual learning experiment.

Selects questions stratified by family, splits into train + probe,
and outputs a manifest JSON for run_curriculum_guidance_experiment.py.

Usage:
    python playground/sleep_time/build_curriculum_manifest.py \
        --dataset playground_data/2wiki_platinum.json \
        --num_per_family 20 \
        --probe_per_family 3 \
        --batch_size 8 \
        --output manifests/2wiki_150q_continual.json
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


COMPARISON_WORDS = [
    "older", "younger", "earlier", "first", "longer", "later",
    "which film", "which album", "which university",
    "who is older", "who is younger", "who lived longer",
]


def classify_family(question: str) -> str:
    """Classify a question into a relation family."""
    text = question.lower()
    if "place of birth" in text or ("born" in text and ("where" in text or "place" in text)):
        return "birth_place"
    if "place of death" in text or ("died" in text and "where" in text) or ("die" in text and "where" in text):
        return "death_place"
    if "place of burial" in text or "buried" in text:
        return "burial_place"
    if "spouse" in text or "husband" in text or "wife" in text:
        return "spouse"
    if "father" in text or "paternal" in text:
        return "paternal_parent"
    if "mother" in text or "maternal" in text:
        return "maternal_parent"
    if "child" in text or "son of" in text or "daughter of" in text:
        return "child"
    if "country" in text or "nationality" in text:
        return "nationality"
    if "work at" in text or "work for" in text or "study" in text or "educated" in text:
        return "affiliation"
    if "when" in text or "date" in text:
        return "date"
    return "other"


def is_comparison(question: str) -> bool:
    """Check if a question is a comparison question (not a lookup)."""
    text = question.lower()
    return any(w in text for w in COMPARISON_WORDS)


def build_manifest(
    dataset_path: str,
    full_dataset_path: str = "playground_data/2wikimultihopqa.json",
    corpus_path: str = "playground_data/2wikimultihopqa_corpus.json",
    gsw_path: str = "",
    num_per_family: int = 20,
    probe_per_family: int = 3,
    batch_size: int = 8,
    seed: int = 42,
    min_family_size: int = 15,
) -> dict:
    """Build a curriculum manifest with train + probe splits.

    Selects questions from ``dataset_path`` (e.g. 2wiki_platinum) but resolves
    their full 10-doc context from ``full_dataset_path`` (2wikimultihopqa.json).
    Optionally verifies that the corresponding GSW directories exist on disk.
    """
    random.seed(seed)

    with open(dataset_path) as f:
        data = json.load(f)

    # Load the full dataset for context resolution
    with open(full_dataset_path) as f:
        full_data = json.load(f)
    full_by_id = {q["_id"]: q for q in full_data}

    # Build corpus title → doc index mapping for GSW verification
    title_to_idx: dict[str, int] = {}
    if corpus_path:
        try:
            with open(corpus_path) as f:
                corpus = json.load(f)
            for i, doc in enumerate(corpus):
                title_to_idx[doc.get("title", "")] = i
        except FileNotFoundError:
            print(f"  Warning: corpus file not found at {corpus_path}")

    # Filter: lookup only (no Yes/No, no comparisons)
    lookup_qs = [
        q for q in data
        if q["answer"] not in ("Yes", "No") and not is_comparison(q["question"])
    ]

    # Resolve full context and verify GSWs
    resolved = []
    skipped_no_full = 0
    skipped_no_gsw = 0
    for q in lookup_qs:
        full_q = full_by_id.get(q["_id"])
        if not full_q:
            skipped_no_full += 1
            continue
        full_titles = [c[0] for c in full_q["context"]]
        # Verify GSW directories exist
        if gsw_path and title_to_idx:
            missing = [t for t in full_titles if t not in title_to_idx]
            if missing:
                skipped_no_gsw += 1
                continue
            doc_indices = [title_to_idx[t] for t in full_titles]
            missing_dirs = [
                f"doc_{idx}" for idx in doc_indices
                if not Path(gsw_path).joinpath(f"doc_{idx}").exists()
            ]
            if missing_dirs:
                skipped_no_gsw += 1
                continue
        resolved.append({"platinum": q, "full": full_q, "full_titles": full_titles})

    if skipped_no_full:
        print(f"  Skipped {skipped_no_full} questions (not found in full dataset)")
    if skipped_no_gsw:
        print(f"  Skipped {skipped_no_gsw} questions (missing GSW directories)")
    print(f"  Resolved {len(resolved)} questions with full 10-doc context")

    # Classify into families
    by_family: dict[str, list] = defaultdict(list)
    for item in resolved:
        family = classify_family(item["platinum"]["question"])
        by_family[family].append(item)

    # Select families with enough questions
    eligible_families = sorted(
        [f for f, qs in by_family.items() if len(qs) >= min_family_size]
    )
    print(f"Eligible families (>= {min_family_size} questions): {eligible_families}")
    for f in eligible_families:
        print(f"  {f}: {len(by_family[f])} available")

    # Sample and split
    manifest_questions = []
    order_index = 0
    total_train = 0
    total_probe = 0

    for family in eligible_families:
        pool = by_family[family][:]
        random.shuffle(pool)

        take = min(num_per_family, len(pool))
        selected = pool[:take]

        train_count = take - probe_per_family
        train_qs = selected[:train_count]
        probe_qs = selected[train_count:]

        # Train questions first (they feed guidance)
        for item in train_qs:
            q = item["platinum"]
            manifest_questions.append({
                "order_index": order_index,
                "original_index": full_data.index(item["full"]),
                "question_id": q["_id"],
                "question": q["question"],
                "family": family,
                "split": "train",
                "answer": q["answer"],
                "answer_aliases": q.get("answer_aliases", []),
                "context_titles": item["full_titles"],
            })
            order_index += 1
            total_train += 1

        # Probe questions (held out, evaluated after every batch)
        for item in probe_qs:
            q = item["platinum"]
            manifest_questions.append({
                "order_index": order_index,
                "original_index": full_data.index(item["full"]),
                "question_id": q["_id"],
                "question": q["question"],
                "family": family,
                "split": "probe",
                "answer": q["answer"],
                "answer_aliases": q.get("answer_aliases", []),
                "context_titles": item["full_titles"],
            })
            order_index += 1
            total_probe += 1

    num_batches = (total_train + batch_size - 1) // batch_size

    # Verify doc counts
    doc_counts = [len(q["context_titles"]) for q in manifest_questions]

    manifest = {
        "dataset": "2wikimultihopqa",
        "source_file": str(dataset_path),
        "questions_path": str(full_dataset_path),
        "corpus_path": str(corpus_path),
        "experiment_type": "continual_learning",
        "family_label": f"continual_{len(eligible_families)}fam_{total_train + total_probe}q",
        "curriculum_batch_size": batch_size,
        "curriculum_seed_batch_size": batch_size,
        "total_questions": total_train + total_probe,
        "total_train": total_train,
        "total_probe": total_probe,
        "num_families": len(eligible_families),
        "families": eligible_families,
        "num_batches": num_batches,
        "seed": seed,
        "questions": manifest_questions,
    }

    print(f"\nManifest summary:")
    print(f"  Families: {len(eligible_families)}")
    print(f"  Train: {total_train}")
    print(f"  Probe: {total_probe}")
    print(f"  Total: {total_train + total_probe}")
    print(f"  Batches: {num_batches}")
    print(f"  Docs per question: min={min(doc_counts)}, max={max(doc_counts)}, avg={sum(doc_counts)/len(doc_counts):.1f}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build curriculum manifest for continual learning experiment")
    parser.add_argument("--dataset", default="playground_data/2wiki_platinum.json", help="Source dataset (question selection)")
    parser.add_argument("--full_dataset", default="playground_data/2wikimultihopqa.json", help="Full dataset (10-doc context resolution)")
    parser.add_argument("--corpus_path", default="playground_data/2wikimultihopqa_corpus.json", help="Corpus JSON for title→doc_id mapping")
    parser.add_argument("--gsw_path", default="", help="GSW networks directory (for verification). Leave empty to skip.")
    parser.add_argument("--num_per_family", type=int, default=20, help="Questions per family")
    parser.add_argument("--probe_per_family", type=int, default=3, help="Probe questions per family")
    parser.add_argument("--batch_size", type=int, default=8, help="Curriculum batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_family_size", type=int, default=15, help="Minimum questions in a family to include")
    parser.add_argument("--output", default="manifests/2wiki_continual.json", help="Output manifest path")
    args = parser.parse_args()

    manifest = build_manifest(
        dataset_path=args.dataset,
        full_dataset_path=args.full_dataset,
        corpus_path=args.corpus_path,
        gsw_path=args.gsw_path,
        num_per_family=args.num_per_family,
        probe_per_family=args.probe_per_family,
        batch_size=args.batch_size,
        seed=args.seed,
        min_family_size=args.min_family_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
