#!/usr/bin/env python3
"""
Plot a histogram of final entity scores from doc_entities.json.

final_score = llm_importance (1-5) × idf(entity)
where idf = log(N / df), N = total docs, df = docs containing the entity.

Usage:
    python playground/plot_entity_scores.py
    python playground/plot_entity_scores.py --input doc_entities.json --output entity_score_histogram.png
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="doc_entities.json")
    p.add_argument("--output", default="entity_score_histogram.png")
    return p.parse_args()


def main():
    args = parse_args()

    results = json.load(open(args.input))
    N = len(results)

    # Compute IDF
    df: Counter = Counter()
    for entry in results.values():
        for e in entry["entities"]:
            df[e["name"]] += 1
    idf = {name: math.log(N / count) for name, count in df.items()}

    # Collect all final scores + metadata
    scores = []
    types = []
    for entry in results.values():
        for e in entry["entities"]:
            score = e["importance"] * idf.get(e["name"], 0.0)
            scores.append(score)
            types.append(e.get("type", "OTHER"))

    scores = np.array(scores)
    print(f"Entities : {len(scores)}")
    print(f"Min      : {scores.min():.3f}")
    print(f"Max      : {scores.max():.3f}")
    print(f"Mean     : {scores.mean():.3f}")
    print(f"Median   : {np.median(scores):.3f}")
    print(f"Unique values: {len(np.unique(scores.round(3)))}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = 40
    counts, bin_edges, patches = ax.hist(
        scores, bins=n_bins,
        color="#4C8BCC", edgecolor="white", linewidth=0.5, alpha=0.9
    )

    # Colour bars by score: low = orange (common/penalised), high = blue (distinctive)
    score_range = scores.max() - scores.min()
    cmap = plt.cm.RdYlGn
    for patch, left in zip(patches, bin_edges[:-1]):
        norm = (left - scores.min()) / score_range if score_range > 0 else 0.5
        patch.set_facecolor(cmap(norm))

    # Mean line
    ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean = {scores.mean():.1f}")
    ax.axvline(np.median(scores), color="gray", linestyle=":", linewidth=1.5,
               label=f"Median = {np.median(scores):.1f}")

    ax.set_xlabel("Final Score  (LLM importance × IDF)", fontsize=13)
    ax.set_ylabel("Number of entities", fontsize=13)
    ax.set_title(
        f"Entity Score Distribution  (N={N} docs, {len(scores)} entities)\n"
        "Low score = common across docs (penalised)  |  High score = distinctive to doc",
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved → {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
