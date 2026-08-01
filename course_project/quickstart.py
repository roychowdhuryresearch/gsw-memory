#!/usr/bin/env python3
"""Smoke-test a standalone Panini course release without loading an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from panini_course import (
    BM25Index,
    CoursePackage,
    DenseIndex,
    QueryEmbeddingStore,
    TfidfIndex,
)


def render_hits(label, hits, metadata):
    print(f"\n{label}")
    for hit in hits:
        row = metadata[hit.item_id]
        print(
            f"{hit.rank:>2}. {hit.score: .4f} | "
            f"{row.get('question', row.get('text', ''))} -> "
            f"{row.get('answer_names', row.get('name', ''))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    package = CoursePackage(args.package)
    question = package.questions("public")[0]["question"]
    query_store = QueryEmbeddingStore.load(
        package.root / "embeddings" / "query_embeddings.npy",
        package.root / "embeddings" / "query_ids.json",
        package.root / "embeddings" / "queries.jsonl",
    )
    query_vector = query_store.get(question)
    qa_metadata = package.metadata_by_id("qa")

    dense = DenseIndex.load(
        package.root / "indices" / "qa_qwen3_8b_ip.faiss",
        package.root / "indices" / "qa_ids.json",
    )
    tfidf = TfidfIndex.load(
        package.root / "indices" / "qa_tfidf.npz",
        package.root / "indices" / "qa_tfidf_vectorizer.joblib",
        package.root / "indices" / "qa_ids.json",
    )
    bm25 = BM25Index.load(
        package.root / "indices" / "qa_bm25.joblib",
        package.root / "indices" / "qa_ids.json",
    )

    print(f"Package counts: {package.manifest['counts']}")
    print(f"Question: {question}")
    render_hits("Dense QA retrieval", dense.search(query_vector, 3), qa_metadata)
    render_hits("TF-IDF QA retrieval", tfidf.search(question, 3), qa_metadata)
    render_hits("BM25 QA retrieval", bm25.search(question, 3), qa_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
