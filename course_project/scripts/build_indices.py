#!/usr/bin/env python3
"""Build portable dense, TF-IDF, and BM25 indices for the course package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import faiss
import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN_PATTERN = re.compile(r"\w+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def build_dense_index(matrix_path: Path, output_path: Path) -> dict[str, Any]:
    """Build a float16 scalar-quantized inner-product FAISS index."""

    matrix = np.load(matrix_path, allow_pickle=False, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected rank-2 matrix at {matrix_path}")
    dimension = int(matrix.shape[1])
    index = faiss.IndexScalarQuantizer(
        dimension,
        faiss.ScalarQuantizer.QT_fp16,
        faiss.METRIC_INNER_PRODUCT,
    )
    batch_size = 2048
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        index.add(np.asarray(matrix[start:stop], dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    return {
        "rows": int(matrix.shape[0]),
        "dimension": dimension,
        "metric": "inner_product",
        "storage": "faiss_scalar_quantizer_fp16",
    }


def build_tfidf_index(
    texts: Sequence[str],
    ids: Sequence[str],
    output: Path,
    name: str,
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    sparse.save_npz(output / f"{name}_tfidf.npz", matrix, compressed=True)
    joblib.dump(
        vectorizer,
        output / f"{name}_tfidf_vectorizer.joblib",
        compress=3,
    )
    write_json(output / f"{name}_ids.json", list(ids))
    return {
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "ngram_range": [1, 2],
        "sublinear_tf": True,
    }


def build_bm25_index(
    texts: Sequence[str],
    ids: Sequence[str],
    output: Path,
    name: str,
) -> dict[str, Any]:
    tokenized = [tokenize(text) for text in texts]
    index = BM25Okapi(tokenized)
    joblib.dump(index, output / f"{name}_bm25.joblib", compress=3)
    write_json(output / f"{name}_ids.json", list(ids))
    return {
        "rows": len(texts),
        "tokenizer": r"casefold + \w+",
        "k1": float(index.k1),
        "b": float(index.b),
        "epsilon": float(index.epsilon),
    }


def build_indices(args: argparse.Namespace) -> dict[str, Any]:
    package = args.package.resolve()
    output = package / "indices"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Index output is not empty: {output}. Pass --overwrite to replace."
        )
    output.mkdir(parents=True, exist_ok=True)

    entity_rows = load_jsonl(package / "metadata" / "entities.jsonl")
    qa_rows = load_jsonl(package / "metadata" / "qa_pairs.jsonl")
    tables = {
        "entity": {
            "ids": [row["entity_uid"] for row in entity_rows],
            "texts": [row["search_text"] for row in entity_rows],
        },
        "qa": {
            "ids": [row["qa_uid"] for row in qa_rows],
            "texts": [row["search_text"] for row in qa_rows],
        },
    }

    manifest: dict[str, Any] = {
        "dense": {},
        "tfidf": {},
        "bm25": {},
        "versions": {
            "faiss": faiss.__version__,
        },
    }
    for name, table in tables.items():
        dense_path = output / f"{name}_qwen3_8b_ip.faiss"
        manifest["dense"][name] = build_dense_index(
            package / "embeddings" / f"{name}_embeddings.npy",
            dense_path,
        )
        manifest["tfidf"][name] = build_tfidf_index(
            table["texts"], table["ids"], output, name
        )
        manifest["bm25"][name] = build_bm25_index(
            table["texts"], table["ids"], output, name
        )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "index_manifest.json"
    }
    manifest["artifacts"] = artifact_hashes
    write_json(output / "index_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build_indices(parse_args(argv))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
