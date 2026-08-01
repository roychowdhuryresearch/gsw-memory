#!/usr/bin/env python3
"""Merge reference-run instantiated-query vectors into a course package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def stable_query_id(text: str) -> str:
    return f"query::{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def merge(package: Path, cache: Path) -> dict[str, int]:
    embeddings = package / "embeddings"
    matrix_path = embeddings / "query_embeddings.npy"
    ids_path = embeddings / "query_ids.json"
    rows_path = embeddings / "queries.jsonl"
    matrix = np.load(matrix_path, allow_pickle=False)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    rows = load_jsonl(rows_path)
    if len(ids) != len(rows) or matrix.shape[0] != len(ids):
        raise ValueError("existing query artifacts are not aligned")

    existing_text = {row["text"] for row in rows}
    existing_ids = set(ids)
    additions: list[tuple[str, str, np.ndarray]] = []
    for record in load_jsonl(cache / "query_vector_manifest.jsonl"):
        text = str(record["query"])
        if text in existing_text:
            continue
        query_id = stable_query_id(text)
        if query_id in existing_ids:
            raise ValueError(f"query ID collision for {text}")
        vector = np.load(cache / record["file"], allow_pickle=False).reshape(-1)
        if vector.shape[0] != matrix.shape[1]:
            raise ValueError(f"dimension mismatch for {text}")
        additions.append((query_id, text, vector.astype(matrix.dtype)))
        existing_text.add(text)
        existing_ids.add(query_id)

    additions.sort(key=lambda item: item[0])
    if additions:
        matrix = np.concatenate(
            [matrix, np.stack([item[2] for item in additions])], axis=0
        )
        ids.extend(item[0] for item in additions)
        rows.extend(
            {
                "query_id": query_id,
                "text": text,
                "sources": [{"kind": "reference_ricr_instantiation"}],
            }
            for query_id, text, _vector in additions
        )
        np.save(matrix_path, matrix, allow_pickle=False)
        write_json(ids_path, ids)
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = embeddings / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"]["query"]["rows"] = len(ids)
    manifest["reference_runtime_queries_added"] = len(additions)
    manifest["student_query_embedding_generation_required"] = False
    write_json(manifest_path, manifest)
    return {"existing": len(ids) - len(additions), "added": len(additions), "total": len(ids)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.package.resolve(), args.cache.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
