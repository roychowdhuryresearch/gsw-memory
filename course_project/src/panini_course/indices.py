"""Load and search the supplied dense and lexical indices."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse

from .retrieval import SearchHit


TOKEN_PATTERN = re.compile(r"\w+")


def _load_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Expected a JSON string list at {path}")
    return value


@dataclass
class DenseIndex:
    """A supplied FAISS inner-product index aligned to stable IDs."""

    index: Any
    ids: list[str]
    source: str = "dense"

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        ids_path: str | Path,
        *,
        source: str = "dense",
    ) -> "DenseIndex":
        try:
            import faiss
        except ImportError as error:
            raise ImportError(
                "DenseIndex requires faiss-cpu; install requirements-colab.txt"
            ) from error
        index = faiss.read_index(str(index_path))
        ids = _load_ids(Path(ids_path))
        if index.ntotal != len(ids):
            raise ValueError(
                f"FAISS rows ({index.ntotal}) do not match IDs ({len(ids)})"
            )
        return cls(index=index, ids=ids, source=source)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index "
                f"dimension {self.index.d}"
            )
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        if float(norm[0, 0]) <= 1e-12:
            raise ValueError("Query embedding has zero norm")
        query /= norm
        scores, indices = self.index.search(query, min(top_k, len(self.ids)))
        return [
            SearchHit(
                item_id=self.ids[int(index)],
                score=float(score),
                source=self.source,
                rank=rank,
            )
            for rank, (score, index) in enumerate(
                zip(scores[0], indices[0]), start=1
            )
            if index >= 0
        ]


@dataclass
class TfidfIndex:
    """A supplied TF-IDF vectorizer and sparse document matrix."""

    vectorizer: Any
    matrix: Any
    ids: list[str]
    source: str = "tfidf"

    @classmethod
    def load(
        cls,
        matrix_path: str | Path,
        vectorizer_path: str | Path,
        ids_path: str | Path,
        *,
        source: str = "tfidf",
    ) -> "TfidfIndex":
        matrix = sparse.load_npz(matrix_path)
        vectorizer = joblib.load(vectorizer_path)
        ids = _load_ids(Path(ids_path))
        if matrix.shape[0] != len(ids):
            raise ValueError("TF-IDF matrix and ID count do not match")
        return cls(vectorizer=vectorizer, matrix=matrix, ids=ids, source=source)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().reshape(-1)
        count = min(top_k, len(self.ids))
        candidates = np.argpartition(-scores, count - 1)[:count]
        ordered = candidates[np.argsort(-scores[candidates])]
        return [
            SearchHit(
                item_id=self.ids[int(index)],
                score=float(scores[index]),
                source=self.source,
                rank=rank,
            )
            for rank, index in enumerate(ordered, start=1)
        ]


@dataclass
class BM25Index:
    """A supplied rank-bm25 index aligned to stable IDs."""

    index: Any
    ids: list[str]
    source: str = "bm25"

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        ids_path: str | Path,
        *,
        source: str = "bm25",
    ) -> "BM25Index":
        index = joblib.load(index_path)
        ids = _load_ids(Path(ids_path))
        if len(index.doc_len) != len(ids):
            raise ValueError("BM25 index and ID count do not match")
        return cls(index=index, ids=ids, source=source)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        tokens = TOKEN_PATTERN.findall(query.casefold())
        scores = np.asarray(self.index.get_scores(tokens), dtype=np.float32)
        count = min(top_k, len(self.ids))
        candidates = np.argpartition(-scores, count - 1)[:count]
        ordered = candidates[np.argsort(-scores[candidates])]
        return [
            SearchHit(
                item_id=self.ids[int(index)],
                score=float(scores[index]),
                source=self.source,
                rank=rank,
            )
            for rank, index in enumerate(ordered, start=1)
        ]


@dataclass
class QueryEmbeddingStore:
    """Exact text lookup for instructor-provided fixed query embeddings."""

    by_text: dict[str, np.ndarray]
    by_id: dict[str, np.ndarray]

    @classmethod
    def load(
        cls,
        matrix_path: str | Path,
        ids_path: str | Path,
        queries_path: str | Path,
    ) -> "QueryEmbeddingStore":
        matrix = np.load(matrix_path, allow_pickle=False)
        ids = _load_ids(Path(ids_path))
        with Path(queries_path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if matrix.shape[0] != len(ids) or len(rows) != len(ids):
            raise ValueError("Query embedding artifacts are not aligned")
        row_by_id = {row["query_id"]: row for row in rows}
        if set(row_by_id) != set(ids):
            raise ValueError("Query metadata and ID sets do not match")
        vectors = np.asarray(matrix, dtype=np.float32)
        by_id = {query_id: vectors[index] for index, query_id in enumerate(ids)}
        by_text = {
            row_by_id[query_id]["text"]: by_id[query_id] for query_id in ids
        }
        return cls(by_text=by_text, by_id=by_id)

    def get(self, text: str) -> np.ndarray:
        try:
            return self.by_text[text]
        except KeyError as error:
            raise KeyError(
                "No supplied embedding for this query. In a required run this "
                "means the decomposition or RICR trace diverged from the "
                "specified deterministic pipeline."
            ) from error
