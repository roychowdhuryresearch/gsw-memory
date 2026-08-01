"""Load and validate instructor-provided project artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSONL records from *path*."""

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected an object at {path}:{line_number}, "
                    f"found {type(value).__name__}"
                )
            yield value


@dataclass(frozen=True)
class EmbeddingTable:
    """An aligned ID list and dense embedding matrix supplied by instructors."""

    ids: tuple[str, ...]
    matrix: np.ndarray

    @classmethod
    def load(
        cls,
        matrix_path: str | Path,
        ids_path: str | Path,
        *,
        normalize: bool = True,
    ) -> "EmbeddingTable":
        matrix = np.load(Path(matrix_path), allow_pickle=False)
        with Path(ids_path).open(encoding="utf-8") as handle:
            ids_value = json.load(handle)
        if not isinstance(ids_value, list) or not all(
            isinstance(value, str) for value in ids_value
        ):
            raise TypeError("Embedding IDs must be a JSON list of strings")
        if matrix.ndim != 2:
            raise ValueError(
                f"Embedding matrix must be rank 2, found shape {matrix.shape}"
            )
        if len(ids_value) != matrix.shape[0]:
            raise ValueError(
                f"ID count ({len(ids_value)}) does not match embedding rows "
                f"({matrix.shape[0]})"
            )
        if len(set(ids_value)) != len(ids_value):
            raise ValueError("Embedding IDs must be unique")
        matrix = np.asarray(matrix, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return cls(ids=tuple(ids_value), matrix=matrix)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Return top cosine-similarity matches for a supplied query vector."""

        if top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match "
                f"index dimension {self.dimension}"
            )
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 1e-12:
            raise ValueError("Query embedding has zero norm")
        scores = self.matrix @ (query / query_norm)
        count = min(top_k, len(self.ids))
        if count == len(self.ids):
            indices = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, count - 1)[:count]
            indices = candidates[np.argsort(-scores[candidates])]
        return [(self.ids[int(index)], float(scores[index])) for index in indices]
