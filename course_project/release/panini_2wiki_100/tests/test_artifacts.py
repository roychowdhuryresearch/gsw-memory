from __future__ import annotations

import json

import numpy as np
import pytest

from panini_course.artifacts import EmbeddingTable


def test_embedding_table_loads_normalizes_and_searches(tmp_path):
    matrix_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "ids.json"
    np.save(
        matrix_path,
        np.asarray([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float16),
    )
    ids_path.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")

    table = EmbeddingTable.load(matrix_path, ids_path)

    assert table.dimension == 2
    assert table.search(np.asarray([1.0, 0.0]), top_k=2)[0][0] == "a"


def test_embedding_table_rejects_misaligned_ids(tmp_path):
    matrix_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "ids.json"
    np.save(matrix_path, np.ones((2, 3), dtype=np.float16))
    ids_path.write_text(json.dumps(["only-one"]), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        EmbeddingTable.load(matrix_path, ids_path)
