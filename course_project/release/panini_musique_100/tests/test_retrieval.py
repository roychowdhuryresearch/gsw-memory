from __future__ import annotations

import pytest

from panini_course.retrieval import SearchHit, reciprocal_rank_fusion


def hit(item_id: str, rank: int, source: str) -> SearchHit:
    return SearchHit(
        item_id=item_id,
        score=1.0 / rank,
        source=source,
        rank=rank,
    )


def test_rrf_rewards_items_found_by_multiple_retrievers():
    dense = [hit("a", 1, "dense"), hit("b", 2, "dense")]
    sparse = [hit("b", 1, "bm25"), hit("c", 2, "bm25")]

    results = reciprocal_rank_fusion(
        [dense, sparse], rank_constant=0.0
    )

    assert results[0].item_id == "b"
    assert results[0].metadata["fusion_sources"] == ("dense", "bm25")


def test_rrf_validates_arguments():
    with pytest.raises(ValueError, match="rank_constant"):
        reciprocal_rank_fusion([], rank_constant=-1)
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], top_k=-1)
