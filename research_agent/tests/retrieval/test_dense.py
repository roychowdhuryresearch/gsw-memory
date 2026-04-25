"""Unit tests for DenseRetriever + HybridRetriever.

Tests run with a stubbed encoder so we don't load Qwen3-8B every time.
The real encoder is exercised by the smoke script, not unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub corpus
# ---------------------------------------------------------------------------


@dataclass
class _Chunk:
    chunk_id: str
    title: str
    text: str
    char_start: int = 0
    char_end: int = 0


class _StubCorpus:
    """Minimal ArticleCorpus-compatible stub."""

    def __init__(self, chunks: list[_Chunk]) -> None:
        self.chunks = chunks
        self._by_title = {c.title: c.text for c in chunks}
        self._by_id = {c.chunk_id: c for c in chunks}

    def get_chunk(self, chunk_id: str):
        return self._by_id.get(chunk_id)

    def article_text(self, title: str) -> str:
        return self._by_title.get(title, "")


# ---------------------------------------------------------------------------
# Stub SentenceTransformer — monkey-patched in
# ---------------------------------------------------------------------------


class _StubST:
    """Deterministic tiny encoder: maps each text to a 4-dim vector
    via a hash-based projection. Not semantic — just stable for tests."""

    def __init__(self, model_name: str, device: str = "cpu", **kwargs) -> None:
        # Accept (and ignore) model_kwargs, torch_dtype, etc.
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts,
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ):
        out = []
        for t in texts:
            h = abs(hash(t))
            vec = np.array(
                [(h >> (8 * i)) & 0xFF for i in range(4)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                vec = vec / (np.linalg.norm(vec) + 1e-12)
            out.append(vec)
        return np.vstack(out)


@pytest.fixture
def stub_corpus():
    return _StubCorpus(
        [
            _Chunk("c1", "Picasso", "Pablo Picasso died in 1973 in France."),
            _Chunk("c2", "PinkFloyd", "Pink Floyd released Dark Side of the Moon in 1973."),
            _Chunk("c3", "Rivers", "The Amazon flows through South America."),
        ]
    )


@pytest.fixture
def patch_sentence_transformer(monkeypatch):
    """Install the stub encoder in place of the real SentenceTransformer."""
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _StubST)
    yield


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


def test_dense_retriever_builds_and_searches(
    stub_corpus, patch_sentence_transformer, tmp_path
):
    from research_agent.retrieval.dense import DenseRetriever

    r = DenseRetriever(
        stub_corpus,
        model_name="fake/stub-model",
        cache_dir=tmp_path / "cache",
        device="cpu",
    )
    hits = r.search("Pablo Picasso", top_k=2)
    assert len(hits) == 2
    # Each hit has the interface the adapters expect.
    for h in hits:
        assert hasattr(h, "chunk")
        assert hasattr(h, "score")
        assert isinstance(h.score, float)
        assert h.chunk.chunk_id in {"c1", "c2", "c3"}


def test_dense_retriever_cache_roundtrip(
    stub_corpus, patch_sentence_transformer, tmp_path
):
    """Build once, then reload from disk — second init should not rebuild."""
    from research_agent.retrieval.dense import DenseRetriever

    cache_dir = tmp_path / "cache"
    r1 = DenseRetriever(
        stub_corpus, model_name="fake/stub-model", cache_dir=cache_dir, device="cpu"
    )
    hits1 = r1.search("Picasso", top_k=3)

    # Reload — should pick up cached files.
    assert (cache_dir / "chunk_embeddings.npy").exists()
    assert (cache_dir / "chunk_ids.pkl").exists()
    assert (cache_dir / "faiss.index").exists()

    r2 = DenseRetriever(
        stub_corpus, model_name="fake/stub-model", cache_dir=cache_dir, device="cpu"
    )
    hits2 = r2.search("Picasso", top_k=3)
    # Same scores, same order.
    assert [h.chunk.chunk_id for h in hits1] == [h.chunk.chunk_id for h in hits2]


def test_dense_retriever_top_k_respects_corpus_size(
    stub_corpus, patch_sentence_transformer, tmp_path
):
    from research_agent.retrieval.dense import DenseRetriever

    r = DenseRetriever(
        stub_corpus, model_name="fake/stub-model", cache_dir=tmp_path, device="cpu"
    )
    hits = r.search("anything", top_k=100)
    # Corpus has 3 chunks; can't return more than that.
    assert len(hits) == 3


# ---------------------------------------------------------------------------
# HybridRetriever RRF
# ---------------------------------------------------------------------------


class _FakeBM25:
    """Returns a fixed ranking per query."""

    def __init__(self, ranking: dict[str, list[_Chunk]]):
        self.ranking = ranking

    def search(self, query, *, top_k: int = 5):
        from research_agent.retrieval.bm25 import RetrievedChunk

        chunks = self.ranking.get(query, [])[:top_k]
        return [RetrievedChunk(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(chunks)]


class _FakeDense:
    def __init__(self, ranking: dict[str, list[_Chunk]]):
        self.ranking = ranking

    def search(self, query, *, top_k: int = 5):
        from research_agent.retrieval.bm25 import RetrievedChunk

        chunks = self.ranking.get(query, [])[:top_k]
        return [RetrievedChunk(chunk=c, score=0.9 - i * 0.05) for i, c in enumerate(chunks)]


def test_hybrid_retriever_rrf_math(stub_corpus):
    """A chunk at rank 1 in both retrievers gets RRF ≈ 2/(k+1).
    A chunk only in one retriever gets ≈ 1/(k+rank)."""
    from research_agent.retrieval.dense import HybridRetriever

    c1 = stub_corpus.chunks[0]
    c2 = stub_corpus.chunks[1]
    c3 = stub_corpus.chunks[2]

    bm25 = _FakeBM25({"q": [c1, c2, c3]})  # c1 rank 1, c2 rank 2, c3 rank 3
    dense = _FakeDense({"q": [c3, c1, c2]})  # c3 rank 1, c1 rank 2, c2 rank 3

    h = HybridRetriever(bm25, dense, k=60, per_retriever_top_k=3)
    hits = h.search("q", top_k=3)

    # Compute expected RRF manually.
    k = 60
    expected = {
        "c1": 1 / (k + 1) + 1 / (k + 2),
        "c2": 1 / (k + 2) + 1 / (k + 3),
        "c3": 1 / (k + 3) + 1 / (k + 1),
    }
    # c1 and c3 have the same score (both at ranks 1+2); c2 has a lower score.
    assert hits[0].chunk.chunk_id in {"c1", "c3"}
    assert hits[1].chunk.chunk_id in {"c1", "c3"}
    assert hits[2].chunk.chunk_id == "c2"
    for h in hits:
        assert abs(h.score - expected[h.chunk.chunk_id]) < 1e-9


def test_hybrid_retriever_handles_disjoint_rankings(stub_corpus):
    """Chunks unique to one retriever should still appear in the fused list."""
    from research_agent.retrieval.dense import HybridRetriever

    c1 = stub_corpus.chunks[0]
    c2 = stub_corpus.chunks[1]
    bm25 = _FakeBM25({"q": [c1]})  # c1 only
    dense = _FakeDense({"q": [c2]})  # c2 only

    h = HybridRetriever(bm25, dense, k=60, per_retriever_top_k=5)
    hits = h.search("q", top_k=5)
    ids = [h.chunk.chunk_id for h in hits]
    assert set(ids) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_retriever_factory(stub_corpus, patch_sentence_transformer, tmp_path):
    from research_agent.retrieval.dense import build_retriever

    r_bm25 = build_retriever("bm25", stub_corpus)
    assert r_bm25.__class__.__name__ == "BM25Retriever"

    r_dense = build_retriever(
        "dense",
        stub_corpus,
        model_name="fake/stub-model",
        cache_dir=tmp_path / "dense",
        device="cpu",
    )
    assert r_dense.__class__.__name__ == "DenseRetriever"

    r_hybrid = build_retriever(
        "hybrid",
        stub_corpus,
        model_name="fake/stub-model",
        cache_dir=tmp_path / "hybrid",
        device="cpu",
    )
    assert r_hybrid.__class__.__name__ == "HybridRetriever"

    with pytest.raises(ValueError):
        build_retriever("bogus", stub_corpus)
