"""Smoke tests for the FRAMES corpus + BM25 retriever."""

from __future__ import annotations

import pytest

from research_agent.retrieval import BM25Retriever, load_frames_corpus


@pytest.fixture(scope="module")
def corpus():
    return load_frames_corpus()


def test_corpus_loads_and_has_chunks(corpus):
    assert len(corpus) > 0
    titles = corpus.titles()
    assert len(titles) > 100
    sample = titles[0]
    assert len(corpus.article_text(sample)) > 0


def test_bm25_retrieves_relevant_titles(corpus):
    retriever = BM25Retriever(corpus)
    hits = retriever.search("Tyson Fury heavyweight boxing", top_k=5)
    assert hits, "expected at least one hit for boxer query"
    assert any("Tyson Fury" in h.chunk.title for h in hits)


def test_bm25_empty_query_returns_nothing(corpus):
    retriever = BM25Retriever(corpus)
    assert retriever.search("", top_k=5) == []
