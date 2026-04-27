"""Unit tests for Search-R1's inline-tag parser helpers."""

from __future__ import annotations

from research_agent.adapters.baselines.search_r1 import SearchR1Adapter
from research_agent.adapters.base import AdapterContext
from research_agent.retrieval.corpus import ArticleCorpus, Chunk


def _stub_adapter(monkeypatch) -> SearchR1Adapter:
    """Build a Search-R1 adapter with a stubbed corpus + no LLM connection."""
    corpus = ArticleCorpus(chunks=[], by_title={})
    ctx = AdapterContext(
        system_id="search_r1",
        model_id="test-stub",
        model_name="test-stub",
        base_url="http://127.0.0.1:9",
        api_key="dummy",
        extra={"corpus": corpus, "retriever": None},
    )

    # Avoid instantiating LLMClient (no openai server).
    class _NoLLM:
        def chat(self, *a, **k):  # noqa: ANN001
            raise RuntimeError("not invoked in parser tests")

    # Subclass to avoid the __init__ network side-effects.
    class _TestAdapter(SearchR1Adapter):
        def __init__(self, ctx):  # noqa: D401
            self.ctx = ctx
            self.corpus = corpus

            class _R:
                def search(self, q, top_k=3):
                    return []

            self.retriever = _R()
            self.llm = _NoLLM()
            self._top_k = 3

    return _TestAdapter(ctx)


def test_extract_answer(monkeypatch):
    ad = _stub_adapter(monkeypatch)
    q, a = ad._extract_search_and_answer("<think>think</think><answer> Boeing 777 </answer>")
    assert q == ""
    assert a == "Boeing 777"


def test_extract_search(monkeypatch):
    ad = _stub_adapter(monkeypatch)
    q, a = ad._extract_search_and_answer(
        "<think>need to look up</think><search>Boeing 777 first flight date</search>"
    )
    assert q == "Boeing 777 first flight date"
    assert a == ""


def test_extract_neither(monkeypatch):
    ad = _stub_adapter(monkeypatch)
    q, a = ad._extract_search_and_answer("just plain text no tags")
    assert q == ""
    assert a == ""


def test_collect_reasoning(monkeypatch):
    ad = _stub_adapter(monkeypatch)
    reasoning = ad._collect_reasoning("<think>first thought</think> blah <think>second</think>")
    assert "first thought" in reasoning
    assert "second" in reasoning
