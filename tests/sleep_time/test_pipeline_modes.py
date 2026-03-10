from gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler


class _DummySearcher:
    def __init__(self):
        self.gsw_by_doc_id = {"doc_0": object()}
        self.entities = [{"name": "Source", "doc_id": "doc_0"}]

    def search(self, query, top_k=10, verbose=False):
        return []


def _build_agent(monkeypatch, pipeline_mode: str = "legacy", hybrid_scope: str = "doc_edge"):
    def _fake_init(self, model_name, base_url=None):
        self.provider = "openai"
        self.client = None
        self.litellm = None

    monkeypatch.setattr(AgenticReconciler, "_initialize_client", _fake_init)
    return AgenticReconciler(
        entity_searcher=_DummySearcher(),
        model_name="gpt-4o-mini",
        verbose=False,
        bridge_verifier_enabled=False,
        pipeline_mode=pipeline_mode,
        hybrid_scope=hybrid_scope,
    )


def test_hybrid_scope_defaults_when_invalid(monkeypatch):
    agent = _build_agent(monkeypatch, pipeline_mode="hybrid", hybrid_scope="invalid_scope")
    assert agent.pipeline_mode == "hybrid"
    assert agent.hybrid_scope == "doc_edge"


def test_run_exploration_dispatches_to_hybrid_corpus(monkeypatch):
    agent = _build_agent(monkeypatch, pipeline_mode="hybrid")
    called = {"hybrid": 0}

    def _fake_run_hybrid_corpus():
        called["hybrid"] += 1
        return {"mode": "hybrid"}

    monkeypatch.setattr(agent, "run_hybrid_corpus", _fake_run_hybrid_corpus)
    result = agent.run_exploration()

    assert result["mode"] == "hybrid"
    assert called["hybrid"] == 1


def test_explore_entity_dispatches_doc_id_to_hybrid_document(monkeypatch):
    agent = _build_agent(monkeypatch, pipeline_mode="hybrid")
    called = {"doc": 0}

    def _fake_run_hybrid_document(doc_id):
        called["doc"] += 1
        return {"mode": "hybrid", "entity": doc_id}

    monkeypatch.setattr(agent, "run_hybrid_document", _fake_run_hybrid_document)
    result = agent.explore_entity(doc_id="doc_0")

    assert result["mode"] == "hybrid"
    assert result["entity"] == "doc_0"
    assert called["doc"] == 1
