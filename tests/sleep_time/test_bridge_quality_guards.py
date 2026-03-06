from dataclasses import dataclass

from gsw_memory.sleep_time.tools import GSWTools


@dataclass
class _Entity:
    id: str
    name: str


class _GSW:
    def __init__(self, entities):
        self.entity_nodes = entities
        self.verb_phrase_nodes = []

    def get_entity_by_id(self, entity_id):
        for entity in self.entity_nodes:
            if entity.id == entity_id:
                return entity
        return None


class _Searcher:
    def __init__(self):
        self.gsw_by_doc_id = {
            "doc_0": _GSW([_Entity("e1", "Source"), _Entity("e2", "Neighbor")]),
            "doc_1": _GSW([_Entity("e1", "Target"), _Entity("e2", "Witness")]),
        }
        self.entities = [
            {"name": "Source", "doc_id": "doc_0"},
            {"name": "Neighbor", "doc_id": "doc_0"},
            {"name": "Target", "doc_id": "doc_1"},
            {"name": "Witness", "doc_id": "doc_1"},
        ]

    def search(self, query, top_k=10, verbose=False):
        return []


def _make_tools():
    return GSWTools(_Searcher())


def test_create_bridge_rejects_single_doc_refs_even_with_multi_source_docs():
    tools = _make_tools()

    result = tools.create_bridge_qa(
        question="When did Source's ally die?",
        answers=["doc_0::e1"],
        reverse_question="Whose ally died in 851?",
        reverse_answers=["doc_0::e2"],
        source_docs=["doc_0", "doc_1"],
        reasoning="Single-doc refs only",
    )

    assert result["success"] is False
    assert "not cross-document" in result["error"]


def test_create_bridge_allows_semantic_near_duplicate_for_verifier_decision():
    tools = _make_tools()

    first = tools.create_bridge_qa(
        question="When did Source's ally die?",
        answers=["doc_0::e1"],
        reverse_question="Whose ally died in 851?",
        reverse_answers=["doc_1::e1"],
        source_docs=["doc_0", "doc_1"],
        reasoning="Chain 1",
    )
    assert first["success"] is True

    second = tools.create_bridge_qa(
        question="When did the ally of Source die",
        answers=["doc_0::e1"],
        reverse_question="Whose ally passed away in 851?",
        reverse_answers=["doc_1::e1"],
        source_docs=["doc_0", "doc_1"],
        reasoning="Chain 2",
    )

    assert second["success"] is True
    assert second["validation"].get("similarity_signal")


def test_create_bridge_canonicalizes_source_docs_to_grounded_ref_docs():
    tools = _make_tools()

    result = tools.create_bridge_qa(
        question="When did Source's ally die?",
        answers=["doc_0::e1"],
        reverse_question="Whose ally died in 851?",
        reverse_answers=["doc_1::e1"],
        source_docs=["doc_0", "doc_1", "doc_9"],
        reasoning="Chain",
    )

    assert result["success"] is True
    stored = tools.bridges_created[-1]
    assert stored["source_docs"] == ["doc_0", "doc_1"]
