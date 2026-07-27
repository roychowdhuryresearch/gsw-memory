from __future__ import annotations

from panini_course.retrieval import DualRetriever, SearchHit


class FakeIndex:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k):
        return self.hits[:top_k]


def test_dual_retriever_expands_entity_verbs_and_merges_direct_hits():
    entity_uid = "doc_1::gsw_1.json::e1"
    qa_rows = [
        {
            "qa_uid": "qa1",
            "document_id": "doc_1",
            "gsw_file": "gsw_1.json",
            "verb_phrase_id": "v1",
            "answer_local_ids": ["e1"],
        },
        {
            "qa_uid": "qa2",
            "document_id": "doc_1",
            "gsw_file": "gsw_1.json",
            "verb_phrase_id": "v1",
            "answer_local_ids": ["e2"],
        },
    ]
    retriever = DualRetriever(
        entity_index=FakeIndex(
            [SearchHit(entity_uid, 1.0, "entity", 1)]
        ),
        qa_index=FakeIndex([SearchHit("qa2", 0.9, "dense", 1)]),
        entity_rows=[{"entity_uid": entity_uid}],
        qa_rows=qa_rows,
    )

    hits = retriever.search("query", fused_top_k=2, rank_constant=0)

    assert hits[0].item_id == "qa2"
    assert {hit.item_id for hit in hits} == {"qa1", "qa2"}
