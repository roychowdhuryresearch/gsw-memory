import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gsw_memory.query_then_sleep.query_agent import QueryAgent, QueryToolAdapter, run_query_agent_batch
from gsw_memory.query_then_sleep.transport import TransportResponse
from gsw_memory.sleep_time.curriculum import BatchQueryItem


def test_normalize_final_answer_reduces_yes_no_sentences():
    assert (
        QueryAgent._normalize_final_answer(
            "Were Jimmy Santiago Baca and Duane Armstrong from the same country?",
            "Yes, Jimmy Santiago Baca and Duane Armstrong are from the same country – the United States.",
        )
        == "yes"
    )


def test_normalize_final_answer_extracts_answer_marker():
    assert (
        QueryAgent._normalize_final_answer(
            "Where was the director of film En Aasai Rasave born?",
            "Answer: Mallingapuram.",
        )
        == "Mallingapuram"
    )


def test_extract_short_answer_candidate_strips_sentence_and_doc_ref():
    assert (
        QueryAgent._extract_short_answer_candidate(
            "Where did John Ii, Duke Of Cleves's wife die?",
            "John II's wife, Mathilde of Hesse, died in **Cologne**【doc_4626::e5】",
        )
        == "Cologne"
    )


def test_choose_final_answer_prefers_short_candidate_when_structured_answer_is_long():
    assert (
        QueryAgent._choose_final_answer(
            question="Who is the spouse of the director of film Pratisodh (2004 Film)?",
            synthesis_answer="The spouse of the director of Pratisodh (2004 Film) is Piya Sengupta.",
            synthesis_raw_text='{"answer": "The spouse of the director of Pratisodh (2004 Film) is Piya Sengupta."}',
            tool_loop_final_text="",
        )
        == "Piya Sengupta"
    )


class _FakeRegistry:
    def __init__(self, score=0.9):
        self.records = {"bridge_1": object()}
        self.score = score

    def search(self, query, top_k=10, embed_texts_fn=None):
        return [
            {
                "bridge_id": "bridge_1",
                "question": "Who directed Pride and Prejudice?",
                "answer_text": "Joe Wright",
                "score": self.score,
            }
        ]


class _FakeTransport:
    def __init__(self, *args, **kwargs):
        self.provider = "fake"

    def generate_json(self, **kwargs):
        stage = kwargs.get("stage")
        response = TransportResponse(provider="fake", model_name="fake", text="{}")
        if stage == "decomposition":
            return {"sub_questions": [{"sub_question": "Atomic?", "relation_hint": "other"}]}, response
        return {
            "answer": "Joe Wright",
            "reasoning_chain": "",
            "sub_questions": [],
            "found_relations": [],
            "missing_relations": [],
            "bridge_ids_used": ["bridge_1"],
        }, response

    def chat(self, *args, **kwargs):
        return TransportResponse(provider="fake", model_name="fake", text="Joe Wright")


class _FakeTransportNoBridge(_FakeTransport):
    def generate_json(self, **kwargs):
        payload, response = super().generate_json(**kwargs)
        if kwargs.get("stage") == "final_synthesis":
            payload = dict(payload)
            payload["bridge_ids_used"] = []
        return payload, response


class _FakeSearcher:
    entities = []
    gsw_by_doc_id = {}


def test_low_score_auto_bridge_is_seen_but_not_injected(monkeypatch):
    monkeypatch.setattr("gsw_memory.query_then_sleep.query_agent.StageTransport", _FakeTransportNoBridge)
    item = BatchQueryItem(question="Who composed a score?", doc_ids=["doc_0"], gold_answer="Joe Wright")
    tools = QueryToolAdapter(entity_searcher=_FakeSearcher(), bridge_registry=_FakeRegistry(score=0.18))
    agent = QueryAgent("fake-model", bridge_inject_min_score=0.30)

    trace = agent.answer_question(item=item, tools=tools)

    assert trace.bridge_evidence_used is True
    assert trace.bridge_evidence_injected is False
    assert trace.bridge_evidence_used_in_answer is False
    assert trace.tool_calls == []
    audit = trace.metadata["bridge_auto_retrieval"]
    assert audit["attempted"] is True
    assert audit["raw_hit_count"] == 1
    assert audit["injected_hit_count"] == 0
    assert audit["max_score"] == 0.18
    assert audit["skipped_reason"] == "no_hit_above_threshold"
    assert audit["raw_hits"][0]["bridge_id"] == "bridge_1"
    assert audit["raw_hits"][0]["injected"] is False


def test_bridge_answer_usage_rate_tracks_synthesis_usage(monkeypatch):
    monkeypatch.setattr("gsw_memory.query_then_sleep.query_agent.StageTransport", _FakeTransport)
    result = run_query_agent_batch(
        batch_index=0,
        batch_items=[BatchQueryItem(question="Who directed Pride and Prejudice?", doc_ids=["doc_0"], gold_answer="Joe Wright")],
        entity_searcher=_FakeSearcher(),
        model_name="fake-model",
        bridge_registry=_FakeRegistry(score=0.9),
    )

    trace = result.traces[0]
    assert trace.bridge_evidence_injected is True
    assert trace.bridge_evidence_used_in_answer is True
    audit = trace.metadata["bridge_auto_retrieval"]
    assert audit["attempted"] is True
    assert audit["raw_hit_count"] == 1
    assert audit["injected_hit_count"] == 1
    assert audit["skipped_reason"] == ""
    assert audit["raw_hits"][0]["injected"] is True
    assert audit["injected_hits"][0]["bridge_id"] == "bridge_1"
    assert result.overall_metrics["bridge_answer_usage_rate"] == 1.0
