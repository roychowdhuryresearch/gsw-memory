"""Tests for chain_models Pydantic data models."""

from gsw_memory.qa.chain_models import (
    ChainFollowingResult,
    DecomposedQuestion,
    DecomposedQuestionList,
    QAPairResult,
)


class TestDecomposedQuestion:
    def test_basic_creation(self):
        q = DecomposedQuestion(
            question="Who directed Casablanca?", requires_retrieval=True
        )
        assert q.question == "Who directed Casablanca?"
        assert q.requires_retrieval is True

    def test_serialization_roundtrip(self):
        q = DecomposedQuestion(question="What year?", requires_retrieval=False)
        data = q.model_dump()
        q2 = DecomposedQuestion.model_validate(data)
        assert q == q2


class TestDecomposedQuestionList:
    def test_list_creation(self):
        ql = DecomposedQuestionList(
            questions=[
                DecomposedQuestion(question="Q1?", requires_retrieval=True),
                DecomposedQuestion(question="Q2?", requires_retrieval=False),
            ]
        )
        assert len(ql.questions) == 2

    def test_json_roundtrip(self):
        ql = DecomposedQuestionList(
            questions=[DecomposedQuestion(question="Q1?", requires_retrieval=True)]
        )
        json_str = ql.model_dump_json()
        ql2 = DecomposedQuestionList.model_validate_json(json_str)
        assert ql == ql2


class TestQAPairResult:
    def test_defaults(self):
        qa = QAPairResult(question="Who?")
        assert qa.answer_names == []
        assert qa.similarity_score == 0.0

    def test_full_creation(self):
        qa = QAPairResult(
            question="Who directed the film?",
            answer_names=["Robert Zemeckis"],
            answer_ids=["e1"],
            verb_phrase="directed",
            source_file="/tmp/test.json",
            similarity_score=0.95,
        )
        assert qa.answer_names == ["Robert Zemeckis"]
        assert qa.similarity_score == 0.95


class TestChainFollowingResult:
    def test_basic(self):
        r = ChainFollowingResult(question="test?", answer="42")
        assert r.question == "test?"
        assert r.answer == "42"
        assert r.evidence == []
        assert r.time_taken == 0.0

    def test_full(self):
        r = ChainFollowingResult(
            question="What?",
            answer="Something",
            evidence=["Q: Who? A: Alice"],
            evidence_count=1,
            decomposed_questions=[{"question": "Q1?", "requires_retrieval": True}],
            chains_info={"total_chains": 1},
            time_taken=1.5,
        )
        assert r.evidence_count == 1
        assert r.chains_info["total_chains"] == 1
