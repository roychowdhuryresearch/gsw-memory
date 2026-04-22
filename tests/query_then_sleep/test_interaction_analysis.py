import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gsw_memory.query_then_sleep.interaction_analysis import analyze_query_interactions
from gsw_memory.query_then_sleep.types import QueryInteractionTrace, QuerySubQuestionTrace


def test_analyze_query_interactions_summarizes_gaps():
    traces = [
        QueryInteractionTrace(
            question_id="q1",
            question="Where was the place of death of Lothair II's mother?",
            doc_ids=["doc_0", "doc_1"],
            sub_questions=[
                QuerySubQuestionTrace(sub_question="Who is Lothair II's mother?", relation_hint="mother", answer_found=True),
                QuerySubQuestionTrace(sub_question="Where did Ermengarde of Tours die?", relation_hint="death_place", answer_found=False),
            ],
            final_answer="",
            found_relations=["mother"],
            missing_relations=["death_place for Ermengarde of Tours"],
            f1=0.0,
        ),
        QueryInteractionTrace(
            question_id="q2",
            question="Who is the spouse of Fernando Cortes?",
            doc_ids=["doc_2"],
            sub_questions=[
                QuerySubQuestionTrace(sub_question="Who is Fernando Cortes married to?", relation_hint="spouse", answer_found=False),
            ],
            final_answer="",
            missing_relations=["spouse for Fernando Cortes"],
            f1=0.0,
        ),
    ]

    summary = analyze_query_interactions(traces, batch_index=0)

    assert summary.guidance_mode == "interaction_v1"
    assert summary.questions_answered == 2
    assert summary.relation_gap_profile[0]["relation_label"] in {"death_place", "spouse"}
    assert any(row["relation_label"] == "death_place" for row in summary.relation_gap_profile)
    assert any(row["entity"] == "Ermengarde of Tours" for row in summary.entity_gap_targets)
    assert summary.bridge_chain_profile
    assert any("bridge" in row["chain_label"] or "links" in row["chain_label"] for row in summary.bridge_chain_profile)
    assert summary.undercovered_chain_families
    assert "bridge" in summary.natural_language_summary.lower() or "links" in summary.natural_language_summary.lower()


def test_analyze_query_interactions_uses_llm_summary_when_available(monkeypatch):
    traces = [
        QueryInteractionTrace(
            question_id="q1",
            question="Who is Fernando Cortes married to?",
            doc_ids=["doc_2"],
            sub_questions=[
                QuerySubQuestionTrace(sub_question="Who is Fernando Cortes married to?", relation_hint="spouse", answer_found=False),
            ],
            final_answer="",
            missing_relations=["spouse for Fernando Cortes"],
            f1=0.0,
        ),
    ]

    class _DummyResponse:
        def as_log_dict(self):
            return {"provider": "bedrock", "model_name": "dummy"}

    class _DummyTransport:
        def __init__(self, *args, **kwargs):
            pass

        def generate_json(self, **kwargs):
            return (
                {
                    "natural_language_summary": "Recent query traces show under-covered spouse links for film-director chains. Focus next-batch bridge generation on those missing connecting hops and deprioritize already-covered direct creator lookups."
                },
                _DummyResponse(),
            )

    monkeypatch.setattr("gsw_memory.query_then_sleep.interaction_analysis.StageTransport", _DummyTransport)

    summary = analyze_query_interactions(
        traces,
        batch_index=0,
        model_name="bedrock/openai.gpt-oss-120b-1:0",
    )

    assert summary.natural_language_summary.startswith("Recent query traces show under-covered spouse links")
