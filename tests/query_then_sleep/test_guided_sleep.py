import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gsw_memory.query_then_sleep.guided_sleep import (
    InteractionGuidedRecursiveEdgeWorker,
    build_sleep_guidance_payload,
    format_interaction_guidance_block,
    prioritize_edges_with_reserve,
)
from gsw_memory.sleep_time.rlm_pipeline import EdgeKey, EdgePacket


class _DummyReconciler:
    pass


def test_prioritize_edges_with_reserve_promotes_targeted_edges():
    guidance = build_sleep_guidance_payload(
        {
            "relation_gap_profile": [{"relation_label": "death_place", "asked": 3, "found": 1, "missing": 2}],
            "entity_gap_targets": [{"entity": "Ermengarde of Tours", "relation_label": "death_place", "count": 1}],
            "supporting_examples": [{"question": "Where did Ermengarde die?", "missing_relation": "death_place"}],
        },
        targeted_edge_ratio=0.7,
    )
    targeted = EdgeKey("doc_1", "Ermengarde of Tours", "Tours", "death_place")
    exploratory = EdgeKey("doc_9", "Other Person", "Other Place", "title")

    ordered = prioritize_edges_with_reserve([exploratory, targeted], guidance)

    assert ordered[0] == targeted
    assert exploratory in ordered


def test_deprioritized_relations_are_penalized_not_targeted():
    guidance = build_sleep_guidance_payload(
        {
            "relation_gap_profile": [
                {"relation_label": "spouse", "asked": 8, "found": 8, "missing": 0},
                {"relation_label": "birth_place", "asked": 6, "found": 2, "missing": 4},
            ],
            "deprioritized_chain_families": [
                {"relation_labels": ["spouse"], "found": 8},
            ],
        }
    )
    birth_edge = EdgeKey("doc_1", "Joan", "Paris", "birth_place")
    spouse_edge = EdgeKey("doc_1", "Joan", "Partner", "spouse")

    ordered = prioritize_edges_with_reserve([spouse_edge, birth_edge], guidance)

    assert guidance["sleep_focus"]["target_relations"] == ["birth_place"]
    assert "spouse" in guidance["sleep_focus"]["deprioritized_relations"]
    assert ordered[0] == birth_edge


def test_guided_worker_includes_interaction_guidance_block():
    guidance = build_sleep_guidance_payload(
        {
            "natural_language_summary": "Focus on death_place bridges.",
            "relation_gap_profile": [{"relation_label": "death_place", "asked": 3, "found": 1, "missing": 2}],
            "entity_gap_targets": [{"entity": "Ermengarde of Tours", "relation_label": "death_place", "count": 1}],
        }
    )
    packet = EdgePacket(
        edge=EdgeKey("doc_1", "Ermengarde of Tours", "Tours", "death_place"),
        source_docs=["doc_1", "doc_2"],
        mandatory_docs=["doc_2"],
        optional_docs=[],
        source_context={"qa_pairs": []},
        mandatory_contexts=[{"doc_id": "doc_2", "qa_pairs": []}],
        optional_contexts=[],
        constraints={"curriculum_guidance": guidance},
        budget={"edge_max_tokens": 2000},
    )
    worker = InteractionGuidedRecursiveEdgeWorker(reconciler=_DummyReconciler(), model_name="dummy")

    messages = worker._build_messages(packet, depth=0, include_optional=False)

    assert "Interaction guidance" in messages[1]["content"]
    assert "death_place" in messages[1]["content"]


def test_guidance_block_prefers_narrative_summary_over_structured_bullets():
    block = format_interaction_guidance_block(
        {
            "interaction_guidance": {
                "natural_language_summary": "The agent is missing spouse and death_place links. Focus on those relation families and avoid over-investing in already-covered director links.",
                "relation_gap_profile": [{"relation_label": "spouse", "asked": 2, "found": 1, "missing": 1}],
                "missing_bridge_targets": [{"entity": "Fernando Cortes", "relation_label": "spouse"}],
            }
        }
    )

    assert "The agent is missing spouse and death_place links." in block
    assert "Top missing relation families:" not in block
    assert "Specific missing targets:" not in block
