from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler
from gsw_memory.sleep_time.rlm_pipeline import EdgeKey, HybridScheduler, RecursiveEdgeWorker, RootScheduler
from gsw_memory.sleep_time.tools import normalize_similarity_text

from .prompts import render_interaction_guidance_block
from .types import InteractionGuidanceSummary


def _guidance_payload_dict(guidance: Any) -> Dict[str, Any]:
    if isinstance(guidance, InteractionGuidanceSummary):
        return guidance.as_dict()
    if isinstance(guidance, dict):
        return dict(guidance)
    return {}
def build_sleep_guidance_payload(
    summary: InteractionGuidanceSummary | Dict[str, Any],
    *,
    targeted_edge_ratio: float = 0.7,
) -> Dict[str, Any]:
    summary_payload = _guidance_payload_dict(summary)
    gap_rows = list(summary_payload.get("relation_gap_profile", []) or [])
    target_rows = list(summary_payload.get("entity_gap_targets", []) or summary_payload.get("missing_bridge_targets", []) or [])
    target_relations = [
        str(row.get("relation_label", "") or row.get("relation", "")).strip()
        for row in gap_rows[:8]
        if str(row.get("relation_label", "") or row.get("relation", "")).strip()
    ]
    target_entities = [
        str(row.get("entity", "") or "").strip()
        for row in target_rows[:12]
        if str(row.get("entity", "") or "").strip()
    ]
    return {
        "interaction_guidance": summary_payload,
        "sleep_focus": {
            "targeted_edge_ratio": max(0.0, min(1.0, float(targeted_edge_ratio))),
            "target_relations": target_relations,
            "target_entities": target_entities,
            "deprioritized_relations": [],
        },
    }


def format_interaction_guidance_block(guidance: Dict[str, Any] | None) -> str:
    return render_interaction_guidance_block(guidance)


def _relation_match_score(relationship: str, target_relations: Sequence[str]) -> float:
    normalized_relationship = normalize_similarity_text(relationship)
    if not normalized_relationship:
        return 0.0
    best = 0.0
    for relation in target_relations:
        normalized_relation = normalize_similarity_text(relation)
        if not normalized_relation:
            continue
        if normalized_relation == normalized_relationship:
            best = max(best, 1.0)
            continue
        if normalized_relation in normalized_relationship or normalized_relationship in normalized_relation:
            best = max(best, 0.7)
    return best


def _entity_match_score(entity_name: str, neighbor_name: str, target_entities: Sequence[str]) -> float:
    normalized_names = {
        normalize_similarity_text(entity_name),
        normalize_similarity_text(neighbor_name),
    }
    best = 0.0
    for entity in target_entities:
        normalized_entity = normalize_similarity_text(entity)
        if not normalized_entity:
            continue
        if normalized_entity in normalized_names:
            best = max(best, 1.0)
            continue
        if any(normalized_entity in candidate or candidate in normalized_entity for candidate in normalized_names if candidate):
            best = max(best, 0.6)
    return best


def _supporting_example_match(edge: EdgeKey, examples: Sequence[Dict[str, Any]]) -> float:
    edge_text = " ".join(
        [
            str(edge.entity_name or ""),
            str(edge.neighbor_name or ""),
            str(edge.relationship or ""),
        ]
    )
    normalized_edge = normalize_similarity_text(edge_text)
    if not normalized_edge:
        return 0.0
    best = 0.0
    for example in examples:
        if not isinstance(example, dict):
            continue
        example_text = " ".join(
            [
                str(example.get("question", "") or ""),
                str(example.get("missing_relation", "") or example.get("relation_label", "") or ""),
                str(example.get("entity", "") or ""),
            ]
        )
        normalized_example = normalize_similarity_text(example_text)
        if normalized_example and (normalized_example in normalized_edge or normalized_edge in normalized_example):
            best = max(best, 0.4)
    return best


def score_edge_for_guidance(edge: EdgeKey, guidance: Dict[str, Any] | None) -> float:
    payload = _guidance_payload_dict(guidance)
    interaction = payload.get("interaction_guidance") if isinstance(payload.get("interaction_guidance"), dict) else payload
    if not isinstance(interaction, dict) or not interaction:
        return 0.0
    target_relations = [
        str(row.get("relation_label", "") or row.get("relation", "")).strip()
        for row in interaction.get("relation_gap_profile", []) or []
        if isinstance(row, dict)
    ]
    target_entities = [
        str(row.get("entity", "") or "").strip()
        for row in interaction.get("entity_gap_targets", []) or interaction.get("missing_bridge_targets", []) or []
        if isinstance(row, dict)
    ]
    examples = list(interaction.get("supporting_examples", []) or [])

    entity_score = _entity_match_score(edge.entity_name, edge.neighbor_name, target_entities)
    relation_score = _relation_match_score(edge.relationship, target_relations)
    example_score = _supporting_example_match(edge, examples)
    return (2.0 * entity_score) + (1.5 * relation_score) + example_score


def prioritize_edges_with_reserve(
    edges: Sequence[EdgeKey],
    guidance: Dict[str, Any] | InteractionGuidanceSummary | None,
) -> List[EdgeKey]:
    ordered_edges = list(edges)
    if not ordered_edges:
        return []

    payload = _guidance_payload_dict(guidance)
    sleep_focus = payload.get("sleep_focus", {}) if isinstance(payload.get("sleep_focus"), dict) else {}
    targeted_ratio = max(0.0, min(1.0, float(sleep_focus.get("targeted_edge_ratio", 0.7) or 0.7)))

    scored: List[Tuple[float, EdgeKey]] = [(score_edge_for_guidance(edge, payload), edge) for edge in ordered_edges]
    targeted = [row for row in scored if row[0] > 0]
    exploratory = [row for row in scored if row[0] <= 0]
    targeted.sort(key=lambda row: (-row[0], row[1].as_tuple()))
    exploratory.sort(key=lambda row: row[1].as_tuple())

    if not targeted or targeted_ratio >= 0.999:
        return [edge for _score, edge in targeted + exploratory]

    result: List[EdgeKey] = [targeted[0][1]]
    reserve_ratio = 1.0 - targeted_ratio
    targeted_index = 1
    exploratory_index = 0
    while targeted_index < len(targeted) or exploratory_index < len(exploratory):
        current_ratio = (len(result) and (exploratory_index / float(len(result))) or 0.0)
        take_exploratory = (
            exploratory_index < len(exploratory)
            and (targeted_index >= len(targeted) or current_ratio < reserve_ratio)
        )
        if take_exploratory:
            result.append(exploratory[exploratory_index][1])
            exploratory_index += 1
            continue
        if targeted_index < len(targeted):
            result.append(targeted[targeted_index][1])
            targeted_index += 1
            continue
        if exploratory_index < len(exploratory):
            result.append(exploratory[exploratory_index][1])
            exploratory_index += 1
    return result


class InteractionGuidedRecursiveEdgeWorker(RecursiveEdgeWorker):
    def _build_messages(self, packet, depth, include_optional, retry_brief=None):
        messages = super()._build_messages(packet, depth, include_optional, retry_brief=retry_brief)
        guidance_block = format_interaction_guidance_block(packet.constraints.get("curriculum_guidance"))
        if guidance_block and len(messages) >= 2:
            messages[1]["content"] += f"\n\n{guidance_block}\n"
        return messages

    def _build_render_messages(self, render_input, retry_brief=None):
        messages = super()._build_render_messages(render_input, retry_brief=retry_brief)
        guidance_block = format_interaction_guidance_block(render_input.constraints.get("curriculum_guidance"))
        if guidance_block and len(messages) >= 2:
            messages[1]["content"] += f"\n\n{guidance_block}\n"
        return messages


class InteractionGuidedRootScheduler(RootScheduler):
    def _iter_pending_edges(self, doc_id: str) -> List[EdgeKey]:
        base_edges = super()._iter_pending_edges(doc_id)
        return prioritize_edges_with_reserve(base_edges, getattr(self.reconciler, "curriculum_guidance", None))


class InteractionGuidedHybridScheduler(HybridScheduler):
    def _iter_pending_edges(self, doc_id: str) -> List[EdgeKey]:
        base_edges = super()._iter_pending_edges(doc_id)
        return prioritize_edges_with_reserve(base_edges, getattr(self.reconciler, "curriculum_guidance", None))


class InteractionGuidedReconciler(AgenticReconciler):
    def _create_rlm_scheduler(self):
        worker = InteractionGuidedRecursiveEdgeWorker(
            reconciler=self,
            model_name=self.worker_model,
            max_depth=self.edge_max_depth,
            max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
        )
        return InteractionGuidedRootScheduler(
            reconciler=self,
            worker=worker,
            edge_max_depth=self.edge_max_depth,
            edge_max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
            max_optional_docs_per_edge=self.max_optional_docs_per_edge,
        )

    def _create_hybrid_scheduler(self):
        worker = InteractionGuidedRecursiveEdgeWorker(
            reconciler=self,
            model_name=self.worker_model,
            max_depth=self.edge_max_depth,
            max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
        )
        return InteractionGuidedHybridScheduler(
            reconciler=self,
            worker=worker,
            edge_max_depth=self.edge_max_depth,
            edge_max_calls=self.edge_max_calls,
            edge_max_tokens=self.edge_max_tokens,
            max_optional_docs_per_edge=self.max_optional_docs_per_edge,
            hybrid_scope=self.hybrid_scope,
            edge_parallel_enabled=self.edge_parallel_enabled,
            edge_parallel_workers=self.edge_parallel_workers,
            parallel_progress_heartbeat_seconds=self.parallel_progress_heartbeat_seconds,
            parallel_stuck_warning_seconds=self.parallel_stuck_warning_seconds,
        )
