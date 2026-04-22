from __future__ import annotations

from typing import Sequence

from ...models import EntityNode, GSWStructure, Question, Role, VerbPhraseNode
from .schemas import AtomicFrame, CanonicalEntity, QuestionPairDraft


def assemble_gsw(
    *,
    entities: Sequence[CanonicalEntity],
    mention_map: dict[str, str],
    frames: Sequence[AtomicFrame],
    question_pairs: dict[str, QuestionPairDraft],
    chunk_id: str,
) -> GSWStructure:
    """Deterministic assembly of the final GSWStructure.

    Upstream (orchestrator's _backfill_canonicalization) guarantees every
    frame participant ref resolves to a canonical entity, so by the time we
    reach this function every relation is answerable and every question
    pair's answers resolve to entity ids. We trust that invariant — no
    filtering, no dropping.
    """
    entity_nodes = [
        EntityNode(
            id=entity.id,
            name=entity.name,
            roles=[
                Role(
                    role=role.role,
                    states=list(role.states),
                    chunk_id=chunk_id,
                )
                for role in entity.roles
            ],
            chunk_id=chunk_id,
        )
        for entity in entities
    ]
    valid_entity_ids = {entity.id for entity in entities}

    verb_phrase_nodes: list[VerbPhraseNode] = []
    for frame in frames:
        pair = question_pairs.get(frame.frame_id)
        if pair is None:
            continue
        questions = [
            Question(
                id=f"q_{frame.frame_id}_forward",
                text=pair.forward.text,
                answers=_resolve_answers(pair.forward.answers, mention_map, valid_entity_ids),
                chunk_id=chunk_id,
            ),
            Question(
                id=f"q_{frame.frame_id}_reverse",
                text=pair.reverse.text,
                answers=_resolve_answers(pair.reverse.answers, mention_map, valid_entity_ids),
                chunk_id=chunk_id,
            ),
        ]
        verb_phrase_nodes.append(
            VerbPhraseNode(
                id=frame.frame_id,
                phrase=frame.vp_phrase,
                questions=questions,
                chunk_id=chunk_id,
            )
        )

    return GSWStructure(
        entity_nodes=entity_nodes,
        verb_phrase_nodes=verb_phrase_nodes,
    )


def _resolve_answers(
    answers: Sequence[str],
    mention_map: dict[str, str],
    valid_entity_ids: set[str],
) -> list[str]:
    """Convert answer strings into canonical entity ids. Accepts both direct
    `eN` ids and mention refs that the mention_map translates. Anything else
    is a bug upstream and should never reach here."""
    resolved: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        raw = str(answer or "").strip()
        if not raw:
            continue
        candidate = mention_map.get(raw, raw)
        if candidate in valid_entity_ids and candidate not in seen:
            resolved.append(candidate)
            seen.add(candidate)
    return resolved
