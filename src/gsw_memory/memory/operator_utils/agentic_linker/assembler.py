"""assembler — pure Python translation of the linker pipeline state into
the final ``GSWStructure``.

By the time we reach this stage the orchestrator has guaranteed:
- every Link's subject_id and object_id reference a real CanonicalEntity
- every Link has a forward_question and reverse_question text
- the answers are fixed: forward=[object_id], reverse=[subject_id]

So this stage is a structural translation, not a normalization pass.
"""

from __future__ import annotations

from typing import Sequence

from ...models import EntityNode, GSWStructure, Question, Role, VerbPhraseNode
from .schemas import CanonicalEntity, Link


def assemble_gsw(
    *,
    entities: Sequence[CanonicalEntity],
    links: Sequence[Link],
    chunk_id: str,
) -> GSWStructure:
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

    verb_phrase_nodes: list[VerbPhraseNode] = []
    for link in links:
        questions = [
            Question(
                id=f"q_{link.link_id}_forward",
                text=link.forward_question,
                answers=[link.object_id],
                chunk_id=chunk_id,
            ),
            Question(
                id=f"q_{link.link_id}_reverse",
                text=link.reverse_question,
                answers=[link.subject_id],
                chunk_id=chunk_id,
            ),
        ]
        verb_phrase_nodes.append(
            VerbPhraseNode(
                id=link.link_id,
                phrase=link.vp_label,
                questions=questions,
                chunk_id=chunk_id,
            )
        )

    return GSWStructure(
        entity_nodes=entity_nodes,
        verb_phrase_nodes=verb_phrase_nodes,
    )
