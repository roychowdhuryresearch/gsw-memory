from __future__ import annotations

import logging
from typing import Sequence

from .prompts import EntityRefinerPrompts
from .schemas import AtomicFrame, CanonicalEntity, EntityRefinerOutput, MentionSpan
from .transport_shim import TransportShim

logger = logging.getLogger(__name__)


class EntityRefinerAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        mentions: Sequence[MentionSpan],
        frames: Sequence[AtomicFrame],
        *,
        hint: str = "",
    ) -> tuple[list[CanonicalEntity], dict[str, str]]:
        payload = await self.transport.async_call(
            system_prompt=EntityRefinerPrompts.SYSTEM,
            user_prompt=EntityRefinerPrompts.user(mentions, frames, hint=hint),
            response_model=EntityRefinerOutput,
            stage="entity_refiner",
            max_tokens=16000,
        )
        if mentions and not payload.mention_map:
            logger.warning(
                "entity_refiner_empty_mention_map mentions=%d entities=%d — "
                "model omitted mention_map; orchestrator will back-fill from "
                "each entity's source_mentions but canonicalization will be weaker",
                len(mentions),
                len(payload.entities),
            )
        # Quality guardrail: the refiner is supposed to assign at least one
        # role per canonical entity. If the model came back with mostly
        # empty roles, surface a warning so we notice the regression early.
        if payload.entities:
            empty_role_count = sum(1 for e in payload.entities if not e.roles)
            total = len(payload.entities)
            if empty_role_count == total:
                logger.warning(
                    "entity_refiner_all_empty_roles entities=%d — model "
                    "returned every entity with roles=[]; downstream will "
                    "treat them as unlabeled. Check EntityRefinerPrompts.",
                    total,
                )
            elif empty_role_count > total // 2:
                logger.warning(
                    "entity_refiner_mostly_empty_roles empty=%d/%d — "
                    "more than half of canonical entities have no role label",
                    empty_role_count,
                    total,
                )
        entities, mention_map = self._normalize(payload.entities, payload.mention_map, mentions)
        return entities, mention_map

    @staticmethod
    def _normalize(
        entities: Sequence[CanonicalEntity],
        mention_map: dict[str, str],
        mentions: Sequence[MentionSpan],
    ) -> tuple[list[CanonicalEntity], dict[str, str]]:
        normalized_entities: list[CanonicalEntity] = []
        mention_refs = {mention.ref for mention in mentions}
        for entity in entities:
            name = str(entity.name or "").strip()
            if not name:
                continue
            aliases = sorted({alias.strip() for alias in entity.aliases if alias and alias.strip() and alias.strip() != name})
            source_mentions = [ref for ref in entity.source_mentions if ref in mention_refs]
            normalized_entities.append(
                CanonicalEntity(
                    id=str(entity.id or "").strip(),
                    name=name,
                    aliases=aliases,
                    roles=list(entity.roles),
                    source_mentions=source_mentions,
                )
            )

        normalized_map = {str(key): str(value) for key, value in mention_map.items() if key in mention_refs and value}
        return normalized_entities, normalized_map
