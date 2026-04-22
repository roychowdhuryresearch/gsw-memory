"""EntityListAgent — single LLM call producing a flat canonical entity list."""

from __future__ import annotations

import logging
import re
from typing import List

from ..agentic_gsw.transport_shim import TransportShim
from .prompts import EntityListPrompts
from .schemas import CanonicalEntity, EntityListOutput

logger = logging.getLogger(__name__)


class EntityListAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(self, text: str) -> List[CanonicalEntity]:
        payload = await self.transport.async_call(
            system_prompt=EntityListPrompts.SYSTEM,
            user_prompt=EntityListPrompts.user(text),
            response_model=EntityListOutput,
            stage="entity_list",
            max_tokens=32000,
        )
        return self._normalize(payload.entities, text)

    @staticmethod
    def _normalize(entities: List[CanonicalEntity], text: str) -> List[CanonicalEntity]:
        """Drop empty / placeholder entities and re-id sequentially.

        We accept the model's id assignment but re-number deterministically
        so the linker sees a clean ``e1, e2, ...`` sequence even if the model
        skipped or duplicated ids."""
        cleaned: list[CanonicalEntity] = []
        seen_names: set[str] = set()
        for entity in entities:
            name = (entity.name or "").strip()
            if not name or len(name) <= 1:
                continue
            if name.casefold() in {"a", "an", "the", "generic", "unknown"}:
                continue
            key = name.casefold()
            if key in seen_names:
                continue
            seen_names.add(key)
            cleaned.append(
                CanonicalEntity(
                    id=entity.id or "",  # rewritten below
                    name=name,
                    aliases=[
                        alias.strip()
                        for alias in entity.aliases
                        if alias and alias.strip() and alias.strip() != name
                    ],
                    roles=list(entity.roles),
                )
            )
        for idx, entity in enumerate(cleaned, start=1):
            entity.id = f"e{idx}"

        # Warn loudly on suspicious output.
        if not cleaned:
            logger.warning(
                "entity_list_zero_entities text_chars=%d — pipeline will have "
                "nothing to link",
                len(text),
            )
        else:
            empty_role_count = sum(1 for e in cleaned if not e.roles)
            if empty_role_count:
                logger.warning(
                    "entity_list_empty_roles count=%d/%d — model omitted role "
                    "labels; downstream linker will have weaker grounding",
                    empty_role_count,
                    len(cleaned),
                )
        return cleaned
