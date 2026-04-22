"""VerbPhraseListAgent — single LLM call producing a flat verb phrase list.

Duplicates are intentional: one ``vp_id`` per relation slot, so a verb
phrase like "released" that has both a year slot and a country slot appears
twice with two distinct ``vp_id``s.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from ..agentic_gsw.transport_shim import TransportShim
from .prompts import VerbPhraseListPrompts
from .schemas import CanonicalEntity, VerbPhraseListOutput, VerbPhraseSlot

logger = logging.getLogger(__name__)


class VerbPhraseListAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        text: str,
        *,
        entities: Optional[Sequence[CanonicalEntity]] = None,
    ) -> List[VerbPhraseSlot]:
        payload = await self.transport.async_call(
            system_prompt=VerbPhraseListPrompts.SYSTEM,
            user_prompt=VerbPhraseListPrompts.user(text, entities=entities),
            response_model=VerbPhraseListOutput,
            stage="verb_phrase_list",
            max_tokens=32000,
        )
        return self._normalize(payload.verb_phrases)

    @staticmethod
    def _normalize(slots: List[VerbPhraseSlot]) -> List[VerbPhraseSlot]:
        """Strip empties and re-id sequentially. Duplicates are preserved."""
        cleaned: list[VerbPhraseSlot] = []
        for slot in slots:
            label = (slot.label or "").strip()
            if not label:
                continue
            cleaned.append(
                VerbPhraseSlot(
                    vp_id=slot.vp_id or "",  # rewritten below
                    label=label,
                    hint=(slot.hint or "").strip(),
                )
            )
        for idx, slot in enumerate(cleaned, start=1):
            slot.vp_id = f"vp{idx}"

        if not cleaned:
            logger.warning(
                "verb_phrase_list_empty — linker will have nothing to fill"
            )
        return cleaned
