"""LinkAgent — one structured LLM call per verb phrase.

This replaces the multi-turn LinkerAgent. The orchestrator fans out one
LinkAgent call per VerbPhraseSlot in parallel; each call sees one verb
phrase + the full entity list + the source text and returns a single
``LinkAgentOutput`` with the link + forward/reverse questions + any
proposed new entities. No tool loop, no global planning.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

from ..agentic_gsw.transport_shim import TransportShim
from .prompts import LinkPrompts
from .schemas import (
    BatchedLinkAgentOutput,
    CanonicalEntity,
    LinkAgentOutput,
    VerbPhraseSlot,
)

logger = logging.getLogger(__name__)


class LinkAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        *,
        verb_phrase: VerbPhraseSlot,
        entities: Sequence[CanonicalEntity],
        text: str,
        retry_hint: str = "",
        semaphore=None,
    ) -> LinkAgentOutput:
        """Single-verb-phrase call. Kept for the retry path where we need
        fine-grained per-item retries with individual hints."""
        payload = await self.transport.async_call(
            system_prompt=LinkPrompts.SYSTEM,
            user_prompt=LinkPrompts.user(
                verb_phrase=verb_phrase,
                entities=entities,
                text=text,
                retry_hint=retry_hint,
            ),
            response_model=LinkAgentOutput,
            stage="link",
            max_tokens=4000,
            semaphore=semaphore,
        )
        return payload

    async def run_batch(
        self,
        *,
        verb_phrases: Sequence[VerbPhraseSlot],
        entities: Sequence[CanonicalEntity],
        text: str,
        retry_hints: Optional[Dict[str, str]] = None,
        semaphore=None,
    ) -> Dict[str, LinkAgentOutput]:
        """Batched call: processes N verb phrases in a single LLM call.

        This is the cost-saving path. Each call ships the full source text
        and entity list once and produces N link outputs, keyed by
        ``vp_id``. Missing outputs (e.g. if the model returns fewer than N
        items) are filled with empty ``LinkAgentOutput(action="skip")``
        entries so the orchestrator can detect them and handle them.
        """
        if not verb_phrases:
            return {}
        payload = await self.transport.async_call(
            system_prompt=LinkPrompts.SYSTEM,
            user_prompt=LinkPrompts.batch_user(
                verb_phrases=verb_phrases,
                entities=entities,
                text=text,
                retry_hints=retry_hints,
            ),
            response_model=BatchedLinkAgentOutput,
            stage="link_batch",
            max_tokens=16000,
            semaphore=semaphore,
        )
        results: Dict[str, LinkAgentOutput] = {}
        for item in payload.items:
            vp_id = (item.vp_id or "").strip()
            if not vp_id:
                continue
            results[vp_id] = item.output
        # Fill in missing vp_ids with explicit skips so the orchestrator can
        # see them in logs without silently losing a verb phrase.
        for vp in verb_phrases:
            if vp.vp_id not in results:
                logger.warning(
                    "link_batch_missing vp_id=%s — model skipped it; "
                    "returning an empty skip output",
                    vp.vp_id,
                )
                results[vp.vp_id] = LinkAgentOutput(
                    action="skip",
                    skip_reason="missing from batched output",
                )
        return results
