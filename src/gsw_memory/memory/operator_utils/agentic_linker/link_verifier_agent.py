"""LinkVerifierAgent — one LLM call per Link, judges grounding + correctness."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

from ..agentic_gsw.transport_shim import TransportShim
from .prompts import LinkVerifierPrompts
from .schemas import (
    BatchedLinkVerifierOutput,
    CanonicalEntity,
    Link,
    LinkVerifierOutput,
)

logger = logging.getLogger(__name__)


class LinkVerifierAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        *,
        link: Link,
        subject_entity: CanonicalEntity,
        object_entity: CanonicalEntity,
        text: str,
        semaphore=None,
    ) -> LinkVerifierOutput:
        """Single-link verifier call. Kept for targeted per-item verification."""
        payload = await self.transport.async_call(
            system_prompt=LinkVerifierPrompts.SYSTEM,
            user_prompt=LinkVerifierPrompts.user(
                link=link,
                subject_entity=subject_entity,
                object_entity=object_entity,
                text=text,
            ),
            response_model=LinkVerifierOutput,
            stage="link_verify",
            max_tokens=2000,
            semaphore=semaphore,
        )
        return payload

    async def run_batch(
        self,
        *,
        links: Sequence[Link],
        entity_by_id: Dict[str, CanonicalEntity],
        text: str,
        semaphore=None,
    ) -> Dict[str, LinkVerifierOutput]:
        """Batched verifier call: judges N links in a single LLM call.
        Returns a dict keyed by ``link_id``. Missing verdicts default to
        valid (fail-open) so the orchestrator doesn't drop good links if
        the model omits them."""
        if not links:
            return {}
        payload = await self.transport.async_call(
            system_prompt=LinkVerifierPrompts.SYSTEM,
            user_prompt=LinkVerifierPrompts.batch_user(
                links=links,
                entity_by_id=entity_by_id,
                text=text,
            ),
            response_model=BatchedLinkVerifierOutput,
            stage="link_verify_batch",
            max_tokens=8000,
            semaphore=semaphore,
        )
        results: Dict[str, LinkVerifierOutput] = {}
        for item in payload.items:
            link_id = (item.link_id or "").strip()
            if not link_id:
                continue
            results[link_id] = item.verdict
        for link in links:
            if link.link_id not in results:
                logger.warning(
                    "link_verify_batch_missing link_id=%s — defaulting to "
                    "valid (fail-open)",
                    link.link_id,
                )
                results[link.link_id] = LinkVerifierOutput(
                    valid=True, confidence=0.5, issue="missing from batched verdict"
                )
        return results
