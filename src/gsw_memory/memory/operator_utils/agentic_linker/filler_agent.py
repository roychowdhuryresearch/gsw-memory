"""FillerAgent — one LLM call per orphan entity, mints a new VP + link.

Runs after all VP-driven LinkAgent calls + verification. For each
canonical entity that still has zero links (i.e. wasn't a subject or
object in any per-VP link), this agent finds a grounded relation from
the source text and emits it as if it were a normal LinkAgent output.
The orchestrator then attaches a freshly-minted vp_id and treats the
result as just another Link.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..agentic_gsw.transport_shim import TransportShim
from .prompts import FillerPrompts
from .schemas import CanonicalEntity, LinkAgentOutput

logger = logging.getLogger(__name__)


class FillerAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        *,
        orphan_entity: CanonicalEntity,
        entities: Sequence[CanonicalEntity],
        text: str,
        semaphore=None,
    ) -> LinkAgentOutput:
        payload = await self.transport.async_call(
            system_prompt=FillerPrompts.SYSTEM,
            user_prompt=FillerPrompts.user(
                orphan_entity=orphan_entity,
                entities=entities,
                text=text,
            ),
            response_model=LinkAgentOutput,
            stage="filler",
            max_tokens=4000,
            semaphore=semaphore,
        )
        return payload
