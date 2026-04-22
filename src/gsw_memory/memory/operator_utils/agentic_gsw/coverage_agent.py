"""CoverageAgent — repairs orphan entities (entities that are not referenced
as an answer id in any existing question pair) by asking the LLM to mint one
or more grounded atomic frames + QA pairs that wire the target entity in.

This is Stage 5b of the pipeline: it runs between the verifier pass (which
judges semantic quality of existing content) and the final assembly. The
orchestrator computes the orphan set deterministically in Python — the
coverage agent only handles "create a new relation" for a single entity
given its local paragraph context.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .prompts import CoverageQAPrompts
from .schemas import (
    AtomicFrame,
    CanonicalEntity,
    CoverageFrameQuestionDraft,
    CoverageQAOutput,
    CoverageQAResult,
    MentionSpan,
    QuestionPairDraft,
    TextWindow,
)
from .transport_shim import TransportShim

logger = logging.getLogger(__name__)


class CoverageAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        *,
        target_entity: CanonicalEntity,
        target_mentions: Sequence[MentionSpan],
        available_mentions: Sequence[MentionSpan],
        local_window: TextWindow,
        nearby_entities: Sequence[CanonicalEntity],
        existing_frames: Sequence[AtomicFrame],
        hint: str = "",
        semaphore=None,
    ) -> CoverageQAResult:
        payload = await self.transport.async_call(
            system_prompt=CoverageQAPrompts.SYSTEM,
            user_prompt=CoverageQAPrompts.user(
                target_entity=target_entity,
                target_mentions=target_mentions,
                available_mentions=available_mentions,
                local_window=local_window,
                nearby_entities=nearby_entities,
                existing_frames=existing_frames,
                hint=hint,
            ),
            response_model=CoverageQAOutput,
            stage="coverage_repair",
            max_tokens=8000,
            semaphore=semaphore,
        )
        return self._normalize(payload, target_entity=target_entity)

    @staticmethod
    def _normalize(
        payload: CoverageQAOutput,
        *,
        target_entity: CanonicalEntity,
    ) -> CoverageQAResult:
        """Attach deterministic frame_ids to any returned drafts and pass the
        frames + pair objects through to the orchestrator. The orchestrator
        is responsible for validating that at least one answer in each pair
        references the target entity and for merging the new frames/pairs
        into the running state.
        """
        frames: list[AtomicFrame] = []
        pairs: list[QuestionPairDraft] = []
        for idx, item in enumerate(payload.items or []):
            if not isinstance(item, CoverageFrameQuestionDraft):
                continue
            frame_id = f"cov_{target_entity.id}_{idx}"
            frame = item.frame.model_copy(deep=True)
            frame.frame_id = frame_id
            frames.append(frame)
            pair = item.pair.model_copy(deep=True)
            pair.frame_id = frame_id
            pairs.append(pair)
        if not frames:
            logger.debug(
                "coverage_skipped target=%s reason=%s",
                target_entity.id,
                (payload.skipped_reason or "")[:200],
            )
        return CoverageQAResult(
            frames=frames,
            question_pairs=pairs,
            skipped_reason=payload.skipped_reason or "",
        )
