from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from .prompts import VerifierPrompts
from .schemas import (
    AtomicFrame,
    CanonicalEntity,
    MentionSpan,
    QuestionPairDraft,
    TextWindow,
    VerifierFailure,
    VerifierOutput,
    VerifierReport,
)
from .transport_shim import TransportShim

logger = logging.getLogger(__name__)


class VerifierAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run_all(
        self,
        *,
        source_text: str,
        windows: Sequence[TextWindow],
        mentions: Sequence[MentionSpan],
        entities: Sequence[CanonicalEntity],
        frames: Sequence[AtomicFrame],
        question_pairs: dict[str, QuestionPairDraft],
        semaphore,
    ) -> VerifierReport:
        coverage_task = self.transport.async_call(
            system_prompt=VerifierPrompts.COVERAGE_SYSTEM,
            user_prompt=VerifierPrompts.coverage_user(
                source_text=source_text,
                windows=windows,
                mentions=mentions,
                frames=frames,
            ),
            response_model=VerifierOutput,
            stage="verify_coverage",
            max_tokens=8000,
            semaphore=semaphore,
        )
        grounding_task = self.transport.async_call(
            system_prompt=VerifierPrompts.GROUNDING_SYSTEM,
            user_prompt=VerifierPrompts.grounding_user(
                source_text=source_text,
                entities=entities,
                frames=frames,
            ),
            response_model=VerifierOutput,
            stage="verify_grounding",
            max_tokens=8000,
            semaphore=semaphore,
        )
        qa_task = self.transport.async_call(
            system_prompt=VerifierPrompts.QA_SYSTEM,
            user_prompt=VerifierPrompts.qa_user(
                frames=frames,
                question_pairs=[pair.model_dump(mode="json") for pair in question_pairs.values()],
                entity_ids=[entity.id for entity in entities],
            ),
            response_model=VerifierOutput,
            stage="verify_qa",
            max_tokens=8000,
            semaphore=semaphore,
        )
        coverage, grounding, qa = await asyncio.gather(coverage_task, grounding_task, qa_task)
        failures = [
            *self._filter_routable("coverage", coverage.failures),
            *self._filter_routable("grounding", grounding.failures),
            *self._filter_routable("qa_completeness", qa.failures),
        ]
        return VerifierReport(failures=failures)

    @staticmethod
    def _filter_routable(
        check_name: str,
        failures: Sequence[VerifierFailure],
    ) -> list[VerifierFailure]:
        """Drop failures that can't be routed to a producing agent.

        A failure is routable iff it has both `item_id` and `producer_stage`.
        Missing `check`/`item_type` are back-filled (check ← the verifier that
        produced the batch; item_type ← derived from producer_stage where
        possible). Un-routable failures get logged and dropped so one bad item
        never invalidates the rest of the batch.
        """
        routable: list[VerifierFailure] = []
        dropped = 0
        for failure in failures:
            if not failure.item_id or not failure.producer_stage:
                dropped += 1
                logger.warning(
                    "verifier_dropped_item check=%s reason=missing_routing "
                    "producer_stage=%s item_id=%s hint=%s",
                    check_name,
                    failure.producer_stage,
                    failure.item_id,
                    (failure.hint or "")[:200],
                )
                continue
            if failure.check is None:
                failure = failure.model_copy(update={"check": check_name})
            routable.append(failure)
        if dropped:
            logger.warning(
                "verifier_filter check=%s kept=%d dropped=%d",
                check_name,
                len(routable),
                dropped,
            )
        return routable
