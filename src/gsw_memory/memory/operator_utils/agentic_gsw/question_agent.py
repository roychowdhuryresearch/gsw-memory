from __future__ import annotations

import logging
from typing import Sequence

from .prompts import QuestionPairPrompts
from .schemas import AtomicFrame, CanonicalEntity, QuestionDraft, QuestionPairDraft, QuestionPairOutput
from .transport_shim import TransportShim

logger = logging.getLogger(__name__)


class QuestionAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        frame: AtomicFrame,
        touched_entities: Sequence[CanonicalEntity],
        *,
        supporting_text: str,
        forward_fallback_id: str,
        reverse_fallback_id: str,
        hint: str = "",
        semaphore=None,
    ) -> QuestionPairDraft:
        """``forward_fallback_id`` / ``reverse_fallback_id`` — canonical entity
        ids of the frame's object and subject respectively. These guarantee
        that every question side has ≥1 entity answer even if the model
        deviates from the instructions — the factual-convention invariant
        "every relation is answerable" holds by construction.
        """
        payload = await self.transport.async_call(
            system_prompt=QuestionPairPrompts.SYSTEM,
            user_prompt=QuestionPairPrompts.user(
                frame,
                touched_entities,
                supporting_text=supporting_text,
                hint=hint,
            ),
            response_model=QuestionPairOutput,
            stage="question",
            max_tokens=8000,
            semaphore=semaphore,
        )
        return self._normalize_pair(
            payload.pair,
            frame,
            touched_entities,
            forward_fallback_id=forward_fallback_id,
            reverse_fallback_id=reverse_fallback_id,
        )

    @staticmethod
    def _normalize_pair(
        pair: QuestionPairDraft,
        frame: AtomicFrame,
        entities: Sequence[CanonicalEntity],
        *,
        forward_fallback_id: str,
        reverse_fallback_id: str,
    ) -> QuestionPairDraft:
        """Every answer must resolve to a canonical entity id in scope.
        If the model drifts on a side (emits a name, an alias, a hallucinated
        id, or nothing at all), we fall back to the frame's corresponding
        canonical participant id so that every question ends up with ≥1
        resolved answer — guaranteed.
        """
        valid_ids = {entity.id for entity in entities}
        name_to_id: dict[str, str] = {}
        for entity in entities:
            if entity.name:
                name_to_id[entity.name.casefold()] = entity.id
            for alias in entity.aliases:
                if alias:
                    name_to_id[alias.casefold()] = entity.id

        def _normalize_answers(
            question: QuestionDraft,
            side: str,
            fallback_id: str,
        ) -> QuestionDraft:
            answers: list[str] = []
            for answer in question.answers:
                raw = str(answer or "").strip()
                if not raw:
                    continue
                lower = raw.casefold()
                if raw in valid_ids:
                    answers.append(raw)
                elif lower in name_to_id:
                    answers.append(name_to_id[lower])
                else:
                    logger.debug(
                        "question_answer_unresolved frame=%s side=%s original=%r "
                        "— will back-fill from frame participant",
                        frame.frame_id,
                        side,
                        raw,
                    )
            # Dedupe while preserving order.
            deduped: list[str] = []
            seen: set[str] = set()
            for answer in answers:
                if answer in seen:
                    continue
                seen.add(answer)
                deduped.append(answer)
            # Guarantee every side has at least one entity id by falling back
            # to the frame's canonical participant for that side.
            if not deduped and fallback_id:
                deduped = [fallback_id]
                logger.debug(
                    "question_answer_backfilled frame=%s side=%s fallback=%s",
                    frame.frame_id,
                    side,
                    fallback_id,
                )
            return QuestionDraft(text=str(question.text or "").strip(), answers=deduped)

        return QuestionPairDraft(
            frame_id=frame.frame_id,
            forward=_normalize_answers(pair.forward, "forward", forward_fallback_id),
            reverse=_normalize_answers(pair.reverse, "reverse", reverse_fallback_id),
        )
