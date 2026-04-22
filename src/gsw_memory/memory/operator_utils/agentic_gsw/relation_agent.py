from __future__ import annotations

from typing import Sequence

from .prompts import AtomicFramePrompts
from .schemas import AtomicFrame, RelationFrameOutput, TextWindow, MentionSpan
from .transport_shim import TransportShim


class RelationAgent:
    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        window: TextWindow,
        mentions_in_window: Sequence[MentionSpan],
        *,
        hint: str = "",
        semaphore=None,
    ) -> list[AtomicFrame]:
        payload = await self.transport.async_call(
            system_prompt=AtomicFramePrompts.SYSTEM,
            user_prompt=AtomicFramePrompts.user(window, mentions_in_window, hint=hint),
            response_model=RelationFrameOutput,
            stage="relation",
            max_tokens=32000,
            semaphore=semaphore,
        )
        frames = self._normalize_frames(payload.frames, window)
        return frames

    @staticmethod
    def _normalize_frames(
        frames: Sequence[AtomicFrame],
        window: TextWindow,
    ) -> list[AtomicFrame]:
        normalized: list[AtomicFrame] = []
        for frame in frames:
            phrase = str(frame.vp_phrase or "").strip()
            if not phrase:
                continue
            subject_ref = str(frame.subject.ref or "").strip()
            object_ref = str(frame.object.ref or "").strip()
            if not subject_ref or not object_ref:
                continue
            support_start = int(frame.support_char_start or 0)
            support_end = int(frame.support_char_end or 0)
            if support_end <= support_start:
                support_start = window.char_start
                support_end = window.char_end
            normalized.append(
                AtomicFrame(
                    frame_id="",
                    vp_phrase=phrase,
                    subject=frame.subject,
                    object=frame.object,
                    slot_kind=frame.slot_kind,
                    window_id=window.id,
                    supporting_text=str(frame.supporting_text or "").strip() or window.text,
                    support_char_start=max(window.char_start, support_start),
                    support_char_end=min(window.char_end, support_end),
                )
            )
        normalized.sort(
            key=lambda item: (
                item.slot_kind,
                item.vp_phrase.casefold(),
                item.subject.ref,
                item.object.ref,
                item.support_char_start,
                item.support_char_end,
            )
        )
        for idx, frame in enumerate(normalized, start=1):
            frame.frame_id = f"f_{window.id}_{idx}"
        return normalized
