from __future__ import annotations

import logging
from typing import Sequence

from .prompts import EntitySeedPrompts
from .schemas import EntitySeedOutput, MentionSeed, MentionSpan, TextWindow
from .transport_shim import TransportShim
from .windowing import assign_primary_window_id

logger = logging.getLogger(__name__)


class EntitySeedAgent:
    # Heuristic: expect at least one mention per ~500 chars of text. Below this
    # we emit a WARNING so silent-zero-mentions regressions don't go unnoticed.
    MIN_MENTIONS_PER_CHARS = 500

    def __init__(self, transport: TransportShim) -> None:
        self.transport = transport

    async def run(
        self,
        text: str,
        windows: Sequence[TextWindow],
        *,
        hint: str = "",
    ) -> list[MentionSpan]:
        payload = await self.transport.async_call(
            system_prompt=EntitySeedPrompts.SYSTEM,
            user_prompt=EntitySeedPrompts.user(text, hint=hint),
            response_model=EntitySeedOutput,
            stage="entity_seed",
            max_tokens=8000,
        )
        mentions = self._materialize_mentions(payload.mentions, text, windows)
        self._warn_if_suspiciously_few(mentions, text)
        return mentions

    @staticmethod
    def _materialize_mentions(
        seeds: Sequence[MentionSeed],
        text: str,
        windows: Sequence[TextWindow],
    ) -> list[MentionSpan]:
        """Search the document for each model-emitted surface string and
        build full MentionSpan objects with deterministic char offsets.

        The LLM only provides `text` + `aliases`; this method finds each
        surface in the source document (first occurrence per distinct
        surface) and assigns the primary window_id. Surfaces we can't
        locate are skipped with a DEBUG log — they're usually model
        hallucinations or paraphrases."""
        materialized: list[MentionSpan] = []
        seen: set[tuple[int, int, str]] = set()
        unlocated: list[str] = []
        for seed in seeds:
            surface = str(getattr(seed, "text", "") or "").strip()
            if not surface:
                continue
            char_start = text.find(surface)
            if char_start < 0:
                unlocated.append(surface)
                continue
            char_end = char_start + len(surface)
            key = (char_start, char_end, surface.casefold())
            if key in seen:
                continue
            seen.add(key)
            aliases = [
                alias.strip()
                for alias in getattr(seed, "aliases", []) or []
                if alias and alias.strip() and alias.strip() != surface
            ]
            materialized.append(
                MentionSpan(
                    ref="",
                    text=surface,
                    aliases=sorted(set(aliases)),
                    char_start=char_start,
                    char_end=char_end,
                    window_id=assign_primary_window_id(char_start, windows),
                )
            )
        materialized.sort(key=lambda item: (item.char_start, item.char_end, item.text.casefold()))
        for idx, mention in enumerate(materialized, start=1):
            mention.ref = f"m{idx}"
        if unlocated:
            logger.debug(
                "entity_seed_unlocated_mentions count=%d first=%s",
                len(unlocated),
                unlocated[:5],
            )
        return materialized

    @classmethod
    def _warn_if_suspiciously_few(cls, mentions: Sequence[MentionSpan], text: str) -> None:
        expected = max(1, len(text) // cls.MIN_MENTIONS_PER_CHARS)
        if len(mentions) == 0:
            logger.warning(
                "entity_seed_zero_mentions text_chars=%d expected_at_least=%d — "
                "downstream stages will run without priors; check EntitySeedPrompts",
                len(text),
                expected,
            )
        elif len(mentions) < expected // 4:
            logger.warning(
                "entity_seed_few_mentions got=%d expected_at_least=%d text_chars=%d",
                len(mentions),
                expected,
                len(text),
            )

    @staticmethod
    def _normalize_mentions(
        mentions: Sequence[MentionSpan],
        text: str,
        windows: Sequence[TextWindow],
    ) -> list[MentionSpan]:
        normalized: list[MentionSpan] = []
        seen: set[tuple[int, int, str]] = set()
        for mention in mentions:
            char_start = max(0, int(mention.char_start))
            char_end = min(len(text), int(mention.char_end))
            if char_end <= char_start:
                continue
            surface = text[char_start:char_end].strip() or mention.text.strip()
            if not surface:
                continue
            key = (char_start, char_end, surface.casefold())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                MentionSpan(
                    ref="",
                    text=surface,
                    aliases=[alias for alias in mention.aliases if alias and alias != surface],
                    char_start=char_start,
                    char_end=char_end,
                    window_id=assign_primary_window_id(char_start, windows),
                )
            )
        normalized.sort(key=lambda item: (item.char_start, item.char_end, item.text.casefold()))
        for idx, mention in enumerate(normalized, start=1):
            mention.ref = f"m{idx}"
        return normalized
