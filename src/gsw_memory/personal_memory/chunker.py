"""
Topic-boundary session chunker for personal memory.

Splits a dialogue session into topically coherent chunks using an LLM to
detect topic shifts, then falls back to fixed-turn splitting if the LLM
call fails or returns no boundaries.

Each returned chunk is a plain string already formatted with turn IDs
(e.g. "[D1:3] Caroline: I went to a support group yesterday") so that the
downstream GSWOperator (CONVERSATIONAL) can populate evidence_turn_ids.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List

from openai import OpenAI

from .data_ingestion.locomo import Session, Turn

logger = logging.getLogger(__name__)

# Prompt for topic boundary detection
_TOPIC_DETECTION_SYSTEM = (
    "You are a dialogue analyst. Your task is to identify topic boundaries "
    "in a two-person conversation."
)

_TOPIC_DETECTION_USER = """Below is a numbered list of dialogue turns from a single conversation session.

{numbered_turns}

Identify the turn INDICES (0-based) where a clear topic shift occurs — i.e. the conversation moves to a noticeably different subject. Return a JSON object with one key "split_indices" containing a list of integers. Do not include index 0.

Example output:
{{"split_indices": [8, 15]}}

If there are no clear topic shifts, return:
{{"split_indices": []}}"""


def _format_turns_with_ids(turns: List[Turn]) -> str:
    """Format turns as '[dia_id] Speaker: text' lines for the GSW operator."""
    return "\n".join(
        f"[{turn.dia_id}] {turn.speaker}: {turn.text}" for turn in turns
    )


def _chunk_by_indices(turns: List[Turn], split_indices: List[int]) -> List[List[Turn]]:
    """Split a turn list at the given 0-based indices."""
    boundaries = sorted(set(split_indices))
    chunks: List[List[Turn]] = []
    start = 0
    for idx in boundaries:
        if 0 < idx < len(turns):
            chunks.append(turns[start:idx])
            start = idx
    chunks.append(turns[start:])
    return [c for c in chunks if c]  # drop empty chunks


class TopicBoundaryChunker:
    """Split a Session into topically coherent turn-groups using an LLM.

    Args:
        model_name: OpenAI model name for topic detection.
        max_turns_per_chunk: Hard cap — if no LLM boundaries are detected,
            fall back to splitting every N turns.
        client: Optional pre-constructed OpenAI client (useful for testing).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_turns_per_chunk: int = 15,
        client: OpenAI | None = None,
    ):
        self.model_name = model_name
        self.max_turns_per_chunk = max_turns_per_chunk
        self._client = client or OpenAI()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_session(self, session: Session) -> List[str]:
        """Split a session into topically coherent chunks.

        Each returned string is formatted as:

            [Session N — <date_time>]
            [D1:2] Speaker: text
            [D1:3] Speaker: text
            ...

        Returns:
            List of chunk strings ready for the GSW operator.
        """
        turns = session.turns
        if not turns:
            return []

        split_indices = self._detect_topic_boundaries(turns)

        if not split_indices:
            # Fall back to fixed-size chunks
            split_indices = list(range(self.max_turns_per_chunk, len(turns), self.max_turns_per_chunk))

        turn_groups = _chunk_by_indices(turns, split_indices)
        return [self._format_chunk(session, group) for group in turn_groups]

    def chunk_session_from_text(
        self, resolved_text: str, session: Session
    ) -> List[str]:
        """Split a coref-resolved session text into topically coherent chunks.

        Topic boundaries are detected from the original ``session.turns``
        (which have accurate dialogue content before coref substitution).
        The chunk strings are assembled by slicing the resolved_text lines,
        so that the GSW operator receives coref-resolved dialogue while
        preserving the same topic splits.

        Args:
            resolved_text: Coref-resolved session text (full session string).
            session: Original Session object — used for topic boundary detection
                     and to carry the session header metadata.

        Returns:
            List of chunk strings ready for the GSW operator.
        """
        turns = session.turns
        if not turns:
            return []

        # Determine split boundaries from the original session turns
        split_indices = self._detect_topic_boundaries(turns)
        if not split_indices:
            split_indices = list(
                range(self.max_turns_per_chunk, len(turns), self.max_turns_per_chunk)
            )

        # Parse the resolved text: extract dialogue lines.
        # Dialogue lines are non-empty lines that don't begin with "[Session".
        # The coref model returns Speaker: text lines (no [D...] prefix).
        all_lines = resolved_text.splitlines()
        dialogue_lines: List[str] = []
        for line in all_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("[Session"):
                dialogue_lines.append(line)

        session_header = f"[Session {session.session_id} — {session.date_time}]"

        # If we couldn't parse dialogue lines fall back to the original turns
        if not dialogue_lines:
            logger.warning(
                "Could not extract dialogue lines from resolved text for session %s; "
                "using original turns.",
                session.session_id,
            )
            turn_groups = _chunk_by_indices(turns, split_indices)
            return [self._format_chunk(session, group) for group in turn_groups]

        # Distribute resolved dialogue lines across chunks proportionally to
        # how many turns each original chunk contains.
        turn_groups = _chunk_by_indices(turns, split_indices)
        chunks: List[str] = []
        line_cursor = 0
        for group in turn_groups:
            n_turns = len(group)
            slice_lines = dialogue_lines[line_cursor : line_cursor + n_turns]
            line_cursor += n_turns
            body = "\n".join(slice_lines)
            chunks.append(f"{session_header}\n\n{body}")

        # Any leftover resolved lines (coref may have added/split lines) go
        # into the last chunk
        if line_cursor < len(dialogue_lines) and chunks:
            remaining = "\n".join(dialogue_lines[line_cursor:])
            chunks[-1] = chunks[-1] + "\n" + remaining

        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_topic_boundaries(self, turns: List[Turn]) -> List[int]:
        """Call the LLM to find topic-shift indices. Returns [] on failure."""
        numbered = "\n".join(
            f"{i}. [{turn.dia_id}] {turn.speaker}: {turn.text}"
            for i, turn in enumerate(turns)
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _TOPIC_DETECTION_SYSTEM},
                    {
                        "role": "user",
                        "content": _TOPIC_DETECTION_USER.format(
                            numbered_turns=numbered
                        ),
                    },
                ],
                temperature=0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ""
            # Extract JSON even if surrounded by markdown fences
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                indices = [int(x) for x in data.get("split_indices", [])]
                return indices
        except Exception as exc:
            logger.warning("Topic boundary detection failed (%s); using fixed chunking.", exc)
        return []

    def _format_chunk(self, session: Session, turns: List[Turn]) -> str:
        """Format a turn group as a single chunk string."""
        header = f"[Session {session.session_id} — {session.date_time}]"
        body = _format_turns_with_ids(turns)
        return f"{header}\n\n{body}"
