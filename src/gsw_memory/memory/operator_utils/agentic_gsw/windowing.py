from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .schemas import TextWindow


def split_into_windows(text: str, strategy: str = "paragraph") -> List[TextWindow]:
    """Split text into overlapping windows with stable document offsets.

    Strategies:
    - ``"whole"``: treat the entire document as a single window. Relations,
      grounding, and repair routing all lose window-level granularity, but
      cross-paragraph relations are preserved and the prompt overhead of
      fanning out drops to one call.
    - ``"paragraph"`` (default): one window per paragraph with 1-sentence
      neighbor bleed.
    - fallback: 5-sentence sliding windows with 2-sentence stride.
    """
    raw_text = str(text or "")

    if strategy == "whole":
        stripped = raw_text.strip()
        if not stripped:
            return []
        start = raw_text.index(stripped)
        end = start + len(stripped)
        return [
            TextWindow(
                id="w1",
                text=raw_text[start:end],
                char_start=start,
                char_end=end,
                core_char_start=start,
                core_char_end=end,
            )
        ]

    sentence_spans = _split_sentence_spans(raw_text)
    if not sentence_spans:
        stripped = raw_text.strip()
        if not stripped:
            return []
        start = raw_text.index(stripped)
        end = start + len(stripped)
        return [
            TextWindow(
                id="w1",
                text=raw_text[start:end],
                char_start=start,
                char_end=end,
                core_char_start=start,
                core_char_end=end,
            )
        ]

    if strategy == "paragraph":
        paragraph_spans = _split_paragraph_spans(raw_text)
        if len(paragraph_spans) > 1:
            return _paragraph_windows(raw_text, sentence_spans, paragraph_spans)

    return _sentence_windows(raw_text, sentence_spans)


def overlaps_window(char_start: int, char_end: int, window: TextWindow) -> bool:
    return char_start < window.char_end and char_end > window.char_start


def assign_primary_window_id(
    char_start: int,
    windows: Sequence[TextWindow],
) -> str | None:
    for window in windows:
        if window.core_char_start <= char_start < window.core_char_end:
            return window.id
    for window in windows:
        if window.char_start <= char_start < window.char_end:
            return window.id
    return None


def _paragraph_windows(
    text: str,
    sentence_spans: Sequence[Tuple[int, int]],
    paragraph_spans: Sequence[Tuple[int, int]],
) -> List[TextWindow]:
    windows: List[TextWindow] = []
    for idx, (para_start, para_end) in enumerate(paragraph_spans, start=1):
        sentence_indices = [
            i
            for i, (sent_start, sent_end) in enumerate(sentence_spans)
            if sent_start < para_end and sent_end > para_start
        ]
        if not sentence_indices:
            continue
        expanded_start_idx = max(0, sentence_indices[0] - 1)
        expanded_end_idx = min(len(sentence_spans) - 1, sentence_indices[-1] + 1)
        win_start = sentence_spans[expanded_start_idx][0]
        win_end = sentence_spans[expanded_end_idx][1]
        windows.append(
            TextWindow(
                id=f"w{idx}",
                text=text[win_start:win_end],
                char_start=win_start,
                char_end=win_end,
                core_char_start=para_start,
                core_char_end=para_end,
            )
        )
    return windows


def _sentence_windows(text: str, sentence_spans: Sequence[Tuple[int, int]]) -> List[TextWindow]:
    windows: List[TextWindow] = []
    window_size = 5
    stride = 2
    idx = 0
    window_num = 1
    while idx < len(sentence_spans):
        chunk = sentence_spans[idx : idx + window_size]
        if not chunk:
            break
        win_start = chunk[0][0]
        win_end = chunk[-1][1]
        windows.append(
            TextWindow(
                id=f"w{window_num}",
                text=text[win_start:win_end],
                char_start=win_start,
                char_end=win_end,
                core_char_start=win_start,
                core_char_end=win_end,
            )
        )
        if idx + window_size >= len(sentence_spans):
            break
        idx += stride
        window_num += 1
    return windows


def _split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: int | None = None
    idx = 0
    length = len(text)
    while idx < length:
        char = text[idx]
        if start is None:
            if char.isspace():
                idx += 1
                continue
            start = idx

        if char in ".?!" and (idx + 1 == length or text[idx + 1].isspace()):
            spans.append((start, idx + 1))
            start = None
        elif char == "\n":
            match = re.match(r"\n\s*\n+", text[idx:])
            if match:
                end = idx
                if start is not None and start < end:
                    spans.append((start, end))
                start = None
                idx += len(match.group(0))
                continue
        idx += 1

    if start is not None and start < length:
        spans.append((start, length))

    return [(start, end) for start, end in spans if text[start:end].strip()]


def _split_paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n+", text):
        start = cursor
        end = match.start()
        if text[start:end].strip():
            trimmed_start = start + len(text[start:end]) - len(text[start:end].lstrip())
            trimmed_end = start + len(text[start:end].rstrip())
            spans.append((trimmed_start, trimmed_end))
        cursor = match.end()
    if text[cursor:].strip():
        segment = text[cursor:]
        trimmed_start = cursor + len(segment) - len(segment.lstrip())
        trimmed_end = cursor + len(segment.rstrip())
        spans.append((trimmed_start, trimmed_end))
    return spans
