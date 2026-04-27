"""FRAMES article corpus loader + chunker.

The FRAMES article cache lives at
``{GSW_MEMORY_ROOT}/data/sleep_time/frames/articles/articles_cache.json``
as a ``{title: full_text}`` dict. We chunk each article into ~512-token
paragraphs and expose a simple store.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GSW_ROOT = Path(os.environ.get("GSW_MEMORY_ROOT", "/home/yigit/codebase/gsw-memory"))
DEFAULT_CACHE_PATH = DEFAULT_GSW_ROOT / "data" / "sleep_time" / "frames" / "articles" / "articles_cache.json"

# Conservative token estimate: 1 token ≈ 4 chars. ~512 tokens ≈ 2000 chars.
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(slots=True)
class Chunk:
    """One searchable chunk."""

    chunk_id: str
    title: str
    text: str
    char_start: int
    char_end: int


def _split_on_paragraphs(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Greedy paragraph-aware chunking with hard-cap fallback.

    A "paragraph" is a run of non-empty lines (separated by blank
    lines). When a paragraph itself exceeds ``max_chars`` (e.g., a
    Wikipedia table flattened to one-row-per-line with no blank
    separators), we force-split it on line boundaries. As a final
    safety net, any single line longer than ``max_chars`` is
    character-sliced.
    """
    paragraphs = [m for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text)]
    if not paragraphs:
        return [(0, len(text))]

    # First split each over-long paragraph into line-bounded
    # sub-spans of <= max_chars each. Paragraphs under the cap pass
    # through unchanged.
    sub_spans: list[tuple[int, int]] = []
    for p in paragraphs:
        if p.end() - p.start() <= max_chars:
            sub_spans.append((p.start(), p.end()))
            continue
        # Walk lines inside the over-long paragraph, accumulating up
        # to max_chars per sub-span.
        body_start = p.start()
        body_end = p.end()
        body = text[body_start:body_end]
        line_offsets: list[tuple[int, int]] = []  # (start, end) within body
        cur = 0
        for line_match in re.finditer(r"[^\n]*(?:\n|$)", body):
            ls, le = line_match.start(), line_match.end()
            if le == ls:
                continue
            line_offsets.append((ls, le))
        if not line_offsets:
            line_offsets = [(0, len(body))]
        sub_start = line_offsets[0][0]
        sub_end = sub_start
        for ls, le in line_offsets:
            if le - sub_start > max_chars:
                if sub_end > sub_start:
                    sub_spans.append((body_start + sub_start, body_start + sub_end))
                sub_start = ls
                sub_end = le
                # Final safety net: a single line longer than max_chars
                # gets character-sliced into max_chars pieces.
                if le - ls > max_chars:
                    pos = ls
                    while pos < le:
                        chunk_end = min(pos + max_chars, le)
                        sub_spans.append((body_start + pos, body_start + chunk_end))
                        pos = chunk_end
                    sub_start = le
                    sub_end = le
            else:
                sub_end = le
        if sub_end > sub_start:
            sub_spans.append((body_start + sub_start, body_start + sub_end))

    # Now greedily re-merge adjacent sub-spans up to max_chars (this
    # gives us paragraph-aware chunks where possible while never
    # exceeding the cap on a single chunk).
    spans: list[tuple[int, int]] = []
    if not sub_spans:
        return [(0, len(text))]
    start = sub_spans[0][0]
    cur_end = sub_spans[0][1]
    for s, e in sub_spans[1:]:
        if e - start > max_chars and cur_end > start:
            spans.append((start, cur_end))
            start = max(cur_end - overlap, s)
            cur_end = e
        else:
            cur_end = e
    if start < cur_end:
        spans.append((start, cur_end))
    return spans


class ArticleCorpus:
    """In-memory corpus of FRAMES articles, chunked for retrieval."""

    def __init__(self, chunks: list[Chunk], by_title: dict[str, str]):
        self.chunks = chunks
        self._by_title = by_title
        self._by_id = {c.chunk_id: c for c in chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def article_text(self, title: str) -> str:
        return self._by_title.get(title, "")

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def titles(self) -> list[str]:
        return sorted(self._by_title.keys())


def load_frames_corpus(
    *,
    cache_path: Path | str | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ArticleCorpus:
    """Load the pre-cached FRAMES article dict and produce an ArticleCorpus."""
    path = Path(cache_path or DEFAULT_CACHE_PATH)
    with path.open() as f:
        data: dict[str, str] = json.load(f)

    chunks: list[Chunk] = []
    for title, text in data.items():
        if not text:
            continue
        spans = _split_on_paragraphs(text, chunk_chars, chunk_overlap)
        for i, (s, e) in enumerate(spans):
            chunks.append(
                Chunk(
                    chunk_id=f"{title}__{i}",
                    title=title,
                    text=text[s:e],
                    char_start=s,
                    char_end=e,
                )
            )
    return ArticleCorpus(chunks=chunks, by_title=data)
