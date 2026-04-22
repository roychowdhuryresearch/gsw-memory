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
    """Greedy paragraph-aware chunking. Returns (start, end) offsets."""
    paragraphs = [m for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text)]
    if not paragraphs:
        return [(0, len(text))]

    spans: list[tuple[int, int]] = []
    start = paragraphs[0].start()
    cur_end = start
    for p in paragraphs:
        if p.end() - start > max_chars and cur_end > start:
            spans.append((start, cur_end))
            start = max(cur_end - overlap, p.start())
            cur_end = p.end()
        else:
            cur_end = p.end()
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
