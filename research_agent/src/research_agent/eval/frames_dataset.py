"""FRAMES dataset loader.

Reuses the pre-cached FRAMES dev / full splits living under the
sister ``gsw-memory`` repo at
``data/sleep_time/frames/{frames_dev.json, frames_split.json}``. No
re-download needed. The article payloads are also already cached
under ``data/sleep_time/frames/articles/`` — we look those up on demand.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_GSW_ROOT = Path(os.environ.get("GSW_MEMORY_ROOT", "/home/yigit/codebase/gsw-memory"))
FRAMES_DEFAULT_DEV_PATH = DEFAULT_GSW_ROOT / "data" / "sleep_time" / "frames" / "frames_dev.json"
FRAMES_DEFAULT_FULL_PATH = DEFAULT_GSW_ROOT / "data" / "sleep_time" / "frames" / "frames_split.json"
FRAMES_ARTICLES_DIR = DEFAULT_GSW_ROOT / "data" / "sleep_time" / "frames" / "articles"


class FramesQuestion(BaseModel):
    """One FRAMES question."""

    id: str
    question: str
    answer: str
    articles: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_types: list[str] = Field(default_factory=list)
    num_hops: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "FramesQuestion":
        # The FRAMES dev split stores two variants:
        #   - ``original_prompt`` — the well-formed full question,
        #     e.g. "Joel Oshiro Dyck played ice hockey for three teams.
        #     Which of those teams were dissolved after 2018-19?"
        #   - ``test_question``   — just the tail, e.g. "Which of those
        #     teams were dissolved after 2018-19?"  — contains unbound
        #     pronouns ("those teams", "that film") that make the question
        #     ambiguous without the preceding context sentence.
        # We want the grounded variant, so ``original_prompt`` wins.
        question = (
            raw.get("original_prompt")
            or raw.get("test_question")
            or ""
        )
        return cls(
            id=str(raw.get("id", "")),
            question=question,
            answer=raw.get("answer", ""),
            articles=list(raw.get("articles") or []),
            reasoning_types=list(raw.get("reasoning_types") or []),
            num_hops=int(raw.get("num_hops") or 0),
            raw=raw,
        )


def load_frames(
    *,
    split: str = "dev",
    path: Path | str | None = None,
) -> list[FramesQuestion]:
    """Load the dev (100 Qs) or full (824 Qs) FRAMES split.

    Parameters
    ----------
    split:
        ``"dev"`` (default) or ``"full"``. Ignored if ``path`` is given.
    path:
        Explicit path override.
    """
    if path is None:
        path = FRAMES_DEFAULT_DEV_PATH if split == "dev" else FRAMES_DEFAULT_FULL_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FRAMES file not found at {path!s}. Expected the pre-cached split under "
            "gsw-memory/data/sleep_time/frames/. Override GSW_MEMORY_ROOT if needed."
        )
    with path.open() as f:
        raw = json.load(f)
    return [FramesQuestion.from_raw(item) for item in raw]
