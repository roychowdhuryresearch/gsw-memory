"""Per-dataset loaders that normalise questions + gold decompositions.

Each loader returns ``list[dict]`` with at minimum:

    {"question": str, "gold_decomposition": list[dict]}

The synthesizer and eval CLI consume this shape directly. Dataset-specific
metadata (hop count, supporting paragraph ids, ...) are preserved on the
same dict so a richer grader can use them later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_monaco(path: str | Path, *, limit: int | None = None) -> List[dict]:
    """Load MoNaCo question + QDMR decomposition pairs.

    MoNaCo ships decompositions as a list of natural-language steps with
    ``#N`` references plus QDMR-ish operators (``return #1, #2``,
    ``return different #3``, ``[list]`` markers). Each step maps to one
    sub-question dict with keys ``id`` (step index, 1-based) and
    ``question`` (the raw step text, preserving ``#N`` references).

    Source file: ``monaco_version_1_release.json``. The JSON is a dict
    keyed by question text; each value has ``{canary, ex_num, question,
    decomposition, validated_answer}``.
    """
    data = json.loads(Path(path).read_text())
    out: List[dict] = []
    # Iteration order is insertion order (Python 3.7+), which is stable
    for i, (_key, row) in enumerate(data.items()):
        if limit is not None and i >= limit:
            break
        steps = row.get("decomposition") or []
        step_count = len(steps)
        # Bucket by decomposition length (matches the grader's per-hop
        # breakdown convention). 1-2 / 3-4 / 5-6 / 7-8 / 9+ groups keep
        # the MuSiQue-style naming (2hop/3hop/4hop/...) readable.
        if step_count <= 2:
            hop = f"{max(1, step_count)}hop"
        elif step_count <= 9:
            hop = f"{step_count}hop"
        else:
            hop = "10plus"
        out.append(
            {
                "id": f"monaco_{row.get('ex_num', i)}",
                "question": row.get("question", ""),
                "answer": row.get("validated_answer"),
                "hop": hop,
                "gold_decomposition": [
                    {"id": idx + 1, "question": step, "answer": None}
                    for idx, step in enumerate(steps)
                ],
            }
        )
    return out


def load_monaco_stratified(
    path: str | Path,
    *,
    per_bucket: int = 40,
    buckets: tuple[str, ...] = ("3hop", "4hop", "5hop", "6hop", "7hop"),
    seed: int = 0,
) -> List[dict]:
    """Sample ``per_bucket`` questions from each decomposition-length bucket.

    Defaults to 3/4/5/6/7-step questions — the dense middle of the MoNaCo
    length distribution (median 5). Hop buckets 1/2 are too trivial and
    8+ are rare.
    """
    import hashlib

    items = load_monaco(path)
    out: List[dict] = []
    for bucket in buckets:
        pool = [x for x in items if x.get("hop") == bucket]
        pool.sort(
            key=lambda x: hashlib.sha256(
                f"{seed}:stratify_monaco:{x.get('id', '')}".encode()
            ).hexdigest()
        )
        out.extend(pool[:per_bucket])
    return out


def load_musique_stratified(
    path: str | Path,
    *,
    per_bucket: int = 40,
    buckets: tuple[str, ...] = ("2hop", "3hop", "4hop"),
    seed: int = 0,
) -> List[dict]:
    """Sample ``per_bucket`` questions from each hop bucket.

    Used to balance the training pool so the Script-Writer sees plenty of
    3/4-hop examples, not just the 2-hop majority class. Uses
    deterministic hash-based selection so repeated calls are stable.
    """
    import hashlib

    items = load_musique(path)
    out: List[dict] = []
    for bucket in buckets:
        pool = [x for x in items if x.get("hop") == bucket]
        pool.sort(
            key=lambda x: hashlib.sha256(
                f"{seed}:stratify:{x.get('id', '')}".encode()
            ).hexdigest()
        )
        out.extend(pool[:per_bucket])
    return out


def load_musique(path: str | Path, *, limit: int | None = None) -> List[dict]:
    """Load MuSiQue question + question_decomposition pairs.

    MuSiQue ships the annotated decomposition inline. Each sub-question
    already uses ``#1``/``#2`` reference syntax.
    """
    data = json.loads(Path(path).read_text())
    out: List[dict] = []
    for i, row in enumerate(data):
        if limit is not None and i >= limit:
            break
        qid = str(row.get("id") or "")
        hop = ""
        if qid.startswith("2hop"):
            hop = "2hop"
        elif qid.startswith("3hop"):
            hop = "3hop"
        elif qid.startswith("4hop"):
            hop = "4hop"
        out.append(
            {
                "id": qid,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "hop": hop,
                "gold_decomposition": [
                    {
                        "id": step.get("id"),
                        "question": step.get("question", ""),
                        "answer": step.get("answer", ""),
                        "paragraph_support_idx": step.get("paragraph_support_idx"),
                    }
                    for step in (row.get("question_decomposition") or [])
                ],
            }
        )
    return out


def split_train_dev(
    items: List[dict], *, dev_fraction: float = 0.25, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/dev split by a stable hash of ``id``.

    No reshuffle across calls with the same seed, so the synthesizer and
    the eval harness grade on identical dev splits when given the same
    seed.
    """
    import hashlib

    train: List[dict] = []
    dev: List[dict] = []
    for ex in items:
        key = f"{seed}:{ex.get('id', '')}:{ex.get('question', '')}"
        h = hashlib.sha256(key.encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        (dev if bucket < dev_fraction else train).append(ex)
    return train, dev
