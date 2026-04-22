"""Pilot subset selector + loader.

Every adapter runs against exactly the same subset across all cells of
the grid. For reproducibility we persist the selected subset to a JSON
file (subset_id + ordered list of question ids) and load it back on
each run.

The stratification target: balance num_hops buckets so we see behavior
on easy (2-hop), medium (3-4), and hard (5+) multi-hop in the pilot.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from research_agent.eval.frames_dataset import FramesQuestion


class PilotSubset(BaseModel):
    """Persisted subset definition."""

    subset_id: str
    benchmark: str
    split: str
    question_ids: list[str] = Field(default_factory=list)
    stratification: dict[str, int] = Field(
        default_factory=dict,
        description="Counts per stratum that the selector targeted.",
    )
    seed: int = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PilotSubset":
        return cls.model_validate_json(Path(path).read_text())


def _bucket(hops: int) -> str:
    if hops <= 2:
        return "2_hop"
    if hops == 3:
        return "3_hop"
    if hops <= 5:
        return "4_5_hop"
    return "6plus_hop"


def select_stratified_subset(
    questions: Iterable[FramesQuestion],
    *,
    subset_id: str = "frames_pilot_v1",
    benchmark: str = "frames",
    split: str = "dev",
    targets: dict[str, int] | None = None,
    seed: int = 0,
) -> PilotSubset:
    """Pick a stratified subset from the given FRAMES questions.

    ``targets`` maps bucket -> count. Default targets produce a 30-Q
    pilot: 10 easy + 10 medium + 8 hard + 2 long.
    """
    targets = targets or {
        "2_hop": 10,
        "3_hop": 10,
        "4_5_hop": 8,
        "6plus_hop": 2,
    }

    by_bucket: dict[str, list[FramesQuestion]] = defaultdict(list)
    for q in questions:
        by_bucket[_bucket(q.num_hops)].append(q)

    rng = random.Random(seed)
    picked_ids: list[str] = []
    actual_counts: dict[str, int] = {}

    for bucket, n_target in targets.items():
        pool = by_bucket.get(bucket, [])
        if not pool:
            actual_counts[bucket] = 0
            continue
        n = min(n_target, len(pool))
        sample = rng.sample(pool, n)
        picked_ids.extend(q.id for q in sample)
        actual_counts[bucket] = n

    return PilotSubset(
        subset_id=subset_id,
        benchmark=benchmark,
        split=split,
        question_ids=picked_ids,
        stratification=actual_counts,
        seed=seed,
    )


def load_subset(path: str | Path) -> PilotSubset:
    """Load a persisted subset."""
    return PilotSubset.load(path)


def filter_to_subset(
    questions: Iterable[FramesQuestion],
    subset: PilotSubset,
) -> list[FramesQuestion]:
    """Return only the questions whose id is in the subset, preserving order."""
    id_set = set(subset.question_ids)
    by_id = {q.id: q for q in questions}
    return [by_id[qid] for qid in subset.question_ids if qid in id_set and qid in by_id]
