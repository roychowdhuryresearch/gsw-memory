"""High-level access to a standalone Panini course dataset package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import load_jsonl


@dataclass
class CoursePackage:
    """Paths and metadata for one materialized course-project release."""

    root: Path

    def __init__(self, root: str | Path):
        root_path = Path(root).expanduser().resolve()
        if not (root_path / "manifest.json").exists():
            raise FileNotFoundError(
                f"Not a Panini course package: {root_path}"
            )
        self.root = root_path

    @property
    def manifest(self) -> dict[str, Any]:
        with (self.root / "manifest.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    def questions(self, split: str = "public") -> list[dict[str, Any]]:
        if split not in {"public", "held_out", "instructor"}:
            raise ValueError(f"Unknown question split: {split}")
        path = (
            self.root / "instructor" / "questions_with_gold.jsonl"
            if split == "instructor"
            else self.root / "questions" / f"{split}.jsonl"
        )
        return list(load_jsonl(path))

    def documents(self) -> list[dict[str, Any]]:
        return list(load_jsonl(self.root / "documents.jsonl"))

    def entities(self) -> list[dict[str, Any]]:
        return list(load_jsonl(self.root / "metadata" / "entities.jsonl"))

    def qa_pairs(self) -> list[dict[str, Any]]:
        return list(load_jsonl(self.root / "metadata" / "qa_pairs.jsonl"))

    def decompositions(self, *, instructor: bool = False) -> dict[str, list[dict]]:
        path = (
            self.root / "instructor" / "reviewed_decompositions.jsonl"
            if instructor
            else self.root / "questions" / "decomposition_validation.jsonl"
        )
        return {
            row["question_id"]: row["decomposed_questions"]
            for row in load_jsonl(path)
        }

    def gsw_paths(self) -> list[Path]:
        return sorted((self.root / "gsws").glob("doc_*/gsw_*.json"))

    def metadata_by_id(self, kind: str) -> dict[str, dict[str, Any]]:
        if kind == "entity":
            rows = self.entities()
            id_field = "entity_uid"
        elif kind == "qa":
            rows = self.qa_pairs()
            id_field = "qa_uid"
        else:
            raise ValueError("kind must be 'entity' or 'qa'")
        return {row[id_field]: row for row in rows}
