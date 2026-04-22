"""Canonical output shape for every emitted decomposer.

Every deterministic decomposer, whether specialized per-dataset or
cross-dataset generalized, returns ``list[SubQuestion]``. The shape is a
superset of both downstream consumers:

* ``gsw_memory.query_then_sleep.QuerySubQuestionTrace`` — uses
  ``sub_question / relation_hint / entity_hint``.
* ``research_agent.adapters.ours_gsw_v1.SubQuestion`` — uses
  ``sub_id / question / entity_focus / hop_type``.

The rule_decomp_gsw adapter maps from this canonical shape to whichever
consumer it is wrapping. The synthesis loop grades emitted decomposers
by comparing this shape against the dataset's annotated decomposition.

Invariant: references to earlier sub-answers use ``#1``, ``#2`` style —
same convention MuSiQue ships in its ``question_decomposition`` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SubQuestion:
    sub_id: str
    question: str
    entity_focus: str = ""
    relation_hint: str = ""
    hop_type: str = ""
    references: List[str] = field(default_factory=list)

    def to_query_trace_kwargs(self) -> dict:
        return {
            "sub_question": self.question or "",
            "relation_hint": self.relation_hint or "",
            "entity_hint": self.entity_focus or "",
        }

    def to_ours_gsw_kwargs(self) -> dict:
        return {
            "sub_id": self.sub_id or "",
            "question": self.question or "",
            "entity_focus": self.entity_focus or "",
            "hop_type": self.hop_type or "lookup",
        }
