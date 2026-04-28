from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallTrace:
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    call_id: str = ""
    iteration: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuerySubQuestionTrace:
    sub_question: str
    relation_hint: str = ""
    entity_hint: str = ""
    answer_found: bool = False
    answer: str = ""
    searched_for: str = ""
    found_instead: List[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryInteractionTrace:
    question_id: str
    question: str
    doc_ids: List[str]
    gold_answers: List[str] = field(default_factory=list)
    sub_questions: List[QuerySubQuestionTrace] = field(default_factory=list)
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    final_answer: str = ""
    final_reasoning: str = ""
    exact_match: Optional[float] = None
    f1: Optional[float] = None
    found_relations: List[str] = field(default_factory=list)
    missing_relations: List[str] = field(default_factory=list)
    bridge_evidence_used: bool = False
    bridge_evidence_injected: bool = False
    bridge_evidence_used_in_answer: bool = False
    bridge_ids_used: List[str] = field(default_factory=list)
    bridge_ids_seen: List[str] = field(default_factory=list)
    bridge_ids_injected: List[str] = field(default_factory=list)
    tool_iterations: int = 0
    status: str = "ok"
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "sub_questions": [item.as_dict() for item in self.sub_questions],
            "tool_calls": [item.as_dict() for item in self.tool_calls],
        }


@dataclass
class QueryAgentBatchResult:
    batch_index: int
    traces: List[QueryInteractionTrace]
    overall_metrics: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_index": int(self.batch_index),
            "traces": [trace.as_dict() for trace in self.traces],
            "overall_metrics": dict(self.overall_metrics),
        }


@dataclass
class InteractionGuidanceSummary:
    guidance_mode: str
    batch_index: int
    questions_answered: int
    avg_f1: float
    relation_gap_profile: List[Dict[str, Any]] = field(default_factory=list)
    bridge_chain_profile: List[Dict[str, Any]] = field(default_factory=list)
    undercovered_chain_families: List[Dict[str, Any]] = field(default_factory=list)
    deprioritized_chain_families: List[Dict[str, Any]] = field(default_factory=list)
    entity_gap_targets: List[Dict[str, Any]] = field(default_factory=list)
    successful_relation_profile: List[Dict[str, Any]] = field(default_factory=list)
    missing_bridge_targets: List[Dict[str, Any]] = field(default_factory=list)
    natural_language_summary: str = ""
    supporting_examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuerySleepBatchRecord:
    batch_index: int
    questions_answered: int
    avg_f1: float
    avg_em: float
    bridges_before_batch: int
    bridges_generated_for_next_batch: int
    bridges_after_sleep: int
    bridge_usage_rate: float
    bridge_answer_usage_rate: float = 0.0
    sleep_targeted_edge_ratio: float = 0.7
    missing_relation_counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
