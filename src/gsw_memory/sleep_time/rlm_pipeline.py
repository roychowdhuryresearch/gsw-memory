"""RLM-style deterministic root scheduler with recursive edge workers.

This module keeps exploration state in Python and only sends compact edge-local
packets to worker model calls to control token growth.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .tools import normalize_similarity_text, tokenize_similarity_text

logger = logging.getLogger(__name__)


WORKER_PROMPT_VERSION = "legacy_reverse_focus_4x4_v2"
PLANNER_PROMPT_VERSION = "hybrid_planner_ds_v2"
COMMIT_SIMILARITY_RECHECK_THRESHOLD = 0.70
GUIDED_RETRY_LIMIT = 2

DROP_REASON_CODES = (
    "missing_source_ref",
    "missing_target_ref",
    "reverse_not_invertible",
    "no_cross_doc_evidence",
    "all_candidates_rejected",
)

WORKER_PRE_OUTPUT_CHECKLIST = """
Pre-output checklist (must pass all):
1. Forward chain dependency: hop-2 must require hop-1 output (#1).
2. Reverse inversion proof is mandatory in reasoning:
   - Forward target variable = X
   - Reverse target variable = source-side variable recovered by inverting hop-2 then hop-1
3. Reverse question truly inverts the forward mapping, not an independent fact.
4. Reasoning includes both:
   - Sub-Q1: [source entity] -> [relationship] -> #1 (doc_X)
   - Sub-Q2: #1 -> [second-hop fact] -> answer (doc_Y)
5. answers/reverse_answers are grounded refs present in provided evidence.
6. Avoid answer leakage and one-hop restatements.
7. If candidates are non-empty, notes must briefly justify why reverse is a true inversion.
"""

WORKER_LEGACY_EXAMPLES = """
Legacy examples (4 good + 4 bad):

GOOD #1
Forward: "Where did Master Aldric's patron build a chapel?"
Reverse: "Whose patron built a chapel in Bruges?"
Forward mapping: Master Aldric -> Lady Margaux -> Bruges
Reverse inversion: Bruges -> Lady Margaux -> Master Aldric
Reasoning:
Forward target variable = Bruges
Reverse target variable = Master Aldric
Sub-Q1: Master Aldric -> patron -> Lady Margaux (doc_5)
Sub-Q2: Lady Margaux -> built chapel in -> Bruges (doc_11)

GOOD #2
Forward: "Who founded the Abbey of Saint Benedict's sister monastery?"
Reverse: "Abbess Hildegard founded the sister monastery of which abbey?"
Forward mapping: Abbey of Saint Benedict -> Convent of Holy Cross -> Abbess Hildegard
Reverse inversion: Abbess Hildegard -> Convent of Holy Cross -> Abbey of Saint Benedict
Reasoning:
Forward target variable = Abbess Hildegard
Reverse target variable = Abbey of Saint Benedict
Sub-Q1: Abbey of Saint Benedict -> sister monastery -> Convent of Holy Cross (doc_15)
Sub-Q2: Convent of Holy Cross -> founded by -> Abbess Hildegard (doc_34)

GOOD #3
Forward: "Where did Silversmith Marco's apprentice open their workshop?"
Reverse: "Whose apprentice opened a workshop in Venice?"
Forward mapping: Silversmith Marco -> Paolo -> Venice
Reverse inversion: Venice -> Paolo -> Silversmith Marco
Reasoning:
Forward target variable = Venice
Reverse target variable = Silversmith Marco
Sub-Q1: Silversmith Marco -> apprentice -> Paolo (doc_8)
Sub-Q2: Paolo -> opened workshop in -> Venice (doc_23)

GOOD #4
Forward: "When was the forge where Master Aldric trained established?"
Reverse: "Who trained at the forge established in 1203?"
Forward mapping: Master Aldric -> Ironworks of Ghent -> 1203
Reverse inversion: 1203 -> Ironworks of Ghent -> Master Aldric
Reasoning:
Forward target variable = 1203
Reverse target variable = Master Aldric
Sub-Q1: Master Aldric -> trained at -> Ironworks of Ghent (doc_5)
Sub-Q2: Ironworks of Ghent -> established in -> 1203 (doc_29)

BAD #1 (REVERSE_INVALID independent reverse)
Forward: "Where did Master Aldric's patron build a chapel?" -> Bruges
Reverse: "Who founded Lady Margaux's chapel?" -> Bishop Renald
Reason: reverse asks a different branch and does not invert Bruges -> Lady Margaux -> Master Aldric.

BAD #2 (NOT_CHAIN conjunction)
Forward: "Which NATO member country lacks a traditional army and participated in the ideological conflict with the Warsaw Pact?"
Reason: independent conjunction; hop-2 does not require hop-1 output.

BAD #3 (ANSWER_LEAK / TOO_TRIVIAL one-hop)
Forward: "What city did Lady Margaux build a chapel in Bruges?"
Reason: answer is leaked in the question; no two-hop reasoning needed.

BAD #4 (CIRCULAR reverse)
Forward: "Where did Silversmith Marco's apprentice open their workshop?" -> Venice
Reverse: "Where did Silversmith Marco's apprentice open their workshop?" -> Venice
Reason: reverse repeats forward question; it does not invert back to the source-side variable.
"""



def _format_curriculum_guidance_block(guidance: Optional[Dict[str, Any]]) -> str:
    if not isinstance(guidance, dict) or not guidance:
        return ""

    sections: List[str] = ["Curriculum guidance (advisory only):"]

    demand = guidance.get("prior_query_demand_sketch")
    if not isinstance(demand, dict):
        demand = guidance.get("current_batch_demand_sketch")
    if isinstance(demand, dict):
        top_patterns = demand.get("top_patterns", []) or []
        if top_patterns:
            sections.append("Prior Query Demand Sketch:")
            prior_batches_seen = int(guidance.get("prior_batches_seen", demand.get("batches_seen", 0)) or 0)
            if prior_batches_seen > 0:
                sections.append(f"- Derived from {prior_batches_seen} earlier answered batch(es).")
            for row in top_patterns[:4]:
                pattern = str(row.get("pattern", "")).strip()
                count = int(row.get("count", 0) or 0)
                representative = str(row.get("representative_query", "")).strip()
                if not pattern:
                    continue
                line = f"- {pattern} x{count}"
                if representative:
                    line += f" | rep: {representative}"
                sections.append(line)

    feedback = guidance.get("prior_batch_feedback_brief")
    if isinstance(feedback, dict):
        summary_lines = feedback.get("summary_lines", []) or []
        if summary_lines:
            sections.append("Prior Batch Feedback Brief:")
            for line in summary_lines[:4]:
                cleaned = str(line).strip()
                if cleaned:
                    sections.append(f"- {cleaned}")

    exemplars = guidance.get("bridge_exemplars")
    if isinstance(exemplars, list) and exemplars:
        sections.append("Relevant Prior Bridge Exemplars:")
        for row in exemplars[:3]:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "")).strip()
            reverse_question = str(row.get("reverse_question", "")).strip()
            pattern_tags = row.get("pattern_tags", []) or []
            tags = ", ".join(str(tag).strip() for tag in pattern_tags if str(tag).strip())
            if question:
                sections.append(f"- Forward: {question}")
            if reverse_question:
                sections.append(f"  Reverse: {reverse_question}")
            if tags:
                sections.append(f"  Tags: {tags}")

    sections.append(
        "Use this historical guidance from earlier answered batches to prefer useful question types and missing patterns, but stay grounded in the provided evidence and do not copy wording mechanically."
    )
    return "\n".join(sections)

PLANNER_SHARED_CHECKLIST = """
Planner checklist (must pass all):
1. Choose only from provided candidates/actions.
2. Never output out-of-range ids/indexes.
3. Obey action/depth/include_optional constraints.
4. Return exactly one JSON object and no markdown.
"""

PLANNER_CORPUS_DOC_EXAMPLES = """
Decision: corpus_doc_selection
Required output schema:
{"doc_id": "<candidate doc_id>", "reason": "<one short sentence>"}

GOOD #1
Candidates: doc_3 (pending_neighbors=8), doc_7 (pending_neighbors=3)
Output: {"doc_id": "doc_3", "reason": "It has the highest pending neighbors among active docs."}

GOOD #2
Candidates: doc_1 (in_progress), doc_2 (unplanned)
Output: {"doc_id": "doc_1", "reason": "Continue the in-progress doc with unresolved neighbors."}

BAD #1 (out-of-candidate id)
Output: {"doc_id": "doc_99", "reason": "Looks promising."}

BAD #2 (missing reason + markdown)
Output: ```json {"doc_id":"doc_3"} ```
"""

PLANNER_DOC_EDGE_EXAMPLES = """
Decision: doc_edge_selection
Required output schema:
{"edge_index": <integer>, "reason": "<one short sentence>"}

GOOD #1
Candidates edge_index: 0,1,2
Output: {"edge_index": 1, "reason": "This edge has stronger cross-doc coverage signal."}

GOOD #2
Candidates edge_index: 0,1
Output: {"edge_index": 0, "reason": "Process deterministic front edge due equal priority."}

BAD #1 (out-of-range index)
Output: {"edge_index": 5, "reason": "Best edge."}

BAD #2 (wrong field)
Output: {"entity_name": "Lothair II", "reason": "Relevant."}
"""

PLANNER_EDGE_ACTION_EXAMPLES = """
Decision: edge_action
Required output schema:
{"action": "run_worker|stop_edge", "depth": <integer>, "include_optional": <boolean>, "reason": "<one short sentence>"}

GOOD #1
State: current_depth=0, max_depth=1, has_optional_contexts=true, no_candidate_calls=0
Output: {"action":"run_worker","depth":1,"include_optional":true,"reason":"New optional contexts can add missing evidence for a valid reverse inversion."}

GOOD #2
State: call_count near cap, rejected_total>0 with repeated reverse-invalid style failures, no_candidate_calls increasing, unchanged evidence
Output: {"action":"stop_edge","depth":0,"include_optional":false,"reason":"Stop because repeated reverse-quality failures show no progress with current evidence."}

BAD #1 (invalid action)
Output: {"action":"scan_more","depth":0,"include_optional":false,"reason":"Need more info."}

BAD #2 (invalid depth)
Output: {"action":"run_worker","depth":99,"include_optional":false,"reason":"Try deeper."}

BAD #3 (wasteful continuation after unchanged failures)
State: call_count high, rejected_total rising, no new optional evidence
Output: {"action":"run_worker","depth":1,"include_optional":false,"reason":"Keep trying same setup."}
"""


class WorkerCandidateSchema(BaseModel):
    """Candidate structure for worker structured outputs.

    Uses extra="ignore" so that extra fields from the model (e.g. source_fact,
    neighbor_fact) are silently dropped instead of causing validation errors
    when structured output enforcement falls back to json_object mode.
    """

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)
    answers: List[str] = Field(min_length=1)
    reverse_question: str = Field(min_length=1)
    reverse_answers: List[str] = Field(min_length=1)
    source_docs: List[str] = Field(min_length=2)
    reasoning: str = Field(min_length=1)
    confidence: float = 0.9
    entities_involved: List[Any] = Field(default_factory=list)


class WorkerResponseSchema(BaseModel):
    """Top-level worker response shape."""

    model_config = ConfigDict(extra="ignore")

    status: str
    candidates: List[WorkerCandidateSchema] = Field(default_factory=list)
    need_recursion: bool
    notes: str


class VerifierOutputSchema(BaseModel):
    """Verifier output structure for bridge candidate validation.

    Uses extra="ignore" so extra fields are silently dropped when structured
    output enforcement falls back to json_object mode.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    pass_field: bool = Field(alias="pass")
    score: float = Field(ge=0.0, le=1.0)
    failure_codes: List[str] = Field(default_factory=list)
    notes: str = ""
    retryable: bool = False
    retry_hint: str = "stop_edge"
    retry_reason: str = ""


class PlannerCorpusDocSelection(BaseModel):
    """Planner output for corpus document selection."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    reason: str


class PlannerDocEdgeSelection(BaseModel):
    """Planner output for document edge selection."""

    model_config = ConfigDict(extra="forbid")

    edge_index: int
    reason: str


class PlannerEdgeAction(BaseModel):
    """Planner output for edge action decision."""

    model_config = ConfigDict(extra="forbid")

    action: str
    depth: int = Field(ge=0)
    include_optional: bool
    reason: str


class OptionalDocSelection(BaseModel):
    """Output for optional document selection."""

    model_config = ConfigDict(extra="forbid")

    selected_doc_ids: List[str] = Field(default_factory=list)


@dataclass(frozen=True)
class EdgeKey:
    """Canonical edge tuple for one source entity -> neighbor relationship."""

    doc_id: str
    entity_name: str
    neighbor_name: str
    relationship: str

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return (self.doc_id, self.entity_name, self.neighbor_name, self.relationship)

    def as_dict(self) -> Dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "entity_name": self.entity_name,
            "neighbor_name": self.neighbor_name,
            "relationship": self.relationship,
        }


@dataclass
class EdgePacket:
    """Compact worker input for one edge."""

    edge: EdgeKey
    source_docs: List[str]
    mandatory_docs: List[str]
    optional_docs: List[str]
    source_context: Dict[str, Any]
    mandatory_contexts: List[Dict[str, Any]]
    optional_contexts: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    budget: Dict[str, int]


@dataclass
class PathProof:
    """Deterministic 2-hop path proof mined from source+neighbor contexts."""

    edge: EdgeKey
    source_fact: Dict[str, Any]
    neighbor_fact: Dict[str, Any]
    path_docs: List[str]
    target_refs: List[str]
    source_subject_refs: List[str]


@dataclass
class RenderInput:
    """Strict renderer input: only path proofs, no free context search."""

    edge: EdgeKey
    proofs: List[PathProof]
    constraints: Dict[str, Any]


@dataclass
class EdgeRetryBrief:
    """Structured verifier feedback for same-edge regeneration."""

    edge: EdgeKey
    failure_codes: List[str]
    retryable: bool
    retry_hint: str
    retry_reason: str
    rejected_question: str
    rejected_reverse_question: str
    retry_attempt: int = 0

    def signature(self) -> Tuple[Any, ...]:
        return (
            bool(self.retryable),
            str(self.retry_hint or ""),
            tuple(str(code).strip().upper() for code in self.failure_codes if str(code).strip()),
            normalize_similarity_text(self.rejected_question),
            normalize_similarity_text(self.rejected_reverse_question),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "edge": self.edge.as_dict(),
            "failure_codes": list(self.failure_codes),
            "retryable": bool(self.retryable),
            "retry_hint": str(self.retry_hint or ""),
            "retry_reason": str(self.retry_reason or ""),
            "rejected_question": str(self.rejected_question or ""),
            "rejected_reverse_question": str(self.rejected_reverse_question or ""),
            "retry_attempt": int(self.retry_attempt),
        }


@dataclass
class WorkerOutput:
    """Normalized output from one worker invocation."""

    status: str
    candidates: List[Dict[str, Any]]
    need_recursion: bool
    notes: str
    parse_stage: str
    raw_preview: str
    token_usage: int = 0
    budgeted_token_usage: int = 0
    parse_attempts: int = 1
    parse_recovery_used: bool = False


@dataclass
class EdgeRunResult:
    """Execution summary for one edge."""

    edge: EdgeKey
    accepted: int
    attempted: int
    rejected: int
    budget_exhausted: bool
    edge_tokens: int
    # Kept for backward compatibility in summaries/events.
    property_attempted: bool
    guided_retry_count: int = 0
    last_retry_hint: str = ""
    last_retry_failure_codes: List[str] = field(default_factory=list)


@dataclass
class PreparedEdgeWork:
    """Immutable packetized edge work item prepared under serial focus lock."""

    edge: EdgeKey
    packet: EdgePacket
    signal: Dict[str, Any]
    edge_call_limit: int
    edge_token_limit: int
    viability: Dict[str, Any]
    retry_brief: Optional[EdgeRetryBrief] = None
    guided_retry_count: int = 0


@dataclass
class EdgeWorkerProposal:
    """Parallel worker-generation proposal awaiting serial commit."""

    edge: EdgeKey
    packet: EdgePacket
    signal: Dict[str, Any]
    attempted: int
    prefilter_rejected: int
    sanitized_candidates: List[Dict[str, Any]]
    budget_exhausted: bool
    edge_tokens: int
    worker_calls_made: int
    no_candidate_calls: int
    close_reason: str
    close_detail: str
    planner_decisions: int
    planner_fallbacks: int
    planner_stop_actions: int
    edge_call_limit: int = 0
    edge_token_limit: int = 0
    drop_reason_counts: Dict[str, int] = field(default_factory=dict)
    retry_brief: Optional[EdgeRetryBrief] = None
    guided_retry_count: int = 0
    last_retry_hint: str = ""
    last_retry_failure_codes: List[str] = field(default_factory=list)
    last_run_signature: Optional[Tuple[Any, ...]] = None


class RecursiveEdgeWorker:
    """Runs bounded worker-model calls for compact edge packets."""

    def __init__(
        self,
        reconciler: Any,
        model_name: str,
        max_depth: int = 1,
        max_calls: int = 2,
        edge_max_tokens: int = 3000,
    ):
        self.reconciler = reconciler
        self.model_name = model_name
        self.max_depth = max(0, int(max_depth))
        self.max_calls = max(1, int(max_calls))
        self.edge_max_tokens = max(1, int(edge_max_tokens))

    @staticmethod
    def _worker_response_format() -> type:
        """Return Pydantic model class for litellm structured output."""
        return WorkerResponseSchema

    def _build_messages(
        self,
        packet: EdgePacket,
        depth: int,
        include_optional: bool,
        retry_brief: Optional[EdgeRetryBrief] = None,
    ) -> List[Dict[str, str]]:
        contexts = list(packet.mandatory_contexts)
        if include_optional:
            contexts.extend(packet.optional_contexts)

        payload = {
            "edge": packet.edge.as_dict(),
            "source_docs": packet.source_docs,
            "mandatory_docs": packet.mandatory_docs,
            "source_context": packet.source_context,
            "contexts": contexts,
            "constraints": packet.constraints,
            "depth": depth,
            "include_optional": include_optional,
        }
        if retry_brief is not None:
            payload["retry_brief"] = {
                "failure_codes": list(retry_brief.failure_codes),
                "retry_hint": retry_brief.retry_hint,
                "retry_reason": retry_brief.retry_reason,
                "rejected_question": retry_brief.rejected_question,
                "rejected_reverse_question": retry_brief.rejected_reverse_question,
                "retry_attempt": retry_brief.retry_attempt,
            }

        system_prompt = (
            "You are an edge-local bridge generator. "
            "You receive a source entity context from one doc and neighbor contexts from other docs. "
            "Propose only valid two-hop bridge candidates that chain source-side fact -> neighbor-side fact "
            "across at least two documents.\n\n"
            f"Prompt version: {WORKER_PROMPT_VERSION}\n\n"
            f"{WORKER_PRE_OUTPUT_CHECKLIST}\n"
            f"{WORKER_LEGACY_EXAMPLES}\n"
            "If retry_brief is present, stay on the SAME edge, address that verifier failure directly, "
            "and do not restate the rejected mapping.\n"
            "Return JSON only with keys: status, candidates, need_recursion, notes."
        )

        guidance_block = _format_curriculum_guidance_block(packet.constraints.get("curriculum_guidance"))

        user_prompt = (
            "Generate high-quality bridge candidates for this edge packet.\n"
            "Each candidate MUST include keys: question, answers, reverse_question, "
            "reverse_answers, source_docs, reasoning.\n"
            "reasoning MUST include: Forward target variable = X; Reverse target variable = Y; "
            "Sub-Q1 and Sub-Q2 dependency chain.\n"
            "answers and reverse_answers MUST use doc_x::ey refs from contexts.\n"
            "If candidates are non-empty, notes MUST briefly explain why reverse is a true inversion.\n"
            "Do not emit fixed-size filler lists; emit only defensible candidates from evidence.\n"
            "If retry_brief is present, keep the same relationship, avoid the rejected question pair, "
            "and use a different mapping, reverse, or path proof when requested.\n"
            "If no good candidate, return empty candidates and need_recursion boolean.\n"
            + (f"\n{guidance_block}\n" if guidance_block else "\n")
            + f"\nEDGE_PACKET:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_render_messages(
        self,
        render_input: RenderInput,
        retry_brief: Optional[EdgeRetryBrief] = None,
    ) -> List[Dict[str, str]]:
        payload = {
            "edge": render_input.edge.as_dict(),
            "path_proofs": [
                {
                    "source_fact": proof.source_fact,
                    "neighbor_fact": proof.neighbor_fact,
                    "path_docs": proof.path_docs,
                    "target_refs": proof.target_refs,
                    "source_subject_refs": proof.source_subject_refs,
                }
                for proof in render_input.proofs
            ],
            "constraints": render_input.constraints,
        }
        if retry_brief is not None:
            payload["retry_brief"] = {
                "failure_codes": list(retry_brief.failure_codes),
                "retry_hint": retry_brief.retry_hint,
                "retry_reason": retry_brief.retry_reason,
                "rejected_question": retry_brief.rejected_question,
                "rejected_reverse_question": retry_brief.rejected_reverse_question,
                "retry_attempt": retry_brief.retry_attempt,
            }

        system_prompt = (
            "You are a bridge QA renderer. "
            "Use ONLY the provided path_proofs to render two-hop bridge candidates. "
            "Do NOT invent facts or refs.\n\n"
            f"Prompt version: {WORKER_PROMPT_VERSION}\n\n"
            f"{WORKER_PRE_OUTPUT_CHECKLIST}\n"
            f"{WORKER_LEGACY_EXAMPLES}\n"
            "Render mode grounding rule: every answer ref and reverse answer ref must come from path_proofs. "
            "If a fact is missing from path_proofs, do not output that candidate. "
            "If retry_brief is present, stay on the SAME edge and fix that verifier failure without reusing the rejected mapping. "
            "Return JSON only with keys: status, candidates, need_recursion, notes."
        )
        guidance_block = _format_curriculum_guidance_block(render_input.constraints.get("curriculum_guidance"))

        user_prompt = (
            "Render bridge candidates from these path proofs.\n"
            "Each candidate MUST include keys: question, answers, reverse_question, "
            "reverse_answers, source_docs, reasoning.\n"
            "reasoning MUST include: Forward target variable = X; Reverse target variable = Y; "
            "Sub-Q1 and Sub-Q2 dependency chain.\n"
            "answers and reverse_answers MUST be refs appearing in path_proofs target_refs, "
            "source_subject_refs, or source facts.\n"
            "If candidates are non-empty, notes MUST briefly explain why reverse is a true inversion.\n"
            "Do not emit fixed-size filler lists; emit only defensible candidates from evidence.\n"
            "If retry_brief is present, use it to choose a different inversion, target, or path proof on the SAME edge.\n"
            "If no good candidate, return empty candidates.\n"
            + (f"\n{guidance_block}\n" if guidance_block else "\n")
            + f"\nRENDER_INPUT:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _is_non_empty_str(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _normalize_candidates(self, raw_candidates: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_candidates, list):
            return []

        normalized: List[Dict[str, Any]] = []
        required = ["question", "answers", "reverse_question", "reverse_answers", "source_docs", "reasoning"]

        for item in raw_candidates:
            if not isinstance(item, dict):
                continue

            missing = [k for k in required if k not in item]
            if missing:
                continue

            if not self._is_non_empty_str(item.get("question")):
                continue
            if not self._is_non_empty_str(item.get("reverse_question")):
                continue
            if not self._is_non_empty_str(item.get("reasoning")):
                continue

            answers = item.get("answers")
            reverse_answers = item.get("reverse_answers")
            source_docs = item.get("source_docs")
            if not isinstance(answers, list) or not answers:
                continue
            if not isinstance(reverse_answers, list) or not reverse_answers:
                continue
            if not isinstance(source_docs, list) or len(source_docs) < 2:
                continue

            normalized.append(
                {
                    "question": item["question"].strip(),
                    "answers": [str(v).strip() for v in answers if str(v).strip()],
                    "reverse_question": item["reverse_question"].strip(),
                    "reverse_answers": [str(v).strip() for v in reverse_answers if str(v).strip()],
                    "source_docs": [str(v).strip() for v in source_docs if str(v).strip()],
                    "reasoning": item["reasoning"].strip(),
                    "confidence": float(item.get("confidence", 0.9)) if isinstance(item.get("confidence", 0.9), (int, float)) else 0.9,
                    "entities_involved": item.get("entities_involved", []),
                }
            )

        return normalized

    def _parse_worker_output(
        self,
        raw_text: str,
        parse_stage: str,
        token_usage: int = 0,
        budgeted_token_usage: int = 0,
        parse_attempts: int = 1,
        parse_recovery_used: bool = False,
    ) -> WorkerOutput:
        parsed = self.reconciler._extract_json_from_text(raw_text)
        validated = WorkerResponseSchema.model_validate(parsed)
        normalized_payload = validated.model_dump()
        status = str(normalized_payload.get("status", "ok"))
        candidates = self._normalize_candidates(normalized_payload.get("candidates", []))
        need_recursion = bool(normalized_payload.get("need_recursion", False))
        notes = str(normalized_payload.get("notes", ""))
        preview = self.reconciler._preview_text(raw_text)
        return WorkerOutput(
            status=status,
            candidates=candidates,
            need_recursion=need_recursion,
            notes=notes,
            parse_stage=parse_stage,
            raw_preview=preview,
            token_usage=max(0, int(token_usage)),
            budgeted_token_usage=max(0, int(budgeted_token_usage)),
            parse_attempts=max(1, int(parse_attempts)),
            parse_recovery_used=bool(parse_recovery_used),
        )

    def generate(
        self,
        packet: EdgePacket,
        depth: int,
        include_optional: bool,
        max_response_tokens: int,
        render_input: Optional[RenderInput] = None,
        retry_brief: Optional[EdgeRetryBrief] = None,
    ) -> WorkerOutput:
        def _metric_inc(metric_key: str, amount: int = 1) -> None:
            metrics = getattr(self.reconciler, "rlm_metrics", None)
            if not isinstance(metrics, dict):
                return
            lock = getattr(self.reconciler, "_usage_lock", None)
            if lock:
                with lock:
                    base = metrics.get(metric_key, 0)
                    try:
                        base_int = int(base)
                    except (TypeError, ValueError):
                        base_int = 0
                    metrics[metric_key] = base_int + int(amount)
                return
            base = metrics.get(metric_key, 0)
            try:
                base_int = int(base)
            except (TypeError, ValueError):
                base_int = 0
            metrics[metric_key] = base_int + int(amount)

        def _call_worker_model(worker_messages: List[Dict[str, str]], temperature: float, max_tokens: int):
            try:
                response = self.reconciler._call_model_for_stage(
                    messages=worker_messages,
                    model_name=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stage="worker",
                    response_format=self._worker_response_format(),
                    return_usage=True,
                )
            except TypeError:
                # Backward compatibility for test doubles or older wrappers.
                message_only = self.reconciler._call_model_for_stage(
                    messages=worker_messages,
                    model_name=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stage="worker",
                    response_format=self._worker_response_format(),
                )
                return message_only, {"total_tokens": 0}

            if isinstance(response, tuple) and len(response) == 2:
                message_obj, usage_obj = response
                return message_obj, usage_obj or {"total_tokens": 0}
            return response, {"total_tokens": 0}

        messages = (
            self._build_render_messages(render_input, retry_brief=retry_brief)
            if render_input is not None
            else self._build_messages(packet, depth, include_optional, retry_brief=retry_brief)
        )

        total_token_usage = 0
        budgeted_token_usage = 0
        message, usage = _call_worker_model(
            worker_messages=messages,
            temperature=0.0,
            max_tokens=max(256, int(max_response_tokens)),
        )
        try:
            initial_tokens = int((usage or {}).get("total_tokens", 0) or 0)
        except (TypeError, ValueError):
            initial_tokens = 0
        total_token_usage += initial_tokens
        budgeted_token_usage += initial_tokens
        raw_text, _source = self.reconciler._extract_message_content_with_source(message)

        # Extract thinking content from reasoning models
        thinking_text = ""
        for attr in ("reasoning", "reasoning_content"):
            val = getattr(message, attr, None)
            if val:
                thinking_text = self.reconciler._coerce_to_text(val).strip()
                break
        if not thinking_text and raw_text and "<think>" in raw_text:
            match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
            if match:
                thinking_text = match.group(1).strip()
        if thinking_text:
            logger.info(
                "Worker thinking [edge=%s]: %s",
                packet.edge.as_tuple(),
                self.reconciler._preview_text(thinking_text, max_len=300),
            )
            callback = getattr(self.reconciler, "output_callback", None)
            if callback:
                callback("rlm_worker_thinking", {
                    "edge": packet.edge.as_dict(),
                    "thinking": thinking_text,
                    "depth": depth,
                })

        # Strip <think> tags before JSON parsing
        if raw_text and "<think>" in raw_text:
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            raw_text = re.sub(r"<think>.*", "", raw_text, flags=re.DOTALL).strip()

        try:
            return self._parse_worker_output(
                raw_text,
                parse_stage="initial",
                token_usage=total_token_usage,
                budgeted_token_usage=budgeted_token_usage,
                parse_attempts=1,
                parse_recovery_used=False,
            )
        except Exception as initial_error:
            logger.warning(
                "Worker parse failed [stage=initial edge=%s] error=%s preview=%s",
                packet.edge.as_tuple(),
                initial_error,
                self.reconciler._preview_text(raw_text),
            )

        repair_messages = messages + [
            {"role": "assistant", "content": raw_text or "[empty_response]"},
            {
                "role": "user",
                "content": (
                    "Re-output ONLY one valid JSON object with keys: "
                    "status (string), candidates (array), need_recursion (bool), notes (string). "
                    "No markdown."
                ),
            },
        ]

        repaired_message, repair_usage = _call_worker_model(
            worker_messages=repair_messages,
            temperature=0.0,
            max_tokens=max(256, int(max_response_tokens)),
        )
        try:
            total_token_usage += int((repair_usage or {}).get("total_tokens", 0) or 0)
        except (TypeError, ValueError):
            pass
        repaired_text, _repaired_source = self.reconciler._extract_message_content_with_source(repaired_message)

        try:
            repaired_output = self._parse_worker_output(
                repaired_text,
                parse_stage="repair_retry",
                token_usage=total_token_usage,
                budgeted_token_usage=budgeted_token_usage,
                parse_attempts=2,
                parse_recovery_used=True,
            )
            _metric_inc("worker_parse_recovery_success", 1)
            return repaired_output
        except Exception as retry_error:
            logger.warning(
                "Worker parse failed [stage=repair_retry edge=%s] error=%s preview=%s",
                packet.edge.as_tuple(),
                retry_error,
                self.reconciler._preview_text(repaired_text),
            )

        regenerate_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Return ONLY one strict JSON object matching this schema exactly: "
                    '{"status":"<string>","candidates":[],"need_recursion":<bool>,"notes":"<string>"}'
                ),
            }
        ]
        regenerated_text = ""
        try:
            regenerated_message, regen_usage = _call_worker_model(
                worker_messages=regenerate_messages,
                temperature=0.0,
                max_tokens=max(256, int(max_response_tokens)),
            )
            try:
                total_token_usage += int((regen_usage or {}).get("total_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
            regenerated_text, _regenerated_source = self.reconciler._extract_message_content_with_source(regenerated_message)
            regenerated_output = self._parse_worker_output(
                regenerated_text,
                parse_stage="regenerate_retry",
                token_usage=total_token_usage,
                budgeted_token_usage=budgeted_token_usage,
                parse_attempts=3,
                parse_recovery_used=True,
            )
            _metric_inc("worker_parse_recovery_success", 1)
            return regenerated_output
        except Exception as regenerate_error:
            logger.error(
                "Worker parse failed [stage=regenerate_retry edge=%s] error=%s preview=%s",
                packet.edge.as_tuple(),
                regenerate_error,
                self.reconciler._preview_text(regenerated_text),
            )
            _metric_inc("worker_parse_recovery_fail", 1)
            return WorkerOutput(
                status="parse_error",
                candidates=[],
                need_recursion=False,
                notes=f"Worker parse error: {regenerate_error}",
                parse_stage="regenerate_retry",
                raw_preview=self.reconciler._preview_text(regenerated_text),
                token_usage=max(0, int(total_token_usage)),
                budgeted_token_usage=max(0, int(budgeted_token_usage)),
                parse_attempts=3,
                parse_recovery_used=True,
            )


class RootScheduler:
    """Deterministic root scheduler for corpus/doc edge progression."""

    def __init__(
        self,
        reconciler: Any,
        worker: RecursiveEdgeWorker,
        edge_max_depth: int = 1,
        edge_max_calls: int = 2,
        edge_max_tokens: int = 3000,
        max_optional_docs_per_edge: int = 2,
    ):
        self.reconciler = reconciler
        self.worker = worker
        self.edge_max_depth = max(0, int(edge_max_depth))
        self.edge_max_calls = max(1, int(edge_max_calls))
        self.edge_max_tokens = max(1, int(edge_max_tokens))
        self.max_optional_docs_per_edge = max(0, int(max_optional_docs_per_edge))
        self._doc_entity_index_cache: Dict[str, Dict[str, Any]] = {}
        self._pending_retry_briefs: Dict[Tuple[str, str, str, str], EdgeRetryBrief] = {}
        self._guided_retry_counts: Dict[Tuple[str, str, str, str], int] = {}
        self._last_guided_retry_signatures: Dict[Tuple[str, str, str, str], Tuple[Any, ...]] = {}

    def _metrics_lock(self):
        return getattr(self.reconciler, "_usage_lock", None)

    def _metric_value(self, key: str) -> int:
        lock = self._metrics_lock()
        if lock:
            with lock:
                value = self.reconciler.rlm_metrics.get(key, 0)
        else:
            value = self.reconciler.rlm_metrics.get(key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _metric_inc(self, key: str, amount: int = 1) -> None:
        lock = self._metrics_lock()
        if lock:
            with lock:
                base = self.reconciler.rlm_metrics.get(key, 0)
                try:
                    base_int = int(base)
                except (TypeError, ValueError):
                    base_int = 0
                self.reconciler.rlm_metrics[key] = base_int + int(amount)
            return
        self.reconciler.rlm_metrics[key] = self._metric_value(key) + int(amount)

    def _metric_add_float(self, key: str, amount: float) -> None:
        lock = self._metrics_lock()
        if lock:
            with lock:
                base = self.reconciler.rlm_metrics.get(key, 0.0)
                try:
                    base_float = float(base)
                except (TypeError, ValueError):
                    base_float = 0.0
                self.reconciler.rlm_metrics[key] = base_float + float(amount)
            return
        current = self.reconciler.rlm_metrics.get(key, 0.0)
        try:
            current_float = float(current)
        except (TypeError, ValueError):
            current_float = 0.0
        self.reconciler.rlm_metrics[key] = current_float + float(amount)

    def _metric_set_max(self, key: str, candidate: int) -> None:
        lock = self._metrics_lock()
        if lock:
            with lock:
                base = self.reconciler.rlm_metrics.get(key, 0)
                try:
                    base_int = int(base)
                except (TypeError, ValueError):
                    base_int = 0
                if int(candidate) > base_int:
                    self.reconciler.rlm_metrics[key] = int(candidate)
            return

        base = self._metric_value(key)
        if int(candidate) > base:
            self.reconciler.rlm_metrics[key] = int(candidate)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        callback = self.reconciler.output_callback
        if callback:
            callback(event_type, data)

    def _emit_worker_call(
        self,
        edge: EdgeKey,
        call_index: int,
        depth: int,
        include_optional: bool,
        remaining_edge_tokens: int,
        planner_action: str,
        planner_source: str,
        planner_reason: str,
        worker_invoked: bool = True,
        retry_brief: Optional[EdgeRetryBrief] = None,
    ) -> None:
        payload = {
            "edge": edge.as_dict(),
            "call_index": int(call_index),
            "depth": int(depth),
            "include_optional": bool(include_optional),
            "remaining_edge_tokens": int(remaining_edge_tokens),
            "planner_action": str(planner_action),
            "planner_source": str(planner_source),
            "planner_reason": str(planner_reason or ""),
            "worker_invoked": bool(worker_invoked),
        }
        if retry_brief is not None:
            payload["guided_retry"] = True
            payload["retry_hint"] = retry_brief.retry_hint
            payload["retry_attempt"] = int(retry_brief.retry_attempt)
            payload["retry_failure_codes"] = list(retry_brief.failure_codes)
        self._emit_event("rlm_worker_call", payload)

    def _emit_worker_result(
        self,
        edge: EdgeKey,
        call_index: int,
        worker_output: WorkerOutput,
        accepted_delta: int,
        rejected_delta: int,
    ) -> None:
        self._emit_event(
            "rlm_worker_result",
            {
                "edge": edge.as_dict(),
                "call_index": int(call_index),
                "candidates_count": int(len(worker_output.candidates)),
                "need_recursion": bool(worker_output.need_recursion),
                "parse_stage": str(worker_output.parse_stage),
                "final_parse_stage": str(worker_output.parse_stage),
                "parse_attempts": int(worker_output.parse_attempts),
                "parse_recovery_used": bool(worker_output.parse_recovery_used),
                "worker_notes": str(worker_output.notes or ""),
                "accepted_delta": int(accepted_delta),
                "rejected_delta": int(rejected_delta),
            },
        )

    @staticmethod
    def _edge_state_key(edge: EdgeKey) -> Tuple[str, str, str, str]:
        return edge.as_tuple()

    def _peek_pending_retry_brief(self, edge: EdgeKey) -> Optional[EdgeRetryBrief]:
        return self._pending_retry_briefs.get(self._edge_state_key(edge))

    def _pop_pending_retry_brief(self, edge: EdgeKey) -> Optional[EdgeRetryBrief]:
        return self._pending_retry_briefs.pop(self._edge_state_key(edge), None)

    def _set_pending_retry_brief(self, brief: EdgeRetryBrief) -> None:
        self._pending_retry_briefs[self._edge_state_key(brief.edge)] = brief

    def _guided_retry_count(self, edge: EdgeKey) -> int:
        return int(self._guided_retry_counts.get(self._edge_state_key(edge), 0))

    def _set_guided_retry_count(self, edge: EdgeKey, count: int) -> None:
        self._guided_retry_counts[self._edge_state_key(edge)] = max(0, int(count))

    def _last_guided_retry_signature(self, edge: EdgeKey) -> Optional[Tuple[Any, ...]]:
        return self._last_guided_retry_signatures.get(self._edge_state_key(edge))

    def _set_last_guided_retry_signature(self, edge: EdgeKey, signature: Tuple[Any, ...]) -> None:
        self._last_guided_retry_signatures[self._edge_state_key(edge)] = signature

    def _clear_guided_retry_state(self, edge: EdgeKey) -> None:
        edge_key = self._edge_state_key(edge)
        self._pending_retry_briefs.pop(edge_key, None)
        self._guided_retry_counts.pop(edge_key, None)
        self._last_guided_retry_signatures.pop(edge_key, None)

    @staticmethod
    def _trim_retry_text(value: Any, max_len: int = 160) -> str:
        text = str(value or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    def _build_retry_brief(
        self,
        edge: EdgeKey,
        candidate: Dict[str, Any],
        verdict: Dict[str, Any],
        retry_attempt: int = 0,
    ) -> EdgeRetryBrief:
        failure_codes = [
            str(code).strip().upper()
            for code in verdict.get("failure_codes", []) or []
            if str(code).strip()
        ]
        return EdgeRetryBrief(
            edge=edge,
            failure_codes=failure_codes,
            retryable=bool(verdict.get("retryable", False)),
            retry_hint=str(verdict.get("retry_hint", "stop_edge") or "stop_edge"),
            retry_reason=self._trim_retry_text(verdict.get("retry_reason") or verdict.get("notes") or "Stop this edge."),
            rejected_question=self._trim_retry_text(candidate.get("question", "")),
            rejected_reverse_question=self._trim_retry_text(candidate.get("reverse_question", "")),
            retry_attempt=max(0, int(retry_attempt)),
        )

    @staticmethod
    def _retry_brief_priority(brief: EdgeRetryBrief) -> Tuple[int, int, int]:
        hint_priority = {
            "stop_edge": 6,
            "use_different_path_proof": 5,
            "change_mapping": 4,
            "fix_reverse_inversion": 3,
            "strengthen_multihop_chain": 2,
            "remove_answer_leak": 1,
        }
        return (
            1 if brief.retry_hint == "stop_edge" else 0,
            1 if brief.retryable else 0,
            hint_priority.get(str(brief.retry_hint or ""), 0),
        )

    def _prefer_retry_brief(
        self,
        current: Optional[EdgeRetryBrief],
        candidate: Optional[EdgeRetryBrief],
    ) -> Optional[EdgeRetryBrief]:
        if candidate is None:
            return current
        if current is None:
            return candidate
        if self._retry_brief_priority(candidate) > self._retry_brief_priority(current):
            return candidate
        return current

    def _guided_retry_change_room(
        self,
        packet: EdgePacket,
        path_proofs: List[PathProof],
        include_optional: bool,
        can_recurse: bool,
        brief: EdgeRetryBrief,
    ) -> bool:
        if not path_proofs:
            return False
        if brief.retry_hint in {"fix_reverse_inversion", "remove_answer_leak"}:
            return True
        if len(path_proofs) > 1:
            return True
        return (not include_optional and bool(packet.optional_contexts) and can_recurse)

    def _guided_retry_next_state(
        self,
        run_depth: int,
        run_include_optional: bool,
        brief: EdgeRetryBrief,
        packet: EdgePacket,
    ) -> Tuple[int, bool]:
        next_depth = int(run_depth)
        next_include_optional = bool(run_include_optional)
        if (
            brief.retry_hint in {"use_different_path_proof", "change_mapping", "strengthen_multihop_chain"}
            and not next_include_optional
            and bool(packet.optional_contexts)
            and run_depth < self.edge_max_depth
        ):
            next_include_optional = True
            next_depth = min(self.edge_max_depth, max(run_depth + 1, 1))
        return next_depth, next_include_optional

    def _guided_retry_signature(
        self,
        run_signature: Tuple[Any, ...],
        candidate_signature: Tuple[Tuple[Any, ...], ...],
        brief: EdgeRetryBrief,
    ) -> Tuple[Any, ...]:
        return (
            run_signature,
            candidate_signature,
            brief.signature(),
        )

    def _emit_guided_retry_event(
        self,
        event_type: str,
        brief: EdgeRetryBrief,
        **extra: Any,
    ) -> None:
        payload = brief.as_dict()
        payload.update(extra)
        self._emit_event(event_type, payload)

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any], doc_id: Optional[str] = None) -> Any:
        self._emit_event("tool_call", {"tool": tool_name, "arguments": arguments})
        previous_doc = self.reconciler._current_doc_exploration_doc_id
        if doc_id:
            self.reconciler._current_doc_exploration_doc_id = doc_id
        try:
            result = self.reconciler._execute_tool(tool_name, arguments)
        finally:
            self.reconciler._current_doc_exploration_doc_id = previous_doc
        self._emit_event("tool_result", {"tool": tool_name, "result": result, "is_error": isinstance(result, dict) and "error" in result})
        return result

    def _iter_pending_edges(self, doc_id: str) -> List[EdgeKey]:
        plan = self.reconciler.tools.doc_exploration_plans.get(doc_id)
        if not plan:
            return []
        scored_edges: List[Tuple[int, str, str, str, EdgeKey]] = []
        for entity_plan in plan.get("entities_to_explore", []):
            for neighbor in entity_plan.get("neighbors", []):
                if neighbor.get("status") != "pending":
                    continue
                edge = EdgeKey(
                    doc_id=doc_id,
                    entity_name=entity_plan["name"],
                    neighbor_name=neighbor["entity"],
                    relationship=neighbor["relationship"],
                )
                other_docs_count = len(neighbor.get("other_docs", []) or [])
                scored_edges.append(
                    (
                        -other_docs_count,
                        str(entity_plan.get("name", "")).lower(),
                        str(neighbor.get("entity", "")).lower(),
                        str(neighbor.get("relationship", "")).lower(),
                        edge,
                    )
                )
        scored_edges.sort()
        return [row[-1] for row in scored_edges]

    @staticmethod
    def _compact_context_payload(context: Dict[str, Any]) -> Dict[str, Any]:
        qa_pairs = context.get("qa_pairs", [])
        compact_qas = []
        for qa in qa_pairs[:12]:
            compact_qas.append(
                {
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "answer_refs": qa.get("answer_refs", []),
                }
            )
        return {
            "doc_id": context.get("doc_id"),
            "entity": context.get("entity"),
            "qa_pairs": compact_qas,
            "relationships": context.get("relationships", {}),
            "roles": context.get("roles", []),
            "states": context.get("states", []),
        }

    @staticmethod
    def _candidate_signature(candidates: List[Dict[str, Any]]) -> Tuple[Tuple[Any, ...], ...]:
        normalized_rows: List[Tuple[Any, ...]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            question = normalize_similarity_text(candidate.get("question", ""))
            reverse_question = normalize_similarity_text(candidate.get("reverse_question", ""))
            answers = tuple(sorted(normalize_similarity_text(v) for v in candidate.get("answers", []) if str(v).strip()))
            reverse_answers = tuple(sorted(normalize_similarity_text(v) for v in candidate.get("reverse_answers", []) if str(v).strip()))
            source_docs = tuple(sorted(str(v).strip() for v in candidate.get("source_docs", []) if str(v).strip()))
            normalized_rows.append((question, reverse_question, answers, reverse_answers, source_docs))
        return tuple(sorted(normalized_rows))

    @staticmethod
    def _exact_candidate_signature(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
        """Exact in-response dedupe signature for sanitized candidates."""
        question = str(candidate.get("question", "")).strip()
        reverse_question = str(candidate.get("reverse_question", "")).strip()
        answers = tuple(str(v).strip() for v in candidate.get("answers", []) if str(v).strip())
        reverse_answers = tuple(str(v).strip() for v in candidate.get("reverse_answers", []) if str(v).strip())
        source_docs = tuple(str(v).strip() for v in candidate.get("source_docs", []) if str(v).strip())
        reasoning = str(candidate.get("reasoning", "")).strip()
        return (question, reverse_question, answers, reverse_answers, source_docs, reasoning)

    @staticmethod
    def _safe_qa_count(contexts: List[Dict[str, Any]]) -> int:
        total = 0
        for ctx in contexts:
            if isinstance(ctx, dict):
                total += len(ctx.get("qa_pairs", []) or [])
        return total

    def _edge_cross_doc_fact_count(
        self,
        packet: EdgePacket,
        include_optional: bool,
    ) -> int:
        contexts = list(packet.mandatory_contexts)
        if include_optional:
            contexts.extend(packet.optional_contexts)
        count = 0
        for ctx in contexts:
            if not isinstance(ctx, dict):
                continue
            for qa in ctx.get("qa_pairs", []) or []:
                if not isinstance(qa, dict):
                    continue
                refs = self._extract_valid_refs(qa)
                if not refs:
                    continue
                docs = self._refs_to_docs(refs)
                if any(doc and doc != packet.edge.doc_id for doc in docs):
                    count += 1
        return count

    @staticmethod
    def _non_empty_context_count(contexts: List[Dict[str, Any]]) -> int:
        count = 0
        for ctx in contexts:
            if isinstance(ctx, dict) and (ctx.get("qa_pairs", []) or []):
                count += 1
        return count

    def _evaluate_edge_viability(
        self,
        packet: EdgePacket,
        include_optional: bool,
    ) -> Dict[str, Any]:
        path_proofs = self._mine_path_proofs(packet, include_optional=include_optional)
        mandatory_context_count = self._non_empty_context_count(packet.mandatory_contexts)
        optional_context_count = self._non_empty_context_count(packet.optional_contexts)
        cross_doc_fact_count = self._edge_cross_doc_fact_count(packet, include_optional=include_optional)
        evidence_signature = (
            bool(include_optional),
            int(len(path_proofs)),
            int(cross_doc_fact_count),
            int(mandatory_context_count),
            int(optional_context_count),
        )
        return {
            "path_proofs": path_proofs,
            "path_proof_count": len(path_proofs),
            "mandatory_context_count": mandatory_context_count,
            "optional_context_count": optional_context_count,
            "cross_doc_fact_count": cross_doc_fact_count,
            "evidence_signature": evidence_signature,
        }

    def _score_edge_signal(
        self,
        coverage: Dict[str, Any],
        mandatory_contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        mandatory_docs_count = len(coverage.get("mandatory_docs", []) or [])
        optional_fuzzy_docs = coverage.get("optional_fuzzy_docs", []) or []
        best_fuzzy_score = 0.0
        if optional_fuzzy_docs and isinstance(optional_fuzzy_docs[0], dict):
            best_fuzzy_score = float(optional_fuzzy_docs[0].get("score", 0.0))

        qa_count = self._safe_qa_count(mandatory_contexts)
        signal_score = 0.0
        signal_score += min(2.0, mandatory_docs_count * 0.8)
        signal_score += min(2.0, qa_count * 0.15)
        signal_score += min(1.0, max(0.0, best_fuzzy_score / 2.0))

        if signal_score >= 2.0:
            tier = "high"
        elif signal_score >= 1.0:
            tier = "medium"
        else:
            tier = "low"

        probe_only = tier == "low" and mandatory_docs_count == 0 and qa_count == 0
        return {
            "tier": tier,
            "score": round(signal_score, 4),
            "mandatory_docs_count": mandatory_docs_count,
            "mandatory_qa_count": qa_count,
            "best_fuzzy_score": round(best_fuzzy_score, 4),
            "probe_only": probe_only,
        }

    def _low_signal_budgets(
        self,
        signal: Dict[str, Any],
    ) -> Tuple[int, int]:
        tier = signal.get("tier")
        if tier == "high":
            return self.edge_max_calls, self.edge_max_tokens
        if tier == "medium":
            return min(self.edge_max_calls, 3), min(self.edge_max_tokens, 7000)
        # low signal: one cheap probe unless stronger evidence appears later.
        return min(self.edge_max_calls, 1), min(self.edge_max_tokens, 2500)

    @staticmethod
    def _qa_mentions_entity_name(qa: Dict[str, Any], entity_name: str) -> bool:
        target = normalize_similarity_text(entity_name)
        if not target:
            return False
        question = normalize_similarity_text(qa.get("question", ""))
        answer = normalize_similarity_text(qa.get("answer", ""))
        return target in question or target in answer

    @staticmethod
    def _extract_valid_refs(qa: Dict[str, Any]) -> List[str]:
        refs = []
        for ref in qa.get("answer_refs", []) or []:
            if isinstance(ref, str) and "::" in ref:
                refs.append(ref.strip())
        return refs

    @staticmethod
    def _refs_to_docs(refs: List[str]) -> Set[str]:
        docs = set()
        for ref in refs:
            if isinstance(ref, str) and "::" in ref:
                docs.add(ref.split("::", 1)[0].strip())
        return docs

    @staticmethod
    def _new_drop_reason_counts() -> Dict[str, int]:
        return {code: 0 for code in DROP_REASON_CODES}

    @staticmethod
    def _merge_drop_reason_counts(target: Dict[str, int], reasons: Set[str]) -> None:
        for code in reasons:
            if code not in DROP_REASON_CODES:
                continue
            target[code] = int(target.get(code, 0)) + 1

    @staticmethod
    def _source_link_orientation(source_qa: Dict[str, Any], neighbor_name: str) -> str:
        neighbor_norm = normalize_similarity_text(neighbor_name)
        question_norm = normalize_similarity_text(source_qa.get("question", ""))
        answer_norm = normalize_similarity_text(source_qa.get("answer", ""))
        if neighbor_norm and answer_norm == neighbor_norm:
            return "neighbor_as_answer"
        if neighbor_norm and neighbor_norm in question_norm:
            return "neighbor_in_question"
        return "other"

    def _resolve_source_subject_refs(
        self,
        edge: EdgeKey,
        source_qa: Dict[str, Any],
        source_answer_refs: List[str],
    ) -> List[str]:
        source_doc = str(source_qa.get("doc_id", edge.doc_id) or edge.doc_id).strip() or edge.doc_id
        normalized_source_entity = self._normalize_entity_text(edge.entity_name)
        refs: List[str] = []

        if normalized_source_entity:
            doc_index = self._get_doc_entity_index(source_doc)
            for entity_id in sorted(doc_index["norm_to_ids"].get(normalized_source_entity, set())):
                if not self._is_valid_entity_id(source_doc, entity_id):
                    continue
                refs.append(f"{source_doc}::{entity_id}")

        if refs:
            return self._dedupe_preserve_order(refs)

        answer_norm = self._normalize_entity_text(source_qa.get("answer", ""))
        if answer_norm and answer_norm == normalized_source_entity:
            return self._dedupe_preserve_order(source_answer_refs)
        return []

    def _mine_path_proofs(
        self,
        packet: EdgePacket,
        include_optional: bool,
        max_paths: int = 6,
    ) -> List[PathProof]:
        source_context = packet.source_context if isinstance(packet.source_context, dict) else {}
        source_qas = source_context.get("qa_pairs", []) or []
        if not source_qas:
            return []

        source_links = [
            qa for qa in source_qas
            if isinstance(qa, dict) and self._qa_mentions_entity_name(qa, packet.edge.neighbor_name)
        ]
        if not source_links:
            return []

        neighbor_contexts = list(packet.mandatory_contexts)
        if include_optional:
            neighbor_contexts.extend(packet.optional_contexts)

        proofs_by_orientation: Dict[str, List[PathProof]] = {
            "neighbor_as_answer": [],
            "neighbor_in_question": [],
            "other": [],
        }
        seen = set()
        for source_qa in source_links:
            source_refs = self._extract_valid_refs(source_qa)
            source_docs = self._refs_to_docs(source_refs) or {packet.edge.doc_id}
            source_subject_refs = self._resolve_source_subject_refs(packet.edge, source_qa, source_refs)
            orientation = self._source_link_orientation(source_qa, packet.edge.neighbor_name)
            for ctx in neighbor_contexts:
                if not isinstance(ctx, dict):
                    continue
                ctx_doc_id = str(ctx.get("doc_id", "")).strip()
                for neighbor_qa in ctx.get("qa_pairs", []) or []:
                    if not isinstance(neighbor_qa, dict):
                        continue
                    target_refs = self._extract_valid_refs(neighbor_qa)
                    if not target_refs:
                        continue
                    answer_norm = normalize_similarity_text(neighbor_qa.get("answer", ""))
                    if not answer_norm:
                        continue
                    if answer_norm == normalize_similarity_text(packet.edge.neighbor_name):
                        continue
                    if answer_norm == normalize_similarity_text(packet.edge.entity_name):
                        continue

                    path_docs = sorted(
                        source_docs
                        | self._refs_to_docs(target_refs)
                        | ({ctx_doc_id} if ctx_doc_id else set())
                    )
                    if len(path_docs) < 2:
                        continue

                    signature = (
                        orientation,
                        normalize_similarity_text(source_qa.get("question", "")),
                        normalize_similarity_text(neighbor_qa.get("question", "")),
                        tuple(sorted(source_subject_refs)),
                        tuple(sorted(target_refs)),
                        tuple(path_docs),
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)

                    proofs_by_orientation[orientation].append(
                        PathProof(
                            edge=packet.edge,
                            source_fact={
                                "doc_id": source_qa.get("doc_id", packet.edge.doc_id),
                                "question": source_qa.get("question", ""),
                                "answer": source_qa.get("answer", ""),
                                "answer_refs": source_refs,
                            },
                            neighbor_fact={
                                "doc_id": neighbor_qa.get("doc_id", ctx_doc_id),
                                "question": neighbor_qa.get("question", ""),
                                "answer": neighbor_qa.get("answer", ""),
                                "answer_refs": target_refs,
                            },
                            path_docs=path_docs,
                            target_refs=target_refs,
                            source_subject_refs=list(source_subject_refs),
                        )
                    )

        if max_paths <= 0:
            return []

        selected: List[PathProof] = []
        orientation_order = ("neighbor_as_answer", "neighbor_in_question", "other")
        cursors = {key: 0 for key in orientation_order}

        while len(selected) < max_paths:
            added = False
            for key in orientation_order:
                bucket = proofs_by_orientation.get(key, [])
                cursor = cursors[key]
                if cursor >= len(bucket):
                    continue
                selected.append(bucket[cursor])
                cursors[key] = cursor + 1
                added = True
                if len(selected) >= max_paths:
                    break
            if not added:
                break
        return selected

    def _build_edge_packet(
        self,
        edge: EdgeKey,
        coverage: Dict[str, Any],
        source_context: Dict[str, Any],
        mandatory_contexts: List[Dict[str, Any]],
        optional_contexts: List[Dict[str, Any]],
    ) -> EdgePacket:
        compact_source = self._compact_context_payload(source_context) if isinstance(source_context, dict) else {
            "doc_id": edge.doc_id,
            "entity": edge.entity_name,
            "qa_pairs": [],
            "relationships": {},
            "roles": [],
            "states": [],
        }
        compact_mandatory = [self._compact_context_payload(c) for c in mandatory_contexts if isinstance(c, dict)]
        compact_optional = [self._compact_context_payload(c) for c in optional_contexts if isinstance(c, dict)]

        source_docs = sorted({edge.doc_id} | set(coverage.get("mandatory_docs", [])) | {c.get("doc_id") for c in compact_optional if c.get("doc_id")})

        return EdgePacket(
            edge=edge,
            source_docs=source_docs,
            mandatory_docs=list(coverage.get("mandatory_docs", [])),
            optional_docs=[c.get("doc_id") for c in compact_optional if c.get("doc_id")],
            source_context=compact_source,
            mandatory_contexts=compact_mandatory,
            optional_contexts=compact_optional,
            constraints={
                "must_be_chain": True,
                "reverse_required": True,
                "curriculum_guidance": getattr(self.reconciler, "curriculum_guidance", None),
            },
            budget={
                "max_depth": self.edge_max_depth,
                "max_calls": self.edge_max_calls,
                "max_tokens": self.edge_max_tokens,
            },
        )

    def _collect_contexts_for_docs(self, neighbor_name: str, doc_ids: List[str], source_doc: str) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        requested_norm = normalize_similarity_text(neighbor_name)
        for target_doc in doc_ids:
            ctx = self._call_tool(
                "get_entity_context",
                {"entity_name": neighbor_name, "doc_id": target_doc},
                doc_id=source_doc,
            )
            if isinstance(ctx, dict) and "error" not in ctx:
                resolved_entity = normalize_similarity_text(ctx.get("entity", ""))
                if requested_norm and resolved_entity and resolved_entity != requested_norm:
                    self._metric_inc("alias_resolution_hits", 1)
                contexts.append(ctx)
        return contexts

    def _select_optional_doc_ids(self, edge: EdgeKey, optional_fuzzy_docs: List[Dict[str, Any]]) -> List[str]:
        """Use root model to pick optional docs when candidates are large; deterministic fallback otherwise."""
        if not optional_fuzzy_docs:
            return []

        ranked = sorted(
            [d for d in optional_fuzzy_docs if isinstance(d, dict) and d.get("doc_id")],
            key=lambda x: (-float(x.get("score", 0.0)), str(x.get("doc_id"))),
        )
        if len(ranked) <= self.max_optional_docs_per_edge:
            return [str(d["doc_id"]) for d in ranked]

        prompt_payload = {
            "edge": edge.as_dict(),
            "optional_docs": [
                {"doc_id": str(d.get("doc_id")), "name": str(d.get("name", "")), "score": float(d.get("score", 0.0))}
                for d in ranked[:8]
            ],
            "limit": self.max_optional_docs_per_edge,
            "goal": "Pick docs most likely to contain evidence for cross-document second-hop bridge generation.",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Select optional docs for edge expansion. Return only JSON with key "
                    "'selected_doc_ids' (array of doc_id strings)."
                ),
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]

        try:
            message = self.reconciler._call_model_for_stage(
                messages=messages,
                model_name=self.reconciler.root_model,
                max_tokens=256,
                temperature=0.0,
                stage="root",
                response_format=OptionalDocSelection,
            )
            raw_text, _source = self.reconciler._extract_message_content_with_source(message)
            parsed = self.reconciler._extract_json_from_text(raw_text)
            result = OptionalDocSelection.model_validate(parsed)
            selected = result.selected_doc_ids
            if isinstance(selected, list):
                allowed = {str(d["doc_id"]) for d in ranked}
                cleaned = [str(doc).strip() for doc in selected if str(doc).strip() in allowed]
                if cleaned:
                    return cleaned[: self.max_optional_docs_per_edge]
        except Exception as e:
            logger.debug("Root optional-doc selection fallback for edge=%s due to: %s", edge.as_tuple(), e)

        return [str(d["doc_id"]) for d in ranked[: self.max_optional_docs_per_edge]]

    @staticmethod
    def _normalize_entity_text(value: Any) -> str:
        return normalize_similarity_text(value)

    @staticmethod
    def _tokenize_entity_text(text: str) -> Set[str]:
        return tokenize_similarity_text(text)

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen: Set[str] = set()
        output: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output

    def _get_doc_entity_index(self, doc_id: str) -> Dict[str, Any]:
        cached = self._doc_entity_index_cache.get(doc_id)
        if cached is not None:
            return cached

        id_to_name: Dict[str, str] = {}
        id_to_norm: Dict[str, str] = {}
        norm_to_ids: Dict[str, Set[str]] = {}

        gsw = self.reconciler.tools.entity_searcher.gsw_by_doc_id.get(doc_id)
        entity_nodes = getattr(gsw, "entity_nodes", None) if gsw is not None else None

        if entity_nodes:
            for entity in entity_nodes:
                entity_id = str(getattr(entity, "id", "")).strip()
                entity_name = str(getattr(entity, "name", "")).strip()
                if not entity_id:
                    continue
                normalized = self._normalize_entity_text(entity_name)
                id_to_name[entity_id] = entity_name
                id_to_norm[entity_id] = normalized
                if normalized:
                    norm_to_ids.setdefault(normalized, set()).add(entity_id)

        payload = {
            "id_to_name": id_to_name,
            "id_to_norm": id_to_norm,
            "norm_to_ids": norm_to_ids,
        }
        self._doc_entity_index_cache[doc_id] = payload
        return payload

    def _is_valid_entity_id(self, doc_id: str, entity_id: str) -> bool:
        gsw = self.reconciler.tools.entity_searcher.gsw_by_doc_id.get(doc_id)
        if gsw is not None and hasattr(gsw, "get_entity_by_id"):
            try:
                return gsw.get_entity_by_id(entity_id) is not None
            except Exception:
                return False
        # Fallback for lightweight/mocked tests where GSW object has no schema.
        return bool(re.fullmatch(r"e\d+", str(entity_id).strip()))

    def _build_context_answer_ref_map(self, packet: EdgePacket) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        contexts = [packet.source_context] + list(packet.mandatory_contexts) + list(packet.optional_contexts)
        for context in contexts:
            if not isinstance(context, dict):
                continue
            for qa in context.get("qa_pairs", []):
                if not isinstance(qa, dict):
                    continue
                answer = self._normalize_entity_text(qa.get("answer", ""))
                if not answer:
                    continue
                refs = []
                for ref in qa.get("answer_refs", []):
                    if not isinstance(ref, str) or "::" not in ref:
                        continue
                    refs.append(ref.strip())
                if not refs:
                    continue
                existing = mapping.get(answer, [])
                mapping[answer] = self._dedupe_preserve_order(existing + refs)
        return mapping

    def _resolve_entity_for_doc(
        self,
        doc_id: str,
        raw_entity: str,
        context_ref_map: Dict[str, List[str]],
        allowed_refs: Set[str],
    ) -> Optional[str]:
        raw = str(raw_entity or "").strip()
        if not raw:
            return None

        normalized = self._normalize_entity_text(raw)
        if not normalized:
            return None

        context_refs = [ref for ref in context_ref_map.get(normalized, []) if ref.startswith(f"{doc_id}::")]
        if len(context_refs) == 1:
            return context_refs[0] if context_refs[0] in allowed_refs else None

        doc_index = self._get_doc_entity_index(doc_id)
        norm_to_ids = doc_index["norm_to_ids"]
        direct_ids = norm_to_ids.get(normalized, set())
        if len(direct_ids) == 1:
            candidate_ref = f"{doc_id}::{next(iter(direct_ids))}"
            if candidate_ref in allowed_refs:
                return candidate_ref

        query_tokens = self._tokenize_entity_text(normalized)
        if not query_tokens:
            return None

        best_id: Optional[str] = None
        best_score = 0.0
        tied = False
        for entity_id, entity_norm in doc_index["id_to_norm"].items():
            entity_tokens = self._tokenize_entity_text(entity_norm)
            if not entity_tokens:
                continue

            if entity_norm and (entity_norm in normalized or normalized in entity_norm):
                score = 1.0 + (0.01 * len(entity_tokens))
            else:
                overlap = len(query_tokens & entity_tokens)
                if overlap == 0:
                    continue
                # Favors exact/near-exact token cover of the entity name.
                score = overlap / max(1, len(entity_tokens))

            if score < 0.8:
                continue

            if score > best_score:
                best_id = entity_id
                best_score = score
                tied = False
            elif score == best_score and best_id != entity_id:
                tied = True

        if best_id and not tied:
            candidate_ref = f"{doc_id}::{best_id}"
            if candidate_ref in allowed_refs:
                return candidate_ref
        return None

    def _resolve_answer_ref(
        self,
        ref: str,
        source_docs: List[str],
        context_ref_map: Dict[str, List[str]],
        allowed_refs: Set[str],
    ) -> Optional[str]:
        raw = str(ref or "").strip()
        if not raw:
            return None

        if "::" in raw:
            doc_id, entity_part = raw.split("::", 1)
            doc_id = doc_id.strip()
            entity_part = entity_part.strip()
            if doc_id not in source_docs:
                return None
            explicit_ref = f"{doc_id}::{entity_part}"
            if explicit_ref in allowed_refs:
                return explicit_ref
            return self._resolve_entity_for_doc(doc_id, entity_part, context_ref_map, allowed_refs)

        matches = []
        for doc_id in source_docs:
            resolved = self._resolve_entity_for_doc(doc_id, raw, context_ref_map, allowed_refs)
            if resolved:
                matches.append(resolved)

        matches = self._dedupe_preserve_order(matches)
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _collect_path_proof_allowlist(path_proofs: List[PathProof]) -> Set[str]:
        refs: Set[str] = set()
        for proof in path_proofs:
            for ref in proof.target_refs:
                if isinstance(ref, str) and "::" in ref:
                    refs.add(ref.strip())
            for ref in proof.source_subject_refs:
                if isinstance(ref, str) and "::" in ref:
                    refs.add(ref.strip())
            for ref in proof.source_fact.get("answer_refs", []) or []:
                if isinstance(ref, str) and "::" in ref:
                    refs.add(ref.strip())
        return refs

    def _sanitize_candidate_refs_with_reasons(
        self,
        packet: EdgePacket,
        candidate: Dict[str, Any],
        extra_allowed_refs: Optional[Set[str]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Set[str]]:
        raw_source_docs = candidate.get("source_docs", [])
        allowed_docs = [str(d).strip() for d in packet.source_docs if str(d).strip()]
        if not allowed_docs:
            return None, {"missing_target_ref"}

        allowed_docs_set = set(allowed_docs)
        candidate_docs = [str(d).strip() for d in raw_source_docs if str(d).strip() in allowed_docs_set]
        source_docs_for_resolution = self._dedupe_preserve_order(candidate_docs + allowed_docs)
        if len(source_docs_for_resolution) < 2:
            return None, {"no_cross_doc_evidence"}

        context_ref_map = self._build_context_answer_ref_map(packet)
        allowed_refs = {
            ref
            for refs in context_ref_map.values()
            for ref in refs
            if isinstance(ref, str) and "::" in ref
        }
        if extra_allowed_refs:
            allowed_refs.update(
                ref.strip()
                for ref in extra_allowed_refs
                if isinstance(ref, str) and "::" in ref
            )
        if not allowed_refs:
            return None, {"missing_target_ref"}

        resolved_answers: List[str] = []
        unresolved_forward: List[str] = []
        for answer in candidate.get("answers", []):
            resolved = self._resolve_answer_ref(str(answer), source_docs_for_resolution, context_ref_map, allowed_refs)
            if resolved:
                resolved_answers.append(resolved)
            else:
                unresolved_forward.append(str(answer))

        resolved_reverse: List[str] = []
        unresolved_reverse: List[str] = []
        for answer in candidate.get("reverse_answers", []):
            resolved = self._resolve_answer_ref(str(answer), source_docs_for_resolution, context_ref_map, allowed_refs)
            if resolved:
                resolved_reverse.append(resolved)
            else:
                unresolved_reverse.append(str(answer))

        resolved_answers = self._dedupe_preserve_order(resolved_answers)
        resolved_reverse = self._dedupe_preserve_order(resolved_reverse)
        if unresolved_forward or unresolved_reverse or not resolved_answers or not resolved_reverse:
            reasons: Set[str] = set()
            if unresolved_forward or not resolved_answers:
                reasons.add("missing_target_ref")
            if unresolved_reverse or not resolved_reverse:
                reasons.add("missing_source_ref")
            logger.debug(
                "Dropping unresolved worker candidate [edge=%s unresolved_forward=%s unresolved_reverse=%s]",
                packet.edge.as_tuple(),
                unresolved_forward[:4],
                unresolved_reverse[:4],
            )
            return None, reasons

        docs_from_refs = {
            ref.split("::", 1)[0]
            for ref in (resolved_answers + resolved_reverse)
            if "::" in ref
        }
        if len(docs_from_refs) < 2:
            logger.debug(
                "Dropping single-doc worker candidate [edge=%s docs=%s]",
                packet.edge.as_tuple(),
                sorted(docs_from_refs),
            )
            return None, {"no_cross_doc_evidence"}
        canonical_source_docs = self._dedupe_preserve_order(
            [doc for doc in source_docs_for_resolution if doc in docs_from_refs]
        )
        if len(canonical_source_docs) < 2:
            return None, {"no_cross_doc_evidence"}

        sanitized = dict(candidate)
        sanitized["answers"] = resolved_answers
        sanitized["reverse_answers"] = resolved_reverse
        sanitized["source_docs"] = canonical_source_docs
        return sanitized, set()

    def _sanitize_candidate_refs(
        self,
        packet: EdgePacket,
        candidate: Dict[str, Any],
        extra_allowed_refs: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        sanitized, _drop_reasons = self._sanitize_candidate_refs_with_reasons(
            packet,
            candidate,
            extra_allowed_refs=extra_allowed_refs,
        )
        return sanitized

    def _prevalidate_commit_candidates(self, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Run non-mutating validation/canonicalization before verifier calls."""
        prepared: List[Dict[str, Any]] = []
        rejected = 0
        seen_signatures: Set[Tuple[Any, ...]] = set()
        lookup_cache = getattr(self.reconciler, "_lookup_rejected_bridge_cache", None)
        emit_cache_hit = getattr(self.reconciler, "_emit_pre_verifier_cache_hit", None)
        signature_builder = getattr(self.reconciler, "_bridge_signature", None)

        for candidate in candidates:
            prevalidated = self.reconciler.tools.prevalidate_bridge_qa(candidate)
            if not isinstance(prevalidated, dict) or not prevalidated.get("valid"):
                rejected += 1
                continue

            bridge_spec = prevalidated.get("bridge_spec", {})
            if not isinstance(bridge_spec, dict):
                rejected += 1
                continue

            cached = lookup_cache(bridge_spec) if callable(lookup_cache) else None
            if cached is not None:
                if callable(emit_cache_hit):
                    emit_cache_hit(bridge_spec, cached)
                rejected += 1
                continue

            if callable(signature_builder):
                signature = signature_builder(bridge_spec)
            else:
                signature = (
                    normalize_similarity_text(bridge_spec.get("question", "")),
                    normalize_similarity_text(bridge_spec.get("reverse_question", "")),
                    tuple(sorted(str(v).strip().lower() for v in bridge_spec.get("answers", []) if str(v).strip())),
                    tuple(sorted(str(v).strip().lower() for v in bridge_spec.get("reverse_answers", []) if str(v).strip())),
                    tuple(sorted(str(v).strip().lower() for v in bridge_spec.get("source_docs", []) if str(v).strip())),
                )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            prepared.append(bridge_spec)

        return prepared, rejected

    @staticmethod
    def _max_similarity_from_signal(similarity_signal: Any) -> float:
        """Compute the max similarity score from a similarity_signal payload."""
        if not isinstance(similarity_signal, list):
            return 0.0
        max_similarity = 0.0
        for hit in similarity_signal:
            if not isinstance(hit, dict):
                continue
            try:
                score = float(hit.get("max_similarity", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if score > max_similarity:
                max_similarity = score
        return max(0.0, min(1.0, max_similarity))

    def _parallel_verifier_screen(
        self,
        edge: EdgeKey,
        candidates: List[Dict[str, Any]],
        retry_attempt: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, int], Optional[EdgeRetryBrief]]:
        """Run verifier in parallel and keep only passing candidates."""
        if not candidates:
            return [], 0, {}, None

        verifier_enabled = bool(getattr(self.reconciler, "bridge_verifier_enabled", False))
        if not verifier_enabled:
            return list(candidates), 0, {}, None

        max_workers = max(1, int(getattr(self.reconciler, "verifier_parallel_workers", 4)))
        workers_used = min(max_workers, len(candidates))
        self._metric_set_max("verifier_parallel_workers_used", workers_used)
        verify_gate = getattr(self.reconciler, "verify_bridge_candidate_gate", None)
        verify_once = getattr(self.reconciler, "_verify_bridge_candidate", None)
        if not callable(verify_gate) and not callable(verify_once):
            return list(candidates), 0, {}, None

        start = time.perf_counter()
        verdicts: Dict[int, Tuple[Dict[str, Any], bool]] = {}
        with ThreadPoolExecutor(max_workers=workers_used) as pool:
            def _verify(candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
                if callable(verify_gate):
                    return verify_gate(candidate)
                verdict = verify_once(candidate)
                score = float(verdict.get("score", 0.0) or 0.0)
                passes_gate = bool(verdict.get("pass")) and score >= float(getattr(self.reconciler, "bridge_verifier_threshold", 0.75))
                return verdict, passes_gate

            futures = {
                pool.submit(_verify, candidate): idx
                for idx, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    verdicts[idx] = future.result()
                except Exception as exc:
                    verdicts[idx] = (
                        {
                            "pass": False,
                            "score": 0.0,
                            "failure_codes": ["VERIFIER_ERROR"],
                            "notes": f"Parallel verifier error: {exc}",
                        },
                        False,
                    )
        self._metric_add_float("parallel_verifier_seconds", time.perf_counter() - start)

        passing: List[Dict[str, Any]] = []
        rejected = 0
        failure_code_counts: Dict[str, int] = {}
        retry_brief: Optional[EdgeRetryBrief] = None
        for idx, candidate in enumerate(candidates):
            verdict, passes_gate = verdicts.get(
                idx,
                ({"pass": False, "score": 0.0, "failure_codes": ["VERIFIER_MISSING"], "notes": "Missing verifier verdict."}, False),
            )
            if passes_gate:
                passing.append(candidate)
            else:
                retry_brief = self._prefer_retry_brief(
                    retry_brief,
                    self._build_retry_brief(edge, candidate, verdict, retry_attempt=retry_attempt),
                )
                for code in verdict.get("failure_codes", []) or []:
                    if not isinstance(code, str) or not code.strip():
                        continue
                    norm_code = code.strip().upper()
                    failure_code_counts[norm_code] = int(failure_code_counts.get(norm_code, 0)) + 1
                rejected += 1
        return passing, rejected, failure_code_counts, retry_brief

    def _commit_preverified_candidates(
        self,
        edge: EdgeKey,
        candidates: List[Dict[str, Any]],
        retry_attempt: int = 0,
    ) -> Tuple[int, int, bool, Optional[EdgeRetryBrief]]:
        """Create already-verified candidates serially with deterministic ordering."""
        accepted = 0
        rejected = 0
        cap_hit = False
        retry_brief: Optional[EdgeRetryBrief] = None
        accepted_cap = max(1, int(getattr(self.reconciler, "edge_max_accepted_bridges", 10)))
        verifier_enabled = bool(getattr(self.reconciler, "bridge_verifier_enabled", False))
        verify_gate = getattr(self.reconciler, "verify_bridge_candidate_gate", None)

        for candidate in candidates:
            if accepted >= accepted_cap:
                cap_hit = True
                break

            refreshed = self.reconciler.tools.prevalidate_bridge_qa(candidate)
            if not isinstance(refreshed, dict) or not refreshed.get("valid"):
                rejected += 1
                continue

            bridge_spec = refreshed.get("bridge_spec", {})
            if not isinstance(bridge_spec, dict):
                rejected += 1
                continue

            similarity_max = self._max_similarity_from_signal(bridge_spec.get("similarity_signal", []))
            if (
                verifier_enabled
                and callable(verify_gate)
                and similarity_max >= COMMIT_SIMILARITY_RECHECK_THRESHOLD
            ):
                verdict, passes_gate = verify_gate(bridge_spec)
                if not passes_gate:
                    retry_brief = self._prefer_retry_brief(
                        retry_brief,
                        self._build_retry_brief(edge, bridge_spec, verdict, retry_attempt=retry_attempt),
                    )
                    rejected += 1
                    continue

            result = self._call_tool(
                "create_bridge_qa",
                {"bridges": [bridge_spec], "__preverified": True},
                doc_id=edge.doc_id,
            )
            items = result if isinstance(result, list) else [result]
            for item in items:
                if isinstance(item, dict) and item.get("success"):
                    accepted += 1
                else:
                    rejected += 1

        if cap_hit:
            self._metric_inc("edge_accept_cap_hits", 1)
        return accepted, rejected, cap_hit, retry_brief

    def _apply_candidates(
        self,
        packet: EdgePacket,
        candidates: List[Dict[str, Any]],
        path_proofs: Optional[List[PathProof]] = None,
        drop_reason_counts: Optional[Dict[str, int]] = None,
        retry_attempt: int = 0,
    ) -> Tuple[int, int, bool, Optional[EdgeRetryBrief]]:
        accepted = 0
        rejected = 0
        seen_signatures: Set[Tuple[Any, ...]] = set()
        deduped_sanitized: List[Dict[str, Any]] = []
        reasons = drop_reason_counts if drop_reason_counts is not None else self._new_drop_reason_counts()
        proof_allowlist = self._collect_path_proof_allowlist(path_proofs or [])

        for candidate in candidates:
            sanitized, sanitize_reasons = self._sanitize_candidate_refs_with_reasons(
                packet,
                candidate,
                extra_allowed_refs=proof_allowlist,
            )
            if sanitized is None:
                rejected += 1
                self._merge_drop_reason_counts(reasons, sanitize_reasons)
                continue

            signature = self._exact_candidate_signature(sanitized)
            if signature in seen_signatures:
                logger.debug(
                    "Skipping duplicate worker candidate [edge=%s]",
                    packet.edge.as_tuple(),
                )
                continue
            seen_signatures.add(signature)
            deduped_sanitized.append(sanitized)

        prepared, prevalidation_rejected = self._prevalidate_commit_candidates(deduped_sanitized)
        rejected += prevalidation_rejected
        passing, verifier_rejected, verifier_failure_codes, retry_brief = self._parallel_verifier_screen(
            packet.edge,
            prepared,
            retry_attempt=retry_attempt,
        )
        rejected += verifier_rejected
        if verifier_failure_codes.get("REVERSE_INVALID"):
            reasons["reverse_not_invertible"] = int(reasons.get("reverse_not_invertible", 0)) + int(
                verifier_failure_codes["REVERSE_INVALID"]
            )
        accepted, commit_rejected, cap_hit, commit_retry_brief = self._commit_preverified_candidates(
            packet.edge,
            passing,
            retry_attempt=retry_attempt,
        )
        retry_brief = self._prefer_retry_brief(retry_brief, commit_retry_brief)
        rejected += commit_rejected
        if candidates and accepted == 0 and rejected > 0:
            reasons["all_candidates_rejected"] = int(reasons.get("all_candidates_rejected", 0)) + 1
        return accepted, rejected, cap_hit, retry_brief

    def _run_edge(self, edge: EdgeKey) -> EdgeRunResult:
        start_tokens = self.reconciler.tokens_used
        self._clear_guided_retry_state(edge)

        source_context = self._call_tool(
            "get_entity_context",
            {"entity_name": edge.entity_name, "doc_id": edge.doc_id},
        )
        if not isinstance(source_context, dict) or "error" in source_context:
            source_context = {
                "entity": edge.entity_name,
                "doc_id": edge.doc_id,
                "qa_pairs": [],
                "roles": [],
                "states": [],
                "relationships": {},
            }

        focus_result = self._call_tool("begin_neighbor_focus", edge.as_dict(), doc_id=edge.doc_id)
        if isinstance(focus_result, dict) and focus_result.get("error"):
            return EdgeRunResult(edge=edge, accepted=0, attempted=0, rejected=0, budget_exhausted=False, edge_tokens=0, property_attempted=False)

        coverage = self._call_tool("plan_neighbor_doc_coverage", edge.as_dict(), doc_id=edge.doc_id)
        if not isinstance(coverage, dict) or "error" in coverage:
            return EdgeRunResult(edge=edge, accepted=0, attempted=0, rejected=0, budget_exhausted=False, edge_tokens=0, property_attempted=False)

        mandatory_docs = list(coverage.get("mandatory_docs", []))
        optional_docs = self._select_optional_doc_ids(edge, coverage.get("optional_fuzzy_docs", []))

        mandatory_contexts = self._collect_contexts_for_docs(edge.neighbor_name, mandatory_docs, source_doc=edge.doc_id)
        optional_contexts = self._collect_contexts_for_docs(edge.neighbor_name, optional_docs, source_doc=edge.doc_id) if optional_docs else []

        packet = self._build_edge_packet(edge, coverage, source_context, mandatory_contexts, optional_contexts)
        signal = self._score_edge_signal(coverage, mandatory_contexts)
        edge_call_limit, edge_token_limit = self._low_signal_budgets(signal)
        if signal.get("tier") == "low":
            self.reconciler.rlm_metrics["low_signal_edges"] = self.reconciler.rlm_metrics.get("low_signal_edges", 0) + 1

        accepted_total = 0
        attempted_total = 0
        rejected_total = 0
        budget_exhausted = False
        call_count = 0
        no_candidate_calls = 0
        drop_reason_counts = self._new_drop_reason_counts()
        last_failure_signature: Optional[Tuple[Tuple[Any, ...], ...]] = None
        last_failure_state: Optional[Tuple[int, bool, bool, int]] = None
        close_reason = "call_limit_reached"
        close_detail = f"Reached edge call limit ({edge_call_limit})."
        last_worker_output: Optional[WorkerOutput] = None
        last_run_signature: Optional[Tuple[Any, ...]] = None
        active_retry_brief: Optional[EdgeRetryBrief] = None
        guided_retry_count = 0
        last_retry_hint = ""
        last_retry_failure_codes: List[str] = []

        depth = 0
        include_optional = False
        edge_budget_spend = 0

        while call_count < edge_call_limit:
            remaining_edge_tokens = edge_token_limit - edge_budget_spend
            if remaining_edge_tokens <= 0:
                budget_exhausted = True
                close_reason = "budget_exhausted"
                close_detail = "Per-edge token budget exhausted before worker call."
                break

            viability = self._evaluate_edge_viability(packet, include_optional=include_optional)
            path_proofs = viability["path_proofs"]
            render_input = RenderInput(
                edge=edge,
                proofs=path_proofs,
                constraints=packet.constraints,
            ) if path_proofs else None

            # Fail-closed viability gate: skip worker when no path proof can be mined.
            no_path_proofs = viability["path_proof_count"] == 0
            can_optional_probe = (
                call_count == 0
                and not include_optional
                and not packet.mandatory_contexts
                and bool(packet.optional_contexts)
                and depth < self.edge_max_depth
            )
            if no_path_proofs:
                if can_optional_probe:
                    include_optional = True
                    depth = min(self.edge_max_depth, max(depth + 1, 1))
                    self._metric_inc("recursive_invocations", 1)
                    continue

                self._metric_inc("edges_skipped_no_path", 1)
                close_reason = "no_viable_path_proof"
                close_detail = (
                    "No viable path proofs before worker call "
                    f"(mandatory_contexts={viability['mandatory_context_count']}, "
                    f"optional_contexts={viability['optional_context_count']}, "
                    f"cross_doc_facts={viability['cross_doc_fact_count']})."
                )
                break

            run_signature = (
                int(depth),
                bool(include_optional),
                viability["evidence_signature"],
            )
            if call_count > 0 and run_signature == last_run_signature and accepted_total == 0 and active_retry_brief is None:
                self._metric_inc("calls_blocked_no_progress", 1)
                close_reason = "no_progress_repeat"
                close_detail = (
                    "Blocked repeated worker call with unchanged evidence state "
                    f"(depth={depth}, include_optional={include_optional}, "
                    f"path_proofs={viability['path_proof_count']})."
                )
                break
            last_run_signature = run_signature

            call_index = call_count + 1
            current_retry_brief = active_retry_brief
            if current_retry_brief is not None:
                self._metric_inc("guided_retry_started", 1)
                self._emit_guided_retry_event(
                    "guided_retry_started",
                    current_retry_brief,
                    mode="rlm",
                    call_index=call_index,
                    depth=depth,
                    include_optional=include_optional,
                )
            self._emit_worker_call(
                edge=edge,
                call_index=call_index,
                depth=depth,
                include_optional=include_optional,
                remaining_edge_tokens=remaining_edge_tokens,
                planner_action="run_worker",
                planner_source="deterministic",
                planner_reason="root_scheduler_default",
                worker_invoked=True,
                retry_brief=current_retry_brief,
            )
            worker_output = self.worker.generate(
                packet=packet,
                depth=depth,
                include_optional=include_optional,
                max_response_tokens=min(1400, remaining_edge_tokens),
                render_input=render_input,
                retry_brief=current_retry_brief,
            )
            active_retry_brief = None
            call_count = call_index
            last_worker_output = worker_output
            edge_budget_spend += max(0, int(getattr(worker_output, "budgeted_token_usage", 0)))

            attempted_total += len(worker_output.candidates)
            if not worker_output.candidates:
                no_candidate_calls += 1

            accepted, rejected, cap_hit, retry_brief = self._apply_candidates(
                packet,
                worker_output.candidates,
                path_proofs=path_proofs,
                drop_reason_counts=drop_reason_counts,
                retry_attempt=guided_retry_count + 1,
            )
            accepted_total += accepted
            rejected_total += rejected
            self._emit_worker_result(
                edge=edge,
                call_index=call_index,
                worker_output=worker_output,
                accepted_delta=accepted,
                rejected_delta=rejected,
            )

            # Enforce hard edge budget even if the latest model call overshot.
            if edge_budget_spend >= edge_token_limit:
                budget_exhausted = True
                close_reason = "budget_exhausted"
                close_detail = "Per-edge token budget exhausted after worker call."
                break

            if accepted_total > 0:
                if current_retry_brief is not None:
                    self._metric_inc("guided_retry_succeeded", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_succeeded",
                        current_retry_brief,
                        mode="rlm",
                        accepted_total=accepted_total,
                    )
                close_reason = "accepted_bridge"
                if cap_hit:
                    close_detail = (
                        f"Accepted {accepted_total} bridge(s); hit per-edge acceptance cap "
                        f"({getattr(self.reconciler, 'edge_max_accepted_bridges', 10)})."
                    )
                else:
                    close_detail = f"Accepted {accepted_total} bridge(s)."
                break

            can_recurse = depth < self.edge_max_depth
            state_key = (depth, include_optional, bool(render_input), len(path_proofs))
            candidate_signature = self._candidate_signature(worker_output.candidates)
            repeated_failure = (
                accepted == 0
                and bool(worker_output.candidates)
                and last_failure_state == state_key
                and last_failure_signature == candidate_signature
            )
            if accepted == 0 and worker_output.candidates:
                last_failure_state = state_key
                last_failure_signature = candidate_signature
            elif accepted > 0:
                last_failure_state = None
                last_failure_signature = None

            if accepted == 0 and retry_brief is not None:
                last_retry_hint = retry_brief.retry_hint
                last_retry_failure_codes = list(retry_brief.failure_codes)
                retry_signature = self._guided_retry_signature(run_signature, candidate_signature, retry_brief)
                if retry_brief.retry_hint == "stop_edge" or not retry_brief.retryable:
                    close_reason = "verifier_stop_edge"
                    close_detail = retry_brief.retry_reason or "Verifier indicated this edge should stop."
                    break
                if guided_retry_count >= GUIDED_RETRY_LIMIT:
                    self._metric_inc("guided_retry_exhausted", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_exhausted",
                        retry_brief,
                        mode="rlm",
                        guided_retry_count=guided_retry_count,
                        reason="retry_limit",
                    )
                    close_reason = "guided_retry_exhausted"
                    close_detail = f"Reached guided retry limit ({GUIDED_RETRY_LIMIT}) for this edge."
                    break
                if not self._guided_retry_change_room(packet, path_proofs, include_optional, can_recurse, retry_brief):
                    self._metric_inc("guided_retry_exhausted", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_exhausted",
                        retry_brief,
                        mode="rlm",
                        guided_retry_count=guided_retry_count,
                        reason="no_alternate_evidence",
                    )
                    close_reason = "guided_retry_exhausted"
                    close_detail = "Verifier suggested retry, but no alternate same-edge evidence remains."
                    break
                if self._last_guided_retry_signature(edge) == retry_signature:
                    self._metric_inc("guided_retry_blocked_no_progress", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_blocked_no_progress",
                        retry_brief,
                        mode="rlm",
                        guided_retry_count=guided_retry_count,
                    )
                    close_reason = "guided_retry_blocked_no_progress"
                    close_detail = "Blocked guided retry with unchanged verifier hint and evidence state."
                    break

                guided_retry_count += 1
                self._set_guided_retry_count(edge, guided_retry_count)
                self._set_last_guided_retry_signature(edge, retry_signature)
                active_retry_brief = EdgeRetryBrief(
                    edge=retry_brief.edge,
                    failure_codes=list(retry_brief.failure_codes),
                    retryable=retry_brief.retryable,
                    retry_hint=retry_brief.retry_hint,
                    retry_reason=retry_brief.retry_reason,
                    rejected_question=retry_brief.rejected_question,
                    rejected_reverse_question=retry_brief.rejected_reverse_question,
                    retry_attempt=guided_retry_count,
                )
                self._metric_inc("guided_retry_scheduled", 1)
                self._emit_guided_retry_event(
                    "guided_retry_scheduled",
                    active_retry_brief,
                    mode="rlm",
                    guided_retry_count=guided_retry_count,
                )
                depth, include_optional = self._guided_retry_next_state(depth, include_optional, active_retry_brief, packet)
                continue

            if can_recurse and not include_optional and packet.optional_contexts:
                include_optional = True
                depth += 1
                self._metric_inc("recursive_invocations", 1)
                continue

            if can_recurse and worker_output.need_recursion:
                depth += 1
                include_optional = True
                self._metric_inc("recursive_invocations", 1)
                continue

            if repeated_failure:
                self._metric_inc("repeated_failure_stops", 1)
                close_reason = "repeated_failure"
                close_detail = "Repeated candidate signature failure at same depth/state."
                break

            if not worker_output.candidates:
                # No candidates and no recursion path left.
                close_reason = "no_candidates_no_progression"
                close_detail = (
                    f"No candidates (notes='{worker_output.notes}', parse_stage={worker_output.parse_stage}, "
                    f"parse_attempts={worker_output.parse_attempts}, "
                    f"accepted_delta={accepted}, rejected_delta={rejected})."
                )
                break

            if accepted == 0:
                if signal.get("probe_only"):
                    close_reason = "probe_only_no_success"
                    close_detail = "Probe-only edge produced no accepted candidates."
                    break
                # Retry only when the candidate signature changed at the same depth/state.
                continue

            # Accepted candidates exist; continue only if loop conditions require it.
            continue

        if close_reason == "call_limit_reached":
            tail = ""
            if last_worker_output is not None:
                tail = (
                    f" last_parse_stage={last_worker_output.parse_stage};"
                    f" last_parse_attempts={last_worker_output.parse_attempts};"
                    f" last_notes={last_worker_output.notes!r}"
                )
            close_detail = (
                f"Reached edge call limit ({edge_call_limit}); accepted_total={accepted_total}; "
                f"rejected_total={rejected_total}; no_candidate_calls={no_candidate_calls};{tail}"
            )

        mark_args = {
            "doc_id": edge.doc_id,
            "entity_name": edge.entity_name,
            "neighbor_name": edge.neighbor_name,
            "relationship": edge.relationship,
            "bridges_created": accepted_total,
        }
        _mark_result = self._call_tool("mark_neighbor_explored", mark_args, doc_id=edge.doc_id)

        edge_tokens = self.reconciler.tokens_used - start_tokens
        self.reconciler.rlm_metrics["edges_explored"] += 1
        if accepted_total > 0:
            self.reconciler.rlm_metrics["edges_with_bridges"] += 1

        edge_result = EdgeRunResult(
            edge=edge,
            accepted=accepted_total,
            attempted=attempted_total,
            rejected=rejected_total,
            budget_exhausted=budget_exhausted,
            edge_tokens=max(0, edge_tokens),
            property_attempted=False,
            guided_retry_count=guided_retry_count,
            last_retry_hint=last_retry_hint,
            last_retry_failure_codes=list(last_retry_failure_codes),
        )
        self._clear_guided_retry_state(edge)

        self._emit_event(
            "rlm_edge_summary",
            {
                "edge": edge.as_dict(),
                "accepted": edge_result.accepted,
                "attempted": edge_result.attempted,
                "rejected": edge_result.rejected,
                "budget_exhausted": edge_result.budget_exhausted,
                "edge_tokens": edge_result.edge_tokens,
                "property_attempted": edge_result.property_attempted,
                "signal_tier": signal.get("tier"),
                "signal_score": signal.get("score"),
                "no_candidate_calls": no_candidate_calls,
                "call_limit": edge_call_limit,
                "token_limit": edge_token_limit,
                "worker_prompt_version": WORKER_PROMPT_VERSION,
                "worker_calls_made": call_count,
                "close_reason": close_reason,
                "close_detail": close_detail if close_detail else (
                    str(last_worker_output.notes) if last_worker_output is not None and last_worker_output.notes else ""
                ),
                "drop_reason_counts": drop_reason_counts,
                "guided_retry_count": guided_retry_count,
                "last_retry_hint": last_retry_hint,
                "last_retry_failure_codes": list(last_retry_failure_codes),
            },
        )

        return edge_result

    def run_document(self, doc_id: str) -> Dict[str, Any]:
        plan = self._call_tool("plan_document_exploration", {"doc_id": doc_id}, doc_id=doc_id)
        if isinstance(plan, dict) and "error" in plan:
            return {
                "entity": doc_id,
                "iterations": 0,
                "tool_calls": 1,
                "bridges_created": 0,
                "mode": "rlm",
                "error": plan.get("error"),
            }

        self.reconciler.rlm_metrics["docs_attempted"] += 1

        edges = self._iter_pending_edges(doc_id)
        edge_results: List[EdgeRunResult] = []
        max_edge_attempts = max(10, len(edges) * 3)
        edge_attempts = 0
        while edges and edge_attempts < max_edge_attempts:
            edge_attempts += 1
            edge_results.append(self._run_edge(edges[0]))
            edges = self._iter_pending_edges(doc_id)
        if edges:
            logger.warning(
                "RLM document loop stopped due to safety cap [doc=%s pending_edges=%s cap=%s]",
                doc_id,
                len(edges),
                max_edge_attempts,
            )

        status = self._call_tool("get_doc_exploration_status", {"doc_id": doc_id}, doc_id=doc_id)
        pending_count = int(status.get("pending_count", 0)) if isinstance(status, dict) else 0

        if pending_count == 0:
            bridges_created = sum(er.accepted for er in edge_results)
            _doc_mark = self._call_tool(
                "mark_document_explored",
                {"doc_id": doc_id, "num_bridges_created": bridges_created},
                doc_id=doc_id,
            )
            self.reconciler.rlm_metrics["docs_completed"] += 1
        else:
            bridges_created = sum(er.accepted for er in edge_results)

        self.reconciler.entities_explored += 1

        summary = {
            "entity": doc_id,
            "iterations": len(edge_results),
            "tool_calls": 0,
            "bridges_created": bridges_created,
            "mode": "rlm",
            "pending_edges": pending_count,
            "edge_results": [
                {
                    "edge": er.edge.as_dict(),
                    "accepted": er.accepted,
                    "attempted": er.attempted,
                    "rejected": er.rejected,
                    "budget_exhausted": er.budget_exhausted,
                    "edge_tokens": er.edge_tokens,
                    "property_attempted": er.property_attempted,
                    "guided_retry_count": er.guided_retry_count,
                    "last_retry_hint": er.last_retry_hint,
                    "last_retry_failure_codes": list(er.last_retry_failure_codes),
                    "worker_prompt_version": WORKER_PROMPT_VERSION,
                }
                for er in edge_results
            ],
        }

        self._emit_event("rlm_doc_summary", summary)
        return summary

    def run_corpus(self, max_documents: Optional[int] = None) -> Dict[str, Any]:
        docs_processed = 0
        doc_results: List[Dict[str, Any]] = []

        while True:
            if max_documents is not None and docs_processed >= max_documents:
                break
            if "max_tokens" in self.reconciler.budget and self.reconciler.tokens_used >= self.reconciler.budget["max_tokens"]:
                break

            plan = self._call_tool("plan_corpus_exploration", {"strategy": "max_pending_neighbors", "limit": 20, "include_completed": False})
            if not isinstance(plan, dict):
                break

            next_doc = plan.get("next_doc")
            if not next_doc:
                break

            doc_id = next_doc.get("doc_id")
            if not doc_id:
                break

            # Skip already completed docs from planner glitches
            if str(next_doc.get("status")) == "completed":
                break

            doc_result = self.run_document(doc_id)
            doc_results.append(doc_result)
            docs_processed += 1

        final_stats = self.reconciler.tools.get_bridge_statistics()
        return {
            "mode": "rlm",
            "documents_explored": docs_processed,
            "total_bridges": final_stats.get("total_bridges", 0),
            "tokens_used": self.reconciler.tokens_used,
            "input_tokens": self.reconciler.input_tokens,
            "output_tokens": self.reconciler.output_tokens,
            "rlm_metrics": dict(self.reconciler.rlm_metrics),
            "document_results": doc_results,
        }


class HybridScheduler(RootScheduler):
    """Planner-guided scheduler with hard guardrails and deterministic fallback."""

    VALID_SCOPES = {"edge", "doc_edge", "corpus_doc_edge"}

    def __init__(
        self,
        reconciler: Any,
        worker: RecursiveEdgeWorker,
        edge_max_depth: int = 1,
        edge_max_calls: int = 2,
        edge_max_tokens: int = 3000,
        max_optional_docs_per_edge: int = 2,
        hybrid_scope: str = "doc_edge",
        edge_parallel_enabled: bool = False,
        edge_parallel_workers: int = 1,
        parallel_progress_heartbeat_seconds: float = 10.0,
        parallel_stuck_warning_seconds: float = 60.0,
    ):
        super().__init__(
            reconciler=reconciler,
            worker=worker,
            edge_max_depth=edge_max_depth,
            edge_max_calls=edge_max_calls,
            edge_max_tokens=edge_max_tokens,
            max_optional_docs_per_edge=max_optional_docs_per_edge,
        )
        self.hybrid_scope = hybrid_scope if hybrid_scope in self.VALID_SCOPES else "doc_edge"
        self.edge_parallel_enabled = bool(edge_parallel_enabled)
        self.edge_parallel_workers = max(1, int(edge_parallel_workers))
        self.parallel_progress_heartbeat_seconds = max(0.1, float(parallel_progress_heartbeat_seconds))
        self.parallel_stuck_warning_seconds = max(0.1, float(parallel_stuck_warning_seconds))

    @staticmethod
    def _inflight_edges_from_futures(
        pending: Set[Any],
        future_to_edge: Dict[Any, EdgeKey],
    ) -> List[Dict[str, Any]]:
        inflight = [future_to_edge[future].as_dict() for future in pending if future in future_to_edge]
        inflight.sort(
            key=lambda edge: (
                str(edge.get("doc_id", "")),
                str(edge.get("entity_name", "")),
                str(edge.get("neighbor_name", "")),
                str(edge.get("relationship", "")),
            )
        )
        return inflight

    @staticmethod
    def _oldest_inflight_seconds(
        pending: Set[Any],
        start_times: Dict[Any, float],
        now_ts: float,
    ) -> float:
        if not pending:
            return 0.0
        ages = [max(0.0, now_ts - float(start_times.get(future, now_ts))) for future in pending]
        return max(ages) if ages else 0.0

    def _planner_fallback(
        self,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
        decision_type: Optional[str] = None,
    ) -> None:
        self._metric_inc("planner_fallbacks", 1)
        event_payload = {
            "decision_type": str(decision_type or (payload or {}).get("decision_type", "")),
            "source": "deterministic_fallback",
            "planner_prompt_version": PLANNER_PROMPT_VERSION,
            "fallback_reason": str(reason),
        }
        self._emit_event("hybrid_planner_decision", event_payload)
        logger.debug(
            "Hybrid planner fallback [scope=%s reason=%s payload=%s]",
            self.hybrid_scope,
            reason,
            payload or {},
        )

    @staticmethod
    def _planner_decision_examples(decision_type: str) -> str:
        if decision_type == "corpus_doc_selection":
            return PLANNER_CORPUS_DOC_EXAMPLES
        if decision_type == "doc_edge_selection":
            return PLANNER_DOC_EDGE_EXAMPLES
        if decision_type == "edge_action":
            return PLANNER_EDGE_ACTION_EXAMPLES
        return ""

    _PLANNER_RESPONSE_MODELS: Dict[str, type] = {
        "corpus_doc_selection": PlannerCorpusDocSelection,
        "doc_edge_selection": PlannerDocEdgeSelection,
        "edge_action": PlannerEdgeAction,
    }

    def _planner_response_model(self, decision_type: str) -> Optional[type]:
        """Return Pydantic model class for the given planner decision type."""
        return self._PLANNER_RESPONSE_MODELS.get(decision_type)

    @staticmethod
    def _planner_schema_hint(decision_type: str) -> Dict[str, str]:
        if decision_type == "corpus_doc_selection":
            return {"doc_id": "string", "reason": "string"}
        if decision_type == "doc_edge_selection":
            return {"edge_index": "integer", "reason": "string"}
        if decision_type == "edge_action":
            return {
                "action": "run_worker|stop_edge",
                "depth": "integer",
                "include_optional": "boolean",
                "reason": "string",
            }
        return {"reason": "string"}

    def _build_planner_messages(self, decision_type: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        examples = self._planner_decision_examples(decision_type)
        schema_hint = self._planner_schema_hint(decision_type)
        system_prompt = (
            "You are a strict planner for hybrid GSW exploration.\n\n"
            f"Prompt version: {PLANNER_PROMPT_VERSION}\n\n"
            f"{PLANNER_SHARED_CHECKLIST}\n"
            f"{examples}\n"
            f"Decision type: {decision_type}\n"
            f"Required output schema: {json.dumps(schema_hint, ensure_ascii=False)}\n"
            "Return exactly one JSON object and no markdown."
        )
        user_payload = {
            "decision_type": decision_type,
            "hybrid_scope": self.hybrid_scope,
            "payload": payload,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def _validate_planner_output(
        self,
        decision_type: str,
        payload: Dict[str, Any],
        parsed: Any,
    ) -> Tuple[bool, str]:
        if not isinstance(parsed, dict):
            return False, "non_dict_json"
        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return False, "missing_reason"

        if decision_type == "corpus_doc_selection":
            chosen_doc = parsed.get("doc_id")
            if not isinstance(chosen_doc, str) or not chosen_doc.strip():
                return False, "invalid_doc_choice"
            candidates = payload.get("candidates", [])
            valid_ids = {
                str(item.get("doc_id"))
                for item in candidates
                if isinstance(item, dict) and item.get("doc_id")
            }
            if valid_ids and chosen_doc not in valid_ids:
                return False, "invalid_doc_choice"
            return True, ""

        if decision_type == "doc_edge_selection":
            edge_index = parsed.get("edge_index")
            candidates = payload.get("candidates", [])
            if not isinstance(edge_index, int):
                return False, "invalid_edge_choice"
            if edge_index < 0 or edge_index >= len(candidates):
                return False, "invalid_edge_choice"
            return True, ""

        if decision_type == "edge_action":
            action = parsed.get("action")
            allowed_actions = payload.get("allowed_actions", [])
            if action not in allowed_actions:
                return False, "invalid_edge_action"

            depth = parsed.get("depth")
            include_optional = parsed.get("include_optional")
            max_depth = int(payload.get("state", {}).get("max_depth", self.edge_max_depth))
            has_optional_contexts = bool(payload.get("state", {}).get("has_optional_contexts", False))

            if not isinstance(depth, int) or not (0 <= depth <= max_depth):
                return False, "invalid_edge_depth"
            if not isinstance(include_optional, bool):
                return False, "invalid_edge_include_optional"
            if include_optional and not has_optional_contexts:
                return False, "invalid_edge_include_without_optional_contexts"
            return True, ""

        return False, "unknown_decision_type"

    def _call_planner_json(
        self,
        decision_type: str,
        payload: Dict[str, Any],
        max_tokens: int = 384,
    ) -> Optional[Dict[str, Any]]:
        self._metric_inc("planner_decisions", 1)
        messages = self._build_planner_messages(decision_type, payload)

        def _attempt(
            call_messages: List[Dict[str, str]],
            attempt_label: str,
            use_structured: bool = True,
        ) -> Tuple[Optional[Dict[str, Any]], str, str]:
            schema_cls = self._planner_response_model(decision_type) if use_structured else None
            response_format = schema_cls or {"type": "json_object"}
            try:
                message = self.reconciler._call_model_for_stage(
                    messages=call_messages,
                    model_name=self.reconciler.root_model,
                    max_tokens=max(128, int(max_tokens)),
                    temperature=0.0,
                    stage="root",
                    response_format=response_format,
                )
            except Exception as e:
                return None, "", f"planner_call_error:{e}"

            raw_text, _source = self.reconciler._extract_message_content_with_source(message)
            try:
                parsed = self.reconciler._extract_json_from_text(raw_text)
            except Exception:
                return None, raw_text, "planner_parse_error"

            valid, error_code = self._validate_planner_output(decision_type, payload, parsed)
            if not valid:
                return None, raw_text, error_code

            self._emit_event(
                "hybrid_planner_decision",
                {
                    "decision_type": decision_type,
                    "source": "planner",
                    "planner_prompt_version": PLANNER_PROMPT_VERSION,
                    "reason": str(parsed.get("reason", "")),
                    "attempt": attempt_label,
                    "decision": parsed,
                },
            )
            return parsed, raw_text, ""

        parsed, first_raw, first_error = _attempt(messages, "initial")
        if parsed is not None:
            return parsed

        schema_hint = self._planner_schema_hint(decision_type)
        repair_messages = list(messages)
        repair_messages.append(
            {
                "role": "assistant",
                "content": first_raw or "[empty_response]",
            }
        )
        repair_messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous output was invalid. Re-output exactly one JSON object matching this schema: "
                    f"{json.dumps(schema_hint, ensure_ascii=False)}. "
                    "Include a non-empty 'reason' string. No markdown. No extra wrapper text."
                ),
            }
        )

        parsed, retry_raw, retry_error = _attempt(repair_messages, "repair_retry", use_structured=False)
        if parsed is not None:
            return parsed

        self._planner_fallback(
            "planner_unresolved_after_retry",
            {
                "decision_type": decision_type,
                "initial_error": first_error,
                "retry_error": retry_error,
                "initial_preview": self.reconciler._preview_text(first_raw),
                "retry_preview": self.reconciler._preview_text(retry_raw),
            },
            decision_type=decision_type,
        )
        return None

    def _choose_doc_for_corpus(self, plan: Dict[str, Any]) -> Optional[str]:
        queue = plan.get("queue", [])
        if not isinstance(queue, list):
            queue = []
        candidates = [
            item for item in queue
            if isinstance(item, dict) and str(item.get("status")) != "completed" and item.get("doc_id")
        ]
        deterministic_doc = None
        next_doc = plan.get("next_doc", {})
        if isinstance(next_doc, dict) and next_doc.get("doc_id"):
            deterministic_doc = str(next_doc["doc_id"])
        elif candidates:
            deterministic_doc = sorted(candidates, key=lambda x: str(x.get("doc_id")))[0].get("doc_id")

        if self.hybrid_scope != "corpus_doc_edge":
            return deterministic_doc

        if not candidates:
            return deterministic_doc

        payload = {
            "candidates": [
                {
                    "doc_id": str(item.get("doc_id")),
                    "status": str(item.get("status", "")),
                    "pending_neighbors": int(item.get("pending_neighbors", 0) or 0),
                    "pending_entities": int(item.get("pending_entities", 0) or 0),
                }
                for item in candidates[:30]
            ],
            "selection_rule": "Pick one doc_id from candidates.",
            "output_schema": {"doc_id": "string", "reason": "string"},
        }
        parsed = self._call_planner_json("corpus_doc_selection", payload, max_tokens=320)
        if not parsed:
            return deterministic_doc

        chosen_doc = parsed.get("doc_id")
        valid_ids = {str(item.get("doc_id")) for item in candidates}
        if isinstance(chosen_doc, str) and chosen_doc in valid_ids:
            return chosen_doc

        self._planner_fallback(
            "invalid_doc_choice",
            {"chosen_doc": chosen_doc, "valid_ids": sorted(valid_ids)[:10]},
            decision_type="corpus_doc_selection",
        )
        return deterministic_doc

    def _choose_edge_for_doc(self, doc_id: str, edges: List[EdgeKey]) -> EdgeKey:
        deterministic_edge = edges[0]
        if self.hybrid_scope == "edge":
            return deterministic_edge

        payload = {
            "doc_id": doc_id,
            "candidates": [
                {
                    "edge_index": idx,
                    "doc_id": edge.doc_id,
                    "entity_name": edge.entity_name,
                    "neighbor_name": edge.neighbor_name,
                    "relationship": edge.relationship,
                }
                for idx, edge in enumerate(edges[:50])
            ],
            "selection_rule": "Pick exactly one candidate by edge_index.",
            "output_schema": {"edge_index": "integer", "reason": "string"},
        }
        parsed = self._call_planner_json("doc_edge_selection", payload, max_tokens=320)
        if not parsed:
            return deterministic_edge

        edge_index = parsed.get("edge_index")
        if isinstance(edge_index, int) and 0 <= edge_index < len(edges):
            return edges[edge_index]

        self._planner_fallback(
            "invalid_edge_choice",
            {"edge_index": edge_index, "num_edges": len(edges)},
            decision_type="doc_edge_selection",
        )
        return deterministic_edge

    def _plan_edge_action(
        self,
        edge: EdgeKey,
        current_depth: int,
        current_include_optional: bool,
        call_count: int,
        remaining_edge_tokens: int,
        accepted_total: int,
        attempted_total: int,
        rejected_total: int,
        no_candidate_calls: int,
        has_optional_contexts: bool,
        path_proof_count: int = 0,
        mandatory_context_count: int = 0,
        optional_context_count: int = 0,
        cross_doc_fact_count: int = 0,
        last_worker_notes: str = "",
        last_candidates_count: int = 0,
        last_rejected_delta: int = 0,
        last_close_reason: str = "",
    ) -> Dict[str, Any]:
        deterministic = {
            "action": "run_worker",
            "depth": current_depth,
            "include_optional": bool(current_include_optional and has_optional_contexts),
            "source": "deterministic",
            "reason": "Deterministic fallback action.",
        }

        payload = {
            "edge": edge.as_dict(),
            "state": {
                "current_depth": int(current_depth),
                "current_include_optional": bool(current_include_optional),
                "call_count": int(call_count),
                "remaining_edge_tokens": int(remaining_edge_tokens),
                "accepted_total": int(accepted_total),
                "attempted_total": int(attempted_total),
                "rejected_total": int(rejected_total),
                "no_candidate_calls": int(no_candidate_calls),
                "has_optional_contexts": bool(has_optional_contexts),
                "max_depth": int(self.edge_max_depth),
                "path_proof_count": int(path_proof_count),
                "mandatory_context_count": int(mandatory_context_count),
                "optional_context_count": int(optional_context_count),
                "cross_doc_fact_count": int(cross_doc_fact_count),
                "last_worker_notes": str(last_worker_notes or "")[:300],
                "last_candidates_count": int(last_candidates_count),
                "last_rejected_delta": int(last_rejected_delta),
                "last_close_reason": str(last_close_reason or ""),
            },
            "allowed_actions": ["run_worker", "stop_edge"],
            "output_schema": {
                "action": "run_worker|stop_edge",
                "depth": "integer",
                "include_optional": "boolean",
                "reason": "string",
            },
        }
        parsed = self._call_planner_json("edge_action", payload, max_tokens=320)
        if not parsed:
            return deterministic

        action = parsed.get("action")
        reason = str(parsed.get("reason", "")).strip()
        if action == "stop_edge":
            return {
                "action": "stop_edge",
                "depth": int(parsed.get("depth", current_depth)),
                "include_optional": bool(parsed.get("include_optional", False)),
                "source": "planner",
                "reason": reason,
            }
        if action != "run_worker":
            self._planner_fallback("invalid_edge_action", {"action": action}, decision_type="edge_action")
            return deterministic

        depth = parsed.get("depth")
        include_optional = parsed.get("include_optional")
        if not isinstance(depth, int) or not (0 <= depth <= self.edge_max_depth):
            self._planner_fallback(
                "invalid_edge_depth",
                {"depth": depth, "max_depth": self.edge_max_depth},
                decision_type="edge_action",
            )
            return deterministic
        if not isinstance(include_optional, bool):
            self._planner_fallback(
                "invalid_edge_include_optional",
                {"include_optional": include_optional},
                decision_type="edge_action",
            )
            return deterministic
        if include_optional and not has_optional_contexts:
            self._planner_fallback(
                "invalid_edge_include_without_optional_contexts",
                {},
                decision_type="edge_action",
            )
            return deterministic

        return {
            "action": "run_worker",
            "depth": depth,
            "include_optional": bool(include_optional),
            "source": "planner",
            "reason": reason,
        }

    def _release_focus_internal(self, edge: EdgeKey) -> None:
        release_fn = getattr(self.reconciler.tools, "release_neighbor_focus_internal", None)
        if callable(release_fn):
            release_fn(
                doc_id=edge.doc_id,
                entity_name=edge.entity_name,
                neighbor_name=edge.neighbor_name,
                relationship=edge.relationship,
            )
            return

        active = getattr(self.reconciler.tools, "active_neighbor_focus", None)
        if not isinstance(active, dict):
            return
        if (
            str(active.get("doc_id", "")).strip().lower() == edge.doc_id.strip().lower()
            and str(active.get("entity_name", "")).strip().lower() == edge.entity_name.strip().lower()
            and str(active.get("neighbor_name", "")).strip().lower() == edge.neighbor_name.strip().lower()
            and str(active.get("relationship", "")).strip().lower() == edge.relationship.strip().lower()
        ):
            self.reconciler.tools.active_neighbor_focus = None

    def _is_edge_pending(self, edge: EdgeKey) -> bool:
        plan = self.reconciler.tools.doc_exploration_plans.get(edge.doc_id)
        if not isinstance(plan, dict):
            return False
        entity_key = edge.entity_name.strip().lower()
        neighbor_key = edge.neighbor_name.strip().lower()
        rel_key = edge.relationship.strip().lower()
        for entity_plan in plan.get("entities_to_explore", []) or []:
            if str(entity_plan.get("name", "")).strip().lower() != entity_key:
                continue
            for neighbor in entity_plan.get("neighbors", []) or []:
                if (
                    str(neighbor.get("entity", "")).strip().lower() == neighbor_key
                    and str(neighbor.get("relationship", "")).strip().lower() == rel_key
                    and str(neighbor.get("status", "")).strip().lower() == "pending"
                ):
                    return True
        return False

    def _prioritize_retry_edges(self, doc_id: str, edges: List[EdgeKey]) -> List[EdgeKey]:
        retry_edge_keys = {
            edge_key
            for edge_key, brief in self._pending_retry_briefs.items()
            if brief.edge.doc_id == doc_id
        }
        if not retry_edge_keys:
            return edges
        prioritized = sorted(
            edges,
            key=lambda edge: (
                0 if edge.as_tuple() in retry_edge_keys else 1,
                edges.index(edge),
            ),
        )
        return prioritized

    def _prepare_edge_work_parallel(self, edge: EdgeKey) -> Optional[PreparedEdgeWork]:
        focus_acquired = False
        retry_brief = self._peek_pending_retry_brief(edge)
        guided_retry_count = self._guided_retry_count(edge)
        try:
            source_context = self._call_tool(
                "get_entity_context",
                {"entity_name": edge.entity_name, "doc_id": edge.doc_id},
            )
            if not isinstance(source_context, dict) or "error" in source_context:
                source_context = {
                    "entity": edge.entity_name,
                    "doc_id": edge.doc_id,
                    "qa_pairs": [],
                    "roles": [],
                    "states": [],
                    "relationships": {},
                }

            focus_result = self._call_tool("begin_neighbor_focus", edge.as_dict(), doc_id=edge.doc_id)
            if isinstance(focus_result, dict) and focus_result.get("error"):
                self._emit_event(
                    "edge_packet_prepared",
                    {
                        "edge": edge.as_dict(),
                        "prepared": False,
                        "reason": "focus_acquire_failed",
                        "detail": str(focus_result.get("error", "")),
                    },
                )
                return None
            focus_acquired = True

            coverage = self._call_tool("plan_neighbor_doc_coverage", edge.as_dict(), doc_id=edge.doc_id)
            if not isinstance(coverage, dict) or "error" in coverage:
                self._emit_event(
                    "edge_packet_prepared",
                    {
                        "edge": edge.as_dict(),
                        "prepared": False,
                        "reason": "coverage_plan_failed",
                        "detail": str((coverage or {}).get("error", "invalid_coverage")),
                    },
                )
                return None

            mandatory_docs = list(coverage.get("mandatory_docs", []))
            optional_docs = self._select_optional_doc_ids(edge, coverage.get("optional_fuzzy_docs", []))

            mandatory_contexts = self._collect_contexts_for_docs(edge.neighbor_name, mandatory_docs, source_doc=edge.doc_id)
            optional_contexts = self._collect_contexts_for_docs(edge.neighbor_name, optional_docs, source_doc=edge.doc_id) if optional_docs else []

            packet = self._build_edge_packet(edge, coverage, source_context, mandatory_contexts, optional_contexts)
            signal = self._score_edge_signal(coverage, mandatory_contexts)
            edge_call_limit, edge_token_limit = self._low_signal_budgets(signal)
            viability = self._evaluate_edge_viability(packet, include_optional=False)
            if signal.get("tier") == "low":
                self._metric_inc("low_signal_edges", 1)

            self._emit_event(
                "edge_packet_prepared",
                {
                    "edge": edge.as_dict(),
                    "prepared": True,
                    "signal_tier": signal.get("tier"),
                    "signal_score": signal.get("score"),
                    "mandatory_docs": len(mandatory_docs),
                    "optional_docs": len(optional_docs),
                    "path_proof_count": int(viability.get("path_proof_count", 0)),
                    "call_limit": int(edge_call_limit),
                    "token_limit": int(edge_token_limit),
                },
            )
            return PreparedEdgeWork(
                edge=edge,
                packet=packet,
                signal=signal,
                edge_call_limit=edge_call_limit,
                edge_token_limit=edge_token_limit,
                viability=viability,
                retry_brief=retry_brief,
                guided_retry_count=guided_retry_count,
            )
        finally:
            if focus_acquired:
                self._release_focus_internal(edge)

    def _generate_parallel_worker_proposal(self, work: PreparedEdgeWork) -> EdgeWorkerProposal:
        edge = work.edge
        packet = work.packet
        signal = work.signal

        start_planner_decisions = self._metric_value("planner_decisions")
        start_planner_fallbacks = self._metric_value("planner_fallbacks")
        start_planner_stops = self._metric_value("planner_stop_actions")

        attempted_total = 0
        prefilter_rejected = 0
        sanitized_candidates: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[Any, ...]] = set()
        budget_exhausted = False
        call_count = 0
        no_candidate_calls = 0
        drop_reason_counts = self._new_drop_reason_counts()
        close_reason = "call_limit_reached"
        close_detail = f"Reached edge call limit ({work.edge_call_limit})."
        last_worker_output: Optional[WorkerOutput] = None
        last_run_signature: Optional[Tuple[Any, ...]] = None
        last_failure_signature: Optional[Tuple[Tuple[Any, ...], ...]] = None
        last_failure_state: Optional[Tuple[int, bool, bool, int]] = None
        last_worker_notes = ""
        last_candidates_count = 0
        last_rejected_delta = 0
        active_retry_brief = work.retry_brief
        guided_retry_count = int(work.guided_retry_count)
        last_retry_hint = active_retry_brief.retry_hint if active_retry_brief is not None else ""
        last_retry_failure_codes = list(active_retry_brief.failure_codes) if active_retry_brief is not None else []

        depth = 0
        include_optional = False
        local_edge_budget_tokens = 0
        local_edge_total_tokens = 0

        while call_count < work.edge_call_limit:
            edge_spend = max(0, int(local_edge_budget_tokens))
            remaining_edge_tokens = work.edge_token_limit - edge_spend
            if remaining_edge_tokens <= 0:
                budget_exhausted = True
                close_reason = "budget_exhausted"
                close_detail = "Per-edge token budget exhausted before worker call."
                break

            viability = self._evaluate_edge_viability(packet, include_optional=include_optional)
            action = self._plan_edge_action(
                edge=edge,
                current_depth=depth,
                current_include_optional=include_optional,
                call_count=call_count,
                remaining_edge_tokens=remaining_edge_tokens,
                accepted_total=0,
                attempted_total=attempted_total,
                rejected_total=prefilter_rejected,
                no_candidate_calls=no_candidate_calls,
                has_optional_contexts=bool(packet.optional_contexts),
                path_proof_count=viability["path_proof_count"],
                mandatory_context_count=viability["mandatory_context_count"],
                optional_context_count=viability["optional_context_count"],
                cross_doc_fact_count=viability["cross_doc_fact_count"],
                last_worker_notes=last_worker_notes,
                last_candidates_count=last_candidates_count,
                last_rejected_delta=last_rejected_delta,
                last_close_reason=close_reason,
            )

            if action.get("action") == "stop_edge":
                self._emit_worker_call(
                    edge=edge,
                    call_index=call_count + 1,
                    depth=depth,
                    include_optional=include_optional,
                    remaining_edge_tokens=remaining_edge_tokens,
                    planner_action="stop_edge",
                    planner_source=str(action.get("source", "")),
                    planner_reason=str(action.get("reason", "")),
                    worker_invoked=False,
                    retry_brief=active_retry_brief,
                )
                self._metric_inc("planner_stop_actions", 1)
                close_reason = "planner_stop"
                close_detail = f"Planner stop accepted (reason='{str(action.get('reason', '')).strip()}')."
                break

            run_depth = int(action.get("depth", depth))
            run_include_optional = bool(action.get("include_optional", include_optional)) and bool(packet.optional_contexts)
            depth = run_depth
            include_optional = run_include_optional
            viability = self._evaluate_edge_viability(packet, include_optional=run_include_optional)
            path_proofs = viability["path_proofs"]
            proof_allowlist = self._collect_path_proof_allowlist(path_proofs)
            render_input = RenderInput(
                edge=edge,
                proofs=path_proofs,
                constraints=packet.constraints,
            ) if path_proofs else None

            no_path_proofs = viability["path_proof_count"] == 0
            can_optional_probe = (
                call_count == 0
                and not run_include_optional
                and not packet.mandatory_contexts
                and bool(packet.optional_contexts)
                and run_depth < self.edge_max_depth
            )
            if no_path_proofs:
                self._metric_inc("planner_actions_overridden", 1)
                self._planner_fallback(
                    "run_worker_overridden_no_path",
                    {
                        "edge": edge.as_dict(),
                        "mandatory_contexts": viability["mandatory_context_count"],
                        "optional_contexts": viability["optional_context_count"],
                        "cross_doc_facts": viability["cross_doc_fact_count"],
                    },
                    decision_type="edge_action",
                )
                if can_optional_probe:
                    include_optional = True
                    depth = min(self.edge_max_depth, max(run_depth + 1, 1))
                    self._metric_inc("recursive_invocations", 1)
                    continue

                self._emit_worker_call(
                    edge=edge,
                    call_index=call_count + 1,
                    depth=run_depth,
                    include_optional=run_include_optional,
                    remaining_edge_tokens=remaining_edge_tokens,
                    planner_action="run_worker",
                    planner_source=str(action.get("source", "")),
                    planner_reason=str(action.get("reason", "")),
                    worker_invoked=False,
                    retry_brief=active_retry_brief,
                )
                self._metric_inc("edges_skipped_no_path", 1)
                close_reason = "no_viable_path_proof"
                close_detail = (
                    "No viable path proofs before worker call "
                    f"(mandatory_contexts={viability['mandatory_context_count']}, "
                    f"optional_contexts={viability['optional_context_count']}, "
                    f"cross_doc_facts={viability['cross_doc_fact_count']})."
                )
                break

            run_signature = (
                int(run_depth),
                bool(run_include_optional),
                viability["evidence_signature"],
            )
            if call_count > 0 and run_signature == last_run_signature and active_retry_brief is None:
                self._metric_inc("planner_actions_overridden", 1)
                self._metric_inc("calls_blocked_no_progress", 1)
                self._planner_fallback(
                    "run_worker_overridden_no_progress",
                    {
                        "edge": edge.as_dict(),
                        "run_depth": run_depth,
                        "run_include_optional": run_include_optional,
                        "evidence_signature": list(viability["evidence_signature"]),
                    },
                    decision_type="edge_action",
                )
                self._emit_worker_call(
                    edge=edge,
                    call_index=call_count + 1,
                    depth=run_depth,
                    include_optional=run_include_optional,
                    remaining_edge_tokens=remaining_edge_tokens,
                    planner_action="run_worker",
                    planner_source=str(action.get("source", "")),
                    planner_reason=str(action.get("reason", "")),
                    worker_invoked=False,
                    retry_brief=active_retry_brief,
                )
                close_reason = "repeated_failure"
                close_detail = "Blocked repeated worker call with unchanged evidence state."
                break
            last_run_signature = run_signature

            call_index = call_count + 1
            if active_retry_brief is not None:
                self._metric_inc("guided_retry_started", 1)
                self._emit_guided_retry_event(
                    "guided_retry_started",
                    active_retry_brief,
                    mode="hybrid_parallel_generation",
                    call_index=call_index,
                    depth=run_depth,
                    include_optional=run_include_optional,
                )
            self._emit_event(
                "edge_worker_dispatched",
                {
                    "edge": edge.as_dict(),
                    "call_index": call_index,
                    "depth": run_depth,
                    "include_optional": run_include_optional,
                },
            )
            self._emit_worker_call(
                edge=edge,
                call_index=call_index,
                depth=run_depth,
                include_optional=run_include_optional,
                remaining_edge_tokens=remaining_edge_tokens,
                planner_action=str(action.get("action", "run_worker")),
                planner_source=str(action.get("source", "")),
                planner_reason=str(action.get("reason", "")),
                worker_invoked=True,
                retry_brief=active_retry_brief,
            )
            worker_output = self.worker.generate(
                packet=packet,
                depth=run_depth,
                include_optional=run_include_optional,
                max_response_tokens=min(1400, remaining_edge_tokens),
                render_input=render_input,
                retry_brief=active_retry_brief,
            )
            active_retry_brief = None
            self._emit_event(
                "edge_worker_completed",
                {
                    "edge": edge.as_dict(),
                    "call_index": call_index,
                    "candidates_count": len(worker_output.candidates),
                    "parse_stage": worker_output.parse_stage,
                },
            )
            call_count = call_index
            last_worker_output = worker_output
            attempted_total += len(worker_output.candidates)
            local_edge_total_tokens += max(0, int(getattr(worker_output, "token_usage", 0)))
            local_edge_budget_tokens += max(0, int(getattr(worker_output, "budgeted_token_usage", 0)))
            if not worker_output.candidates:
                no_candidate_calls += 1

            per_call_rejected = 0
            per_call_sanitized = 0
            for candidate in worker_output.candidates:
                sanitized, sanitize_reasons = self._sanitize_candidate_refs_with_reasons(
                    packet,
                    candidate,
                    extra_allowed_refs=proof_allowlist,
                )
                if sanitized is None:
                    per_call_rejected += 1
                    self._merge_drop_reason_counts(drop_reason_counts, sanitize_reasons)
                    continue
                signature = self._exact_candidate_signature(sanitized)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                sanitized_candidates.append(sanitized)
                per_call_sanitized += 1
            prefilter_rejected += per_call_rejected
            if worker_output.candidates and per_call_sanitized == 0:
                drop_reason_counts["all_candidates_rejected"] = int(
                    drop_reason_counts.get("all_candidates_rejected", 0)
                ) + 1
            self._emit_worker_result(
                edge=edge,
                call_index=call_index,
                worker_output=worker_output,
                accepted_delta=0,
                rejected_delta=per_call_rejected,
            )

            last_worker_notes = str(worker_output.notes or "")
            last_candidates_count = int(len(worker_output.candidates))
            last_rejected_delta = int(per_call_rejected)

            edge_spend = max(0, int(local_edge_budget_tokens))
            if edge_spend >= work.edge_token_limit:
                budget_exhausted = True
                close_reason = "budget_exhausted"
                close_detail = "Per-edge token budget exhausted after worker call."
                break

            can_recurse = run_depth < self.edge_max_depth
            state_key = (run_depth, run_include_optional, bool(render_input), len(path_proofs))
            candidate_signature = self._candidate_signature(worker_output.candidates)
            repeated_failure = (
                bool(worker_output.candidates)
                and last_failure_state == state_key
                and last_failure_signature == candidate_signature
            )
            if worker_output.candidates:
                last_failure_state = state_key
                last_failure_signature = candidate_signature

            if can_recurse and not run_include_optional and packet.optional_contexts and (
                worker_output.need_recursion or not worker_output.candidates
            ):
                include_optional = True
                depth = min(self.edge_max_depth, run_depth + 1)
                self._metric_inc("recursive_invocations", 1)
            elif can_recurse and worker_output.need_recursion and run_include_optional:
                new_depth = min(self.edge_max_depth, run_depth + 1)
                if new_depth > run_depth:
                    depth = new_depth
                    self._metric_inc("recursive_invocations", 1)

            if repeated_failure:
                self._metric_inc("repeated_failure_stops", 1)
                close_reason = "repeated_failure"
                close_detail = "Repeated candidate signature failure at same depth/state."
                break

            if not worker_output.candidates:
                if include_optional != run_include_optional or depth != run_depth:
                    continue
                close_reason = "no_candidates_no_progression"
                close_detail = (
                    f"No candidates (notes='{worker_output.notes}', parse_stage={worker_output.parse_stage}, "
                    f"parse_attempts={worker_output.parse_attempts}, "
                    f"rejected_delta={per_call_rejected})."
                )
                break

            if signal.get("probe_only"):
                close_reason = "probe_only_no_success"
                close_detail = "Probe-only edge produced no accepted candidates."
                break

            continue

        if close_reason == "call_limit_reached":
            tail = ""
            if last_worker_output is not None:
                tail = (
                    f" last_parse_stage={last_worker_output.parse_stage};"
                    f" last_parse_attempts={last_worker_output.parse_attempts};"
                    f" last_notes={last_worker_output.notes!r}"
                )
            close_detail = (
                f"Reached edge call limit ({work.edge_call_limit}); attempted_total={attempted_total}; "
                f"prefilter_rejected={prefilter_rejected}; no_candidate_calls={no_candidate_calls};{tail}"
            )

        planner_decisions_delta = self._metric_value("planner_decisions") - start_planner_decisions
        planner_fallbacks_delta = self._metric_value("planner_fallbacks") - start_planner_fallbacks
        planner_stops_delta = self._metric_value("planner_stop_actions") - start_planner_stops

        return EdgeWorkerProposal(
            edge=edge,
            packet=packet,
            signal=signal,
            attempted=attempted_total,
            prefilter_rejected=prefilter_rejected,
            sanitized_candidates=sanitized_candidates,
            budget_exhausted=budget_exhausted,
            edge_tokens=max(0, int(local_edge_total_tokens)),
            worker_calls_made=call_count,
            no_candidate_calls=no_candidate_calls,
            close_reason=close_reason,
            close_detail=close_detail,
            planner_decisions=planner_decisions_delta,
            planner_fallbacks=planner_fallbacks_delta,
            planner_stop_actions=planner_stops_delta,
            edge_call_limit=int(work.edge_call_limit),
            edge_token_limit=int(work.edge_token_limit),
            drop_reason_counts=drop_reason_counts,
            retry_brief=work.retry_brief,
            guided_retry_count=guided_retry_count,
            last_retry_hint=last_retry_hint,
            last_retry_failure_codes=last_retry_failure_codes,
            last_run_signature=last_run_signature,
        )

    def _apply_sanitized_candidates(
        self,
        edge: EdgeKey,
        candidates: List[Dict[str, Any]],
        drop_reason_counts: Optional[Dict[str, int]] = None,
        retry_attempt: int = 0,
    ) -> Tuple[int, int, bool, Optional[EdgeRetryBrief]]:
        reasons = drop_reason_counts if drop_reason_counts is not None else self._new_drop_reason_counts()
        prepared, prevalidation_rejected = self._prevalidate_commit_candidates(candidates)
        passing, verifier_rejected, verifier_failure_codes, retry_brief = self._parallel_verifier_screen(
            edge,
            prepared,
            retry_attempt=retry_attempt,
        )
        if verifier_failure_codes.get("REVERSE_INVALID"):
            reasons["reverse_not_invertible"] = int(reasons.get("reverse_not_invertible", 0)) + int(
                verifier_failure_codes["REVERSE_INVALID"]
            )
        accepted, commit_rejected, cap_hit, commit_retry_brief = self._commit_preverified_candidates(
            edge,
            passing,
            retry_attempt=retry_attempt,
        )
        retry_brief = self._prefer_retry_brief(retry_brief, commit_retry_brief)
        rejected = prevalidation_rejected + verifier_rejected + commit_rejected
        if candidates and accepted == 0 and rejected > 0:
            reasons["all_candidates_rejected"] = int(reasons.get("all_candidates_rejected", 0)) + 1
        return accepted, rejected, cap_hit, retry_brief

    def _commit_edge_worker_proposal(self, proposal: EdgeWorkerProposal) -> EdgeRunResult:
        edge = proposal.edge
        accepted_total = 0
        rejected_total = proposal.prefilter_rejected
        close_reason = proposal.close_reason
        close_detail = proposal.close_detail
        drop_reason_counts = dict(proposal.drop_reason_counts or self._new_drop_reason_counts())
        guided_retry_count = int(proposal.guided_retry_count)
        last_retry_hint = str(proposal.last_retry_hint or "")
        last_retry_failure_codes = list(proposal.last_retry_failure_codes or [])
        mark_explored = True
        self._pop_pending_retry_brief(edge)

        focus_result = self._call_tool("begin_neighbor_focus", edge.as_dict(), doc_id=edge.doc_id)
        if isinstance(focus_result, dict) and focus_result.get("error"):
            close_reason = "stale_edge_skipped"
            close_detail = str(focus_result.get("error", "focus_acquire_failed_at_commit"))
            result = EdgeRunResult(
                edge=edge,
                accepted=0,
                attempted=proposal.attempted,
                rejected=rejected_total,
                budget_exhausted=proposal.budget_exhausted,
                edge_tokens=max(0, proposal.edge_tokens),
                property_attempted=False,
                guided_retry_count=guided_retry_count,
                last_retry_hint=last_retry_hint,
                last_retry_failure_codes=list(last_retry_failure_codes),
            )
            self._emit_event(
                "rlm_edge_summary",
                {
                    "edge": edge.as_dict(),
                    "accepted": result.accepted,
                    "attempted": result.attempted,
                    "rejected": result.rejected,
                    "budget_exhausted": result.budget_exhausted,
                    "edge_tokens": result.edge_tokens,
                    "property_attempted": result.property_attempted,
                    "signal_tier": proposal.signal.get("tier"),
                    "signal_score": proposal.signal.get("score"),
                    "no_candidate_calls": proposal.no_candidate_calls,
                    "call_limit": int(proposal.edge_call_limit),
                    "token_limit": int(proposal.edge_token_limit),
                    "worker_prompt_version": WORKER_PROMPT_VERSION,
                    "mode": "hybrid",
                    "hybrid_scope": self.hybrid_scope,
                    "planner_decisions": proposal.planner_decisions,
                    "planner_fallbacks": proposal.planner_fallbacks,
                    "planner_stop_actions": proposal.planner_stop_actions,
                    "worker_calls_made": proposal.worker_calls_made,
                    "close_reason": close_reason,
                    "close_detail": close_detail,
                    "drop_reason_counts": drop_reason_counts,
                    "guided_retry_count": guided_retry_count,
                    "last_retry_hint": last_retry_hint,
                    "last_retry_failure_codes": list(last_retry_failure_codes),
                },
            )
            return result

        try:
            coverage = self._call_tool("plan_neighbor_doc_coverage", edge.as_dict(), doc_id=edge.doc_id)
            if not isinstance(coverage, dict) or "error" in coverage:
                close_reason = "coverage_guard"
                close_detail = str((coverage or {}).get("error", "coverage_refresh_failed"))
            else:
                accepted_apply, rejected_apply, cap_hit, retry_brief = self._apply_sanitized_candidates(
                    edge,
                    proposal.sanitized_candidates,
                    drop_reason_counts=drop_reason_counts,
                    retry_attempt=guided_retry_count + 1,
                )
                accepted_total += accepted_apply
                rejected_total += rejected_apply
                if accepted_total > 0:
                    if proposal.retry_brief is not None:
                        self._metric_inc("guided_retry_succeeded", 1)
                        self._emit_guided_retry_event(
                            "guided_retry_succeeded",
                            proposal.retry_brief,
                            mode="hybrid_parallel_commit",
                            accepted_total=accepted_total,
                        )
                    close_reason = "accepted_bridge"
                    if cap_hit:
                        close_detail = (
                            f"Accepted {accepted_total} bridge(s); hit per-edge acceptance cap "
                            f"({getattr(self.reconciler, 'edge_max_accepted_bridges', 10)})."
                        )
                    else:
                        close_detail = f"Accepted {accepted_total} bridge(s)."
                    self._clear_guided_retry_state(edge)
                elif retry_brief is not None:
                    last_retry_hint = retry_brief.retry_hint
                    last_retry_failure_codes = list(retry_brief.failure_codes)
                    current_include_optional = False
                    if isinstance(proposal.last_run_signature, tuple) and len(proposal.last_run_signature) >= 2:
                        current_include_optional = bool(proposal.last_run_signature[1])
                    viability = self._evaluate_edge_viability(proposal.packet, include_optional=current_include_optional)
                    current_path_proofs = viability.get("path_proofs", [])
                    can_recurse = (
                        bool(proposal.packet.optional_contexts)
                        and isinstance(proposal.last_run_signature, tuple)
                        and len(proposal.last_run_signature) >= 1
                        and int(proposal.last_run_signature[0]) < self.edge_max_depth
                    ) if proposal.last_run_signature is not None else bool(proposal.packet.optional_contexts)
                    retry_candidate_signature = (
                        (normalize_similarity_text(retry_brief.rejected_question), normalize_similarity_text(retry_brief.rejected_reverse_question)),
                    )
                    retry_signature = self._guided_retry_signature(
                        proposal.last_run_signature or ("parallel_commit", current_include_optional, ()),
                        retry_candidate_signature,
                        retry_brief,
                    )
                    if retry_brief.retry_hint == "stop_edge" or not retry_brief.retryable:
                        close_reason = "verifier_stop_edge"
                        close_detail = retry_brief.retry_reason or "Verifier indicated this edge should stop."
                    elif guided_retry_count >= GUIDED_RETRY_LIMIT:
                        self._metric_inc("guided_retry_exhausted", 1)
                        self._emit_guided_retry_event(
                            "guided_retry_exhausted",
                            retry_brief,
                            mode="hybrid_parallel_commit",
                            guided_retry_count=guided_retry_count,
                            reason="retry_limit",
                        )
                        close_reason = "guided_retry_exhausted"
                        close_detail = f"Reached guided retry limit ({GUIDED_RETRY_LIMIT}) for this edge."
                    elif not self._guided_retry_change_room(
                        proposal.packet,
                        current_path_proofs,
                        current_include_optional,
                        can_recurse,
                        retry_brief,
                    ):
                        self._metric_inc("guided_retry_exhausted", 1)
                        self._emit_guided_retry_event(
                            "guided_retry_exhausted",
                            retry_brief,
                            mode="hybrid_parallel_commit",
                            guided_retry_count=guided_retry_count,
                            reason="no_alternate_evidence",
                        )
                        close_reason = "guided_retry_exhausted"
                        close_detail = "Verifier suggested retry, but no alternate same-edge evidence remains."
                    elif self._last_guided_retry_signature(edge) == retry_signature:
                        self._metric_inc("guided_retry_blocked_no_progress", 1)
                        self._emit_guided_retry_event(
                            "guided_retry_blocked_no_progress",
                            retry_brief,
                            mode="hybrid_parallel_commit",
                            guided_retry_count=guided_retry_count,
                        )
                        close_reason = "guided_retry_blocked_no_progress"
                        close_detail = "Blocked guided retry with unchanged verifier hint and evidence state."
                    else:
                        guided_retry_count += 1
                        scheduled_retry = EdgeRetryBrief(
                            edge=retry_brief.edge,
                            failure_codes=list(retry_brief.failure_codes),
                            retryable=retry_brief.retryable,
                            retry_hint=retry_brief.retry_hint,
                            retry_reason=retry_brief.retry_reason,
                            rejected_question=retry_brief.rejected_question,
                            rejected_reverse_question=retry_brief.rejected_reverse_question,
                            retry_attempt=guided_retry_count,
                        )
                        self._set_guided_retry_count(edge, guided_retry_count)
                        self._set_last_guided_retry_signature(edge, retry_signature)
                        self._set_pending_retry_brief(scheduled_retry)
                        self._metric_inc("guided_retry_scheduled", 1)
                        self._emit_guided_retry_event(
                            "guided_retry_scheduled",
                            scheduled_retry,
                            mode="hybrid_parallel_commit",
                            guided_retry_count=guided_retry_count,
                        )
                        close_reason = "guided_retry_scheduled"
                        close_detail = scheduled_retry.retry_reason or "Requeued same edge with verifier-guided retry."
                        mark_explored = False

            if mark_explored:
                mark_result = self._call_tool(
                    "mark_neighbor_explored",
                    {
                        "doc_id": edge.doc_id,
                        "entity_name": edge.entity_name,
                        "neighbor_name": edge.neighbor_name,
                        "relationship": edge.relationship,
                        "bridges_created": accepted_total,
                    },
                    doc_id=edge.doc_id,
                )
                if isinstance(mark_result, dict) and mark_result.get("error"):
                    close_reason = "coverage_guard" if mark_result.get("stage") == "coverage_guard" else "commit_mark_failed"
                    close_detail = str(mark_result.get("error", "mark_neighbor_explored_failed"))
                self._clear_guided_retry_state(edge)
        finally:
            self._release_focus_internal(edge)

        if mark_explored:
            self._metric_inc("edges_explored", 1)
            if accepted_total > 0:
                self._metric_inc("edges_with_bridges", 1)

        edge_result = EdgeRunResult(
            edge=edge,
            accepted=accepted_total,
            attempted=proposal.attempted,
            rejected=rejected_total,
            budget_exhausted=proposal.budget_exhausted,
            edge_tokens=max(0, proposal.edge_tokens),
            property_attempted=False,
            guided_retry_count=guided_retry_count,
            last_retry_hint=last_retry_hint,
            last_retry_failure_codes=list(last_retry_failure_codes),
        )

        self._emit_event(
            "rlm_edge_summary",
            {
                "edge": edge.as_dict(),
                "accepted": edge_result.accepted,
                "attempted": edge_result.attempted,
                "rejected": edge_result.rejected,
                "budget_exhausted": edge_result.budget_exhausted,
                "edge_tokens": edge_result.edge_tokens,
                "property_attempted": edge_result.property_attempted,
                "signal_tier": proposal.signal.get("tier"),
                "signal_score": proposal.signal.get("score"),
                "no_candidate_calls": proposal.no_candidate_calls,
                "call_limit": int(proposal.edge_call_limit),
                "token_limit": int(proposal.edge_token_limit),
                "worker_prompt_version": WORKER_PROMPT_VERSION,
                "mode": "hybrid",
                "hybrid_scope": self.hybrid_scope,
                "planner_decisions": proposal.planner_decisions,
                "planner_fallbacks": proposal.planner_fallbacks,
                "planner_stop_actions": proposal.planner_stop_actions,
                "worker_calls_made": proposal.worker_calls_made,
                "close_reason": close_reason,
                "close_detail": close_detail,
                "drop_reason_counts": drop_reason_counts,
                "guided_retry_count": guided_retry_count,
                "last_retry_hint": last_retry_hint,
                "last_retry_failure_codes": list(last_retry_failure_codes),
            },
        )
        return edge_result

    def _run_document_parallel(self, doc_id: str, max_edge_attempts: int) -> List[EdgeRunResult]:
        edge_results: List[EdgeRunResult] = []
        edge_attempts = 0

        while edge_attempts < max_edge_attempts:
            edges = self._prioritize_retry_edges(doc_id, self._iter_pending_edges(doc_id))
            if not edges:
                break

            batch_edges = edges[: self.edge_parallel_workers]
            if not batch_edges:
                break

            edge_attempts += len(batch_edges)
            self._metric_inc("parallel_batches", 1)
            self._metric_inc("parallel_workers_used", len(batch_edges))
            self._metric_set_max("commit_queue_size_peak", len(batch_edges))

            prepared_batch: List[PreparedEdgeWork] = []
            for edge in batch_edges:
                prepared = self._prepare_edge_work_parallel(edge)
                if prepared is not None:
                    prepared_batch.append(prepared)

            if not prepared_batch:
                break

            generation_start = time.perf_counter()
            proposals_by_edge: Dict[Tuple[str, str, str, str], EdgeWorkerProposal] = {}
            batch_id = self._metric_value("parallel_batches")
            batch_started_at_iso = datetime.now().isoformat()
            with ThreadPoolExecutor(max_workers=min(self.edge_parallel_workers, len(prepared_batch))) as pool:
                futures: Dict[Any, Tuple[str, str, str, str]] = {}
                future_to_edge: Dict[Any, EdgeKey] = {}
                for item in prepared_batch:
                    future = pool.submit(self._generate_parallel_worker_proposal, item)
                    futures[future] = item.edge.as_tuple()
                    future_to_edge[future] = item.edge
                future_start_times = {future: time.perf_counter() for future in futures}
                warning_windows: Dict[Any, int] = {}
                pending: Set[Any] = set(futures.keys())
                completed = 0
                total = len(futures)

                self._emit_event(
                    "parallel_batch_started",
                    {
                        "doc_id": doc_id,
                        "batch_id": batch_id,
                        "started_at": batch_started_at_iso,
                        "completed": 0,
                        "total": total,
                        "inflight_edges": self._inflight_edges_from_futures(pending, future_to_edge),
                        "oldest_inflight_seconds": 0.0,
                        "elapsed_seconds": 0.0,
                    },
                )

                while pending:
                    done, still_pending = wait(
                        pending,
                        timeout=self.parallel_progress_heartbeat_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    now_ts = time.perf_counter()
                    elapsed = max(0.0, now_ts - generation_start)

                    if done:
                        for future in done:
                            proposal = future.result()
                            proposals_by_edge[proposal.edge.as_tuple()] = proposal
                            pending.remove(future)
                            completed += 1
                            inflight_edges = self._inflight_edges_from_futures(pending, future_to_edge)
                            self._emit_event(
                                "parallel_batch_progress",
                                {
                                    "doc_id": doc_id,
                                    "batch_id": batch_id,
                                    "completed": completed,
                                    "total": total,
                                    "inflight_edges": inflight_edges,
                                    "oldest_inflight_seconds": self._oldest_inflight_seconds(
                                        pending, future_start_times, now_ts
                                    ),
                                    "elapsed_seconds": elapsed,
                                    "completed_edge": proposal.edge.as_dict(),
                                },
                            )
                    else:
                        inflight_edges = self._inflight_edges_from_futures(still_pending, future_to_edge)
                        self._emit_event(
                            "parallel_batch_heartbeat",
                            {
                                "doc_id": doc_id,
                                "batch_id": batch_id,
                                "completed": completed,
                                "total": total,
                                "inflight_edges": inflight_edges,
                                "oldest_inflight_seconds": self._oldest_inflight_seconds(
                                    still_pending, future_start_times, now_ts
                                ),
                                "elapsed_seconds": elapsed,
                            },
                        )

                    for future in list(still_pending):
                        age = max(0.0, now_ts - float(future_start_times.get(future, now_ts)))
                        if age < self.parallel_stuck_warning_seconds:
                            continue
                        current_window = int(age // self.parallel_stuck_warning_seconds)
                        previous_window = warning_windows.get(future, -1)
                        if current_window <= previous_window:
                            continue
                        warning_windows[future] = current_window
                        stuck_edge = future_to_edge.get(future)
                        if stuck_edge is None:
                            continue
                        self._emit_event(
                            "parallel_worker_stuck_warning",
                            {
                                "doc_id": doc_id,
                                "batch_id": batch_id,
                                "completed": completed,
                                "total": total,
                                "edge": stuck_edge.as_dict(),
                                "inflight_edges": self._inflight_edges_from_futures(still_pending, future_to_edge),
                                "worker_inflight_seconds": age,
                                "oldest_inflight_seconds": self._oldest_inflight_seconds(
                                    still_pending, future_start_times, now_ts
                                ),
                                "elapsed_seconds": elapsed,
                                "warning_threshold_seconds": self.parallel_stuck_warning_seconds,
                            },
                        )

                self._emit_event(
                    "parallel_batch_completed",
                    {
                        "doc_id": doc_id,
                        "batch_id": batch_id,
                        "completed": completed,
                        "total": total,
                        "inflight_edges": [],
                        "oldest_inflight_seconds": 0.0,
                        "elapsed_seconds": max(0.0, time.perf_counter() - generation_start),
                    },
                )
            self._metric_add_float("parallel_generation_seconds", time.perf_counter() - generation_start)

            commit_start = time.perf_counter()
            for edge in batch_edges:
                self._emit_event("edge_commit_started", {"edge": edge.as_dict()})
                if not self._is_edge_pending(edge):
                    self._metric_inc("stale_edge_skipped", 1)
                    edge_result = EdgeRunResult(
                        edge=edge,
                        accepted=0,
                        attempted=0,
                        rejected=0,
                        budget_exhausted=False,
                        edge_tokens=0,
                        property_attempted=False,
                    )
                    self._emit_event(
                        "rlm_edge_summary",
                        {
                            "edge": edge.as_dict(),
                            "accepted": 0,
                            "attempted": 0,
                            "rejected": 0,
                            "budget_exhausted": False,
                            "edge_tokens": 0,
                            "property_attempted": False,
                            "signal_tier": "n/a",
                            "signal_score": 0.0,
                            "no_candidate_calls": 0,
                            "call_limit": 0,
                            "token_limit": 0,
                            "worker_prompt_version": WORKER_PROMPT_VERSION,
                            "mode": "hybrid",
                            "hybrid_scope": self.hybrid_scope,
                            "planner_decisions": 0,
                            "planner_fallbacks": 0,
                            "planner_stop_actions": 0,
                            "worker_calls_made": 0,
                            "close_reason": "stale_edge_skipped",
                            "close_detail": "Edge already explored before commit.",
                            "drop_reason_counts": self._new_drop_reason_counts(),
                        },
                    )
                    self._emit_event(
                        "edge_commit_finished",
                        {
                            "edge": edge.as_dict(),
                            "close_reason": "stale_edge_skipped",
                            "accepted": 0,
                        },
                    )
                    edge_results.append(edge_result)
                    continue

                proposal = proposals_by_edge.get(edge.as_tuple())
                if proposal is None:
                    self._metric_inc("stale_edge_skipped", 1)
                    self._emit_event(
                        "edge_commit_finished",
                        {
                            "edge": edge.as_dict(),
                            "close_reason": "stale_edge_skipped",
                            "accepted": 0,
                            "detail": "No proposal generated for edge.",
                        },
                    )
                    continue

                edge_result = self._commit_edge_worker_proposal(proposal)
                edge_results.append(edge_result)
                self._emit_event(
                    "edge_commit_finished",
                    {
                        "edge": edge.as_dict(),
                        "close_reason": "committed",
                        "accepted": edge_result.accepted,
                    },
                )
            self._metric_add_float("serial_commit_seconds", time.perf_counter() - commit_start)

        return edge_results

    def _run_edge(self, edge: EdgeKey) -> EdgeRunResult:
        start_tokens = self.reconciler.tokens_used
        start_planner_decisions = self._metric_value("planner_decisions")
        start_planner_fallbacks = self._metric_value("planner_fallbacks")
        start_planner_stops = self._metric_value("planner_stop_actions")
        self._clear_guided_retry_state(edge)

        source_context = self._call_tool(
            "get_entity_context",
            {"entity_name": edge.entity_name, "doc_id": edge.doc_id},
        )
        if not isinstance(source_context, dict) or "error" in source_context:
            source_context = {
                "entity": edge.entity_name,
                "doc_id": edge.doc_id,
                "qa_pairs": [],
                "roles": [],
                "states": [],
                "relationships": {},
            }

        focus_result = self._call_tool("begin_neighbor_focus", edge.as_dict(), doc_id=edge.doc_id)
        if isinstance(focus_result, dict) and focus_result.get("error"):
            return EdgeRunResult(edge=edge, accepted=0, attempted=0, rejected=0, budget_exhausted=False, edge_tokens=0, property_attempted=False)

        coverage = self._call_tool("plan_neighbor_doc_coverage", edge.as_dict(), doc_id=edge.doc_id)
        if not isinstance(coverage, dict) or "error" in coverage:
            return EdgeRunResult(edge=edge, accepted=0, attempted=0, rejected=0, budget_exhausted=False, edge_tokens=0, property_attempted=False)

        mandatory_docs = list(coverage.get("mandatory_docs", []))
        optional_docs = self._select_optional_doc_ids(edge, coverage.get("optional_fuzzy_docs", []))

        mandatory_contexts = self._collect_contexts_for_docs(edge.neighbor_name, mandatory_docs, source_doc=edge.doc_id)
        optional_contexts = self._collect_contexts_for_docs(edge.neighbor_name, optional_docs, source_doc=edge.doc_id) if optional_docs else []

        packet = self._build_edge_packet(edge, coverage, source_context, mandatory_contexts, optional_contexts)
        signal = self._score_edge_signal(coverage, mandatory_contexts)
        edge_call_limit, edge_token_limit = self._low_signal_budgets(signal)
        if signal.get("tier") == "low":
            self.reconciler.rlm_metrics["low_signal_edges"] = self.reconciler.rlm_metrics.get("low_signal_edges", 0) + 1

        accepted_total = 0
        attempted_total = 0
        rejected_total = 0
        budget_exhausted = False
        call_count = 0
        no_candidate_calls = 0
        drop_reason_counts = self._new_drop_reason_counts()
        last_failure_signature: Optional[Tuple[Tuple[Any, ...], ...]] = None
        last_failure_state: Optional[Tuple[int, bool, bool, int]] = None
        close_reason = "call_limit_reached"
        close_detail = f"Reached edge call limit ({edge_call_limit})."
        last_worker_output: Optional[WorkerOutput] = None
        last_worker_notes = ""
        last_candidates_count = 0
        last_rejected_delta = 0
        last_edge_reason = ""
        last_run_signature: Optional[Tuple[Any, ...]] = None
        active_retry_brief: Optional[EdgeRetryBrief] = None
        guided_retry_count = 0
        last_retry_hint = ""
        last_retry_failure_codes: List[str] = []

        depth = 0
        include_optional = False
        edge_budget_spend = 0

        while call_count < edge_call_limit:
            remaining_edge_tokens = edge_token_limit - edge_budget_spend
            if remaining_edge_tokens <= 0:
                budget_exhausted = True
                close_reason = "budget_exhausted"
                close_detail = "Per-edge token budget exhausted before worker call."
                break

            viability = self._evaluate_edge_viability(packet, include_optional=include_optional)
            path_proofs = viability["path_proofs"]
            render_input = RenderInput(
                edge=edge,
                proofs=path_proofs,
                constraints=packet.constraints,
            ) if path_proofs else None

            action = self._plan_edge_action(
                edge=edge,
                current_depth=depth,
                current_include_optional=include_optional,
                call_count=call_count,
                remaining_edge_tokens=remaining_edge_tokens,
                accepted_total=accepted_total,
                attempted_total=attempted_total,
                rejected_total=rejected_total,
                no_candidate_calls=no_candidate_calls,
                has_optional_contexts=bool(packet.optional_contexts),
                path_proof_count=viability["path_proof_count"],
                mandatory_context_count=viability["mandatory_context_count"],
                optional_context_count=viability["optional_context_count"],
                cross_doc_fact_count=viability["cross_doc_fact_count"],
                last_worker_notes=last_worker_notes,
                last_candidates_count=last_candidates_count,
                last_rejected_delta=last_rejected_delta,
                last_close_reason=last_edge_reason,
            )

            if action.get("action") == "stop_edge":
                coverage_snapshot = self._call_tool("plan_neighbor_doc_coverage", edge.as_dict(), doc_id=edge.doc_id)
                pending_docs = []
                if isinstance(coverage_snapshot, dict):
                    pending_docs = list(coverage_snapshot.get("pending_mandatory_docs", []) or [])
                if pending_docs:
                    self._planner_fallback(
                        "stop_blocked_pending_docs",
                        {"edge": edge.as_dict(), "pending": pending_docs},
                        decision_type="edge_action",
                    )
                else:
                    self._emit_worker_call(
                        edge=edge,
                        call_index=call_count + 1,
                        depth=depth,
                        include_optional=include_optional,
                        remaining_edge_tokens=remaining_edge_tokens,
                        planner_action=str(action.get("action", "stop_edge")),
                        planner_source=str(action.get("source", "")),
                        planner_reason=str(action.get("reason", "")),
                        worker_invoked=False,
                        retry_brief=active_retry_brief,
                    )
                    self._metric_inc("planner_stop_actions", 1)
                    close_reason = "planner_stop"
                    close_detail = (
                        f"Planner stop accepted (reason='{str(action.get('reason', '')).strip()}')."
                    )
                    break

            run_depth = int(action.get("depth", depth))
            run_include_optional = bool(action.get("include_optional", include_optional)) and bool(packet.optional_contexts)
            depth = run_depth
            include_optional = run_include_optional
            viability = self._evaluate_edge_viability(packet, include_optional=run_include_optional)
            path_proofs = viability["path_proofs"]
            render_input = RenderInput(
                edge=edge,
                proofs=path_proofs,
                constraints=packet.constraints,
            ) if path_proofs else None

            # Planner can request run_worker, but hard viability guard is fail-closed.
            no_path_proofs = viability["path_proof_count"] == 0
            can_optional_probe = (
                call_count == 0
                and not run_include_optional
                and not packet.mandatory_contexts
                and bool(packet.optional_contexts)
                and run_depth < self.edge_max_depth
            )
            if no_path_proofs:
                self._metric_inc("planner_actions_overridden", 1)
                self._planner_fallback(
                    "run_worker_overridden_no_path",
                    {
                        "edge": edge.as_dict(),
                        "mandatory_contexts": viability["mandatory_context_count"],
                        "optional_contexts": viability["optional_context_count"],
                        "cross_doc_facts": viability["cross_doc_fact_count"],
                    },
                    decision_type="edge_action",
                )
                if can_optional_probe:
                    include_optional = True
                    depth = min(self.edge_max_depth, max(run_depth + 1, 1))
                    self._metric_inc("recursive_invocations", 1)
                    continue

                self._emit_worker_call(
                    edge=edge,
                    call_index=call_count + 1,
                    depth=run_depth,
                    include_optional=run_include_optional,
                    remaining_edge_tokens=remaining_edge_tokens,
                    planner_action="run_worker",
                    planner_source=str(action.get("source", "")),
                    planner_reason=str(action.get("reason", "")),
                    worker_invoked=False,
                    retry_brief=active_retry_brief,
                )
                self._metric_inc("edges_skipped_no_path", 1)
                close_reason = "no_viable_path_proof"
                close_detail = (
                    "No viable path proofs before worker call "
                    f"(mandatory_contexts={viability['mandatory_context_count']}, "
                    f"optional_contexts={viability['optional_context_count']}, "
                    f"cross_doc_facts={viability['cross_doc_fact_count']})."
                )
                break

            run_signature = (
                int(run_depth),
                bool(run_include_optional),
                viability["evidence_signature"],
            )
            if call_count > 0 and run_signature == last_run_signature and accepted_total == 0 and active_retry_brief is None:
                self._metric_inc("planner_actions_overridden", 1)
                self._metric_inc("calls_blocked_no_progress", 1)
                self._planner_fallback(
                    "run_worker_overridden_no_progress",
                    {
                        "edge": edge.as_dict(),
                        "run_depth": run_depth,
                        "run_include_optional": run_include_optional,
                        "evidence_signature": list(viability["evidence_signature"]),
                    },
                    decision_type="edge_action",
                )
                self._emit_worker_call(
                    edge=edge,
                    call_index=call_count + 1,
                    depth=run_depth,
                    include_optional=run_include_optional,
                    remaining_edge_tokens=remaining_edge_tokens,
                    planner_action="run_worker",
                    planner_source=str(action.get("source", "")),
                    planner_reason=str(action.get("reason", "")),
                    worker_invoked=False,
                    retry_brief=active_retry_brief,
                )
                close_reason = "no_progress_repeat"
                close_detail = (
                    "Blocked repeated worker call with unchanged evidence state "
                    f"(depth={run_depth}, include_optional={run_include_optional}, "
                    f"path_proofs={viability['path_proof_count']})."
                )
                break
            last_run_signature = run_signature

            call_index = call_count + 1
            current_retry_brief = active_retry_brief
            if current_retry_brief is not None:
                self._metric_inc("guided_retry_started", 1)
                self._emit_guided_retry_event(
                    "guided_retry_started",
                    current_retry_brief,
                    mode="hybrid",
                    call_index=call_index,
                    depth=run_depth,
                    include_optional=run_include_optional,
                )
            self._emit_worker_call(
                edge=edge,
                call_index=call_index,
                depth=run_depth,
                include_optional=run_include_optional,
                remaining_edge_tokens=remaining_edge_tokens,
                planner_action=str(action.get("action", "run_worker")),
                planner_source=str(action.get("source", "")),
                planner_reason=str(action.get("reason", "")),
                worker_invoked=True,
                retry_brief=current_retry_brief,
            )
            worker_output = self.worker.generate(
                packet=packet,
                depth=run_depth,
                include_optional=run_include_optional,
                max_response_tokens=min(1400, remaining_edge_tokens),
                render_input=render_input,
                retry_brief=current_retry_brief,
            )
            active_retry_brief = None
            call_count = call_index
            last_worker_output = worker_output
            edge_budget_spend += max(0, int(getattr(worker_output, "budgeted_token_usage", 0)))

            attempted_total += len(worker_output.candidates)
            if not worker_output.candidates:
                no_candidate_calls += 1

            accepted, rejected, cap_hit, retry_brief = self._apply_candidates(
                packet,
                worker_output.candidates,
                path_proofs=path_proofs,
                drop_reason_counts=drop_reason_counts,
                retry_attempt=guided_retry_count + 1,
            )
            accepted_total += accepted
            rejected_total += rejected
            self._emit_worker_result(
                edge=edge,
                call_index=call_index,
                worker_output=worker_output,
                accepted_delta=accepted,
                rejected_delta=rejected,
            )
            last_worker_notes = str(worker_output.notes or "")
            last_candidates_count = int(len(worker_output.candidates))
            last_rejected_delta = int(rejected)
            last_edge_reason = "accepted_bridge" if accepted > 0 else "worker_call_no_accept"

            if edge_budget_spend >= edge_token_limit:
                budget_exhausted = True
                close_reason = "budget_exhausted"
                close_detail = "Per-edge token budget exhausted after worker call."
                break

            if accepted_total > 0:
                if current_retry_brief is not None:
                    self._metric_inc("guided_retry_succeeded", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_succeeded",
                        current_retry_brief,
                        mode="hybrid",
                        accepted_total=accepted_total,
                    )
                close_reason = "accepted_bridge"
                if cap_hit:
                    close_detail = (
                        f"Accepted {accepted_total} bridge(s); hit per-edge acceptance cap "
                        f"({getattr(self.reconciler, 'edge_max_accepted_bridges', 10)})."
                    )
                else:
                    close_detail = f"Accepted {accepted_total} bridge(s)."
                break

            can_recurse = run_depth < self.edge_max_depth
            state_key = (run_depth, run_include_optional, bool(render_input), len(path_proofs))
            candidate_signature = self._candidate_signature(worker_output.candidates)
            repeated_failure = (
                accepted == 0
                and bool(worker_output.candidates)
                and last_failure_state == state_key
                and last_failure_signature == candidate_signature
            )
            if accepted == 0 and worker_output.candidates:
                last_failure_state = state_key
                last_failure_signature = candidate_signature
            elif accepted > 0:
                last_failure_state = None
                last_failure_signature = None

            if accepted == 0 and retry_brief is not None:
                last_retry_hint = retry_brief.retry_hint
                last_retry_failure_codes = list(retry_brief.failure_codes)
                retry_signature = self._guided_retry_signature(run_signature, candidate_signature, retry_brief)
                if retry_brief.retry_hint == "stop_edge" or not retry_brief.retryable:
                    close_reason = "verifier_stop_edge"
                    close_detail = retry_brief.retry_reason or "Verifier indicated this edge should stop."
                    break
                if guided_retry_count >= GUIDED_RETRY_LIMIT:
                    self._metric_inc("guided_retry_exhausted", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_exhausted",
                        retry_brief,
                        mode="hybrid",
                        guided_retry_count=guided_retry_count,
                        reason="retry_limit",
                    )
                    close_reason = "guided_retry_exhausted"
                    close_detail = f"Reached guided retry limit ({GUIDED_RETRY_LIMIT}) for this edge."
                    break
                if not self._guided_retry_change_room(packet, path_proofs, run_include_optional, can_recurse, retry_brief):
                    self._metric_inc("guided_retry_exhausted", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_exhausted",
                        retry_brief,
                        mode="hybrid",
                        guided_retry_count=guided_retry_count,
                        reason="no_alternate_evidence",
                    )
                    close_reason = "guided_retry_exhausted"
                    close_detail = "Verifier suggested retry, but no alternate same-edge evidence remains."
                    break
                if self._last_guided_retry_signature(edge) == retry_signature:
                    self._metric_inc("guided_retry_blocked_no_progress", 1)
                    self._emit_guided_retry_event(
                        "guided_retry_blocked_no_progress",
                        retry_brief,
                        mode="hybrid",
                        guided_retry_count=guided_retry_count,
                    )
                    close_reason = "guided_retry_blocked_no_progress"
                    close_detail = "Blocked guided retry with unchanged verifier hint and evidence state."
                    break

                guided_retry_count += 1
                self._set_guided_retry_count(edge, guided_retry_count)
                self._set_last_guided_retry_signature(edge, retry_signature)
                active_retry_brief = EdgeRetryBrief(
                    edge=retry_brief.edge,
                    failure_codes=list(retry_brief.failure_codes),
                    retryable=retry_brief.retryable,
                    retry_hint=retry_brief.retry_hint,
                    retry_reason=retry_brief.retry_reason,
                    rejected_question=retry_brief.rejected_question,
                    rejected_reverse_question=retry_brief.rejected_reverse_question,
                    retry_attempt=guided_retry_count,
                )
                self._metric_inc("guided_retry_scheduled", 1)
                self._emit_guided_retry_event(
                    "guided_retry_scheduled",
                    active_retry_brief,
                    mode="hybrid",
                    guided_retry_count=guided_retry_count,
                )
                depth, include_optional = self._guided_retry_next_state(
                    run_depth,
                    run_include_optional,
                    active_retry_brief,
                    packet,
                )
                continue

            # Deterministic fallback progression for next iteration.
            if can_recurse and not run_include_optional and packet.optional_contexts and (
                worker_output.need_recursion or not worker_output.candidates
            ):
                include_optional = True
                depth = min(self.edge_max_depth, run_depth + 1)
                self._metric_inc("recursive_invocations", 1)
            elif can_recurse and worker_output.need_recursion and run_include_optional:
                new_depth = min(self.edge_max_depth, run_depth + 1)
                if new_depth > run_depth:
                    depth = new_depth
                    self._metric_inc("recursive_invocations", 1)

            if repeated_failure:
                self._metric_inc("repeated_failure_stops", 1)
                close_reason = "repeated_failure"
                close_detail = "Repeated candidate signature failure at same depth/state."
                break

            if not worker_output.candidates:
                # No candidates and no additional deterministic progression.
                if include_optional != run_include_optional or depth != run_depth:
                    continue
                close_reason = "no_candidates_no_progression"
                close_detail = (
                    f"No candidates (notes='{worker_output.notes}', parse_stage={worker_output.parse_stage}, "
                    f"parse_attempts={worker_output.parse_attempts}, "
                    f"accepted_delta={accepted}, rejected_delta={rejected})."
                )
                break

            if accepted == 0:
                if signal.get("probe_only"):
                    close_reason = "probe_only_no_success"
                    close_detail = "Probe-only edge produced no accepted candidates."
                    break
                continue

            continue

        if close_reason == "call_limit_reached":
            tail = ""
            if last_worker_output is not None:
                tail = (
                    f" last_parse_stage={last_worker_output.parse_stage};"
                    f" last_parse_attempts={last_worker_output.parse_attempts};"
                    f" last_notes={last_worker_output.notes!r}"
                )
            close_detail = (
                f"Reached edge call limit ({edge_call_limit}); accepted_total={accepted_total}; "
                f"rejected_total={rejected_total}; no_candidate_calls={no_candidate_calls};{tail}"
            )

        mark_args = {
            "doc_id": edge.doc_id,
            "entity_name": edge.entity_name,
            "neighbor_name": edge.neighbor_name,
            "relationship": edge.relationship,
            "bridges_created": accepted_total,
        }
        _mark_result = self._call_tool("mark_neighbor_explored", mark_args, doc_id=edge.doc_id)

        edge_tokens = self.reconciler.tokens_used - start_tokens
        self.reconciler.rlm_metrics["edges_explored"] += 1
        if accepted_total > 0:
            self.reconciler.rlm_metrics["edges_with_bridges"] += 1

        edge_result = EdgeRunResult(
            edge=edge,
            accepted=accepted_total,
            attempted=attempted_total,
            rejected=rejected_total,
            budget_exhausted=budget_exhausted,
            edge_tokens=max(0, edge_tokens),
            property_attempted=False,
            guided_retry_count=guided_retry_count,
            last_retry_hint=last_retry_hint,
            last_retry_failure_codes=list(last_retry_failure_codes),
        )
        self._clear_guided_retry_state(edge)

        planner_decisions_delta = self._metric_value("planner_decisions") - start_planner_decisions
        planner_fallbacks_delta = self._metric_value("planner_fallbacks") - start_planner_fallbacks
        planner_stops_delta = self._metric_value("planner_stop_actions") - start_planner_stops

        self._emit_event(
            "rlm_edge_summary",
            {
                "edge": edge.as_dict(),
                "accepted": edge_result.accepted,
                "attempted": edge_result.attempted,
                "rejected": edge_result.rejected,
                "budget_exhausted": edge_result.budget_exhausted,
                "edge_tokens": edge_result.edge_tokens,
                "property_attempted": edge_result.property_attempted,
                "signal_tier": signal.get("tier"),
                "signal_score": signal.get("score"),
                "no_candidate_calls": no_candidate_calls,
                "call_limit": edge_call_limit,
                "token_limit": edge_token_limit,
                "worker_prompt_version": WORKER_PROMPT_VERSION,
                "mode": "hybrid",
                "hybrid_scope": self.hybrid_scope,
                "planner_decisions": planner_decisions_delta,
                "planner_fallbacks": planner_fallbacks_delta,
                "planner_stop_actions": planner_stops_delta,
                "worker_calls_made": call_count,
                "close_reason": close_reason,
                "close_detail": close_detail if close_detail else (
                    str(last_worker_output.notes) if last_worker_output is not None and last_worker_output.notes else ""
                ),
                "drop_reason_counts": drop_reason_counts,
                "guided_retry_count": guided_retry_count,
                "last_retry_hint": last_retry_hint,
                "last_retry_failure_codes": list(last_retry_failure_codes),
            },
        )

        return edge_result

    def run_document(self, doc_id: str) -> Dict[str, Any]:
        start_planner_decisions = self._metric_value("planner_decisions")
        start_planner_fallbacks = self._metric_value("planner_fallbacks")
        start_planner_stops = self._metric_value("planner_stop_actions")

        plan = self._call_tool("plan_document_exploration", {"doc_id": doc_id}, doc_id=doc_id)
        if isinstance(plan, dict) and "error" in plan:
            return {
                "entity": doc_id,
                "iterations": 0,
                "tool_calls": 1,
                "bridges_created": 0,
                "mode": "hybrid",
                "hybrid_scope": self.hybrid_scope,
                "error": plan.get("error"),
            }

        self.reconciler.rlm_metrics["docs_attempted"] += 1

        max_edge_attempts = max(10, len(self._iter_pending_edges(doc_id)) * 3)
        if self.edge_parallel_enabled and self.edge_parallel_workers > 1 and self.hybrid_scope == "doc_edge":
            edge_results = self._run_document_parallel(doc_id=doc_id, max_edge_attempts=max_edge_attempts)
        else:
            edge_results: List[EdgeRunResult] = []
            edge_attempts = 0
            while edge_attempts < max_edge_attempts:
                edges = self._prioritize_retry_edges(doc_id, self._iter_pending_edges(doc_id))
                if not edges:
                    break
                edge_attempts += 1
                selected_edge = edges[0] if self._peek_pending_retry_brief(edges[0]) is not None else self._choose_edge_for_doc(doc_id, edges)
                edge_results.append(self._run_edge(selected_edge))

        remaining_edges = self._iter_pending_edges(doc_id)
        if remaining_edges:
            logger.warning(
                "Hybrid document loop stopped due to safety cap [doc=%s pending_edges=%s cap=%s]",
                doc_id,
                len(remaining_edges),
                max_edge_attempts,
            )

        status = self._call_tool("get_doc_exploration_status", {"doc_id": doc_id}, doc_id=doc_id)
        pending_count = int(status.get("pending_count", 0)) if isinstance(status, dict) else 0
        bridges_created = sum(er.accepted for er in edge_results)

        if pending_count == 0:
            _doc_mark = self._call_tool(
                "mark_document_explored",
                {"doc_id": doc_id, "num_bridges_created": bridges_created},
                doc_id=doc_id,
            )
            self.reconciler.rlm_metrics["docs_completed"] += 1

        self.reconciler.entities_explored += 1

        planner_decisions_delta = self._metric_value("planner_decisions") - start_planner_decisions
        planner_fallbacks_delta = self._metric_value("planner_fallbacks") - start_planner_fallbacks
        planner_stops_delta = self._metric_value("planner_stop_actions") - start_planner_stops

        summary = {
            "entity": doc_id,
            "iterations": len(edge_results),
            "tool_calls": 0,
            "bridges_created": bridges_created,
            "mode": "hybrid",
            "hybrid_scope": self.hybrid_scope,
            "pending_edges": pending_count,
            "planner_decisions": planner_decisions_delta,
            "planner_fallbacks": planner_fallbacks_delta,
            "planner_stop_actions": planner_stops_delta,
            "edge_results": [
                {
                    "edge": er.edge.as_dict(),
                    "accepted": er.accepted,
                    "attempted": er.attempted,
                    "rejected": er.rejected,
                    "budget_exhausted": er.budget_exhausted,
                    "edge_tokens": er.edge_tokens,
                    "property_attempted": er.property_attempted,
                    "guided_retry_count": er.guided_retry_count,
                    "last_retry_hint": er.last_retry_hint,
                    "last_retry_failure_codes": list(er.last_retry_failure_codes),
                    "worker_prompt_version": WORKER_PROMPT_VERSION,
                }
                for er in edge_results
            ],
        }

        self._emit_event("rlm_doc_summary", summary)
        return summary

    def run_corpus(self, max_documents: Optional[int] = None) -> Dict[str, Any]:
        docs_processed = 0
        doc_results: List[Dict[str, Any]] = []

        while True:
            if max_documents is not None and docs_processed >= max_documents:
                break
            if "max_tokens" in self.reconciler.budget and self.reconciler.tokens_used >= self.reconciler.budget["max_tokens"]:
                break

            plan = self._call_tool(
                "plan_corpus_exploration",
                {"strategy": "max_pending_neighbors", "limit": 20, "include_completed": False},
            )
            if not isinstance(plan, dict):
                break

            doc_id = self._choose_doc_for_corpus(plan)
            if not doc_id:
                break

            queue = plan.get("queue", [])
            queue_row = None
            if isinstance(queue, list):
                for item in queue:
                    if isinstance(item, dict) and str(item.get("doc_id")) == str(doc_id):
                        queue_row = item
                        break
            if queue_row and str(queue_row.get("status")) == "completed":
                self._planner_fallback(
                    "selected_completed_doc",
                    {"doc_id": doc_id},
                    decision_type="corpus_doc_selection",
                )
                # Deterministic fallback to planner's next_doc.
                next_doc = plan.get("next_doc", {})
                if isinstance(next_doc, dict) and next_doc.get("doc_id"):
                    doc_id = str(next_doc["doc_id"])
                else:
                    break

            doc_result = self.run_document(doc_id)
            doc_results.append(doc_result)
            docs_processed += 1

        final_stats = self.reconciler.tools.get_bridge_statistics()
        return {
            "mode": "hybrid",
            "hybrid_scope": self.hybrid_scope,
            "documents_explored": docs_processed,
            "total_bridges": final_stats.get("total_bridges", 0),
            "tokens_used": self.reconciler.tokens_used,
            "input_tokens": self.reconciler.input_tokens,
            "output_tokens": self.reconciler.output_tokens,
            "rlm_metrics": dict(self.reconciler.rlm_metrics),
            "document_results": doc_results,
        }
