"""RLM-style deterministic root scheduler with recursive edge workers.

This module keeps exploration state in Python and only sends compact edge-local
packets to worker model calls to control token growth.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .tools import normalize_similarity_text, tokenize_similarity_text

logger = logging.getLogger(__name__)


WORKER_PROMPT_VERSION = "legacy_rich_4x4_v1"

WORKER_PRE_OUTPUT_CHECKLIST = """
Pre-output checklist (must pass all):
1. Forward question requires resolving hop-1 output (#1) before hop-2.
2. Reverse question truly inverts the forward mapping, not an independent fact.
3. Reasoning includes both:
   - Sub-Q1: [source entity] -> [relationship] -> #1 (doc_X)
   - Sub-Q2: #1 -> [second-hop fact] -> answer (doc_Y)
4. answers/reverse_answers are grounded refs present in provided evidence.
5. Avoid answer leakage and one-hop restatements.
"""

WORKER_LEGACY_EXAMPLES = """
Legacy examples (4 good + 4 bad):

GOOD #1
Forward: "Where did Master Aldric's patron build a chapel?"
Reverse: "Whose patron built a chapel in Bruges?"
Reasoning:
Sub-Q1: Aldric -> patron -> Lady Margaux (doc_5)
Sub-Q2: Lady Margaux -> built chapel -> Bruges (doc_11)

GOOD #2
Forward: "Who founded the Abbey of Saint Benedict's sister monastery?"
Reverse: "Abbess Hildegard founded the sister monastery of which abbey?"
Reasoning:
Sub-Q1: Abbey of Saint Benedict -> sister monastery -> Convent of Holy Cross (doc_15)
Sub-Q2: Convent of Holy Cross -> founded by -> Abbess Hildegard (doc_34)

GOOD #3
Forward: "Where did Silversmith Marco's apprentice open their workshop?"
Reverse: "Whose apprentice opened a workshop in Venice?"
Reasoning:
Sub-Q1: Marco -> apprentice -> Paolo (doc_8)
Sub-Q2: Paolo -> workshop location -> Venice (doc_23)

GOOD #4
Forward: "When was the forge where Master Aldric trained established?"
Reverse: "Who trained at the forge established in 1203?"
Reasoning:
Sub-Q1: Aldric -> trained at -> Ironworks of Ghent (doc_5)
Sub-Q2: Ironworks of Ghent -> established -> 1203 (doc_29)

BAD #1 (NOT_CHAIN conjunction)
Forward: "Which NATO member country lacks a traditional army but participated in the ideological conflict with the Warsaw Pact?"
Reason: independent facts; hop-2 does not depend on hop-1 output.

BAD #2 (parallel hub facts / NOT_CHAIN + REVERSE_INVALID)
Forward: "Who was the queen of Lotharingia?" -> Teutberga
Reverse: "Who was the king of Lotharingia?" -> Lothair II
Reason: two direct lookups on the same hub entity, not inversion of one mapping.

BAD #3 (REVERSE_INVALID)
Forward: "Where did Master Aldric's patron build a chapel?"
Reverse: "Where did Master Aldric's patron build a chapel?"
Reason: reverse is identical to forward; no inversion.

BAD #4 (ANSWER_LEAK / TOO_TRIVIAL)
Forward: "What defense alliance comprising nuclear-armed nations was established to counter Warsaw Pact?"
Reason: question text leaks answer directly.
"""

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


@dataclass
class RenderInput:
    """Strict renderer input: only path proofs, no free context search."""

    edge: EdgeKey
    proofs: List[PathProof]
    constraints: Dict[str, Any]


@dataclass
class WorkerOutput:
    """Normalized output from one worker invocation."""

    status: str
    candidates: List[Dict[str, Any]]
    need_recursion: bool
    notes: str
    parse_stage: str
    raw_preview: str


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

    def _build_messages(
        self,
        packet: EdgePacket,
        depth: int,
        include_optional: bool,
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

        system_prompt = (
            "You are an edge-local bridge generator. "
            "You receive a source entity context from one doc and neighbor contexts from other docs. "
            "Propose only valid two-hop bridge candidates that chain source-side fact -> neighbor-side fact "
            "across at least two documents.\n\n"
            f"Prompt version: {WORKER_PROMPT_VERSION}\n\n"
            f"{WORKER_PRE_OUTPUT_CHECKLIST}\n"
            f"{WORKER_LEGACY_EXAMPLES}\n"
            "Return JSON only with keys: status, candidates, need_recursion, notes."
        )

        user_prompt = (
            "Generate high-quality bridge candidates for this edge packet.\n"
            "Each candidate MUST include keys: question, answers, reverse_question, "
            "reverse_answers, source_docs, reasoning.\n"
            "answers and reverse_answers MUST use doc_x::ey refs from contexts.\n"
            "Do not emit fixed-size filler lists; emit only defensible candidates from evidence.\n"
            "If no good candidate, return empty candidates and need_recursion boolean.\n\n"
            f"EDGE_PACKET:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_render_messages(
        self,
        render_input: RenderInput,
    ) -> List[Dict[str, str]]:
        payload = {
            "edge": render_input.edge.as_dict(),
            "path_proofs": [
                {
                    "source_fact": proof.source_fact,
                    "neighbor_fact": proof.neighbor_fact,
                    "path_docs": proof.path_docs,
                    "target_refs": proof.target_refs,
                }
                for proof in render_input.proofs
            ],
            "constraints": render_input.constraints,
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
            "Return JSON only with keys: status, candidates, need_recursion, notes."
        )
        user_prompt = (
            "Render bridge candidates from these path proofs.\n"
            "Each candidate MUST include keys: question, answers, reverse_question, "
            "reverse_answers, source_docs, reasoning.\n"
            "answers and reverse_answers MUST be refs appearing in path_proofs target_refs or source facts.\n"
            "Do not emit fixed-size filler lists; emit only defensible candidates from evidence.\n"
            "If no good candidate, return empty candidates.\n\n"
            f"RENDER_INPUT:\n{json.dumps(payload, ensure_ascii=False)}"
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

    def _parse_worker_output(self, raw_text: str, parse_stage: str) -> WorkerOutput:
        parsed = self.reconciler._extract_json_from_text(raw_text)
        status = str(parsed.get("status", "ok"))
        candidates = self._normalize_candidates(parsed.get("candidates", []))
        need_recursion = bool(parsed.get("need_recursion", False))
        notes = str(parsed.get("notes", ""))
        preview = self.reconciler._preview_text(raw_text)
        return WorkerOutput(
            status=status,
            candidates=candidates,
            need_recursion=need_recursion,
            notes=notes,
            parse_stage=parse_stage,
            raw_preview=preview,
        )

    def generate(
        self,
        packet: EdgePacket,
        depth: int,
        include_optional: bool,
        max_response_tokens: int,
        render_input: Optional[RenderInput] = None,
    ) -> WorkerOutput:
        messages = (
            self._build_render_messages(render_input)
            if render_input is not None
            else self._build_messages(packet, depth, include_optional)
        )

        message = self.reconciler._call_model_for_stage(
            messages=messages,
            model_name=self.model_name,
            max_tokens=max(256, int(max_response_tokens)),
            temperature=0.2,
            stage="worker",
            response_format={"type": "json_object"},
        )
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
            return self._parse_worker_output(raw_text, parse_stage="initial")
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

        repaired_message = self.reconciler._call_model_for_stage(
            messages=repair_messages,
            model_name=self.model_name,
            max_tokens=max(256, int(max_response_tokens)),
            temperature=0.0,
            stage="worker",
            response_format={"type": "json_object"},
        )
        repaired_text, _repaired_source = self.reconciler._extract_message_content_with_source(repaired_message)

        try:
            return self._parse_worker_output(repaired_text, parse_stage="repair_retry")
        except Exception as retry_error:
            logger.error(
                "Worker parse failed [stage=repair_retry edge=%s] error=%s preview=%s",
                packet.edge.as_tuple(),
                retry_error,
                self.reconciler._preview_text(repaired_text),
            )
            return WorkerOutput(
                status="parse_error",
                candidates=[],
                need_recursion=False,
                notes=f"Worker parse error: {retry_error}",
                parse_stage="repair_retry",
                raw_preview=self.reconciler._preview_text(repaired_text),
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

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        callback = self.reconciler.output_callback
        if callback:
            callback(event_type, data)

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

        proofs: List[PathProof] = []
        seen = set()
        for source_qa in source_links:
            source_refs = self._extract_valid_refs(source_qa)
            source_docs = self._refs_to_docs(source_refs) or {packet.edge.doc_id}
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
                        normalize_similarity_text(source_qa.get("question", "")),
                        normalize_similarity_text(neighbor_qa.get("question", "")),
                        tuple(sorted(target_refs)),
                        tuple(path_docs),
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)

                    proofs.append(
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
                        )
                    )
                    if len(proofs) >= max_paths:
                        return proofs
        return proofs

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
            },
            budget={
                "max_depth": self.edge_max_depth,
                "max_calls": self.edge_max_calls,
                "max_tokens": self.edge_max_tokens,
            },
        )

    def _collect_contexts_for_docs(self, neighbor_name: str, doc_ids: List[str], source_doc: str) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        for target_doc in doc_ids:
            ctx = self._call_tool(
                "get_entity_context",
                {"entity_name": neighbor_name, "doc_id": target_doc},
                doc_id=source_doc,
            )
            if isinstance(ctx, dict) and "error" not in ctx:
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
                response_format={"type": "json_object"},
            )
            raw_text, _source = self.reconciler._extract_message_content_with_source(message)
            parsed = self.reconciler._extract_json_from_text(raw_text)
            selected = parsed.get("selected_doc_ids", [])
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

    def _sanitize_candidate_refs(self, packet: EdgePacket, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_source_docs = candidate.get("source_docs", [])
        allowed_docs = [str(d).strip() for d in packet.source_docs if str(d).strip()]
        if not allowed_docs:
            return None

        allowed_docs_set = set(allowed_docs)
        candidate_docs = [str(d).strip() for d in raw_source_docs if str(d).strip() in allowed_docs_set]
        source_docs_for_resolution = self._dedupe_preserve_order(candidate_docs + allowed_docs)
        if len(source_docs_for_resolution) < 2:
            return None

        context_ref_map = self._build_context_answer_ref_map(packet)
        allowed_refs = {
            ref
            for refs in context_ref_map.values()
            for ref in refs
            if isinstance(ref, str) and "::" in ref
        }
        if not allowed_refs:
            return None

        resolved_answers: List[str] = []
        unresolved: List[str] = []
        for answer in candidate.get("answers", []):
            resolved = self._resolve_answer_ref(str(answer), source_docs_for_resolution, context_ref_map, allowed_refs)
            if resolved:
                resolved_answers.append(resolved)
            else:
                unresolved.append(str(answer))

        resolved_reverse: List[str] = []
        for answer in candidate.get("reverse_answers", []):
            resolved = self._resolve_answer_ref(str(answer), source_docs_for_resolution, context_ref_map, allowed_refs)
            if resolved:
                resolved_reverse.append(resolved)
            else:
                unresolved.append(str(answer))

        resolved_answers = self._dedupe_preserve_order(resolved_answers)
        resolved_reverse = self._dedupe_preserve_order(resolved_reverse)
        if unresolved or not resolved_answers or not resolved_reverse:
            logger.debug(
                "Dropping unresolved worker candidate [edge=%s unresolved=%s]",
                packet.edge.as_tuple(),
                unresolved[:6],
            )
            return None

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
            return None
        canonical_source_docs = self._dedupe_preserve_order(
            [doc for doc in source_docs_for_resolution if doc in docs_from_refs]
        )
        if len(canonical_source_docs) < 2:
            return None

        sanitized = dict(candidate)
        sanitized["answers"] = resolved_answers
        sanitized["reverse_answers"] = resolved_reverse
        sanitized["source_docs"] = canonical_source_docs
        return sanitized

    def _apply_candidates(self, packet: EdgePacket, candidates: List[Dict[str, Any]]) -> Tuple[int, int]:
        accepted = 0
        rejected = 0
        seen_signatures: Set[Tuple[Any, ...]] = set()

        for candidate in candidates:
            sanitized = self._sanitize_candidate_refs(packet, candidate)
            if sanitized is None:
                rejected += 1
                continue

            signature = self._exact_candidate_signature(sanitized)
            if signature in seen_signatures:
                logger.debug(
                    "Skipping duplicate worker candidate [edge=%s]",
                    packet.edge.as_tuple(),
                )
                continue
            seen_signatures.add(signature)

            result = self._call_tool(
                "create_bridge_qa",
                {"bridges": [sanitized]},
                doc_id=packet.edge.doc_id,
            )
            items = result if isinstance(result, list) else [result]
            for item in items:
                if isinstance(item, dict) and item.get("success"):
                    accepted += 1
                else:
                    rejected += 1

        return accepted, rejected

    def _run_edge(self, edge: EdgeKey) -> EdgeRunResult:
        start_tokens = self.reconciler.tokens_used

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
        last_failure_signature: Optional[Tuple[Tuple[Any, ...], ...]] = None
        last_failure_state: Optional[Tuple[int, bool, bool, int]] = None

        depth = 0
        include_optional = False

        while call_count < edge_call_limit:
            edge_spend = self.reconciler.tokens_used - start_tokens
            remaining_edge_tokens = edge_token_limit - edge_spend
            if remaining_edge_tokens <= 0:
                budget_exhausted = True
                break

            path_proofs = self._mine_path_proofs(packet, include_optional=include_optional)
            render_input = RenderInput(
                edge=edge,
                proofs=path_proofs,
                constraints=packet.constraints,
            ) if path_proofs else None

            worker_output = self.worker.generate(
                packet=packet,
                depth=depth,
                include_optional=include_optional,
                max_response_tokens=min(1400, remaining_edge_tokens),
                render_input=render_input,
            )
            call_count += 1

            attempted_total += len(worker_output.candidates)
            if not worker_output.candidates:
                no_candidate_calls += 1

            accepted, rejected = self._apply_candidates(packet, worker_output.candidates)
            accepted_total += accepted
            rejected_total += rejected

            # Enforce hard edge budget even if the latest model call overshot.
            edge_spend = self.reconciler.tokens_used - start_tokens
            if edge_spend >= edge_token_limit:
                budget_exhausted = True
                break

            if accepted_total > 0:
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

            if can_recurse and not include_optional and packet.optional_contexts:
                include_optional = True
                depth += 1
                self.reconciler.rlm_metrics["recursive_invocations"] += 1
                continue

            if can_recurse and worker_output.need_recursion:
                depth += 1
                include_optional = True
                self.reconciler.rlm_metrics["recursive_invocations"] += 1
                continue

            if repeated_failure:
                self.reconciler.rlm_metrics["repeated_failure_stops"] = self.reconciler.rlm_metrics.get("repeated_failure_stops", 0) + 1
                break

            if not worker_output.candidates:
                # No candidates and no recursion path left.
                break

            if accepted == 0:
                if signal.get("probe_only"):
                    break
                # Retry only when the candidate signature changed at the same depth/state.
                continue

            # Accepted candidates exist; continue only if loop conditions require it.
            continue

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
                "signal_tier": signal.get("tier"),
                "signal_score": signal.get("score"),
                "no_candidate_calls": no_candidate_calls,
                "call_limit": edge_call_limit,
                "token_limit": edge_token_limit,
                "worker_prompt_version": WORKER_PROMPT_VERSION,
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
