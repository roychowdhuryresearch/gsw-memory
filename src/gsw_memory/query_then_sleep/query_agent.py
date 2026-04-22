from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from gsw_memory.sleep_time.curriculum import (
    BatchQueryItem,
    BridgeRegistry,
    parse_answer_from_response,
    score_predicted_answer,
)
from gsw_memory.sleep_time.tools import GSWTools

from .prompts import (
    QUERY_DECOMPOSITION_SYSTEM_PROMPT,
    QUERY_FINAL_SYNTHESIS_SYSTEM_PROMPT,
    QUERY_TOOL_AGENT_INSTRUCTIONS,
)
from .transport import StageTransport, TransportResponse
from .types import (
    QueryAgentBatchResult,
    QueryInteractionTrace,
    QuerySubQuestionTrace,
    ToolCallTrace,
)


class QueryToolAdapter:
    """Query-time tool surface backed by sleep-time GSW tools plus bridge retrieval."""

    def __init__(self, entity_searcher: Any, bridge_registry: Optional[BridgeRegistry] = None, bridge_top_k: int = 10):
        self.entity_searcher = entity_searcher
        self.gsw_tools = GSWTools(entity_searcher)
        self.bridge_registry = bridge_registry or BridgeRegistry()
        self.bridge_top_k = max(1, int(bridge_top_k))
        self.bridge_hits_seen: List[Dict[str, Any]] = []
        self.bridge_ids_seen: List[str] = []

    def _embed_texts(self, texts: Sequence[str]) -> Optional[Sequence[Any]]:
        if not texts or not hasattr(self.entity_searcher, "_embed_query"):
            return None
        try:
            return self.entity_searcher._embed_query(list(texts))
        except Exception:
            return None

    def search_entities(self, query: str, top_k: int = 10, exclude_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.gsw_tools.search_entities(query=query, top_k=top_k, exclude_doc_id=exclude_doc_id)

    def get_entity_context(self, entity_name: str, doc_id: Optional[Any] = None) -> Any:
        return self.gsw_tools.get_entity_context(entity_name=entity_name, doc_id=doc_id)

    def find_entity_neighbors(self, entity_name: str, relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.gsw_tools.find_entity_neighbors(entity_name=entity_name, relationship_types=relationship_types)

    def trace_relationship_chain(self, start_entity: str, end_entity: str, max_hops: int = 4) -> List[List[Dict[str, Any]]]:
        return self.gsw_tools.trace_relationship_chain(start_entity=start_entity, end_entity=end_entity, max_hops=max_hops)

    def search_qa_pairs(self, query: str, entity_filter: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.gsw_tools.search_qa_pairs(query=query, entity_filter=entity_filter, top_k=top_k)

    def get_bridge_evidence(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        hits = self.bridge_registry.search(
            query=query,
            top_k=int(top_k or self.bridge_top_k),
            embed_texts_fn=self._embed_texts,
        )
        if hits:
            self.bridge_hits_seen.extend(hits)
            for row in hits:
                bridge_id = str(row.get("bridge_id", "") or "").strip()
                if bridge_id and bridge_id not in self.bridge_ids_seen:
                    self.bridge_ids_seen.append(bridge_id)
        return hits


class QueryAgent:
    """Agentic query-time answerer using shared provider transport and tool calling."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        max_iterations: int = 8,
        generation_params: Optional[Dict[str, Any]] = None,
        bridge_top_k: int = 10,
        reasoning_effort: str = "medium",
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.base_url = str(base_url).strip() if base_url else None
        self.max_iterations = max(1, int(max_iterations))
        self.generation_params = dict(generation_params or {"temperature": 0.0})
        self.bridge_top_k = max(1, int(bridge_top_k))
        self.reasoning_effort = str(reasoning_effort or "medium").strip() or "medium"
        self.event_callback = event_callback
        self.transport = StageTransport(
            model_name=self.model_name,
            base_url=self.base_url,
            reasoning_effort=self.reasoning_effort,
        )

    @staticmethod
    def _tool_definitions() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_entities",
                    "description": "Search for entities by name. Returns entities with their QA pairs (question-answer facts like nationality, birth date, relationships), roles, and states. This is your primary tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Entity name or description to search for"},
                            "top_k": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_qa_pairs",
                    "description": "Search QA pairs across all documents by question text. Less precise than search_entities — use only when you don't know which entity to search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Question to search for"},
                            "top_k": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bridge_evidence",
                    "description": "Search pre-computed bridge evidence from prior sleep batches.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    @staticmethod
    def _json_safe(value: Any, max_items: int = 20) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_items:
                    result["..."] = f"truncated_after_{max_items}_items"
                    break
                result[str(key)] = QueryAgent._json_safe(item, max_items=max_items)
            return result
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            trimmed = [QueryAgent._json_safe(item, max_items=max_items) for item in items[:max_items]]
            if len(items) > max_items:
                trimmed.append(f"truncated_after_{max_items}_items")
            return trimmed
        if hasattr(value, "model_dump"):
            return QueryAgent._json_safe(value.model_dump(), max_items=max_items)
        if hasattr(value, "as_dict"):
            return QueryAgent._json_safe(value.as_dict(), max_items=max_items)
        return str(value)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _normalize_final_answer(question: str, answer: str) -> str:
        raw_answer = parse_answer_from_response(str(answer or ""))
        cleaned = re.sub(r"\s+", " ", raw_answer).strip()
        if not cleaned:
            return ""

        cleaned = re.sub(r"^\**answer:\**\s*", "", cleaned, flags=re.IGNORECASE).strip()
        # Strip doc references like 【doc_123::e4】
        cleaned = re.sub(r"【[^】]*】", "", cleaned).strip()
        # Strip markdown bold
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned).strip()

        question_lower = str(question or "").strip().lower()
        if question_lower.startswith(
            (
                "is ",
                "are ",
                "was ",
                "were ",
                "do ",
                "does ",
                "did ",
                "has ",
                "have ",
                "had ",
                "can ",
                "could ",
                "will ",
                "would ",
                "should ",
            )
        ):
            lowered = cleaned.lower()
            if lowered.startswith("yes"):
                return "yes"
            if lowered.startswith("no"):
                return "no"

        cleaned = cleaned.strip().strip("`").strip()
        if cleaned.endswith(".") and cleaned.count(".") == 1:
            cleaned = cleaned[:-1].strip()
        return cleaned

    @staticmethod
    def _looks_like_long_answer(answer: str) -> bool:
        text = str(answer or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if len(text.split()) > 8:
            return True
        if any(token in lowered for token in (" the answer is ", " because ", " according to ", " via ", " found in ")):
            return True
        if any(text.startswith(prefix) for prefix in ("The ", "This ", "That ")):
            return True
        return False

    @classmethod
    def _extract_short_answer_candidate(cls, question: str, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        cleaned = re.sub(r"【[^】]*】", "", raw)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        normalized = cls._normalize_final_answer(question, cleaned)
        if normalized and not cls._looks_like_long_answer(normalized):
            return normalized

        question_lower = str(question or "").strip().lower()
        patterns: List[str] = []
        if question_lower.startswith("who"):
            patterns = [
                r"\b(?:was|is|were|are)\s+([^.;]+?)(?:[.;]|$)",
                r"\b(?:husband|wife|spouse)\s+(?:was|is)\s+([^.;]+?)(?:[.;]|$)",
            ]
        elif question_lower.startswith("where"):
            patterns = [
                r"\b(?:in|at)\s+([^.;]+?)(?:[.;]|$)",
            ]
        elif question_lower.startswith("when"):
            patterns = [
                r"\b(?:on|in)\s+([^.;]+?)(?:[.;]|$)",
            ]
        elif question_lower.startswith("what"):
            patterns = [
                r"\b(?:was|is|were|are)\s+([^.;]+?)(?:[.;]|$)",
            ]

        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = cls._normalize_final_answer(question, match.group(1))
            if candidate and not cls._looks_like_long_answer(candidate):
                return candidate
        return ""

    @classmethod
    def _choose_final_answer(
        cls,
        *,
        question: str,
        synthesis_answer: str,
        synthesis_raw_text: str,
        tool_loop_final_text: str,
    ) -> str:
        normalized_answer = cls._normalize_final_answer(question, synthesis_answer)
        if normalized_answer and not cls._looks_like_long_answer(normalized_answer):
            return normalized_answer

        for source_text in (synthesis_raw_text, tool_loop_final_text, synthesis_answer):
            candidate = cls._extract_short_answer_candidate(question, source_text)
            if candidate:
                return candidate
        return normalized_answer

    def _emit(self, event_type: str, **data: Any) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, data)

    def _log_model_response(self, response: TransportResponse, *, stage: str, question_id: str, iteration: int = 0) -> None:
        payload = response.as_log_dict()
        payload.update({
            "stage": stage,
            "question_id": question_id,
            "iteration": int(iteration),
        })
        self._emit("model_response", **payload)

    def _call_json_response(
        self,
        *,
        instructions: str,
        user_text: str,
        stage: str,
        question_id: str,
        max_output_tokens: int = 800,
    ) -> tuple[Dict[str, Any], TransportResponse]:
        payload, response = self.transport.generate_json(
            instructions=instructions,
            user_text=user_text,
            max_output_tokens=max_output_tokens,
            temperature=float(self.generation_params.get("temperature", 0.0) or 0.0),
            stage=stage,
        )
        self._log_model_response(response, stage=stage, question_id=question_id)
        return payload, response

    def _decompose_question(self, question: str, *, question_id: str) -> List[QuerySubQuestionTrace]:
        try:
            payload, _response = self._call_json_response(
                instructions=QUERY_DECOMPOSITION_SYSTEM_PROMPT,
                user_text=f"Question: {question}",
                stage="decomposition",
                question_id=question_id,
                max_output_tokens=600,
            )
        except Exception as exc:
            self._emit("query_model_error", stage="decomposition", question_id=question_id, error=str(exc))
            payload = {}
        rows = payload.get("sub_questions", []) if isinstance(payload, dict) else []
        traces: List[QuerySubQuestionTrace] = []
        if isinstance(rows, list):
            for row in rows[:4]:
                if not isinstance(row, dict):
                    continue
                sub_question = str(row.get("sub_question", "") or "").strip()
                if not sub_question:
                    continue
                traces.append(
                    QuerySubQuestionTrace(
                        sub_question=sub_question,
                        relation_hint=str(row.get("relation_hint", "") or "").strip(),
                        entity_hint=str(row.get("entity_hint", "") or "").strip(),
                    )
                )
        if traces:
            return traces
        return [QuerySubQuestionTrace(sub_question=question)]

    def _tool_summary(self, tool_calls: Sequence[ToolCallTrace]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in tool_calls:
            result = item.result
            preview = self._json_safe(result, max_items=4)
            rows.append(
                {
                    "iteration": int(item.iteration),
                    "tool_name": item.tool_name,
                    "arguments": self._json_safe(item.arguments, max_items=8),
                    "result_preview": preview,
                }
            )
        return rows

    def _synthesize_trace(
        self,
        *,
        question: str,
        question_id: str,
        sub_questions: Sequence[QuerySubQuestionTrace],
        tool_calls: Sequence[ToolCallTrace],
        bridge_ids_seen: Sequence[str],
    ) -> tuple[Dict[str, Any], str]:
        payload = {
            "question": question,
            "sub_questions": [item.as_dict() for item in sub_questions],
            "tool_trace": self._tool_summary(tool_calls),
            "bridge_ids_seen": list(bridge_ids_seen),
        }
        try:
            result, response = self._call_json_response(
                instructions=QUERY_FINAL_SYNTHESIS_SYSTEM_PROMPT,
                user_text=json.dumps(payload, ensure_ascii=False),
                stage="final_synthesis",
                question_id=question_id,
                max_output_tokens=1200,
            )
            return result, str(response.text or "").strip()
        except Exception as exc:
            self._emit("query_model_error", stage="final_synthesis", question_id=question_id, error=str(exc))
            return (
                {
                    "answer": "",
                    "reasoning_chain": "",
                    "sub_questions": [item.as_dict() for item in sub_questions],
                    "found_relations": [],
                    "missing_relations": [],
                    "bridge_ids_used": list(bridge_ids_seen),
                },
                "",
            )

    def answer_question(
        self,
        *,
        item: BatchQueryItem,
        tools: QueryToolAdapter,
    ) -> QueryInteractionTrace:
        question_id = str(item.question_id or "")
        self._emit(
            "query_started",
            question_id=question_id,
            question=str(item.question or ""),
            doc_ids=list(item.doc_ids),
        )
        sub_questions = self._decompose_question(item.question, question_id=question_id)
        self._emit(
            "query_decomposition",
            question_id=question_id,
            sub_questions=[row.as_dict() for row in sub_questions],
        )

        tool_calls: List[ToolCallTrace] = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": QUERY_TOOL_AGENT_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Question: {item.question}\n"
                    f"Decomposition: {json.dumps([row.as_dict() for row in sub_questions], ensure_ascii=False)}"
                ),
            },
        ]
        tool_map = {
            "search_entities": tools.search_entities,
            "search_qa_pairs": tools.search_qa_pairs,
            "get_bridge_evidence": tools.get_bridge_evidence,
        }

        # Auto-search bridges if registry has any
        if len(tools.bridge_registry.records) > 0:
            bridge_hits = tools.get_bridge_evidence(item.question)
            if bridge_hits:
                bridge_text = "\n".join(
                    f"- Bridge: Q: {h.get('question', '')} A: {h.get('answer_text', '')}"
                    for h in bridge_hits[:5]
                )
                messages[1]["content"] += (
                    f"\n\nPre-computed bridge evidence (hints, verify before using):\n{bridge_text}"
                    f"\n\nThese bridges suggest which entities and relations to look for. "
                    f"Bridge answers are entity names that may not exactly match the expected format — "
                    f"verify with search_qa_pairs before giving your final answer."
                )
                tool_calls.append(ToolCallTrace(
                    tool_name="get_bridge_evidence",
                    arguments={"query": item.question, "auto": True},
                    result=bridge_hits,
                    call_id="auto_bridge_search",
                ))
                self._emit(
                    "tool_call",
                    question_id=question_id,
                    iteration=0,
                    tool="get_bridge_evidence",
                    arguments={"query": item.question, "auto": True},
                    call_id="auto_bridge_search",
                )
                self._emit(
                    "tool_result",
                    question_id=question_id,
                    iteration=0,
                    tool="get_bridge_evidence",
                    result={"hits": len(bridge_hits)},
                    call_id="auto_bridge_search",
                )

        final_text = ""
        error_text = ""
        iterations = 0

        try:
            while iterations < self.max_iterations:
                iterations += 1
                response = self.transport.chat(
                    messages,
                    tools=self._tool_definitions(),
                    tool_choice="auto",
                    temperature=float(self.generation_params.get("temperature", 0.0) or 0.0),
                    max_tokens=1200,
                    stage="tool_loop",
                )
                self._log_model_response(response, stage="tool_loop", question_id=question_id, iteration=iterations)

                if response.tool_calls:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.text or "",
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False),
                                },
                            }
                            for call in response.tool_calls
                        ],
                    }
                    messages.append(assistant_message)
                    for function_call in response.tool_calls:
                        parsed_arguments = dict(function_call.arguments)
                        tool_fn = tool_map.get(function_call.name)
                        self._emit(
                            "tool_call",
                            question_id=question_id,
                            iteration=iterations,
                            tool=function_call.name,
                            arguments=self._json_safe(parsed_arguments),
                            call_id=function_call.call_id,
                        )
                        if tool_fn is None:
                            result = {"error": f"Unknown tool: {function_call.name}"}
                        else:
                            try:
                                result = tool_fn(**parsed_arguments)
                            except Exception as exc:
                                result = {"error": str(exc), "tool": function_call.name}
                        safe_result = self._json_safe(result)
                        self._emit(
                            "tool_result",
                            question_id=question_id,
                            iteration=iterations,
                            tool=function_call.name,
                            result=safe_result,
                            call_id=function_call.call_id,
                        )
                        tool_calls.append(
                            ToolCallTrace(
                                tool_name=function_call.name,
                                arguments=self._json_safe(parsed_arguments),
                                result=safe_result,
                                call_id=function_call.call_id,
                                iteration=iterations,
                            )
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": function_call.call_id,
                                "content": json.dumps(safe_result, ensure_ascii=False),
                            }
                        )
                    continue

                final_text = (response.text or response.reasoning_text or "").strip()
                break
        except Exception as exc:
            error_text = str(exc)
            self._emit("query_model_error", stage="tool_loop", question_id=question_id, error=error_text, iteration=iterations)

        synthesis, synthesis_raw_text = self._synthesize_trace(
            question=item.question,
            question_id=question_id,
            sub_questions=sub_questions,
            tool_calls=tool_calls,
            bridge_ids_seen=tools.bridge_ids_seen,
        )

        synthesis_sub_questions: List[QuerySubQuestionTrace] = []
        for row in synthesis.get("sub_questions", []) or []:
            if not isinstance(row, dict):
                continue
            synthesis_sub_questions.append(
                QuerySubQuestionTrace(
                    sub_question=str(row.get("sub_question", "") or "").strip(),
                    relation_hint=str(row.get("relation_hint", "") or "").strip(),
                    entity_hint=str(row.get("entity_hint", "") or "").strip(),
                    answer_found=bool(row.get("answer_found", False)),
                    answer=str(row.get("answer", "") or "").strip(),
                    searched_for=str(row.get("searched_for", "") or "").strip(),
                    found_instead=[str(v).strip() for v in row.get("found_instead", []) or [] if str(v).strip()],
                    notes=str(row.get("notes", "") or "").strip(),
                )
            )
        if synthesis_sub_questions:
            sub_questions = synthesis_sub_questions

        final_answer = self._choose_final_answer(
            question=item.question,
            synthesis_answer=str(synthesis.get("answer", "") or "").strip(),
            synthesis_raw_text=synthesis_raw_text,
            tool_loop_final_text=final_text.strip(),
        )

        gold_answers = []
        if item.gold_answer:
            gold_answers.append(item.gold_answer)
        for alias in (item.answer_aliases or []):
            if alias and alias not in gold_answers:
                gold_answers.append(alias)
        exact_match, f1 = score_predicted_answer(gold_answers, final_answer)
        trace = QueryInteractionTrace(
            question_id=question_id,
            question=str(item.question or ""),
            doc_ids=list(item.doc_ids),
            gold_answers=gold_answers,
            sub_questions=list(sub_questions),
            tool_calls=tool_calls,
            final_answer=final_answer,
            final_reasoning=str(synthesis.get("reasoning_chain", "") or final_text or "").strip(),
            exact_match=exact_match,
            f1=f1,
            found_relations=[str(v).strip() for v in synthesis.get("found_relations", []) or [] if str(v).strip()],
            missing_relations=[str(v).strip() for v in synthesis.get("missing_relations", []) or [] if str(v).strip()],
            bridge_evidence_used=bool(tools.bridge_ids_seen),
            bridge_ids_used=[str(v).strip() for v in synthesis.get("bridge_ids_used", []) or tools.bridge_ids_seen if str(v).strip()],
            tool_iterations=iterations,
            status="error" if error_text else "ok",
            error=error_text,
            metadata={
                "raw_final_text": final_text,
                "raw_synthesis_text": synthesis_raw_text,
                "question_metadata": dict(item.metadata or {}),
                "provider": self.transport.provider,
            },
        )
        self._emit(
            "query_finished",
            question_id=question_id,
            question=str(item.question or ""),
            answer=str(trace.final_answer or ""),
            gold_answer=str(trace.gold_answers[0] if trace.gold_answers else ""),
            status=trace.status,
            f1=trace.f1,
            exact_match=trace.exact_match,
            bridge_evidence_used=trace.bridge_evidence_used,
            bridge_ids_used=list(trace.bridge_ids_used),
            tool_iterations=trace.tool_iterations,
            missing_relations=list(trace.missing_relations),
        )
        return trace


def run_query_agent_batch(
    *,
    batch_index: int,
    batch_items: Sequence[BatchQueryItem],
    entity_searcher: Any,
    model_name: str,
    base_url: Optional[str] = None,
    bridge_registry: Optional[BridgeRegistry] = None,
    max_iterations: int = 8,
    generation_params: Optional[Dict[str, Any]] = None,
    bridge_top_k: int = 10,
    reasoning_effort: str = "medium",
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> QueryAgentBatchResult:
    traces: List[QueryInteractionTrace] = []
    query_agent = QueryAgent(
        model_name=model_name,
        base_url=base_url,
        max_iterations=max_iterations,
        generation_params=generation_params,
        bridge_top_k=bridge_top_k,
        reasoning_effort=reasoning_effort,
        event_callback=event_callback,
    )

    for item in batch_items:
        tools = QueryToolAdapter(entity_searcher=entity_searcher, bridge_registry=bridge_registry, bridge_top_k=bridge_top_k)
        traces.append(query_agent.answer_question(item=item, tools=tools))

    scored_f1 = [float(trace.f1) for trace in traces if trace.f1 is not None]
    scored_em = [float(trace.exact_match) for trace in traces if trace.exact_match is not None]
    bridge_usage_rate = (
        sum(1 for trace in traces if trace.bridge_evidence_used) / float(len(traces)) if traces else 0.0
    )
    overall_metrics = {
        "F1": sum(scored_f1) / len(scored_f1) if scored_f1 else 0.0,
        "ExactMatch": sum(scored_em) / len(scored_em) if scored_em else 0.0,
        "bridge_usage_rate": bridge_usage_rate,
        "questions_answered": float(len(traces)),
    }
    return QueryAgentBatchResult(batch_index=int(batch_index), traces=traces, overall_metrics=overall_metrics)
