import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from gsw_memory.sleep_time.rlm_pipeline import (
    EdgeKey,
    EdgePacket,
    EdgeRunResult,
    EdgeWorkerProposal,
    HybridScheduler,
    PLANNER_PROMPT_VERSION,
    PreparedEdgeWork,
    RenderInput,
    RecursiveEdgeWorker,
    RootScheduler,
    WORKER_PROMPT_VERSION,
    WorkerResponseSchema,
)
from gsw_memory.sleep_time.tools import GSWTools


class _DummyMessage:
    def __init__(self, content=None, parsed=None, reasoning=None, reasoning_content=None, finish_reason=None, tool_calls=None):
        self.content = content
        self.parsed = parsed
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls


@dataclass
class _EntityNode:
    id: str
    name: str


class _MiniGSW:
    def __init__(self, entities: List[_EntityNode]):
        self.entity_nodes = entities
        self.verb_phrase_nodes = []

    def get_entity_by_id(self, entity_id: str):
        for entity in self.entity_nodes:
            if entity.id == entity_id:
                return entity
        return None


class _DummySearcher:
    def __init__(self):
        self.gsw_by_doc_id = {
            "doc_0": _MiniGSW([_EntityNode("e1", "Source")]),
            "doc_1": _MiniGSW([_EntityNode("e2", "20 March 851"), _EntityNode("e1", "Neighbor")]),
            "doc_2": _MiniGSW([_EntityNode("e5", "Abbey X")]),
            "doc_3": _MiniGSW([_EntityNode("e1", "795")]),
            "doc_4": _MiniGSW([_EntityNode("e9", "Lotharingia")]),
        }
        self.entities = [
            {"name": "Source", "doc_id": "doc_0"},
            {"name": "Neighbor", "doc_id": "doc_0"},
            {"name": "Neighbor", "doc_id": "doc_1"},
            {"name": "Neighbor", "doc_id": "doc_2"},
        ]

    def search(self, query, top_k=10, verbose=False):
        q = str(query).strip().lower()
        if q != "neighbor":
            return []
        return [
            ({"name": "Neighbor alt", "doc_id": "doc_3", "roles": [], "all_states": []}, 0.91),
            ({"name": "Neighbor variant", "doc_id": "doc_4", "roles": [], "all_states": []}, 0.78),
        ]


class _FakeReconciler:
    def __init__(self, token_increment: int = 100):
        self.tools = GSWTools(_DummySearcher())
        self.callback_events: List[Tuple[str, Dict[str, Any]]] = []
        self.output_callback = lambda event_type, data: self.callback_events.append((event_type, data))
        self.tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.token_increment = token_increment
        self.provider = "openai"
        self.budget = {"max_tokens": 10_000_000}
        self._current_doc_exploration_doc_id = None
        self.root_model = "root-model"
        self.worker_model = "worker-model"
        self.bridge_verifier_enabled = False
        self.bridge_verifier_threshold = 0.75
        self.verifier_parallel_workers = 4
        self.edge_max_accepted_bridges = 2
        self.entities_explored = 0
        self.rlm_metrics: Dict[str, Any] = {
            "root_calls": 0,
            "worker_calls": 0,
            "verifier_calls": 0,
            "root_input_tokens": 0,
            "root_output_tokens": 0,
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
            "verifier_input_tokens": 0,
            "verifier_output_tokens": 0,
            "edges_explored": 0,
            "edges_with_bridges": 0,
            "recursive_invocations": 0,
            "low_signal_edges": 0,
            "repeated_failure_stops": 0,
            "edges_skipped_no_path": 0,
            "planner_actions_overridden": 0,
            "prefilter_reject_count": 0,
            "alias_resolution_hits": 0,
            "calls_blocked_no_progress": 0,
            "planner_decisions": 0,
            "planner_fallbacks": 0,
            "planner_stop_actions": 0,
            "docs_attempted": 0,
            "docs_completed": 0,
            "parallel_batches": 0,
            "parallel_workers_used": 0,
            "commit_queue_size_peak": 0,
            "stale_edge_skipped": 0,
            "parallel_generation_seconds": 0.0,
            "serial_commit_seconds": 0.0,
            "pre_verifier_cache_hits": 0,
            "verifier_parallel_workers_used": 0,
            "parallel_verifier_seconds": 0.0,
            "edge_accept_cap_hits": 0,
            "guided_retry_scheduled": 0,
            "guided_retry_started": 0,
            "guided_retry_succeeded": 0,
            "guided_retry_exhausted": 0,
            "guided_retry_blocked_no_progress": 0,
            "worker_parse_recovery_success": 0,
            "worker_parse_recovery_fail": 0,
            "verifier_parse_recovery_success": 0,
            "verifier_parse_recovery_fail": 0,
        }

        self.stage_payloads: Dict[str, List[str]] = {
            "root": [],
            "worker": [],
            "verifier": [],
        }
        self.captured_response_formats: List[Any] = []

        self.context_map: Dict[str, Dict[str, Any]] = {
            "doc_0": {
                "entity": "Source",
                "doc_id": "doc_0",
                "qa_pairs": [
                    {
                        "question": "Who is Source related to?",
                        "answer": "Neighbor",
                        "answer_refs": ["doc_0::e1"],
                    }
                ],
                "roles": ["person"],
                "states": ["historical figure"],
                "relationships": {"related to": ["Neighbor"]},
            },
            "doc_1": {
                "entity": "Neighbor",
                "doc_id": "doc_1",
                "qa_pairs": [
                    {
                        "question": "When did Neighbor die?",
                        "answer": "20 March 851",
                        "answer_refs": ["doc_1::e2"],
                    }
                ],
                "roles": ["person"],
                "states": ["deceased"],
                "relationships": {"died on": ["20 March 851"]},
            },
            "doc_2": {
                "entity": "Neighbor",
                "doc_id": "doc_2",
                "qa_pairs": [
                    {
                        "question": "Where was Neighbor buried?",
                        "answer": "Abbey X",
                        "answer_refs": ["doc_2::e5"],
                    }
                ],
                "roles": ["person"],
                "states": ["historical figure"],
                "relationships": {"buried in": ["Abbey X"]},
            },
            "doc_3": {
                "entity": "Neighbor",
                "doc_id": "doc_3",
                "qa_pairs": [
                    {
                        "question": "What year was Neighbor born?",
                        "answer": "795",
                        "answer_refs": ["doc_3::e1"],
                    }
                ],
                "roles": ["person"],
                "states": ["historical figure"],
                "relationships": {"born in": ["795"]},
            },
            "doc_4": {
                "entity": "Neighbor",
                "doc_id": "doc_4",
                "qa_pairs": [
                    {
                        "question": "Where did Neighbor rule?",
                        "answer": "Lotharingia",
                        "answer_refs": ["doc_4::e9"],
                    }
                ],
                "roles": ["person"],
                "states": ["ruler"],
                "relationships": {"ruled": ["Lotharingia"]},
            },
        }

        self.tools.doc_exploration_plans["doc_0"] = {
            "doc_id": "doc_0",
            "entities_to_explore": [
                {
                    "name": "Source",
                    "roles": ["person"],
                    "status": "pending",
                    "neighbors": [
                        {
                            "entity": "Neighbor",
                            "relationship": "related to",
                            "other_docs": ["doc_1", "doc_2"],
                            "status": "pending",
                        }
                    ],
                }
            ],
            "total_neighbors_to_explore": 1,
            "explored_count": 0,
            "pending_count": 1,
        }

    def _increment_stage_usage(self, stage: str):
        prompt = self.token_increment // 2
        completion = self.token_increment - prompt
        self.tokens_used += self.token_increment
        self.input_tokens += prompt
        self.output_tokens += completion

        if stage == "root":
            self.rlm_metrics["root_calls"] += 1
            self.rlm_metrics["root_input_tokens"] += prompt
            self.rlm_metrics["root_output_tokens"] += completion
        elif stage == "worker":
            self.rlm_metrics["worker_calls"] += 1
            self.rlm_metrics["worker_input_tokens"] += prompt
            self.rlm_metrics["worker_output_tokens"] += completion
        elif stage == "verifier":
            self.rlm_metrics["verifier_calls"] += 1
            self.rlm_metrics["verifier_input_tokens"] += prompt
            self.rlm_metrics["verifier_output_tokens"] += completion

    def _call_model_for_stage(
        self,
        messages,
        model_name=None,
        max_tokens=1024,
        temperature=0.2,
        stage="worker",
        response_format=None,
        tools=None,
        tool_choice=None,
        return_usage=False,
    ):
        self.captured_response_formats.append(response_format)
        self._increment_stage_usage(stage)
        queue = self.stage_payloads.get(stage, [])
        if queue:
            payload = queue.pop(0)
        else:
            payload = '{"status": "ok", "candidates": [], "need_recursion": false, "notes": "default"}'
        message = payload if isinstance(payload, _DummyMessage) else _DummyMessage(payload)
        if return_usage:
            prompt_tokens = self.token_increment // 2
            completion_tokens = self.token_increment - prompt_tokens
            return message, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": self.token_increment,
                "finish_reason": getattr(message, "finish_reason", None),
            }
        return message

    def _extract_message_content_with_source(self, message, stage=None, structured_output=False):
        if self.provider == "vllm" and structured_output and stage in {"worker", "verifier"}:
            if getattr(message, "parsed", None) is not None:
                if isinstance(message.parsed, str):
                    parsed_text = message.parsed.strip()
                else:
                    parsed_text = json.dumps(message.parsed).strip()
                if parsed_text:
                    return parsed_text, "parsed"
                return "", "parsed_empty"
            content = (message.content or "").strip()
            if content:
                return content, "content"
            return "", "content_empty"
        return (message.content or ""), "content"

    def _extract_json_from_text(self, text: str, payload_label: str = "response"):
        if not (text or "").strip():
            raise ValueError(f"Empty {payload_label} response")
        return json.loads(text)

    def _preview_text(self, text: str, max_len: int = 120):
        text = (text or "").strip()
        return text[:max_len]

    def _coerce_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None)
        return str(value)

    def _summarize_assistant_message_fields(self, message: Any, max_preview_len: int = 240):
        parsed_value = getattr(message, "parsed", None)
        parsed_preview = ""
        parsed_type = None
        if parsed_value is not None:
            parsed_type = type(parsed_value).__name__
            if isinstance(parsed_value, str):
                parsed_preview = parsed_value[:max_preview_len]
            else:
                parsed_preview = json.dumps(parsed_value)[:max_preview_len]
        content_value = (getattr(message, "content", None) or "")
        reasoning_content = (getattr(message, "reasoning_content", None) or "")
        return {
            "has_parsed": parsed_value is not None,
            "parsed_type": parsed_type,
            "parsed_preview": parsed_preview,
            "has_content": bool(str(content_value).strip()),
            "content_preview": str(content_value)[:max_preview_len],
            "has_reasoning_content": bool(str(reasoning_content).strip()),
            "reasoning_content_preview": str(reasoning_content)[:max_preview_len],
            "has_tool_calls": bool(getattr(message, "tool_calls", None)),
        }

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        if tool_name == "plan_document_exploration":
            return self.tools.doc_exploration_plans.get(arguments["doc_id"], {"error": "missing plan"})

        if tool_name == "get_entity_context":
            entity_name = arguments.get("entity_name")
            doc_id = arguments.get("doc_id")
            self.tools._mark_focus_mandatory_doc_explored(entity_name, doc_id)
            if doc_id in self.context_map:
                return self.context_map[doc_id]
            return {
                "entity": entity_name,
                "doc_id": doc_id,
                "qa_pairs": [],
                "roles": [],
                "states": [],
                "relationships": {},
            }

        if tool_name == "create_bridge_qa":
            bridges = arguments.get("bridges", [])
            results = []
            for idx, bridge in enumerate(bridges):
                question = str(bridge.get("question", "")).lower()
                if "reject" in question:
                    results.append({"success": False, "error": "rejected"})
                else:
                    results.append(
                        {
                            "success": True,
                            "bridge_id": f"bridge_{idx}",
                            "validation": {"valid": True, "confidence": 0.9},
                        }
                    )
            return results

        method = getattr(self.tools, tool_name)
        return method(**arguments)


def _build_packet() -> EdgePacket:
    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")
    return EdgePacket(
        edge=edge,
        source_docs=["doc_0", "doc_1"],
        mandatory_docs=["doc_1"],
        optional_docs=[],
        source_context={
            "doc_id": "doc_0",
            "entity": "Source",
            "qa_pairs": [{"question": "Who is Source related to?", "answer": "Neighbor", "answer_refs": ["doc_0::e1"]}],
            "relationships": {"related to": ["Neighbor"]},
            "roles": ["person"],
            "states": ["historical figure"],
        },
        mandatory_contexts=[
            {
                "doc_id": "doc_1",
                "entity": "Neighbor",
                "qa_pairs": [{"question": "When did Neighbor die?", "answer": "20 March 851", "answer_refs": ["doc_1::e2"]}],
                "relationships": {"died on": ["20 March 851"]},
                "roles": ["person"],
                "states": ["deceased"],
            }
        ],
        optional_contexts=[],
        constraints={"must_be_chain": True, "reverse_required": True},
        budget={"max_depth": 1, "max_calls": 2, "max_tokens": 3000},
    )


def test_worker_parse_repair_retry_path():
    reconciler = _FakeReconciler(token_increment=50)
    reconciler.stage_payloads["worker"] = [
        "not-json",
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "When did Source's neighbor die?",
                        "answers": ["doc_1::e2"],
                        "reverse_question": "Whose neighbor died on 20 March 851?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_1"],
                        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": False,
                "notes": "ok",
            }
        ),
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=2, edge_max_tokens=3000)
    output = worker.generate(packet=_build_packet(), depth=0, include_optional=False, max_response_tokens=500)

    assert output.parse_stage == "repair_retry"
    assert output.parse_attempts == 2
    assert output.parse_recovery_used is True
    assert output.budgeted_token_usage == 50
    assert output.token_usage == 100
    assert len(output.candidates) == 1
    assert reconciler.rlm_metrics["worker_calls"] == 2
    assert reconciler.rlm_metrics["worker_parse_recovery_success"] == 1


def test_worker_parse_regenerate_retry_path():
    reconciler = _FakeReconciler(token_increment=40)
    reconciler.stage_payloads["worker"] = [
        "not-json",
        "still-not-json",
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "Where was Source's neighbor buried?",
                        "answers": ["doc_2::e5"],
                        "reverse_question": "Whose neighbor was buried in Abbey X?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_2"],
                        "reasoning": "Forward target variable = Abbey X. Reverse target variable = Source. Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": False,
                "notes": "Recovered on regenerate.",
            }
        ),
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=2, edge_max_tokens=3000)
    output = worker.generate(packet=_build_packet(), depth=0, include_optional=False, max_response_tokens=500)

    assert output.parse_stage == "regenerate_retry"
    assert output.parse_attempts == 3
    assert output.parse_recovery_used is True
    assert output.budgeted_token_usage == 40
    assert output.token_usage == 120
    assert len(output.candidates) == 1
    assert reconciler.rlm_metrics["worker_calls"] == 3
    assert reconciler.rlm_metrics["worker_parse_recovery_success"] == 1


def test_worker_uses_pydantic_structured_output():
    reconciler = _FakeReconciler(token_increment=20)
    reconciler.stage_payloads["worker"] = [
        json.dumps(
            {
                "status": "ok",
                "candidates": [],
                "need_recursion": False,
                "notes": "schema ok",
            }
        )
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=1, edge_max_tokens=1000)
    output = worker.generate(packet=_build_packet(), depth=0, include_optional=False, max_response_tokens=300)

    assert output.status == "ok"
    response_format = reconciler.captured_response_formats[0]
    # Should pass the Pydantic model class directly for litellm structured output
    assert response_format is WorkerResponseSchema


def test_worker_prefers_vllm_parsed_structured_output():
    reconciler = _FakeReconciler(token_increment=20)
    reconciler.provider = "vllm"
    reconciler.stage_payloads["worker"] = [
        _DummyMessage(
            content='{"status": "wrong", "candidates": [], "need_recursion": true, "notes": "content"}',
            parsed={
                "status": "ok",
                "candidates": [],
                "need_recursion": False,
                "notes": "parsed wins",
            },
        )
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=1, edge_max_tokens=1000)
    output = worker.generate(packet=_build_packet(), depth=0, include_optional=False, max_response_tokens=300)

    assert output.status == "ok"
    assert output.notes == "parsed wins"
    assert output.source_field == "parsed"


def test_worker_vllm_empty_parsed_fails_without_content_fallback():
    reconciler = _FakeReconciler(token_increment=20)
    reconciler.provider = "vllm"
    reconciler.stage_payloads["worker"] = [
        _DummyMessage(
            content='{"status": "ok", "candidates": [], "need_recursion": false, "notes": "content only"}',
            parsed="",
        ),
        _DummyMessage(
            content='{"status": "ok", "candidates": [], "need_recursion": false, "notes": "content only"}',
            parsed="",
        ),
        _DummyMessage(
            content='{"status": "ok", "candidates": [], "need_recursion": false, "notes": "content only"}',
            parsed="",
        ),
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=1, edge_max_tokens=1000)
    output = worker.generate(packet=_build_packet(), depth=0, include_optional=False, max_response_tokens=300)

    assert output.status == "parse_error"
    assert output.source_field == "parsed_empty"
    assert output.notes == "Worker parse error: Empty worker response"


def test_worker_parse_failure_emits_diagnostics_event_with_finish_reason():
    reconciler = _FakeReconciler(token_increment=20)
    reconciler.provider = "vllm"
    reconciler.stage_payloads["worker"] = [
        _DummyMessage(content="", parsed=None, reasoning_content="hidden", finish_reason="length"),
        _DummyMessage(content="", parsed=None, reasoning_content="hidden", finish_reason="length"),
        _DummyMessage(content="", parsed=None, reasoning_content="hidden", finish_reason="length"),
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=1, edge_max_tokens=1000)
    output = worker.generate(packet=_build_packet(), depth=0, include_optional=False, max_response_tokens=300)

    assert output.status == "parse_error"
    assert output.finish_reason == "length"
    assert output.message_field_summary["has_content"] is False
    assert output.message_field_summary["has_reasoning_content"] is True

    parse_failure_events = [
        data
        for event_type, data in reconciler.callback_events
        if event_type == "rlm_worker_parse_failure"
    ]
    assert [event["stage"] for event in parse_failure_events] == [
        "initial",
        "repair_retry",
        "regenerate_retry",
    ]
    assert all(event["finish_reason"] == "length" for event in parse_failure_events)
    assert all(event["message_fields"]["has_reasoning_content"] is True for event in parse_failure_events)


def test_worker_invalid_structured_payload_reports_schema_error():
    reconciler = _FakeReconciler(token_increment=20)
    reconciler.stage_payloads["worker"] = [
        json.dumps({"status": "ok", "candidates": [], "need_recursion": "bad"}),
        json.dumps({"status": "ok", "candidates": [], "need_recursion": "bad"}),
        json.dumps({"status": "ok", "candidates": [], "need_recursion": "bad"}),
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=1, edge_max_tokens=1000)
    output = worker.generate(packet=_build_packet(), depth=0, include_optional=False, max_response_tokens=300)

    assert output.status == "parse_error"
    assert "Worker schema validation failed" in output.notes
    assert output.parse_stage == "regenerate_retry"


def test_packet_construction_includes_source_context_and_docs():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=2000,
        max_optional_docs_per_edge=2,
    )

    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")
    coverage = {
        "mandatory_docs": ["doc_1", "doc_2"],
        "optional_fuzzy_docs": [{"doc_id": "doc_3", "name": "Neighbor alt", "score": 0.91}],
    }
    packet = scheduler._build_edge_packet(
        edge=edge,
        coverage=coverage,
        source_context=reconciler.context_map["doc_0"],
        mandatory_contexts=[reconciler.context_map["doc_1"], reconciler.context_map["doc_2"]],
        optional_contexts=[reconciler.context_map["doc_3"]],
    )

    assert "doc_1" in packet.source_docs
    assert packet.source_context["doc_id"] == "doc_0"


def test_scheduler_collects_optional_context_via_alias_fallback(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=2000,
        max_optional_docs_per_edge=2,
    )

    reconciler.tools.active_neighbor_focus = {
        "doc_id": "doc_0",
        "entity_name": "Source",
        "neighbor_name": "Neighbor",
        "relationship": "related to",
    }
    edge_key = reconciler.tools._active_focus_edge_key()
    reconciler.tools.neighbor_doc_coverage_plans[edge_key] = {
        "doc_id": "doc_0",
        "entity_name": "Source",
        "neighbor_name": "Neighbor",
        "relationship": "related to",
        "mandatory_docs": [],
        "optional_fuzzy_docs": [{"doc_id": "doc_3", "name": "Neighbor alt", "score": 0.91}],
        "explored_mandatory_docs": [],
        "include_fuzzy": True,
        "top_k": 10,
    }

    calls = []

    def _fake_single_context(entity_name, doc_id):
        calls.append((entity_name, doc_id))
        if entity_name == "Neighbor alt" and doc_id == "doc_3":
            return {
                "entity": entity_name,
                "doc_id": doc_id,
                "qa_pairs": [{"question": "When was Neighbor alt born?", "answer": "795", "answer_refs": ["doc_3::e1"]}],
                "roles": ["person"],
                "states": ["historical figure"],
                "relationships": {"born in": ["795"]},
            }
        return {
            "entity": entity_name,
            "doc_id": doc_id,
            "qa_pairs": [],
            "roles": [],
            "states": [],
            "relationships": {},
        }

    monkeypatch.setattr(reconciler.tools, "_get_single_doc_context", _fake_single_context)

    original_execute_tool = reconciler._execute_tool

    def _patched_execute_tool(tool_name, arguments):
        if tool_name == "get_entity_context":
            return reconciler.tools.get_entity_context(**arguments)
        return original_execute_tool(tool_name, arguments)

    monkeypatch.setattr(reconciler, "_execute_tool", _patched_execute_tool)

    optional_contexts = scheduler._collect_contexts_for_docs("Neighbor", ["doc_3"], source_doc="doc_0")
    assert len(optional_contexts) == 1
    assert optional_contexts[0]["qa_pairs"]
    assert calls == [("Neighbor", "doc_3"), ("Neighbor alt", "doc_3")]

    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")
    packet = scheduler._build_edge_packet(
        edge=edge,
        coverage={"mandatory_docs": [], "optional_fuzzy_docs": [{"doc_id": "doc_3", "name": "Neighbor alt", "score": 0.91}]},
        source_context=reconciler.context_map["doc_0"],
        mandatory_contexts=[],
        optional_contexts=optional_contexts,
    )
    proofs = scheduler._mine_path_proofs(packet, include_optional=True)
    assert proofs


def test_path_proofs_include_source_subject_refs():
    reconciler = _FakeReconciler(token_increment=30)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )
    proofs = scheduler._mine_path_proofs(_build_packet(), include_optional=False, max_paths=4)
    assert proofs
    assert any("doc_0::e1" in proof.source_subject_refs for proof in proofs)


def test_scheduler_resolves_reverse_answer_from_source_subject_refs(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )
    packet = EdgePacket(
        edge=EdgeKey("doc_0", "Source", "Neighbor", "related to"),
        source_docs=["doc_0", "doc_1"],
        mandatory_docs=["doc_1"],
        optional_docs=[],
        source_context={
            "doc_id": "doc_0",
            "entity": "Source",
            "qa_pairs": [
                {
                    "question": "Who is Source related to?",
                    "answer": "Neighbor",
                    "answer_refs": ["doc_0::e9"],
                }
            ],
            "relationships": {"related to": ["Neighbor"]},
            "roles": ["person"],
            "states": ["historical figure"],
        },
        mandatory_contexts=[
            {
                "doc_id": "doc_1",
                "entity": "Neighbor",
                "qa_pairs": [
                    {
                        "question": "When did Neighbor die?",
                        "answer": "20 March 851",
                        "answer_refs": ["doc_1::e2"],
                    }
                ],
                "relationships": {"died on": ["20 March 851"]},
                "roles": ["person"],
                "states": ["deceased"],
            }
        ],
        optional_contexts=[],
        constraints={"must_be_chain": True, "reverse_required": True},
        budget={"max_depth": 1, "max_calls": 2, "max_tokens": 3000},
    )
    path_proofs = scheduler._mine_path_proofs(packet, include_optional=False, max_paths=4)
    assert path_proofs

    captured = {}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        assert tool_name == "create_bridge_qa"
        captured["bridge"] = arguments["bridges"][0]
        return [{"success": True, "bridge_id": "bridge_subject_ref"}]

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)
    drop_reason_counts = scheduler._new_drop_reason_counts()
    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(
        packet,
        [
            {
                "question": "When did Source's neighbor die?",
                "answers": ["doc_1::e2"],
                "reverse_question": "Whose neighbor died on 20 March 851?",
                "reverse_answers": ["Source"],
                "source_docs": ["doc_0", "doc_1"],
                "reasoning": "Sub-Q1 ... Sub-Q2 ...",
            }
        ],
        path_proofs=path_proofs,
        drop_reason_counts=drop_reason_counts,
    )

    assert accepted == 1
    assert rejected == 0
    assert cap_hit is False
    assert captured["bridge"]["reverse_answers"] == ["doc_0::e1"]
    assert drop_reason_counts["missing_source_ref"] == 0


def test_mine_path_proofs_round_robin_balances_orientations():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )
    packet = EdgePacket(
        edge=EdgeKey("doc_0", "Source", "Neighbor", "related to"),
        source_docs=["doc_0", "doc_1", "doc_2"],
        mandatory_docs=["doc_1", "doc_2"],
        optional_docs=[],
        source_context={
            "doc_id": "doc_0",
            "entity": "Source",
            "qa_pairs": [
                {
                    "question": "Who is Source related to?",
                    "answer": "Neighbor",
                    "answer_refs": ["doc_0::e1"],
                },
                {
                    "question": "Who was married to Neighbor?",
                    "answer": "Source",
                    "answer_refs": ["doc_0::e1"],
                },
            ],
            "relationships": {"related to": ["Neighbor"]},
            "roles": ["person"],
            "states": ["historical figure"],
        },
        mandatory_contexts=[
            {
                "doc_id": "doc_1",
                "entity": "Neighbor",
                "qa_pairs": [
                    {
                        "question": "When did Neighbor die?",
                        "answer": "20 March 851",
                        "answer_refs": ["doc_1::e2"],
                    }
                ],
                "relationships": {"died on": ["20 March 851"]},
                "roles": ["person"],
                "states": ["deceased"],
            },
            {
                "doc_id": "doc_2",
                "entity": "Neighbor",
                "qa_pairs": [
                    {
                        "question": "Where was Neighbor buried?",
                        "answer": "Abbey X",
                        "answer_refs": ["doc_2::e5"],
                    }
                ],
                "relationships": {"buried in": ["Abbey X"]},
                "roles": ["person"],
                "states": ["historical figure"],
            },
        ],
        optional_contexts=[],
        constraints={"must_be_chain": True, "reverse_required": True},
        budget={"max_depth": 1, "max_calls": 2, "max_tokens": 3000},
    )
    proofs = scheduler._mine_path_proofs(packet, include_optional=False, max_paths=2)
    assert len(proofs) == 2
    orientations = {
        scheduler._source_link_orientation(proof.source_fact, packet.edge.neighbor_name)
        for proof in proofs
    }
    assert orientations == {"neighbor_as_answer", "neighbor_in_question"}


def test_scheduler_enforces_edge_token_budget():
    reconciler = _FakeReconciler(token_increment=220)
    reconciler.stage_payloads["worker"] = [
        json.dumps({"status": "ok", "candidates": [], "need_recursion": True, "notes": "more"}),
    ]
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=2, edge_max_tokens=250)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=250,
        max_optional_docs_per_edge=2,
    )

    summary = scheduler.run_document("doc_0")
    edge_result = summary["edge_results"][0]

    assert edge_result["budget_exhausted"] is True
    assert reconciler.rlm_metrics["edges_explored"] == 1


def test_scheduler_repairs_non_id_entity_refs_before_create_bridge(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    captured = {}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        assert tool_name == "create_bridge_qa"
        captured["bridge"] = arguments["bridges"][0]
        return [{"success": True, "bridge_id": "bridge_x"}]

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)

    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(
        _build_packet(),
        [
            {
                "question": "When did Source's neighbor die?",
                "answers": ["doc_1::20 March 851"],
                "reverse_question": "Who is Source related to?",
                "reverse_answers": ["Neighbor"],
                "source_docs": ["doc_0", "doc_1"],
                "reasoning": "Sub-Q1 ... Sub-Q2 ...",
            }
        ],
    )

    assert accepted == 1
    assert rejected == 0
    assert cap_hit is False
    assert captured["bridge"]["answers"] == ["doc_1::e2"]
    assert captured["bridge"]["reverse_answers"] == ["doc_0::e1"]


def test_scheduler_drops_unresolved_entity_refs_before_create_bridge(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    call_counter = {"create_bridge_qa": 0}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "create_bridge_qa":
            call_counter["create_bridge_qa"] += 1
        return [{"success": True, "bridge_id": "bridge_x"}]

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)

    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(
        _build_packet(),
        [
            {
                "question": "What was unresolved?",
                "answers": ["Unknown Entity"],
                "reverse_question": "Whose unresolved entity?",
                "reverse_answers": ["doc_1::NotARealEntity"],
                "source_docs": ["doc_0", "doc_1"],
                "reasoning": "Sub-Q1 ... Sub-Q2 ...",
            }
        ],
    )

    assert accepted == 0
    assert rejected == 1
    assert cap_hit is False
    assert call_counter["create_bridge_qa"] == 0


def test_scheduler_early_stops_after_first_bridge_acceptance():
    reconciler = _FakeReconciler(token_increment=80)
    reconciler.stage_payloads["worker"] = [
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "When did Source's neighbor die?",
                        "answers": ["doc_1::e2"],
                        "reverse_question": "Whose neighbor died on 20 March 851?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_1"],
                        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": True,
                "notes": "ok",
            }
        ),
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=2, edge_max_tokens=3000)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=3000,
        max_optional_docs_per_edge=2,
    )

    summary = scheduler.run_document("doc_0")

    assert summary["bridges_created"] == 1
    assert reconciler.rlm_metrics["worker_calls"] == 1
    assert reconciler.tools.active_neighbor_focus is None


def test_scheduler_does_not_mark_doc_explored_when_edges_remain(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    def _stuck_run_edge(_edge):
        return EdgeRunResult(
            edge=_edge,
            accepted=0,
            attempted=0,
            rejected=0,
            budget_exhausted=False,
            edge_tokens=0,
            property_attempted=False,
        )

    monkeypatch.setattr(scheduler, "_run_edge", _stuck_run_edge)

    summary = scheduler.run_document("doc_0")

    assert summary["pending_edges"] > 0
    assert "doc_0" not in reconciler.tools.explored_documents


def test_stage_token_metrics_accounting_for_root_and_worker():
    reconciler = _FakeReconciler(token_increment=90)
    reconciler.stage_payloads["root"] = [json.dumps({"selected_doc_ids": ["doc_3"]})]
    reconciler.stage_payloads["worker"] = [
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "Where was Source's neighbor buried?",
                        "answers": ["doc_2::e5"],
                        "reverse_question": "Whose neighbor was buried in Abbey X?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_2"],
                        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": False,
                "notes": "ok",
            }
        )
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=2, edge_max_tokens=3000)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=3000,
        max_optional_docs_per_edge=1,
    )

    scheduler.run_document("doc_0")

    assert reconciler.rlm_metrics["root_calls"] >= 1
    assert reconciler.rlm_metrics["worker_calls"] >= 1
    assert reconciler.rlm_metrics["root_input_tokens"] > 0
    assert reconciler.rlm_metrics["worker_input_tokens"] > 0
    assert reconciler.tokens_used >= (reconciler.rlm_metrics["root_input_tokens"] + reconciler.rlm_metrics["root_output_tokens"])


def test_worker_payload_includes_source_context():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    messages = worker._build_messages(
        packet=_build_packet(),
        depth=0,
        include_optional=False,
    )

    user_payload = messages[1]["content"].split("EDGE_PACKET:\n", 1)[1]
    parsed = json.loads(user_payload)

    assert "source_context" in parsed
    assert parsed["source_context"]["doc_id"] == "doc_0"
    assert parsed["source_context"]["entity"] == "Source"


def test_worker_prompt_contains_rich_legacy_examples():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    messages = worker._build_messages(
        packet=_build_packet(),
        depth=0,
        include_optional=False,
    )
    system_prompt = messages[0]["content"]

    assert f"Prompt version: {WORKER_PROMPT_VERSION}" in system_prompt
    assert "GOOD #1" in system_prompt
    assert "GOOD #4" in system_prompt
    assert "BAD #1 (REVERSE_INVALID independent reverse)" in system_prompt
    assert "BAD #4 (CIRCULAR reverse)" in system_prompt
    assert "Forward chain dependency" in system_prompt
    assert "Forward target variable = X" in system_prompt


def test_render_prompt_contains_rich_legacy_examples_and_grounding_rule():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    messages = worker._build_render_messages(
        RenderInput(edge=_build_packet().edge, proofs=[], constraints={"must_be_chain": True})
    )
    system_prompt = messages[0]["content"]

    assert f"Prompt version: {WORKER_PROMPT_VERSION}" in system_prompt
    assert "GOOD #2" in system_prompt
    assert "BAD #4 (CIRCULAR reverse)" in system_prompt
    assert "Render mode grounding rule" in system_prompt
    assert "Use ONLY the provided path_proofs" in system_prompt


def test_worker_normalize_candidates_is_not_truncated():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    candidates = []
    for idx in range(5):
        candidates.append(
            {
                "question": f"When did Source's neighbor die #{idx}?",
                "answers": ["doc_1::e2"],
                "reverse_question": f"Whose neighbor died on 20 March 851? #{idx}",
                "reverse_answers": ["doc_0::e1"],
                "source_docs": ["doc_0", "doc_1"],
                "reasoning": "Sub-Q1 ... Sub-Q2 ...",
            }
        )

    normalized = worker._normalize_candidates(candidates)
    assert len(normalized) == 5


def test_scheduler_dedupes_identical_candidates_before_create_bridge(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    call_counter = {"create_bridge_qa": 0}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "create_bridge_qa":
            call_counter["create_bridge_qa"] += 1
            return [{"success": True, "bridge_id": "bridge_x"}]
        return {"success": True}

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)

    dup_candidate = {
        "question": "When did Source's neighbor die?",
        "answers": ["doc_1::20 March 851"],
        "reverse_question": "Who is Source related to?",
        "reverse_answers": ["Neighbor"],
        "source_docs": ["doc_0", "doc_1"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
    }

    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(
        _build_packet(),
        [dup_candidate, dict(dup_candidate)],
    )

    assert accepted == 1
    assert rejected == 0
    assert cap_hit is False
    assert call_counter["create_bridge_qa"] == 1


def test_scheduler_canonicalizes_source_docs_to_ref_docs_only(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    captured = {}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "create_bridge_qa":
            captured["bridge"] = arguments["bridges"][0]
            return [{"success": True, "bridge_id": "bridge_x"}]
        return {"success": True}

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)

    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(
        _build_packet(),
        [
            {
                "question": "When did Source's neighbor die?",
                "answers": ["doc_1::e2"],
                "reverse_question": "Whose neighbor died on 20 March 851?",
                "reverse_answers": ["doc_0::e1"],
                "source_docs": ["doc_0", "doc_1", "doc_2", "doc_4"],
                "reasoning": "Sub-Q1 ... Sub-Q2 ...",
            }
        ],
    )

    assert accepted == 1
    assert rejected == 0
    assert cap_hit is False
    assert captured["bridge"]["source_docs"] == ["doc_0", "doc_1"]


def test_scheduler_enforces_edge_accept_cap_with_preverified_commit(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    reconciler.bridge_verifier_enabled = True
    reconciler.edge_max_accepted_bridges = 2
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    verifier_calls = {"count": 0}

    def _fake_verify_gate(_bridge_spec):
        verifier_calls["count"] += 1
        return {"pass": True, "score": 0.99, "failure_codes": [], "notes": "ok"}, True

    reconciler.verify_bridge_candidate_gate = _fake_verify_gate

    create_calls = {"count": 0}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "create_bridge_qa":
            create_calls["count"] += 1
            assert arguments.get("__preverified") is True
            return [{"success": True, "bridge_id": f"bridge_{create_calls['count']}"}]
        return {"success": True}

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)

    candidates = []
    for idx in range(4):
        candidates.append(
            {
                "question": f"When did Source's neighbor die #{idx}?",
                "answers": ["doc_1::e2"],
                "reverse_question": f"Whose neighbor died on 20 March 851? #{idx}",
                "reverse_answers": ["doc_0::e1"],
                "source_docs": ["doc_0", "doc_1"],
                "reasoning": "Sub-Q1 ... Sub-Q2 ...",
            }
        )

    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(_build_packet(), candidates)

    assert accepted == 2
    assert rejected == 0
    assert cap_hit is True
    assert verifier_calls["count"] == 4
    assert create_calls["count"] == 2
    assert reconciler.rlm_metrics["edge_accept_cap_hits"] >= 1


def test_scheduler_reverifies_high_similarity_sibling_at_commit(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    reconciler.bridge_verifier_enabled = True
    reconciler.edge_max_accepted_bridges = 8
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    verifier_calls = {"count": 0}

    def _fake_verify_gate(bridge_spec):
        verifier_calls["count"] += 1
        similarity_max = scheduler._max_similarity_from_signal(bridge_spec.get("similarity_signal", []))
        if similarity_max >= 0.70:
            return {"pass": False, "score": 0.2, "failure_codes": ["NEAR_DUPLICATE"], "notes": "dup"}, False
        return {"pass": True, "score": 0.99, "failure_codes": [], "notes": "ok"}, True

    reconciler.verify_bridge_candidate_gate = _fake_verify_gate

    create_calls = {"count": 0}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "create_bridge_qa":
            create_calls["count"] += 1
            assert arguments.get("__preverified") is True
            create_args = dict(arguments)
            create_args.pop("__preverified", None)
            return reconciler.tools.create_bridge_qa(**create_args)
        return {"success": True}

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)

    candidates = [
        {
            "question": "When did Source's neighbor die?",
            "answers": ["doc_1::e2"],
            "reverse_question": "Whose neighbor died on 20 March 851?",
            "reverse_answers": ["doc_0::e1"],
            "source_docs": ["doc_0", "doc_1"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        },
        {
            "question": "When did the neighbor of Source die?",
            "answers": ["doc_1::e2"],
            "reverse_question": "Whose neighbor died on 20 March 851?",
            "reverse_answers": ["doc_0::e1"],
            "source_docs": ["doc_0", "doc_1"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        },
    ]

    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(_build_packet(), candidates)

    assert accepted == 1
    assert rejected == 1
    assert cap_hit is False
    assert verifier_calls["count"] == 3
    assert create_calls["count"] == 1
    assert len(reconciler.tools.bridges_created) == 1


def test_scheduler_keeps_low_similarity_commit_on_fast_path(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    reconciler.bridge_verifier_enabled = True
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=2,
    )

    verifier_calls = {"count": 0}

    def _fake_verify_gate(_bridge_spec):
        verifier_calls["count"] += 1
        return {"pass": True, "score": 0.99, "failure_codes": [], "notes": "ok"}, True

    reconciler.verify_bridge_candidate_gate = _fake_verify_gate

    create_calls = {"count": 0}

    def _fake_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "create_bridge_qa":
            create_calls["count"] += 1
            assert arguments.get("__preverified") is True
            create_args = dict(arguments)
            create_args.pop("__preverified", None)
            return reconciler.tools.create_bridge_qa(**create_args)
        return {"success": True}

    monkeypatch.setattr(scheduler, "_call_tool", _fake_call_tool)

    accepted, rejected, cap_hit, retry_brief = scheduler._apply_candidates(
        _build_packet(),
        [
            {
                "question": "When did Source's neighbor die?",
                "answers": ["doc_1::e2"],
                "reverse_question": "Whose neighbor died on 20 March 851?",
                "reverse_answers": ["doc_0::e1"],
                "source_docs": ["doc_0", "doc_1"],
                "reasoning": "Sub-Q1 ... Sub-Q2 ...",
            }
        ],
    )

    assert accepted == 1
    assert rejected == 0
    assert cap_hit is False
    assert verifier_calls["count"] == 1
    assert create_calls["count"] == 1


def test_scheduler_stops_repeated_failed_candidate_signatures():
    reconciler = _FakeReconciler(token_increment=40)
    repeated_payload = json.dumps(
        {
            "status": "ok",
            "candidates": [
                {
                    "question": "reject bridge candidate",
                    "answers": ["doc_1::e2"],
                    "reverse_question": "reject reverse candidate",
                    "reverse_answers": ["doc_0::e1"],
                    "source_docs": ["doc_0", "doc_1"],
                    "reasoning": "same failing chain",
                }
            ],
            "need_recursion": False,
            "notes": "same candidate",
        }
    )
    reconciler.stage_payloads["worker"] = [repeated_payload] * 5

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=0, max_calls=5, edge_max_tokens=5000)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=0,
        edge_max_calls=5,
        edge_max_tokens=5000,
        max_optional_docs_per_edge=0,
    )

    summary = scheduler.run_document("doc_0")

    assert summary["bridges_created"] == 0
    assert reconciler.rlm_metrics["worker_calls"] < 5


def test_low_signal_edges_use_minimal_probe_budget():
    reconciler = _FakeReconciler(token_increment=25)
    # Make exact neighbor coverage empty (only in current doc).
    reconciler.tools.entity_to_docs["neighbor"] = {"doc_0"}
    reconciler.tools.entity_degrees["neighbor"] = 1

    # Keep optional fuzzy results available, but low-signal policy should still cap calls.
    reconciler.stage_payloads["worker"] = [
        json.dumps(
            {
                "status": "ok",
                "candidates": [],
                "need_recursion": False,
                "notes": "no path",
            }
        )
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=5, edge_max_tokens=10000)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=5,
        edge_max_tokens=10000,
        max_optional_docs_per_edge=2,
    )

    summary = scheduler.run_document("doc_0")
    assert summary["bridges_created"] == 0
    assert reconciler.rlm_metrics["worker_calls"] == 1


def test_rlm_edge_summary_includes_worker_prompt_version():
    reconciler = _FakeReconciler(token_increment=25)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    reconciler.stage_payloads["worker"] = [
        json.dumps({"status": "ok", "candidates": [], "need_recursion": False, "notes": "none"})
    ]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=0, max_calls=1, edge_max_tokens=1000)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=0,
        edge_max_calls=1,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
    )

    scheduler.run_document("doc_0")

    edge_events = [e for e in events if e["event"] == "rlm_edge_summary"]
    assert edge_events
    assert edge_events[0]["data"]["worker_prompt_version"] == WORKER_PROMPT_VERSION
    assert edge_events[0]["data"]["close_reason"] == "no_candidates_no_progression"
    assert edge_events[0]["data"]["worker_calls_made"] == 1
    assert edge_events[0]["data"]["close_detail"]
    assert edge_events[0]["data"]["drop_reason_counts"]["missing_source_ref"] == 0
    assert edge_events[0]["data"]["drop_reason_counts"]["missing_target_ref"] == 0
    assert edge_events[0]["data"]["drop_reason_counts"]["reverse_not_invertible"] == 0
    assert edge_events[0]["data"]["drop_reason_counts"]["no_cross_doc_evidence"] == 0


def test_root_scheduler_guided_retry_same_edge():
    reconciler = _FakeReconciler(token_increment=40)
    reconciler.bridge_verifier_enabled = True
    reconciler.stage_payloads["worker"] = [
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "When did Source's neighbor die?",
                        "answers": ["doc_1::e2"],
                        "reverse_question": "Whose neighbor died on 20 March 851?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_1"],
                        "reasoning": "Forward target variable = 20 March 851. Reverse target variable = Source. Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": False,
                "notes": "first attempt",
            }
        ),
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "When did the mother of Source's neighbor die?",
                        "answers": ["doc_1::e2"],
                        "reverse_question": "Whose neighbor's mother died on 20 March 851?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_1"],
                        "reasoning": "Forward target variable = 20 March 851. Reverse target variable = Source. Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": False,
                "notes": "retry attempt",
            }
        ),
    ]

    def _fake_verify_gate(bridge_spec):
        if "mother" in bridge_spec.get("question", "").lower():
            return {"pass": True, "score": 0.97, "failure_codes": [], "notes": "ok", "retryable": False, "retry_hint": "stop_edge", "retry_reason": "accepted"}, True
        return {
            "pass": False,
            "score": 0.31,
            "failure_codes": ["REVERSE_INVALID"],
            "notes": "reverse mapping is weak",
            "retryable": True,
            "retry_hint": "fix_reverse_inversion",
            "retry_reason": "Keep the same edge and repair the reverse inversion.",
        }, False

    reconciler.verify_bridge_candidate_gate = _fake_verify_gate

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=3, edge_max_tokens=5000)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=3,
        edge_max_tokens=5000,
        max_optional_docs_per_edge=1,
    )

    summary = scheduler.run_document("doc_0")

    assert summary["bridges_created"] == 1
    assert reconciler.rlm_metrics["guided_retry_scheduled"] == 1
    assert reconciler.rlm_metrics["guided_retry_started"] == 1
    assert reconciler.rlm_metrics["guided_retry_succeeded"] == 1
    assert summary["edge_results"][0]["guided_retry_count"] == 1
    assert summary["edge_results"][0]["last_retry_hint"] == "fix_reverse_inversion"


def test_hybrid_parallel_requeues_guided_retry_edge(monkeypatch):
    reconciler = _FakeReconciler(token_increment=40)
    reconciler.bridge_verifier_enabled = True
    reconciler.stage_payloads["worker"] = [
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "When did Source's neighbor die?",
                        "answers": ["doc_1::e2"],
                        "reverse_question": "Whose neighbor died on 20 March 851?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_1"],
                        "reasoning": "Forward target variable = 20 March 851. Reverse target variable = Source. Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": False,
                "notes": "first attempt",
            }
        ),
        json.dumps(
            {
                "status": "ok",
                "candidates": [
                    {
                        "question": "Where was Source's neighbor buried?",
                        "answers": ["doc_2::e5"],
                        "reverse_question": "Whose neighbor was buried in Abbey X?",
                        "reverse_answers": ["doc_0::e1"],
                        "source_docs": ["doc_0", "doc_2"],
                        "reasoning": "Forward target variable = Abbey X. Reverse target variable = Source. Sub-Q1 ... Sub-Q2 ...",
                    }
                ],
                "need_recursion": False,
                "notes": "retry attempt",
            }
        ),
    ]

    def _fake_verify_gate(bridge_spec):
        if "buried" in bridge_spec.get("question", "").lower():
            return {"pass": True, "score": 0.96, "failure_codes": [], "notes": "ok", "retryable": False, "retry_hint": "stop_edge", "retry_reason": "accepted"}, True
        return {
            "pass": False,
            "score": 0.28,
            "failure_codes": ["NOT_CHAIN"],
            "notes": "a different proof on the same edge looks more promising",
            "retryable": True,
            "retry_hint": "use_different_path_proof",
            "retry_reason": "Try a different path proof on the same edge.",
        }, False

    reconciler.verify_bridge_candidate_gate = _fake_verify_gate

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=1, max_calls=2, edge_max_tokens=5000)
    retry_briefs = []
    original_generate = worker.generate

    def _wrapped_generate(*args, **kwargs):
        retry_briefs.append(kwargs.get("retry_brief"))
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(worker, "generate", _wrapped_generate)

    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=5000,
        max_optional_docs_per_edge=1,
        hybrid_scope="doc_edge",
        edge_parallel_enabled=True,
        edge_parallel_workers=2,
    )

    summary = scheduler.run_document("doc_0")

    assert summary["bridges_created"] == 1
    assert reconciler.rlm_metrics["guided_retry_scheduled"] == 1
    assert reconciler.rlm_metrics["guided_retry_started"] >= 1
    assert retry_briefs[0] is None
    assert any(brief is not None for brief in retry_briefs[1:])
    assert any(result["guided_retry_count"] == 1 for result in summary["edge_results"])
    assert scheduler._pending_retry_briefs == {}


def test_hybrid_stop_edge_emits_non_invoked_worker_event_and_close_reason(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=0, max_calls=1, edge_max_tokens=1000)
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=0,
        edge_max_calls=1,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="edge",
    )
    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")

    original_call_tool = scheduler._call_tool

    def _patched_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "plan_neighbor_doc_coverage":
            return {
                "doc_id": "doc_0",
                "entity_name": "Source",
                "neighbor_name": "Neighbor",
                "relationship": "related to",
                "mandatory_docs": [],
                "optional_fuzzy_docs": [],
                "explored_mandatory_docs": [],
                "pending_mandatory_docs": [],
                "ready_to_close": True,
            }
        return original_call_tool(tool_name, arguments, doc_id=doc_id)

    def _always_stop(*_args, **_kwargs):
        return {"action": "stop_edge", "source": "planner", "reason": "No reliable cross-doc evidence."}

    monkeypatch.setattr(scheduler, "_call_tool", _patched_call_tool)
    monkeypatch.setattr(scheduler, "_plan_edge_action", _always_stop)

    edge_result = scheduler._run_edge(edge)

    assert edge_result.accepted == 0
    assert reconciler.rlm_metrics["worker_calls"] == 0

    worker_call_events = [e for e in events if e["event"] == "rlm_worker_call"]
    assert worker_call_events
    assert worker_call_events[0]["data"]["worker_invoked"] is False
    assert worker_call_events[0]["data"]["planner_action"] == "stop_edge"

    edge_events = [e for e in events if e["event"] == "rlm_edge_summary"]
    assert edge_events
    assert edge_events[0]["data"]["close_reason"] == "planner_stop"
    assert "Planner stop accepted" in edge_events[0]["data"]["close_detail"]


def test_rlm_parse_error_propagates_into_close_detail():
    reconciler = _FakeReconciler(token_increment=20)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    reconciler.stage_payloads["worker"] = ["not-json", "still-not-json", "still not json"]

    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=0, max_calls=1, edge_max_tokens=1000)
    scheduler = RootScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=0,
        edge_max_calls=1,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
    )

    scheduler.run_document("doc_0")

    edge_events = [e for e in events if e["event"] == "rlm_edge_summary"]
    assert edge_events
    edge_data = edge_events[0]["data"]
    assert edge_data["close_reason"] == "no_candidates_no_progression"
    assert "parse_stage=regenerate_retry" in edge_data["close_detail"]
    assert "parse_attempts=3" in edge_data["close_detail"]
    assert "Worker parse error" in edge_data["close_detail"]

    worker_result_events = [e for e in events if e["event"] == "rlm_worker_result"]
    assert worker_result_events
    assert worker_result_events[0]["data"]["parse_stage"] == "regenerate_retry"
    assert worker_result_events[0]["data"]["parse_attempts"] == 3
    assert worker_result_events[0]["data"]["parse_recovery_used"] is True
    assert worker_result_events[0]["data"]["rejected_delta"] == 0
    assert reconciler.rlm_metrics["worker_parse_recovery_fail"] == 1


def test_hybrid_doc_scope_uses_planner_edge_choice():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
    )

    edges = [
        EdgeKey("doc_0", "SourceA", "NeighborA", "relA"),
        EdgeKey("doc_0", "SourceB", "NeighborB", "relB"),
    ]
    reconciler.stage_payloads["root"] = [json.dumps({"edge_index": 1, "reason": "Edge 1 has higher cross-doc signal."})]

    selected = scheduler._choose_edge_for_doc("doc_0", edges)
    assert selected == edges[1]
    assert reconciler.rlm_metrics["planner_decisions"] >= 1
    assert reconciler.rlm_metrics["planner_fallbacks"] == 0


def test_hybrid_planner_prompts_include_version_examples_and_checklist():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
    )

    corpus_messages = scheduler._build_planner_messages("corpus_doc_selection", {"candidates": []})
    doc_edge_messages = scheduler._build_planner_messages("doc_edge_selection", {"candidates": []})
    edge_action_messages = scheduler._build_planner_messages("edge_action", {"state": {"max_depth": 1}})

    corpus_system = corpus_messages[0]["content"]
    doc_edge_system = doc_edge_messages[0]["content"]
    edge_action_system = edge_action_messages[0]["content"]

    assert PLANNER_PROMPT_VERSION in corpus_system
    assert "Planner checklist" in corpus_system
    assert "Decision: corpus_doc_selection" in corpus_system
    assert "GOOD #1" in corpus_system and "BAD #2" in corpus_system
    assert "Decision: doc_edge_selection" in doc_edge_system
    assert "Decision: edge_action" in edge_action_system
    assert "repeated reverse-invalid style failures" in edge_action_system
    assert "BAD #3 (wasteful continuation after unchanged failures)" in edge_action_system


def test_hybrid_edge_scope_keeps_deterministic_edge_choice():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="edge",
    )

    edges = [
        EdgeKey("doc_0", "SourceA", "NeighborA", "relA"),
        EdgeKey("doc_0", "SourceB", "NeighborB", "relB"),
    ]
    reconciler.stage_payloads["root"] = [json.dumps({"edge_index": 1, "reason": "Selecting second edge."})]

    selected = scheduler._choose_edge_for_doc("doc_0", edges)
    assert selected == edges[0]
    assert reconciler.rlm_metrics["planner_decisions"] == 0


def test_hybrid_invalid_edge_choice_falls_back_deterministic():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
    )

    edges = [
        EdgeKey("doc_0", "SourceA", "NeighborA", "relA"),
        EdgeKey("doc_0", "SourceB", "NeighborB", "relB"),
    ]
    reconciler.stage_payloads["root"] = [json.dumps({"edge_index": 99, "reason": "Invalid index for fallback test."})]

    selected = scheduler._choose_edge_for_doc("doc_0", edges)
    assert selected == edges[0]
    assert reconciler.rlm_metrics["planner_fallbacks"] >= 1


def test_hybrid_missing_reason_falls_back_deterministic():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
    )

    edges = [
        EdgeKey("doc_0", "SourceA", "NeighborA", "relA"),
        EdgeKey("doc_0", "SourceB", "NeighborB", "relB"),
    ]
    reconciler.stage_payloads["root"] = [
        json.dumps({"edge_index": 1}),
        json.dumps({"edge_index": 1}),
    ]

    selected = scheduler._choose_edge_for_doc("doc_0", edges)
    assert selected == edges[0]
    assert reconciler.rlm_metrics["planner_fallbacks"] >= 1


def test_hybrid_corpus_scope_uses_planner_doc_choice():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="corpus_doc_edge",
    )

    plan = {
        "next_doc": {"doc_id": "doc_0", "status": "unplanned"},
        "queue": [
            {"doc_id": "doc_0", "status": "unplanned", "pending_neighbors": 1, "pending_entities": 1},
            {"doc_id": "doc_1", "status": "in_progress", "pending_neighbors": 5, "pending_entities": 3},
        ],
    }
    reconciler.stage_payloads["root"] = [json.dumps({"doc_id": "doc_1", "reason": "Highest pending neighbors."})]

    selected = scheduler._choose_doc_for_corpus(plan)
    assert selected == "doc_1"


def test_hybrid_planner_repair_retry_applies_valid_second_attempt():
    reconciler = _FakeReconciler(token_increment=10)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
    )
    edges = [
        EdgeKey("doc_0", "SourceA", "NeighborA", "relA"),
        EdgeKey("doc_0", "SourceB", "NeighborB", "relB"),
    ]
    reconciler.stage_payloads["root"] = [
        "not-json",
        json.dumps({"edge_index": 1, "reason": "Edge 1 has stronger pending signal."}),
    ]

    selected = scheduler._choose_edge_for_doc("doc_0", edges)
    assert selected == edges[1]
    assert reconciler.rlm_metrics["planner_fallbacks"] == 0
    assert reconciler.rlm_metrics["root_calls"] == 2

    planner_events = [e for e in events if e["event"] == "hybrid_planner_decision"]
    assert planner_events
    latest = planner_events[-1]["data"]
    assert latest["source"] == "planner"
    assert latest["attempt"] == "repair_retry"
    assert latest["planner_prompt_version"] == PLANNER_PROMPT_VERSION


def test_hybrid_invalid_edge_action_falls_back_to_deterministic():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="edge",
    )
    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")
    reconciler.stage_payloads["root"] = [
        json.dumps({"action": "run_worker", "depth": 99, "include_optional": False, "reason": "Try unsupported deep step."})
    ]

    decision = scheduler._plan_edge_action(
        edge=edge,
        current_depth=0,
        current_include_optional=False,
        call_count=0,
        remaining_edge_tokens=1000,
        accepted_total=0,
        attempted_total=0,
        rejected_total=0,
        no_candidate_calls=0,
        has_optional_contexts=False,
    )
    assert decision["action"] == "run_worker"
    assert decision["depth"] == 0
    assert decision["source"] == "deterministic"
    assert reconciler.rlm_metrics["planner_fallbacks"] >= 1


def test_hybrid_planner_retry_failure_emits_deterministic_fallback_event():
    reconciler = _FakeReconciler(token_increment=10)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="corpus_doc_edge",
    )

    plan = {
        "next_doc": {"doc_id": "doc_0", "status": "unplanned"},
        "queue": [
            {"doc_id": "doc_0", "status": "unplanned", "pending_neighbors": 1, "pending_entities": 1},
            {"doc_id": "doc_1", "status": "in_progress", "pending_neighbors": 5, "pending_entities": 3},
        ],
    }
    reconciler.stage_payloads["root"] = ["not-json", "still-not-json"]

    selected = scheduler._choose_doc_for_corpus(plan)
    assert selected == "doc_0"
    assert reconciler.rlm_metrics["planner_fallbacks"] >= 1

    fallback_events = [
        e["data"]
        for e in events
        if e["event"] == "hybrid_planner_decision" and e["data"].get("source") == "deterministic_fallback"
    ]
    assert fallback_events
    assert fallback_events[-1]["fallback_reason"] == "planner_unresolved_after_retry"
    assert fallback_events[-1]["planner_prompt_version"] == PLANNER_PROMPT_VERSION


def test_hybrid_stop_action_blocked_by_pending_mandatory_docs_falls_back_to_worker(monkeypatch):
    reconciler = _FakeReconciler(token_increment=20)
    reconciler.stage_payloads["worker"] = [
        json.dumps({"status": "ok", "candidates": [], "need_recursion": False, "notes": "none"})
    ]
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model", max_depth=0, max_calls=1, edge_max_tokens=2000)
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=0,
        edge_max_calls=1,
        edge_max_tokens=2000,
        max_optional_docs_per_edge=0,
        hybrid_scope="edge",
    )
    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")

    coverage_calls = {"count": 0}
    original_call_tool = scheduler._call_tool

    def _patched_call_tool(tool_name, arguments, doc_id=None):
        if tool_name == "plan_neighbor_doc_coverage":
            coverage_calls["count"] += 1
            if coverage_calls["count"] == 1:
                return {
                    "doc_id": "doc_0",
                    "entity_name": "Source",
                    "neighbor_name": "Neighbor",
                    "relationship": "related to",
                    "mandatory_docs": [],
                    "optional_fuzzy_docs": [],
                    "explored_mandatory_docs": [],
                    "pending_mandatory_docs": [],
                    "ready_to_close": True,
                }
            return {
                "doc_id": "doc_0",
                "entity_name": "Source",
                "neighbor_name": "Neighbor",
                "relationship": "related to",
                "mandatory_docs": ["doc_1"],
                "optional_fuzzy_docs": [],
                "explored_mandatory_docs": [],
                "pending_mandatory_docs": ["doc_1"],
                "ready_to_close": False,
            }
        return original_call_tool(tool_name, arguments, doc_id=doc_id)

    def _always_stop(*_args, **_kwargs):
        return {"action": "stop_edge", "source": "planner"}

    monkeypatch.setattr(scheduler, "_call_tool", _patched_call_tool)
    monkeypatch.setattr(scheduler, "_plan_edge_action", _always_stop)

    edge_result = scheduler._run_edge(edge)

    assert edge_result.attempted == 0
    assert reconciler.rlm_metrics["planner_fallbacks"] >= 1
    assert reconciler.rlm_metrics["worker_calls"] == 0
    assert reconciler.rlm_metrics["edges_skipped_no_path"] >= 1


def test_hybrid_parallel_packetization_releases_focus_lock():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=1,
        hybrid_scope="doc_edge",
        edge_parallel_enabled=True,
        edge_parallel_workers=2,
    )
    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")

    prepared = scheduler._prepare_edge_work_parallel(edge)
    assert prepared is not None
    assert reconciler.tools.active_neighbor_focus is None


def test_hybrid_parallel_run_document_commits_in_deterministic_batch_order(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
        edge_parallel_enabled=True,
        edge_parallel_workers=2,
    )

    edge_a = EdgeKey("doc_0", "SourceA", "NeighborA", "relA")
    edge_b = EdgeKey("doc_0", "SourceB", "NeighborB", "relB")
    reconciler.tools.doc_exploration_plans["doc_0"] = {
        "doc_id": "doc_0",
        "entities_to_explore": [
            {
                "name": "SourceA",
                "roles": ["person"],
                "status": "pending",
                "neighbors": [{"entity": "NeighborA", "relationship": "relA", "other_docs": ["doc_1"], "status": "pending"}],
            },
            {
                "name": "SourceB",
                "roles": ["person"],
                "status": "pending",
                "neighbors": [{"entity": "NeighborB", "relationship": "relB", "other_docs": ["doc_2"], "status": "pending"}],
            },
        ],
        "total_neighbors_to_explore": 2,
        "explored_count": 0,
        "pending_count": 2,
    }

    packet_template = _build_packet()

    def _packet_for(edge: EdgeKey) -> EdgePacket:
        return EdgePacket(
            edge=edge,
            source_docs=list(packet_template.source_docs),
            mandatory_docs=list(packet_template.mandatory_docs),
            optional_docs=[],
            source_context=dict(packet_template.source_context),
            mandatory_contexts=list(packet_template.mandatory_contexts),
            optional_contexts=[],
            constraints=dict(packet_template.constraints),
            budget=dict(packet_template.budget),
        )

    def _fake_prepare(edge: EdgeKey) -> PreparedEdgeWork:
        return PreparedEdgeWork(
            edge=edge,
            packet=_packet_for(edge),
            signal={"tier": "medium", "score": 1.2, "probe_only": False},
            edge_call_limit=1,
            edge_token_limit=1000,
            viability={"path_proof_count": 1},
        )

    def _fake_generate(work: PreparedEdgeWork) -> EdgeWorkerProposal:
        if work.edge == edge_a:
            time.sleep(0.02)
        return EdgeWorkerProposal(
            edge=work.edge,
            packet=work.packet,
            signal=work.signal,
            attempted=0,
            prefilter_rejected=0,
            sanitized_candidates=[],
            budget_exhausted=False,
            edge_tokens=0,
            worker_calls_made=1,
            no_candidate_calls=1,
            close_reason="planner_stop",
            close_detail="stop",
            planner_decisions=0,
            planner_fallbacks=0,
            planner_stop_actions=0,
        )

    commit_order: List[Tuple[str, str]] = []

    def _fake_commit(proposal: EdgeWorkerProposal) -> EdgeRunResult:
        commit_order.append((proposal.edge.entity_name, proposal.edge.neighbor_name))
        return EdgeRunResult(
            edge=proposal.edge,
            accepted=0,
            attempted=proposal.attempted,
            rejected=proposal.prefilter_rejected,
            budget_exhausted=False,
            edge_tokens=proposal.edge_tokens,
            property_attempted=False,
        )

    call_count = {"iter": 0}

    def _fake_iter_pending_edges(_doc_id: str):
        call_count["iter"] += 1
        if call_count["iter"] <= 2:
            return [edge_a, edge_b]
        return []

    monkeypatch.setattr(scheduler, "_prepare_edge_work_parallel", _fake_prepare)
    monkeypatch.setattr(scheduler, "_generate_parallel_worker_proposal", _fake_generate)
    monkeypatch.setattr(scheduler, "_commit_edge_worker_proposal", _fake_commit)
    monkeypatch.setattr(scheduler, "_iter_pending_edges", _fake_iter_pending_edges)

    summary = scheduler.run_document("doc_0")
    assert summary["mode"] == "hybrid"
    assert commit_order == [("SourceA", "NeighborA"), ("SourceB", "NeighborB")]
    assert reconciler.rlm_metrics["parallel_batches"] == 1
    assert reconciler.rlm_metrics["parallel_workers_used"] == 2
    assert reconciler.rlm_metrics["commit_queue_size_peak"] == 2
    assert any(e["event"] == "parallel_batch_started" for e in events)
    assert any(e["event"] == "parallel_batch_progress" for e in events)
    assert any(e["event"] == "parallel_batch_completed" for e in events)
    assert any(e["event"] == "edge_commit_started" for e in events)
    assert any(e["event"] == "edge_commit_finished" for e in events)
    batch_completed_idx = next(i for i, e in enumerate(events) if e["event"] == "parallel_batch_completed")
    first_commit_idx = next(i for i, e in enumerate(events) if e["event"] == "edge_commit_started")
    assert batch_completed_idx < first_commit_idx


def test_hybrid_parallel_stale_edge_is_skipped(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
        edge_parallel_enabled=True,
        edge_parallel_workers=2,
    )
    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")

    def _fake_prepare(_edge: EdgeKey) -> PreparedEdgeWork:
        return PreparedEdgeWork(
            edge=edge,
            packet=_build_packet(),
            signal={"tier": "low", "score": 0.2, "probe_only": True},
            edge_call_limit=1,
            edge_token_limit=1000,
            viability={"path_proof_count": 0},
        )

    def _fake_generate(work: PreparedEdgeWork) -> EdgeWorkerProposal:
        return EdgeWorkerProposal(
            edge=work.edge,
            packet=work.packet,
            signal=work.signal,
            attempted=0,
            prefilter_rejected=0,
            sanitized_candidates=[],
            budget_exhausted=False,
            edge_tokens=0,
            worker_calls_made=0,
            no_candidate_calls=0,
            close_reason="no_viable_path_proof",
            close_detail="none",
            planner_decisions=0,
            planner_fallbacks=0,
            planner_stop_actions=0,
        )

    call_count = {"iter": 0}

    def _fake_iter_pending_edges(_doc_id: str):
        call_count["iter"] += 1
        if call_count["iter"] <= 2:
            return [edge]
        return []

    monkeypatch.setattr(scheduler, "_prepare_edge_work_parallel", _fake_prepare)
    monkeypatch.setattr(scheduler, "_generate_parallel_worker_proposal", _fake_generate)
    monkeypatch.setattr(scheduler, "_iter_pending_edges", _fake_iter_pending_edges)
    monkeypatch.setattr(scheduler, "_is_edge_pending", lambda _edge: False)

    summary = scheduler.run_document("doc_0")
    assert summary["mode"] == "hybrid"
    assert reconciler.rlm_metrics["stale_edge_skipped"] >= 1

    stale_events = [
        e["data"]
        for e in events
        if e["event"] == "rlm_edge_summary" and e["data"].get("close_reason") == "stale_edge_skipped"
    ]
    assert stale_events


def test_hybrid_parallel_heartbeat_and_stuck_warning_visibility(monkeypatch):
    reconciler = _FakeReconciler(token_increment=10)
    events: List[Dict[str, Any]] = []

    def _capture(event_type, data):
        events.append({"event": event_type, "data": data})

    reconciler.output_callback = _capture
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    scheduler = HybridScheduler(
        reconciler=reconciler,
        worker=worker,
        edge_max_depth=1,
        edge_max_calls=2,
        edge_max_tokens=1000,
        max_optional_docs_per_edge=0,
        hybrid_scope="doc_edge",
        edge_parallel_enabled=True,
        edge_parallel_workers=2,
        parallel_progress_heartbeat_seconds=0.01,
        parallel_stuck_warning_seconds=0.02,
    )
    edge = EdgeKey("doc_0", "Source", "Neighbor", "related to")

    def _fake_prepare(_edge: EdgeKey) -> PreparedEdgeWork:
        return PreparedEdgeWork(
            edge=edge,
            packet=_build_packet(),
            signal={"tier": "medium", "score": 1.2, "probe_only": False},
            edge_call_limit=1,
            edge_token_limit=1000,
            viability={"path_proof_count": 1},
        )

    def _fake_generate(work: PreparedEdgeWork) -> EdgeWorkerProposal:
        time.sleep(0.2)
        return EdgeWorkerProposal(
            edge=work.edge,
            packet=work.packet,
            signal=work.signal,
            attempted=0,
            prefilter_rejected=0,
            sanitized_candidates=[],
            budget_exhausted=False,
            edge_tokens=0,
            worker_calls_made=1,
            no_candidate_calls=1,
            close_reason="planner_stop",
            close_detail="stop",
            planner_decisions=0,
            planner_fallbacks=0,
            planner_stop_actions=0,
        )

    call_count = {"iter": 0}

    def _fake_iter_pending_edges(_doc_id: str):
        call_count["iter"] += 1
        if call_count["iter"] <= 2:
            return [edge]
        return []

    monkeypatch.setattr(scheduler, "_prepare_edge_work_parallel", _fake_prepare)
    monkeypatch.setattr(scheduler, "_generate_parallel_worker_proposal", _fake_generate)
    monkeypatch.setattr(scheduler, "_iter_pending_edges", _fake_iter_pending_edges)

    summary = scheduler.run_document("doc_0")
    assert summary["mode"] == "hybrid"
    assert any(e["event"] == "parallel_batch_started" for e in events)
    assert any(e["event"] == "parallel_batch_heartbeat" for e in events)
    assert any(e["event"] == "parallel_worker_stuck_warning" for e in events)
    assert any(e["event"] == "parallel_batch_completed" for e in events)
    timeout_like = [
        e for e in events
        if e["event"] == "rlm_edge_summary"
        and "timeout" in str((e.get("data") or {}).get("close_reason", "")).lower()
    ]
    assert not timeout_like


def test_worker_prompt_includes_curriculum_guidance_block():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    packet = _build_packet()
    packet.constraints = {
        **packet.constraints,
        "curriculum_guidance": {
            "prior_query_demand_sketch": {
                "top_patterns": [
                    {
                        "pattern": "mother",
                        "count": 2,
                        "representative_query": "When did Mira's mother die?",
                    }
                ],
                "batches_seen": 1,
            },
            "prior_batches_seen": 1,
            "prior_batch_feedback_brief": {
                "summary_lines": ["Weak coverage: mother x2"],
            },
            "bridge_exemplars": [
                {
                    "question": "When did Mira's mother die?",
                    "reverse_question": "Whose mother died in 912?",
                    "pattern_tags": ["mother", "death_date"],
                }
            ],
        },
    }

    messages = worker._build_messages(packet=packet, depth=0, include_optional=False)
    user_prompt = messages[1]["content"]

    assert "Prior Query Demand Sketch:" in user_prompt
    assert "- mother x2" in user_prompt
    assert "Prior Batch Feedback Brief:" in user_prompt
    assert "Weak coverage: mother x2" in user_prompt
    assert "Relevant Prior Bridge Exemplars:" in user_prompt
    assert "Forward: When did Mira's mother die?" in user_prompt
    assert "Reverse: Whose mother died in 912?" in user_prompt
    assert "Current Batch Demand Sketch:" not in user_prompt


def test_render_prompt_includes_curriculum_guidance_block():
    reconciler = _FakeReconciler(token_increment=10)
    worker = RecursiveEdgeWorker(reconciler, model_name="worker-model")
    render_input = RenderInput(
        edge=_build_packet().edge,
        proofs=[],
        constraints={
            "must_be_chain": True,
            "curriculum_guidance": {
                "current_batch_demand_sketch": {
                    "top_patterns": [
                        {
                            "pattern": "burial_place",
                            "count": 1,
                            "representative_query": "Where was Mira buried?",
                        }
                    ],
                    "batches_seen": 1,
                },
                "prior_batches_seen": 1,
                "prior_batch_feedback_brief": {
                    "summary_lines": ["Doc-backed more than bridge-backed: burial_place x1"],
                },
                "bridge_exemplars": [
                    {
                        "question": "Where was Mira buried?",
                        "reverse_question": "Who was buried in Abbey X?",
                        "pattern_tags": ["burial_place"],
                    }
                ],
            },
        },
    )

    messages = worker._build_render_messages(render_input)
    user_prompt = messages[1]["content"]

    assert "Prior Query Demand Sketch:" in user_prompt
    assert "- burial_place x1" in user_prompt
    assert "Prior Batch Feedback Brief:" in user_prompt
    assert "Doc-backed more than bridge-backed: burial_place x1" in user_prompt
    assert "Relevant Prior Bridge Exemplars:" in user_prompt
    assert "Forward: Where was Mira buried?" in user_prompt
    assert "Current Batch Demand Sketch:" not in user_prompt
