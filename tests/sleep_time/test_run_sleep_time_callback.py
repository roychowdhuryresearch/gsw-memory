import io
import json
import logging
import sys
import threading
import uuid
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.sleep_time.run_sleep_time import SleepTimeRunner, run_sleep_time_once
from gsw_memory.sleep_time.rlm_pipeline import PLANNER_PROMPT_VERSION


def _build_stream_logger(prefix: str):
    stream = io.StringIO()
    logger = logging.getLogger(f"test.{prefix}.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, stream


def _build_runner_for_callback_tests():
    runner = SleepTimeRunner.__new__(SleepTimeRunner)
    runner.trace_logger, runner._trace_stream = _build_stream_logger("trace")
    runner.tool_logger, runner._tool_stream = _build_stream_logger("tool")
    return runner


def _patch_runner_init_logging(monkeypatch):
    def _fake_setup_logging(self):
        logger, _ = _build_stream_logger("runner")
        self.logger = logger
        self.error_logger = logger
        self.trace_logger = logger
        self.tool_logger = logger

    monkeypatch.setattr(SleepTimeRunner, "_setup_logging", _fake_setup_logging)


def test_hybrid_planner_decision_logs_trace_and_structured_tool_event():
    runner = _build_runner_for_callback_tests()
    callback = runner._create_callback_handler(display=None)

    payload = {
        "decision_type": "edge_action",
        "source": "planner",
        "reason": "Escalate once with optional contexts.",
        "attempt": "repair_retry",
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
    }
    callback("hybrid_planner_decision", payload)

    trace_text = runner._trace_stream.getvalue()
    assert "[planner]" in trace_text
    assert '"decision_type": "edge_action"' in trace_text
    assert '"reason": "Escalate once with optional contexts."' in trace_text

    tool_lines = [line for line in runner._tool_stream.getvalue().splitlines() if line.strip()]
    assert len(tool_lines) == 1
    event = json.loads(tool_lines[0])
    assert event["event_type"] == "hybrid_planner_decision"
    assert event["data"]["decision_type"] == "edge_action"
    assert event["data"]["source"] == "planner"
    assert event["data"]["reason"] == "Escalate once with optional contexts."
    assert "timestamp" in event and isinstance(event["timestamp"], str) and event["timestamp"]


def test_sleep_time_runner_registers_signal_handlers_only_in_main_thread(monkeypatch, tmp_path):
    _patch_runner_init_logging(monkeypatch)
    signal_calls = []

    def _fake_signal(sig, handler):
        signal_calls.append((threading.current_thread().name, sig, handler))

    monkeypatch.setattr("playground.sleep_time.run_sleep_time.signal.signal", _fake_signal)

    args = SimpleNamespace(output_dir=str(tmp_path))

    main_runner = SleepTimeRunner(args)
    assert main_runner is not None
    assert len(signal_calls) == 2
    assert all(call[0] == threading.main_thread().name for call in signal_calls)

    signal_calls.clear()
    errors = []

    def _build_in_thread():
        try:
            SleepTimeRunner(args)
        except Exception as exc:  # pragma: no cover - test should not hit
            errors.append(exc)

    worker = threading.Thread(target=_build_in_thread, name="sleep-time-test-worker")
    worker.start()
    worker.join()

    assert not errors
    assert signal_calls == []


def test_generate_run_id_is_unique_even_for_same_timestamp(monkeypatch):
    fixed_now = real_datetime(2026, 4, 2, 16, 35, 57, 123456)
    uuid_values = iter(
        [
            SimpleNamespace(hex="aaaabbbbccccdddd1111222233334444"),
            SimpleNamespace(hex="eeeeffff000011112222333344445555"),
        ]
    )

    class _FixedDatetime:
        @staticmethod
        def now():
            return fixed_now

    monkeypatch.setattr("playground.sleep_time.run_sleep_time.datetime", _FixedDatetime)
    monkeypatch.setattr("playground.sleep_time.run_sleep_time.uuid.uuid4", lambda: next(uuid_values))

    first = SleepTimeRunner._generate_run_id()
    second = SleepTimeRunner._generate_run_id()

    assert first == "20260402_163557_123456_aaaabbbb"
    assert second == "20260402_163557_123456_eeeeffff"
    assert first != second


def test_hybrid_planner_fallback_logs_fallback_reason():
    runner = _build_runner_for_callback_tests()
    callback = runner._create_callback_handler(display=None)

    payload = {
        "decision_type": "doc_edge_selection",
        "source": "deterministic_fallback",
        "fallback_reason": "planner_unresolved_after_retry",
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
    }
    callback("hybrid_planner_decision", payload)

    trace_text = runner._trace_stream.getvalue()
    assert "[planner]" in trace_text
    assert '"fallback_reason": "planner_unresolved_after_retry"' in trace_text

    tool_lines = [line for line in runner._tool_stream.getvalue().splitlines() if line.strip()]
    assert len(tool_lines) == 1
    event = json.loads(tool_lines[0])
    assert event["data"]["source"] == "deterministic_fallback"
    assert event["data"]["fallback_reason"] == "planner_unresolved_after_retry"


def test_hybrid_planner_decision_display_only_when_display_present():
    runner = _build_runner_for_callback_tests()

    class _DisplayStub:
        def __init__(self):
            self.calls = []

        def show_planner_decision(self, data):
            self.calls.append(data)

    display = _DisplayStub()
    callback_with_display = runner._create_callback_handler(display=display)
    callback_without_display = runner._create_callback_handler(display=None)

    payload = {
        "decision_type": "corpus_doc_selection",
        "source": "planner",
        "reason": "Continue in-progress doc.",
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
    }
    callback_with_display("hybrid_planner_decision", payload)
    callback_without_display("hybrid_planner_decision", payload)

    assert len(display.calls) == 1
    assert display.calls[0]["decision_type"] == "corpus_doc_selection"


def test_worker_events_log_trace_and_structured_tool_events():
    runner = _build_runner_for_callback_tests()
    callback = runner._create_callback_handler(display=None)

    call_payload = {
        "edge": {"doc_id": "doc_0", "entity_name": "Source", "neighbor_name": "Neighbor", "relationship": "related to"},
        "call_index": 1,
        "depth": 0,
        "include_optional": False,
        "remaining_edge_tokens": 1200,
        "planner_action": "run_worker",
        "planner_source": "planner",
        "planner_reason": "Explore the edge further.",
        "worker_invoked": True,
    }
    result_payload = {
        "edge": {"doc_id": "doc_0", "entity_name": "Source", "neighbor_name": "Neighbor", "relationship": "related to"},
        "call_index": 1,
        "candidates_count": 0,
        "need_recursion": False,
        "parse_stage": "initial",
        "worker_notes": "No usable path proofs.",
        "accepted_delta": 0,
        "rejected_delta": 1,
    }

    callback("rlm_worker_call", call_payload)
    callback("rlm_worker_result", result_payload)

    trace_text = runner._trace_stream.getvalue()
    assert "[worker_call]" in trace_text
    assert "[worker_result]" in trace_text
    assert '"planner_action": "run_worker"' in trace_text
    assert '"rejected_delta": 1' in trace_text

    tool_lines = [line for line in runner._tool_stream.getvalue().splitlines() if line.strip()]
    assert len(tool_lines) == 2
    first = json.loads(tool_lines[0])
    second = json.loads(tool_lines[1])
    assert first["event_type"] == "rlm_worker_call"
    assert second["event_type"] == "rlm_worker_result"
    assert first["data"]["worker_invoked"] is True
    assert second["data"]["worker_notes"] == "No usable path proofs."


def test_verifier_result_logs_trace_and_structured_tool_event():
    runner = _build_runner_for_callback_tests()
    callback = runner._create_callback_handler(display=None)

    payload = {
        "question": "When did Source's neighbor die?",
        "reverse_question": "Whose neighbor died on 20 March 851?",
        "source_docs": ["doc_0", "doc_1"],
        "verifier_pass": False,
        "verifier_raw_pass": False,
        "score": 0.31,
        "failure_codes": ["REVERSE_INVALID"],
        "notes": "reverse mapping is weak",
        "similarity_gate_applied": False,
        "similarity_max": 0.0,
        "similarity_gate_blocked_by": [],
        "retryable": True,
        "retry_hint": "fix_reverse_inversion",
        "retry_reason": "Keep the same edge and repair the reverse inversion.",
        "parse_stage": "initial",
        "parse_attempts": 1,
        "parse_recovery_used": False,
        "verifier_model": "test-model",
        "verifier_threshold": 0.7,
    }
    callback("verifier_result", payload)

    trace_text = runner._trace_stream.getvalue()
    assert "[verifier]" in trace_text
    assert '"failure_codes": ["REVERSE_INVALID"]' in trace_text
    assert '"retry_hint": "fix_reverse_inversion"' in trace_text
    assert '"notes": "reverse mapping is weak"' in trace_text

    tool_lines = [line for line in runner._tool_stream.getvalue().splitlines() if line.strip()]
    assert len(tool_lines) == 1
    event = json.loads(tool_lines[0])
    assert event["event_type"] == "verifier_result"
    assert event["data"]["verifier_pass"] is False
    assert event["data"]["failure_codes"] == ["REVERSE_INVALID"]
    assert event["data"]["retry_hint"] == "fix_reverse_inversion"


def test_verifier_result_display_only_when_display_present():
    runner = _build_runner_for_callback_tests()

    class _DisplayStub:
        def __init__(self):
            self.calls = []

        def show_verifier_event(self, data):
            self.calls.append(data)

    display = _DisplayStub()
    callback_with_display = runner._create_callback_handler(display=display)
    callback_without_display = runner._create_callback_handler(display=None)

    payload = {
        "question": "Q",
        "reverse_question": "RQ",
        "verifier_pass": False,
        "score": 0.22,
        "failure_codes": ["NOT_CHAIN"],
        "notes": "independent facts",
    }
    callback_with_display("verifier_result", payload)
    callback_without_display("verifier_result", payload)

    assert len(display.calls) == 1
    assert display.calls[0]["failure_codes"] == ["NOT_CHAIN"]


def test_worker_event_display_only_when_display_present():
    runner = _build_runner_for_callback_tests()

    class _DisplayStub:
        def __init__(self):
            self.worker_calls = []

        def show_worker_event(self, event_type, data):
            self.worker_calls.append((event_type, data))

    display = _DisplayStub()
    callback_with_display = runner._create_callback_handler(display=display)
    callback_without_display = runner._create_callback_handler(display=None)

    payload = {
        "edge": {"doc_id": "doc_0"},
        "call_index": 1,
        "depth": 0,
        "include_optional": False,
        "remaining_edge_tokens": 1200,
        "planner_action": "run_worker",
        "planner_source": "planner",
        "planner_reason": "Explore.",
        "worker_invoked": True,
    }
    callback_with_display("rlm_worker_call", payload)
    callback_without_display("rlm_worker_call", payload)

    assert len(display.worker_calls) == 1
    assert display.worker_calls[0][0] == "rlm_worker_call"


def test_parallel_stage_events_log_trace_and_tool_stream():
    runner = _build_runner_for_callback_tests()
    callback = runner._create_callback_handler(display=None)

    payload = {
        "edge": {"doc_id": "doc_0", "entity_name": "Source", "neighbor_name": "Neighbor", "relationship": "related to"},
        "call_index": 1,
    }

    callback("edge_packet_prepared", payload)
    callback("edge_worker_dispatched", payload)
    callback("edge_worker_completed", payload)
    callback("edge_commit_started", payload)
    callback("edge_commit_finished", payload)

    trace_text = runner._trace_stream.getvalue()
    assert "[edge_packet_prepared]" in trace_text
    assert "[edge_worker_dispatched]" in trace_text
    assert "[edge_worker_completed]" in trace_text
    assert "[edge_commit_started]" in trace_text
    assert "[edge_commit_finished]" in trace_text

    tool_lines = [line for line in runner._tool_stream.getvalue().splitlines() if line.strip()]
    assert len(tool_lines) == 5
    event_types = [json.loads(line)["event_type"] for line in tool_lines]
    assert event_types == [
        "edge_packet_prepared",
        "edge_worker_dispatched",
        "edge_worker_completed",
        "edge_commit_started",
        "edge_commit_finished",
    ]


def test_rlm_doc_summary_logs_trace_and_structured_tool_event():
    runner = _build_runner_for_callback_tests()
    callback = runner._create_callback_handler(display=None)

    payload = {
        "entity": "doc_0",
        "bridges_created": 3,
        "pending_edges": 1,
        "mode": "hybrid",
    }
    callback("rlm_doc_summary", payload)

    trace_text = runner._trace_stream.getvalue()
    assert "[rlm_doc]" in trace_text
    assert '"entity": "doc_0"' in trace_text

    tool_lines = [line for line in runner._tool_stream.getvalue().splitlines() if line.strip()]
    assert len(tool_lines) == 1
    event = json.loads(tool_lines[0])
    assert event["event_type"] == "rlm_doc_summary"
    assert event["data"]["entity"] == "doc_0"
    assert event["data"]["bridges_created"] == 3


def test_parallel_batch_visibility_events_log_trace_and_tool_stream():
    runner = _build_runner_for_callback_tests()
    callback = runner._create_callback_handler(display=None)

    events = [
        (
            "parallel_batch_started",
            {
                "doc_id": "doc_0",
                "batch_id": 1,
                "completed": 0,
                "total": 2,
                "inflight_edges": [{"doc_id": "doc_0", "entity_name": "A", "neighbor_name": "B", "relationship": "r"}],
                "oldest_inflight_seconds": 0.0,
                "elapsed_seconds": 0.0,
            },
        ),
        (
            "parallel_batch_progress",
            {
                "doc_id": "doc_0",
                "batch_id": 1,
                "completed": 1,
                "total": 2,
                "inflight_edges": [{"doc_id": "doc_0", "entity_name": "C", "neighbor_name": "D", "relationship": "r"}],
                "oldest_inflight_seconds": 1.2,
                "elapsed_seconds": 1.4,
            },
        ),
        (
            "parallel_batch_heartbeat",
            {
                "doc_id": "doc_0",
                "batch_id": 1,
                "completed": 1,
                "total": 2,
                "inflight_edges": [{"doc_id": "doc_0", "entity_name": "C", "neighbor_name": "D", "relationship": "r"}],
                "oldest_inflight_seconds": 8.3,
                "elapsed_seconds": 12.0,
            },
        ),
        (
            "parallel_worker_stuck_warning",
            {
                "doc_id": "doc_0",
                "batch_id": 1,
                "completed": 1,
                "total": 2,
                "edge": {"doc_id": "doc_0", "entity_name": "C", "neighbor_name": "D", "relationship": "r"},
                "inflight_edges": [{"doc_id": "doc_0", "entity_name": "C", "neighbor_name": "D", "relationship": "r"}],
                "worker_inflight_seconds": 61.0,
                "oldest_inflight_seconds": 61.0,
                "elapsed_seconds": 61.5,
            },
        ),
        (
            "parallel_batch_completed",
            {
                "doc_id": "doc_0",
                "batch_id": 1,
                "completed": 2,
                "total": 2,
                "inflight_edges": [],
                "oldest_inflight_seconds": 0.0,
                "elapsed_seconds": 62.0,
            },
        ),
    ]

    for event_type, payload in events:
        callback(event_type, payload)

    trace_text = runner._trace_stream.getvalue()
    assert "[parallel_batch_started]" in trace_text
    assert "[parallel_batch_progress]" in trace_text
    assert "[parallel_batch_heartbeat]" in trace_text
    assert "[parallel_worker_stuck_warning]" in trace_text
    assert "[parallel_batch_completed]" in trace_text

    tool_lines = [line for line in runner._tool_stream.getvalue().splitlines() if line.strip()]
    assert len(tool_lines) == 5
    event_types = [json.loads(line)["event_type"] for line in tool_lines]
    assert event_types == [
        "parallel_batch_started",
        "parallel_batch_progress",
        "parallel_batch_heartbeat",
        "parallel_worker_stuck_warning",
        "parallel_batch_completed",
    ]


def test_parallel_batch_visibility_events_display_only_when_display_present():
    runner = _build_runner_for_callback_tests()

    class _DisplayStub:
        def __init__(self):
            self.calls = []

        def show_parallel_batch_event(self, event_type, data):
            self.calls.append((event_type, data))

    display = _DisplayStub()
    callback_with_display = runner._create_callback_handler(display=display)
    callback_without_display = runner._create_callback_handler(display=None)

    payload = {
        "doc_id": "doc_0",
        "batch_id": 3,
        "completed": 1,
        "total": 4,
        "inflight_edges": [],
        "oldest_inflight_seconds": 0.0,
        "elapsed_seconds": 3.4,
    }
    callback_with_display("parallel_batch_progress", payload)
    callback_without_display("parallel_batch_progress", payload)

    assert len(display.calls) == 1
    assert display.calls[0][0] == "parallel_batch_progress"


def test_run_sleep_time_once_accepts_injected_searcher_and_guidance(monkeypatch, tmp_path):
    captured = {}

    def _fake_run(self):
        captured["entity_searcher"] = self.entity_searcher
        captured["guidance"] = self.injected_curriculum_guidance
        self._cleanup_logging()
        return {"summary": {"ok": True}}

    monkeypatch.setattr(SleepTimeRunner, "run", _fake_run)

    args = SimpleNamespace(output_dir=str(tmp_path), model="test-model")
    injected_searcher = object()
    guidance = {"prior_batches_seen": 2}

    result = run_sleep_time_once(
        args,
        entity_searcher=injected_searcher,
        curriculum_guidance=guidance,
    )

    assert result == {"summary": {"ok": True}}
    assert captured["entity_searcher"] is injected_searcher
    assert captured["guidance"] == guidance
