import io
import json
import logging
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from playground.sleep_time.run_sleep_time import SleepTimeRunner
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
