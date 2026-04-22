import pytest

from gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler
from gsw_memory.sleep_time.rlm_pipeline import VerifierOutputSchema
from gsw_memory.sleep_time.prompts import SLEEP_TIME_BRIDGE_VERIFIER_PROMPT


class _DummySearcher:
    def __init__(self):
        self.gsw_by_doc_id = {}
        self.entities = []


class _DummyMessage:
    def __init__(self, content=None, parsed=None, reasoning=None, reasoning_content=None, tool_calls=None, finish_reason=None):
        self.content = content
        self.parsed = parsed
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason


class _DummyChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = getattr(message, "finish_reason", None)


class _DummyUsage:
    def __init__(self, total_tokens=1):
        self.total_tokens = total_tokens


class _DummyResponse:
    def __init__(self, message, total_tokens=1):
        self.choices = [_DummyChoice(message)]
        self.usage = _DummyUsage(total_tokens=total_tokens)


class _SequencedCompletions:
    def __init__(self, messages):
        self._messages = list(messages)
        self.call_count = 0
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.call_count >= len(self._messages):
            raise RuntimeError("No more fake responses configured")
        msg = self._messages[self.call_count]
        self.call_count += 1
        return _DummyResponse(msg)

    def completion(self, **kwargs):
        """Mimic litellm.completion() interface — same sequenced responses."""
        return self.create(**kwargs)


class _NeverCallLiteLLM:
    def __init__(self):
        self.call_count = 0
        self.calls = []

    def completion(self, **kwargs):
        self.call_count += 1
        self.calls.append(kwargs)
        raise AssertionError("LiteLLM structured path should not be used for vLLM structured calls")


class _DummyChat:
    def __init__(self, completions):
        self.completions = completions


class _DummyClient:
    def __init__(self, completions):
        self.chat = _DummyChat(completions)
        self.base_url = "http://127.0.0.1:6379/v1"


def _new_agent_for_parser():
    return AgenticReconciler.__new__(AgenticReconciler)


def _new_agent_with_fake_client(monkeypatch, messages):
    completions = _SequencedCompletions(messages)
    client = _DummyClient(completions)

    def _fake_init(self, model_name, base_url=None):
        self.provider = "openai"
        self.client = client
        self.litellm = completions  # shares response sequence for structured output calls

    monkeypatch.setattr(AgenticReconciler, "_initialize_client", _fake_init)
    agent = AgenticReconciler(
        entity_searcher=_DummySearcher(),
        model_name="gpt-4o-mini",
        verbose=False,
        bridge_verifier_enabled=True,
        bridge_verifier_fail_open=False,
    )
    return agent, completions


def _new_vllm_agent_with_fake_client(monkeypatch, messages):
    completions = _SequencedCompletions(messages)
    litellm = _NeverCallLiteLLM()
    client = _DummyClient(completions)

    def _fake_init(self, model_name, base_url=None):
        self.provider = "vllm"
        self.client = client
        self.litellm = litellm

    monkeypatch.setattr(AgenticReconciler, "_initialize_client", _fake_init)
    agent = AgenticReconciler(
        entity_searcher=_DummySearcher(),
        model_name="openai/gpt-oss-120b",
        base_url="http://127.0.0.1:6379/v1",
        verbose=False,
        bridge_verifier_enabled=True,
        bridge_verifier_fail_open=False,
    )
    return agent, completions, litellm


def test_extract_json_from_text_raw_json():
    agent = _new_agent_for_parser()
    parsed = agent._extract_json_from_text('{"pass": true, "score": 0.9, "failure_codes": [], "notes": "ok"}')
    assert parsed["pass"] is True
    assert parsed["score"] == 0.9


def test_extract_json_from_text_fenced_json():
    agent = _new_agent_for_parser()
    raw = """```json
{"pass": false, "score": 0.2, "failure_codes": ["NOT_CHAIN"], "notes": "bad"}
```"""
    parsed = agent._extract_json_from_text(raw)
    assert parsed["pass"] is False
    assert parsed["failure_codes"] == ["NOT_CHAIN"]


def test_extract_json_from_text_think_wrapped_json():
    agent = _new_agent_for_parser()
    raw = '<think>reasoning here</think>\n{"pass": true, "score": 0.88, "failure_codes": [], "notes": "ok"}'
    parsed = agent._extract_json_from_text(raw)
    assert parsed["pass"] is True
    assert parsed["score"] == 0.88


def test_extract_json_from_text_embedded_json():
    agent = _new_agent_for_parser()
    raw = 'I think this is valid. {"pass": true, "score": 0.8, "failure_codes": [], "notes": "embedded"} Thanks.'
    parsed = agent._extract_json_from_text(raw)
    assert parsed["pass"] is True
    assert parsed["notes"] == "embedded"


def test_summarize_assistant_message_fields_reports_raw_payload_inventory():
    agent = _new_agent_for_parser()
    message = _DummyMessage(
        content='{"status":"ok"}',
        parsed={"status": "ok"},
        reasoning_content="hidden reasoning",
        tool_calls=[{"id": "tool_1"}],
    )

    summary = agent._summarize_assistant_message_fields(message)

    assert summary["has_parsed"] is True
    assert summary["parsed_type"] == "dict"
    assert '"status": "ok"' in summary["parsed_preview"]
    assert summary["has_content"] is True
    assert '{"status":"ok"}' in summary["content_preview"]
    assert summary["has_reasoning_content"] is True
    assert "hidden reasoning" in summary["reasoning_content_preview"]
    assert summary["has_tool_calls"] is True


def test_call_model_for_stage_usage_includes_finish_reason(monkeypatch):
    first = _DummyMessage(content='{"pass": true, "score": 0.9, "failure_codes": [], "notes": "ok"}', finish_reason="stop")
    agent, _completions = _new_agent_with_fake_client(monkeypatch, [first])

    _message, usage = agent._call_model_for_stage(
        messages=[{"role": "user", "content": "hi"}],
        model_name="gpt-4o-mini",
        stage="worker",
        return_usage=True,
    )

    assert usage["finish_reason"] == "stop"


@pytest.mark.parametrize("raw", ["", "No JSON here"])
def test_extract_json_from_text_invalid(raw):
    agent = _new_agent_for_parser()
    with pytest.raises(ValueError):
        agent._extract_json_from_text(raw)


def test_verify_bridge_candidate_repair_retry_success(monkeypatch):
    first = _DummyMessage(content="I cannot comply with strict JSON.")
    second = _DummyMessage(content='{"pass": true, "score": 0.91, "failure_codes": [], "notes": "valid chain"}')
    agent, completions = _new_agent_with_fake_client(monkeypatch, [first, second])

    verdict = agent._verify_bridge_candidate(
        {
            "question": "Q",
            "answers": ["doc_1::e1"],
            "reverse_question": "RQ",
            "reverse_answers": ["doc_2::e2"],
            "source_docs": ["doc_1", "doc_2"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        }
    )

    assert verdict["pass"] is True
    assert verdict["parse_stage"] == "repair_retry"
    assert verdict["parse_attempts"] == 2
    assert verdict["parse_recovery_used"] is True
    assert completions.call_count == 2
    assert agent.rlm_metrics["verifier_parse_recovery_success"] == 1


def test_verify_bridge_candidate_repair_retry_parse_failure(monkeypatch):
    first = _DummyMessage(content="Not JSON")
    second = _DummyMessage(content="Still not JSON")
    third = _DummyMessage(content="Still malformed JSON")
    agent, completions = _new_agent_with_fake_client(monkeypatch, [first, second, third])

    verdict = agent._verify_bridge_candidate(
        {
            "question": "Q",
            "answers": ["doc_1::e1"],
            "reverse_question": "RQ",
            "reverse_answers": ["doc_2::e2"],
            "source_docs": ["doc_1", "doc_2"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        }
    )

    assert verdict["pass"] is False
    assert "VERIFIER_PARSE_ERROR" in verdict["failure_codes"]
    assert verdict["parse_stage"] == "regenerate_retry"
    assert verdict["parse_attempts"] == 3
    assert verdict["parse_recovery_used"] is True
    assert completions.call_count == 3
    assert agent.rlm_metrics["verifier_parse_recovery_fail"] == 1


def test_verify_bridge_candidate_regenerate_retry_success(monkeypatch):
    first = _DummyMessage(content="Not JSON")
    second = _DummyMessage(content="Still not JSON")
    third = _DummyMessage(content='{"pass": false, "score": 0.43, "failure_codes": ["NOT_CHAIN"], "notes": "regenerated"}')
    agent, completions = _new_agent_with_fake_client(monkeypatch, [first, second, third])

    verdict = agent._verify_bridge_candidate(
        {
            "question": "Q",
            "answers": ["doc_1::e1"],
            "reverse_question": "RQ",
            "reverse_answers": ["doc_2::e2"],
            "source_docs": ["doc_1", "doc_2"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        }
    )

    assert verdict["pass"] is False
    assert verdict["parse_stage"] == "regenerate_retry"
    assert verdict["parse_attempts"] == 3
    assert verdict["parse_recovery_used"] is True
    assert verdict["failure_codes"] == ["NOT_CHAIN"]
    assert completions.call_count == 3
    assert agent.rlm_metrics["verifier_parse_recovery_success"] == 1


def test_normalize_verifier_output_falls_back_to_retry_guidance():
    agent = _new_agent_for_parser()

    verdict = agent._normalize_verifier_output(
        {
            "pass": False,
            "score": 0.32,
            "failure_codes": ["NOT_CHAIN"],
            "notes": "same edge may work with a different proof",
        }
    )

    assert verdict["retryable"] is True
    assert verdict["retry_hint"] == "use_different_path_proof"
    assert verdict["retry_reason"] == "same edge may work with a different proof"


def test_normalize_verifier_output_keeps_explicit_retry_guidance():
    agent = _new_agent_for_parser()

    verdict = agent._normalize_verifier_output(
        {
            "pass": False,
            "score": 0.41,
            "failure_codes": ["REVERSE_INVALID"],
            "notes": "weak reverse",
            "retryable": True,
            "retry_hint": "fix_reverse_inversion",
            "retry_reason": "Keep the same edge and repair the reverse mapping.",
        }
    )

    assert verdict["retryable"] is True
    assert verdict["retry_hint"] == "fix_reverse_inversion"
    assert verdict["retry_reason"] == "Keep the same edge and repair the reverse mapping."


def test_verify_bridge_candidate_reads_reasoning_content(monkeypatch):
    msg = _DummyMessage(
        content="",
        reasoning_content='{"pass": true, "score": 0.86, "failure_codes": [], "notes": "from reasoning_content"}',
    )
    agent, completions = _new_agent_with_fake_client(monkeypatch, [msg])

    verdict = agent._verify_bridge_candidate(
        {
            "question": "Q",
            "answers": ["doc_1::e1"],
            "reverse_question": "RQ",
            "reverse_answers": ["doc_2::e2"],
            "source_docs": ["doc_1", "doc_2"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        }
    )

    assert verdict["pass"] is True
    assert verdict["parse_stage"] == "initial"
    assert verdict["parse_attempts"] == 1
    assert verdict["parse_recovery_used"] is False
    assert verdict["source_field"] == "reasoning_content"
    assert completions.call_count == 1


def test_vllm_structured_stage_call_preserves_pydantic_schema(monkeypatch):
    msg = _DummyMessage(
        parsed={"pass": True, "score": 0.88, "failure_codes": [], "notes": "ok"},
        content='{"pass": false, "score": 0.1, "failure_codes": ["NOT_CHAIN"], "notes": "content"}',
    )
    agent, completions, litellm = _new_vllm_agent_with_fake_client(monkeypatch, [msg])

    message = agent._call_model_for_stage(
        [{"role": "user", "content": "hello"}],
        model_name="openai/gpt-oss-120b",
        stage="verifier",
        response_format=VerifierOutputSchema,
    )

    assert message is msg
    call = completions.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == "verifier_output"
    assert call["response_format"]["json_schema"]["schema"] == VerifierOutputSchema.model_json_schema()
    assert call["response_format"]["json_schema"]["strict"] is True
    assert litellm.call_count == 0


def test_vllm_structured_verifier_prefers_parsed(monkeypatch):
    msg = _DummyMessage(
        parsed={"pass": True, "score": 0.86, "failure_codes": [], "notes": "from parsed"},
        content='{"pass": false, "score": 0.11, "failure_codes": ["NOT_CHAIN"], "notes": "from content"}',
    )
    agent, completions, litellm = _new_vllm_agent_with_fake_client(monkeypatch, [msg])

    verdict = agent._verify_bridge_candidate(
        {
            "question": "Q",
            "answers": ["doc_1::e1"],
            "reverse_question": "RQ",
            "reverse_answers": ["doc_2::e2"],
            "source_docs": ["doc_1", "doc_2"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        }
    )

    assert verdict["pass"] is True
    assert verdict["source_field"] == "parsed"
    assert completions.call_count == 1
    assert litellm.call_count == 0


def test_vllm_structured_verifier_uses_content_when_parsed_absent(monkeypatch):
    msg = _DummyMessage(
        content='{"pass": true, "score": 0.77, "failure_codes": [], "notes": "from content"}',
    )
    agent, completions, litellm = _new_vllm_agent_with_fake_client(monkeypatch, [msg])

    verdict = agent._verify_bridge_candidate(
        {
            "question": "Q",
            "answers": ["doc_1::e1"],
            "reverse_question": "RQ",
            "reverse_answers": ["doc_2::e2"],
            "source_docs": ["doc_1", "doc_2"],
            "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        }
    )

    assert verdict["pass"] is True
    assert verdict["source_field"] == "content"
    assert completions.call_count == 1
    assert litellm.call_count == 0


def test_vllm_structured_empty_parsed_is_reported(monkeypatch):
    agent, _completions, _litellm = _new_vllm_agent_with_fake_client(monkeypatch, [])
    message = _DummyMessage(
        parsed="",
        content='{"pass": true, "score": 0.9, "failure_codes": [], "notes": "content fallback not allowed"}',
    )

    raw_text, source = agent._extract_message_content_with_source(
        message,
        stage="verifier",
        structured_output=True,
    )

    assert raw_text == ""
    assert source == "parsed_empty"


def test_vllm_structured_empty_content_is_reported(monkeypatch):
    agent, _completions, _litellm = _new_vllm_agent_with_fake_client(monkeypatch, [])
    message = _DummyMessage(content="")

    raw_text, source = agent._extract_message_content_with_source(
        message,
        stage="verifier",
        structured_output=True,
    )

    assert raw_text == ""
    assert source == "content_empty"


def test_verify_bridge_candidate_gate_caches_structural_reject(monkeypatch):
    msg = _DummyMessage(content='{"pass": false, "score": 0.21, "failure_codes": ["NOT_CHAIN"], "notes": "independent facts"}')
    agent, completions = _new_agent_with_fake_client(monkeypatch, [msg])
    bridge_spec = {
        "question": "Q",
        "answers": ["doc_1::e1"],
        "reverse_question": "RQ",
        "reverse_answers": ["doc_2::e2"],
        "source_docs": ["doc_1", "doc_2"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
    }

    verdict, passes = agent.verify_bridge_candidate_gate(bridge_spec)

    assert passes is False
    assert verdict["failure_codes"] == ["NOT_CHAIN"]
    assert completions.call_count == 1
    cached = agent._lookup_rejected_bridge_cache(bridge_spec)
    assert cached is not None
    assert "NOT_CHAIN" in cached.get("failure_codes", [])


def test_verify_gate_high_similarity_blocks_near_duplicate_even_if_pass_true(monkeypatch):
    msg = _DummyMessage(content='{"pass": true, "score": 0.97, "failure_codes": ["NEAR_DUPLICATE"], "notes": "same bridge"}')
    agent, completions = _new_agent_with_fake_client(monkeypatch, [msg])
    bridge_spec = {
        "question": "Q",
        "answers": ["doc_1::e1"],
        "reverse_question": "RQ",
        "reverse_answers": ["doc_2::e2"],
        "source_docs": ["doc_1", "doc_2"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        "similarity_signal": [
            {"max_similarity": 0.91, "existing_question_snippet": "Q old", "existing_reverse_question_snippet": "RQ old"}
        ],
    }

    verdict, passes = agent.verify_bridge_candidate_gate(bridge_spec)

    assert passes is False
    assert completions.call_count == 1
    assert verdict["similarity_gate_applied"] is True
    assert verdict["similarity_max"] == 0.91
    assert verdict["similarity_gate_blocked_by"] == ["NEAR_DUPLICATE"]
    cached = agent._lookup_rejected_bridge_cache(bridge_spec)
    assert cached is not None
    assert "NEAR_DUPLICATE" in cached.get("failure_codes", [])


def test_verify_gate_high_similarity_allows_non_blocking_codes(monkeypatch):
    msg = _DummyMessage(content='{"pass": true, "score": 0.91, "failure_codes": ["LOW_SPECIFICITY"], "notes": "acceptable"}')
    agent, _completions = _new_agent_with_fake_client(monkeypatch, [msg])
    bridge_spec = {
        "question": "Q",
        "answers": ["doc_1::e1"],
        "reverse_question": "RQ",
        "reverse_answers": ["doc_2::e2"],
        "source_docs": ["doc_1", "doc_2"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        "similarity_signal": [{"max_similarity": 0.83}],
    }

    verdict, passes = agent.verify_bridge_candidate_gate(bridge_spec)

    assert passes is True
    assert verdict["similarity_gate_applied"] is True
    assert verdict["similarity_gate_blocked_by"] == []


def test_verify_gate_low_similarity_does_not_apply_strict_similarity_block(monkeypatch):
    msg = _DummyMessage(content='{"pass": true, "score": 0.96, "failure_codes": ["NEAR_DUPLICATE"], "notes": "low sim case"}')
    agent, _completions = _new_agent_with_fake_client(monkeypatch, [msg])
    bridge_spec = {
        "question": "Q",
        "answers": ["doc_1::e1"],
        "reverse_question": "RQ",
        "reverse_answers": ["doc_2::e2"],
        "source_docs": ["doc_1", "doc_2"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
        "similarity_signal": [{"max_similarity": 0.45}],
    }

    verdict, passes = agent.verify_bridge_candidate_gate(bridge_spec)

    assert passes is True
    assert verdict["similarity_gate_applied"] is False
    assert verdict["similarity_gate_blocked_by"] == []


def test_apply_verifier_result_includes_similarity_gate_diagnostics(monkeypatch):
    msg = _DummyMessage(content='{"pass": true, "score": 0.96, "failure_codes": ["CIRCULAR"], "notes": "same mapping"}')
    agent, _completions = _new_agent_with_fake_client(monkeypatch, [msg])
    result = {
        "success": True,
        "bridge_id": "bridge_x",
        "validation": {
            "valid": True,
            "confidence": 0.9,
            "similarity_signal": [{"max_similarity": 0.88}],
        },
    }
    bridge_spec = {
        "question": "Q",
        "answers": ["doc_1::e1"],
        "reverse_question": "RQ",
        "reverse_answers": ["doc_2::e2"],
        "source_docs": ["doc_1", "doc_2"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
    }

    final_result = agent._apply_verifier_to_bridge_result(result, bridge_spec)

    assert final_result["success"] is False
    validation = final_result["validation"]
    assert validation["similarity_gate_applied"] is True
    assert validation["similarity_max"] == 0.88
    assert validation["similarity_gate_blocked_by"] == ["CIRCULAR"]


def test_verifier_prompt_mentions_near_duplicate_failure_code():
    assert "NEAR_DUPLICATE" in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert "High-similarity duplicate examples (aggressive policy):" in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert "GOOD (distinct, should not get NEAR_DUPLICATE):" in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert "BAD (near-duplicate, should emit NEAR_DUPLICATE):" in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert "Similarity checklist (required when similarity_signal is present):" in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert "Compare semantic mapping, not wording style." in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert "Do not treat punctuation, tense, casing, or trivial synonym changes as meaningfully distinct." in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert '"retryable": true or false' in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert '"retry_hint": "fix_anchor|fix_reverse_inversion|change_mapping|strengthen_multihop_chain|remove_answer_leak|use_different_path_proof|stop_edge"' in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT
    assert "Retry guidance checklist:" in SLEEP_TIME_BRIDGE_VERIFIER_PROMPT


def test_execute_create_bridge_tool_uses_pre_verifier_cache(monkeypatch):
    msg = _DummyMessage(content='{"pass": true, "score": 0.91, "failure_codes": [], "notes": "ok"}')
    agent, completions = _new_agent_with_fake_client(monkeypatch, [msg])
    bridge_spec = {
        "question": "When did source die?",
        "answers": ["doc_1::e1"],
        "reverse_question": "Whose source died in 851?",
        "reverse_answers": ["doc_2::e2"],
        "source_docs": ["doc_1", "doc_2"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
    }
    signature = agent._bridge_signature(bridge_spec)
    with agent._usage_lock:
        agent._rejected_bridge_cache[signature] = {
            "signature_hash": "deadbeefcafe",
            "failure_codes": ["NOT_CHAIN"],
            "notes": "cached rejection",
            "score": 0.2,
        }

    def _unexpected_tool(**kwargs):
        raise AssertionError("tool should not be invoked on cache hit")

    result = agent._execute_create_bridge_tool(_unexpected_tool, dict(bridge_spec))

    assert result["success"] is False
    assert result["validation"]["stage"] == "pre_verifier_cache"
    assert completions.call_count == 0


def test_execute_create_bridge_tool_preverified_skips_verifier(monkeypatch):
    msg = _DummyMessage(content='{"pass": false, "score": 0.01, "failure_codes": ["NOT_CHAIN"], "notes": "should not run"}')
    agent, completions = _new_agent_with_fake_client(monkeypatch, [msg])

    def _tool_method(**kwargs):
        return [
            {
                "success": True,
                "bridge_id": "bridge_x",
                "validation": {"valid": True, "confidence": 0.9},
            }
        ]

    bridge_spec = {
        "question": "When did source die?",
        "answers": ["doc_1::e1"],
        "reverse_question": "Whose source died in 851?",
        "reverse_answers": ["doc_2::e2"],
        "source_docs": ["doc_1", "doc_2"],
        "reasoning": "Sub-Q1 ... Sub-Q2 ...",
    }

    result = agent._execute_create_bridge_tool(
        _tool_method,
        {"bridges": [bridge_spec], "__preverified": True},
    )

    assert isinstance(result, list)
    assert result[0]["success"] is True
    assert result[0]["validation"]["preverified"] is True
    assert result[0]["validation"]["verifier_pass"] is True
    assert completions.call_count == 0
