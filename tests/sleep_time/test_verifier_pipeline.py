import pytest

from gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler


class _DummySearcher:
    def __init__(self):
        self.gsw_by_doc_id = {}
        self.entities = []


class _DummyMessage:
    def __init__(self, content=None, reasoning=None, reasoning_content=None):
        self.content = content
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content


class _DummyChoice:
    def __init__(self, message):
        self.message = message


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

    def create(self, **kwargs):
        if self.call_count >= len(self._messages):
            raise RuntimeError("No more fake responses configured")
        msg = self._messages[self.call_count]
        self.call_count += 1
        return _DummyResponse(msg)


class _DummyChat:
    def __init__(self, completions):
        self.completions = completions


class _DummyClient:
    def __init__(self, completions):
        self.chat = _DummyChat(completions)


def _new_agent_for_parser():
    return AgenticReconciler.__new__(AgenticReconciler)


def _new_agent_with_fake_client(monkeypatch, messages):
    completions = _SequencedCompletions(messages)
    client = _DummyClient(completions)

    def _fake_init(self, model_name, base_url=None):
        self.provider = "openai"
        self.client = client
        self.litellm = None

    monkeypatch.setattr(AgenticReconciler, "_initialize_client", _fake_init)
    agent = AgenticReconciler(
        entity_searcher=_DummySearcher(),
        model_name="gpt-4o-mini",
        verbose=False,
        bridge_verifier_enabled=True,
        bridge_verifier_fail_open=False,
    )
    return agent, completions


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
    assert completions.call_count == 2


def test_verify_bridge_candidate_repair_retry_parse_failure(monkeypatch):
    first = _DummyMessage(content="Not JSON")
    second = _DummyMessage(content="Still not JSON")
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

    assert verdict["pass"] is False
    assert "VERIFIER_PARSE_ERROR" in verdict["failure_codes"]
    assert verdict["parse_stage"] == "repair_retry"
    assert completions.call_count == 2


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
    assert verdict["source_field"] == "reasoning_content"
    assert completions.call_count == 1
