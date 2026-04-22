import types

import pytest

from gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler


class _DummySearcher:
    def __init__(self):
        self.gsw_by_doc_id = {"doc_0": object()}
        self.entities = [{"name": "Source", "doc_id": "doc_0"}]

    def search(self, query, top_k=10, verbose=False):
        return []


class _DummyMessage:
    def __init__(self, content=None, tool_calls=None, reasoning=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content


class _DummyChoice:
    def __init__(self, message):
        self.message = message


class _DummyUsage:
    def __init__(self, prompt_tokens=1, completion_tokens=1):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _DummyResponse:
    def __init__(self, message, prompt_tokens=1, completion_tokens=1):
        self.choices = [_DummyChoice(message)]
        self.usage = _DummyUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class _FakeServiceUnavailableError(Exception):
    pass


class _FakeAPIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class _FakeLiteLLM:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.call_count = 0
        self.exceptions = types.SimpleNamespace(
            ServiceUnavailableError=_FakeServiceUnavailableError,
            APIError=_FakeAPIError,
        )

    def completion(self, **kwargs):
        if self.call_count >= len(self._sequence):
            raise RuntimeError("No more fake responses configured")
        item = self._sequence[self.call_count]
        self.call_count += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeOpenAICompletions:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.call_count = 0

    def create(self, **kwargs):
        if self.call_count >= len(self._sequence):
            raise RuntimeError("No more fake responses configured")
        item = self._sequence[self.call_count]
        self.call_count += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeOpenAIClient:
    def __init__(self, sequence):
        self.chat = types.SimpleNamespace(completions=_FakeOpenAICompletions(sequence))


def _build_bedrock_agent(monkeypatch, sequence):
    litellm_client = _FakeLiteLLM(sequence)

    def _fake_init(self, model_name, base_url=None):
        self.provider = "bedrock"
        self.client = None
        self.litellm = litellm_client

    monkeypatch.setattr(AgenticReconciler, "_initialize_client", _fake_init)
    agent = AgenticReconciler(
        entity_searcher=_DummySearcher(),
        model_name="bedrock/openai.gpt-oss-120b-1:0",
        verbose=False,
        bridge_verifier_enabled=False,
    )
    return agent, litellm_client


def _build_openai_agent(monkeypatch, responses):
    openai_client = _FakeOpenAIClient(responses)
    litellm_client = _FakeLiteLLM([])

    def _fake_init(self, model_name, base_url=None):
        self.provider = "openai"
        self.client = openai_client
        self.litellm = litellm_client

    monkeypatch.setattr(AgenticReconciler, "_initialize_client", _fake_init)
    agent = AgenticReconciler(
        entity_searcher=_DummySearcher(),
        model_name="gpt-4o-mini",
        verbose=False,
        bridge_verifier_enabled=False,
    )
    return agent, openai_client, litellm_client


def test_bedrock_stage_call_retries_on_transient_service_unavailable(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "gsw_memory.sleep_time.agentic_reconciler.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    agent, litellm_client = _build_bedrock_agent(
        monkeypatch,
        [
            _FakeServiceUnavailableError("temporary outage"),
            _DummyResponse(_DummyMessage(content="ok")),
        ],
    )

    message = agent._call_model_for_stage(
        [{"role": "user", "content": "hello"}],
        stage="worker",
        max_tokens=256,
    )

    assert message.content == "ok"
    assert litellm_client.call_count == 2
    assert sleep_calls == [1.0]


def test_bedrock_stage_call_raises_after_retry_exhaustion(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "gsw_memory.sleep_time.agentic_reconciler.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    error = _FakeServiceUnavailableError("still unavailable")
    agent, litellm_client = _build_bedrock_agent(
        monkeypatch,
        [error, error, error, error],
    )

    with pytest.raises(_FakeServiceUnavailableError):
        agent._call_model_for_stage(
            [{"role": "user", "content": "hello"}],
            stage="worker",
            max_tokens=256,
        )

    assert litellm_client.call_count == 4
    assert sleep_calls == [1.0, 2.0, 4.0]


def test_bedrock_stage_call_does_not_retry_non_retryable_api_error(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "gsw_memory.sleep_time.agentic_reconciler.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    agent, litellm_client = _build_bedrock_agent(
        monkeypatch,
        [_FakeAPIError("bad request", status_code=400)],
    )

    with pytest.raises(_FakeAPIError):
        agent._call_model_for_stage(
            [{"role": "user", "content": "hello"}],
            stage="worker",
            max_tokens=256,
        )

    assert litellm_client.call_count == 1
    assert sleep_calls == []


def test_bedrock_legacy_loop_uses_same_retry_helper(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "gsw_memory.sleep_time.agentic_reconciler.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    agent, litellm_client = _build_bedrock_agent(
        monkeypatch,
        [
            _FakeServiceUnavailableError("temporary outage"),
            _DummyResponse(_DummyMessage(content="done", tool_calls=[])),
        ],
    )

    result = agent.explore_entity(entity_name="Source", max_iterations=1)

    assert result["iterations"] == 1
    assert result["tool_calls"] == 0
    assert litellm_client.call_count == 2
    assert sleep_calls == [1.0]


def test_non_bedrock_stage_call_skips_bedrock_retry_wrapper(monkeypatch):
    response = _DummyResponse(_DummyMessage(content="openai-ok"))
    agent, openai_client, litellm_client = _build_openai_agent(monkeypatch, [response])

    message = agent._call_model_for_stage(
        [{"role": "user", "content": "hello"}],
        stage="worker",
        max_tokens=256,
    )

    assert message.content == "openai-ok"
    assert openai_client.chat.completions.call_count == 1
    assert litellm_client.call_count == 0
