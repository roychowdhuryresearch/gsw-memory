import types

from gsw_memory.query_then_sleep.transport import StageTransport


class _DummyFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _DummyToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _DummyFunction(name, arguments)


class _DummyMessage:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content
        self.reasoning = reasoning


class _DummyChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _DummyUsage:
    def __init__(self, prompt_tokens=1, completion_tokens=1):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _DummyResponse:
    def __init__(self, message, finish_reason="stop", prompt_tokens=1, completion_tokens=1):
        self.choices = [_DummyChoice(message, finish_reason=finish_reason)]
        self.usage = _DummyUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class _FakeOpenAICompletions:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.call_count = 0

    def create(self, **kwargs):
        item = self._sequence[self.call_count]
        self.call_count += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeOpenAIClient:
    def __init__(self, sequence):
        self.chat = types.SimpleNamespace(completions=_FakeOpenAICompletions(sequence))


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
        item = self._sequence[self.call_count]
        self.call_count += 1
        if isinstance(item, Exception):
            raise item
        return item


def test_stage_transport_normalizes_openai_tool_calls(monkeypatch):
    openai_client = _FakeOpenAIClient(
        [
            _DummyResponse(
                _DummyMessage(
                    content="",
                    tool_calls=[_DummyToolCall("call_1", "search_entities", '{"query": "ermengarde"}')],
                ),
                prompt_tokens=12,
                completion_tokens=5,
            )
        ]
    )

    def _fake_init(self):
        self.provider = "openai"
        self.model_name = self.original_model_name
        self.client = openai_client
        self.litellm = None

    monkeypatch.setattr(StageTransport, "_initialize_client", _fake_init)

    transport = StageTransport("gpt-4o-mini")
    response = transport.chat(
        [{"role": "user", "content": "find Ermengarde"}],
        tools=[{"type": "function", "function": {"name": "search_entities", "parameters": {"type": "object"}}}],
        tool_choice="auto",
    )

    assert response.provider == "openai"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 5
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "search_entities"
    assert response.tool_calls[0].arguments == {"query": "ermengarde"}


def test_stage_transport_retries_bedrock_calls(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("gsw_memory.query_then_sleep.transport.time.sleep", lambda seconds: sleep_calls.append(seconds))

    litellm_client = _FakeLiteLLM(
        [
            _FakeServiceUnavailableError("temporary outage"),
            _DummyResponse(_DummyMessage(content="ok"), prompt_tokens=3, completion_tokens=4),
        ]
    )

    def _fake_init(self):
        self.provider = "bedrock"
        self.model_name = "bedrock/converse/openai.gpt-oss-120b-1:0"
        self.client = None
        self.litellm = litellm_client

    monkeypatch.setattr(StageTransport, "_initialize_client", _fake_init)

    transport = StageTransport("bedrock/openai.gpt-oss-120b-1:0")
    response = transport.chat([{"role": "user", "content": "say ok"}], stage="query")

    assert response.text == "ok"
    assert litellm_client.call_count == 2
    assert sleep_calls == [1.0]
