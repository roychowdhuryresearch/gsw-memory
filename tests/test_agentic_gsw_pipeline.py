import types

from pydantic import BaseModel

from gsw_memory.memory.models import Role
from gsw_memory.memory.operator_utils.agentic_gsw.assembler import assemble_gsw
from gsw_memory.memory.operator_utils.agentic_gsw.schemas import (
    AtomicFrame,
    CanonicalEntity,
    ProvisionalParticipant,
    QuestionDraft,
    QuestionPairDraft,
    RelationFrameOutput,
)
from gsw_memory.memory.operator_utils.agentic_gsw.transport_shim import TransportShim
from gsw_memory.memory.operator_utils.agentic_gsw.windowing import split_into_windows
from gsw_memory.query_then_sleep.transport import StageTransport


class _DummyMessage:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = []
        self.reasoning_content = None
        self.reasoning = None


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
    def __init__(self, content):
        self.choices = [_DummyChoice(_DummyMessage(content=content))]
        self.usage = _DummyUsage()


class _FakeOpenAICompletions:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.call_count = 0

    def create(self, **kwargs):
        item = self._sequence[self.call_count]
        self.call_count += 1
        return item


class _FakeOpenAIClient:
    def __init__(self, sequence):
        self.chat = types.SimpleNamespace(completions=_FakeOpenAICompletions(sequence))


class _FakeLiteLLM:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.call_count = 0
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        item = self._sequence[self.call_count]
        self.call_count += 1
        return item


class _SimpleOutput(BaseModel):
    value: str


def test_split_into_windows_paragraph_bleed():
    text = (
        "Alpha leads the first paragraph.\n\n"
        "Beta anchors the second paragraph.\n\n"
        "Gamma closes the third paragraph."
    )

    windows = split_into_windows(text, strategy="paragraph")

    assert len(windows) == 3
    assert windows[1].text.startswith("Alpha leads")
    assert "Gamma closes" in windows[1].text


def test_split_into_windows_sentence_fallback_overlaps():
    text = "One. Two. Three. Four. Five. Six. Seven."

    windows = split_into_windows(text, strategy="paragraph")

    assert len(windows) >= 2
    assert windows[1].char_start < windows[0].char_end


def test_assemble_gsw_builds_valid_directional_questions():
    entities = [
        CanonicalEntity(
            id="e1",
            name="Alice",
            roles=[Role(role="person", states=["sender"])],
            source_mentions=["m1"],
        ),
        CanonicalEntity(
            id="e2",
            name="Bob",
            roles=[Role(role="person", states=["recipient"])],
            source_mentions=["m2"],
        ),
    ]
    frame = AtomicFrame(
        frame_id="f_w1_1",
        vp_phrase="sent to",
        subject=ProvisionalParticipant(role="agent", ref="m1"),
        object=ProvisionalParticipant(role="recipient", ref="m2"),
        slot_kind="recipient",
        window_id="w1",
        supporting_text="Alice sent the parcel to Bob.",
        support_char_start=0,
        support_char_end=29,
    )
    pair = QuestionPairDraft(
        frame_id="f_w1_1",
        forward=QuestionDraft(
            text="To whom did Alice send the parcel?",
            answers=["e2"],
        ),
        reverse=QuestionDraft(
            text="Who sent the parcel to Bob?",
            answers=["e1"],
        ),
    )

    gsw = assemble_gsw(
        entities=entities,
        mention_map={"m1": "e1", "m2": "e2"},
        frames=[frame],
        question_pairs={"f_w1_1": pair},
        chunk_id="0_0",
    )

    assert len(gsw.entity_nodes) == 2
    assert len(gsw.verb_phrase_nodes) == 1
    assert len(gsw.verb_phrase_nodes[0].questions) == 2
    assert gsw.verb_phrase_nodes[0].questions[0].answers == ["e2"]
    assert gsw.verb_phrase_nodes[0].questions[1].answers == ["e1"]


def test_transport_shim_repairs_invalid_json(monkeypatch):
    client = _FakeOpenAIClient(
        [
            _DummyResponse("not json"),
            _DummyResponse('{"value": "ok"}'),
        ]
    )

    def _fake_init(self):
        self.provider = "openai"
        self.model_name = self.original_model_name
        self.client = client
        self.litellm = None

    monkeypatch.setattr(StageTransport, "_initialize_client", _fake_init)

    shim = TransportShim("gpt-4o-mini")
    result = shim.call(
        system_prompt="Return JSON.",
        user_prompt="Say ok.",
        response_model=_SimpleOutput,
        stage="unit_test",
    )

    assert result.value == "ok"


def test_transport_shim_retries_validation_errors_with_structured_repair(monkeypatch):
    client = _FakeOpenAIClient(
        [
            _DummyResponse('{"wrong": "shape"}'),
            _DummyResponse('{"value": "ok"}'),
        ]
    )

    def _fake_init(self):
        self.provider = "openai"
        self.model_name = self.original_model_name
        self.client = client
        self.litellm = None

    monkeypatch.setattr(StageTransport, "_initialize_client", _fake_init)

    shim = TransportShim("gpt-4o-mini")
    result = shim.call(
        system_prompt="Return JSON.",
        user_prompt="Say ok.",
        response_model=_SimpleOutput,
        stage="unit_test_validation_repair",
    )

    assert result.value == "ok"
    assert client.chat.completions.call_count == 2


def test_relation_frame_output_accepts_incomplete_drafts():
    payload = RelationFrameOutput.model_validate(
        {
            "frames": [
                {
                    "vp_phrase": "good friends with",
                }
            ]
        }
    )

    assert len(payload.frames) == 1
    assert payload.frames[0].subject.ref == ""
    assert payload.frames[0].object.ref == ""


def test_transport_shim_uses_structured_bedrock_path(monkeypatch):
    litellm = _FakeLiteLLM([
        _DummyResponse('{"value": "ok"}'),
    ])

    def _fake_init(self):
        self.provider = "bedrock"
        self.model_name = "bedrock/converse/openai.gpt-oss-120b-1:0"
        self.client = None
        self.litellm = litellm

    monkeypatch.setattr(StageTransport, "_initialize_client", _fake_init)

    shim = TransportShim("bedrock/openai.gpt-oss-120b-1:0", reasoning_effort="medium")
    result = shim.call(
        system_prompt="Return JSON.",
        user_prompt="Say ok.",
        response_model=_SimpleOutput,
        stage="bedrock_structured",
    )

    assert result.value == "ok"
    assert litellm.call_count == 1
    assert litellm.calls[0]["response_format"] is _SimpleOutput
    assert litellm.calls[0]["reasoning_effort"] == "medium"
