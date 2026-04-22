import datetime
from types import SimpleNamespace

from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse

from gsw_memory.memory.models import GSWStructure
from gsw_memory.memory.operator import GSWProcessor
from gsw_memory.memory.operator_utils import alibaba_thinking as thinking_module


class _FakeOperator:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def __call__(self, inputs):
        dataset = []
        for input_row in inputs:
            dataset.append(
                {
                    "text": input_row["text"],
                    "idx": input_row.get("idx", 0),
                    "gsw": {"entity_nodes": [], "verb_phrase_nodes": []},
                    "doc_idx": input_row.get("doc_idx", 0),
                    "global_id": input_row.get("global_id", "0_0"),
                }
            )
        return SimpleNamespace(dataset=dataset)


class _CaptureThinkingOperator:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs


def _build_processor(**kwargs):
    return GSWProcessor(
        model_name="qwen-plus",
        enable_coref=False,
        enable_chunking=False,
        enable_context=False,
        enable_spacetime=False,
        **kwargs,
    )


def _make_generic_response(idx, *, error=None, usage=None):
    now = datetime.datetime.now()
    return GenericResponse(
        response_message=None,
        response_errors=[error] if error else None,
        raw_response={"usage": usage or {}} if usage is not None else {},
        raw_request=None,
        generic_request=GenericRequest(
            model="qwen-plus",
            messages=[],
            response_format=None,
            original_row_idx=idx,
            original_row={"idx": idx},
            generation_params={},
            multimodal_prompt=False,
        ),
        created_at=now,
        finished_at=now,
        token_usage=None,
        response_cost=None,
        finish_reason=None,
    )


def test_dashscope_non_thinking_uses_openai_backend(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr("gsw_memory.memory.operator.GSWOperator", _FakeOperator)

    processor = _build_processor(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    processor.process_documents(["Example document"])

    assert _FakeOperator.last_kwargs["backend"] == "openai"
    assert _FakeOperator.last_kwargs["backend_params"]["base_url"] == (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    assert _FakeOperator.last_kwargs["backend_params"]["api_key"] == "dash-key"


def test_legacy_vllm_alias_maps_to_openai_compatible_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("gsw_memory.memory.operator.GSWOperator", _FakeOperator)

    processor = _build_processor(vllm_base_url="http://127.0.0.1:6379/v1")
    processor.process_documents(["Example document"])

    assert _FakeOperator.last_kwargs["backend"] == "openai"
    assert _FakeOperator.last_kwargs["backend_params"]["base_url"] == (
        "http://127.0.0.1:6379/v1"
    )
    assert _FakeOperator.last_kwargs["backend_params"]["api_key"] == "EMPTY"


def test_dashscope_skips_curator_header_probe(monkeypatch):
    def fail_post(*args, **kwargs):
        raise AssertionError("DashScope header probe should not call requests.post")

    monkeypatch.setattr(thinking_module.curator_openai_module.requests, "post", fail_post)
    processor = SimpleNamespace(
        config=SimpleNamespace(
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
        ),
        api_key="dash-key",
        url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    )

    limits = thinking_module.curator_openai_module.OpenAIOnlineRequestProcessor.get_header_based_rate_limits(
        processor
    )

    assert limits == (0, 0)


def test_alibaba_thinking_uses_curator_enable_thinking(monkeypatch):
    captured = {"run_calls": 0}

    def fake_run_curator_operator(operator, inputs, working_dir):
        captured["run_calls"] += 1
        return (
            [
                {
                    "text": inputs[0]["text"],
                    "idx": inputs[0]["idx"],
                    "doc_idx": inputs[0]["doc_idx"],
                    "global_id": inputs[0]["global_id"],
                    "start_sentence": 0,
                    "end_sentence": 0,
                    "context": "",
                    "gsw": {"entity_nodes": [], "verb_phrase_nodes": []},
                    "llm_debug": {
                        "provider": "dashscope",
                        "mode": "thinking",
                        "parse_stage": "initial",
                        "raw_content": '{"entity_nodes": [], "verb_phrase_nodes": []}',
                        "reasoning_content": "thoughts",
                        "repaired_content": "",
                        "thinking_usage": {"total_tokens": 11},
                        "repair_usage": {},
                        "finish_reason": "stop",
                        "error": None,
                    },
                }
            ],
            {},
            None,
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(thinking_module, "AlibabaThinkingOperator", _CaptureThinkingOperator)
    monkeypatch.setattr(thinking_module, "_run_curator_operator", fake_run_curator_operator)

    results = thinking_module.process_documents_with_alibaba_thinking(
        documents=["Example document"],
        model_name="qwen3.5-plus",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        thinking_budget=123,
        max_workers=3,
        print_progress=False,
    )

    row = results[0]["0_0"]
    assert row["gsw"] is not None
    assert row["llm_debug"]["reasoning_content"] == "thoughts"
    assert captured["run_calls"] == 1
    assert _CaptureThinkingOperator.init_kwargs["backend"] == "openai"
    assert "response_format" not in _CaptureThinkingOperator.init_kwargs
    assert _CaptureThinkingOperator.init_kwargs["generation_params"]["enable_thinking"] is True
    assert _CaptureThinkingOperator.init_kwargs["generation_params"]["thinking_budget"] == 123
    assert _CaptureThinkingOperator.init_kwargs["backend_params"]["base_url"] == (
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    )
    assert _CaptureThinkingOperator.init_kwargs["backend_params"]["api_key"] == "dash-key"
    assert _CaptureThinkingOperator.init_kwargs["backend_params"]["max_concurrent_requests"] == 3
    assert _CaptureThinkingOperator.init_kwargs["backend_params"]["max_requests_per_minute"] == 30000
    assert _CaptureThinkingOperator.init_kwargs["backend_params"]["max_tokens_per_minute"] == 5000000


def test_alibaba_thinking_no_repair_by_default(monkeypatch):
    def fake_run_curator_operator(operator, inputs, working_dir):
        return (
            [
                {
                    "text": inputs[0]["text"],
                    "idx": inputs[0]["idx"],
                    "doc_idx": inputs[0]["doc_idx"],
                    "global_id": inputs[0]["global_id"],
                    "start_sentence": 0,
                    "end_sentence": 0,
                    "context": "",
                    "gsw": None,
                    "llm_debug": {
                        "provider": "dashscope",
                        "mode": "thinking",
                        "parse_stage": "initial_failed_no_repair",
                        "raw_content": "not json at all",
                        "reasoning_content": "",
                        "repaired_content": "",
                        "thinking_usage": {},
                        "repair_usage": {},
                        "finish_reason": "stop",
                        "error": "initial parse failed: bad json",
                    },
                }
            ],
            {},
            None,
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(thinking_module, "AlibabaThinkingOperator", _CaptureThinkingOperator)
    monkeypatch.setattr(thinking_module, "_run_curator_operator", fake_run_curator_operator)

    results = thinking_module.process_documents_with_alibaba_thinking(
        documents=["Example document"],
        model_name="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        print_progress=False,
    )

    chunk_data = results[0]["0_0"]
    assert chunk_data["gsw"] is None
    assert chunk_data["llm_debug"]["parse_stage"] == "initial_failed_no_repair"


def test_alibaba_thinking_opt_in_repair(monkeypatch):
    def fake_run_curator_operator(operator, inputs, working_dir):
        return (
            [
                {
                    "text": inputs[0]["text"],
                    "idx": inputs[0]["idx"],
                    "doc_idx": inputs[0]["doc_idx"],
                    "global_id": inputs[0]["global_id"],
                    "start_sentence": 0,
                    "end_sentence": 0,
                    "context": "",
                    "gsw": None,
                    "llm_debug": {
                        "provider": "dashscope",
                        "mode": "thinking",
                        "parse_stage": "initial_failed_no_repair",
                        "raw_content": "not json at all",
                        "reasoning_content": "thought trace",
                        "repaired_content": "",
                        "thinking_usage": {"total_tokens": 10},
                        "repair_usage": {},
                        "finish_reason": "stop",
                        "error": "initial parse failed: bad json",
                    },
                }
            ],
            {},
            None,
        )

    def fake_run_repair_pass(*, failed_rows, **kwargs):
        assert failed_rows[0]["idx"] == 0
        return (
            {
                0: {
                    "idx": 0,
                    "doc_idx": 0,
                    "global_id": "0_0",
                    "gsw": {"entity_nodes": [], "verb_phrase_nodes": []},
                }
            },
            {
                0: _make_generic_response(
                    0,
                    usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                )
            },
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(thinking_module, "AlibabaThinkingOperator", _CaptureThinkingOperator)
    monkeypatch.setattr(thinking_module, "_run_curator_operator", fake_run_curator_operator)
    monkeypatch.setattr(thinking_module, "_run_repair_pass", fake_run_repair_pass)

    results = thinking_module.process_documents_with_alibaba_thinking(
        documents=["Example document"],
        model_name="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        repair_model="qwen-flash",
        print_progress=False,
    )

    chunk_data = results[0]["0_0"]
    assert chunk_data["gsw"] is not None
    assert chunk_data["llm_debug"]["parse_stage"] == "repair"
    assert chunk_data["llm_debug"]["repair_usage"]["total_tokens"] == 7


def test_alibaba_thinking_request_failure_keeps_error(monkeypatch):
    def fake_run_curator_operator(operator, inputs, working_dir):
        return (
            [],
            {
                0: _make_generic_response(
                    0,
                    error="API error: invalid_api_key",
                    usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                )
            },
            Exception("all requests failed"),
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(thinking_module, "AlibabaThinkingOperator", _CaptureThinkingOperator)
    monkeypatch.setattr(thinking_module, "_run_curator_operator", fake_run_curator_operator)

    results = thinking_module.process_documents_with_alibaba_thinking(
        documents=["Example document"],
        model_name="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        print_progress=False,
    )

    chunk_data = results[0]["0_0"]
    assert chunk_data["gsw"] is None
    assert chunk_data["llm_debug"]["parse_stage"] == "request_failed"
    assert "invalid_api_key" in chunk_data["llm_debug"]["error"]


def test_alibaba_thinking_request_failure_formats_transport_error(monkeypatch):
    transport_error = thinking_module._build_request_error_message(
        thinking_module.curator_openai_module.aiohttp.ClientOSError(
            104, "Connection reset by peer"
        ),
        stage="transport",
        url="https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        request_payload={
            "model": "qwen3.5-plus",
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": True,
        },
        timeout_seconds=3600,
    )

    def fake_run_curator_operator(operator, inputs, working_dir):
        return (
            [],
            {0: _make_generic_response(0, error=transport_error)},
            Exception("all requests failed"),
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(thinking_module, "AlibabaThinkingOperator", _CaptureThinkingOperator)
    monkeypatch.setattr(thinking_module, "_run_curator_operator", fake_run_curator_operator)

    results = thinking_module.process_documents_with_alibaba_thinking(
        documents=["Example document"],
        model_name="qwen3.5-plus",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        print_progress=False,
    )

    error_text = results[0]["0_0"]["llm_debug"]["error"]
    assert "type=ClientOSError" in error_text
    assert "errno=104" in error_text
    assert "thinking=true" in error_text
    assert "model=qwen3.5-plus" in error_text


def test_alibaba_thinking_progress_logging(monkeypatch, capsys):
    def fake_run_curator_operator(operator, inputs, working_dir):
        rows = []
        for input_row in inputs:
            rows.append(
                {
                    "text": input_row["text"],
                    "idx": input_row["idx"],
                    "doc_idx": input_row["doc_idx"],
                    "global_id": input_row["global_id"],
                    "start_sentence": 0,
                    "end_sentence": 0,
                    "context": "",
                    "gsw": {"entity_nodes": [], "verb_phrase_nodes": []},
                    "llm_debug": {
                        "provider": "dashscope",
                        "mode": "thinking",
                        "parse_stage": "initial",
                        "raw_content": '{"entity_nodes": [], "verb_phrase_nodes": []}',
                        "reasoning_content": "",
                        "repaired_content": "",
                        "thinking_usage": {},
                        "repair_usage": {},
                        "finish_reason": "stop",
                        "error": None,
                    },
                }
            )
        return rows, {}, None

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(thinking_module, "AlibabaThinkingOperator", _CaptureThinkingOperator)
    monkeypatch.setattr(thinking_module, "_run_curator_operator", fake_run_curator_operator)

    thinking_module.process_documents_with_alibaba_thinking(
        documents=["doc0", "doc1", "doc2"],
        model_name="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        max_workers=2,
        progress_label="documents",
        print_progress=True,
    )

    captured = capsys.readouterr().out
    assert "Thinking batch 1" in captured
    assert "doc 0: ok" in captured
    assert "Batch 2 complete" in captured


def test_raw_outputs_keep_debug_but_final_network_does_not(tmp_path):
    processor = _build_processor()
    gsw = GSWStructure(entity_nodes=[], verb_phrase_nodes=[])
    all_documents_data = [
        {
            "0_0": {
                "text": "Example document",
                "doc_idx": 0,
                "chunk_idx": 0,
                "global_id": "0_0",
                "start_sentence": 0,
                "end_sentence": 0,
                "gsw": gsw,
                "context": "",
                "spacetime": {},
                "llm_debug": {"reasoning_content": "hidden from final output"},
            }
        }
    ]

    processor._save_outputs_unified(
        output_dir=str(tmp_path),
        save_intermediates=False,
        resolved_documents={0: "Example document"},
        all_documents_data=all_documents_data,
        do_visualization=False,
        batch_idx=1,
    )

    raw_path = tmp_path / "networks_raw" / "doc_0" / "gsw_0_0.json"
    final_path = tmp_path / "networks" / "doc_0" / "gsw_0_0.json"

    raw_payload = raw_path.read_text()
    final_payload = final_path.read_text()

    assert "llm_debug" in raw_payload
    assert "llm_debug" not in final_payload
