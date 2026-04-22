"""
Alibaba DashScope helpers for GSW generation.

This module provides:
- OpenAI-compatible base URL and API key resolution helpers
- Curator-based DashScope thinking-mode generation
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from bespokelabs import curator
from bespokelabs.curator.request_processor.online import (
    openai_online_request_processor as curator_openai_module,
)
from bespokelabs.curator.request_processor.online.base_online_request_processor import (
    BaseOnlineRequestProcessor,
)
from bespokelabs.curator.types.generic_response import GenericResponse

from ...prompts.operator_prompts import PromptType
from ..models import GSWStructure
from .gsw_operator import build_gsw_messages
from .utils import parse_gsw

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MAX_REQUESTS_PER_MINUTE = 30_000
DASHSCOPE_MAX_TOKENS_PER_MINUTE = 5_000_000
_LOCAL_BASE_URL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0")
_REQUEST_ERROR_PREFIX = "request_error: "


def repair_model_enabled(repair_model: Optional[str]) -> bool:
    """Return True when repair should run."""
    if repair_model is None:
        return False
    return repair_model.strip().lower() not in {"", "none", "off", "false", "disable", "disabled"}


def normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize OpenAI-compatible base URLs and treat 'None' as unset."""
    if base_url is None:
        return None
    normalized = str(base_url).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized.rstrip("/")


def is_dashscope_base_url(base_url: Optional[str]) -> bool:
    """Return True when a base URL points to DashScope compatible-mode."""
    normalized = normalize_openai_base_url(base_url)
    return bool(normalized and "dashscope" in normalized.lower())


def _is_local_base_url(base_url: str) -> bool:
    return any(host in base_url.lower() for host in _LOCAL_BASE_URL_HOSTS)


def resolve_api_key_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Resolve the environment-backed API key for an OpenAI-compatible base URL."""
    normalized = normalize_openai_base_url(base_url)
    if normalized is None:
        return None

    if is_dashscope_base_url(normalized):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable not set for DashScope base URL"
            )
        return api_key

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return openai_key

    if _is_local_base_url(normalized):
        return "EMPTY"

    return "EMPTY"


def _summarize_exception(exc: BaseException) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "repr": repr(exc),
    }

    errno = getattr(exc, "errno", None)
    if errno is not None:
        details["errno"] = int(errno)

    strerror = getattr(exc, "strerror", None)
    if strerror:
        details["strerror"] = str(strerror)

    status = getattr(exc, "status", None)
    if status is not None:
        details["status"] = int(status)

    request_info = getattr(exc, "request_info", None)
    real_url = getattr(request_info, "real_url", None)
    if real_url is not None:
        details["request_url"] = str(real_url)

    os_error = getattr(exc, "os_error", None)
    if os_error is not None and os_error is not exc:
        details["os_error"] = _summarize_exception(os_error)

    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        details["cause"] = _summarize_exception(cause)

    return details


def _build_request_error_message(
    exc: BaseException,
    *,
    stage: str,
    url: str,
    request_payload: Dict[str, Any],
    timeout_seconds: Optional[int],
    response_status: Optional[int] = None,
    response_payload: Optional[Dict[str, Any]] = None,
) -> str:
    messages = request_payload.get("messages") or []
    details: Dict[str, Any] = {
        "stage": stage,
        "url": url,
        "model": request_payload.get("model"),
        "enable_thinking": bool(request_payload.get("enable_thinking")),
        "message_count": len(messages),
        "timeout_seconds": timeout_seconds,
        "generation_keys": sorted(
            key for key in request_payload.keys() if key not in {"model", "messages"}
        ),
        "exception": _summarize_exception(exc),
    }

    if response_status is not None:
        details["response_status"] = int(response_status)

    if isinstance(response_payload, dict):
        request_id = response_payload.get("request_id")
        if request_id:
            details["request_id"] = str(request_id)
        provider_error = response_payload.get("error")
        if provider_error is not None:
            details["provider_error"] = provider_error

    return _REQUEST_ERROR_PREFIX + json.dumps(details, ensure_ascii=True, sort_keys=True)


def _parse_request_error_message(error_text: str) -> Optional[Dict[str, Any]]:
    if not error_text.startswith(_REQUEST_ERROR_PREFIX):
        return None
    payload = error_text[len(_REQUEST_ERROR_PREFIX) :].strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _render_request_error_message(error_details: Dict[str, Any]) -> str:
    exception = error_details.get("exception") or {}
    parts = [
        f"stage={error_details.get('stage', 'unknown')}",
        f"type={exception.get('type', 'UnknownError')}",
        f"message={exception.get('message', '')}",
    ]

    errno = exception.get("errno")
    if errno is not None:
        parts.append(f"errno={errno}")

    response_status = error_details.get("response_status")
    if response_status is not None:
        parts.append(f"status={response_status}")

    request_id = error_details.get("request_id")
    if request_id:
        parts.append(f"request_id={request_id}")

    url = error_details.get("url")
    if url:
        parts.append(f"url={url}")

    model = error_details.get("model")
    if model:
        parts.append(f"model={model}")

    if error_details.get("enable_thinking"):
        parts.append("thinking=true")

    timeout_seconds = error_details.get("timeout_seconds")
    if timeout_seconds:
        parts.append(f"timeout_seconds={timeout_seconds}")

    return "request_error(" + ", ".join(parts) + ")"


def _patch_curator_openai_request_processor() -> None:
    """Patch Curator's OpenAI processor for providers that reject empty-message probes."""
    cls = curator_openai_module.OpenAIOnlineRequestProcessor
    if getattr(cls, "_gsw_dashscope_patch_applied", False):
        return

    def _patched_init(self, config, compatible_provider=None):
        BaseOnlineRequestProcessor.__init__(self, config)
        self._cost_processor = curator_openai_module.cost_processor_factory(
            compatible_provider or self.backend
        )

        if self.config.base_url is None:
            if "OPENAI_BASE_URL" in os.environ:
                key_url = os.environ["OPENAI_BASE_URL"].strip().rstrip("/")
                self.url = key_url + self._DEFAULT_COMPLETION_SUFFIX
            else:
                self.url = curator_openai_module._DEFAULT_OPENAI_URL
        else:
            self.url = self.config.base_url + self._DEFAULT_COMPLETION_SUFFIX

        if self.config.base_url == "https://api.deepseek.com":
            self.api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            (
                self.header_based_max_requests_per_minute,
                self.header_based_max_tokens_per_minute,
            ) = self.get_header_based_rate_limits()

        self.token_encoding = self.get_token_encoding()

    def _patched_get_header_based_rate_limits(self) -> tuple[int, int]:
        normalized_base_url = normalize_openai_base_url(self.config.base_url)
        if is_dashscope_base_url(normalized_base_url):
            curator_openai_module.logger.info(
                "Skipping Curator header-based rate-limit probe for DashScope because "
                "DashScope rejects empty-message chat completion requests."
            )
            return 0, 0

        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API Key - Please set OPENAI_API_KEY in your environment vars"
            )

        response = curator_openai_module.requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.config.model, "messages": []},
        )
        if response.status_code >= 400:
            curator_openai_module.logger.warning(
                "Curator header-based rate-limit probe returned status %s for %s; "
                "falling back to manual/default limits.",
                response.status_code,
                self.url,
            )
            return 0, 0

        rpm = int(response.headers.get("x-ratelimit-limit-requests", 0) or 0)
        tpm = int(response.headers.get("x-ratelimit-limit-tokens", 0) or 0)
        return rpm, tpm

    async def _patched_call_single_request(self, request, session, status_tracker):
        import asyncio as _asyncio
        import random as _random

        request_header = {"Authorization": f"Bearer {self.api_key}"}
        if "/deployments" in self.url:
            request_header = {"api-key": f"{self.api_key}"}

        _MAX_TRANSPORT_RETRIES = 3
        last_transport_exc = None

        for _transport_attempt in range(_MAX_TRANSPORT_RETRIES + 1):
            if _transport_attempt > 0:
                delay = min(60, (2 ** _transport_attempt)) + _random.uniform(0, 2)
                curator_openai_module.logger.info(
                    f"Transport retry {_transport_attempt}/{_MAX_TRANSPORT_RETRIES} "
                    f"for request {request.task_id} after {delay:.1f}s backoff"
                )
                await _asyncio.sleep(delay)

            try:
                async with session.post(
                    self.url,
                    headers=request_header,
                    json=request.api_specific_request,
                    timeout=self.config.request_timeout,
                ) as response_obj:
                    try:
                        response = await response_obj.json()
                    except Exception as exc:
                        raise RuntimeError(
                            _build_request_error_message(
                                exc,
                                stage="response_decode",
                                url=self.url,
                                request_payload=request.api_specific_request,
                                timeout_seconds=self.config.request_timeout,
                                response_status=response_obj.status,
                            )
                        ) from exc

                    if "error" in response:
                        status_tracker.num_api_errors += 1
                        error = response["error"]
                        error_message = error if isinstance(error, str) else error.get("message", "")
                        if "rate limit" in error_message.lower():
                            status_tracker.time_of_last_rate_limit_error = curator_openai_module.time.time()
                            status_tracker.num_rate_limit_errors += 1
                            status_tracker.num_api_errors -= 1
                            status_tracker.num_other_errors -= 1
                        raise RuntimeError(
                            _build_request_error_message(
                                Exception(error_message or str(error)),
                                stage="provider_error",
                                url=self.url,
                                request_payload=request.api_specific_request,
                                timeout_seconds=self.config.request_timeout,
                                response_status=response_obj.status,
                                response_payload=response,
                            )
                        )

                    if response_obj.status != 200:
                        raise RuntimeError(
                            _build_request_error_message(
                                Exception(f"API request failed with status {response_obj.status}"),
                                stage="http_status",
                                url=self.url,
                                request_payload=request.api_specific_request,
                                timeout_seconds=self.config.request_timeout,
                                response_status=response_obj.status,
                                response_payload=response,
                            )
                        )

                    if self.config.return_completions_object:
                        response_message = dict(response)
                    else:
                        response_message = response["choices"][0]["message"]["content"]
                    finish_reason = response["choices"][0].get("finish_reason", "unkown")
                    usage = response["usage"]
                    token_usage = curator_openai_module.TokenUsage(
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                        total_tokens=usage["total_tokens"],
                    )

                    cost = self.completion_cost(response)

                    return GenericResponse(
                        response_message=response_message,
                        response_errors=None,
                        raw_request=request.api_specific_request,
                        raw_response=response,
                        generic_request=request.generic_request,
                        created_at=request.created_at,
                        finished_at=curator_openai_module.datetime.datetime.now(),
                        token_usage=token_usage,
                        response_cost=cost,
                        finish_reason=finish_reason,
                    )
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc).startswith(_REQUEST_ERROR_PREFIX):
                    raise
                # Transport-level error — retry with backoff before giving up
                last_transport_exc = exc
                if _transport_attempt < _MAX_TRANSPORT_RETRIES:
                    continue
                raise RuntimeError(
                    _build_request_error_message(
                        exc,
                        stage="transport",
                        url=self.url,
                        request_payload=request.api_specific_request,
                        timeout_seconds=self.config.request_timeout,
                    )
                ) from exc

    cls.__init__ = _patched_init
    cls.get_header_based_rate_limits = _patched_get_header_based_rate_limits
    cls.call_single_request = _patched_call_single_request
    cls._gsw_dashscope_patch_applied = True


_patch_curator_openai_request_processor()


@dataclass
class AlibabaThinkingResult:
    gsw: Optional[GSWStructure]
    raw_content: str
    reasoning_content: str
    repaired_content: str
    thinking_usage: Dict[str, int]
    repair_usage: Dict[str, int]
    parse_stage: str
    finish_reason: Optional[str]
    error: Optional[str] = None


def _usage_to_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _coerce_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts: List[str] = []
        for item in payload:
            parts.append(_coerce_text(item))
        return "".join(part for part in parts if part)
    if isinstance(payload, dict):
        for key in ("text", "content", "thinking", "reasoning_content"):
            value = payload.get(key)
            if value is not None:
                return _coerce_text(value)
        return ""
    for attr in ("text", "content", "thinking", "reasoning_content"):
        value = getattr(payload, attr, None)
        if value is not None:
            return _coerce_text(value)
    return str(payload)


def _parse_gsw_candidate(text: str) -> GSWStructure:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Empty model output")
    try:
        return GSWStructure.model_validate_json(stripped)
    except Exception:
        return parse_gsw(stripped)


def _extract_choice_message(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def _extract_finish_reason(response: Dict[str, Any]) -> Optional[str]:
    choices = response.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def _format_response_error(response: GenericResponse) -> str:
    errors = response.response_errors or []
    if errors:
        formatted_errors = []
        for error in errors:
            error_text = str(error)
            structured = _parse_request_error_message(error_text)
            if structured is not None:
                formatted_errors.append(_render_request_error_message(structured))
            else:
                formatted_errors.append(error_text)
        return "; ".join(formatted_errors)

    raw_response = response.raw_response or {}
    raw_error = raw_response.get("error")
    if isinstance(raw_error, dict):
        return str(raw_error.get("message") or raw_error)
    if raw_error:
        return str(raw_error)
    return "Request failed without an error payload"


def _load_curator_responses(working_dir: str) -> Dict[int, GenericResponse]:
    responses: Dict[int, GenericResponse] = {}
    pattern = str(Path(working_dir) / "**" / "responses_*.jsonl")
    for path in sorted(glob(pattern, recursive=True)):
        with open(path, "r") as handle:
            for line in handle:
                response = GenericResponse.model_validate_json(line)
                responses[response.generic_request.original_row_idx] = response
    return responses


def _build_backend_params(
    model_name: str,
    base_url: str,
    max_concurrent_requests: int,
) -> Dict[str, Any]:
    backend_params = {
        "base_url": base_url,
        "api_key": resolve_api_key_for_base_url(base_url),
        "request_timeout": 3600,
        "max_concurrent_requests": max(1, int(max_concurrent_requests)),
        "require_all_responses": False,
        "model": model_name,
    }
    if is_dashscope_base_url(base_url):
        backend_params["max_requests_per_minute"] = DASHSCOPE_MAX_REQUESTS_PER_MINUTE
        backend_params["max_tokens_per_minute"] = DASHSCOPE_MAX_TOKENS_PER_MINUTE
    else:
        backend_params["max_requests_per_minute"] = max(60, int(max_concurrent_requests) * 60)
    return backend_params


class AlibabaThinkingOperator(curator.LLM):
    """Curator operator that requests DashScope thinking responses."""

    return_completions_object = True

    def __init__(self, prompt_type: PromptType = PromptType.FACTUAL_GPT_OSS, **kwargs):
        super().__init__(**kwargs)
        self.prompt_type = prompt_type

    def prompt(self, input_row):
        return build_gsw_messages(
            prompt_type=self.prompt_type,
            input_text=input_row["text"],
            background_context=input_row.get("context", ""),
        )

    def parse(self, input_row, response: Dict[str, Any]):
        message = _extract_choice_message(response)
        raw_content = _coerce_text(message.get("content"))
        reasoning_content = _coerce_text(
            message.get("reasoning_content") or message.get("reasoning")
        )

        try:
            gsw = _parse_gsw_candidate(raw_content)
            parse_stage = "initial"
            error = None
            gsw_payload = gsw.model_dump()
        except Exception as exc:
            parse_stage = "initial_failed_no_repair"
            error = f"initial parse failed: {exc}"
            gsw_payload = None

        return {
            "text": input_row["text"],
            "idx": input_row.get("idx", 0),
            "doc_idx": input_row.get("doc_idx", input_row.get("idx", 0)),
            "global_id": input_row.get("global_id", "unknown"),
            "start_sentence": input_row.get("start_sentence", 0),
            "end_sentence": input_row.get("end_sentence", 0),
            "context": input_row.get("context", ""),
            "gsw": gsw_payload,
            "llm_debug": {
                "provider": "dashscope",
                "mode": "thinking",
                "parse_stage": parse_stage,
                "raw_content": raw_content,
                "reasoning_content": reasoning_content,
                "repaired_content": "",
                "thinking_usage": _usage_to_dict(response.get("usage")),
                "repair_usage": {},
                "finish_reason": _extract_finish_reason(response),
                "error": error,
            },
        }


class AlibabaThinkingRepairOperator(curator.LLM):
    """Curator operator that repairs malformed thinking outputs into structured JSON."""

    response_format = GSWStructure

    def __init__(self, prompt_type: PromptType = PromptType.FACTUAL_GPT_OSS, **kwargs):
        super().__init__(**kwargs)
        self.prompt_type = prompt_type

    def prompt(self, input_row):
        messages = build_gsw_messages(
            prompt_type=self.prompt_type,
            input_text=input_row["text"],
            background_context=input_row.get("context", ""),
        )
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": input_row.get("raw_content") or "[empty_response]",
                },
                {
                    "role": "user",
                    "content": (
                        "Re-output the previous extraction as valid JSON only. "
                        "Return JSON only, with no markdown or explanation."
                    ),
                },
            ]
        )
        return messages

    def parse(self, input_row, response: GSWStructure):
        return {
            "idx": input_row["idx"],
            "doc_idx": input_row["doc_idx"],
            "global_id": input_row["global_id"],
            "gsw": response.model_dump(),
        }


def _normalize_dataset_rows(dataset_obj: Any) -> List[Dict[str, Any]]:
    if dataset_obj is None:
        return []
    dataset = dataset_obj.dataset if hasattr(dataset_obj, "dataset") else dataset_obj
    return [dict(row) for row in dataset]


def _run_curator_operator(
    operator: curator.LLM,
    inputs: Iterable[Dict[str, Any]],
    working_dir: str,
) -> tuple[List[Dict[str, Any]], Dict[int, GenericResponse], Optional[Exception]]:
    error: Optional[Exception] = None
    rows: List[Dict[str, Any]] = []
    input_rows = list(inputs)
    try:
        rows = _normalize_dataset_rows(operator(input_rows, working_dir=working_dir))
    except Exception as exc:
        error = exc
    responses = _load_curator_responses(working_dir)
    return rows, responses, error


def _build_thinking_result_from_row(row: Dict[str, Any]) -> AlibabaThinkingResult:
    gsw_payload = row.get("gsw")
    return AlibabaThinkingResult(
        gsw=GSWStructure(**gsw_payload) if gsw_payload else None,
        raw_content=row["llm_debug"].get("raw_content", ""),
        reasoning_content=row["llm_debug"].get("reasoning_content", ""),
        repaired_content=row["llm_debug"].get("repaired_content", ""),
        thinking_usage=row["llm_debug"].get("thinking_usage", {}),
        repair_usage=row["llm_debug"].get("repair_usage", {}),
        parse_stage=row["llm_debug"].get("parse_stage", "initial_failed_no_repair"),
        finish_reason=row["llm_debug"].get("finish_reason"),
        error=row["llm_debug"].get("error"),
    )


def generate_gsw_with_alibaba_thinking(
    text: str,
    model_name: str,
    prompt_type: PromptType = PromptType.FACTUAL_GPT_OSS,
    base_url: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    repair_model: Optional[str] = "none",
    background_context: str = "",
) -> AlibabaThinkingResult:
    """Generate a single GSW with DashScope thinking mode via Curator."""
    results = process_documents_with_alibaba_thinking(
        documents=[text],
        model_name=model_name,
        prompt_type=prompt_type,
        base_url=base_url,
        thinking_budget=thinking_budget,
        repair_model=repair_model,
        max_workers=1,
        progress_label="single-document request",
        print_progress=False,
        contexts=[background_context],
    )
    row = next(iter(results[0].values()))
    return _build_thinking_result_from_row(row)


def _run_repair_pass(
    failed_rows: List[Dict[str, Any]],
    model_name: str,
    prompt_type: PromptType,
    base_url: str,
    repair_model: str,
    max_workers: int,
    working_dir: str,
) -> tuple[Dict[int, Dict[str, Any]], Dict[int, GenericResponse]]:
    repair_operator = AlibabaThinkingRepairOperator(
        model_name=repair_model or model_name,
        prompt_type=prompt_type,
        backend="openai",
        response_format=GSWStructure,
        generation_params={"temperature": 0.0},
        backend_params=_build_backend_params(
            model_name=repair_model or model_name,
            base_url=base_url,
            max_concurrent_requests=max_workers,
        ),
    )
    repair_rows, repair_responses, _ = _run_curator_operator(
        operator=repair_operator,
        inputs=failed_rows,
        working_dir=working_dir,
    )
    return {row["idx"]: row for row in repair_rows}, repair_responses


def _build_chunk_data(
    row: Dict[str, Any],
    *,
    gsw: Optional[GSWStructure],
    llm_debug: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "text": row["text"],
        "doc_idx": row["doc_idx"],
        "chunk_idx": row.get("chunk_idx", 0),
        "global_id": row["global_id"],
        "start_sentence": row.get("start_sentence", 0),
        "end_sentence": row.get("end_sentence", 0),
        "gsw": gsw,
        "context": row.get("context", ""),
        "spacetime": {},
        "llm_debug": llm_debug,
    }


def process_documents_with_alibaba_thinking(
    documents: Sequence[str],
    model_name: str,
    prompt_type: PromptType = PromptType.FACTUAL_GPT_OSS,
    base_url: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    repair_model: Optional[str] = "none",
    max_workers: int = 4,
    progress_label: str = "documents",
    print_progress: bool = True,
    contexts: Optional[Sequence[str]] = None,
) -> List[Dict[str, Dict[str, Any]]]:
    """Process documents with bounded Curator-powered DashScope thinking requests."""
    normalized_base_url = normalize_openai_base_url(base_url) or DASHSCOPE_BASE_URL
    if not is_dashscope_base_url(normalized_base_url):
        raise ValueError(
            "Alibaba thinking mode is only supported with a DashScope base URL"
        )

    max_workers = max(1, int(max_workers))
    batch_size = max_workers
    results: List[Optional[Dict[str, Dict[str, Any]]]] = [None] * len(documents)
    contexts = list(contexts) if contexts is not None else [""] * len(documents)

    operator = AlibabaThinkingOperator(
        model_name=model_name,
        prompt_type=prompt_type,
        backend="openai",
        generation_params={
            "temperature": 0.0,
            "enable_thinking": True,
            **(
                {"thinking_budget": int(thinking_budget)}
                if thinking_budget is not None and int(thinking_budget) > 0
                else {}
            ),
        },
        backend_params=_build_backend_params(
            model_name=model_name,
            base_url=normalized_base_url,
            max_concurrent_requests=max_workers,
        ),
    )

    for batch_start in range(0, len(documents), batch_size):
        batch_number = (batch_start // batch_size) + 1
        batch_end = min(batch_start + batch_size, len(documents))
        batch_inputs = []
        for doc_idx in range(batch_start, batch_end):
            batch_inputs.append(
                {
                    "text": documents[doc_idx],
                    "idx": doc_idx,
                    "doc_idx": doc_idx,
                    "global_id": f"{doc_idx}_0",
                    "start_sentence": 0,
                    "end_sentence": 0,
                    "context": contexts[doc_idx],
                }
            )

        if print_progress:
            print(
                f"   Thinking batch {batch_number}: docs {batch_start}-{batch_end - 1} "
                f"({len(batch_inputs)} {progress_label})"
            )

        batch_working_dir = tempfile.mkdtemp(prefix="gsw_alibaba_thinking_")
        rows, responses, batch_error = _run_curator_operator(
            operator=operator,
            inputs=batch_inputs,
            working_dir=batch_working_dir,
        )

        row_by_idx = {row["idx"]: row for row in rows}

        repair_candidates = [
            row
            for row in rows
            if row["gsw"] is None and repair_model_enabled(repair_model)
        ]

        repair_rows_by_idx: Dict[int, Dict[str, Any]] = {}
        repair_responses: Dict[int, GenericResponse] = {}
        if repair_candidates:
            repair_working_dir = tempfile.mkdtemp(prefix="gsw_alibaba_repair_")
            repair_rows_by_idx, repair_responses = _run_repair_pass(
                failed_rows=repair_candidates,
                model_name=model_name,
                prompt_type=prompt_type,
                base_url=normalized_base_url,
                repair_model=repair_model or model_name,
                max_workers=max_workers,
                working_dir=repair_working_dir,
            )
            shutil.rmtree(repair_working_dir, ignore_errors=True)

        batch_successes = 0
        for input_row in batch_inputs:
            doc_idx = input_row["idx"]
            global_id = input_row["global_id"]
            base_row = row_by_idx.get(doc_idx)
            response = responses.get(doc_idx)

            if base_row is None:
                error = _format_response_error(response) if response is not None else (
                    str(batch_error) if batch_error is not None else "No response captured"
                )
                llm_debug = {
                    "provider": "dashscope",
                    "mode": "thinking",
                    "parse_stage": "request_failed",
                    "raw_content": "",
                    "reasoning_content": "",
                    "repaired_content": "",
                    "thinking_usage": _usage_to_dict(
                        (response.raw_response or {}).get("usage") if response else None
                    ),
                    "repair_usage": {},
                    "finish_reason": None,
                    "error": error,
                }
                chunk_data = _build_chunk_data(input_row, gsw=None, llm_debug=llm_debug)
            else:
                llm_debug = dict(base_row["llm_debug"])
                gsw_payload = base_row.get("gsw")
                gsw = GSWStructure(**gsw_payload) if gsw_payload else None

                if gsw is None and repair_model_enabled(repair_model):
                    repaired_row = repair_rows_by_idx.get(doc_idx)
                    repair_response = repair_responses.get(doc_idx)
                    llm_debug["repair_usage"] = _usage_to_dict(
                        (repair_response.raw_response or {}).get("usage")
                        if repair_response is not None
                        else None
                    )
                    if repaired_row is not None and repaired_row.get("gsw") is not None:
                        llm_debug["parse_stage"] = "repair"
                        llm_debug["repaired_content"] = json.dumps(repaired_row["gsw"])
                        llm_debug["error"] = None
                        gsw = GSWStructure(**repaired_row["gsw"])
                    else:
                        repair_error = (
                            _format_response_error(repair_response)
                            if repair_response is not None
                            else "Repair response missing"
                        )
                        llm_debug["parse_stage"] = "repair_failed"
                        llm_debug["error"] = (
                            f"{llm_debug.get('error') or 'initial parse failed'}; "
                            f"repair failed: {repair_error}"
                        )

                chunk_data = _build_chunk_data(input_row, gsw=gsw, llm_debug=llm_debug)

            results[doc_idx] = {global_id: chunk_data}
            status = "ok" if chunk_data["gsw"] is not None else chunk_data["llm_debug"]["parse_stage"]
            if chunk_data["gsw"] is not None:
                batch_successes += 1
            if print_progress:
                print(f"      doc {doc_idx}: {status}")

        if print_progress:
            print(
                f"   Batch {batch_number} complete: {batch_successes}/{len(batch_inputs)} "
                "valid GSWs"
            )
        shutil.rmtree(batch_working_dir, ignore_errors=True)

    return [doc_result for doc_result in results if doc_result is not None]
