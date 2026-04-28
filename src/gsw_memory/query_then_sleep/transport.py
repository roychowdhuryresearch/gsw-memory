from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_BEDROCK_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_BEDROCK_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


@dataclass
class TransportToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw_arguments: str = ""
    arguments_error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "call_id": self.call_id,
            "raw_arguments": self.raw_arguments,
            "arguments_error": self.arguments_error,
        }


@dataclass
class TransportResponse:
    provider: str
    model_name: str
    text: str = ""
    tool_calls: List[TransportToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_text: str = ""
    raw: Any = None

    def as_log_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "finish_reason": self.finish_reason,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "tool_calls": [item.as_dict() for item in self.tool_calls],
            "text_preview": _compact_text(self.text),
            "reasoning_preview": _compact_text(self.reasoning_text),
        }


class StageTransport:
    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        reasoning_effort: str = "medium",
    ) -> None:
        self.original_model_name = str(model_name or "").strip()
        self.base_url = str(base_url).strip() if base_url else None
        self.reasoning_effort = str(reasoning_effort or "medium").strip() or "medium"
        self.provider = ""
        self.model_name = self.original_model_name
        self.client = None
        self.litellm = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if self.base_url:
            from openai import OpenAI

            self.provider = "vllm"
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=self.base_url)
            return

        if self.original_model_name.startswith("gpt-"):
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.provider = "openai"
            self.client = OpenAI(api_key=api_key)
            return

        try:
            import litellm
        except ImportError as exc:
            raise ImportError("LiteLLM package not installed. Run: pip install litellm") from exc

        self.litellm = litellm
        if self.original_model_name.startswith("bedrock/"):
            self.provider = "bedrock"
            if self.original_model_name.startswith("bedrock/converse/"):
                self.model_name = self.original_model_name
            elif self.original_model_name.startswith("bedrock/invoke/"):
                self.model_name = self.original_model_name.replace("bedrock/invoke/", "bedrock/", 1)
            else:
                self.model_name = self.original_model_name.replace("bedrock/", "bedrock/converse/", 1)
            return

        if self.original_model_name.startswith("bedrock/invoke/"):
            self.provider = "bedrock"
            self.model_name = self.original_model_name.replace("bedrock/invoke/", "bedrock/", 1)
            return

        if self.original_model_name.startswith(("together_ai/", "anthropic/", "vertex_ai/", "gemini/", "openai/")):
            self.provider = "litellm"
            self.model_name = self.original_model_name
            return

        if "/" in self.original_model_name:
            self.provider = "together"
            self.model_name = (
                self.original_model_name
                if self.original_model_name.startswith("together_ai/")
                else f"together_ai/{self.original_model_name}"
            )
            return

        raise ValueError(
            f"Unsupported query-then-sleep model '{self.original_model_name}'. "
            "Expected gpt-*, bedrock/..., provider/model, or provide --base_url for an OpenAI-compatible endpoint."
        )

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        stage: str = "chat",
    ) -> TransportResponse:
        if self.provider in {"openai", "vllm"}:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=list(messages),
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max(1, int(max_tokens)),
            )
            return self._normalize_chat_response(response)

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max(1, int(max_tokens)),
        }
        if tools is not None:
            api_params["tools"] = tools
        if tool_choice is not None:
            api_params["tool_choice"] = tool_choice

        if self.provider == "bedrock":
            api_params["reasoning_effort"] = self.reasoning_effort
            allowed_params = ["reasoning_effort"]
            if tools is not None:
                allowed_params.extend(["tools", "tool_choice"])
            api_params["allowed_openai_params"] = allowed_params
            api_params.pop("max_tokens", None)
            response = self._call_litellm_with_bedrock_retry(api_params, stage=stage)
        else:
            response = self.litellm.completion(**api_params)
        return self._normalize_chat_response(response)

    def generate_json(
        self,
        *,
        instructions: str,
        user_text: str,
        max_output_tokens: int = 800,
        temperature: float = 0.0,
        stage: str = "json",
    ) -> tuple[Dict[str, Any], TransportResponse]:
        response = self.chat(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stage=stage,
        )
        payload = _extract_json_dict(response.text)
        if payload is None:
            raise ValueError("Model did not return valid JSON")
        return payload, response

    def _call_litellm_with_bedrock_retry(self, api_params: Dict[str, Any], *, stage: str) -> Any:
        if self.provider != "bedrock":
            return self.litellm.completion(**api_params)

        last_error: Optional[Exception] = None
        max_attempts = len(_BEDROCK_RETRY_DELAYS_SECONDS) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return self.litellm.completion(**api_params)
            except Exception as error:  # pragma: no cover - exercised via tests with fakes
                last_error = error
                if not _is_retryable_bedrock_error(self.litellm, error):
                    raise
                if attempt >= max_attempts:
                    raise
                delay_seconds = float(_BEDROCK_RETRY_DELAYS_SECONDS[attempt - 1])
                logger.warning(
                    "Retrying Bedrock query_then_sleep completion [stage=%s model=%s attempt=%s/%s delay_s=%.1f error=%s]",
                    stage,
                    self.model_name,
                    attempt,
                    max_attempts,
                    delay_seconds,
                    error,
                )
                time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Bedrock call failed without a recorded error")

    def _normalize_chat_response(self, response: Any) -> TransportResponse:
        choice = _safe_get(getattr(response, "choices", None), 0)
        message = getattr(choice, "message", None)
        usage = getattr(response, "usage", None)
        text = _extract_message_text(message)
        reasoning_text = _extract_reasoning_text(message)
        tool_calls = _extract_tool_calls(message)
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return TransportResponse(
            provider=self.provider,
            model_name=self.model_name,
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_text=reasoning_text,
            raw=response,
        )


def _safe_get(value: Any, index: int) -> Any:
    try:
        return value[index]
    except Exception:
        return None


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: List[str] = []
        for chunk in content:
            text = None
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content")
            else:
                text = getattr(chunk, "text", None) or getattr(chunk, "content", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text.strip()
    return ""


def _extract_reasoning_text(message: Any) -> str:
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str):
        return reasoning.strip()
    if isinstance(reasoning, list):
        chunks: List[str] = []
        for chunk in reasoning:
            text = None
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content")
            else:
                text = getattr(chunk, "text", None) or getattr(chunk, "content", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str):
        return reasoning.strip()
    return ""


def _extract_tool_calls(message: Any) -> List[TransportToolCall]:
    tool_calls = getattr(message, "tool_calls", None) or []
    normalized: List[TransportToolCall] = []
    for item in tool_calls:
        function = getattr(item, "function", None)
        name = ""
        raw_arguments = ""
        arguments: Dict[str, Any] = {}
        arguments_error = ""
        call_id = str(getattr(item, "id", None) or getattr(item, "call_id", None) or "")
        if function is not None:
            name = str(getattr(function, "name", "") or "")
            raw_arguments = str(getattr(function, "arguments", "") or "")
        elif isinstance(item, dict):
            function_payload = item.get("function", {}) or {}
            name = str(function_payload.get("name", "") or item.get("name", "") or "")
            raw_arguments = str(function_payload.get("arguments", "") or item.get("arguments", "") or "")
            call_id = str(item.get("id", "") or item.get("call_id", "") or call_id)
        if raw_arguments:
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    arguments = parsed
            except Exception as exc:
                arguments_error = str(exc)
                arguments = {}
        normalized.append(
            TransportToolCall(
                name=name,
                arguments=arguments,
                call_id=call_id,
                raw_arguments=raw_arguments,
                arguments_error=arguments_error,
            )
        )
    return normalized


def _is_retryable_bedrock_error(litellm_module: Any, error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code in _BEDROCK_RETRYABLE_STATUS_CODES:
        return True
    exceptions_module = getattr(litellm_module, "exceptions", None)
    service_unavailable = getattr(exceptions_module, "ServiceUnavailableError", None)
    api_error = getattr(exceptions_module, "APIError", None)
    if service_unavailable is not None and isinstance(error, service_unavailable):
        return True
    if api_error is not None and isinstance(error, api_error):
        code = getattr(error, "status_code", None)
        return isinstance(code, int) and code in _BEDROCK_RETRYABLE_STATUS_CODES
    text = str(error).lower()
    return any(token in text for token in ("service unavailable", "503", "throttl", "too many requests", "timed out"))


def _compact_text(text: str, limit: int = 400) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "..."
