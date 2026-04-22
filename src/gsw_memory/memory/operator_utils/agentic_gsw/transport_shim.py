from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ....query_then_sleep.transport import StageTransport

logger = logging.getLogger(__name__)
# Dedicated logger for agent reasoning traces. The CLI can route this
# stream independently of the noisy transport DEBUG logs so users can see
# what each agent is "thinking" without dumping full prompts.
reasoning_logger = logging.getLogger("gsw_memory.agents.reasoning")

T = TypeVar("T", bound=BaseModel)


def _preview(text: str, n: int = 600) -> str:
    text = str(text or "")
    if len(text) <= n:
        return text
    return text[:n] + f"...[{len(text) - n} more chars]"


class StructuredResponseError(ValueError):
    def __init__(self, message: str, *, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


class TransportShim:
    """Small structured-call shim on top of StageTransport."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        reasoning_effort: str = "medium",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self._thread_local = threading.local()
        # Per-call ledger for the markdown trace writer. Each entry holds
        # the FULL reasoning text, FULL content text, and the parsed payload
        # — never truncated. The orchestrator drains this list at the end
        # of each document.
        self._call_log: list[dict] = []
        self._call_log_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API for the markdown trace writer
    # ------------------------------------------------------------------

    def drain_call_log(self) -> list[dict]:
        """Return and clear the per-call ledger. Called by the orchestrator
        after each document so traces don't bleed across docs."""
        with self._call_log_lock:
            log = list(self._call_log)
            self._call_log.clear()
        return log

    def _record_call(
        self,
        *,
        stage: str,
        response: Any,
        payload: Any,
        path: str,
    ) -> None:
        """Capture a single call's reasoning + content + token usage for
        the markdown trace and cost calculation. Stores FULL text, no preview."""
        try:
            choice = response.choices[0]
            message = choice.message
        except Exception:
            return

        reasoning_text = ""
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message, attr, None)
            if value is None:
                continue
            text = self._coerce_to_text(value).strip()
            if text:
                reasoning_text = text
                break

        content_text = self._coerce_to_text(getattr(message, "content", None)).strip()

        # Token usage. Most providers (OpenAI, litellm, bedrock-via-litellm)
        # surface these on response.usage. Default to 0 if missing.
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0) if usage else (prompt_tokens + completion_tokens)

        entry = {
            "stage": stage,
            "path": path,
            "reasoning": reasoning_text,
            "content": content_text,
            "payload": payload,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "timestamp": time.time(),
        }
        with self._call_log_lock:
            self._call_log.append(entry)

    async def async_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        semaphore: asyncio.Semaphore | None = None,
    ) -> T:
        async def _inner() -> T:
            return await asyncio.to_thread(
                self.call,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if semaphore is None:
            return await _inner()
        async with semaphore:
            return await _inner()

    def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 8000,
    ) -> T:
        transport = self._transport()
        # For providers where response_format gets stripped (bedrock via litellm with
        # restricted allowed_openai_params), embed the schema directly in the user prompt
        # so the model still has an explicit contract.
        if transport.provider in {"bedrock", "litellm", "together"}:
            schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            user_prompt = (
                f"{user_prompt}\n\n"
                "Respond with a single JSON object — no prose, no code fences, no commentary. "
                "It must validate against this JSON schema:\n"
                f"{schema_text}"
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(
            "stage=%s start provider=%s model=%s user_chars=%d max_tokens=%d",
            stage,
            transport.provider,
            transport.model_name,
            len(user_prompt),
            max_tokens,
        )
        logger.debug("stage=%s system_preview=%s", stage, _preview(system_prompt, 400))
        logger.debug("stage=%s user_preview=%s", stage, _preview(user_prompt, 1200))
        t_start = time.time()
        payload: dict | None = None
        try:
            payload = self._call_structured(
                messages=messages,
                response_model=response_model,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            validated = response_model.model_validate(payload)
            logger.info(
                "stage=%s ok duration=%.2fs path=structured",
                stage,
                time.time() - t_start,
            )
            return validated
        except ValidationError as exc:
            raw_text = json.dumps(payload, ensure_ascii=False, indent=2) if payload is not None else ""
            logger.warning(
                "stage=%s validation_error duration=%.2fs error=%s raw_preview=%s",
                stage,
                time.time() - t_start,
                _preview(str(exc), 400),
                _preview(raw_text, 600),
            )
            return self._repair_json(
                messages=messages,
                raw_text=raw_text,
                response_model=response_model,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
                validation_error=str(exc),
            )
        except StructuredResponseError as exc:
            logger.warning(
                "stage=%s structured_error duration=%.2fs error=%s raw_preview=%s",
                stage,
                time.time() - t_start,
                str(exc),
                _preview(exc.raw_text, 600),
            )
            return self._repair_json(
                messages=messages,
                raw_text=exc.raw_text,
                response_model=response_model,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "stage=%s transport_error duration=%.2fs error=%s",
                stage,
                time.time() - t_start,
                _preview(repr(exc), 400),
            )
            response = self._transport().chat(
                messages,
                temperature=temperature,
                max_tokens=max(8000, int(max_tokens)),
                stage=stage,
            )
            logger.debug("stage=%s plain_text_fallback preview=%s", stage, _preview(response.text, 600))
            return self._repair_json(
                messages=messages,
                raw_text=response.text,
                response_model=response_model,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _repair_json(
        self,
        *,
        messages: list[dict],
        raw_text: str,
        response_model: Type[T],
        stage: str,
        temperature: float,
        max_tokens: int,
        validation_error: str = "",
    ) -> T:
        schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        validation_block = (
            f"\nValidation errors from the previous response:\n{validation_error}\n"
            if validation_error
            else ""
        )
        logger.info(
            "stage=%s repair_start validation_error=%s raw_preview=%s",
            stage,
            bool(validation_error),
            _preview(raw_text, 400),
        )
        # Compacted repair: do NOT append to the original messages. The original
        # user payload may have been 100k+ chars, and re-sending it plus the
        # broken response plus a retry nudge blows through the token budget.
        # Instead: fresh system + user with ONLY (broken_response, schema).
        compact_raw = (raw_text or "[empty_response]")[:8000]
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON repair assistant. Re-emit one valid JSON object "
                    "that satisfies the provided schema. Be concise. No prose, no code "
                    "fences, no commentary — only the JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The previous model response was not valid JSON for the target schema."
                    f"{validation_block}"
                    "\nPrevious response:\n"
                    f"{compact_raw}\n"
                    "\nTarget JSON schema:\n"
                    f"{schema_text}\n"
                    "\nRe-output exactly one JSON object that validates against the schema."
                ),
            },
        ]
        # Repair gets at least 2x the caller's budget, with a 16k floor, so
        # reasoning models have room to emit the full corrected payload.
        repair_budget = max(16000, int(max_tokens) * 2)
        try:
            payload = self._call_structured(
                messages=repair_messages,
                response_model=response_model,
                stage=f"{stage}_repair",
                temperature=temperature,
                max_tokens=repair_budget,
            )
            validated = response_model.model_validate(payload)
            logger.info("stage=%s repair_ok path=structured budget=%d", stage, repair_budget)
            return validated
        except (ValidationError, StructuredResponseError) as exc:
            logger.warning(
                "stage=%s repair_structured_failed error=%s — falling back to plain text",
                stage,
                _preview(repr(exc), 300),
            )

        repaired = self._transport().chat(
            repair_messages,
            temperature=temperature,
            max_tokens=repair_budget,
            stage=f"{stage}_repair_text",
        )
        logger.info(
            "stage=%s repair_text_response chars=%d preview=%s",
            stage,
            len(repaired.text or ""),
            _preview(repaired.text, 600),
        )
        if not (repaired.text or "").strip():
            raise StructuredResponseError(
                f"Repair pass returned empty text for stage={stage}; "
                "model likely consumed all tokens on reasoning",
                raw_text="",
            )
        try:
            payload = self._extract_json_dict(repaired.text)
        except StructuredResponseError as exc:
            raise StructuredResponseError(
                f"Repair pass for stage={stage} returned non-JSON text",
                raw_text=exc.raw_text or repaired.text[:2000],
            ) from exc
        validated = response_model.model_validate(payload)
        logger.info("stage=%s repair_ok path=plain_text", stage)
        return validated

    def _call_structured(
        self,
        *,
        messages: list[dict],
        response_model: Type[T],
        stage: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        transport = self._transport()
        api_params: Dict[str, Any] = {
            "model": transport.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(128, int(max_tokens)),
        }

        use_direct_openai_schema = transport.provider in {"openai", "vllm"}
        if use_direct_openai_schema:
            api_params["response_format"] = self._pydantic_model_to_openai_json_schema(response_model)
        else:
            api_params["response_format"] = response_model

        raw_text = ""
        try:
            response = self._execute_structured_call(
                transport=transport,
                api_params=api_params,
                stage=stage,
            )
            payload = self._extract_payload_from_response(
                response=response,
                transport=transport,
                structured_output=True,
            )
            self._emit_reasoning(stage, response)
            self._record_call(stage=stage, response=response, payload=payload, path="structured")
            logger.debug(
                "stage=%s structured_extract_ok keys=%s",
                stage,
                list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
            )
            return payload
        except Exception as inner_exc:
            if "response" in locals():
                raw_text = self._extract_raw_text(response, transport)
            logger.warning(
                "stage=%s structured_call_failed error=%s raw_preview=%s",
                stage,
                _preview(repr(inner_exc), 300),
                _preview(raw_text, 500),
            )

        fallback_params = dict(api_params)
        fallback_params["response_format"] = {"type": "json_object"}
        logger.info("stage=%s trying_json_object_fallback", stage)
        try:
            fallback_response = self._execute_structured_call(
                transport=transport,
                api_params=fallback_params,
                stage=f"{stage}_structured_fallback",
            )
            payload = self._extract_payload_from_response(
                response=fallback_response,
                transport=transport,
                structured_output=False,
            )
            self._emit_reasoning(stage, fallback_response)
            self._record_call(
                stage=stage, response=fallback_response, payload=payload, path="json_object_fallback"
            )
            logger.info("stage=%s json_object_fallback_ok", stage)
            return payload
        except Exception as exc:
            if "fallback_response" in locals():
                raw_text = self._extract_raw_text(fallback_response, transport)
            logger.warning(
                "stage=%s json_object_fallback_failed error=%s raw_preview=%s",
                stage,
                _preview(repr(exc), 300),
                _preview(raw_text, 500),
            )
            raise StructuredResponseError("Structured output failed", raw_text=raw_text) from exc

    def _execute_structured_call(
        self,
        *,
        transport: StageTransport,
        api_params: Dict[str, Any],
        stage: str,
    ) -> Any:
        if transport.provider in {"openai", "vllm"}:
            return transport.client.chat.completions.create(**api_params)

        if transport.provider == "bedrock":
            api_params["reasoning_effort"] = transport.reasoning_effort
            # Bedrock rejects max_completion_tokens (conflicts with maxTokens).
            # Forward only max_tokens, set to a budget large enough for gpt-oss
            # reasoning models which burn most of the output channel on thinking.
            budget = max(int(api_params.get("max_tokens", 0) or 0), 8000)
            api_params["max_tokens"] = budget
            api_params.pop("max_completion_tokens", None)
            api_params["allowed_openai_params"] = ["reasoning_effort", "max_tokens"]
            return transport._call_litellm_with_bedrock_retry(api_params, stage=stage)

        if transport.provider == "together" and "gpt-oss" in str(api_params.get("model", "")).lower():
            api_params["reasoning_effort"] = transport.reasoning_effort

        return transport.litellm.completion(**api_params)

    def _transport(self) -> StageTransport:
        transport = getattr(self._thread_local, "transport", None)
        if transport is None:
            transport = StageTransport(
                self.model_name,
                base_url=self.base_url,
                reasoning_effort=self.reasoning_effort,
            )
            self._thread_local.transport = transport
        return transport

    @staticmethod
    def _pydantic_model_to_openai_json_schema(response_format: Any) -> Dict[str, Any]:
        schema_name = re.sub(r"(?<!^)(?=[A-Z])", "_", response_format.__name__).lower()
        if schema_name.endswith("_schema"):
            schema_name = schema_name[: -len("_schema")]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": response_format.model_json_schema(),
                "strict": True,
            },
        }

    @classmethod
    def _emit_reasoning(cls, stage: str, response: Any) -> None:
        """Pull reasoning_content / reasoning from the response and emit it
        via the dedicated reasoning logger. Routed independently of the
        noisy transport DEBUG stream so users can see model thinking
        without enabling full DEBUG."""
        try:
            choice = response.choices[0]
            message = choice.message
        except Exception:
            return
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message, attr, None)
            if value is None:
                continue
            text = cls._coerce_to_text(value).strip()
            if not text:
                continue
            reasoning_logger.info(
                "stage=%s reasoning=%s",
                stage,
                _preview(text, 800),
            )
            return  # First non-empty source wins

    @classmethod
    def _extract_payload_from_response(
        cls,
        *,
        response: Any,
        transport: StageTransport,
        structured_output: bool,
    ) -> dict:
        choice = response.choices[0]
        message = choice.message

        parsed_value = getattr(message, "parsed", None)
        if structured_output and parsed_value is not None:
            if isinstance(parsed_value, dict):
                return parsed_value
            if isinstance(parsed_value, str) and parsed_value.strip():
                return cls._extract_json_dict(parsed_value.strip())

        # Only `content` is a valid JSON source. Reasoning fields hold the model's
        # internal monologue (gpt-oss / Bedrock thinking traces) and must never be
        # parsed as JSON — doing so picks up a brace-delimited substring of narrative.
        content = cls._coerce_to_text(getattr(message, "content", None)).strip()
        if content:
            return cls._extract_json_dict(content)

        normalized = transport._normalize_chat_response(response)
        if normalized.text and normalized.text.strip():
            return cls._extract_json_dict(normalized.text.strip())

        raise StructuredResponseError(
            "Empty content from model (likely all tokens went into reasoning)",
            raw_text="",
        )

    @classmethod
    def _extract_raw_text(cls, response: Any, transport: StageTransport) -> str:
        choice = response.choices[0]
        message = choice.message
        text = cls._coerce_to_text(getattr(message, "content", None)).strip()
        if text:
            return text
        normalized = transport._normalize_chat_response(response).text
        return normalized or ""

    @staticmethod
    def _coerce_to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for block in value:
                if isinstance(block, dict):
                    text = block.get("text")
                    if text is not None:
                        parts.append(str(text))
                elif block is not None:
                    parts.append(str(block))
            return "\n".join(parts).strip()
        return str(value)

    @classmethod
    def _extract_json_dict(cls, text: str) -> dict:
        raw = str(text or "").strip()
        if not raw:
            raise StructuredResponseError("Empty model response", raw_text="")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return cls._normalize_top_level_keys(parsed)
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return cls._normalize_top_level_keys(parsed)
            except Exception:
                pass

        # Balanced-brace scan: walk the string, track nesting, and try to parse each
        # complete top-level {...} block. The greedy regex `\{.*\}` can grab a substring
        # spanning unrelated braces inside narrative text, which then fails to parse.
        depth = 0
        start = -1
        in_string = False
        escape = False
        for idx, ch in enumerate(raw):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidate = raw[start : idx + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return cls._normalize_top_level_keys(parsed)
                        except Exception:
                            pass
                        start = -1

        raise StructuredResponseError("Model did not return JSON", raw_text=raw[:2000])

    @staticmethod
    def _normalize_top_level_keys(payload: dict) -> dict:
        """Lowercase any top-level keys that would otherwise silently fail
        Pydantic validation. Models occasionally return ``{"Frames": [...]}``,
        ``{"Mentions": [...]}`` or similar capitalized variants; our schemas
        use lowercase field names and would default those fields to empty
        lists, replacing good state with nothing on the repair path.
        """
        if not isinstance(payload, dict):
            return payload
        result: dict = {}
        for key, value in payload.items():
            if isinstance(key, str) and key and not key.islower():
                lowered = key.lower()
                # Don't clobber a lowercase key that already exists.
                if lowered not in payload:
                    result[lowered] = value
                    continue
            result[key] = value
        return result
