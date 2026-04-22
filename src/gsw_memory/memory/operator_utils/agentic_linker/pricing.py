"""Per-model pricing + cost calculation for linker pipeline runs.

Strategy: try ``litellm.completion_cost()`` first since it has the largest
maintained pricing table for OpenAI / Bedrock / Together / Anthropic /
Google. Fall back to a small built-in table for the models we use most
when litellm isn't available or doesn't recognize the model.

All rates are USD per **1 million tokens**.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# Built-in fallback table — only consulted when litellm doesn't know the model.
# Format: model_substring → (input_$_per_1M, output_$_per_1M)
_FALLBACK_RATES: list[tuple[str, tuple[float, float]]] = [
    # OpenAI
    ("gpt-4.1-mini", (0.40, 1.60)),
    ("gpt-4.1", (2.00, 8.00)),
    ("gpt-4o-mini", (0.15, 0.60)),
    ("gpt-4o", (2.50, 10.00)),
    # Bedrock — gpt-oss
    ("openai.gpt-oss-120b", (0.15, 0.60)),
    ("openai.gpt-oss-20b", (0.07, 0.30)),
    # Bedrock — Anthropic
    ("claude-opus-4", (15.00, 75.00)),
    ("claude-sonnet-4", (3.00, 15.00)),
    ("claude-haiku-4", (0.80, 4.00)),
    ("claude-3-5-sonnet", (3.00, 15.00)),
    ("claude-3-haiku", (0.25, 1.25)),
    # Together — Qwen
    ("qwen3-235b", (0.20, 0.60)),
    ("qwen3-32b", (0.10, 0.30)),
]


def estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a single LLM call. 0.0 if pricing unknown."""
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return 0.0

    # 1) Try litellm if available
    try:
        import litellm  # type: ignore

        # litellm.completion_cost wants either a response object or
        # explicit token counts via the helper signature below.
        cost = litellm.completion_cost(  # type: ignore[attr-defined]
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if isinstance(cost, (int, float)) and cost > 0:
            return float(cost)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("litellm completion_cost failed for %s: %r", model_name, exc)

    # 2) Built-in fallback by substring match
    rates = _lookup_fallback_rates(model_name)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate


def _lookup_fallback_rates(model_name: str) -> Tuple[float, float] | None:
    """Find the first fallback table entry whose substring matches the model."""
    name = (model_name or "").lower()
    for needle, rates in _FALLBACK_RATES:
        if needle.lower() in name:
            return rates
    return None
