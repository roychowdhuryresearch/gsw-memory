"""LLM-as-judge for FRAMES-style QA.

Mirrors the evaluation protocol from the original FRAMES paper: exact
match is too strict (misses verbose-but-correct predictions) and token
F1 is noisy for long answers, so a small frontier model is asked to
judge whether ``predicted`` expresses the same answer as ``gold``.

Defaults to ``gpt-4o`` (fast + cheap at the pilot scale: 30 Qs × a few
hundred tokens ≈ $0.03 / cell). Any OpenAI-compatible endpoint works.

The judge is provider-agnostic: it uses the same ``openai`` SDK the
adapters use. For vLLM-served judge models, pass ``base_url`` +
``api_key``. For Bedrock judges, use the sleep-time litellm path
instead; that's out of scope for this first pass.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

_log = logging.getLogger("research_agent.judge")


JUDGE_SYSTEM_PROMPT = """You are an answer-grading assistant for a question \
answering benchmark. For each case you receive:
- the question,
- the gold (reference) answer,
- the model's predicted answer.

Decide whether the predicted answer expresses the same FACTS as gold. \
Grade on substance, not surface form. A correct answer can look very \
different from the gold string and still be correct.

## What to IGNORE (cosmetic — mark CORRECT if these are the only differences)

- Punctuation and separators between items: "A; B" / "A, B" / "A and B" / \
"A & B" / "A B" are all equivalent. A predicted "Arianne; Kathleen" matches a \
gold "Arianne Kathleen" — same two name parts, different delimiter.
- Accents and diacritics: "Mooré" / "Mòoré" / "Moore", \
"Biruté Galdikas" / "Birutė Galdikas".
- Whitespace, casing, articles ("the", "a"), trailing periods, quotes, \
parens, possessive "'s", ordinal vs cardinal ("first" / "1st").
- Date formatting: "1972" / "in 1972" / "1972-01" / "January 1972" / \
"Jan 1972" all match a gold "1972" if the year is what matters; full date \
"Jan 5 1972" / "January 5, 1972" / "5/1/1972" likewise match.
- Unit normalization: "84512" / "84,512" / "84.5K", "12 years" / "12", \
"32 minutes" / "32 mins".
- Number formats: "1.0" / "1", "10646" / "10,646", scientific notation.
- Verbose-but-equivalent: gold "Bob Dylan" / pred "Bob Dylan (Robert \
Zimmerman)" / pred "It's Bob Dylan." — all match.
- Yes/no synonyms: "yes" / "Yes." / "yes, that's right" / "True" / "1".
- Substring of a longer correct answer that contains the key fact: gold \
"The Boeing 777 was first flown on June 12, 1994 by Emirates" / pred \
"Boeing 777" — match if the question asked which jetliner.

## What to FLAG as INCORRECT

- Wrong fact: different name, different number/year off by any amount, \
different entity. "Sekiro" vs "The Last of Us Part II" — INCORRECT.
- Off-by-one or off-by-small-delta on a number when precision matters: \
"43" vs "44", "10" vs "12" — INCORRECT (the answer is the number, so a \
different number is wrong).
- Missing key part of a multi-part answer: gold "Bowens, Thomas, & \
Madison" / pred "Bowens, Thomas" — INCORRECT (missing one of three names). \
But "Bowens; Thomas; Madison" / "Bowens Thomas Madison" — CORRECT (same \
three names, cosmetic delimiter).
- Empty / hallucinated tool-call JSON / refusals.

## Rule of thumb

If a careful human reader would say "yes, the model got it right, just \
formatted differently," mark CORRECT. If they'd say "the model got the \
fact wrong," mark INCORRECT.

Respond with a single JSON object — no preamble, no code fence — of the form:
{"correct": true|false, "reason": "<one sentence>"}"""


JUDGE_USER_TEMPLATE = """Question:
{question}

Gold answer:
{gold}

Predicted answer:
{predicted}

Respond with the JSON object only."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class JudgeVerdict:
    correct: bool
    reason: str = ""
    raw_text: str = ""
    error: str = ""


def _parse_verdict(text: str) -> JudgeVerdict:
    if not text:
        return JudgeVerdict(correct=False, reason="empty judge response", raw_text=text)
    candidates: list[str] = []
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.append(match.group(0))
    candidates.append(text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and "correct" in parsed:
            correct = bool(parsed.get("correct"))
            reason = str(parsed.get("reason", "") or "")
            return JudgeVerdict(correct=correct, reason=reason, raw_text=text)
    return JudgeVerdict(
        correct=False, reason="could not parse verdict JSON", raw_text=text
    )


class LLMJudge:
    """Grade (question, gold, predicted) triples via an OpenAI-compatible LLM."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package missing; install with `pip install openai`")
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "LLMJudge needs OPENAI_API_KEY (or pass api_key=). "
                "Disable the judge with --no-llm-judge on the CLI if you don't have one."
            )
        self._client = OpenAI(base_url=base_url or None, api_key=resolved_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def judge(self, *, question: str, gold: str, predicted: str) -> JudgeVerdict:
        if not predicted.strip():
            # Empty predictions are never correct — skip the call entirely.
            return JudgeVerdict(correct=False, reason="empty prediction")
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(
                            question=question,
                            gold=gold,
                            predicted=predicted,
                        ),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            text = completion.choices[0].message.content or ""
            verdict = _parse_verdict(text)
            _log.debug(
                "judge q=%s correct=%s reason=%s",
                question[:80], verdict.correct, verdict.reason[:120],
            )
            return verdict
        except Exception as exc:  # noqa: BLE001
            _log.warning("judge call failed: %s", exc)
            return JudgeVerdict(correct=False, reason="judge call failed", error=str(exc))

    def close(self) -> None:
        """Optional cleanup hook."""
