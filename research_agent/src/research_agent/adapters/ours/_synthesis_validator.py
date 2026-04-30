"""Synthesis validator for the orchestrator's submit_answer.

Targets the wrong-fact-pick failure mode (Bucket C in the 33-✗
analysis): all blanks are resolved correctly but the final synthesis
picks the wrong target value. Examples:

  - q345 (MLB hitting coach in Moana's release year): pred=Mike
    Hargrove, gold=Ty Van Burkleo. Both names appeared in retrieved
    chunks; model picked the wrong role.
  - q702 (Digimon trivia): pred=Birdramon, gold=Kabuterimon. Same
    pattern.

Mechanism: one extra LLM call right before submit_answer finalizes,
giving the validator the question + each resolved blank's value AND
evidence-chunk snippets. The orchestrator decides whether the verdict
is log-only or reject-capable; the adapter default is currently off so
evals do not add another stochastic model call unless requested.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from research_agent.adapters.ours._planner_exec import (
    BlankResult,
    GSWPlan,
    _extract_json_object,
)


_log = logging.getLogger(__name__)


class ValidatorVerdict(BaseModel):
    """Output of the synthesis validator."""

    verdict: str = Field(description="'correct' or 'wrong'")
    reason: str = Field(description="≤1 sentence explanation")
    flagged_blank: Optional[str] = Field(
        default=None,
        description="Blank id that looks wrong, if any",
    )
    suggested_correction: Optional[str] = Field(
        default=None,
        description="What the answer should be, if wrong",
    )
    corrected_answer: Optional[str] = Field(
        default=None,
        description="Final answer to submit if confidently repairable",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="'high', 'medium', or 'low'",
    )
    error_type: Optional[str] = Field(
        default=None,
        description=(
            "format, arithmetic, spelling, wrong_entity, missing_detail, "
            "structural_plan_error, or other"
        ),
    )
    evidence_support: Optional[str] = Field(
        default=None,
        description="Short note tying corrected_answer to shown evidence",
    )
    raw_response: str = Field(default="", exclude=True)


VALIDATOR_SYSTEM = """You are a synthesis validator for a multi-hop research agent.

The agent retrieved intermediate facts (each with an evidence chunk
snippet) to answer a question, and is about to submit a final answer.
Your job is to decide whether the final answer follows from the
retrieved evidence.

CRITICAL RULES:
1. Be conservative. Only flag "wrong" if you have CLEAR evidence the
   proposed answer is inconsistent with the retrieved facts.
2. When in doubt, return "correct". A false flag forces the agent to
   redo work and frustrates the loop. Missing one wrong answer is
   strictly preferable to false-flagging a correct one.
3. Do NOT use your prior knowledge of the world to second-guess the
   answer. Only use the retrieved evidence shown to you.
4. Numeric arithmetic: re-do the computation from the cited values.
   If the math is consistent, the answer is correct.
5. Format / cosmetic differences are NOT wrong. "Katie" vs "Kathleen"
   = correct. "84512" vs "84,512" = correct.

If the verdict is "wrong", provide a concise `corrected_answer` ONLY
when the shown retrieved facts support the exact final answer. Do not
guess from prior knowledge. If the plan itself is structurally wrong,
set `error_type="structural_plan_error"` and leave `corrected_answer`
null unless the final answer is directly supported anyway.

Output a single JSON object, no other text:
{
  "verdict": "correct" | "wrong",
  "reason": "<one sentence>",
  "flagged_blank": "<blank id whose committed value seems wrong, or null>",
  "suggested_correction": "<backward-compatible alias for corrected_answer, or null>",
  "corrected_answer": "<final answer only, or null>",
  "confidence": "high" | "medium" | "low",
  "error_type": "format" | "arithmetic" | "spelling" | "wrong_entity" | "missing_detail" | "structural_plan_error" | "other",
  "evidence_support": "<brief reference to the retrieved facts, or null>"
}
"""


SAFE_AUTO_REPAIR_ERROR_TYPES = {
    "format",
    "arithmetic",
    "spelling",
    "missing_detail",
}


def auto_repair_candidate(
    verdict: ValidatorVerdict,
    proposed_answer: str,
) -> tuple[Optional[str], str]:
    """Return a gated corrected answer, or ``(None, reason)``.

    The gate is intentionally conservative: it only accepts high-confidence
    final-value repairs for local answer mistakes. Structural-plan and
    wrong-entity repairs stay in the normal reject/rethink path.
    """
    if verdict.verdict != "wrong":
        return None, "verdict is not wrong"

    corrected = (
        (verdict.corrected_answer or "").strip()
        or (verdict.suggested_correction or "").strip()
    )
    if not corrected:
        return None, "missing corrected_answer"

    if corrected == (proposed_answer or "").strip():
        return None, "corrected_answer equals proposed_answer"

    confidence = (verdict.confidence or "").strip().lower()
    if confidence != "high":
        return None, f"confidence is not high: {confidence or 'missing'}"

    error_type = (verdict.error_type or "").strip().lower()
    if error_type not in SAFE_AUTO_REPAIR_ERROR_TYPES:
        return None, f"unsafe error_type: {error_type or 'missing'}"

    # Avoid accepting explanatory validator prose as the final answer.
    if len(corrected) > 240:
        return None, "corrected_answer too long"
    if corrected[:1] in {"{", "["} and corrected[-1:] in {"}", "]"}:
        return None, "corrected_answer looks like structured data"

    return corrected, "accepted"


def _format_evidence_snippets(
    state: dict[str, BlankResult],
    plan: GSWPlan,
    corpus,
    snippet_chars: int = 900,
    max_chunks_per_blank: int = 3,
) -> str:
    """Render each resolved blank as: id = value\n  EVIDENCE: snippet."""
    lines = []
    for blank in plan.blank_entities():
        res = state.get(blank.id)
        if not res or res.status != "resolved":
            continue
        role = blank.role or ""
        target_marker = " [TARGET]" if blank.is_target else ""
        lines.append(f"• {blank.id} ({role}){target_marker} = {res.value!r}")
        chunk_ids = (res.evidence_chunk_ids or [])[:max_chunks_per_blank]
        for cid in chunk_ids:
            chunk = corpus.get_chunk(cid)
            if chunk is None:
                lines.append(f"    EVIDENCE [{cid}]: (chunk not found)")
                continue
            text = (chunk.text or "").strip().replace("\n", " ")
            if len(text) > snippet_chars:
                text = text[:snippet_chars] + "…"
            lines.append(f"    EVIDENCE [{cid}]: {text}")
    return "\n".join(lines) if lines else "(no evidence)"


def build_validator_messages(
    question: str,
    plan: GSWPlan,
    state: dict[str, BlankResult],
    proposed_answer: str,
    corpus,
) -> list[dict[str, str]]:
    """Build the validator chat messages."""
    evidence = _format_evidence_snippets(state, plan, corpus)
    user = (
        f"Question: {question.strip()}\n\n"
        f"Retrieved facts:\n{evidence}\n\n"
        f"Proposed answer: {proposed_answer!r}\n\n"
        "Output JSON only."
    )
    return [
        {"role": "system", "content": VALIDATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


def validate_synthesis(
    *,
    question: str,
    plan: GSWPlan,
    state: dict[str, BlankResult],
    proposed_answer: str,
    corpus,
    llm: Any,
    llm_seed: Optional[int] = None,
    max_tokens: int = 600,
) -> ValidatorVerdict:
    """One LLM call. Returns a ValidatorVerdict.

    Falls back to verdict="correct" on any LLM error or parse failure
    (conservative — never block submit on validator infrastructure
    issues).
    """
    if not (proposed_answer or "").strip():
        # Don't validate empty submits — nothing to check.
        return ValidatorVerdict(
            verdict="correct",
            reason="empty submit; validator skipped",
        )

    messages = build_validator_messages(
        question=question,
        plan=plan,
        state=state,
        proposed_answer=proposed_answer,
        corpus=corpus,
    )

    try:
        resp = llm.chat(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            stream=False,
            seed=llm_seed,
        )
    except Exception as exc:  # noqa: BLE001
        _log.info("validator LLM call failed: %s — falling back to correct", exc)
        return ValidatorVerdict(
            verdict="correct",
            reason=f"validator LLM error: {exc}",
        )

    text = (getattr(resp, "text", "") or "").strip()
    try:
        data = _extract_json_object(text)
    except Exception as exc:  # noqa: BLE001
        _log.info("validator parse failed: %s — falling back to correct", exc)
        return ValidatorVerdict(
            verdict="correct",
            reason="validator output unparseable",
            raw_response=text[:400],
        )

    verdict = (data.get("verdict") or "").strip().lower()
    if verdict not in ("correct", "wrong"):
        return ValidatorVerdict(
            verdict="correct",
            reason="validator returned unexpected verdict",
            raw_response=text[:400],
        )

    suggested = (data.get("suggested_correction") or "").strip() or None
    corrected = (data.get("corrected_answer") or "").strip() or None
    if corrected is None:
        corrected = suggested

    return ValidatorVerdict(
        verdict=verdict,
        reason=(data.get("reason") or "")[:240],
        flagged_blank=data.get("flagged_blank") or None,
        suggested_correction=suggested,
        corrected_answer=corrected,
        confidence=(data.get("confidence") or "").strip().lower() or None,
        error_type=(data.get("error_type") or "").strip().lower() or None,
        evidence_support=(data.get("evidence_support") or "")[:240] or None,
        raw_response=text[:400],
    )


def build_rejection_message(verdict: ValidatorVerdict, proposed_answer: str) -> str:
    """The user-message we inject when the validator flags wrong.

    v1 wording: surface the validator's reason and any optional
    suggested correction, give the orchestrator ONE remaining attempt,
    and lay out the two recovery paths (re-dispatch or re-submit).
    """
    flagged_line = (
        f"Flagged intermediate blank: {verdict.flagged_blank!r}\n"
        if verdict.flagged_blank else ""
    )
    suggested_line = (
        f"Validator's suggested correction: {verdict.suggested_correction!r}\n"
        if verdict.suggested_correction else ""
    )
    return (
        f"⚠ SYNTHESIS VALIDATOR REJECTED your previous submit_answer({proposed_answer!r}).\n"
        f"\n"
        f"Reason: {verdict.reason}\n"
        f"{flagged_line}"
        f"{suggested_line}"
        f"\n"
        "You have ONE remaining attempt. Either:\n"
        "1. Re-dispatch the flagged blank with sharper hints to gather the right evidence.\n"
        "2. Re-submit a different answer if the validator's suggestion is correct.\n"
        "\n"
        "Your next submit_answer will be accepted as-is."
    )
