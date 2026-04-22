"""Heuristic failure-mode classifier.

Runs against a scored ``QuestionResult`` (so scoring has already
flagged correct vs incorrect) and assigns one ``FailureMode`` tag.
Deterministic — no LLM calls. The classifier's job is to put each
question in the right bucket for the failure-mode histogram, not to
explain why a reasoning chain went wrong.

The rules are intentionally simple and applied in order; first match
wins. Future work: an LLM-backed auditor that explains the failure,
keyed off this tag.
"""

from __future__ import annotations

from collections import Counter

from research_agent.models.failure_modes import FailureMode
from research_agent.models.trace import QuestionResult


# Tunable thresholds for the heuristic rules.
MIN_EXPECTED_TURNS_PER_HOP = 1
LOOP_REPEAT_THRESHOLD = 3


def classify(result: QuestionResult, *, num_hops: int | None = None) -> FailureMode:
    """Assign a single failure mode to a scored question result."""

    if result.exact_match or (result.judge_correct is True):
        return FailureMode.CORRECT

    traj = result.trajectory

    # 1) Adapter crash / no trajectory.
    if traj.turns == 0 and not result.predicted_answer and not traj.tool_calls:
        return FailureMode.UNKNOWN

    # 2) Tool error the agent couldn't recover from.
    tool_errors = [tc for tc in traj.tool_calls if tc.error]
    if tool_errors and not result.predicted_answer:
        return FailureMode.TOOL_ERROR

    # 3) Budget exceeded.
    budget_marker = (traj.extra.get("stopped_reason") or "").lower()
    if budget_marker in {"max_turns", "max_tokens", "budget_exceeded"}:
        return FailureMode.BUDGET_EXCEEDED

    # 4) Looped tool calls — same (name, args) repeated >= threshold.
    if traj.tool_calls:
        repeats = Counter(
            (tc.name, repr(sorted(tc.args.items())) if tc.args else "")
            for tc in traj.tool_calls
        )
        if any(c >= LOOP_REPEAT_THRESHOLD for c in repeats.values()):
            return FailureMode.LOOP

    # 5) Hallucination — model emitted a (wrong) answer without any retrieval
    #    at all. Must come before early_stop: "no retrieval calls + a
    #    confident answer" is a stronger signal than "trajectory was short."
    retrieval_calls = [
        tc for tc in traj.tool_calls
        if tc.name.lower() in {"search", "retrieve", "read", "visit", "browse", "query"}
    ]
    if not retrieval_calls and result.predicted_answer:
        return FailureMode.HALLUCINATION

    # 6) Early stop — fewer turns than hops suggest, but the agent DID retrieve.
    if num_hops and traj.turns and traj.turns < max(1, num_hops * MIN_EXPECTED_TURNS_PER_HOP):
        return FailureMode.EARLY_STOP

    # 7) Decomposition gaps — adapter recorded < num_hops sub-questions.
    n_sub_qs = int(traj.extra.get("n_sub_questions") or 0)
    if num_hops and 0 < n_sub_qs < num_hops:
        return FailureMode.INCOMPLETE_DECOMPOSITION

    # 8) Wrong retrieval vs wrong synthesis.
    #    - Retrieval happened but relevant docs (from gold article titles)
    #      never appeared in any tool result = wrong_retrieval.
    #    - Otherwise = wrong_synthesis.

    gold_titles = [
        a.get("title", "").lower() for a in (traj.extra.get("gold_articles") or []) if a.get("title")
    ]
    if gold_titles:
        seen = " ".join(tc.result_preview.lower() for tc in traj.tool_calls)
        if not any(t in seen for t in gold_titles if t):
            return FailureMode.WRONG_RETRIEVAL

    return FailureMode.WRONG_SYNTHESIS


def histogram(results: list[QuestionResult]) -> dict[str, int]:
    """Build a failure-mode histogram over a batch of results."""
    hist: Counter[str] = Counter()
    for r in results:
        hist[r.failure_mode.value] += 1
    return dict(hist)
