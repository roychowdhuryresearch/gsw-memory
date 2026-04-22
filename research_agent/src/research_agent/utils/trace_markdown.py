"""Render a ``QuestionResult`` / ``Trajectory`` into a readable markdown doc.

JSON traces are the machine-readable source of truth; these .md files
sit beside them for human auditing. We interleave reasoning blocks
(parsed out of ``hidden_reasoning`` with ``[turn N]`` tags) with the
tool calls and their full results, producing one section per turn.

File layout per cell:
    logs/<cell>/traces/q_<id>.json   (machine)
    logs/<cell>/traces/q_<id>.md     (human)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from research_agent.models.trace import QuestionResult, ToolCall, Trajectory


_TURN_HEADER_RE = re.compile(r"\[(?:turn|stage|retrieval turn|reasoner)\s*[^\]]*\]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verdict_badge(qr: QuestionResult) -> str:
    if qr.exact_match:
        return "✅ **CORRECT (exact-match)**"
    if qr.judge_correct is True:
        return "🟢 **CORRECT (LLM-judge)** — predicted wording differs from gold but is semantically right"
    if qr.judge_correct is False:
        return f"❌ **WRONG** (judge=no, failure_mode = `{qr.failure_mode.value}`)"
    return f"❌ **WRONG** (failure_mode = `{qr.failure_mode.value}`)"


def _pretty_json(payload: str | dict | list, max_lines: int = 80) -> str:
    """Return a ```json code block for a payload, truncated for very long lists."""
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except Exception:  # noqa: BLE001
            return f"```\n{payload[:4000]}\n```"
    else:
        data = payload
    pretty = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    lines = pretty.splitlines()
    if len(lines) > max_lines:
        kept = "\n".join(lines[:max_lines])
        dropped = len(lines) - max_lines
        return f"```json\n{kept}\n… ({dropped} more lines truncated in markdown — full JSON is in q_*.json)\n```"
    return f"```json\n{pretty}\n```"


def _reasoning_by_turn(hidden_reasoning: str) -> dict[str, str]:
    """Parse ``[turn N]`` (or [stage], [reasoner]) tags in hidden_reasoning and
    bucket the text that follows each tag."""
    if not hidden_reasoning:
        return {}
    parts = _TURN_HEADER_RE.split(hidden_reasoning)
    headers = _TURN_HEADER_RE.findall(hidden_reasoning)
    out: dict[str, str] = {}
    # First chunk before any header is unattributed — dump under "preamble".
    if parts and parts[0].strip():
        out["preamble"] = parts[0].strip()
    for h, text in zip(headers, parts[1:]):
        key = h.strip("[]").strip()
        out.setdefault(key, "")
        out[key] += text.strip() + "\n"
    return out


def _tool_calls_by_turn(tool_calls: list[ToolCall]) -> dict[int, list[ToolCall]]:
    bucketed: dict[int, list[ToolCall]] = defaultdict(list)
    for tc in tool_calls:
        bucketed[tc.turn].append(tc)
    return dict(sorted(bucketed.items()))


def _turn_section(turn: int, reasoning: str, tcs: list[ToolCall]) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Turn {turn}")
    lines.append("")

    if reasoning:
        lines.append("**Reasoning**")
        lines.append("")
        lines.append("> " + reasoning.strip().replace("\n", "\n> "))
        lines.append("")

    if not tcs:
        lines.append("_(no tool calls this turn)_")
        lines.append("")
        return lines

    for i, tc in enumerate(tcs, 1):
        lines.append(f"### 🔧 Tool call {i}: `{tc.name}`  _(took {tc.duration_s:.2f}s)_")
        lines.append("")
        if tc.error:
            lines.append(f"❗ **error**: `{tc.error}`")
            lines.append("")
        if tc.args:
            lines.append("**Args**")
            lines.append("")
            lines.append(_pretty_json(tc.args, max_lines=40))
            lines.append("")
        lines.append("**Result**")
        lines.append("")
        # Prefer the full payload when available; fall back to preview.
        payload = tc.result_full or tc.result_preview or "(empty)"
        lines.append(_pretty_json(payload, max_lines=120))
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_question_markdown(qr: QuestionResult, *, context: dict[str, Any] | None = None) -> str:
    """Produce a markdown document for one scored question."""
    traj = qr.trajectory
    context = context or {}

    header: list[str] = []
    header.append(f"# Q{qr.question_id} — {_verdict_badge(qr)}")
    header.append("")
    header.append("## Question")
    header.append("")
    header.append(f"> {qr.question.strip()}")
    header.append("")

    # Summary table
    summary_rows = [
        ("System", f"`{traj.system_id}`"),
        ("Model", f"`{traj.model_id}`"),
        ("Turns", str(traj.turns)),
        ("Tool calls", str(len(traj.tool_calls))),
        ("Prompt tokens", str(traj.prompt_tokens)),
        ("Completion tokens", str(traj.completion_tokens)),
        ("Wall time", f"{traj.wall_time_s:.2f}s"),
        ("Exact match", "✓" if qr.exact_match else "✗"),
        ("Token F1", f"{qr.f1:.3f}"),
        (
            "LLM judge",
            "—" if qr.judge_correct is None
            else ("✓ correct" if qr.judge_correct else "✗ wrong"),
        ),
        ("Failure mode", f"`{qr.failure_mode.value}`"),
        ("Stopped reason", f"`{traj.extra.get('stopped_reason', '?')}`"),
    ]
    header.append("## Summary")
    header.append("")
    header.append("| Field | Value |")
    header.append("|---|---|")
    for k, v in summary_rows:
        header.append(f"| {k} | {v} |")
    header.append("")

    # Gold + predicted side-by-side
    header.append("## Answers")
    header.append("")
    header.append(f"**Gold**\n\n> {qr.gold_answer.strip()}")
    header.append("")
    header.append(f"**Predicted**\n\n> {qr.predicted_answer.strip() or '*(empty)*'}")
    header.append("")

    # Adapter-synthesized reasoning summary (tag contents / scratchpad)
    if traj.reasoning:
        header.append("## Adapter reasoning (synthesized)")
        header.append("")
        header.append("> " + traj.reasoning.strip().replace("\n", "\n> ")[:4000])
        header.append("")

    # Per-turn detail
    reasoning_buckets = _reasoning_by_turn(traj.hidden_reasoning)
    tools_by_turn = _tool_calls_by_turn(traj.tool_calls)

    # Preamble reasoning (rare: ours_gsw_v1 uses [decompose] / [aggregate] tags
    # rather than numeric turns).
    if "preamble" in reasoning_buckets:
        header.append("## Reasoning preamble")
        header.append("")
        header.append("> " + reasoning_buckets["preamble"].replace("\n", "\n> "))
        header.append("")

    # Either integer turns or stage-based tags (decompose, build_gsw, aggregate).
    numeric_turns = sorted(
        {int(re.search(r"\d+", k).group()) for k in reasoning_buckets if re.search(r"\d+", k)}
        | set(tools_by_turn.keys())
    )

    if numeric_turns:
        header.append("## Turns")
        header.append("")
        for turn in numeric_turns:
            matching_keys = [k for k in reasoning_buckets if f"turn {turn}" in k.lower() or f"turn{turn}" in k.lower()]
            reasoning_text = "\n".join(reasoning_buckets[k] for k in matching_keys)
            header.extend(_turn_section(turn, reasoning_text, tools_by_turn.get(turn, [])))

    # Stage-based sections (for ours_gsw_v1 and Context-1)
    stage_keys = [
        k for k in reasoning_buckets
        if not re.search(r"\d+", k) and k != "preamble"
    ]
    if stage_keys:
        header.append("## Stages")
        header.append("")
        for k in stage_keys:
            header.append(f"### 🧠 {k}")
            header.append("")
            header.append("> " + reasoning_buckets[k].strip().replace("\n", "\n> "))
            header.append("")

    # Messages (full conversation)
    if traj.messages:
        header.append("## Full message trace")
        header.append("")
        header.append("<details><summary>Click to expand</summary>")
        header.append("")
        header.append(_pretty_json(traj.messages, max_lines=300))
        header.append("")
        header.append("</details>")
        header.append("")

    # Adapter-specific extras (GSW snapshot, kept chunk ids, Q+ state, etc.)
    extra_items = {
        k: v for k, v in traj.extra.items()
        if k not in {"stopped_reason", "gold_articles", "traceback", "adapter_error"}
        and not k.startswith("_")
    }
    if extra_items:
        header.append("## Adapter state (extras)")
        header.append("")
        header.append(_pretty_json(extra_items, max_lines=200))
        header.append("")

    # Gold articles (from FRAMES) for context
    gold_articles = traj.extra.get("gold_articles") or []
    if gold_articles:
        header.append("## Gold articles required (per FRAMES)")
        header.append("")
        for a in gold_articles:
            title = a.get("title", "?")
            url = a.get("url", "")
            header.append(f"- [{title}]({url})" if url else f"- {title}")
        header.append("")

    return "\n".join(header).rstrip() + "\n"


def write_trace_pair(
    result: QuestionResult,
    *,
    trace_dir: Path | str,
) -> None:
    """Persist both q_<id>.json and q_<id>.md for a scored question."""
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    traj = result.trajectory
    (trace_dir / f"q_{result.question_id}.json").write_text(
        traj.model_dump_json(indent=2)
    )
    (trace_dir / f"q_{result.question_id}.md").write_text(
        render_question_markdown(result)
    )
