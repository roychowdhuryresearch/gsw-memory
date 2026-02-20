"""
veRL reward function interface for GSW sleep-time RLVR training.

veRL calls compute_score() for each generated response. This module:
  1. Parses bridge QA pairs from the agent's tool call trace
  2. Delegates to reward.compute_reward() for the final score

veRL interface (required signature):
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

Where:
    data_source:  dataset name string (e.g. "musique")
    solution_str: the agent's full generated response text
    ground_truth: gold answer string
    extra_info:   dict with {gsw_dirs, decomposition, answer_aliases} (from parquet extra_info column)
"""

import json
import re
from typing import Any, Dict, List, Optional

from .reward import compute_reward


# ---------------------------------------------------------------------------
# Parse bridge QA pairs from agent response
# ---------------------------------------------------------------------------

# Pattern for tool calls in the agent's response
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _extract_bridges_from_response(solution_str: str) -> List[Dict[str, Any]]:
    """
    Parse all create_bridge_qa tool calls from the agent's response text
    and collect the bridge QA pairs that were submitted.

    Handles both single-bridge and batch-bridge modes:
      - Single: create_bridge_qa(question=..., answer=..., source_docs=..., reasoning=...)
      - Batch:  create_bridge_qa(bridges=[{...}, {...}])
    """
    bridges = []

    for match in _TOOL_CALL_PATTERN.finditer(solution_str):
        try:
            call = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue

        if call.get("name") != "create_bridge_qa":
            continue

        args = call.get("arguments", {})

        if "bridges" in args:
            # Batch mode
            for b in args["bridges"]:
                if isinstance(b, dict) and "question" in b and "answer" in b:
                    bridges.append({
                        "question": b.get("question", ""),
                        "answer": b.get("answer", ""),
                        "source_docs": b.get("source_docs", []),
                        "reasoning": b.get("reasoning", ""),
                    })
        elif "question" in args and "answer" in args:
            # Single mode
            bridges.append({
                "question": args.get("question", ""),
                "answer": args.get("answer", ""),
                "source_docs": args.get("source_docs", []),
                "reasoning": args.get("reasoning", ""),
            })

    return bridges


# ---------------------------------------------------------------------------
# veRL reward function — required interface
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Any] = None,
) -> float:
    """
    veRL reward function for GSW sleep-time bridge compilation.

    Args:
        data_source:  Dataset name (e.g. "musique"). Unused but required by veRL.
        solution_str: Full agent response text including all tool calls.
        ground_truth: Gold answer string for the MuSiQue question.
        extra_info:   Optional dict or JSON string with:
                        - answer_aliases: list of alternative correct answers
                        - decomposition:  MuSiQue sub-question decomposition steps

    Returns:
        Non-negative float reward score.
    """
    # Parse extra_info
    aliases: List[str] = []
    decomposition: List[Dict[str, Any]] = []

    if extra_info is not None:
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except (json.JSONDecodeError, ValueError):
                extra_info = {}
        if isinstance(extra_info, dict):
            aliases = extra_info.get("answer_aliases", [])
            decomposition = extra_info.get("decomposition", [])

    # Extract bridges from the agent's response
    bridges = _extract_bridges_from_response(solution_str)

    # No bridges created → zero reward
    if not bridges:
        return 0.0

    return compute_reward(
        bridges=bridges,
        gold_answer=ground_truth,
        gold_decomposition=decomposition,
        gold_aliases=aliases,
    )
