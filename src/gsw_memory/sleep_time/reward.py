"""
Reward function for RLVR training of the sleep-time GSW exploration agent.

Reward signal:
    - Main reward:  10 × token_F1(best bridge answer, gold answer)
                    → 0 if best F1 < 0.5 (no partial credit below threshold)
    - Bridge bonus: per bridge — multi-hop depth + novelty bonuses
    - Decomp bonus: bridges that directly answer decomposition sub-questions

All components are non-negative. Total reward ≥ 0.
"""

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Token-level F1 (standard MuSiQue / SQuAD metric)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1_against_aliases(pred: str, gold: str, aliases: List[str]) -> float:
    """Return max token F1 against gold answer and all its aliases."""
    candidates = [gold] + (aliases or [])
    return max(token_f1(pred, c) for c in candidates)


# ---------------------------------------------------------------------------
# Bridge novelty
# ---------------------------------------------------------------------------

def _bridge_key(bridge: Dict[str, Any]) -> str:
    """Canonical string for deduplication."""
    q = _normalize(bridge.get("question", ""))
    a = _normalize(bridge.get("answer", ""))
    return f"{q}||{a}"


def is_novel(bridge: Dict[str, Any], all_bridges: List[Dict[str, Any]]) -> bool:
    """True if this bridge's (question, answer) pair is unique in the episode."""
    key = _bridge_key(bridge)
    return sum(1 for b in all_bridges if _bridge_key(b) == key) == 1


# ---------------------------------------------------------------------------
# Decomposition coverage bonus
# ---------------------------------------------------------------------------

def decomp_coverage_bonus(
    bridges: List[Dict[str, Any]],
    gold_decomposition: List[Dict[str, Any]],
    f1_threshold: float = 0.5,
) -> float:
    """
    Reward bridges that directly answer one of the MuSiQue sub-questions.

    Each covered sub-question adds +1.0. A sub-question is covered if any
    bridge's answer has F1 ≥ f1_threshold against the sub-question's answer.
    """
    if not gold_decomposition or not bridges:
        return 0.0

    bonus = 0.0
    bridge_answers = [b.get("answer", "") for b in bridges]

    for step in gold_decomposition:
        sub_answer = step.get("answer", "")
        if not sub_answer:
            continue
        if any(token_f1(ba, sub_answer) >= f1_threshold for ba in bridge_answers):
            bonus += 1.0

    return bonus


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    bridges: List[Dict[str, Any]],
    gold_answer: str,
    gold_decomposition: Optional[List[Dict[str, Any]]] = None,
    gold_aliases: Optional[List[str]] = None,
    main_scale: float = 10.0,
    f1_threshold: float = 0.5,
) -> float:
    """
    Compute the GRPO reward for one episode.

    Args:
        bridges:          All bridge QA pairs created during the episode.
                          Each bridge is a dict with keys:
                          question, answer, source_docs, reasoning.
        gold_answer:      Gold final answer for the MuSiQue question.
        gold_decomposition: MuSiQue decomposition steps (list of sub-QA dicts).
        gold_aliases:     Alternative correct answers for the gold answer.
        main_scale:       Multiplier for the main F1 reward (default 10).
        f1_threshold:     Minimum F1 to count as a correct answer (default 0.5).

    Returns:
        Non-negative scalar reward.

    Reward breakdown:
        main_reward  = main_scale × best_F1(bridge answers, gold answer)
                       → 0.0 if best F1 < f1_threshold

        bridge_bonus = Σ over bridges:
                         (is_multihop) × (
                             1.0                               # base multi-hop bonus
                           + 0.5 × max(0, num_source_docs - 2) # depth bonus
                           + 0.5 × is_novel(bridge)            # novelty bonus
                         )
                       where is_multihop = (num_source_docs >= 2)

        decomp_bonus = number of gold decomposition sub-answers covered by bridges
    """
    if not bridges:
        return 0.0

    # --- Main reward ---
    bridge_answers = [b.get("answer", "") for b in bridges]
    best_f1 = max(
        best_f1_against_aliases(a, gold_answer, gold_aliases or [])
        for a in bridge_answers
    )
    if best_f1 < f1_threshold:
        return 0.0
    main_reward = main_scale * best_f1

    # --- Bridge quality bonus ---
    bridge_bonus = 0.0
    for bridge in bridges:
        num_docs = len(bridge.get("source_docs", []))
        is_multihop = num_docs >= 2
        if not is_multihop:
            continue
        depth_bonus = 0.5 * max(0, num_docs - 2)
        novelty_bonus = 0.5 * float(is_novel(bridge, bridges))
        bridge_bonus += 1.0 + depth_bonus + novelty_bonus

    # --- Decomposition coverage bonus ---
    decomp_bonus = decomp_coverage_bonus(bridges, gold_decomposition or [])

    return main_reward + bridge_bonus + decomp_bonus


# ---------------------------------------------------------------------------
# Reward breakdown (for logging / debugging)
# ---------------------------------------------------------------------------

def reward_breakdown(
    bridges: List[Dict[str, Any]],
    gold_answer: str,
    gold_decomposition: Optional[List[Dict[str, Any]]] = None,
    gold_aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return a detailed breakdown of all reward components (for logging).
    """
    if not bridges:
        return {
            "total": 0.0,
            "main_reward": 0.0,
            "best_f1": 0.0,
            "bridge_bonus": 0.0,
            "decomp_bonus": 0.0,
            "num_bridges": 0,
            "num_multihop_bridges": 0,
        }

    bridge_answers = [b.get("answer", "") for b in bridges]
    best_f1 = max(
        best_f1_against_aliases(a, gold_answer, gold_aliases or [])
        for a in bridge_answers
    )

    main_reward = 10.0 * best_f1 if best_f1 >= 0.5 else 0.0
    bridge_bonus = 0.0
    multihop_count = 0
    per_bridge = []

    for bridge in bridges:
        num_docs = len(bridge.get("source_docs", []))
        is_mh = num_docs >= 2
        if is_mh:
            multihop_count += 1
        depth_b = 0.5 * max(0, num_docs - 2) if is_mh else 0.0
        novelty_b = 0.5 * float(is_novel(bridge, bridges)) if is_mh else 0.0
        b_contrib = (1.0 + depth_b + novelty_b) if is_mh else 0.0
        bridge_bonus += b_contrib
        per_bridge.append({
            "question": bridge.get("question", ""),
            "answer": bridge.get("answer", ""),
            "source_docs": bridge.get("source_docs", []),
            "is_multihop": is_mh,
            "depth_bonus": depth_b,
            "novelty_bonus": novelty_b,
            "contribution": b_contrib,
        })

    decomp_bonus = decomp_coverage_bonus(bridges, gold_decomposition or [])
    total = main_reward + bridge_bonus + decomp_bonus

    return {
        "total": round(total, 4),
        "main_reward": round(main_reward, 4),
        "best_f1": round(best_f1, 4),
        "bridge_bonus": round(bridge_bonus, 4),
        "decomp_bonus": round(decomp_bonus, 4),
        "num_bridges": len(bridges),
        "num_multihop_bridges": multihop_count,
        "per_bridge": per_bridge,
    }
