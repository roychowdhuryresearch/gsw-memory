"""Scoring utilities for FRAMES-style QA.

- ``exact_match``: canonicalize strings and test equality.
- ``token_f1``: word-level precision/recall F1 on normalized tokens.

Matches the conventions used by the original FRAMES paper and by
downstream work (ASearcher, Search-o1, etc.): lowercase, strip
articles, drop punctuation, collapse whitespace.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _ARTICLE_RE.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = _WS_RE.sub(" ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def any_alias_match(pred: str, gold_list: list[str]) -> bool:
    """Exact-match against any of the gold aliases."""
    pred_norm = normalize_answer(pred)
    return any(pred_norm == normalize_answer(g) for g in gold_list if g)


def best_f1(pred: str, gold_list: list[str]) -> float:
    """Best F1 across all gold aliases."""
    return max((token_f1(pred, g) for g in gold_list if g), default=0.0)
