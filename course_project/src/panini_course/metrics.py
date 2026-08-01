"""Retrieval and answer metrics used by the course notebooks."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(
            normalized_prediction == normalize_answer(answer)
            for answer in gold_answers
        )
    )


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    best = 0.0
    for answer in gold_answers:
        gold_tokens = normalize_answer(answer).split()
        common = Counter(prediction_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if not prediction_tokens or not gold_tokens:
            score = float(prediction_tokens == gold_tokens)
        elif overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def recall_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0
