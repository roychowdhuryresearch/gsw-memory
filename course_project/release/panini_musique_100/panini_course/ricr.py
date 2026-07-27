"""RICR data structures and student implementation scaffold."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


PLACEHOLDER_PATTERN = re.compile(r"<ENTITY_Q(\d+)>")


@dataclass(frozen=True)
class Candidate:
    qa_uid: str
    answer: str
    score: float
    question: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainState:
    steps: tuple[Candidate, ...]
    answers_by_step: Mapping[int, str]
    score: float

    @property
    def current_answer(self) -> str:
        return self.steps[-1].answer


def geometric_mean(scores: Sequence[float], epsilon: float = 1e-12) -> float:
    """Return the length-normalized chain score used by RICR."""

    if not scores:
        return 0.0
    safe = [max(float(score), epsilon) for score in scores]
    return float(math.exp(sum(math.log(score) for score in safe) / len(safe)))


def instantiate_question(
    template: str,
    answers_by_step: Mapping[int, str],
) -> str:
    """Replace every ``<ENTITY_Qn>`` with the answer produced at step n."""

    def replace(match: re.Match[str]) -> str:
        step = int(match.group(1))
        if step not in answers_by_step:
            raise KeyError(
                f"Question template references unresolved step Q{step}: "
                f"{template}"
            )
        return answers_by_step[step]

    return PLACEHOLDER_PATTERN.sub(replace, template)


def prune_unique_answers(
    chains: Sequence[ChainState],
    beam_width: int,
) -> list[ChainState]:
    """Keep the best chain for each normalized current answer.

    TODO(student): implement deterministic score ordering, answer
    deduplication, and beam-width pruning.
    """

    raise NotImplementedError("Implement unique-answer beam pruning")


def run_linear_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
) -> list[ChainState]:
    """Run RICR over one linear decomposed sub-question sequence.

    TODO(student): implement Algorithm 1 from the project paper. Use
    ``instantiate_question``, expand every surviving beam at every retrieval
    hop, score chains by geometric mean, and call ``prune_unique_answers``
    after each expansion.
    """

    raise NotImplementedError("Implement linear RICR beam search")
