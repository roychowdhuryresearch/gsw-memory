"""Reference data structures and beam-search core for course-scale RICR."""

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
    if not scores:
        return 0.0
    safe = [max(float(score), epsilon) for score in scores]
    return float(math.exp(sum(math.log(score) for score in safe) / len(safe)))


def instantiate_question(
    template: str,
    answers_by_step: Mapping[int, str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        step = int(match.group(1))
        if step not in answers_by_step:
            raise KeyError(
                f"Question template references unresolved step Q{step}: {template}"
            )
        return answers_by_step[step]

    return PLACEHOLDER_PATTERN.sub(replace, template)


def prune_unique_answers(
    chains: Sequence[ChainState],
    beam_width: int,
) -> list[ChainState]:
    """Keep the best-scoring chain for each normalized current answer."""

    if beam_width <= 0:
        return []
    ordered = sorted(
        chains,
        key=lambda chain: (
            -chain.score,
            chain.current_answer.casefold(),
            tuple(step.qa_uid for step in chain.steps),
        ),
    )
    kept: list[ChainState] = []
    seen: set[str] = set()
    for chain in ordered:
        answer_key = " ".join(chain.current_answer.casefold().split())
        if answer_key in seen:
            continue
        seen.add(answer_key)
        kept.append(chain)
        if len(kept) == beam_width:
            break
    return kept


def run_linear_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
) -> list[ChainState]:
    """Run Algorithm 1 for one linear decomposed sub-question sequence."""

    retrieval_steps = [
        row
        for row in decomposed_questions
        if bool(row.get("requires_retrieval", True))
    ]
    if not retrieval_steps:
        return []

    first_question = instantiate_question(
        str(retrieval_steps[0]["question"]), {}
    )
    first_candidates = retrieve_and_score(
        first_question, candidates_per_hop
    )
    beams = prune_unique_answers(
        [
            ChainState(
                steps=(candidate,),
                answers_by_step={1: candidate.answer},
                score=max(candidate.score, 1e-12),
            )
            for candidate in first_candidates
        ],
        beam_width,
    )

    for step_number, row in enumerate(retrieval_steps[1:], start=2):
        expanded: list[ChainState] = []
        for beam in beams:
            concrete_question = instantiate_question(
                str(row["question"]), beam.answers_by_step
            )
            for candidate in retrieve_and_score(
                concrete_question, candidates_per_hop
            ):
                steps = beam.steps + (candidate,)
                answers = dict(beam.answers_by_step)
                answers[step_number] = candidate.answer
                expanded.append(
                    ChainState(
                        steps=steps,
                        answers_by_step=answers,
                        score=geometric_mean(
                            [step.score for step in steps]
                        ),
                    )
                )
        beams = prune_unique_answers(expanded, beam_width)
        if not beams:
            break
    return beams
