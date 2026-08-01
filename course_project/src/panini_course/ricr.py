"""Student-scale reference implementation of PANINI's DAG RICR algorithm."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Mapping, Sequence


PLACEHOLDER_PATTERN = re.compile(r"<ENTITY_Q(\d+)>")


@dataclass(frozen=True)
class Candidate:
    """One reranked GSW QA pair, before answer-entity expansion.

    ``answer_ids`` must be globally meaningful within the package. Because the
    supplied GSWs have document-local IDs, the package adapter uses values such
    as ``doc_17::e1``. It does not pretend that equal names across documents
    have been entity-reconciled.
    """

    qa_uid: str
    answer_names: tuple[str, ...]
    score: float
    question: str = ""
    answer_ids: tuple[str, ...] = ()
    answer_role_states: tuple[str, ...] = ()
    document_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainState:
    steps: tuple[Candidate, ...]
    answers_by_step: Mapping[int, str | tuple[str, ...]]
    score: float
    last_hop_score: float = 0.0

    @property
    def current_answers(self) -> tuple[str, ...]:
        if not self.steps:
            return ()
        return self.steps[-1].answer_names


@dataclass(frozen=True)
class RICRResult:
    components: tuple[tuple[int, ...], ...]
    chains: tuple[ChainState, ...]
    evidence: tuple[Candidate, ...]
    issued_queries: tuple[str, ...]
    fallback: bool = False


def normalize_entity_name(value: str) -> str:
    return " ".join(value.casefold().split())


def geometric_mean(scores: Sequence[float], epsilon: float = 1e-12) -> float:
    if not scores:
        return 0.0
    safe = [max(float(score), epsilon) for score in scores]
    return float(math.exp(sum(math.log(score) for score in safe) / len(safe)))


def panini_chain_score(steps: Sequence[Candidate]) -> float:
    """Match PANINI's cumulative score: normalize each score, then geometric mean."""

    normalized = [max(1e-6, min(1.0, 0.5 * (step.score + 1.0))) for step in steps]
    return geometric_mean(normalized, epsilon=1e-6)


def harmonic_mean(scores: Sequence[float]) -> float:
    valid = [float(score) for score in scores if float(score) > 1e-6]
    return len(valid) / sum(1.0 / score for score in valid) if valid else 1e-6


def instantiate_question(
    template: str,
    answers_by_step: Mapping[int, str | tuple[str, ...]],
) -> str:
    def replace(match: re.Match[str]) -> str:
        step = int(match.group(1))
        if step not in answers_by_step:
            raise KeyError(f"Question references unresolved Q{step}: {template}")
        value = answers_by_step[step]
        return ", ".join(value) if isinstance(value, tuple) else value

    return PLACEHOLDER_PATTERN.sub(replace, template)


def identify_retrieval_components(
    decomposed_questions: Sequence[Mapping[str, object]],
) -> list[list[int]]:
    """Find connected retrieval DAGs and topologically order each one (1-based IDs)."""

    retrieval = {
        index
        for index, row in enumerate(decomposed_questions, start=1)
        if bool(row.get("requires_retrieval", True))
    }
    parents: dict[int, list[int]] = {}
    children: dict[int, list[int]] = {index: [] for index in retrieval}
    for index in sorted(retrieval):
        references = [
            int(value)
            for value in PLACEHOLDER_PATTERN.findall(
                str(decomposed_questions[index - 1]["question"])
            )
        ]
        parents[index] = [value for value in references if value in retrieval]
        for parent in parents[index]:
            children[parent].append(index)

    components: list[list[int]] = []
    visited: set[int] = set()
    for start in sorted(retrieval):
        if start in visited:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(parents.get(node, []))
            stack.extend(children.get(node, []))
        visited.update(component)
        indegree = {
            node: len([parent for parent in parents[node] if parent in component])
            for node in component
        }
        queue = sorted(node for node, degree in indegree.items() if degree == 0)
        ordered: list[int] = []
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for child in sorted(children.get(node, [])):
                if child in component:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        queue.append(child)
                        queue.sort()
        if len(ordered) != len(component):
            raise ValueError(f"retrieval dependency cycle: {sorted(component)}")
        # PANINI sends a plan with no multi-question component through simple
        # retrieval of the original question.
        if len(ordered) > 1:
            components.append(ordered)
    return components


def _state_order(state: ChainState):
    return (-state.score, -state.last_hop_score, tuple(step.qa_uid for step in state.steps))


def _combine_parent_beams(
    parent_groups: Sequence[Sequence[ChainState]],
    beam_width: int,
    quality_threshold: float,
) -> list[ChainState]:
    combinations = [
        (harmonic_mean([state.score for state in states]), states)
        for states in product(*parent_groups)
    ]
    combinations.sort(
        key=lambda item: (
            -item[0],
            tuple(tuple(step.qa_uid for step in state.steps) for state in item[1]),
        )
    )
    selected = [item for item in combinations[:beam_width] if item[0] >= quality_threshold]
    if not selected and combinations:
        selected = combinations[:1]
    merged: list[ChainState] = []
    for score, states in selected:
        answers: dict[int, str | tuple[str, ...]] = {}
        steps: list[Candidate] = []
        for state in states:
            answers.update(state.answers_by_step)
            steps.extend(state.steps)
        merged.append(ChainState(tuple(steps), answers, score))
    return merged


def run_panini_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    original_question: str,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
    multi_dependency_threshold: float = 0.3,
    unique_intermediate_entities: bool = True,
) -> RICRResult:
    """Run faithful PANINI RICR over connected retrieval DAG components."""

    components = identify_retrieval_components(decomposed_questions)
    issued: list[str] = []
    if not components:
        issued.append(original_question)
        candidates = tuple(retrieve_and_score(original_question, candidates_per_hop))
        chains = tuple(
            ChainState((candidate,), {1: candidate.answer_names}, panini_chain_score([candidate]), candidate.score)
            for candidate in candidates[:beam_width]
        )
        return RICRResult((), chains, candidates, tuple(issued), fallback=True)

    final_states: list[ChainState] = []
    for component in components:
        completed: dict[int, list[ChainState]] = {}
        for position, step_id in enumerate(component):
            row = decomposed_questions[step_id - 1]
            template = str(row["question"])
            references = [int(value) for value in PLACEHOLDER_PATTERN.findall(template)]
            dependencies = [value for value in references if value in component]
            if not dependencies:
                prior = [ChainState((), {}, 1.0)]
            elif len(dependencies) == 1:
                prior = completed.get(dependencies[0], [])
            else:
                groups = [completed.get(parent, [])[:beam_width] for parent in dependencies]
                prior = (
                    _combine_parent_beams(groups, beam_width, multi_dependency_threshold)
                    if all(groups)
                    else []
                )

            final_hop = position == len(component) - 1
            expansions: list[ChainState] = []
            for state in prior:
                query = instantiate_question(template, state.answers_by_step)
                issued.append(query)
                candidates = retrieve_and_score(query, candidates_per_hop)
                if final_hop:
                    for candidate in candidates:
                        answers = dict(state.answers_by_step)
                        answers[step_id] = candidate.answer_names
                        steps = state.steps + (candidate,)
                        expansions.append(
                            ChainState(steps, answers, panini_chain_score(steps), candidate.score)
                        )
                elif unique_intermediate_entities:
                    best_by_entity: dict[str, ChainState] = {}
                    for candidate in candidates:
                        for answer_index, answer in enumerate(candidate.answer_names):
                            if not answer:
                                continue
                            entity_key = (
                                candidate.answer_ids[answer_index]
                                if answer_index < len(candidate.answer_ids)
                                and candidate.answer_ids[answer_index]
                                else normalize_entity_name(answer)
                            )
                            answers = dict(state.answers_by_step)
                            answers[step_id] = answer
                            steps = state.steps + (candidate,)
                            new_state = ChainState(
                                steps, answers, panini_chain_score(steps), candidate.score
                            )
                            previous = best_by_entity.get(entity_key)
                            if previous is None or _state_order(new_state) < _state_order(previous):
                                best_by_entity[entity_key] = new_state
                    expansions.extend(best_by_entity.values())
                else:
                    for candidate in candidates:
                        for answer in candidate.answer_names:
                            answers = dict(state.answers_by_step)
                            answers[step_id] = answer
                            steps = state.steps + (candidate,)
                            expansions.append(
                                ChainState(steps, answers, panini_chain_score(steps), candidate.score)
                            )
            completed[step_id] = sorted(expansions, key=_state_order)[:beam_width]
        final_states.extend(completed.get(component[-1], []))

    evidence: list[Candidate] = []
    seen: set[str] = set()
    for state in final_states:
        for candidate in state.steps:
            if candidate.qa_uid not in seen:
                seen.add(candidate.qa_uid)
                evidence.append(candidate)
    return RICRResult(
        tuple(tuple(component) for component in components),
        tuple(final_states),
        tuple(evidence),
        tuple(issued),
    )


def run_linear_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
) -> list[ChainState]:
    """Compatibility wrapper; the implementation still uses the DAG engine."""

    first = next(
        (str(row["question"]) for row in decomposed_questions if row.get("requires_retrieval", True)),
        "",
    )
    return list(
        run_panini_ricr(
            decomposed_questions,
            retrieve_and_score,
            original_question=first,
            beam_width=beam_width,
            candidates_per_hop=candidates_per_hop,
        ).chains
    )
