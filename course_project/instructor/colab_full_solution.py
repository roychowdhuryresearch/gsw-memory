"""Instructor reference pipeline used by the full Colab answer key.

The functions in this file deliberately keep every expensive stage restartable.
The generated notebook embeds this module so it remains self-contained after it
is uploaded to Colab; this source copy exists so the implementation can be
tested and maintained normally.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLACEHOLDER = re.compile(r"<ENTITY_Q(\d+)>")
WORD = re.compile(r"\w+")
FORBIDDEN_HELDOUT_FIELDS = {
    "answer",
    "answer_aliases",
    "supporting_facts",
    "evidences",
}


@dataclass(frozen=True)
class RunConfig:
    seed: int = 232
    beam_width: int = 5
    candidates_per_hop: int = 15
    retrieval_pool: int = 60
    entity_top_k: int = 20
    qa_top_k: int = 60
    rrf_constant: float = 60.0
    retrieval_weight: float = 0.5
    rerank_batch_size: int = 2
    rerank_max_length: int = 512
    free_colab_rerank_batch_size: int = 1
    free_colab_rerank_max_length: int = 256
    reranker_model: str = "Qwen/Qwen3-Reranker-8B"
    free_colab_reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    reranker_8b_minimum_gib: float = 18.0
    max_new_tokens: int = 768


def select_reranker_model(config: RunConfig, total_gib: float) -> str:
    """Select the 8B reference reranker or the T4-safe 4B fallback."""

    return (
        config.reranker_model
        if total_gib >= config.reranker_8b_minimum_gib
        else config.free_colab_reranker_model
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"[cache warning] ignored incomplete JSONL line {line_number} in {path}",
                    flush=True,
                )
    return records


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = path.exists() and path.stat().st_size > 0
    if needs_separator:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_separator = existing.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(value: object) -> str:
    return " ".join(WORD.findall(str(value).casefold()))


def release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda": True,
            "name": props.name,
            "total_gib": round(props.total_memory / 2**30, 3),
            "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 2**30, 3
            ),
        }
    except ImportError:
        return {"cuda": False}


def audit_package(package, dataset: str) -> tuple[list[dict], dict[str, Any]]:
    """Answer-key implementation for the Question 1 artifact audit."""

    import numpy as np

    public = package.questions("public")
    heldout = package.questions("held_out")
    entities = package.entities()
    qa_rows = package.qa_pairs()
    entity_ids = json.loads(
        (package.root / "embeddings/entity_ids.json").read_text()
    )
    qa_ids = json.loads((package.root / "embeddings/qa_ids.json").read_text())
    entity_matrix = np.load(
        package.root / "embeddings/entity_embeddings.npy", mmap_mode="r"
    )
    qa_matrix = np.load(
        package.root / "embeddings/qa_embeddings.npy", mmap_mode="r"
    )
    assert len(entity_ids) == len(set(entity_ids)) == entity_matrix.shape[0]
    assert len(qa_ids) == len(set(qa_ids)) == qa_matrix.shape[0]
    assert set(entity_ids) == {row["entity_uid"] for row in entities}
    assert set(qa_ids) == {row["qa_uid"] for row in qa_rows}
    assert all(
        not FORBIDDEN_HELDOUT_FIELDS.intersection(row) for row in heldout
    )
    document_ids = {row["document_id"] for row in package.documents()}
    gsw_paths = package.gsw_paths()
    validation_count = len(package.decompositions())
    rows = [
        {
            "dataset": dataset,
            "split": "development",
            "questions": len(public),
            "unique_documents": len(
                {doc for row in public for doc in row["context_document_ids"]}
            ),
            "gsw_files": len(gsw_paths),
            "entities": len(entities),
            "qa_records": len(qa_rows),
            "reviewed_decompositions": validation_count,
        },
        {
            "dataset": dataset,
            "split": "held_out",
            "questions": len(heldout),
            "unique_documents": len(
                {doc for row in heldout for doc in row["context_document_ids"]}
            ),
            "gsw_files": len(gsw_paths),
            "entities": len(entities),
            "qa_records": len(qa_rows),
            "reviewed_decompositions": validation_count,
        },
    ]
    examples = {
        "question": public[0],
        "entity": entities[0],
        "qa": qa_rows[0],
        "document_count": len(document_ids),
    }
    return rows, examples


STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "is", "of",
    "on", "or", "the", "to", "was", "with",
}
GENERIC_ATTRIBUTE_ROLES = {
    "achievement", "classification", "date", "date range", "ethnicity",
    "frequency", "genre", "language", "nationality", "number", "occupation",
    "profession", "quantity", "time", "time period", "title", "year",
}


def _role_parts(attributes: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    labels: set[str] = set()
    state_tokens: set[str] = set()
    for item in attributes.get("roles", []):
        if isinstance(item, Mapping):
            labels.add(normalize_text(item.get("role", "")))
            for state in item.get("states", []):
                state_tokens.update(WORD.findall(str(state).casefold()))
        else:
            labels.add(normalize_text(item))
    state_tokens.difference_update(STOPWORDS)
    return {value for value in labels if value}, state_tokens


def _neighbor_signature(native, node: str) -> set[str]:
    signatures: set[str] = set()
    for verb in native.predecessors(node):
        phrase = native.nodes[verb].get("phrase", "")
        signatures.update(WORD.findall(str(phrase).casefold()))
        for sibling in native.successors(verb):
            if sibling == node:
                continue
            name = native.nodes[sibling].get("name", "")
            signatures.update(WORD.findall(str(name).casefold()))
    signatures.difference_update(STOPWORDS)
    return signatures


def conservative_entity_mapping(native) -> tuple[dict[str, str], list[dict]]:
    """Return a conservative cross-document occurrence-to-identity map.

    Two occurrences are eligible only when canonical surfaces and node types
    match. Generic attribute values and numeric/date surfaces are not merged.
    A multi-token proper surface is joined when role labels overlap and neither
    occurrence is only a generic attribute; otherwise two content-bearing
    neighborhood tokens must overlap. Empty evidence never counts as
    agreement. Union-find makes transitive clusters, and every decision retains
    its evidence for the audit table.
    """

    from panini_course.graph import canonical_entity_name

    occurrences = [
        node
        for node, data in native.nodes(data=True)
        if data.get("node_type") != "verb_phrase"
    ]
    parent = {node: node for node in occurrences}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for node in occurrences:
        data = native.nodes[node]
        key = (
            canonical_entity_name(str(data.get("name", ""))),
            str(data.get("node_type", "unknown")),
        )
        if key[0]:
            grouped[key].append(node)

    decisions: list[dict] = []
    for (surface, node_type), nodes in sorted(grouped.items()):
        if len(nodes) < 2:
            continue
        roles = {node: _role_parts(native.nodes[node]) for node in nodes}
        neighborhoods = {node: _neighbor_signature(native, node) for node in nodes}
        for index, left in enumerate(nodes):
            for right in nodes[index + 1 :]:
                if native.nodes[left].get("document_id") == native.nodes[right].get(
                    "document_id"
                ):
                    continue
                left_labels, left_states = roles[left]
                right_labels, right_states = roles[right]
                state_union = left_states | right_states
                state_jaccard = (
                    len(left_states & right_states) / len(state_union)
                    if left_states and right_states and state_union
                    else 0.0
                )
                neighborhood_overlap = sorted(
                    neighborhoods[left] & neighborhoods[right]
                )
                surface_tokens = surface.split()
                numeric_surface = not any(character.isalpha() for character in surface)
                generic = (
                    bool(left_labels) and left_labels <= GENERIC_ATTRIBUTE_ROLES
                ) or (
                    bool(right_labels) and right_labels <= GENERIC_ATTRIBUTE_ROLES
                )
                compatible_roles = bool(left_labels & right_labels)
                proper_surface_evidence = (
                    len(surface_tokens) >= 2 and compatible_roles and not generic
                )
                neighborhood_evidence = len(neighborhood_overlap) >= 2
                accepted = (
                    not numeric_surface
                    and not generic
                    and (proper_surface_evidence or neighborhood_evidence)
                )
                if accepted:
                    union(left, right)
                decisions.append(
                    {
                        "surface": surface,
                        "node_type": node_type,
                        "left_uid": left,
                        "right_uid": right,
                        "left_document": native.nodes[left].get("document_id"),
                        "right_document": native.nodes[right].get("document_id"),
                        "left_roles": sorted(left_labels),
                        "right_roles": sorted(right_labels),
                        "state_jaccard": state_jaccard,
                        "neighborhood_overlap": neighborhood_overlap,
                        "accepted": accepted,
                    }
                )

    roots: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for node in sorted(occurrences):
        root = find(node)
        if root not in roots:
            roots[root] = len(roots)
        surface = canonical_entity_name(str(native.nodes[node].get("name", node)))
        mapping[node] = f"{surface}::c{roots[root]}"
    return mapping, decisions


def aggregate_projection(unreconciled, mapping: Mapping[str, str]):
    import networkx as nx

    result = nx.Graph()
    for node, data in unreconciled.nodes(data=True):
        target = mapping[node]
        if target not in result:
            result.add_node(
                target,
                name=data.get("name", node),
                node_type=data.get("node_type"),
                occurrences=0,
                documents=set(),
            )
        result.nodes[target]["occurrences"] += 1
        result.nodes[target]["documents"].add(data.get("document_id"))
    for left, right, data in unreconciled.edges(data=True):
        new_left, new_right = mapping[left], mapping[right]
        if new_left == new_right:
            continue
        weight = int(data.get("weight", 1))
        if result.has_edge(new_left, new_right):
            result[new_left][new_right]["weight"] += weight
        else:
            result.add_edge(new_left, new_right, weight=weight)
    return result


def network_statistics(name: str, graph) -> dict[str, Any]:
    import networkx as nx
    import numpy as np

    simple = nx.Graph(graph.to_undirected()) if graph.is_directed() else nx.Graph(graph)
    sizes = sorted((len(group) for group in nx.connected_components(simple)))
    n = simple.number_of_nodes()
    return {
        "graph": name,
        "nodes": n,
        "edges": simple.number_of_edges(),
        "components": len(sizes),
        "giant": sizes[-1] if sizes else 0,
        "giant_fraction": sizes[-1] / n if sizes and n else 0.0,
        "isolates": nx.number_of_isolates(simple),
        "component_min": sizes[0] if sizes else 0,
        "component_median": float(np.median(sizes)) if sizes else 0.0,
        "component_mean": float(np.mean(sizes)) if sizes else 0.0,
        "component_max": sizes[-1] if sizes else 0,
        "average_clustering": nx.average_clustering(simple) if n else 0.0,
        "degree_assortativity": (
            nx.degree_assortativity_coefficient(simple)
            if simple.number_of_edges() > 1
            else float("nan")
        ),
    }


def top_centralities(graph, top_n: int = 10, seed: int = 232) -> dict[str, list]:
    import networkx as nx

    simple = nx.Graph(graph)
    degree = dict(simple.degree(weight="weight"))
    pagerank = nx.pagerank(simple, weight="weight") if simple else {}
    sample = min(500, simple.number_of_nodes())
    between = (
        nx.betweenness_centrality(simple, k=sample, seed=seed, weight=None)
        if sample and simple.number_of_edges()
        else {}
    )

    def highest(values: Mapping[str, float]) -> list[tuple[str, float]]:
        return sorted(values.items(), key=lambda item: (-item[1], item[0]))[:top_n]

    return {
        "weighted_degree": highest(degree),
        "pagerank": highest(pagerank),
        "betweenness": highest(between),
    }


def validate_plan(plan: object) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(plan, list) or not plan:
        return {"valid": False, "errors": ["plan must be a nonempty list"], "edges": []}
    edges: list[tuple[int, int]] = []
    for step_number, row in enumerate(plan, start=1):
        if not isinstance(row, Mapping):
            errors.append(f"Q{step_number} is not an object")
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            errors.append(f"Q{step_number} has no question")
        for raw_parent in PLACEHOLDER.findall(question):
            parent = int(raw_parent)
            edges.append((parent, step_number))
            if parent >= step_number or parent < 1:
                errors.append(f"Q{step_number} has invalid reference Q{parent}")
    return {"valid": not errors, "errors": errors, "edges": sorted(set(edges))}


def decomposition_metrics(
    predicted: Mapping[str, Sequence[Mapping[str, Any]]],
    reviewed: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    valid, count_exact, true_flags, predicted_flags = [], [], [], []
    edge_tp = edge_fp = edge_fn = 0
    evaluated = 0
    for qid, gold in reviewed.items():
        if qid not in predicted:
            continue
        evaluated += 1
        plan = predicted[qid]
        valid.append(float(validate_plan(plan)["valid"]))
        count_exact.append(float(len(plan) == len(gold)))
        gold_edges = set(map(tuple, validate_plan(gold)["edges"]))
        plan_edges = set(map(tuple, validate_plan(plan)["edges"]))
        edge_tp += len(gold_edges & plan_edges)
        edge_fp += len(plan_edges - gold_edges)
        edge_fn += len(gold_edges - plan_edges)
        for left, right in zip(plan, gold):
            predicted_flags.append(bool(left.get("requires_retrieval", True)))
            true_flags.append(bool(right.get("requires_retrieval", True)))
    precision = edge_tp / (edge_tp + edge_fp) if edge_tp + edge_fp else 0.0
    recall = edge_tp / (edge_tp + edge_fn) if edge_tp + edge_fn else 0.0
    return {
        "questions": evaluated,
        "valid_plan_rate": sum(valid) / evaluated if evaluated else 0.0,
        "subquestion_count_exact": sum(count_exact) / evaluated if evaluated else 0.0,
        "dependency_edge_precision": precision,
        "dependency_edge_recall": recall,
        "dependency_edge_f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "retrieval_reasoning_accuracy": (
            sum(a == b for a, b in zip(predicted_flags, true_flags))
            / len(true_flags)
            if true_flags
            else 0.0
        ),
    }


def retrieval_branches(plan: Sequence[Mapping[str, Any]]) -> tuple[list[list[int]], list[str]]:
    """Extract single-parent retrieval chains and report unsupported joins."""

    retrieval = {
        index: row
        for index, row in enumerate(plan, start=1)
        if bool(row.get("requires_retrieval", True))
    }
    parent: dict[int, int | None] = {}
    warnings: list[str] = []
    for index, row in retrieval.items():
        refs = [int(value) for value in PLACEHOLDER.findall(str(row["question"]))]
        retrieval_refs = [value for value in refs if value in retrieval]
        if len(retrieval_refs) > 1:
            warnings.append(
                f"Q{index} is a multi-parent retrieval join; evidence is taken from its parent branches"
            )
            continue
        if refs and not retrieval_refs:
            warnings.append(
                f"Q{index} depends on a reasoning result and is not issued as retrieval"
            )
            continue
        parent[index] = retrieval_refs[0] if retrieval_refs else None
    children: dict[int, set[int]] = {index: set() for index in parent}
    for child, parent_id in parent.items():
        if parent_id in children:
            children[parent_id].add(child)
    leaves = [index for index in parent if not children[index]]
    branches: list[list[int]] = []
    for leaf in sorted(leaves):
        path, current, seen = [], leaf, set()
        while current is not None and current not in seen:
            seen.add(current)
            path.append(current)
            current = parent.get(current)
        path.reverse()
        if path and path not in branches:
            branches.append(path)
    return branches, warnings


def _chain_key(chain: Mapping[str, Any]) -> tuple:
    return (
        -float(chain["score"]),
        tuple(step["qa_uid"] for step in chain["steps"]),
    )


def prune_chains(
    chains: Sequence[Mapping[str, Any]],
    beam_width: int,
    *,
    unique_answers: bool = True,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chain in sorted(chains, key=_chain_key):
        answer_key = normalize_text(chain["steps"][-1]["answer"])
        if unique_answers and answer_key in seen:
            continue
        seen.add(answer_key)
        kept.append(dict(chain))
        if len(kept) == beam_width:
            break
    return kept


def pending_branch_queries(
    plan: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> list[str]:
    """Instantiate the next query once for every current parent beam."""

    step_ids = state["step_ids"]
    hop_index = int(state["hop_index"])
    if hop_index >= len(step_ids):
        return []
    if hop_index > 0 and not state["beams"]:
        return []
    step_id = step_ids[hop_index]
    template = str(plan[step_id - 1]["question"])
    parents = state["beams"] or [{"steps": [], "answers": {}, "score": 1.0}]
    queries: list[str] = []
    for parent in parents:
        answers = parent["answers"]

        def replace_placeholder(match: re.Match[str]) -> str:
            reference = int(match.group(1))
            if reference not in answers:
                raise KeyError(f"unresolved Q{reference} in {template}")
            return str(answers[reference])

        queries.append(PLACEHOLDER.sub(replace_placeholder, template))
    return queries


def advance_branch_one_hop(
    plan: Sequence[Mapping[str, Any]],
    state: dict[str, Any],
    retrieve,
    *,
    beam_width: int,
    candidates_per_hop: int,
    unique_answers: bool = True,
    score_rule: str = "geometric_mean",
) -> None:
    queries = pending_branch_queries(plan, state)
    if not queries:
        return
    step_id = state["step_ids"][state["hop_index"]]
    template = str(plan[step_id - 1]["question"])
    parents = state["beams"] or [{"steps": [], "answers": {}, "score": 1.0}]
    expansions: list[dict[str, Any]] = []
    for parent, concrete in zip(parents, queries):
        for candidate in retrieve(concrete, candidates_per_hop):
            steps = [*parent["steps"], candidate]
            scores = [max(float(item["score"]), 1e-12) for item in steps]
            score = (
                scores[-1]
                if score_rule == "last_hop"
                else math.exp(sum(math.log(value) for value in scores) / len(scores))
            )
            expansions.append(
                {
                    "steps": steps,
                    "answers": {**parent["answers"], step_id: candidate["answer"]},
                    "score": score,
                }
            )
    state["beams"] = prune_chains(
        expansions, beam_width, unique_answers=unique_answers
    )
    state["trace"].append(
        {
            "hop": state["hop_index"] + 1,
            "step_id": step_id,
            "template": template,
            "issued_queries": queries,
            "expansions": len(expansions),
            "kept": [
                {
                    "qa_ids": [step["qa_uid"] for step in beam["steps"]],
                    "answers": [step["answer"] for step in beam["steps"]],
                    "score": beam["score"],
                }
                for beam in state["beams"]
            ],
        }
    )
    state["hop_index"] += 1


def result_from_branch_states(
    branches: Sequence[dict[str, Any]], warnings: Sequence[str]
) -> dict[str, Any]:
    evidence: dict[str, dict[str, Any]] = {}
    for branch in branches:
        for beam in branch["beams"]:
            for step in beam["steps"]:
                previous = evidence.get(step["qa_uid"])
                if previous is None or step["score"] > previous["score"]:
                    evidence[step["qa_uid"]] = step
    return {
        "branches": [branch["step_ids"] for branch in branches],
        "warnings": list(warnings),
        "chains": [beam for branch in branches for beam in branch["beams"]],
        "branch_traces": [
            {"step_ids": branch["step_ids"], "hops": branch["trace"]}
            for branch in branches
        ],
        "evidence": list(evidence.values()),
    }


def run_branch(
    plan: Sequence[Mapping[str, Any]],
    step_ids: Sequence[int],
    retrieve,
    *,
    beam_width: int,
    candidates_per_hop: int,
    unique_answers: bool = True,
    score_rule: str = "geometric_mean",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    beams: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for hop, step_id in enumerate(step_ids, start=1):
        template = str(plan[step_id - 1]["question"])
        parents = beams if beams else [{"steps": [], "answers": {}, "score": 1.0}]
        expansions: list[dict[str, Any]] = []
        issued_queries: list[str] = []
        for parent in parents:
            answers = parent["answers"]

            def replace(match: re.Match[str]) -> str:
                referenced = int(match.group(1))
                if referenced not in answers:
                    raise KeyError(f"unresolved Q{referenced} in {template}")
                return str(answers[referenced])

            concrete = PLACEHOLDER.sub(replace, template)
            issued_queries.append(concrete)
            for candidate in retrieve(concrete, candidates_per_hop):
                steps = [*parent["steps"], candidate]
                scores = [max(float(item["score"]), 1e-12) for item in steps]
                if score_rule == "last_hop":
                    score = scores[-1]
                else:
                    score = math.exp(sum(math.log(value) for value in scores) / len(scores))
                expansions.append(
                    {
                        "steps": steps,
                        "answers": {**answers, step_id: candidate["answer"]},
                        "score": score,
                    }
                )
        beams = prune_chains(
            expansions, beam_width, unique_answers=unique_answers
        )
        trace.append(
            {
                "hop": hop,
                "step_id": step_id,
                "template": template,
                "issued_queries": issued_queries,
                "expansions": len(expansions),
                "kept": [
                    {
                        "qa_ids": [step["qa_uid"] for step in beam["steps"]],
                        "answers": [step["answer"] for step in beam["steps"]],
                        "score": beam["score"],
                    }
                    for beam in beams
                ],
            }
        )
        if not beams:
            break
    return beams, trace


def execute_plan(
    plan: Sequence[Mapping[str, Any]],
    retrieve,
    config: RunConfig,
    *,
    unique_answers: bool = True,
    score_rule: str = "geometric_mean",
) -> dict[str, Any]:
    branches, warnings = retrieval_branches(plan)
    all_beams, branch_traces = [], []
    for branch in branches:
        beams, trace = run_branch(
            plan,
            branch,
            retrieve,
            beam_width=config.beam_width,
            candidates_per_hop=config.candidates_per_hop,
            unique_answers=unique_answers,
            score_rule=score_rule,
        )
        all_beams.append(beams)
        branch_traces.append({"step_ids": branch, "hops": trace})

    # The final answerer receives the best chain from each independent branch.
    evidence: dict[str, dict[str, Any]] = {}
    for beams in all_beams:
        if not beams:
            continue
        for step in beams[0]["steps"]:
            previous = evidence.get(step["qa_uid"])
            if previous is None or step["score"] > previous["score"]:
                evidence[step["qa_uid"]] = step
    return {
        "branches": branches,
        "warnings": warnings,
        "chains": [beam for beams in all_beams for beam in beams],
        "branch_traces": branch_traces,
        "evidence": list(evidence.values()),
    }


def evidence_answers(question: Mapping[str, Any]) -> list[str]:
    answers: list[str] = []
    for evidence in question.get("evidences", []):
        if isinstance(evidence, Mapping):
            answers.append(str(evidence.get("answer", "")))
        elif isinstance(evidence, (list, tuple)) and evidence:
            answers.append(str(evidence[-1]))
    return [answer for answer in answers if answer]


def gold_qa_ids(question: Mapping[str, Any], qa_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    selected: set[str] = set()
    default_documents = set(
        question.get("supporting_document_ids", question.get("context_document_ids", []))
    )
    for task in atomic_tasks(question):
        documents = {task["document_id"]} if task["document_id"] else default_documents
        answer = normalize_text(task["answer"])
        candidates = [
            row
            for row in qa_rows
            if row.get("document_id") in documents
            and answer in {
                normalize_text(value) for value in row.get("answer_names", [])
            }
        ]
        if not candidates and documents != set(question.get("context_document_ids", [])):
            candidates = [
                row
                for row in qa_rows
                if row.get("document_id") in question.get("context_document_ids", [])
                and answer in {
                    normalize_text(value) for value in row.get("answer_names", [])
                }
            ]
        query_tokens = set(WORD.findall(task["question"].casefold())) - STOPWORDS

        def relevance(row: Mapping[str, Any]) -> tuple[float, str]:
            text = f"{row.get('question', '')} {row.get('verb_phrase', '')}"
            row_tokens = set(WORD.findall(text.casefold())) - STOPWORDS
            overlap = len(query_tokens & row_tokens) / max(len(query_tokens | row_tokens), 1)
            return overlap, str(row["qa_uid"])

        if candidates:
            best = sorted(candidates, key=lambda row: (-relevance(row)[0], relevance(row)[1]))[0]
            selected.add(str(best["qa_uid"]))
    return selected


def atomic_tasks(question: Mapping[str, Any]) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    previous: dict[int, str] = {}
    for index, evidence in enumerate(question.get("evidences", []), start=1):
        if isinstance(evidence, Mapping):
            query = str(evidence.get("question", ""))
            query = re.sub(
                r"#(\d+)",
                lambda match: previous.get(int(match.group(1)), match.group(0)),
                query,
            )
            answer = str(evidence.get("answer", ""))
            document_id = str(evidence.get("document_id", ""))
        else:
            subject, relation, answer = map(str, evidence)
            query = f"{subject} >> {relation}"
            document_id = ""
        previous[index] = answer
        tasks.append(
            {"question": query, "answer": answer, "document_id": document_id}
        )
    return tasks


def stratified_sample(
    questions: Sequence[Mapping[str, Any]],
    group_field: str,
    size: int = 20,
    seed: int = 232,
) -> list[Mapping[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in questions:
        groups[str(row[group_field])].append(row)
    rng = random.Random(seed)
    selected: list[Mapping[str, Any]] = []
    ordered_groups = sorted(groups)
    base, remainder = divmod(size, len(ordered_groups))
    for index, group in enumerate(ordered_groups):
        rows = sorted(groups[group], key=lambda row: str(row["question_id"]))
        rng.shuffle(rows)
        selected.extend(rows[: base + (index < remainder)])
    return sorted(selected, key=lambda row: str(row["question_id"]))


def evaluate_ranked_ids(ranked_ids: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    hits = [index for index, qa_uid in enumerate(ranked_ids[:k], start=1) if qa_uid in relevant]
    return {
        "recall": len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.0,
        "reciprocal_rank": 1.0 / hits[0] if hits else 0.0,
    }


def run_decomposition_stage(
    jobs: Sequence[tuple[str, Any, Sequence[Mapping[str, Any]], Path]],
    config: RunConfig,
    *,
    model_name: str = "yigitturali/GSW-QA-Decomposer-Qwen3-4B",
) -> None:
    """Run all datasets with one 4-bit decomposer and append after each question."""

    import torch
    from panini_course.qwen_models import QwenDecomposer

    if not torch.cuda.is_available():
        raise RuntimeError("The full decomposition stage requires a GPU runtime")
    dtype = "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
    prompt_path = jobs[0][1].root / "models/decomposition_prompt.txt"
    model = QwenDecomposer(
        model_name,
        prompt_path,
        quantized=True,
        dtype=dtype,
        device_map="auto",
    )
    for dataset, _package, questions, cache_root in jobs:
        path = cache_root / "decompositions.jsonl"
        completed = {row["question_id"] for row in read_jsonl(path)}
        for question in questions:
            qid = str(question["question_id"])
            if qid in completed:
                continue
            started = time.perf_counter()
            raw = ""
            try:
                raw = model.generate_raw(
                    str(question["question"]),
                    max_new_tokens=config.max_new_tokens,
                )
                plan = model.parse_response(raw)
                validation = validate_plan(plan)
                error = None
            except Exception as exception:  # retained in the trace by design
                plan, validation, error = [], {"valid": False, "errors": []}, repr(exception)
            append_jsonl(
                path,
                {
                    "dataset": dataset,
                    "question_id": qid,
                    "question": question["question"],
                    "raw_response": raw,
                    "predicted_decomposition": plan,
                    "decomposition_valid": validation["valid"],
                    "decomposition_errors": validation["errors"],
                    "error": error,
                    "seconds": time.perf_counter() - started,
                },
            )
            print(
                f"[decompose] {dataset} {len(read_jsonl(path))}/100 {qid}",
                flush=True,
            )
    del model
    release_gpu()


class NeuralRetrievalCache:
    """Persistent exact-query cache for all Q5--Q7 ranking variants."""

    def __init__(self, package, cache_root: Path, config: RunConfig, encoder, reranker):
        import numpy as np
        from panini_course import (
            BM25Index,
            DenseIndex,
            DualRetriever,
            QueryEmbeddingStore,
            TfidfIndex,
        )

        self.np = np
        self.package = package
        self.cache_root = Path(cache_root)
        self.config = config
        self.encoder = encoder
        self.reranker = reranker
        self.qa_rows = package.qa_pairs()
        self.qa_by_id = {row["qa_uid"]: row for row in self.qa_rows}
        self.entity_rows = package.entities()
        root = package.root
        self.entity_bm25 = BM25Index.load(
            root / "indices/entity_bm25.joblib",
            root / "indices/entity_ids.json",
            source="entity_bm25",
        )
        self.qa_bm25 = BM25Index.load(
            root / "indices/qa_bm25.joblib",
            root / "indices/qa_ids.json",
            source="qa_bm25",
        )
        self.qa_tfidf = TfidfIndex.load(
            root / "indices/qa_tfidf.npz",
            root / "indices/qa_tfidf_vectorizer.joblib",
            root / "indices/qa_ids.json",
            source="qa_tfidf",
        )
        self.qa_dense = DenseIndex.load(
            root / "indices/qa_qwen3_8b_ip.faiss",
            root / "indices/qa_ids.json",
            source="qa_dense",
        )
        self.query_store = QueryEmbeddingStore.load(
            root / "embeddings/query_embeddings.npy",
            root / "embeddings/query_ids.json",
            root / "embeddings/queries.jsonl",
        )
        self.dual = DualRetriever(
            entity_index=self.entity_bm25,
            qa_index=self.qa_dense,
            entity_rows=self.entity_rows,
            qa_rows=self.qa_rows,
        )
        self.query_path = self.cache_root / "retrieval_cache.jsonl"
        self.records = {
            row["query_key"]: row for row in read_jsonl(self.query_path)
        }
        self.vector_manifest_path = self.cache_root / "query_vector_manifest.jsonl"
        self.vector_records = {
            row["query_key"]: row for row in read_jsonl(self.vector_manifest_path)
        }
        self.vector_memory: dict[str, Any] = {}

    def vector(self, query: str):
        key = normalize_text(query)
        if key in self.vector_memory:
            return self.vector_memory[key]
        try:
            vector = self.query_store.get(query)
        except KeyError:
            if key in self.vector_records:
                vector = self.np.load(
                    self.cache_root / self.vector_records[key]["file"]
                )
            else:
                if self.encoder is None:
                    raise RuntimeError(
                        f"Query vector is not prepared for: {query}"
                    )
                started = time.perf_counter()
                vector = self.encoder.encode([query], max_length=256)[0]
                relative = f"query_vectors/query_{len(self.vector_records):06d}.npy"
                target = self.cache_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                self.np.save(target, vector)
                record = {
                    "query_key": key,
                    "query": query,
                    "file": relative,
                    "seconds": time.perf_counter() - started,
                }
                append_jsonl(self.vector_manifest_path, record)
                self.vector_records[key] = record
        self.vector_memory[key] = vector
        return vector

    def _expand_answers(self, hits, score_name: str = "score") -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for hit in hits:
            row = self.qa_by_id[hit.item_id]
            for answer in row.get("answer_names", []):
                candidates.append(
                    {
                        "qa_uid": row["qa_uid"],
                        "answer": answer,
                        "question": row["question"],
                        "document_id": row["document_id"],
                        "score": float(getattr(hit, score_name, hit.score)),
                        "rank": int(hit.rank),
                        "source": hit.source,
                    }
                )
        return candidates

    def compute(self, query: str) -> dict[str, Any]:
        from panini_course import reciprocal_rank_fusion

        key = normalize_text(query)
        if key in self.records:
            return self.records[key]
        if self.reranker is None:
            raise RuntimeError(f"Reranker result is not prepared for: {query}")
        started = time.perf_counter()
        pool = self.config.retrieval_pool
        bm25 = self.qa_bm25.search(query, pool)
        tfidf = self.qa_tfidf.search(query, pool)
        dense = self.qa_dense.search(self.vector(query), pool)
        rrf = reciprocal_rank_fusion(
            [tfidf, bm25, dense],
            rank_constant=self.config.rrf_constant,
            top_k=pool,
        )
        dual = self.dual.search(
            query,
            query_vector=self.vector(query),
            entity_top_k=self.config.entity_top_k,
            qa_top_k=self.config.qa_top_k,
            fused_top_k=pool,
            rank_constant=self.config.rrf_constant,
        )
        rows = [self.qa_by_id[hit.item_id] for hit in dual]
        rerank_scores = self.reranker.score(
            query,
            [row["search_text"] for row in rows],
            batch_size=self.config.rerank_batch_size,
        )
        reranked: dict[str, list[dict[str, Any]]] = {
            "reranker_only": [],
            "retrieval_only": [],
            "dual_hybrid": [],
        }
        for hit, row, reranker_score in zip(dual, rows, rerank_scores):
            retrieval_score = 1.0 / hit.rank
            hybrid = (
                self.config.retrieval_weight * retrieval_score
                + (1.0 - self.config.retrieval_weight) * float(reranker_score)
            )
            base = {
                "qa_uid": row["qa_uid"],
                "question": row["question"],
                "document_id": row["document_id"],
                "retrieval_rank": hit.rank,
                "reranker_score": float(reranker_score),
                "routes": list(hit.metadata.get("fusion_sources", [])),
                "source": "dual",
            }
            for answer in row.get("answer_names", []):
                reranked["reranker_only"].append(
                    {**base, "answer": answer, "score": float(reranker_score)}
                )
                reranked["retrieval_only"].append(
                    {**base, "answer": answer, "score": retrieval_score}
                )
                reranked["dual_hybrid"].append(
                    {**base, "answer": answer, "score": hybrid}
                )
        for name in reranked:
            reranked[name].sort(
                key=lambda row: (-row["score"], row["qa_uid"], row["answer"])
            )
        rankings = {
            "bm25": self._expand_answers(bm25),
            "dense": self._expand_answers(dense),
            "rrf": self._expand_answers(rrf),
            **reranked,
        }
        record = {
            "query_key": key,
            "query": query,
            "rankings": rankings,
            "seconds": time.perf_counter() - started,
        }
        append_jsonl(self.query_path, record)
        self.records[key] = record
        return record

    def candidates(self, query: str, top_k: int, backend: str = "dual_hybrid"):
        return self.compute(query)["rankings"][backend][:top_k]


def run_neural_retrieval_stage(
    jobs: Sequence[tuple[str, Any, Sequence[Mapping[str, Any]], Path]],
    config: RunConfig,
    *,
    backend: str = "dual_hybrid",
    warm_gold_atomic_queries: bool = True,
    run_ablations: bool = True,
) -> None:
    """Run Qwen query encoding, reranking, and RICR with one model pair."""

    import torch
    from panini_course.qwen_models import QwenQueryEncoder, QwenReranker

    if not torch.cuda.is_available():
        raise RuntimeError("The full retrieval stage requires a GPU runtime")
    if torch.cuda.get_device_properties(0).total_memory < 14 * 2**30:
        raise RuntimeError("The 4-bit encoder/reranker stage needs a 15–16 GiB GPU")
    dtype = "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
    encoder = QwenQueryEncoder(quantized=True, dtype=dtype, device_map="auto")
    reranker = QwenReranker(
        model_name=config.reranker_model,
        quantized=True,
        dtype=dtype,
        device_map="auto",
        max_length=config.rerank_max_length,
    )
    for dataset, package, questions, cache_root in jobs:
        plans = {
            row["question_id"]: row for row in read_jsonl(cache_root / "decompositions.jsonl")
        }
        trace_path = cache_root / "traces.jsonl"
        completed = {row["question_id"] for row in read_jsonl(trace_path)}
        retrieval = NeuralRetrievalCache(package, cache_root, config, encoder, reranker)
        if warm_gold_atomic_queries:
            for public_question in package.questions("public"):
                for task in atomic_tasks(public_question):
                    retrieval.compute(task["question"])
        for question in questions:
            qid = str(question["question_id"])
            if qid in completed:
                continue
            plan_record = plans.get(qid)
            if not plan_record or not plan_record.get("decomposition_valid"):
                append_jsonl(
                    trace_path,
                    {
                        "dataset": dataset,
                        "question_id": qid,
                        "question": question["question"],
                        "error": "missing or invalid decomposition",
                        "chains": [],
                        "evidence": [],
                        "seconds": 0.0,
                    },
                )
                continue
            started = time.perf_counter()
            try:
                result = execute_plan(
                    plan_record["predicted_decomposition"],
                    lambda query, k: retrieval.candidates(query, k, backend),
                    config,
                )
                error = None
            except Exception as exception:
                result = {"branches": [], "warnings": [], "chains": [], "evidence": []}
                error = repr(exception)
            append_jsonl(
                trace_path,
                {
                    "dataset": dataset,
                    "question_id": qid,
                    "question": question["question"],
                    "predicted_decomposition": plan_record["predicted_decomposition"],
                    "reranker_model": config.reranker_model,
                    **result,
                    "error": error,
                    "seconds": time.perf_counter() - started,
                },
            )
            print(
                f"[retrieve] {dataset} {len(read_jsonl(trace_path))}/100 {qid}",
                flush=True,
            )
        if run_ablations:
            group_field = "type" if dataset == "2wiki" else "hop_count"
            subset = stratified_sample(
                package.questions("public"), group_field, size=20, seed=config.seed
            )
            ablation_path = cache_root / "ablation_traces.jsonl"
            completed_ablations = {
                (row["configuration"], row["question_id"])
                for row in read_jsonl(ablation_path)
            }
            ablations = [
                ("default", config, "dual_hybrid", True, "geometric_mean"),
                ("beam_1", replace(config, beam_width=1), "dual_hybrid", True, "geometric_mean"),
                ("beam_3", replace(config, beam_width=3), "dual_hybrid", True, "geometric_mean"),
                ("k_5", replace(config, candidates_per_hop=5), "dual_hybrid", True, "geometric_mean"),
                ("unique_off", config, "dual_hybrid", False, "geometric_mean"),
                ("last_hop", config, "dual_hybrid", True, "last_hop"),
                ("bm25", config, "bm25", True, "geometric_mean"),
                ("dense", config, "dense", True, "geometric_mean"),
                ("rrf", config, "rrf", True, "geometric_mean"),
            ]
            qa_rows = package.qa_pairs()
            for name, ablation_config, ablation_backend, unique, score_rule in ablations:
                for question in subset:
                    qid = str(question["question_id"])
                    if (name, qid) in completed_ablations:
                        continue
                    plan_record = plans.get(qid)
                    if not plan_record or not plan_record.get("decomposition_valid"):
                        continue
                    started = time.perf_counter()
                    result = execute_plan(
                        plan_record["predicted_decomposition"],
                        lambda query, k, selected=ablation_backend: retrieval.candidates(
                            query, k, selected
                        ),
                        ablation_config,
                        unique_answers=unique,
                        score_rule=score_rule,
                    )
                    append_jsonl(
                        ablation_path,
                        {
                            "dataset": dataset,
                            "configuration": name,
                            "question_id": qid,
                            "question": question["question"],
                            "backend": ablation_backend,
                            "reranker_model": config.reranker_model,
                            "beam_width": ablation_config.beam_width,
                            "candidates_per_hop": ablation_config.candidates_per_hop,
                            "unique_answers": unique,
                            "score_rule": score_rule,
                            **result,
                            **score_trace(question, result, qa_rows),
                            "seconds": time.perf_counter() - started,
                        },
                    )
                print(f"[ablation] {dataset} {name} complete", flush=True)
    del encoder, reranker
    release_gpu()


def _scheduled_retrieval_configuration(
    jobs: Sequence[tuple[str, Any, Sequence[Mapping[str, Any]], Path]],
    config: RunConfig,
    *,
    output_name: str,
    configuration_name: str,
    backend: str,
    unique_answers: bool,
    score_rule: str,
    warm_gold_atomic_queries: bool = False,
) -> None:
    """Execute one configuration without co-resident encoder and reranker."""

    import torch
    from panini_course.qwen_models import QwenQueryEncoder, QwenReranker

    if not torch.cuda.is_available():
        raise RuntimeError("The full retrieval stage requires a GPU runtime")
    dtype = "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    reranker_model = select_reranker_model(config, total_gib)
    if reranker_model == config.free_colab_reranker_model:
        config = replace(
            config,
            rerank_batch_size=config.free_colab_rerank_batch_size,
            rerank_max_length=config.free_colab_rerank_max_length,
        )
    print(
        f"[low-memory] {total_gib:.1f} GiB GPU; reranker={reranker_model}",
        flush=True,
    )
    contexts: list[dict[str, Any]] = []
    for dataset, package, questions, cache_root in jobs:
        output_path = cache_root / output_name
        existing = read_jsonl(output_path)
        completed = {
            row["question_id"]
            for row in existing
            if row.get("configuration", configuration_name) == configuration_name
        }
        plans = {
            row["question_id"]: row
            for row in read_jsonl(cache_root / "decompositions.jsonl")
        }
        states: dict[str, dict[str, Any]] = {}
        for question in questions:
            qid = str(question["question_id"])
            if qid in completed:
                continue
            plan_record = plans.get(qid)
            if not plan_record or not plan_record.get("decomposition_valid"):
                append_jsonl(
                    output_path,
                    {
                        "dataset": dataset,
                        "configuration": configuration_name,
                        "question_id": qid,
                        "question": question["question"],
                        "error": "missing or invalid decomposition",
                        "chains": [],
                        "evidence": [],
                        "seconds": 0.0,
                    },
                )
                continue
            plan = plan_record["predicted_decomposition"]
            branch_ids, warnings = retrieval_branches(plan)
            states[qid] = {
                "question": question,
                "plan": plan,
                "warnings": warnings,
                "branches": [
                    {"step_ids": ids, "hop_index": 0, "beams": [], "trace": []}
                    for ids in branch_ids
                ],
                "started": time.perf_counter(),
            }
        contexts.append(
            {
                "dataset": dataset,
                "package": package,
                "cache_root": cache_root,
                "output_path": output_path,
                "states": states,
                "qa_rows": package.qa_pairs(),
                "wrote": set(),
            }
        )

    round_number = 0
    while True:
        round_number += 1
        queries_by_dataset: dict[str, list[str]] = {}
        any_pending = False
        for context in contexts:
            queries: list[str] = []
            if warm_gold_atomic_queries and round_number == 1:
                for question in context["package"].questions("public"):
                    queries.extend(task["question"] for task in atomic_tasks(question))
            for state in context["states"].values():
                for branch in state["branches"]:
                    pending = pending_branch_queries(state["plan"], branch)
                    queries.extend(pending)
                    any_pending = any_pending or bool(pending)
            queries_by_dataset[context["dataset"]] = list(dict.fromkeys(queries))
        if not any_pending and not (
            warm_gold_atomic_queries and round_number == 1
        ):
            break

        missing_vectors: dict[str, list[str]] = {}
        for context in contexts:
            cache = NeuralRetrievalCache(
                context["package"], context["cache_root"], config, None, None
            )
            missing: list[str] = []
            for query in queries_by_dataset[context["dataset"]]:
                try:
                    cache.vector(query)
                except RuntimeError:
                    missing.append(query)
            missing_vectors[context["dataset"]] = missing

        vector_misses = sum(map(len, missing_vectors.values()))
        print(
            f"[low-memory round {round_number}] query-vector misses: {vector_misses}",
            flush=True,
        )
        if vector_misses:
            encoder = QwenQueryEncoder(
                quantized=True, dtype=dtype, device_map="auto"
            )
            for context in contexts:
                cache = NeuralRetrievalCache(
                    context["package"], context["cache_root"], config, encoder, None
                )
                for query in missing_vectors[context["dataset"]]:
                    cache.vector(query)
            # The last cache owns an encoder reference; release it before the
            # CUDA cleanup or the next model can inherit the old reservation.
            del cache, encoder
            release_gpu()

        ranking_misses: dict[str, list[str]] = {}
        for context in contexts:
            cache = NeuralRetrievalCache(
                context["package"], context["cache_root"], config, None, None
            )
            ranking_misses[context["dataset"]] = [
                query
                for query in queries_by_dataset[context["dataset"]]
                if normalize_text(query) not in cache.records
            ]
        rerank_misses = sum(map(len, ranking_misses.values()))
        print(
            f"[low-memory round {round_number}] reranking misses: {rerank_misses}",
            flush=True,
        )
        reranker = None
        if rerank_misses:
            reranker = QwenReranker(
                model_name=reranker_model,
                quantized=True,
                dtype=dtype,
                device_map="auto",
                max_length=config.rerank_max_length,
            )
        caches: dict[str, NeuralRetrievalCache] = {}
        for context in contexts:
            cache = NeuralRetrievalCache(
                context["package"], context["cache_root"], config, None, reranker
            )
            caches[context["dataset"]] = cache
            for query in ranking_misses[context["dataset"]]:
                cache.compute(query)

        for context in contexts:
            cache = caches[context["dataset"]]
            for qid, state in context["states"].items():
                if qid in context["wrote"]:
                    continue
                for branch in state["branches"]:
                    if pending_branch_queries(state["plan"], branch):
                        advance_branch_one_hop(
                            state["plan"],
                            branch,
                            lambda query, k, selected=backend: cache.candidates(
                                query, k, selected
                            ),
                            beam_width=config.beam_width,
                            candidates_per_hop=config.candidates_per_hop,
                            unique_answers=unique_answers,
                            score_rule=score_rule,
                        )
                finished = all(
                    branch["hop_index"] >= len(branch["step_ids"])
                    or (branch["hop_index"] > 0 and not branch["beams"])
                    for branch in state["branches"]
                )
                if finished:
                    result = result_from_branch_states(
                        state["branches"], state["warnings"]
                    )
                    issued_queries = {
                        query
                        for branch in result["branch_traces"]
                        for hop in branch["hops"]
                        for query in hop["issued_queries"]
                    }
                    query_seconds = 0.0
                    for query in issued_queries:
                        key = normalize_text(query)
                        query_seconds += float(
                            cache.records.get(key, {}).get("seconds", 0.0)
                        )
                        query_seconds += float(
                            cache.vector_records.get(key, {}).get("seconds", 0.0)
                        )
                    record = {
                        "dataset": context["dataset"],
                        "configuration": configuration_name,
                        "question_id": qid,
                        "question": state["question"]["question"],
                        "predicted_decomposition": state["plan"],
                        "backend": backend,
                        "reranker_model": reranker_model,
                        "beam_width": config.beam_width,
                        "candidates_per_hop": config.candidates_per_hop,
                        "unique_answers": unique_answers,
                        "score_rule": score_rule,
                        **result,
                        "error": None,
                        "seconds": query_seconds,
                        "scheduled_wall_seconds": time.perf_counter() - state["started"],
                    }
                    if output_name == "ablation_traces.jsonl":
                        record.update(
                            score_trace(
                                state["question"], result, context["qa_rows"]
                            )
                        )
                    append_jsonl(context["output_path"], record)
                    context["wrote"].add(qid)
        del caches, cache
        if reranker is not None:
            del reranker
            release_gpu()


def run_neural_retrieval_stage_low_memory(
    jobs: Sequence[tuple[str, Any, Sequence[Mapping[str, Any]], Path]],
    config: RunConfig,
    *,
    run_ablations: bool = True,
) -> None:
    """Free-tier path: alternate encoder and reranker by RICR depth round."""

    _scheduled_retrieval_configuration(
        jobs,
        config,
        output_name="traces.jsonl",
        configuration_name="default",
        backend="dual_hybrid",
        unique_answers=True,
        score_rule="geometric_mean",
        warm_gold_atomic_queries=True,
    )
    if not run_ablations:
        return
    ablations = [
        ("default", config, "dual_hybrid", True, "geometric_mean"),
        ("beam_1", replace(config, beam_width=1), "dual_hybrid", True, "geometric_mean"),
        ("beam_3", replace(config, beam_width=3), "dual_hybrid", True, "geometric_mean"),
        ("k_5", replace(config, candidates_per_hop=5), "dual_hybrid", True, "geometric_mean"),
        ("unique_off", config, "dual_hybrid", False, "geometric_mean"),
        ("last_hop", config, "dual_hybrid", True, "last_hop"),
        ("bm25", config, "bm25", True, "geometric_mean"),
        ("dense", config, "dense", True, "geometric_mean"),
        ("rrf", config, "rrf", True, "geometric_mean"),
    ]
    for name, selected_config, backend, unique, score_rule in ablations:
        subset_jobs = []
        for dataset, package, _questions, cache_root in jobs:
            group_field = "type" if dataset == "2wiki" else "hop_count"
            subset = stratified_sample(
                package.questions("public"),
                group_field,
                size=20,
                seed=config.seed,
            )
            subset_jobs.append((dataset, package, subset, cache_root))
        _scheduled_retrieval_configuration(
            subset_jobs,
            selected_config,
            output_name="ablation_traces.jsonl",
            configuration_name=name,
            backend=backend,
            unique_answers=unique,
            score_rule=score_rule,
        )


def score_trace(question: Mapping[str, Any], trace: Mapping[str, Any], qa_rows) -> dict[str, Any]:
    from panini_course.metrics import exact_match

    selected_ids = {row["qa_uid"] for row in trace.get("evidence", [])}
    all_chain_steps = [
        step for chain in trace.get("chains", []) for step in chain.get("steps", [])
    ]
    all_chain_ids = {step["qa_uid"] for step in all_chain_steps}
    gold_answer_sequence = [normalize_text(answer) for answer in evidence_answers(question)]
    final_beams_by_branch = [
        branch.get("hops", [])[-1].get("kept", [])
        for branch in trace.get("branch_traces", [])
        if branch.get("hops")
    ]
    if len(final_beams_by_branch) == 1:
        complete_chain = any(
            all(answer in [normalize_text(value) for value in beam.get("answers", [])]
                for answer in gold_answer_sequence)
            for beam in final_beams_by_branch[0]
        )
    else:
        retained_answers = {
            normalize_text(answer)
            for branch in final_beams_by_branch
            for beam in branch
            for answer in beam.get("answers", [])
        }
        complete_chain = all(answer in retained_answers for answer in gold_answer_sequence)
    gold_ids = gold_qa_ids(question, qa_rows)
    document_by_qa = {str(row["qa_uid"]): row.get("document_id") for row in qa_rows}
    support_docs = set(question.get("supporting_document_ids", [])) or {
        document_by_qa[qa_uid]
        for qa_uid in gold_ids
        if document_by_qa.get(qa_uid)
    }
    selected_docs = {row.get("document_id") for row in trace.get("evidence", [])}
    gold_answers = [question["answer"], *question.get("answer_aliases", [])]
    return {
        "supporting_qa_recall": len(all_chain_ids & gold_ids) / len(gold_ids) if gold_ids else 0.0,
        "supporting_document_recall": (
            len(selected_docs & support_docs) / len(support_docs) if support_docs else 0.0
        ),
        "complete_chain_recovery": float(complete_chain),
        "answer_in_selected_evidence": max(
            (exact_match(row.get("answer", ""), gold_answers) for row in trace.get("evidence", [])),
            default=0.0,
        ),
        "selected_evidence_count": len(selected_ids),
        "surviving_chains": len(trace.get("chains", [])),
        "unique_current_answers": len(
            {
                normalize_text(chain["steps"][-1]["answer"])
                for chain in trace.get("chains", [])
                if chain.get("steps")
            }
        ),
    }


def run_answer_stage(
    jobs: Sequence[tuple[str, Any, Sequence[Mapping[str, Any]], Path]],
    output_root: Path,
    config: RunConfig,
    *,
    run_ablations: bool = True,
) -> None:
    import torch
    from panini_course.metrics import exact_match, token_f1
    from panini_course.qwen_models import QwenAnswerer

    if not torch.cuda.is_available():
        raise RuntimeError("The full answer stage requires a GPU runtime")
    dtype = "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
    answerer = QwenAnswerer(quantized=True, dtype=dtype, device_map="auto")
    for dataset, package, questions, cache_root in jobs:
        plans = {
            row["question_id"]: row for row in read_jsonl(cache_root / "decompositions.jsonl")
        }
        traces = {row["question_id"]: row for row in read_jsonl(cache_root / "traces.jsonl")}
        prediction_path = cache_root / "answers.jsonl"
        completed = {row["question_id"] for row in read_jsonl(prediction_path)}
        qa_rows = package.qa_pairs()
        public_ids = {row["question_id"] for row in package.questions("public")}
        for question in questions:
            qid = str(question["question_id"])
            if qid in completed:
                continue
            trace = traces.get(qid, {"chains": [], "evidence": [], "seconds": 0.0})
            evidence = [
                f"Q: {row['question']} A: {row['answer']}"
                for row in trace.get("evidence", [])
            ]
            started = time.perf_counter()
            prediction = answerer.answer(str(question["question"]), evidence)
            answer_seconds = time.perf_counter() - started
            context = "\n".join(evidence)
            context_tokens = len(answerer.tokenizer(context).input_ids)
            plan_record = plans.get(qid, {})
            issued_queries = {
                query
                for branch in trace.get("branch_traces", [])
                for hop in branch.get("hops", [])
                for query in hop.get("issued_queries", [])
            }
            record: dict[str, Any] = {
                "dataset": dataset,
                "split": "development" if qid in public_ids else "held_out",
                "question_id": qid,
                "question": question["question"],
                "predicted_decomposition": plan_record.get("predicted_decomposition", []),
                "decomposition_valid": bool(plan_record.get("decomposition_valid", False)),
                "retrieval_backend": "dual_hybrid",
                "reranker_model": trace.get(
                    "reranker_model", config.reranker_model
                ),
                "beam_width": config.beam_width,
                "candidates_per_hop": config.candidates_per_hop,
                "chains": [
                    {
                        "qa_ids": [step["qa_uid"] for step in chain.get("steps", [])],
                        "answers": [step["answer"] for step in chain.get("steps", [])],
                        "hop_scores": [step["score"] for step in chain.get("steps", [])],
                        "chain_score": chain.get("score", 0.0),
                    }
                    for chain in trace.get("chains", [])
                ],
                "evidence_qa_ids": [row["qa_uid"] for row in trace.get("evidence", [])],
                "predicted_answer": prediction,
                "latency_ms": {
                    "decomposition": 1000 * float(plan_record.get("seconds", 0.0)),
                    "retrieval_ricr": 1000 * float(trace.get("seconds", 0.0)),
                    "answer": 1000 * answer_seconds,
                },
                "answer_context_tokens": context_tokens,
                "evidence_count": len(evidence),
                "reranked_candidate_count": len(issued_queries) * config.retrieval_pool,
            }
            if qid in public_ids:
                aliases = [question["answer"], *question.get("answer_aliases", [])]
                record.update(score_trace(question, trace, qa_rows))
                record.update(
                    {
                        "gold_answer": question["answer"],
                        "exact_match": exact_match(prediction, aliases),
                        "token_f1": token_f1(prediction, aliases),
                    }
                )
            append_jsonl(prediction_path, record)
            print(
                f"[answer] {dataset} {len(read_jsonl(prediction_path))}/100 {qid}",
                flush=True,
            )

        predictions = read_jsonl(prediction_path)
        for row in predictions:
            row.setdefault("reranker_model", config.reranker_model)
        write_jsonl(prediction_path, predictions)
        write_jsonl(
            output_root / "results" / f"{dataset}_dev.jsonl",
            (row for row in predictions if row["split"] == "development"),
        )
        write_jsonl(
            output_root / "predictions" / f"{dataset}_heldout.jsonl",
            (row for row in predictions if row["split"] == "held_out"),
        )
        if run_ablations:
            ablation_rows = read_jsonl(cache_root / "ablation_traces.jsonl")
            ablation_answer_path = cache_root / "ablation_answers.jsonl"
            completed_ablations = {
                (row["configuration"], row["question_id"])
                for row in read_jsonl(ablation_answer_path)
            }
            question_by_id = {
                row["question_id"]: row for row in package.questions("public")
            }
            for row in ablation_rows:
                key = (row["configuration"], row["question_id"])
                if key in completed_ablations:
                    continue
                evidence = [
                    f"Q: {item['question']} A: {item['answer']}"
                    for item in row.get("evidence", [])
                ]
                started = time.perf_counter()
                prediction = answerer.answer(row["question"], evidence)
                question = question_by_id[row["question_id"]]
                aliases = [question["answer"], *question.get("answer_aliases", [])]
                refreshed_metrics = score_trace(question, row, qa_rows)
                append_jsonl(
                    ablation_answer_path,
                    {
                        "dataset": dataset,
                        "configuration": row["configuration"],
                        "question_id": row["question_id"],
                        "reranker_model": row.get(
                            "reranker_model", config.reranker_model
                        ),
                        "predicted_answer": prediction,
                        "exact_match": exact_match(prediction, aliases),
                        "token_f1": token_f1(prediction, aliases),
                        "answer_seconds": time.perf_counter() - started,
                        "supporting_qa_recall": refreshed_metrics["supporting_qa_recall"],
                        "complete_chain_recovery": refreshed_metrics["complete_chain_recovery"],
                        "retrieval_seconds": row["seconds"],
                        "evidence_count": len(evidence),
                    },
                )
            ablation_predictions = read_jsonl(ablation_answer_path)
            for row in ablation_predictions:
                row.setdefault("reranker_model", config.reranker_model)
            write_jsonl(ablation_answer_path, ablation_predictions)
    del answerer
    release_gpu()


def result_summary(records: Sequence[Mapping[str, Any]], group_field: str) -> list[dict[str, Any]]:
    import numpy as np

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get(group_field, "all"))].append(row)
    output = []
    for group, rows in sorted(groups.items()):
        totals = [sum(item["latency_ms"].values()) for item in rows]
        output.append(
            {
                group_field: group,
                "questions": len(rows),
                "decomposition_valid": float(np.mean([row["decomposition_valid"] for row in rows])),
                "supporting_qa_recall": float(np.mean([row.get("supporting_qa_recall", 0) for row in rows])),
                "supporting_document_recall": float(
                    np.mean([row.get("supporting_document_recall", 0) for row in rows])
                ),
                "complete_chain_recovery": float(np.mean([row.get("complete_chain_recovery", 0) for row in rows])),
                "surviving_chains_mean": float(
                    np.mean([row.get("surviving_chains", 0) for row in rows])
                ),
                "unique_current_answers_mean": float(
                    np.mean([row.get("unique_current_answers", 0) for row in rows])
                ),
                "EM": float(np.mean([row.get("exact_match", 0) for row in rows])),
                "F1": float(np.mean([row.get("token_f1", 0) for row in rows])),
                "latency_mean_ms": float(np.mean(totals)),
                "latency_p95_ms": float(np.percentile(totals, 95)),
                "evidence_mean": float(np.mean([row.get("evidence_count", 0) for row in rows])),
                "reranked_candidates_mean": float(
                    np.mean([row.get("reranked_candidate_count", 0) for row in rows])
                ),
                "answer_tokens_mean": float(np.mean([row.get("answer_context_tokens", 0) for row in rows])),
            }
        )
    return output


def write_environment(path: str | Path, config: RunConfig) -> None:
    import importlib.metadata

    packages = {}
    for name in ("torch", "transformers", "bitsandbytes", "faiss-cpu", "networkx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu_snapshot(),
        "packages": packages,
        "models": {
            "decomposer": "yigitturali/GSW-QA-Decomposer-Qwen3-4B",
            "query_encoder": "Qwen/Qwen3-Embedding-8B",
            "reranker": "Qwen/Qwen3-Reranker-8B",
            "answerer": "Qwen/Qwen3-4B",
        },
        "configuration": asdict(config),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
