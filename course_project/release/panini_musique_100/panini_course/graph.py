"""Construct native and reconciled NetworkX views of packaged GSWs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def canonical_entity_name(name: str) -> str:
    return " ".join(re.findall(r"\w+", name.casefold()))


def build_native_gsw_graph(gsw_paths: Iterable[str | Path]):
    """Build a directed heterogeneous MultiDiGraph.

    Entity and verb-phrase IDs are namespaced by document and GSW filename.
    Every QA record becomes a labeled edge from its verb phrase to an answer
    node. Question text and provenance remain attached to the edge.
    """

    import networkx as nx

    graph = nx.MultiDiGraph()
    for raw_path in gsw_paths:
        path = Path(raw_path)
        document_id = path.parent.name
        gsw_name = path.name
        with path.open(encoding="utf-8") as handle:
            gsw = json.load(handle)

        node_lookup: dict[str, str] = {}
        for node_type, key in (
            ("entity", "entity_nodes"),
            ("space", "space_nodes"),
            ("time", "time_nodes"),
        ):
            for node in gsw.get(key, []):
                local_id = str(node["id"])
                node_uid = f"{document_id}::{gsw_name}::{local_id}"
                name = (
                    node.get("name")
                    or node.get("current_name")
                    or next(iter(node.get("name_history", {}).values()), local_id)
                )
                graph.add_node(
                    node_uid,
                    node_type=node_type,
                    name=str(name),
                    document_id=document_id,
                    local_id=local_id,
                    roles=node.get("roles", []),
                )
                node_lookup[local_id] = node_uid

        for verb_phrase in gsw.get("verb_phrase_nodes", []):
            verb_id = str(verb_phrase["id"])
            verb_uid = f"{document_id}::{gsw_name}::{verb_id}"
            graph.add_node(
                verb_uid,
                node_type="verb_phrase",
                phrase=verb_phrase.get("phrase", ""),
                document_id=document_id,
                local_id=verb_id,
            )
            for question in verb_phrase.get("questions", []):
                for answer_id in question.get("answers", []):
                    answer_uid = node_lookup.get(str(answer_id))
                    if answer_uid is None:
                        continue
                    graph.add_edge(
                        verb_uid,
                        answer_uid,
                        key=str(question["id"]),
                        edge_type="qa",
                        question_id=str(question["id"]),
                        question=question.get("text", ""),
                        document_id=document_id,
                    )
    return graph


def build_entity_projection(native_graph):
    """Project verb-phrase neighborhoods into an undirected entity graph.

    Nodes with the same canonical surface name are reconciled across documents.
    Edge weights count how often two named nodes participate in the same local
    verb-phrase neighborhood.
    """

    import networkx as nx

    projection = nx.Graph()
    for node, attributes in native_graph.nodes(data=True):
        if attributes.get("node_type") == "verb_phrase":
            continue
        name = str(attributes.get("name", node))
        canonical = canonical_entity_name(name)
        if not canonical:
            continue
        if canonical not in projection:
            projection.add_node(
                canonical,
                name=name,
                node_type=attributes.get("node_type"),
                occurrences=0,
                documents=set(),
            )
        projection.nodes[canonical]["occurrences"] += 1
        projection.nodes[canonical]["documents"].add(
            attributes.get("document_id")
        )

    for verb, attributes in native_graph.nodes(data=True):
        if attributes.get("node_type") != "verb_phrase":
            continue
        neighbors = []
        for answer in native_graph.successors(verb):
            name = native_graph.nodes[answer].get("name", answer)
            canonical = canonical_entity_name(str(name))
            if canonical and canonical not in neighbors:
                neighbors.append(canonical)
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1 :]:
                if projection.has_edge(left, right):
                    projection[left][right]["weight"] += 1
                else:
                    projection.add_edge(left, right, weight=1)
    return projection
