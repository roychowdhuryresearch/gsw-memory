from __future__ import annotations

import json

from panini_course.graph import (
    build_entity_projection,
    build_native_gsw_graph,
    build_unreconciled_entity_projection,
)


def test_native_graph_and_entity_projection(tmp_path):
    gsw_dir = tmp_path / "doc_1"
    gsw_dir.mkdir()
    gsw_path = gsw_dir / "gsw_1_0.json"
    gsw_path.write_text(
        json.dumps(
            {
                "entity_nodes": [
                    {"id": "e1", "name": "Ada", "roles": []},
                    {"id": "e2", "name": "London", "roles": []},
                ],
                "verb_phrase_nodes": [
                    {
                        "id": "v1",
                        "phrase": "lived in",
                        "questions": [
                            {
                                "id": "q1",
                                "text": "Who lived in London?",
                                "answers": ["e1"],
                            },
                            {
                                "id": "q2",
                                "text": "Where did Ada live?",
                                "answers": ["e2"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    native = build_native_gsw_graph([gsw_path])
    unreconciled = build_unreconciled_entity_projection(native)
    projection = build_entity_projection(native)

    assert native.number_of_nodes() == 3
    assert native.number_of_edges() == 2
    assert unreconciled.number_of_nodes() == 2
    assert unreconciled.has_edge(
        "doc_1::gsw_1_0.json::e1",
        "doc_1::gsw_1_0.json::e2",
    )
    assert projection.has_edge("ada", "london")


def test_cross_document_surface_reconciliation_is_analysis_only(tmp_path):
    paths = []
    for document_id, spelling in (("doc_1", "Ada"), ("doc_2", "ADA")):
        gsw_dir = tmp_path / document_id
        gsw_dir.mkdir()
        path = gsw_dir / "gsw.json"
        path.write_text(
            json.dumps(
                {
                    "entity_nodes": [
                        {"id": "e1", "name": spelling, "roles": []},
                        {
                            "id": "e2",
                            "name": f"Place {document_id}",
                            "roles": [],
                        },
                    ],
                    "verb_phrase_nodes": [
                        {
                            "id": "v1",
                            "phrase": "visited",
                            "questions": [
                                {
                                    "id": "q1",
                                    "text": "Who visited?",
                                    "answers": ["e1"],
                                },
                                {
                                    "id": "q2",
                                    "text": "Which place?",
                                    "answers": ["e2"],
                                },
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    native = build_native_gsw_graph(paths)
    unreconciled = build_unreconciled_entity_projection(native)
    surface_baseline = build_entity_projection(native)

    ada_occurrences = [
        node
        for node, attributes in unreconciled.nodes(data=True)
        if attributes["name"].casefold() == "ada"
    ]
    assert len(ada_occurrences) == 2
    assert "ada" in surface_baseline
    assert surface_baseline.nodes["ada"]["occurrences"] == 2
