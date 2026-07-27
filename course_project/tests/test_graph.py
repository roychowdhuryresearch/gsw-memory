from __future__ import annotations

import json

from panini_course.graph import build_entity_projection, build_native_gsw_graph


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
    projection = build_entity_projection(native)

    assert native.number_of_nodes() == 3
    assert native.number_of_edges() == 2
    assert projection.has_edge("ada", "london")
