from __future__ import annotations

import pytest

from panini_course.qwen_models import QwenDecomposer


def test_decomposer_response_can_be_persisted_before_parsing():
    raw = """```json
    {"questions": [
      {"question": "Who directed the film?", "requires_retrieval": true},
      {"question": "When did <ENTITY_Q1> die?", "requires_retrieval": "yes"}
    ]}
    ```"""

    assert QwenDecomposer.parse_response(raw) == [
        {"question": "Who directed the film?", "requires_retrieval": True},
        {"question": "When did <ENTITY_Q1> die?", "requires_retrieval": True},
    ]


def test_decomposer_response_rejects_missing_questions_list():
    with pytest.raises(ValueError, match="questions list"):
        QwenDecomposer.parse_response('{"plan": []}')
