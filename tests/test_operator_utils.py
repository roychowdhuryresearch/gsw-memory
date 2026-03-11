"""Tests for operator utility functions (chunking, parsing, JSON extraction)."""

import json

import pytest

from gsw_memory.memory.operator_utils.utils import (
    chunk_text,
    extract_json_from_output,
    parse_gsw,
)


class TestChunkText:
    def test_basic_chunking(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text(text, chunk_size=2, overlap=0)
        assert len(chunks) == 3
        assert chunks[0]["idx"] == 0
        assert chunks[1]["idx"] == 1

    def test_chunking_with_overlap(self):
        text = "A. B. C. D. E. F."
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        # With overlap=1, stride = 3-1 = 2
        # Chunks: [A,B,C], [C,D,E], [E,F]
        assert len(chunks) == 3
        # The overlapping sentence should appear in consecutive chunks
        assert "C" in chunks[0]["text"]
        assert "C" in chunks[1]["text"]

    def test_single_sentence(self):
        text = "Just one sentence."
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=3, overlap=1)
        assert len(chunks) == 0

    def test_chunk_indices(self):
        text = "A. B. C. D."
        chunks = chunk_text(text, chunk_size=2, overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk["idx"] == i
            assert "start_sentence" in chunk
            assert "end_sentence" in chunk


class TestExtractJsonFromOutput:
    def test_valid_json(self):
        result = extract_json_from_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json(self):
        with pytest.raises(ValueError):
            extract_json_from_output("not json at all")


class TestParseGSW:
    def test_parse_json_string(self):
        gsw_json = json.dumps(
            {
                "entity_nodes": [
                    {
                        "id": "e1",
                        "name": "Alice",
                        "roles": [{"role": "Protagonist", "states": ["happy"]}],
                    }
                ],
                "verb_phrase_nodes": [
                    {
                        "id": "vp1",
                        "phrase": "walked",
                        "questions": [
                            {"id": "q1", "text": "Who walked?", "answers": ["e1"]}
                        ],
                    }
                ],
            }
        )
        gsw = parse_gsw(gsw_json)
        assert len(gsw.entity_nodes) == 1
        assert gsw.entity_nodes[0].name == "Alice"
        assert len(gsw.verb_phrase_nodes) == 1
        assert gsw.verb_phrase_nodes[0].phrase == "walked"

    def test_parse_with_code_fence(self):
        text = '```json\n{"entity_nodes": [{"id": "e1", "name": "Bob", "roles": []}], "verb_phrase_nodes": []}\n```'
        gsw = parse_gsw(text)
        assert gsw.entity_nodes[0].name == "Bob"

    def test_parse_with_generic_code_fence(self):
        text = '```\n{"entity_nodes": [{"id": "e1", "name": "Carol", "roles": []}], "verb_phrase_nodes": []}\n```'
        gsw = parse_gsw(text)
        assert gsw.entity_nodes[0].name == "Carol"

    def test_parse_empty_gsw(self):
        gsw_json = json.dumps({"entity_nodes": [], "verb_phrase_nodes": []})
        gsw = parse_gsw(gsw_json)
        assert len(gsw.entity_nodes) == 0
        assert len(gsw.verb_phrase_nodes) == 0

    def test_parse_invalid_json(self):
        with pytest.raises(ValueError):
            parse_gsw("completely invalid {{ not json")

    def test_parse_preserves_chunk_ids(self):
        gsw_json = json.dumps(
            {
                "entity_nodes": [
                    {
                        "id": "e1",
                        "name": "Dave",
                        "roles": [
                            {"role": "Agent", "states": [], "chunk_id": "0_2"}
                        ],
                        "chunk_id": "0_2",
                    }
                ],
                "verb_phrase_nodes": [],
            }
        )
        gsw = parse_gsw(gsw_json)
        assert gsw.entity_nodes[0].chunk_id == "0_2"
        assert gsw.entity_nodes[0].roles[0].chunk_id == "0_2"
