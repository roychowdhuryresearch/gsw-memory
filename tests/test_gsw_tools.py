"""Tests for GSW Tools (BM25 search, entity context retrieval)."""

import json
from unittest.mock import patch

from gsw_memory.qa.gsw_tools import GSWTools


def _build_bm25_only(tools):
    """Build only the BM25 index, skipping embeddings (no API key needed)."""
    with patch.object(tools, "_build_embedding_index"):
        tools.build_index()


class TestGSWToolsBM25:
    def test_build_index(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        _build_bm25_only(tools)
        assert tools._index_built
        assert len(tools.entity_corpus) == 3  # 3 entities in sample_gsw

    def test_search_bm25(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        _build_bm25_only(tools)
        results = tools.search_gsw_bm25("John Smith", limit=5)
        assert len(results) > 0
        assert results[0]["entity_name"] == "John Smith"

    def test_search_bm25_partial(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        _build_bm25_only(tools)
        results = tools.search_gsw_bm25("police", limit=5)
        assert any("Police" in r["entity_name"] for r in results)

    def test_search_bm25_no_results(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        _build_bm25_only(tools)
        results = tools.search_gsw_bm25("xyznonexistent", limit=5)
        assert len(results) == 0

    def test_lazy_index_build(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        assert not tools._index_built
        with patch.object(tools, "_build_embedding_index"):
            results = tools.search_gsw_bm25("John", limit=5)
        assert tools._index_built


class TestGSWToolsMultiFile:
    def test_multi_file_index(self, multi_gsw_json_files):
        tools = GSWTools(multi_gsw_json_files)
        _build_bm25_only(tools)
        assert len(tools.entity_corpus) == 4

    def test_multi_file_search(self, multi_gsw_json_files):
        tools = GSWTools(multi_gsw_json_files)
        _build_bm25_only(tools)
        # Search for an entity unique to one doc (avoids BM25 IDF=0 for common terms)
        results = tools.search_gsw_bm25("Forrest Gump", limit=10)
        assert len(results) >= 1
        assert results[0]["entity_name"] == "Forrest Gump"

    def test_global_ids_unique(self, multi_gsw_json_files):
        tools = GSWTools(multi_gsw_json_files)
        _build_bm25_only(tools)
        global_ids = [m["global_id"] for m in tools.entity_metadata]
        assert len(global_ids) == len(set(global_ids))  # all unique


class TestGSWToolsEntityContext:
    def test_get_entity_context(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        context = tools.get_entity_context("entity_1")
        assert context["entity_name"] == "John Smith"
        assert len(context["questions"]) > 0

    def test_get_entity_context_with_global_id(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        global_id = f"{gsw_json_file}::entity_1"
        context = tools.get_entity_context(global_id)
        assert context["entity_name"] == "John Smith"

    def test_get_entity_context_missing(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        context = tools.get_entity_context("nonexistent_entity")
        assert "error" in context

    def test_get_multiple_entity_contexts(self, gsw_json_file):
        tools = GSWTools(gsw_json_file)
        contexts = tools.get_multiple_entity_contexts(["entity_1", "entity_2"])
        assert len(contexts) == 2
        names = [c["entity_name"] for c in contexts]
        assert "John Smith" in names
        assert "Jane Doe" in names

    def test_entity_context_has_other_entities(self, gsw_json_file):
        """Questions in context should reference other connected entities."""
        tools = GSWTools(gsw_json_file)
        context = tools.get_entity_context("entity_1")
        # entity_1 (John Smith) appears in questions that also reference entity_2
        all_other_entities = []
        for q in context["questions"]:
            all_other_entities.extend(q["other_entities"])
        other_names = [e["entity_name"] for e in all_other_entities]
        assert len(other_names) > 0  # Should have connected entities
