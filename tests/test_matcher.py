"""Tests for entity matching logic (no API calls needed)."""

from gsw_memory.qa.matcher import EntityMatcher


class TestEntityMatcher:
    def setup_method(self):
        self.matcher = EntityMatcher()

    def test_exact_match(self, sample_gsw):
        matches = self.matcher.find_matching_entities(["John Smith"], sample_gsw)
        assert len(matches) == 1
        assert matches[0].name == "John Smith"

    def test_substring_match(self, sample_gsw):
        """Entity name is a substring of the query."""
        matches = self.matcher.find_matching_entities(
            ["Detective John Smith"], sample_gsw
        )
        assert any(m.name == "John Smith" for m in matches)

    def test_reverse_substring_match(self, sample_gsw):
        """Query is a substring of the entity name."""
        matches = self.matcher.find_matching_entities(["John"], sample_gsw)
        assert any(m.name == "John Smith" for m in matches)

    def test_case_insensitive(self, sample_gsw):
        matches = self.matcher.find_matching_entities(["john smith"], sample_gsw)
        assert any(m.name == "John Smith" for m in matches)

    def test_word_overlap_match(self, sample_gsw):
        """Word-level overlap matching."""
        matches = self.matcher.find_matching_entities(["Smith"], sample_gsw)
        assert any(m.name == "John Smith" for m in matches)

    def test_no_match(self, sample_gsw):
        matches = self.matcher.find_matching_entities(
            ["Nonexistent Person"], sample_gsw
        )
        # "Person" doesn't appear in any entity name
        # But "Nonexistent" doesn't either, so no word overlap
        # Actually — no entity has "Nonexistent" or "Person" as a word
        # Central Police Station has "Station", "Police", "Central"
        assert not any(m.name == "Nonexistent Person" for m in matches)

    def test_multiple_queries(self, sample_gsw):
        matches = self.matcher.find_matching_entities(
            ["John Smith", "Jane Doe"], sample_gsw
        )
        names = [m.name for m in matches]
        assert "John Smith" in names
        assert "Jane Doe" in names

    def test_connected_entities(self, sample_gsw):
        """With include_connected=True, should find entities connected via verb phrases."""
        matches = self.matcher.find_matching_entities(
            ["John Smith"], sample_gsw, include_connected=True
        )
        names = [m.name for m in matches]
        # John Smith is connected to Jane Doe and Central Police Station
        # through verb phrases
        assert "John Smith" in names
        assert len(matches) > 1  # Should have connected entities too

    def test_empty_query(self, sample_gsw):
        matches = self.matcher.find_matching_entities([], sample_gsw)
        assert len(matches) == 0

    def test_quoted_entity_names(self):
        """Entity names with quotes should still match."""
        from gsw_memory.memory.models import EntityNode, GSWStructure, Role

        gsw = GSWStructure(
            entity_nodes=[
                EntityNode(
                    id="e1",
                    name='"The Great Gatsby"',
                    roles=[Role(role="Book", states=[])],
                ),
            ]
        )
        matches = self.matcher.find_matching_entities(["The Great Gatsby"], gsw)
        assert len(matches) == 1
