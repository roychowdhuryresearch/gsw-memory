"""Tests for core data models (EntityNode, GSWStructure, etc.)."""

import json

from gsw_memory.memory.models import (
    EntityNode,
    GSWStructure,
    Question,
    Role,
    SpaceNode,
    TimeNode,
    VerbPhraseNode,
)


class TestRole:
    def test_creation(self):
        role = Role(role="Detective", states=["investigating"])
        assert role.role == "Detective"
        assert role.states == ["investigating"]
        assert role.chunk_id is None

    def test_with_chunk_id(self):
        role = Role(role="Suspect", states=["arrested"], chunk_id="0_1")
        assert role.chunk_id == "0_1"

    def test_empty_states(self):
        role = Role(role="Witness", states=[])
        assert role.states == []


class TestEntityNode:
    def test_creation(self, sample_entity):
        assert sample_entity.id == "entity_1"
        assert sample_entity.name == "John Smith"
        assert len(sample_entity.roles) == 1

    def test_serialization_roundtrip(self, sample_entity):
        data = sample_entity.model_dump(mode="json")
        restored = EntityNode(**data)
        assert restored.id == sample_entity.id
        assert restored.name == sample_entity.name
        assert len(restored.roles) == len(sample_entity.roles)


class TestGSWStructure:
    def test_empty_creation(self):
        gsw = GSWStructure()
        assert len(gsw.entity_nodes) == 0
        assert len(gsw.verb_phrase_nodes) == 0

    def test_add_entity(self, sample_entity):
        gsw = GSWStructure()
        gsw.add_entity(sample_entity)
        assert len(gsw.entity_nodes) == 1
        assert gsw.get_entity_by_id("entity_1") is not None

    def test_add_duplicate_entity(self, sample_entity):
        gsw = GSWStructure()
        gsw.add_entity(sample_entity)
        gsw.add_entity(sample_entity)  # duplicate
        assert len(gsw.entity_nodes) == 1

    def test_get_entity_by_id_missing(self, sample_gsw):
        assert sample_gsw.get_entity_by_id("nonexistent") is None

    def test_add_verb_phrase(self, sample_verb_phrase):
        gsw = GSWStructure()
        gsw.add_verb_phrase(sample_verb_phrase)
        assert len(gsw.verb_phrase_nodes) == 1

    def test_add_duplicate_verb_phrase(self, sample_verb_phrase):
        gsw = GSWStructure()
        gsw.add_verb_phrase(sample_verb_phrase)
        gsw.add_verb_phrase(sample_verb_phrase)
        assert len(gsw.verb_phrase_nodes) == 1

    def test_get_verb_phrase_by_id(self, sample_gsw):
        vp = sample_gsw.get_verb_phrase_by_id("vp_1")
        assert vp is not None
        assert vp.phrase == "investigated"

    def test_get_question_by_id(self, sample_gsw):
        result = sample_gsw.get_question_by_id("q_1")
        assert result is not None
        vp, question = result
        assert question.text == "Who investigated the case?"

    def test_get_question_by_id_missing(self, sample_gsw):
        assert sample_gsw.get_question_by_id("nonexistent") is None

    def test_from_json(self, sample_gsw):
        data = sample_gsw.model_dump(mode="json")
        restored = GSWStructure.from_json(data)
        assert len(restored.entity_nodes) == len(sample_gsw.entity_nodes)
        assert len(restored.verb_phrase_nodes) == len(sample_gsw.verb_phrase_nodes)

    def test_json_roundtrip(self, sample_gsw):
        """Full serialize → deserialize roundtrip via JSON string."""
        json_str = json.dumps(sample_gsw.model_dump(mode="json"))
        data = json.loads(json_str)
        restored = GSWStructure.from_json(data)
        assert restored.entity_nodes[0].name == "John Smith"
        assert restored.verb_phrase_nodes[0].phrase == "investigated"

    def test_get_most_connected_entities(self, sample_gsw):
        result = sample_gsw.get_most_connected_entities(top_n=2)
        # entity_1 (John Smith) and entity_2 (Jane Doe) should be most connected
        assert "John Smith" in result
        assert "Jane Doe" in result

    def test_merge_external_entity(self):
        gsw = GSWStructure(
            entity_nodes=[
                EntityNode(id="e1", name="John", roles=[Role(role="A", states=["s1"])]),
                EntityNode(id="e2", name="Johnny", roles=[Role(role="B", states=["s2"])]),
            ],
            verb_phrase_nodes=[
                VerbPhraseNode(
                    id="vp1",
                    phrase="ran",
                    questions=[
                        Question(id="q1", text="Who ran?", answers=["e2"]),
                    ],
                ),
            ],
        )
        external = gsw.get_entity_by_id("e2")
        gsw.merge_external_entity("e1", external)
        # e1 should now have e2's role merged in
        e1 = gsw.get_entity_by_id("e1")
        assert len(e1.roles) == 2
        # Verb phrase answers should be updated to point to e1
        assert gsw.verb_phrase_nodes[0].questions[0].answers == ["e1"]


class TestSpaceTimeNodes:
    def test_space_node_history(self):
        node = SpaceNode(
            id="sp_1",
            name_history={"0_0": "New York", "0_1": "NYC"},
            current_name="NYC",
        )
        assert "New York" in node.formatted_history
        assert "NYC" in node.formatted_history

    def test_time_node_history(self):
        node = TimeNode(
            id="t_1",
            name_history={"0_0": "Morning", "0_1": "Afternoon"},
            current_name="Afternoon",
        )
        assert "Morning" in node.formatted_history
