"""
Shared test fixtures for GSW Memory tests.

Provides reusable GSWStructure instances and components
that don't require API keys or network access.
"""

import json
import os
import tempfile

import pytest

from gsw_memory.memory.models import (
    EntityNode,
    GSWStructure,
    Question,
    Role,
    VerbPhraseNode,
)


@pytest.fixture
def sample_role():
    """A single role with states."""
    return Role(role="Detective", states=["investigating", "on duty"], chunk_id="0_0")


@pytest.fixture
def sample_entity(sample_role):
    """A single entity node."""
    return EntityNode(
        id="entity_1",
        name="John Smith",
        roles=[sample_role],
        chunk_id="0_0",
    )


@pytest.fixture
def sample_question():
    """A single question node."""
    return Question(
        id="q_1",
        text="Who investigated the case?",
        answers=["entity_1", "entity_2"],
        chunk_id="0_0",
    )


@pytest.fixture
def sample_verb_phrase(sample_question):
    """A single verb phrase node with questions."""
    return VerbPhraseNode(
        id="vp_1",
        phrase="investigated",
        questions=[sample_question],
        chunk_id="0_0",
    )


@pytest.fixture
def sample_gsw():
    """A complete GSW structure with multiple entities and verb phrases.

    Represents a simple crime investigation scenario:
    - John Smith (Detective) investigated a case
    - Jane Doe (Suspect) was apprehended
    - The case connects them through verb phrase questions
    """
    entities = [
        EntityNode(
            id="entity_1",
            name="John Smith",
            roles=[Role(role="Detective", states=["investigating"], chunk_id="0_0")],
            chunk_id="0_0",
        ),
        EntityNode(
            id="entity_2",
            name="Jane Doe",
            roles=[Role(role="Suspect", states=["apprehended"], chunk_id="0_0")],
            chunk_id="0_0",
        ),
        EntityNode(
            id="entity_3",
            name="Central Police Station",
            roles=[Role(role="Location", states=["active"], chunk_id="0_0")],
            chunk_id="0_0",
        ),
    ]

    verb_phrases = [
        VerbPhraseNode(
            id="vp_1",
            phrase="investigated",
            questions=[
                Question(
                    id="q_1",
                    text="Who investigated the case?",
                    answers=["entity_1"],
                    chunk_id="0_0",
                ),
                Question(
                    id="q_2",
                    text="Who was the suspect in the investigation?",
                    answers=["entity_2"],
                    chunk_id="0_0",
                ),
            ],
            chunk_id="0_0",
        ),
        VerbPhraseNode(
            id="vp_2",
            phrase="apprehended",
            questions=[
                Question(
                    id="q_3",
                    text="Who was apprehended?",
                    answers=["entity_2"],
                    chunk_id="0_0",
                ),
                Question(
                    id="q_4",
                    text="Who apprehended the suspect?",
                    answers=["entity_1"],
                    chunk_id="0_0",
                ),
                Question(
                    id="q_5",
                    text="Where was the suspect apprehended?",
                    answers=["entity_3"],
                    chunk_id="0_0",
                ),
            ],
            chunk_id="0_0",
        ),
    ]

    return GSWStructure(entity_nodes=entities, verb_phrase_nodes=verb_phrases)


@pytest.fixture
def multi_doc_gsws():
    """Two GSW structures representing different documents for cross-doc testing."""
    gsw_doc1 = GSWStructure(
        entity_nodes=[
            EntityNode(
                id="e1",
                name="Robert Zemeckis",
                roles=[Role(role="Director", states=["active"])],
            ),
            EntityNode(
                id="e2",
                name="Forrest Gump",
                roles=[Role(role="Film", states=["released 1994"])],
            ),
        ],
        verb_phrase_nodes=[
            VerbPhraseNode(
                id="vp1",
                phrase="directed",
                questions=[
                    Question(
                        id="q1",
                        text="Who directed the film?",
                        answers=["e1"],
                    ),
                    Question(
                        id="q2",
                        text="What film was directed?",
                        answers=["e2"],
                    ),
                ],
            ),
        ],
    )

    gsw_doc2 = GSWStructure(
        entity_nodes=[
            EntityNode(
                id="e1",
                name="Robert Zemeckis",
                roles=[Role(role="Director", states=["active"])],
            ),
            EntityNode(
                id="e3",
                name="Back to the Future",
                roles=[Role(role="Film", states=["released 1985"])],
            ),
        ],
        verb_phrase_nodes=[
            VerbPhraseNode(
                id="vp1",
                phrase="directed",
                questions=[
                    Question(
                        id="q1",
                        text="Who directed the film?",
                        answers=["e1"],
                    ),
                    Question(
                        id="q2",
                        text="What film was directed?",
                        answers=["e3"],
                    ),
                ],
            ),
        ],
    )

    return [gsw_doc1, gsw_doc2]


@pytest.fixture
def gsw_json_file(sample_gsw, tmp_path):
    """Write a sample GSW to a temp JSON file and return the path."""
    file_path = tmp_path / "test_gsw.json"
    with open(file_path, "w") as f:
        json.dump(sample_gsw.model_dump(mode="json"), f)
    return str(file_path)


@pytest.fixture
def multi_gsw_json_files(multi_doc_gsws, tmp_path):
    """Write multiple GSW structures to temp JSON files."""
    paths = []
    for i, gsw in enumerate(multi_doc_gsws):
        file_path = tmp_path / f"gsw_doc_{i}.json"
        with open(file_path, "w") as f:
            json.dump(gsw.model_dump(mode="json"), f)
        paths.append(str(file_path))
    return paths
