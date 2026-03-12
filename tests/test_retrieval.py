"""Tests for multi-hop retrieval over GSW structures.

Uses real GSW outputs extracted from the 2wiki MultiHopQA dataset
(stored in tests/fixtures/2wiki/) to test retrieval behavior against
known gold chains.

Test scenarios:
  Q0: "When did Lothair II's mother die?"
      Chain: Lothair II → son_of → Ermengarde of Tours → died_on → 20 March 851
      Files: lothair_ii.json (doc_4), ermengarde.json (doc_5)

  Q2: "What is the place of birth of the performer of Changed It?"
      Chain: Changed It → performed_by → Nicki Minaj → born_in → Saint James, Port of Spain
      Files: changed_it.json (doc_22), nicki_minaj.json (doc_24)
"""

import json
import os
from unittest.mock import patch

import pytest

from gsw_memory.memory.models import GSWStructure
from gsw_memory.qa.gsw_tools import GSWTools

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "2wiki")


def _build_bm25_only(tools: GSWTools):
    """Build only the BM25 index, skipping embeddings (no API key needed)."""
    with patch.object(tools, "_build_embedding_index"):
        tools.build_index()


# ---------------------------------------------------------------------------
# Fixtures: real 2wiki GSW files
# ---------------------------------------------------------------------------


@pytest.fixture
def lothair_ermengarde_files():
    """GSW files for Q0: Lothair II + Ermengarde of Tours."""
    return [
        os.path.join(FIXTURES_DIR, "lothair_ii.json"),
        os.path.join(FIXTURES_DIR, "ermengarde.json"),
    ]


@pytest.fixture
def changed_it_nicki_files():
    """GSW files for Q2: Changed It + Nicki Minaj."""
    return [
        os.path.join(FIXTURES_DIR, "changed_it.json"),
        os.path.join(FIXTURES_DIR, "nicki_minaj.json"),
    ]


@pytest.fixture
def all_fixture_files():
    """All four fixture GSW files."""
    return [
        os.path.join(FIXTURES_DIR, "lothair_ii.json"),
        os.path.join(FIXTURES_DIR, "ermengarde.json"),
        os.path.join(FIXTURES_DIR, "changed_it.json"),
        os.path.join(FIXTURES_DIR, "nicki_minaj.json"),
    ]


@pytest.fixture
def lothair_tools(lothair_ermengarde_files):
    tools = GSWTools(lothair_ermengarde_files)
    _build_bm25_only(tools)
    return tools


@pytest.fixture
def changed_it_tools(changed_it_nicki_files):
    tools = GSWTools(changed_it_nicki_files)
    _build_bm25_only(tools)
    return tools


@pytest.fixture
def all_tools(all_fixture_files):
    tools = GSWTools(all_fixture_files)
    _build_bm25_only(tools)
    return tools


# Gold answers from the 2wiki dataset
GOLD = {
    "lothair_mother_death": {
        "question": "When did Lothair II's mother die?",
        "answer": "20 March 851",
        "evidences": [
            ["Lothair II", "mother", "Ermengarde of Tours"],
            ["Ermengarde of Tours", "date of death", "20 March 851"],
        ],
        "chain_entities": ["Lothair II", "Ermengarde of Tours", "20 March 851"],
    },
    "changed_it_birthplace": {
        "question": "What is the place of birth of the performer of Changed It?",
        "answer": "Port of Spain",
        "evidences": [
            ["Changed It", "performer", "Nicki Minaj"],
            ["Nicki Minaj", "place of birth", "Port of Spain"],
        ],
        "chain_entities": ["Changed It", "Nicki Minaj", "Saint James, Port of Spain"],
    },
}


# ---------------------------------------------------------------------------
# Test: Single-hop entity search
# ---------------------------------------------------------------------------


class TestSingleHopSearch:
    """BM25 search finds entities by name from real GSW data."""

    def test_find_lothair(self, lothair_tools):
        results = lothair_tools.search_gsw_bm25("Lothair", limit=5)
        names = [r["entity_name"] for r in results]
        assert "Lothair II" in names

    def test_find_ermengarde(self, lothair_tools):
        results = lothair_tools.search_gsw_bm25("Ermengarde", limit=5)
        names = [r["entity_name"] for r in results]
        assert "Ermengarde of Tours" in names

    def test_find_changed_it(self, changed_it_tools):
        results = changed_it_tools.search_gsw_bm25("Changed It", limit=5)
        names = [r["entity_name"] for r in results]
        assert "Changed It" in names

    def test_find_nicki_minaj(self, changed_it_tools):
        results = changed_it_tools.search_gsw_bm25("Nicki Minaj", limit=5)
        names = [r["entity_name"] for r in results]
        # Nicki Minaj is stored as "Nicki Minaj" in changed_it.json
        # and as "Onika Tanya Maraj-Petty" (birth name) + "Nicki Minaj" in nicki_minaj.json
        assert any("Nicki Minaj" in n or "Nicki" in n for n in names)

    def test_entity_count(self, lothair_tools):
        """Verify correct number of entities loaded from real data."""
        # lothair_ii.json has 9 entities, ermengarde.json has 12 entities
        assert len(lothair_tools.entity_corpus) == 21


# ---------------------------------------------------------------------------
# Test: Entity context reveals connections (single-hop)
# ---------------------------------------------------------------------------


class TestEntityContext:
    """get_entity_context returns questions and connected entities."""

    def test_lothair_context_has_questions(self, lothair_tools, lothair_ermengarde_files):
        """Lothair II should have verb phrase questions about his relationships."""
        ctx = lothair_tools.get_entity_context(
            f"{lothair_ermengarde_files[0]}::e1"
        )
        assert ctx["entity_name"] == "Lothair II"
        assert len(ctx["questions"]) > 0

    def test_lothair_connected_to_ermengarde(self, lothair_tools, lothair_ermengarde_files):
        """Lothair II's context should show Ermengarde via 'son of' VP."""
        ctx = lothair_tools.get_entity_context(
            f"{lothair_ermengarde_files[0]}::e1"
        )
        connected_names = set()
        for q in ctx["questions"]:
            for other in q["other_entities"]:
                connected_names.add(other["entity_name"])

        assert "Ermengarde of Tours" in connected_names, (
            f"Ermengarde not found in Lothair's connections. Got: {connected_names}"
        )

    def test_changed_it_connected_to_nicki(self, changed_it_tools, changed_it_nicki_files):
        """Changed It's context should show Nicki Minaj via 'performed by' VP."""
        ctx = changed_it_tools.get_entity_context(
            f"{changed_it_nicki_files[0]}::e1"
        )
        assert ctx["entity_name"] == "Changed It"

        connected_names = set()
        for q in ctx["questions"]:
            for other in q["other_entities"]:
                connected_names.add(other["entity_name"])

        assert "Nicki Minaj" in connected_names, (
            f"Nicki Minaj not found in Changed It's connections. Got: {connected_names}"
        )

    def test_ermengarde_connected_to_death_date(self, lothair_tools, lothair_ermengarde_files):
        """Ermengarde's context should reveal her death date via 'died on' VP."""
        ctx = lothair_tools.get_entity_context(
            f"{lothair_ermengarde_files[1]}::e1"
        )
        assert ctx["entity_name"] == "Ermengarde of Tours"

        connected_names = set()
        for q in ctx["questions"]:
            for other in q["other_entities"]:
                connected_names.add(other["entity_name"])

        assert "20 March 851" in connected_names, (
            f"Death date not found in Ermengarde's connections. Got: {connected_names}"
        )

    def test_nicki_connected_to_birthplace(self, changed_it_tools, changed_it_nicki_files):
        """Nicki Minaj's context should reveal birthplace via 'born in' VP."""
        # In nicki_minaj.json, Nicki is e3 (professional name) and e1 (birth name)
        # The 'born in' VP references e3 (Nicki Minaj) and e5 (Saint James, Port of Spain)
        ctx = changed_it_tools.get_entity_context(
            f"{changed_it_nicki_files[1]}::e3"
        )
        assert ctx["entity_name"] == "Nicki Minaj"

        connected_names = set()
        for q in ctx["questions"]:
            for other in q["other_entities"]:
                connected_names.add(other["entity_name"])

        assert "Saint James, Port of Spain" in connected_names, (
            f"Birthplace not found in Nicki's connections. Got: {connected_names}"
        )


# ---------------------------------------------------------------------------
# Test: Multi-hop chain following (2 hops)
# ---------------------------------------------------------------------------


class TestMultiHopChainFollowing:
    """Verify 2-hop chains through real GSW verb phrase links."""

    def test_lothair_to_death_date(self, lothair_tools, lothair_ermengarde_files):
        """Chain: Lothair II → Ermengarde of Tours → 20 March 851.

        Hop 1 (lothair_ii.json): Lothair II (e1) → 'son of' v4 → Ermengarde (e6)
        Hop 2 (ermengarde.json): Ermengarde (e1) → 'died on' v1 → 20 March 851 (e2)
        """
        lothair_file = lothair_ermengarde_files[0]
        ermengarde_file = lothair_ermengarde_files[1]

        # Hop 1: Lothair II → find Ermengarde
        ctx = lothair_tools.get_entity_context(f"{lothair_file}::e1")
        ermengarde_found = any(
            other["entity_name"] == "Ermengarde of Tours"
            for q in ctx["questions"]
            for other in q["other_entities"]
        )
        assert ermengarde_found, "Hop 1 failed: Ermengarde not found from Lothair II"

        # Hop 2: Ermengarde → find death date
        ctx2 = lothair_tools.get_entity_context(f"{ermengarde_file}::e1")
        death_date_found = any(
            other["entity_name"] == "20 March 851"
            for q in ctx2["questions"]
            for other in q["other_entities"]
        )
        assert death_date_found, "Hop 2 failed: Death date not found from Ermengarde"

    def test_changed_it_to_birthplace(self, changed_it_tools, changed_it_nicki_files):
        """Chain: Changed It → Nicki Minaj → Saint James, Port of Spain.

        Hop 1 (changed_it.json): Changed It (e1) → 'performed by' v1 → Nicki Minaj (e2)
        Hop 2 (nicki_minaj.json): Nicki Minaj (e3) → 'born in' v3 → Saint James (e5)
        """
        changed_file = changed_it_nicki_files[0]
        nicki_file = changed_it_nicki_files[1]

        # Hop 1: Changed It → find Nicki Minaj
        ctx = changed_it_tools.get_entity_context(f"{changed_file}::e1")
        nicki_found = any(
            other["entity_name"] == "Nicki Minaj"
            for q in ctx["questions"]
            for other in q["other_entities"]
        )
        assert nicki_found, "Hop 1 failed: Nicki Minaj not found from Changed It"

        # Hop 2: Nicki Minaj (e3 in nicki_minaj.json) → find birthplace
        ctx2 = changed_it_tools.get_entity_context(f"{nicki_file}::e3")
        birthplace_found = any(
            other["entity_name"] == "Saint James, Port of Spain"
            for q in ctx2["questions"]
            for other in q["other_entities"]
        )
        assert birthplace_found, "Hop 2 failed: Birthplace not found from Nicki Minaj"


# ---------------------------------------------------------------------------
# Test: Cross-document entity bridging
# ---------------------------------------------------------------------------


class TestCrossDocumentBridging:
    """The key challenge in multi-hop: the same entity appears in different
    documents with different information. Retrieval must bridge across docs."""

    def test_ermengarde_different_info_per_doc(self, lothair_tools, lothair_ermengarde_files):
        """Ermengarde in lothair_ii.json has family info;
        in ermengarde.json has death date. Both are needed."""
        lothair_file = lothair_ermengarde_files[0]
        ermengarde_file = lothair_ermengarde_files[1]

        # In lothair_ii.json, Ermengarde is e6 — only appears as Lothair's mother
        ctx_lothair = lothair_tools.get_entity_context(f"{lothair_file}::e6")
        assert ctx_lothair["entity_name"] == "Ermengarde of Tours"
        lothair_q_texts = [q["question_text"] for q in ctx_lothair["questions"]]

        # In ermengarde.json, Ermengarde is e1 — has death date, family, marriage
        ctx_erm = lothair_tools.get_entity_context(f"{ermengarde_file}::e1")
        assert ctx_erm["entity_name"] == "Ermengarde of Tours"
        erm_q_texts = [q["question_text"] for q in ctx_erm["questions"]]

        # The death date question is only in ermengarde.json
        assert any("die" in q.lower() or "died" in q.lower() for q in erm_q_texts)
        # The "son of" relationship is only in lothair_ii.json
        assert any("son" in q.lower() for q in lothair_q_texts)

    def test_search_bridges_documents(self, lothair_tools, lothair_ermengarde_files):
        """After finding Ermengarde via Lothair's context, search should find
        her entry in ermengarde.json to get death date."""
        # Search for Ermengarde
        results = lothair_tools.search_gsw_bm25("Ermengarde", limit=10)
        source_files = set(r["source_file"] for r in results if "Ermengarde" in r["entity_name"])

        # Should find Ermengarde in both files
        assert len(source_files) == 2, (
            f"Expected Ermengarde in 2 files, found in {len(source_files)}: {source_files}"
        )

    def test_nicki_different_info_per_doc(self, changed_it_tools, changed_it_nicki_files):
        """Nicki Minaj in changed_it.json → performer info;
        in nicki_minaj.json → birthplace, awards, career."""
        changed_file = changed_it_nicki_files[0]
        nicki_file = changed_it_nicki_files[1]

        # In changed_it.json, Nicki is e2
        ctx_song = changed_it_tools.get_entity_context(f"{changed_file}::e2")
        assert ctx_song["entity_name"] == "Nicki Minaj"
        song_q_texts = [q["question_text"] for q in ctx_song["questions"]]
        assert any("perform" in q.lower() or "song" in q.lower() for q in song_q_texts)

        # In nicki_minaj.json, Nicki is e3
        ctx_bio = changed_it_tools.get_entity_context(f"{nicki_file}::e3")
        assert ctx_bio["entity_name"] == "Nicki Minaj"
        bio_q_texts = [q["question_text"] for q in ctx_bio["questions"]]
        assert any("born" in q.lower() for q in bio_q_texts)


# ---------------------------------------------------------------------------
# Test: Entity matcher on real GSW data
# ---------------------------------------------------------------------------


class TestMatcherOnRealData:
    """EntityMatcher finds seed entities from real GSW structures."""

    def test_matcher_finds_lothair(self, lothair_ermengarde_files):
        from gsw_memory.qa.matcher import EntityMatcher

        with open(lothair_ermengarde_files[0]) as f:
            gsw = GSWStructure.from_json(json.load(f))

        matcher = EntityMatcher()
        matches = matcher.find_matching_entities(["Lothair II"], gsw)
        names = [m.name for m in matches]
        assert "Lothair II" in names

    def test_matcher_finds_changed_it(self, changed_it_nicki_files):
        from gsw_memory.qa.matcher import EntityMatcher

        with open(changed_it_nicki_files[0]) as f:
            gsw = GSWStructure.from_json(json.load(f))

        matcher = EntityMatcher()
        matches = matcher.find_matching_entities(["Changed It"], gsw)
        names = [m.name for m in matches]
        assert "Changed It" in names

    def test_matcher_connected_finds_bridge(self, lothair_ermengarde_files):
        """Finding Lothair with include_connected should also find Ermengarde."""
        from gsw_memory.qa.matcher import EntityMatcher

        with open(lothair_ermengarde_files[0]) as f:
            gsw = GSWStructure.from_json(json.load(f))

        matcher = EntityMatcher()
        matches = matcher.find_matching_entities(
            ["Lothair II"], gsw, include_connected=True
        )
        names = [m.name for m in matches]
        assert "Lothair II" in names
        assert len(matches) > 1, "Connected entities should be found"


# ---------------------------------------------------------------------------
# Test: Full retrieval chain (integration)
# ---------------------------------------------------------------------------


class TestFullRetrievalChain:
    """End-to-end: search → context → chain → gold answer. No LLM calls."""

    def test_lothair_mother_death_chain(self, lothair_tools, lothair_ermengarde_files):
        """Full chain for: 'When did Lothair II's mother die?'

        1. Search 'Lothair' → find Lothair II
        2. Get context → find Ermengarde of Tours (his mother)
        3. Search 'Ermengarde' or use global_id → find in ermengarde.json
        4. Get context → find '20 March 851' (death date)
        """
        gold = GOLD["lothair_mother_death"]
        lothair_file = lothair_ermengarde_files[0]
        ermengarde_file = lothair_ermengarde_files[1]

        # Step 1: Find seed entity
        results = lothair_tools.search_gsw_bm25("Lothair", limit=5)
        assert any(r["entity_name"] == "Lothair II" for r in results)

        # Step 2: Get Lothair's context → find bridge to Ermengarde
        ctx = lothair_tools.get_entity_context(f"{lothair_file}::e1")
        bridge_names = set()
        for q in ctx["questions"]:
            for other in q["other_entities"]:
                bridge_names.add(other["entity_name"])
        assert "Ermengarde of Tours" in bridge_names

        # Step 3: Get Ermengarde's context in her own doc → find answer
        ctx2 = lothair_tools.get_entity_context(f"{ermengarde_file}::e1")
        answer_names = set()
        for q in ctx2["questions"]:
            for other in q["other_entities"]:
                answer_names.add(other["entity_name"])
        assert gold["answer"] in answer_names, (
            f"Gold answer '{gold['answer']}' not found. Got: {answer_names}"
        )

    def test_changed_it_birthplace_chain(self, changed_it_tools, changed_it_nicki_files):
        """Full chain for: 'What is the place of birth of the performer of Changed It?'

        1. Search 'Changed It' → find entity
        2. Get context → find Nicki Minaj (performer)
        3. Search 'Nicki Minaj' → find in nicki_minaj.json
        4. Get context → find 'Saint James, Port of Spain' (birthplace)
        """
        gold = GOLD["changed_it_birthplace"]
        changed_file = changed_it_nicki_files[0]
        nicki_file = changed_it_nicki_files[1]

        # Step 1: Find seed
        results = changed_it_tools.search_gsw_bm25("Changed", limit=5)
        assert any(r["entity_name"] == "Changed It" for r in results)

        # Step 2: Context → bridge to Nicki
        ctx = changed_it_tools.get_entity_context(f"{changed_file}::e1")
        bridge_names = set()
        for q in ctx["questions"]:
            for other in q["other_entities"]:
                bridge_names.add(other["entity_name"])
        assert "Nicki Minaj" in bridge_names

        # Step 3: Nicki's context → birthplace
        ctx2 = changed_it_tools.get_entity_context(f"{nicki_file}::e3")
        answer_names = set()
        for q in ctx2["questions"]:
            for other in q["other_entities"]:
                answer_names.add(other["entity_name"])
        assert "Saint James, Port of Spain" in answer_names, (
            f"Gold answer not found. Got: {answer_names}"
        )

    def test_collected_evidence_covers_chain(self, lothair_tools, lothair_ermengarde_files):
        """Following the chain should collect all entities in the gold evidence."""
        gold = GOLD["lothair_mother_death"]
        lothair_file = lothair_ermengarde_files[0]
        ermengarde_file = lothair_ermengarde_files[1]

        collected = set()

        # Hop 1: Lothair context
        ctx1 = lothair_tools.get_entity_context(f"{lothair_file}::e1")
        collected.add(ctx1["entity_name"])
        for q in ctx1["questions"]:
            for other in q["other_entities"]:
                collected.add(other["entity_name"])

        # Hop 2: Ermengarde context
        ctx2 = lothair_tools.get_entity_context(f"{ermengarde_file}::e1")
        collected.add(ctx2["entity_name"])
        for q in ctx2["questions"]:
            for other in q["other_entities"]:
                collected.add(other["entity_name"])

        for entity in gold["chain_entities"]:
            assert entity in collected, f"'{entity}' missing from collected evidence"

    def test_multi_context_batch(self, lothair_tools, lothair_ermengarde_files):
        """get_multiple_entity_contexts retrieves both hops in one call."""
        lothair_file = lothair_ermengarde_files[0]
        ermengarde_file = lothair_ermengarde_files[1]

        ids = [f"{lothair_file}::e1", f"{ermengarde_file}::e1"]
        contexts = lothair_tools.get_multiple_entity_contexts(ids)

        assert len(contexts) == 2
        names = {c["entity_name"] for c in contexts}
        assert "Lothair II" in names
        assert "Ermengarde of Tours" in names
