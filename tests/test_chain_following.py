"""Tests for ChainFollowingQA pipeline logic.

Tests pure-logic methods (chain identification, scoring, beam pruning,
entity substitution) that don't require API keys or network access.
"""

from gsw_memory.qa.chain_following_qa import ChainFollowingQA


# We can't construct a full ChainFollowingQA without API keys, so we
# test the static/class methods and internal logic directly.


class TestIdentifyReasoningChains:
    """Test identify_reasoning_chains with various decomposition patterns."""

    @staticmethod
    def _make_qa_stub():
        """Create a minimal ChainFollowingQA-like object for testing chain logic."""
        # We only need the method, which is a regular method on the class.
        # We'll create a mock-like object.
        qa = object.__new__(ChainFollowingQA)
        return qa

    def test_linear_chain(self):
        qa = self._make_qa_stub()
        decomposed = [
            {"question": "Who directed Casablanca?", "requires_retrieval": True},
            {"question": "Who was <ENTITY_Q1>'s spouse?", "requires_retrieval": True},
            {
                "question": "What is <ENTITY_Q2>'s birth year?",
                "requires_retrieval": True,
            },
        ]
        chains = qa.identify_reasoning_chains(decomposed)
        assert len(chains) == 1
        assert chains[0] == [0, 1, 2]

    def test_dag_structure(self):
        qa = self._make_qa_stub()
        decomposed = [
            {"question": "Who directed Dune?", "requires_retrieval": True},
            {"question": "Who directed The Dark Knight?", "requires_retrieval": True},
            {
                "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
                "requires_retrieval": True,
            },
            {"question": "Which film?", "requires_retrieval": False},
        ]
        chains = qa.identify_reasoning_chains(decomposed)
        assert len(chains) == 1
        # Q0 and Q1 are roots, Q2 depends on both
        chain = chains[0]
        assert chain.index(0) < chain.index(2)
        assert chain.index(1) < chain.index(2)

    def test_no_retrieval_questions(self):
        qa = self._make_qa_stub()
        decomposed = [
            {"question": "What is 2+2?", "requires_retrieval": False},
        ]
        chains = qa.identify_reasoning_chains(decomposed)
        assert chains == []

    def test_single_question(self):
        qa = self._make_qa_stub()
        decomposed = [
            {"question": "Who is Alice?", "requires_retrieval": True},
        ]
        chains = qa.identify_reasoning_chains(decomposed)
        # Single question doesn't form a chain
        assert chains == []

    def test_two_independent_chains(self):
        qa = self._make_qa_stub()
        decomposed = [
            {"question": "Who directed A?", "requires_retrieval": True},
            {"question": "When was <ENTITY_Q1> born?", "requires_retrieval": True},
            {"question": "Who wrote B?", "requires_retrieval": True},
            {"question": "When was <ENTITY_Q3> born?", "requires_retrieval": True},
        ]
        chains = qa.identify_reasoning_chains(decomposed)
        assert len(chains) == 2


class TestSubstituteFromState:
    def test_indexed_placeholder(self):
        result = ChainFollowingQA._substitute_from_state(
            "Who is <ENTITY_Q1>'s spouse?",
            {0: "Michael Curtiz"},
        )
        assert result == "Who is Michael Curtiz's spouse?"

    def test_multiple_placeholders(self):
        result = ChainFollowingQA._substitute_from_state(
            "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            {0: "Denis Villeneuve", 1: "Christopher Nolan"},
        )
        assert "Denis Villeneuve" in result
        assert "Christopher Nolan" in result

    def test_generic_placeholder(self):
        result = ChainFollowingQA._substitute_from_state(
            "When was <ENTITY> born?",
            {0: "Alice"},
        )
        assert result == "When was Alice born?"

    def test_no_placeholders(self):
        result = ChainFollowingQA._substitute_from_state(
            "Who directed Casablanca?",
            {},
        )
        assert result == "Who directed Casablanca?"

    def test_list_entity_value(self):
        result = ChainFollowingQA._substitute_from_state(
            "What is <ENTITY_Q1>?",
            {0: ["Alice", "Bob"]},
        )
        assert result == "What is Alice?"


class TestScoringFunctions:
    @staticmethod
    def _make_qa_stub(mode="cumulative", alpha=0.5):
        qa = object.__new__(ChainFollowingQA)
        qa.scoring_mode = mode
        qa.alpha = alpha
        qa.gsw_tools = None
        return qa

    def test_cumulative_score_basic(self):
        qa = self._make_qa_stub()
        state = {
            "evidence_pairs": [
                {"similarity_score": 0.8},
                {"similarity_score": 0.6},
            ]
        }
        score = qa._compute_cumulative_score(state)
        assert 0.0 < score <= 1.0

    def test_cumulative_score_empty(self):
        qa = self._make_qa_stub()
        state = {"evidence_pairs": []}
        score = qa._compute_cumulative_score(state)
        assert score < 0.001  # eps

    def test_score_chain_state_cumulative(self):
        qa = self._make_qa_stub(mode="cumulative")
        state = {"evidence_pairs": [{"similarity_score": 0.9}]}
        state = qa._score_chain_state(state)
        assert "chain_score" in state
        assert state["chain_score"] > 0

    def test_score_chain_state_none_mode(self):
        qa = self._make_qa_stub(mode="none")
        state = {"evidence_pairs": [{"similarity_score": 0.9}]}
        state = qa._score_chain_state(state)
        assert state["chain_score"] == 1.0


class TestBeamPruning:
    def test_prune_basic(self):
        candidates = [
            {"chain_score": 0.5, "last_hop_score": 0.1},
            {"chain_score": 0.9, "last_hop_score": 0.2},
            {"chain_score": 0.3, "last_hop_score": 0.3},
        ]
        result = ChainFollowingQA._prune_to_beam_width(candidates, 2)
        assert len(result) == 2
        assert result[0]["chain_score"] == 0.9

    def test_prune_empty(self):
        assert ChainFollowingQA._prune_to_beam_width([], 5) == []

    def test_prune_fewer_than_beam(self):
        candidates = [{"chain_score": 0.5, "last_hop_score": 0.1}]
        result = ChainFollowingQA._prune_to_beam_width(candidates, 5)
        assert len(result) == 1


class TestHarmonicMean:
    def test_basic(self):
        result = ChainFollowingQA._harmonic_mean([0.5, 0.5])
        assert abs(result - 0.5) < 0.01

    def test_penalizes_low_outlier(self):
        high = ChainFollowingQA._harmonic_mean([0.9, 0.9])
        mixed = ChainFollowingQA._harmonic_mean([0.9, 0.1])
        assert high > mixed

    def test_empty(self):
        result = ChainFollowingQA._harmonic_mean([])
        assert result < 0.001


class TestExtractEvidence:
    def test_basic_extraction(self):
        beams = [
            {
                "evidence_pairs": [
                    {
                        "question": "Who?",
                        "answer_names": ["Alice"],
                        "answer_rolestates": [],
                    },
                    {
                        "question": "Where?",
                        "answer_names": ["London"],
                        "answer_rolestates": ["Location: active"],
                    },
                ]
            }
        ]
        evidence = ChainFollowingQA._extract_evidence_from_beams(beams)
        assert len(evidence) == 2
        assert "Q: Who? A: Alice" in evidence[0]
        assert "Location: active" in evidence[1]

    def test_deduplication(self):
        beams = [
            {
                "evidence_pairs": [
                    {
                        "question": "Who?",
                        "answer_names": ["Alice"],
                        "answer_rolestates": [],
                    },
                ]
            },
            {
                "evidence_pairs": [
                    {
                        "question": "Who?",
                        "answer_names": ["Alice"],
                        "answer_rolestates": [],
                    },
                ]
            },
        ]
        evidence = ChainFollowingQA._extract_evidence_from_beams(beams)
        assert len(evidence) == 1


class TestCreateExpansionState:
    def test_basic(self):
        base = {"entities_by_qidx": {}, "evidence_pairs": [], "score": 0.0}
        qa = {"question": "Who?", "answer_names": ["Alice"], "similarity_score": 0.8}
        state = ChainFollowingQA._create_expansion_state(base, qa, q_idx=0)
        assert state["entities_by_qidx"][0] == "Alice"
        assert len(state["evidence_pairs"]) == 1
        assert state["last_hop_score"] == 0.8

    def test_last_hop_keeps_all_answers(self):
        base = {"entities_by_qidx": {}, "evidence_pairs": [], "score": 0.0}
        qa = {
            "question": "Who?",
            "answer_names": ["Alice", "Bob"],
            "similarity_score": 0.5,
        }
        state = ChainFollowingQA._create_expansion_state(
            base, qa, q_idx=0, is_last_hop=True
        )
        assert state["entities_by_qidx"][0] == ["Alice", "Bob"]
