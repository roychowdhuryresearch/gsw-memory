from gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler
from gsw_memory.sleep_time.prompts import SLEEP_TIME_SYSTEM_PROMPT
from gsw_memory.sleep_time.tools import GSWTools


class _DummySearcher:
    def __init__(self):
        self.gsw_by_doc_id = {}
        self.entities = [
            {"name": "Master Aldric", "doc_id": "doc_5"},
            {"name": "Lady Margaux", "doc_id": "doc_5"},
            {"name": "Lady Margaux", "doc_id": "doc_11"},
            {"name": "Prior Anselm", "doc_id": "doc_5"},
            {"name": "Prior Anselm", "doc_id": "doc_18"},
        ]

    def search(self, query, top_k=10, verbose=False):
        q = query.strip().lower()
        if q == "lady margaux":
            return [
                ({"name": "Lady Margaux", "doc_id": "doc_11", "roles": [], "all_states": []}, 0.95),
                ({"name": "Lady Margaux of Bruges", "doc_id": "doc_22", "roles": [], "all_states": []}, 0.72),
            ]
        if q == "prior anselm":
            return [
                ({"name": "Prior Anselm", "doc_id": "doc_18", "roles": [], "all_states": []}, 0.93),
            ]
        return []


def _build_doc_plan():
    return {
        "doc_id": "doc_5",
        "entities_to_explore": [
            {
                "name": "Master Aldric",
                "roles": ["blacksmith"],
                "status": "pending",
                "neighbors": [
                    {
                        "entity": "Lady Margaux",
                        "relationship": "commissioned by",
                        "other_docs": ["doc_11"],
                        "status": "pending",
                    },
                    {
                        "entity": "Prior Anselm",
                        "relationship": "forged for",
                        "other_docs": ["doc_18"],
                        "status": "pending",
                    },
                ],
            }
        ],
        "total_neighbors_to_explore": 2,
        "explored_count": 0,
        "pending_count": 2,
    }


def _new_tools():
    tools = GSWTools(_DummySearcher())
    tools.doc_exploration_plans["doc_5"] = _build_doc_plan()
    return tools


def _new_agent(monkeypatch):
    def _fake_init(self, model_name, base_url=None):
        self.provider = "openai"
        self.client = None
        self.litellm = None

    monkeypatch.setattr(AgenticReconciler, "_initialize_client", _fake_init)
    agent = AgenticReconciler(
        entity_searcher=_DummySearcher(),
        model_name="gpt-4o-mini",
        verbose=False,
        bridge_verifier_enabled=False,
    )
    agent.tools.doc_exploration_plans["doc_5"] = _build_doc_plan()
    agent._current_doc_exploration_doc_id = "doc_5"
    return agent


def test_begin_neighbor_focus_succeeds_for_pending_edge():
    tools = _new_tools()
    result = tools.begin_neighbor_focus("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert result["success"] is True
    assert result["active_focus"]["doc_id"] == "doc_5"
    assert result["active_focus"]["entity_name"] == "Master Aldric"
    assert result["active_focus"]["neighbor_name"] == "Lady Margaux"
    assert result["active_focus"]["relationship"] == "commissioned by"


def test_begin_neighbor_focus_fails_for_already_explored_edge():
    tools = _new_tools()
    tools.doc_exploration_plans["doc_5"]["entities_to_explore"][0]["neighbors"][0]["status"] = "explored"
    result = tools.begin_neighbor_focus("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert "error" in result
    assert "Pending edge not found" in result["error"]


def test_mark_neighbor_explored_marks_exact_relationship_only():
    tools = GSWTools(_DummySearcher())
    tools.doc_exploration_plans["doc_5"] = {
        "doc_id": "doc_5",
        "entities_to_explore": [
            {
                "name": "Lothair II",
                "roles": [],
                "status": "pending",
                "neighbors": [
                    {
                        "entity": "Ermengarde of Tours",
                        "relationship": "son of",
                        "other_docs": ["doc_1"],
                        "status": "pending",
                    },
                    {
                        "entity": "Ermengarde of Tours",
                        "relationship": "allied with",
                        "other_docs": ["doc_2"],
                        "status": "pending",
                    },
                ],
            }
        ],
        "total_neighbors_to_explore": 2,
        "explored_count": 0,
        "pending_count": 2,
    }

    result = tools.mark_neighbor_explored(
        "doc_5",
        "Lothair II",
        "Ermengarde of Tours",
        relationship="allied with",
        bridges_created=1,
    )
    assert result["marked"] == ["Ermengarde of Tours"]

    neighbors = tools.doc_exploration_plans["doc_5"]["entities_to_explore"][0]["neighbors"]
    assert neighbors[0]["status"] == "pending"
    assert neighbors[1]["status"] == "explored"
    assert neighbors[1]["bridges_created"] == 1


def test_marking_active_edge_clears_focus_lock():
    tools = _new_tools()
    begin = tools.begin_neighbor_focus("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert begin["success"] is True
    assert tools.active_neighbor_focus is not None

    coverage = tools.plan_neighbor_doc_coverage(
        "doc_5",
        "Master Aldric",
        "Lady Margaux",
        "commissioned by",
        include_fuzzy=True,
    )
    assert coverage["mandatory_docs"] == ["doc_11"]
    assert coverage["ready_to_close"] is False
    tools.get_entity_context("Lady Margaux", "doc_11")

    mark = tools.mark_neighbor_explored(
        "doc_5",
        "Master Aldric",
        "Lady Margaux",
        relationship="commissioned by",
        bridges_created=2,
    )
    assert mark["marked"] == ["Lady Margaux"]
    assert tools.active_neighbor_focus is None


def test_plan_neighbor_doc_coverage_builds_exact_mandatory_and_optional_fuzzy():
    tools = _new_tools()
    begin = tools.begin_neighbor_focus("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert begin["success"] is True

    coverage = tools.plan_neighbor_doc_coverage(
        "doc_5",
        "Master Aldric",
        "Lady Margaux",
        "commissioned by",
        include_fuzzy=True,
        top_k=10,
    )
    assert coverage["mandatory_docs"] == ["doc_11"]
    assert "doc_5" not in coverage["mandatory_docs"]
    assert coverage["pending_mandatory_docs"] == ["doc_11"]
    assert coverage["ready_to_close"] is False
    assert any(item["doc_id"] == "doc_22" for item in coverage["optional_fuzzy_docs"])


def test_mandatory_coverage_progression_updates_on_get_entity_context():
    tools = _new_tools()
    begin = tools.begin_neighbor_focus("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert begin["success"] is True
    tools.plan_neighbor_doc_coverage("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")

    before = tools.get_neighbor_doc_coverage_status("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert before["pending_mandatory_docs"] == ["doc_11"]
    assert before["ready_to_close"] is False

    tools.get_entity_context("Lady Margaux", "doc_11")

    after = tools.get_neighbor_doc_coverage_status("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert after["explored_mandatory_docs"] == ["doc_11"]
    assert after["pending_mandatory_docs"] == []
    assert after["ready_to_close"] is True


def test_get_entity_context_uses_optional_doc_alias_under_active_focus(monkeypatch, caplog):
    tools = _new_tools()
    tools.active_neighbor_focus = {
        "doc_id": "doc_9",
        "entity_name": "Bertha",
        "neighbor_name": "Hugh of Italy",
        "relationship": "had children",
    }
    edge_key = tools._active_focus_edge_key()
    tools.neighbor_doc_coverage_plans[edge_key] = {
        "doc_id": "doc_9",
        "entity_name": "Bertha",
        "neighbor_name": "Hugh of Italy",
        "relationship": "had children",
        "mandatory_docs": [],
        "optional_fuzzy_docs": [
            {"doc_id": "doc_2", "name": "Hugh, King of Italy", "score": 7.1434},
        ],
        "explored_mandatory_docs": [],
        "include_fuzzy": True,
        "top_k": 10,
    }

    calls = []

    def _fake_single_context(entity_name, doc_id):
        calls.append((entity_name, doc_id))
        if entity_name == "Hugh, King of Italy" and doc_id == "doc_2":
            return {
                "entity": entity_name,
                "doc_id": doc_id,
                "qa_pairs": [{"question": "Who ruled Italy?", "answer": "Hugh", "answer_refs": ["doc_2::e1"], "doc_id": doc_id}],
                "roles": ["person"],
                "states": [],
                "relationships": {"ruled": ["Italy"]},
            }
        return {
            "entity": entity_name,
            "doc_id": doc_id,
            "qa_pairs": [],
            "roles": [],
            "states": [],
            "relationships": {},
        }

    monkeypatch.setattr(tools, "_get_single_doc_context", _fake_single_context)

    with caplog.at_level("INFO"):
        context = tools.get_entity_context("Hugh of Italy", "doc_2")

    assert context["qa_pairs"]
    assert calls == [("Hugh of Italy", "doc_2"), ("Hugh, King of Italy", "doc_2")]
    assert any("Optional-doc alias fallback used" in rec.message for rec in caplog.records)


def test_get_entity_context_does_not_use_optional_alias_without_active_focus(monkeypatch):
    tools = _new_tools()
    calls = []

    def _fake_single_context(entity_name, doc_id):
        calls.append((entity_name, doc_id))
        if entity_name == "Hugh, King of Italy" and doc_id == "doc_2":
            return {
                "entity": entity_name,
                "doc_id": doc_id,
                "qa_pairs": [{"question": "Who ruled Italy?", "answer": "Hugh", "answer_refs": ["doc_2::e1"], "doc_id": doc_id}],
                "roles": ["person"],
                "states": [],
                "relationships": {"ruled": ["Italy"]},
            }
        return {
            "entity": entity_name,
            "doc_id": doc_id,
            "qa_pairs": [],
            "roles": [],
            "states": [],
            "relationships": {},
        }

    monkeypatch.setattr(tools, "_get_single_doc_context", _fake_single_context)
    context = tools.get_entity_context("Hugh of Italy", "doc_2")

    assert context["qa_pairs"] == []
    assert calls == [("Hugh of Italy", "doc_2")]


def test_mark_neighbor_explored_blocked_when_mandatory_docs_pending():
    tools = _new_tools()
    begin = tools.begin_neighbor_focus("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")
    assert begin["success"] is True
    tools.plan_neighbor_doc_coverage("doc_5", "Master Aldric", "Lady Margaux", "commissioned by")

    mark = tools.mark_neighbor_explored(
        "doc_5",
        "Master Aldric",
        "Lady Margaux",
        relationship="commissioned by",
        bridges_created=0,
    )
    assert mark["stage"] == "coverage_guard"
    assert mark["pending_mandatory_docs"] == ["doc_11"]


def test_plan_corpus_exploration_reflects_doc_status_and_next_doc():
    tools = _new_tools()
    # Seed corpus docs for queue behavior.
    tools.entity_searcher.gsw_by_doc_id = {"doc_2": object(), "doc_5": object(), "doc_9": object()}
    tools.doc_exploration_plans["doc_9"] = {
        "doc_id": "doc_9",
        "entities_to_explore": [
            {"name": "X", "status": "pending", "neighbors": [{"entity": "Y", "relationship": "related", "other_docs": [], "status": "pending"}]}
        ],
        "total_neighbors_to_explore": 1,
        "explored_count": 0,
        "pending_count": 1,
    }
    tools.mark_document_explored("doc_5", num_bridges_created=3)

    queue = tools.plan_corpus_exploration(strategy="max_pending_neighbors", limit=10, include_completed=False)
    assert queue["total_docs"] == 3
    docs = [item["doc_id"] for item in queue["queue"]]
    assert "doc_5" not in docs  # completed excluded
    assert queue["next_doc"]["doc_id"] in {"doc_9", "doc_2"}


def test_focus_guard_blocks_get_entity_context_without_lock(monkeypatch):
    agent = _new_agent(monkeypatch)
    agent.tools.active_neighbor_focus = None
    result = agent._execute_tool("get_entity_context", {"entity_name": "Lady Margaux", "doc_id": "doc_11"})
    assert result["stage"] == "focus_guard"
    assert "begin_neighbor_focus" in result["hint"]


def test_focus_guard_blocks_wrong_entity_context_while_locked(monkeypatch):
    agent = _new_agent(monkeypatch)
    agent.tools.active_neighbor_focus = {
        "doc_id": "doc_5",
        "entity_name": "Master Aldric",
        "neighbor_name": "Lady Margaux",
        "relationship": "commissioned by",
    }
    result = agent._execute_tool("get_entity_context", {"entity_name": "Prior Anselm", "doc_id": "doc_18"})
    assert result["stage"] == "focus_guard"
    assert "active focused neighbor" in result["error"]


def test_focus_guard_blocks_batch_doc_context_while_locked(monkeypatch):
    agent = _new_agent(monkeypatch)
    agent.tools.active_neighbor_focus = {
        "doc_id": "doc_5",
        "entity_name": "Master Aldric",
        "neighbor_name": "Lady Margaux",
        "relationship": "commissioned by",
    }
    result = agent._execute_tool("get_entity_context", {"entity_name": "Lady Margaux", "doc_id": ["doc_11", "doc_22"]})
    assert result["stage"] == "focus_guard"
    assert "single doc_id string" in result["error"]


def test_focus_guard_blocks_unrelated_search_query_while_locked(monkeypatch):
    agent = _new_agent(monkeypatch)
    agent.tools.active_neighbor_focus = {
        "doc_id": "doc_5",
        "entity_name": "Master Aldric",
        "neighbor_name": "Lady Margaux",
        "relationship": "commissioned by",
    }
    result = agent._execute_tool("search_entities", {"query": "Prior Anselm", "exclude_doc_id": "doc_5"})
    assert result["stage"] == "focus_guard"
    assert "must match the active focused neighbor" in result["error"]


def test_focus_guard_blocks_plan_neighbor_doc_coverage_without_lock(monkeypatch):
    agent = _new_agent(monkeypatch)
    agent.tools.active_neighbor_focus = None
    result = agent._execute_tool(
        "plan_neighbor_doc_coverage",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Lady Margaux",
            "relationship": "commissioned by",
        },
    )
    assert result["stage"] == "focus_guard"
    assert "requires an active neighbor focus lock" in result["error"]


def test_focus_guard_blocks_plan_neighbor_doc_coverage_for_non_active_edge(monkeypatch):
    agent = _new_agent(monkeypatch)
    agent.tools.active_neighbor_focus = {
        "doc_id": "doc_5",
        "entity_name": "Master Aldric",
        "neighbor_name": "Lady Margaux",
        "relationship": "commissioned by",
    }
    result = agent._execute_tool(
        "plan_neighbor_doc_coverage",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Prior Anselm",
            "relationship": "forged for",
        },
    )
    assert result["stage"] == "focus_guard"
    assert "must match active focused neighbor" in result["error"]


def test_focus_guard_blocks_plan_corpus_exploration_while_locked(monkeypatch):
    agent = _new_agent(monkeypatch)
    agent.tools.active_neighbor_focus = {
        "doc_id": "doc_5",
        "entity_name": "Master Aldric",
        "neighbor_name": "Lady Margaux",
        "relationship": "commissioned by",
    }
    result = agent._execute_tool("plan_corpus_exploration", {})
    assert result["stage"] == "focus_guard"
    assert "blocked while neighbor focus lock is active" in result["error"]


def test_focus_guard_blocks_mark_when_coverage_incomplete(monkeypatch):
    agent = _new_agent(monkeypatch)

    begin = agent._execute_tool(
        "begin_neighbor_focus",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Lady Margaux",
            "relationship": "commissioned by",
        },
    )
    assert begin["success"] is True
    coverage = agent._execute_tool(
        "plan_neighbor_doc_coverage",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Lady Margaux",
            "relationship": "commissioned by",
        },
    )
    assert coverage["ready_to_close"] is False

    result = agent._execute_tool(
        "mark_neighbor_explored",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Lady Margaux",
            "relationship": "commissioned by",
            "bridges_created": 0,
        },
    )
    assert result["stage"] == "coverage_guard"
    assert "mandatory neighbor docs" in result["error"]


def test_mark_releases_lock_then_next_begin_allowed(monkeypatch):
    agent = _new_agent(monkeypatch)

    begin_one = agent._execute_tool(
        "begin_neighbor_focus",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Lady Margaux",
            "relationship": "commissioned by",
        },
    )
    assert begin_one["success"] is True
    assert agent.tools.active_neighbor_focus is not None

    coverage = agent._execute_tool(
        "plan_neighbor_doc_coverage",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Lady Margaux",
            "relationship": "commissioned by",
        },
    )
    assert coverage["mandatory_docs"] == ["doc_11"]

    # Reading a mandatory doc should unlock close gating for this edge.
    _ = agent._execute_tool("get_entity_context", {"entity_name": "Lady Margaux", "doc_id": "doc_11"})

    mark = agent._execute_tool(
        "mark_neighbor_explored",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Lady Margaux",
            "relationship": "commissioned by",
            "bridges_created": 1,
        },
    )
    assert mark["marked"] == ["Lady Margaux"]
    assert agent.tools.active_neighbor_focus is None

    begin_two = agent._execute_tool(
        "begin_neighbor_focus",
        {
            "doc_id": "doc_5",
            "entity_name": "Master Aldric",
            "neighbor_name": "Prior Anselm",
            "relationship": "forged for",
        },
    )
    assert begin_two["success"] is True
    assert begin_two["active_focus"]["neighbor_name"] == "Prior Anselm"


def test_prompt_contains_neighbor_lock_protocol():
    assert "Neighbor Lock Protocol" in SLEEP_TIME_SYSTEM_PROMPT
    assert "begin_neighbor_focus" in SLEEP_TIME_SYSTEM_PROMPT
    assert "plan_neighbor_doc_coverage" in SLEEP_TIME_SYSTEM_PROMPT
    assert "plan_corpus_exploration" in SLEEP_TIME_SYSTEM_PROMPT
    assert "Do NOT use batch context" in SLEEP_TIME_SYSTEM_PROMPT
    assert "get_entity_context(neighbor, [other_doc_ids])" not in SLEEP_TIME_SYSTEM_PROMPT
