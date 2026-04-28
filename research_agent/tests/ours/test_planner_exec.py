"""Unit tests for the GSW-fragment plan executor.

These tests never hit a real LLM or retriever. We use minimal stubs
so we can assert the pure Python DAG + dispatch layers are correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from research_agent.adapters.ours._planner_exec import (
    Constraint,
    Entity,
    ExecutionError,
    GSWPlan,
    VerbPhrase,
    build_dependency_graph,
    execute,
    topological_sort_blanks,
)


# ---------------------------------------------------------------------------
# Stubs for LLM + retriever
# ---------------------------------------------------------------------------


@dataclass
class _StubChunk:
    chunk_id: str
    title: str
    text: str


@dataclass
class _StubHit:
    chunk: _StubChunk
    score: float = 1.0


class _StubRetriever:
    def __init__(self, query_to_chunks: dict[str, list[_StubChunk]]) -> None:
        self.query_to_chunks = query_to_chunks
        self.calls: list[str] = []

    def search(self, query: str, *, top_k: int = 8) -> list[_StubHit]:
        self.calls.append(query)
        for key, chunks in self.query_to_chunks.items():
            if key in query:
                return [_StubHit(chunk=c) for c in chunks]
        return []


@dataclass
class _StubResponse:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _StubLLM:
    """Return scripted JSON blobs keyed by substring match against the
    Retrieved-chunks section of the user prompt (which is blank-specific
    — the context block above it lists every filled entity and would
    otherwise always match the first key in the dict)."""

    def __init__(self, query_to_value: dict[str, Any]) -> None:
        self.query_to_value = query_to_value
        self.calls: list[str] = []

    def chat(self, *, messages: list[dict[str, Any]], max_tokens: int = 600) -> _StubResponse:
        user = messages[-1]["content"]
        self.calls.append(user)
        # Only match against retrieved-chunks section.
        chunks_section = user.split("Retrieved chunks:", 1)[-1] if "Retrieved chunks:" in user else user
        for key, value in self.query_to_value.items():
            if key in chunks_section:
                import json

                payload = {"value": value, "evidence_chunk_ids": ["chunk_1"]}
                return _StubResponse(text=json.dumps(payload))
        return _StubResponse(text='{"value": null, "evidence_chunk_ids": []}')


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def test_topo_sort_simple_chain():
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Pablo_Picasso"),
            Entity(id="e2", kind="filled", name="Pink_Floyd"),
            Entity(id="b1", kind="blank", value_type="date"),
            Entity(id="b2", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="died_in", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="released_in_year", subject_id="e2", object_id="b1"),
            VerbPhrase(id="vp3", phrase="released", subject_id="e2", object_id="b2"),
        ],
        constraints=[],
    )
    order = topological_sort_blanks(plan)
    # b1 has no constraint deps; b2 has no constraint deps either (VP-only).
    # Both end up at indeg 0 here; whatever order is fine.
    assert set(order) == {"b1", "b2"}


def test_topo_sort_diff_chain():
    # Example B: Portland ME/OR population diff.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Portland_Maine"),
            Entity(id="e2", kind="filled", name="Portland_Oregon"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has_population", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="has_population", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b2", "b1"],
                output_blank_id="t",
            )
        ],
    )
    order = topological_sort_blanks(plan)
    # b1 and b2 must appear before t.
    assert order.index("b1") < order.index("t")
    assert order.index("b2") < order.index("t")


def test_topo_sort_argmax_chain():
    # Example C: missile argmax.
    entities = [
        Entity(id=f"e{i}", kind="filled", name=f"missile_{i}") for i in range(1, 5)
    ]
    blanks = [
        Entity(id=f"b{i}", kind="blank", value_type="date") for i in range(1, 5)
    ]
    target = Entity(id="t", kind="blank", value_type="entity", is_target=True)
    vps = [
        VerbPhrase(
            id=f"vp{i}",
            phrase="entered_service_in",
            subject_id=f"e{i}",
            object_id=f"b{i}",
        )
        for i in range(1, 5)
    ]
    plan = GSWPlan(
        entities=entities + blanks + [target],
        verb_phrases=vps,
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=[f"e{i}" for i in range(1, 5)],
                sort_by_blank_ids=[f"b{i}" for i in range(1, 5)],
                output_blank_id="t",
            )
        ],
    )
    order = topological_sort_blanks(plan)
    for b in ("b1", "b2", "b3", "b4"):
        assert order.index(b) < order.index("t")


def test_topo_sort_projection_chain():
    # Example D: b_person (identify) → b_birth_year, b_death_year (project).
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="NYC"),
            Entity(id="e2", kind="filled", name="Catwoman"),
            Entity(id="b_person", kind="blank", value_type="entity"),
            Entity(id="b_birth", kind="blank", value_type="date"),
            Entity(id="b_death", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in", subject_id="b_person", object_id="e1"),
            VerbPhrase(id="vp2", phrase="created", subject_id="b_person", object_id="e2"),
            VerbPhrase(
                id="vp3", phrase="born_in_year", subject_id="b_person", object_id="b_birth"
            ),
            VerbPhrase(
                id="vp4", phrase="died_in_year", subject_id="b_person", object_id="b_death"
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b_death", "b_birth"],
                output_blank_id="t",
            )
        ],
    )
    order = topological_sort_blanks(plan)
    # b_person (entity blank) must come before b_birth and b_death
    # (projection VPs). b_birth and b_death must come before t.
    assert order.index("b_person") < order.index("b_birth")
    assert order.index("b_person") < order.index("b_death")
    assert order.index("b_birth") < order.index("t")
    assert order.index("b_death") < order.index("t")


def test_topo_sort_cycle_raises():
    plan = GSWPlan(
        entities=[
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c1", kind="derived", op="sum", args_blanks=["b2"], output_blank_id="b1"
            ),
            Constraint(
                id="c2", kind="derived", op="sum", args_blanks=["b1"], output_blank_id="b2"
            ),
        ],
    )
    with pytest.raises(ExecutionError) as excinfo:
        topological_sort_blanks(plan)
    assert excinfo.value.kind == "cyclic_plan"


# ---------------------------------------------------------------------------
# execute() — derived ops
# ---------------------------------------------------------------------------


def test_execute_diff_numeric():
    # Example B fully executed with stubs.
    # Names have spaces (not underscores) so they survive
    # _identify_blank's underscore→space normalisation.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Portland Maine"),
            Entity(id="e2", kind="filled", name="Portland Oregon"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has population", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="has population", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b2", "b1"],
                output_blank_id="t",
            )
        ],
    )
    # Use non-overlapping substrings so _StubLLM's first-match rule works.
    retriever = _StubRetriever(
        {
            "Maine": [_StubChunk("ME_chunk", "Portland, Maine", "pop 68408")],
            "Oregon": [_StubChunk("OR_chunk", "Portland, Oregon", "pop 152920")],
        }
    )
    llm = _StubLLM(
        {
            "Maine": 68408,
            "Oregon": 152920,
        }
    )
    final, trace = execute(plan, retriever=retriever, llm_client=llm, top_k=5)
    assert final == "84512"
    assert len(trace.executed_blanks) == 3
    # Last trace call should be the derived constraint, not an LLM extract.
    assert trace.tool_calls[-1].name == "constraint:derived"


def test_execute_diff_uses_years_from_date_strings():
    """Year-difference constraints should still fire when upstream date
    blanks store full dates. FRAMES questions often ask "how many years"
    while evidence snippets contain day/month/year dates."""
    plan = GSWPlan(
        entities=[
            Entity(id="e_birth", kind="filled", name="Ben Darwin"),
            Entity(id="e_opened", kind="filled", name="Australian Institute of Sport"),
            Entity(id="b_birth", kind="blank", value_type="date"),
            Entity(id="b_opened", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in_year", subject_id="e_birth", object_id="b_birth"),
            VerbPhrase(id="vp2", phrase="opened_in_year", subject_id="e_opened", object_id="b_opened"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b_opened", "b_birth"],
                output_blank_id="t",
            )
        ],
    )
    retriever = _StubRetriever(
        {
            "Ben Darwin": [_StubChunk("birth_chunk", "Ben Darwin", "born 17 October 1976")],
            "Australian Institute": [
                _StubChunk("ais_chunk", "Australian Institute of Sport", "headquarters opened in 1981")
            ],
        }
    )
    llm = _StubLLM(
        {
            "Ben Darwin": "1976-10-17",
            "Australian Institute": "1981-01-01",
        }
    )
    final, trace = execute(plan, retriever=retriever, llm_client=llm, top_k=5)
    assert final == "5"
    assert trace.tool_calls[-1].name == "constraint:derived"


def test_execute_sum_digits_from_text_postcode():
    """A sum over a text postcode should add the postcode digits, not
    leave the target unresolved."""
    plan = GSWPlan(
        entities=[
            Entity(id="e_hospital", kind="filled", name="Liverpool Maternity Hospital"),
            Entity(id="b_postcode", kind="blank", value_type="text"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(
                id="vp1",
                phrase="has_postcode",
                subject_id="e_hospital",
                object_id="b_postcode",
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="sum",
                args_blanks=["b_postcode"],
                output_blank_id="t",
            )
        ],
    )
    retriever = _StubRetriever(
        {
            "Liverpool Maternity Hospital": [
                _StubChunk("postcode_chunk", "Liverpool Maternity Hospital", "postcode L7 7AB")
            ],
        }
    )
    llm = _StubLLM({"postcode": "L7 7AB"})
    final, trace = execute(plan, retriever=retriever, llm_client=llm, top_k=5)
    assert final == "14"
    assert trace.tool_calls[-1].name == "constraint:derived"


def test_execute_argmax_entity_selector():
    # Use non-overlapping unique substrings as entity names so stubs match.
    names = ["Alpha", "Bravo", "Charlie", "Delta"]
    years = {"Alpha": "1996", "Bravo": "1968", "Charlie": "1992", "Delta": "1981"}
    entities = [
        Entity(id=f"e{i}", kind="filled", name=names[i - 1]) for i in range(1, 5)
    ]
    blanks = [Entity(id=f"b{i}", kind="blank", value_type="date") for i in range(1, 5)]
    target = Entity(id="t", kind="blank", value_type="entity", is_target=True)
    plan = GSWPlan(
        entities=entities + blanks + [target],
        verb_phrases=[
            VerbPhrase(
                id=f"vp{i}",
                phrase="entered service in",
                subject_id=f"e{i}",
                object_id=f"b{i}",
            )
            for i in range(1, 5)
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=[f"e{i}" for i in range(1, 5)],
                sort_by_blank_ids=[f"b{i}" for i in range(1, 5)],
                output_blank_id="t",
            )
        ],
    )
    retriever = _StubRetriever(
        {name: [_StubChunk(f"{name}_chunk", name, f"entered {years[name]}")] for name in names}
    )
    llm = _StubLLM(years)
    final, _trace = execute(plan, retriever=retriever, llm_client=llm, top_k=5)
    assert final == "Alpha"  # 1996 is max


def test_execute_unknown_cascades_to_target():
    # If a retrieval returns nothing, target must be blank (unknown), not fabricated.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="UnknownEntity"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has_attribute", subject_id="e1", object_id="b1"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="sum",
                args_blanks=["b1"],
                output_blank_id="t",
            )
        ],
    )
    retriever = _StubRetriever({})  # every query returns zero chunks
    llm = _StubLLM({})  # always returns null
    final, trace = execute(plan, retriever=retriever, llm_client=llm, top_k=5)
    assert final == ""  # empty string for unresolved target


def test_execute_no_target_raises():
    # Wire the blanks with a VP so the no-dangling validator passes,
    # but leave is_target unset so execute() still raises no_target.
    plan = GSWPlan(
        entities=[
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="relates_to", subject_id="b1", object_id="b2"),
        ],
    )
    with pytest.raises(ExecutionError) as excinfo:
        execute(plan, retriever=_StubRetriever({}), llm_client=_StubLLM({}))
    assert excinfo.value.kind == "no_target"


def test_execute_empty_plan_raises():
    plan = GSWPlan(entities=[])
    with pytest.raises(ExecutionError) as excinfo:
        execute(plan, retriever=_StubRetriever({}), llm_client=_StubLLM({}))
    assert excinfo.value.kind == "empty_plan"


# ---------------------------------------------------------------------------
# Prompt-v2 additions: role fields + no-dangling validator
# ---------------------------------------------------------------------------


def test_entity_role_and_state_fields_accepted():
    """Plans that include `role` (and optionally `state`) on entities
    should validate cleanly; the executor ignores the fields but the
    schema must round-trip them."""
    plan = GSWPlan(
        entities=[
            Entity(
                id="e1",
                kind="filled",
                name="Pablo Picasso",
                role="subject",
                state="alive_in_question_context",
            ),
            Entity(id="e2", kind="filled", name="Pink Floyd", role="subject"),
            Entity(id="b1", kind="blank", value_type="date", role="bridge-date"),
            Entity(
                id="t",
                kind="blank",
                value_type="entity",
                is_target=True,
                role="target",
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="died_in", subject_id="e1", object_id="b1"),
            VerbPhrase(
                id="vp2", phrase="released_in_year", subject_id="e2", object_id="b1"
            ),
            VerbPhrase(id="vp3", phrase="released", subject_id="e2", object_id="t"),
        ],
    )
    # Round-trip via model_dump / model_validate to ensure fields persist.
    round_tripped = GSWPlan.model_validate(plan.model_dump())
    assert round_tripped.entity_by_id("e1").role == "subject"
    assert round_tripped.entity_by_id("e1").state == "alive_in_question_context"
    assert round_tripped.entity_by_id("t").role == "target"


def test_dangling_entity_rejected():
    """A plan whose entity appears in NO verb-phrase and NO constraint
    reference must raise a ValidationError that names the dangling id."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="Used Entity", role="subject"),
                Entity(id="e_dangles", kind="filled", name="Orphan", role="subject"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="entity",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="relates_to", subject_id="e1", object_id="t"),
            ],
        )
    # The error message should mention the dangling id.
    assert "e_dangles" in str(excinfo.value)


def test_dangling_validator_accepts_constraint_only_references():
    """Entities referenced only by a constraint (never by a VP) must
    count as non-dangling. This covers argmax target blanks whose only
    incoming edge is the constraint's output_blank_id."""
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="A", role="candidate"),
            Entity(id="e2", kind="filled", name="B", role="candidate"),
            Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
            Entity(id="b2", kind="blank", value_type="number", role="bridge-number"),
            Entity(
                id="t",
                kind="blank",
                value_type="entity",
                is_target=True,
                role="target",
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="has_x", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=["e1", "e2"],
                sort_by_blank_ids=["b1", "b2"],
                output_blank_id="t",
            )
        ],
    )
    # Simply constructing without raising is the assertion.
    assert plan.target().id == "t"


def test_build_dependency_graph_respects_vp_projection():
    # Ensure that a blank-entity → blank VP creates a dep edge.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="NYC"),
            Entity(id="b_person", kind="blank", value_type="entity"),
            Entity(id="b_age", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in", subject_id="b_person", object_id="e1"),
            VerbPhrase(id="vp2", phrase="has_age", subject_id="b_person", object_id="b_age"),
        ],
    )
    deps = build_dependency_graph(plan)
    assert "b_person" in deps["b_age"]
    assert deps["b_person"] == set()


# ---------------------------------------------------------------------------
# Phase-3.6.2 — per-kind constraint-shape validation
#
# A constraint missing its required input fields is uncomputable. The
# planner emit-repair loop relies on these ValueErrors to surface a
# concrete problem to the LLM for retry.
# ---------------------------------------------------------------------------


def _base_entities_two_candidates() -> list[Entity]:
    return [
        Entity(id="e1", kind="filled", name="A", role="candidate"),
        Entity(id="e2", kind="filled", name="B", role="candidate"),
        Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
        Entity(id="b2", kind="blank", value_type="number", role="bridge-number"),
        Entity(
            id="t",
            kind="blank",
            value_type="entity",
            is_target=True,
            role="target",
        ),
    ]


def _base_vps_two_candidates() -> list[VerbPhrase]:
    return [
        VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
        VerbPhrase(id="vp2", phrase="has_x", subject_id="e2", object_id="b2"),
    ]


def test_constraint_argmax_without_candidates_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=_base_entities_two_candidates(),
            verb_phrases=_base_vps_two_candidates(),
            constraints=[
                Constraint(
                    id="c1",
                    kind="argmax",
                    candidate_entity_ids=[],  # empty — invalid
                    sort_by_blank_ids=["b1", "b2"],
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "candidate_entity_ids" in msg
    assert "argmax" in msg


def test_constraint_argmin_without_sort_by_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=_base_entities_two_candidates(),
            verb_phrases=_base_vps_two_candidates(),
            constraints=[
                Constraint(
                    id="c1",
                    kind="argmin",
                    candidate_entity_ids=["e1", "e2"],
                    sort_by_blank_ids=[],  # empty — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "sort_by_blank_ids" in msg
    assert "argmin" in msg


def test_constraint_derived_with_empty_args_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="number",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="derived",
                    op="diff",
                    args_blanks=[],  # empty — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "args_blanks" in msg
    assert "derived" in msg


def test_constraint_gt_without_left_ref_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="bool",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="gt",
                    left_ref=None,  # missing — invalid
                    right_ref="b1",
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "left_ref" in msg or "right_ref" in msg
    assert "gt" in msg


def test_constraint_equals_without_right_ref_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="text", role="bridge-text"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="bool",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="equals",
                    left_ref="b1",
                    right_ref=None,  # missing — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "left_ref" in msg or "right_ref" in msg
    assert "equals" in msg


def test_constraint_in_list_with_empty_args_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="list", role="bridge-list"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="bool",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="in_list",
                    args_blanks=[],  # empty — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "args_blanks" in msg
    assert "in_list" in msg


def test_constraint_argmax_with_full_shape_accepted():
    """A well-formed argmax constraint must validate without errors."""
    plan = GSWPlan(
        entities=_base_entities_two_candidates(),
        verb_phrases=_base_vps_two_candidates(),
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=["e1", "e2"],
                sort_by_blank_ids=["b1", "b2"],
                output_blank_id="t",
            )
        ],
    )
    assert plan.constraints[0].kind == "argmax"


def test_constraint_derived_with_args_accepted():
    """A well-formed derived constraint must validate without errors."""
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="A", role="subject"),
            Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
            Entity(id="b2", kind="blank", value_type="number", role="bridge-number"),
            Entity(
                id="t",
                kind="blank",
                value_type="number",
                is_target=True,
                role="target",
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="x", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="y", subject_id="e1", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b1", "b2"],
                output_blank_id="t",
            )
        ],
    )
    assert plan.constraints[0].args_blanks == ["b1", "b2"]
