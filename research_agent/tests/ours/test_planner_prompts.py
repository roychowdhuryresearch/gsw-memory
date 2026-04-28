"""Shape-validation tests for the prompt v3 few-shots.

These tests do NOT call an LLM. They take the 5 synthetic few-shot
plans shipped in ``_planner_prompts._FEW_SHOTS`` and push them through
the executor's Pydantic schema to ensure they never silently ship
broken:

- Every plan validates (no dangling entities; all roles present;
  exactly one target blank).
- Few-shot #5 (collective expansion) has exactly three filled entities
  carrying the ``state="expanded_from_collective:<group>"`` breadcrumb.

If a future prompt edit breaks a few-shot, these tests fail fast.
"""

from __future__ import annotations

import pytest

from research_agent.adapters.ours._planner_exec import GSWPlan
from research_agent.adapters.ours._planner_emit import _parse_plan
from research_agent.adapters.ours._planner_prompts import (
    _FEW_SHOT_1_TEMPORAL_BRIDGE,
    _FEW_SHOT_2_AS_OF_DATE_DIFF,
    _FEW_SHOT_3_ENUMERATED_ARGMAX,
    _FEW_SHOT_4_COMPOUND_SCOPE,
    _FEW_SHOT_5_COLLECTIVE_EXPANSION,
    _FEW_SHOTS,
)


_COLLECTIVE_PREFIX = "expanded_from_collective:"


@pytest.mark.parametrize(
    "example",
    [
        _FEW_SHOT_1_TEMPORAL_BRIDGE,
        _FEW_SHOT_2_AS_OF_DATE_DIFF,
        _FEW_SHOT_3_ENUMERATED_ARGMAX,
        _FEW_SHOT_4_COMPOUND_SCOPE,
        _FEW_SHOT_5_COLLECTIVE_EXPANSION,
    ],
    ids=[
        "temporal_bridge",
        "as_of_date_diff",
        "enumerated_argmax",
        "compound_scope",
        "collective_expansion",
    ],
)
def test_few_shot_plan_validates(example: dict) -> None:
    plan = GSWPlan.model_validate(example["plan"])

    # Exactly one target blank.
    targets = [e for e in plan.blank_entities() if e.is_target]
    assert len(targets) == 1, f"expected exactly one target blank, got {len(targets)}"

    # Every entity has a role.
    missing_roles = [e.id for e in plan.entities if not (e.role or "").strip()]
    assert not missing_roles, f"entities missing role: {missing_roles}"

    # Every entity participates in at least one VP or constraint — this is
    # already enforced by the GSWPlan validator, so just confirm we
    # reached here without ValidationError.
    assert len(plan.entities) >= 2


def test_few_shots_list_has_five() -> None:
    """The prompt feeds exactly 5 few-shots into the user message."""
    assert len(_FEW_SHOTS) == 5


def test_collective_expansion_carries_state_breadcrumb() -> None:
    """Few-shot #5 must demonstrate the Hard-rule-3 escape hatch:
    every expanded member carries the ``expanded_from_collective:`` state,
    and the collective itself is present with category=true."""
    plan = GSWPlan.model_validate(_FEW_SHOT_5_COLLECTIVE_EXPANSION["plan"])

    expanded = [
        e for e in plan.entities
        if e.kind == "filled" and (e.state or "").startswith(_COLLECTIVE_PREFIX)
    ]
    assert len(expanded) == 3, f"expected 3 expanded entities, got {len(expanded)}"

    # All three point at the same collective group.
    groups = {e.state.split(":", 1)[1] for e in expanded}
    assert len(groups) == 1, f"expanded entities reference inconsistent groups: {groups}"

    # The collective itself is in the plan, with category=True.
    collectives = [
        e for e in plan.entities
        if e.kind == "filled" and e.category and groups.pop().lower() in (e.name or "").lower()
    ]
    # (groups was mutated by pop; rebuild for message clarity)
    assert collectives, "collective list-header entity not found in plan"
    assert collectives[0].role == "list-header"


def test_no_other_few_shot_uses_state_breadcrumb() -> None:
    """Shots 1-4 should NOT carry collective-expansion breadcrumbs —
    they cover the grounded cases. Catches accidental copy-paste."""
    non_expansion_shots = [
        _FEW_SHOT_1_TEMPORAL_BRIDGE,
        _FEW_SHOT_2_AS_OF_DATE_DIFF,
        _FEW_SHOT_3_ENUMERATED_ARGMAX,
        _FEW_SHOT_4_COMPOUND_SCOPE,
    ]
    for ex in non_expansion_shots:
        plan = GSWPlan.model_validate(ex["plan"])
        leaked = [
            e.id for e in plan.entities
            if (e.state or "").startswith(_COLLECTIVE_PREFIX)
        ]
        assert not leaked, (
            f"few-shot '{ex['question'][:60]}' unexpectedly carries state "
            f"breadcrumb on entities: {leaked}"
        )


def test_as_of_date_shot_has_date_anchor_role() -> None:
    """Few-shot #2 explicitly demonstrates Hard rule 2 — an
    ``as-of-date`` filled entity should be present."""
    plan = GSWPlan.model_validate(_FEW_SHOT_2_AS_OF_DATE_DIFF["plan"])
    as_of_date_roles = [
        e for e in plan.entities
        if e.kind == "filled" and e.role == "as-of-date"
    ]
    assert as_of_date_roles, "few-shot #2 must carry a filled as-of-date entity"


def test_parse_plan_rejects_ungrounded_placeholder_filled_entity() -> None:
    import json

    bad_plan = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Winner A", "role": "candidate"},
            {"id": "b1", "kind": "blank", "value_type": "number", "role": "bridge-number"},
            {"id": "t", "kind": "blank", "value_type": "entity", "role": "target", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_age", "subject_id": "e1", "object_id": "b1"}
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "argmin",
                "candidate_entity_ids": ["e1"],
                "sort_by_blank_ids": ["b1"],
                "output_blank_id": "t",
            }
        ],
    }
    with pytest.raises(ValueError) as excinfo:
        _parse_plan(
            json.dumps(bad_plan),
            question="Of the non-Americans who have won the Phoenix Open, who was youngest?",
        )
    assert "grounding test" in str(excinfo.value)


def test_temporal_bridge_shot_has_year_anchor_role() -> None:
    """Few-shot #1 demonstrates Hard rule 2 via the ``year-anchor`` role."""
    plan = GSWPlan.model_validate(_FEW_SHOT_1_TEMPORAL_BRIDGE["plan"])
    year_anchors = [
        e for e in plan.entities
        if e.kind == "filled" and e.role == "year-anchor"
    ]
    assert year_anchors, "few-shot #1 must carry a filled year-anchor entity"
