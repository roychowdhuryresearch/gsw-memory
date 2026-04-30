"""Shape-validation tests for the prompt v4 few-shots.

These tests do NOT call an LLM. They take the synthetic few-shot
plans shipped in ``_planner_prompts._FEW_SHOTS`` and push them through
the executor's Pydantic schema to ensure they never silently ship
broken:

- Every plan validates (no dangling entities; all roles present;
  exactly one target blank).
- The literal-offset few-shot uses `literal_value` + `args_refs`, not a
  fake blank for "twelve".
- No few-shot teaches placeholder collective expansion names.

If a future prompt edit breaks a few-shot, these tests fail fast.
"""

from __future__ import annotations

import pytest

from research_agent.adapters.ours._planner_exec import GSWPlan
from research_agent.adapters.ours._planner_prompts import (
    _FEW_SHOT_1_TEMPORAL_BRIDGE,
    _FEW_SHOT_2_AS_OF_DATE_DIFF,
    _FEW_SHOT_3_ENUMERATED_ARGMAX,
    _FEW_SHOT_4_COMPOUND_SCOPE,
    _FEW_SHOT_5_LITERAL_OFFSET,
    _FEW_SHOT_6_LIST_INTERSECTION,
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
        _FEW_SHOT_5_LITERAL_OFFSET,
        _FEW_SHOT_6_LIST_INTERSECTION,
    ],
    ids=[
        "temporal_bridge",
        "as_of_date_diff",
        "enumerated_argmax",
        "compound_scope",
        "literal_offset",
        "list_intersection",
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


def test_few_shots_list_has_six() -> None:
    """The prompt feeds exactly 6 few-shots into the user message."""
    assert len(_FEW_SHOTS) == 6


def test_no_few_shot_uses_state_breadcrumb() -> None:
    """Few-shots should not teach fake collective expansions."""
    for ex in _FEW_SHOTS:
        plan = GSWPlan.model_validate(ex["plan"])
        leaked = [
            e.id for e in plan.entities
            if (e.state or "").startswith(_COLLECTIVE_PREFIX)
        ]
        assert not leaked, (
            f"few-shot '{ex['question'][:60]}' unexpectedly carries state "
            f"breadcrumb on entities: {leaked}"
        )


def test_literal_offset_shot_uses_literal_value_and_args_refs() -> None:
    plan = GSWPlan.model_validate(_FEW_SHOT_5_LITERAL_OFFSET["plan"])
    literal = plan.entity_by_id("e_twelve")
    assert literal.role == "constraint-value"
    assert literal.literal_value == 12

    constraint = plan.constraints[0]
    assert constraint.args_refs == ["e_year", "e_twelve"]
    assert constraint.args_blanks == []


def test_list_intersection_shot_uses_intermediate_list_blanks() -> None:
    plan = GSWPlan.model_validate(_FEW_SHOT_6_LIST_INTERSECTION["plan"])
    list_blanks = [e.id for e in plan.blank_entities() if e.value_type == "list"]
    assert set(list_blanks) == {"b_hr_list", "b_sb_list"}

    target_vps = [
        vp for vp in plan.verb_phrases
        if "t" in (vp.subject_id, vp.object_id)
    ]
    assert len(target_vps) == 2
    assert {vp.object_id for vp in target_vps} == {"b_hr_list", "b_sb_list"}


def test_as_of_date_shot_has_date_anchor_role() -> None:
    """Few-shot #2 explicitly demonstrates Hard rule 2 — an
    ``as-of-date`` filled entity should be present."""
    plan = GSWPlan.model_validate(_FEW_SHOT_2_AS_OF_DATE_DIFF["plan"])
    as_of_date_roles = [
        e for e in plan.entities
        if e.kind == "filled" and e.role == "as-of-date"
    ]
    assert as_of_date_roles, "few-shot #2 must carry a filled as-of-date entity"


def test_temporal_bridge_shot_has_year_anchor_role() -> None:
    """Few-shot #1 demonstrates Hard rule 2 via the ``year-anchor`` role."""
    plan = GSWPlan.model_validate(_FEW_SHOT_1_TEMPORAL_BRIDGE["plan"])
    year_anchors = [
        e for e in plan.entities
        if e.kind == "filled" and e.role == "year-anchor"
    ]
    assert year_anchors, "few-shot #1 must carry a filled year-anchor entity"
