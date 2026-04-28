"""Unit tests for ``_plan_diff`` — diff ops + apply + signature.

Exercises every diff-op type on a small fixture plan, the
trivial-op detector, and the structural-signature equality used as a
post-apply no-op check.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.adapters.ours._plan_diff import (
    AddBlankOp,
    AddConstraintOp,
    AddVPOp,
    ChangeTargetOp,
    ChangeValueTypeOp,
    DiffApplyError,
    RemoveBlankOp,
    RemoveConstraintOp,
    RemoveVPOp,
    ReplaceConstraintOp,
    apply_diff,
    is_trivial_op,
    parse_diff_op,
    plan_signature,
)
from research_agent.adapters.ours._planner_exec import GSWPlan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_blank_plan() -> dict[str, Any]:
    """Pink Floyd → bridge year → target album."""
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "active_in", "subject_id": "e1", "object_id": "b_year"},
            {"id": "vp2", "phrase": "released_in", "subject_id": "t", "object_id": "b_year"},
        ],
        "constraints": [],
    }


def _one_blank_plan() -> dict[str, Any]:
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [{"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": "t"}],
        "constraints": [],
    }


# ---------------------------------------------------------------------------
# parse_diff_op routes raw dicts to the right subclass.
# ---------------------------------------------------------------------------


def test_parse_diff_op_routes_to_add_blank():
    op = parse_diff_op({
        "op": "add_blank",
        "blank": {"id": "b_new", "value_type": "text"},
        "wire_via_vp": {
            "id": "vp_new", "phrase": "x", "subject_id": "e1", "object_id": "b_new",
        },
    })
    assert isinstance(op, AddBlankOp)
    assert op.blank.id == "b_new"
    assert op.wire_via_vp.id == "vp_new"


def test_parse_diff_op_routes_to_change_target():
    op = parse_diff_op({"op": "change_target", "from_id": "t", "to_id": "b_other"})
    assert isinstance(op, ChangeTargetOp)
    assert op.from_id == "t"
    assert op.to_id == "b_other"


def test_parse_diff_op_unknown_op_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        parse_diff_op({"op": "rename_blank", "blank_id": "x"})


def test_parse_diff_op_routes_to_replace_constraint():
    op = parse_diff_op({
        "op": "replace_constraint",
        "constraint_id": "c1",
        "constraint": {
            "id": "c1",
            "kind": "derived",
            "op": "sum",
            "args_blanks": ["b_a", "b_b"],
            "output_blank_id": "t",
        },
    })
    assert isinstance(op, ReplaceConstraintOp)
    assert op.constraint_id == "c1"


# ---------------------------------------------------------------------------
# is_trivial_op rejects no-op-shaped ops before applying.
# ---------------------------------------------------------------------------


def test_trivial_add_blank_existing_id():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({
        "op": "add_blank",
        "blank": {"id": "e1", "value_type": "text"},
        "wire_via_vp": {"id": "vp_new", "phrase": "x", "subject_id": "e1", "object_id": "t"},
    })
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is True
    assert "already exists" in reason


def test_trivial_change_target_same_id():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({"op": "change_target", "from_id": "t", "to_id": "t"})
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is True
    assert "from_id == to_id" in reason


def test_trivial_change_target_wrong_from_id():
    plan = GSWPlan.model_validate(_two_blank_plan())
    op = parse_diff_op({"op": "change_target", "from_id": "b_year", "to_id": "t"})
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is True
    assert "not the current target" in reason


def test_trivial_change_value_type_same_type():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({"op": "change_value_type", "blank_id": "t", "to": "entity"})
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is True
    assert "already has value_type" in reason


def test_trivial_remove_blank_target():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({"op": "remove_blank", "blank_id": "t"})
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is True
    assert "cannot remove target" in reason


def test_trivial_add_vp_self_loop():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({
        "op": "add_vp",
        "vp": {"id": "vp_new", "phrase": "x", "subject_id": "t", "object_id": "t"},
    })
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is True
    assert "self-loop" in reason


def test_trivial_add_vp_duplicate():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({
        "op": "add_vp",
        "vp": {"id": "vp_dup", "phrase": "released", "subject_id": "e1", "object_id": "t"},
    })
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is True
    assert "duplicate" in reason


def test_nontrivial_add_blank_passes():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({
        "op": "add_blank",
        "blank": {"id": "b_year", "value_type": "date", "role": "bridge-date"},
        "wire_via_vp": {"id": "vp_new", "phrase": "released_in", "subject_id": "t", "object_id": "b_year"},
    })
    trivial, reason = is_trivial_op(op, plan)
    assert trivial is False
    assert reason == ""


# ---------------------------------------------------------------------------
# apply_diff produces a structurally-different plan.
# ---------------------------------------------------------------------------


def test_apply_add_blank():
    plan = GSWPlan.model_validate(_one_blank_plan())
    sig0 = plan_signature(plan)
    op = parse_diff_op({
        "op": "add_blank",
        "blank": {"id": "b_year", "value_type": "date", "role": "bridge-date"},
        "wire_via_vp": {"id": "vp_year", "phrase": "released_in", "subject_id": "t", "object_id": "b_year"},
    })
    new = apply_diff(plan, op)
    assert {e.id for e in new.entities} == {"e1", "t", "b_year"}
    assert any(vp.id == "vp_year" for vp in new.verb_phrases)
    assert plan_signature(new) != sig0


def test_apply_remove_blank_invalidates_when_target_dangles():
    """Removing the only blank that wires the target ⇒ target becomes
    dangling ⇒ DiffApplyError (caught by re-validation)."""
    plan = GSWPlan.model_validate(_two_blank_plan())
    # Remove t's only wiring: drop b_year while t had VPs only to b_year.
    # Wait — vp1: e1→b_year, vp2: t→b_year. After removing b_year both
    # VPs go and t becomes dangling. revalidation catches it.
    op = parse_diff_op({"op": "remove_blank", "blank_id": "b_year"})
    with pytest.raises(DiffApplyError):
        apply_diff(plan, op)


def test_apply_change_target():
    plan = GSWPlan.model_validate(_two_blank_plan())
    op = parse_diff_op({"op": "change_target", "from_id": "t", "to_id": "b_year"})
    new = apply_diff(plan, op)
    targets = [e.id for e in new.entities if e.is_target]
    assert targets == ["b_year"]


def test_apply_add_vp():
    plan = GSWPlan.model_validate(_two_blank_plan())
    op = parse_diff_op({
        "op": "add_vp",
        "vp": {"id": "vp3", "phrase": "co_mentioned", "subject_id": "e1", "object_id": "t"},
    })
    new = apply_diff(plan, op)
    assert any(vp.id == "vp3" for vp in new.verb_phrases)


def test_apply_remove_vp_can_invalidate():
    """Removing the VP that wires a blank ⇒ blank dangles ⇒ DiffApplyError."""
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({"op": "remove_vp", "vp_id": "vp1"})
    with pytest.raises(DiffApplyError):
        apply_diff(plan, op)


def test_apply_add_constraint_concat():
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "b_b", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_b"},
            {"id": "vp_t", "phrase": "is_concat_of", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }
    plan = GSWPlan.model_validate(plan_dict)
    op = parse_diff_op({
        "op": "add_constraint",
        "constraint": {
            "id": "c1", "kind": "derived", "op": "concat",
            "args_blanks": ["b_a", "b_b"], "output_blank_id": "t",
        },
    })
    new = apply_diff(plan, op)
    assert len(new.constraints) == 1
    assert new.constraints[0].kind == "derived"
    assert new.constraints[0].args_blanks == ["b_a", "b_b"]


def test_apply_change_value_type():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({"op": "change_value_type", "blank_id": "t", "to": "text"})
    new = apply_diff(plan, op)
    t_ent = next(e for e in new.entities if e.id == "t")
    assert t_ent.value_type == "text"


def test_apply_replace_constraint():
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_b", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "sum",
                "args_blanks": ["b_a"],
                "output_blank_id": "t",
            }
        ],
    }
    plan = GSWPlan.model_validate(plan_dict)
    op = parse_diff_op({
        "op": "replace_constraint",
        "constraint_id": "c1",
        "constraint": {
            "id": "c1",
            "kind": "derived",
            "op": "sum",
            "args_blanks": ["b_a", "b_b"],
            "output_blank_id": "t",
        },
    })
    new = apply_diff(plan, op)
    assert len(new.constraints) == 1
    assert new.constraints[0].args_blanks == ["b_a", "b_b"]


def test_apply_add_constraint_duplicate_output_rejected():
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_b", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "sum",
                "args_blanks": ["b_a"],
                "output_blank_id": "t",
            }
        ],
    }
    plan = GSWPlan.model_validate(plan_dict)
    op = parse_diff_op({
        "op": "add_constraint",
        "constraint": {
            "id": "c2",
            "kind": "derived",
            "op": "sum",
            "args_blanks": ["b_a", "b_b"],
            "output_blank_id": "t",
        },
    })
    with pytest.raises(DiffApplyError):
        apply_diff(plan, op)


def test_apply_add_blank_introducing_cycle_rejected():
    """Adding a constraint or VP that creates a dependency cycle is
    caught by topological_sort_blanks during apply."""
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "X", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_b", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "links", "subject_id": "b_a", "object_id": "b_b"},
            {"id": "vp3", "phrase": "named", "subject_id": "b_b", "object_id": "t"},
        ],
        "constraints": [],
    }
    plan = GSWPlan.model_validate(plan_dict)
    # Add a VP that creates b_b → b_a edge (cycle with vp2).
    op = parse_diff_op({
        "op": "add_vp",
        "vp": {"id": "vp_cycle", "phrase": "back_links", "subject_id": "b_b", "object_id": "b_a"},
    })
    with pytest.raises(DiffApplyError):
        apply_diff(plan, op)


# ---------------------------------------------------------------------------
# plan_signature equality.
# ---------------------------------------------------------------------------


def test_signature_stable_on_identical_plan():
    p1 = GSWPlan.model_validate(_two_blank_plan())
    p2 = GSWPlan.model_validate(_two_blank_plan())
    assert plan_signature(p1) == plan_signature(p2)


def test_signature_changes_after_apply():
    plan = GSWPlan.model_validate(_one_blank_plan())
    op = parse_diff_op({
        "op": "add_blank",
        "blank": {"id": "b_year", "value_type": "date"},
        "wire_via_vp": {"id": "vp_new", "phrase": "released_in", "subject_id": "t", "object_id": "b_year"},
    })
    new = apply_diff(plan, op)
    assert plan_signature(plan) != plan_signature(new)


def test_signature_ignores_entity_ordering():
    """Sorting in plan_signature should make entity-list reordering
    invisible — the executor sees the same plan."""
    p1 = GSWPlan.model_validate(_two_blank_plan())
    plan_dict = _two_blank_plan()
    plan_dict["entities"] = list(reversed(plan_dict["entities"]))
    plan_dict["verb_phrases"] = list(reversed(plan_dict["verb_phrases"]))
    p2 = GSWPlan.model_validate(plan_dict)
    assert plan_signature(p1) == plan_signature(p2)
