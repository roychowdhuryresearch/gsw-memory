"""Structured diff operations for GSWPlan revisions.

The plan-updater used to emit a full revised ``GSWPlan`` JSON. On
gpt-oss-120b, the path of least resistance was to copy the input plan
verbatim — measured at 100% no-op rate across 25 successful calls on
the Phase-3.4 baseline (Phase-3.6 motivation).

This module defines a smaller output contract: the updater emits ONE
:class:`PlanDiffOp` describing a single structural edit
(``add_blank``, ``change_target``, ``add_constraint``, ...). The op is
applied deterministically via :func:`apply_diff`, the resulting plan
re-validates against ``GSWPlan``, and the topology is re-checked. A
``no-op`` is impossible by construction — there is no way to ``add a
blank`` that produces an identical plan, etc.

Companion code:

- :func:`is_trivial_op` rejects ops that the LLM might fall back to
  if it really wants to no-op (e.g. ``change_target`` where
  ``from_id == to_id``). The plan-updater retries with a sharpened
  prompt.
- :func:`plan_signature` is a stable structural fingerprint reused
  from the rolled-back Phase-3.5 patch — the rejection-sampling loop
  in :mod:`_plan_updater` calls it after ``apply_diff`` as a
  belt-and-suspenders no-op check.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from research_agent.adapters.ours._planner_exec import (
    Constraint,
    Entity,
    ExecutionError,
    GSWPlan,
    VerbPhrase,
    build_dependency_graph,
    topological_sort_blanks,
)


# ---------------------------------------------------------------------------
# Diff-op pydantic models — discriminated union on the ``op`` field.
# ---------------------------------------------------------------------------


class _BlankSpec(BaseModel):
    """Subset of Entity fields the LLM may set when adding a blank."""

    id: str
    role: Optional[str] = None
    value_type: Optional[
        Literal["date", "number", "entity", "attribute", "list", "text", "bool"]
    ] = "text"
    is_target: bool = False
    state: Optional[str] = None


class _VPSpec(BaseModel):
    """VerbPhrase the LLM may add when wiring a new blank."""

    id: str
    phrase: str
    subject_id: str
    object_id: str


class _ConstraintSpec(BaseModel):
    """Constraint shape the LLM may add."""

    id: str
    kind: Literal["derived", "argmax", "argmin", "equals", "in_list", "gt", "lt"]
    op: Optional[
        Literal[
            "diff",
            "sum",
            "avg",
            "max",
            "min",
            "count",
            "concat",
            "mul",
            "div",
            "round_nearest",
        ]
    ] = None
    args_blanks: list[str] = Field(default_factory=list)
    args_refs: list[str] = Field(default_factory=list)
    candidate_entity_ids: list[str] = Field(default_factory=list)
    sort_by_blank_ids: list[str] = Field(default_factory=list)
    left_ref: Optional[str] = None
    right_ref: Optional[str] = None
    output_blank_id: Optional[str] = None


class AddBlankOp(BaseModel):
    op: Literal["add_blank"]
    blank: _BlankSpec
    # Required: blank entities must be referenced by ≥ 1 VP/constraint
    # to satisfy ``GSWPlan._check_no_dangling``. The LLM provides the
    # wiring VP atomically with the new blank.
    wire_via_vp: _VPSpec


class RemoveBlankOp(BaseModel):
    op: Literal["remove_blank"]
    blank_id: str


class ChangeTargetOp(BaseModel):
    op: Literal["change_target"]
    from_id: str
    to_id: str


class AddVPOp(BaseModel):
    op: Literal["add_vp"]
    vp: _VPSpec


class RemoveVPOp(BaseModel):
    op: Literal["remove_vp"]
    vp_id: str


class AddConstraintOp(BaseModel):
    op: Literal["add_constraint"]
    constraint: _ConstraintSpec


class RemoveConstraintOp(BaseModel):
    op: Literal["remove_constraint"]
    constraint_id: str


class ChangeValueTypeOp(BaseModel):
    op: Literal["change_value_type"]
    blank_id: str
    to: Literal["date", "number", "entity", "attribute", "list", "text", "bool"]


PlanDiffOp = Annotated[
    Union[
        AddBlankOp,
        RemoveBlankOp,
        ChangeTargetOp,
        AddVPOp,
        RemoveVPOp,
        AddConstraintOp,
        RemoveConstraintOp,
        ChangeValueTypeOp,
    ],
    Field(discriminator="op"),
]


# A pydantic wrapper that applies the discriminator — used to parse a
# raw dict into the right subclass.
class _PlanDiffOpEnvelope(BaseModel):
    op_obj: PlanDiffOp = Field(alias="root")


def parse_diff_op(raw: dict) -> "PlanDiffOp":
    """Parse a raw dict into a typed PlanDiffOp.

    Wraps the discriminated-union root validator so callers don't
    have to know about the envelope.
    """
    # Use TypeAdapter for the union directly.
    from pydantic import TypeAdapter

    adapter = TypeAdapter(PlanDiffOp)
    return adapter.validate_python(raw)


# ---------------------------------------------------------------------------
# Trivial-op detection — rejects "no-op shaped" ops before applying.
# ---------------------------------------------------------------------------


def is_trivial_op(op: "PlanDiffOp", plan: GSWPlan) -> tuple[bool, str]:
    """Return ``(True, reason)`` if the op cannot meaningfully change
    the plan. Used by :mod:`_plan_updater` to reject before applying
    so the rejection-sampling retry sees a clear error message.
    """
    blank_ids = {e.id for e in plan.blank_entities()}
    all_ent_ids = {e.id for e in plan.entities}
    vp_ids = {vp.id for vp in plan.verb_phrases}
    cons_ids = {c.id for c in plan.constraints}
    target_id = next((e.id for e in plan.blank_entities() if e.is_target), None)

    if isinstance(op, AddBlankOp):
        if op.blank.id in all_ent_ids:
            return True, f"add_blank: id {op.blank.id!r} already exists in plan"
        # The wiring VP must reference at least the new blank id and
        # one other entity that exists in the post-add plan.
        new_ids = all_ent_ids | {op.blank.id}
        if op.wire_via_vp.subject_id not in new_ids:
            return True, (
                f"add_blank: wire_via_vp.subject_id "
                f"{op.wire_via_vp.subject_id!r} not in plan"
            )
        if op.wire_via_vp.object_id not in new_ids:
            return True, (
                f"add_blank: wire_via_vp.object_id "
                f"{op.wire_via_vp.object_id!r} not in plan"
            )
        if op.wire_via_vp.id in vp_ids:
            return True, f"add_blank: wire_via_vp.id {op.wire_via_vp.id!r} already exists"
        if op.wire_via_vp.subject_id == op.wire_via_vp.object_id:
            return True, "add_blank: wire_via_vp self-loop"
        return False, ""

    if isinstance(op, RemoveBlankOp):
        if op.blank_id not in blank_ids:
            return True, f"remove_blank: id {op.blank_id!r} not in plan"
        if op.blank_id == target_id:
            return True, (
                f"remove_blank: cannot remove target {op.blank_id!r} — "
                "use change_target first"
            )
        return False, ""

    if isinstance(op, ChangeTargetOp):
        if op.from_id == op.to_id:
            return True, "change_target: from_id == to_id"
        if op.from_id != target_id:
            return True, (
                f"change_target: from_id {op.from_id!r} is not the "
                f"current target ({target_id!r})"
            )
        if op.to_id not in blank_ids:
            return True, f"change_target: to_id {op.to_id!r} not a blank in plan"
        return False, ""

    if isinstance(op, AddVPOp):
        if op.vp.id in vp_ids:
            return True, f"add_vp: id {op.vp.id!r} already exists"
        if op.vp.subject_id not in all_ent_ids:
            return True, f"add_vp: subject_id {op.vp.subject_id!r} not in plan"
        if op.vp.object_id not in all_ent_ids:
            return True, f"add_vp: object_id {op.vp.object_id!r} not in plan"
        if op.vp.subject_id == op.vp.object_id:
            return True, "add_vp: self-loop"
        # Reject duplicate (subject, phrase, object) VPs — they don't
        # change the dependency graph.
        for vp in plan.verb_phrases:
            if (vp.subject_id, vp.phrase, vp.object_id) == (
                op.vp.subject_id,
                op.vp.phrase,
                op.vp.object_id,
            ):
                return True, f"add_vp: duplicate of existing {vp.id!r}"
        return False, ""

    if isinstance(op, RemoveVPOp):
        if op.vp_id not in vp_ids:
            return True, f"remove_vp: id {op.vp_id!r} not in plan"
        return False, ""

    if isinstance(op, AddConstraintOp):
        if op.constraint.id in cons_ids:
            return True, f"add_constraint: id {op.constraint.id!r} already exists"
        if op.constraint.output_blank_id and op.constraint.output_blank_id not in blank_ids:
            return True, (
                f"add_constraint: output_blank_id "
                f"{op.constraint.output_blank_id!r} not a blank in plan"
            )
        return False, ""

    if isinstance(op, RemoveConstraintOp):
        if op.constraint_id not in cons_ids:
            return True, f"remove_constraint: id {op.constraint_id!r} not in plan"
        return False, ""

    if isinstance(op, ChangeValueTypeOp):
        if op.blank_id not in blank_ids:
            return True, f"change_value_type: blank_id {op.blank_id!r} not in plan"
        ent = next(e for e in plan.entities if e.id == op.blank_id)
        if ent.value_type == op.to:
            return True, (
                f"change_value_type: blank {op.blank_id!r} already has "
                f"value_type={op.to!r}"
            )
        return False, ""

    raise ValueError(f"unknown diff op type: {type(op).__name__}")


# ---------------------------------------------------------------------------
# Apply diff — deterministic mutation + re-validation.
# ---------------------------------------------------------------------------


class DiffApplyError(Exception):
    """Raised when applying a diff produces an invalid plan.

    Distinguished from ``is_trivial_op`` which catches "no-op shape"
    pre-apply. ``DiffApplyError`` fires when the result violates a
    structural invariant (cycle introduced, dangling entity, multiple
    targets, etc.).
    """


def apply_diff(old_plan: GSWPlan, op: "PlanDiffOp") -> GSWPlan:
    """Apply ``op`` to a deep copy of ``old_plan`` and return the new plan.

    The result is re-validated against ``GSWPlan`` (catches dangling
    entities, no-target-blank, etc.) and the topology is re-checked
    via ``topological_sort_blanks`` (catches cycles introduced by a
    new VP). On failure raises ``DiffApplyError``.
    """
    new = old_plan.model_copy(deep=True)

    if isinstance(op, AddBlankOp):
        new.entities.append(
            Entity(
                id=op.blank.id,
                kind="blank",
                value_type=op.blank.value_type,
                role=op.blank.role,
                is_target=op.blank.is_target,
                state=op.blank.state,
            )
        )
        new.verb_phrases.append(
            VerbPhrase(
                id=op.wire_via_vp.id,
                phrase=op.wire_via_vp.phrase,
                subject_id=op.wire_via_vp.subject_id,
                object_id=op.wire_via_vp.object_id,
            )
        )

    elif isinstance(op, RemoveBlankOp):
        new.entities = [e for e in new.entities if e.id != op.blank_id]
        # Drop any VP/constraint that referenced the removed blank.
        new.verb_phrases = [
            vp for vp in new.verb_phrases
            if op.blank_id not in (vp.subject_id, vp.object_id)
        ]
        new.constraints = [
            c for c in new.constraints
            if op.blank_id not in c.args_blanks
            and op.blank_id not in c.args_refs
            and op.blank_id not in c.candidate_entity_ids
            and op.blank_id not in c.sort_by_blank_ids
            and c.output_blank_id != op.blank_id
            and c.left_ref != op.blank_id
            and c.right_ref != op.blank_id
        ]

    elif isinstance(op, ChangeTargetOp):
        for e in new.entities:
            if e.id == op.from_id:
                e.is_target = False
            if e.id == op.to_id:
                e.is_target = True

    elif isinstance(op, AddVPOp):
        new.verb_phrases.append(
            VerbPhrase(
                id=op.vp.id,
                phrase=op.vp.phrase,
                subject_id=op.vp.subject_id,
                object_id=op.vp.object_id,
            )
        )

    elif isinstance(op, RemoveVPOp):
        new.verb_phrases = [vp for vp in new.verb_phrases if vp.id != op.vp_id]

    elif isinstance(op, AddConstraintOp):
        new.constraints.append(
            Constraint(
                id=op.constraint.id,
                kind=op.constraint.kind,
                op=op.constraint.op,
                args_blanks=list(op.constraint.args_blanks),
                args_refs=list(op.constraint.args_refs),
                candidate_entity_ids=list(op.constraint.candidate_entity_ids),
                sort_by_blank_ids=list(op.constraint.sort_by_blank_ids),
                left_ref=op.constraint.left_ref,
                right_ref=op.constraint.right_ref,
                output_blank_id=op.constraint.output_blank_id,
            )
        )

    elif isinstance(op, RemoveConstraintOp):
        new.constraints = [c for c in new.constraints if c.id != op.constraint_id]

    elif isinstance(op, ChangeValueTypeOp):
        for e in new.entities:
            if e.id == op.blank_id:
                e.value_type = op.to
                break

    else:
        raise DiffApplyError(f"unknown diff op type: {type(op).__name__}")

    # Re-validate against GSWPlan schema (catches dangling entities,
    # multiple/zero targets via the model validator).
    try:
        revalidated = GSWPlan.model_validate(new.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise DiffApplyError(f"post-apply validation failed: {exc}") from exc

    # Re-check topology (catches cycles introduced by a new VP).
    try:
        topological_sort_blanks(revalidated)
        build_dependency_graph(revalidated)
    except ExecutionError as exc:
        raise DiffApplyError(
            f"post-apply topology invalid: {exc.kind}: {exc.detail}"
        ) from exc

    return revalidated


# ---------------------------------------------------------------------------
# Structural fingerprint (no-op detection belt-and-suspenders).
# ---------------------------------------------------------------------------


def plan_signature(plan: GSWPlan) -> tuple:
    """Stable structural fingerprint over (entities, VPs, constraints).

    Two plans with equal signatures differ in nothing the executor
    sees: same blank ids, same target, same VP graph, same
    constraints. Used as a final no-op check after :func:`apply_diff`
    even though :func:`is_trivial_op` should have already rejected
    most no-op shapes.
    """
    ents = tuple(sorted(
        (
            e.id,
            e.kind,
            e.name or "",
            _literal_signature(e.literal_value),
            e.value_type or "",
            bool(e.is_target),
            e.role or "",
        )
        for e in plan.entities
    ))
    vps = tuple(sorted(
        (vp.id, vp.phrase, vp.subject_id, vp.object_id)
        for vp in plan.verb_phrases
    ))
    cons = tuple(sorted(
        (
            c.id,
            c.kind,
            c.op or "",
            tuple(c.args_blanks),
            tuple(c.args_refs),
            tuple(c.candidate_entity_ids),
            tuple(c.sort_by_blank_ids),
            c.left_ref or "",
            c.right_ref or "",
            c.output_blank_id or "",
        )
        for c in plan.constraints
    ))
    return (ents, vps, cons)


def _literal_signature(value: object) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, default=str)
