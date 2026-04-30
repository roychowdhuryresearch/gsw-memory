"""Executor for GSW-fragment plans.

The executor walks a ``GSWPlan`` emitted by the question-planner LLM and
fills its blank entities in topological order. Python handles the DAG
traversal, value substitution, and numeric / selector constraints; the
LLM is used only to extract a typed value from retrieved chunks for a
blank that needs identification or attribute projection.

The shape here is deliberately generic — no FRAMES-specific slot
types, no per-question templates. The planner is expected to express
any multi-hop question in terms of:

- Filled entities (extracted from the question text via NER).
- Blank entities with a ``value_type`` (to be resolved by retrieval).
- Verb-phrases connecting entities (directional, binary).
- Constraints between blanks (derived / argmax / argmin + a few
  relational kinds for Phase-2).

One blank has ``is_target=True``; the executor returns its resolved
value as the final answer.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from research_agent.models.trace import ToolCall


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """A node in the GSW-fragment plan.

    Filled entities carry a ``name`` (extracted from the question).
    Blank entities carry a ``value_type`` and are resolved by retrieval.
    Exactly one blank should have ``is_target=True``.
    """

    id: str
    kind: Literal["filled", "blank"]
    name: Optional[str] = None
    value_type: Optional[
        Literal["date", "number", "entity", "attribute", "list", "text", "bool"]
    ] = None
    is_target: bool = False
    category: bool = False
    # Phase-1.5 planner metadata — purely informational for the LLM + UI.
    # The executor ignores these fields (they don't change retrieval or
    # constraint evaluation). Optional so older dumps / plans without
    # role annotations keep validating.
    role: Optional[str] = None
    state: Optional[str] = None
    # Literal value for filled constraint constants. Example:
    # name="twelve", literal_value=12 lets the entity remain grounded in
    # the question text while constraints consume the numeric value.
    literal_value: Optional[Any] = None


class VerbPhrase(BaseModel):
    """A typed binary relation between two entities."""

    id: str
    phrase: str
    subject_id: str
    object_id: str


class Constraint(BaseModel):
    """Dependency between blanks.

    Only one of the field groups is populated per instance:
    - ``op`` + ``args_refs`` / ``args_blanks`` for ``kind="derived"``.
    - ``candidate_entity_ids`` + ``sort_by_blank_ids`` for argmax / argmin.
    - ``left_ref`` / ``right_ref`` for relational kinds (Phase-2).

    ``output_blank_id`` names the blank whose value this constraint fills.
    For argmax / argmin the filled blank's value is an entity id drawn
    from ``candidate_entity_ids``.
    """

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
    # Preferred input list for derived constraints. Unlike the legacy
    # args_blanks field, args_refs may point to blanks or filled literal
    # entities such as constraint-value "twelve".
    args_refs: list[str] = Field(default_factory=list)
    candidate_entity_ids: list[str] = Field(default_factory=list)
    sort_by_blank_ids: list[str] = Field(default_factory=list)
    left_ref: Optional[str] = None
    right_ref: Optional[str] = None
    output_blank_id: Optional[str] = None


class GSWPlan(BaseModel):
    """Full plan emitted by the question-planner LLM."""

    entities: list[Entity]
    verb_phrases: list[VerbPhrase] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    def filled_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.kind == "filled"]

    def blank_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.kind == "blank"]

    def target(self) -> Entity:
        targets = [e for e in self.blank_entities() if e.is_target]
        if len(targets) != 1:
            raise ExecutionError(
                kind="no_target",
                detail=f"expected exactly one target blank, found {len(targets)}",
            )
        return targets[0]

    def entity_by_id(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise ExecutionError(kind="bad_ref", detail=f"unknown entity id {eid!r}")

    @model_validator(mode="after")
    def _check_known_refs(self) -> "GSWPlan":
        """Every VP/constraint reference must point at an emitted entity."""
        entity_ids = {e.id for e in self.entities}
        blank_ids = {e.id for e in self.blank_entities()}

        def require_known(ref: Optional[str], *, where: str) -> None:
            if ref and ref not in entity_ids:
                raise ValueError(f"{where} references unknown entity id {ref!r}")

        def require_blank(ref: Optional[str], *, where: str) -> None:
            if ref and ref not in blank_ids:
                raise ValueError(f"{where} references non-blank id {ref!r}")

        for vp in self.verb_phrases:
            require_known(vp.subject_id, where=f"verb-phrase {vp.id!r}.subject_id")
            require_known(vp.object_id, where=f"verb-phrase {vp.id!r}.object_id")

        for c in self.constraints:
            for b in c.args_blanks:
                require_blank(b, where=f"constraint {c.id!r}.args_blanks")
            for ref in c.args_refs:
                require_known(ref, where=f"constraint {c.id!r}.args_refs")
            for b in c.sort_by_blank_ids:
                require_blank(b, where=f"constraint {c.id!r}.sort_by_blank_ids")
            for eid in c.candidate_entity_ids:
                require_known(eid, where=f"constraint {c.id!r}.candidate_entity_ids")
            require_known(c.left_ref, where=f"constraint {c.id!r}.left_ref")
            require_known(c.right_ref, where=f"constraint {c.id!r}.right_ref")
            require_blank(c.output_blank_id, where=f"constraint {c.id!r}.output_blank_id")
        return self

    @model_validator(mode="after")
    def _check_no_dangling(self) -> "GSWPlan":
        """Every entity must be referenced by ≥ 1 verb-phrase or constraint.

        A dangling entity is unreachable by the executor — no retrieval
        signal can use it and no constraint can consume/produce it — so
        it contributes nothing to answering the question. Emit a clear
        ValidationError so the adapter's repair retry can surface it to
        the LLM.
        """
        referenced: set[str] = set()
        for vp in self.verb_phrases:
            referenced.add(vp.subject_id)
            referenced.add(vp.object_id)
        for c in self.constraints:
            referenced.update(c.args_blanks)
            referenced.update(c.args_refs)
            referenced.update(c.candidate_entity_ids)
            referenced.update(c.sort_by_blank_ids)
            for ref in (c.left_ref, c.right_ref, c.output_blank_id):
                if ref:
                    referenced.add(ref)
        dangling = [e.id for e in self.entities if e.id not in referenced]
        if dangling:
            raise ValueError(
                f"dangling entities (no verb-phrase or constraint edge): {dangling}"
            )
        return self

    @model_validator(mode="after")
    def _check_constraint_shape(self) -> "GSWPlan":
        """Validate per-kind required fields on each constraint.

        A constraint with empty inputs is uncomputable — the executor
        can't pick a winner / sum / compare anything. The
        orchestrator then submits whatever placeholder is in scope
        (e.g., the literal hint string ``Winner B``). Catching at
        plan-emit time and raising ``ValueError`` lets the planner
        repair retry surface a concrete error to the LLM.
        """
        for c in self.constraints:
            if c.kind == "derived":
                if not c.op:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=derived) has no op"
                    )
                if not (c.args_refs or c.args_blanks):
                    raise ValueError(
                        f"constraint {c.id!r} (kind=derived, op={c.op!r}) "
                        "has empty args_refs/args_blanks; derived "
                        "constraints must list input refs"
                    )
            elif c.kind in ("argmax", "argmin"):
                if not c.candidate_entity_ids:
                    raise ValueError(
                        f"constraint {c.id!r} (kind={c.kind}) has empty "
                        "candidate_entity_ids; argmax/argmin must list "
                        "the entities to choose between"
                    )
                if not c.sort_by_blank_ids:
                    raise ValueError(
                        f"constraint {c.id!r} (kind={c.kind}) has empty "
                        "sort_by_blank_ids; argmax/argmin must specify "
                        "the blank(s) to rank candidates by"
                    )
            elif c.kind in ("equals", "gt", "lt"):
                if not (c.left_ref and c.right_ref):
                    raise ValueError(
                        f"constraint {c.id!r} (kind={c.kind}) requires "
                        "both left_ref and right_ref"
                    )
            elif c.kind == "in_list":
                has_left_right = bool(c.left_ref and c.right_ref)
                has_args = len(c.args_blanks) >= 2
                if not (has_left_right or has_args):
                    raise ValueError(
                        f"constraint {c.id!r} (kind=in_list) requires "
                        "either left_ref/right_ref or args_blanks="
                        "[member_blank, list_blank]"
                    )
        return self


# ---------------------------------------------------------------------------
# Errors + state
# ---------------------------------------------------------------------------


class ExecutionError(Exception):
    """Raised when the planner-emitted plan cannot be executed.

    ``kind`` is one of: ``cyclic_plan``, ``no_target``, ``bad_ref``,
    ``empty_plan``. The adapter catches this and falls back to flat.
    """

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


@dataclass
class BlankResult:
    blank_id: str
    value: Any = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    status: Literal["unknown", "resolved", "error"] = "unknown"
    llm_calls: int = 0
    wall_time_s: float = 0.0
    error: str = ""


@dataclass
class ExecutionTrace:
    """Captured execution detail for attaching to a Trajectory.extra."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    executed_blanks: list[BlankResult] = field(default_factory=list)
    wall_times: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Dependency DAG + topo sort
# ---------------------------------------------------------------------------


def build_dependency_graph(plan: GSWPlan) -> dict[str, set[str]]:
    """For each blank B, return the set of blanks it depends on.

    B depends on A if (a) a Constraint inputs A and outputs B, or
    (b) a VerbPhrase has A on one side and B on the other and A is
    used to identify B.

    For VP-based dependencies we choose the direction conservatively:
    if one endpoint is an entity blank (type=entity) with no incoming
    filled-neighbor VPs yet, we treat it as the "source" for outgoing
    attribute-projection VPs. But a simpler and equivalent rule is:
    **both endpoints of a VP sit at the same dependency level; if one
    is later resolved we re-visit**. The topo sort uses only constraint
    edges + "identification-requires-incoming-VP" — projection edges
    are executed after the source blank resolves.

    This function returns the **constraint dependency graph** only; the
    executor handles VP-based projection as a second phase after
    identification.
    """

    blanks = {e.id for e in plan.blank_entities()}
    deps: dict[str, set[str]] = {b: set() for b in blanks}

    # Constraint-based deps: output depends on all constraint inputs.
    for c in plan.constraints:
        inputs: set[str] = set()
        if c.kind == "derived":
            inputs.update(x for x in _derived_arg_refs(c) if x in blanks)
        elif c.kind in ("argmax", "argmin"):
            inputs.update(x for x in c.sort_by_blank_ids if x in blanks)
        elif c.kind in ("equals", "gt", "lt"):
            for x in (c.left_ref, c.right_ref):
                if x and x in blanks:
                    inputs.add(x)
        elif c.kind == "in_list":
            if c.left_ref and c.right_ref:
                for x in (c.left_ref, c.right_ref):
                    if x in blanks:
                        inputs.add(x)
            else:
                inputs.update(x for x in c.args_blanks[:2] if x in blanks)

        if c.output_blank_id and c.output_blank_id in blanks:
            # output depends on all inputs (minus self)
            for inp in inputs:
                if inp != c.output_blank_id:
                    deps[c.output_blank_id].add(inp)

    # VP-based "projection" deps: if a blank-entity B has outgoing VPs
    # to other blanks (B is subject, object is blank), those object
    # blanks depend on B being resolved first.
    ent_by_id = {e.id: e for e in plan.entities}
    for vp in plan.verb_phrases:
        sub = ent_by_id.get(vp.subject_id)
        obj = ent_by_id.get(vp.object_id)
        if not sub or not obj:
            raise ExecutionError(
                kind="bad_ref",
                detail=f"verb-phrase {vp.id!r} references unknown entity",
            )
        # Case: blank_entity --[VP]--> blank (projection) — object depends on subject.
        if sub.kind == "blank" and sub.value_type == "entity" and obj.kind == "blank":
            deps[obj.id].add(sub.id)

    return deps


def topological_sort_blanks(plan: GSWPlan) -> list[str]:
    """Kahn's algorithm. Raises ExecutionError on cycle."""

    deps = build_dependency_graph(plan)
    indeg: dict[str, int] = {b: len(ds) for b, ds in deps.items()}
    children: dict[str, set[str]] = defaultdict(set)
    for b, ds in deps.items():
        for d in ds:
            children[d].add(b)

    q: deque[str] = deque([b for b, d in indeg.items() if d == 0])
    order: list[str] = []
    while q:
        b = q.popleft()
        order.append(b)
        for child in children[b]:
            indeg[child] -= 1
            if indeg[child] == 0:
                q.append(child)

    if len(order) != len(indeg):
        unresolved = [b for b, d in indeg.items() if d > 0]
        raise ExecutionError(kind="cyclic_plan", detail=f"unresolved blanks: {unresolved}")

    return order


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


Retriever = Any
"""A retriever with ``.search(query: str, top_k: int = …) -> list[Hit]``
where each ``Hit`` has ``chunk.chunk_id``, ``chunk.title``, ``chunk.text``."""


@dataclass
class _RetrievalChunk:
    chunk_id: str
    title: str
    text: str
    score: float


def _retrieve_chunks(retriever: Retriever, query: str, top_k: int = 8) -> list[_RetrievalChunk]:
    """Unified retrieval wrapper — tolerant to the BM25 Hit shape."""
    hits = retriever.search(query, top_k=top_k)
    out: list[_RetrievalChunk] = []
    for h in hits:
        chunk = getattr(h, "chunk", h)
        out.append(
            _RetrievalChunk(
                chunk_id=getattr(chunk, "chunk_id", ""),
                title=getattr(chunk, "title", ""),
                text=getattr(chunk, "text", "")[:1500],
                score=float(getattr(h, "score", 0.0)),
            )
        )
    return out


def _format_chunks_for_prompt(chunks: list[_RetrievalChunk]) -> str:
    """Plain-text formatting for LLM consumption."""
    if not chunks:
        return "(no retrieved chunks)"
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}] {c.title}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).strip()


_EXTRACT_SYSTEM = """You are extracting a single typed value from a set of \
retrieved text chunks.

Given:
  - the target blank's query (what we want to fill in)
  - expected value type (date / number / entity / attribute / list / text / bool)
  - the retrieved chunks

Return STRICT JSON:
  {"value": <the extracted value>, "evidence_chunk_ids": ["chunk_id_1", ...]}

Rules:
  - If the chunks do not contain the answer, return {"value": null, "evidence_chunk_ids": []}.
  - The "value" field type must match expected_answer_type:
      date    → "1973" or "1973-04-08" or "April 1973"
      number  → a bare number (e.g. 68408, 3.14) — no units, no commas
      entity  → a short entity name (e.g. "Dark Side of the Moon")
      list    → a JSON array of short strings
      bool    → true or false
      text    → a short phrase
  - Prefer chunks whose title directly matches the query subject.
  - NEVER fabricate. Return null if unsure.
"""


_EXTRACT_USER = """Target blank query: {query}
Expected value type: {value_type}

Context from query (relevant filled entities and resolved prior blanks):
{context}

Retrieved chunks:
{chunks}

Return the JSON object now."""


def _llm_extract(
    llm_client: Any,
    query: str,
    value_type: str,
    context: str,
    chunks: list[_RetrievalChunk],
    max_tokens: int = 600,
) -> tuple[Any, list[str], dict[str, int]]:
    """Call the LLM to extract a value from chunks.

    Returns (value, evidence_chunk_ids, usage_counts).
    """
    prompt = _EXTRACT_USER.format(
        query=query,
        value_type=value_type,
        context=context or "(none)",
        chunks=_format_chunks_for_prompt(chunks),
    )
    resp = llm_client.chat(
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    text = getattr(resp, "text", "") or ""
    usage = {
        "prompt_tokens": int(getattr(resp, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp, "completion_tokens", 0) or 0),
    }
    try:
        parsed = _extract_json_object(text)
        value = parsed.get("value", None)
        ev = parsed.get("evidence_chunk_ids", []) or []
        if not isinstance(ev, list):
            ev = []
        return value, [str(x) for x in ev], usage
    except (json.JSONDecodeError, ValueError):
        return None, [], usage


def _extract_json_object(text: str) -> dict[str, Any]:
    """Balanced-brace extractor — safer than a greedy regex."""
    if not text:
        raise ValueError("empty LLM response")
    start = text.find("{")
    if start < 0:
        raise ValueError("no opening brace in response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced braces in LLM response")


def _coerce_number(value: Any) -> Optional[float]:
    """Best-effort coercion of an LLM output into a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_year(value: Any) -> Optional[int]:
    """Pull an integer year out of a string (best effort)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    # Just look for the first 4-digit run.
    import re

    m = re.search(r"\b(\d{4})\b", s)
    if m:
        return int(m.group(1))
    return None


def _coerce_constraint_number(value: Any) -> Optional[float]:
    """Coerce values for numeric constraints.

    Date blanks often store full dates (``2015-02-25``) even when a
    downstream ``diff`` asks for a year difference. Use a normal numeric
    parse first, then fall back to the first 4-digit year.
    """
    n = _coerce_number(value)
    if n is not None:
        return n
    y = _coerce_year(value)
    if y is not None:
        return float(y)
    return None


def _coerce_sum_term(value: Any) -> Optional[float]:
    """Coerce one term for a ``sum`` constraint.

    Normal numbers win. If the value is a text token such as a UK
    postcode, fall back to summing its digits. This lets plans express
    "add up the numbers in the postcode" as ``sum(postcode_blank)``
    while still preserving the actual postcode as the upstream value.
    """
    n = _coerce_number(value)
    if n is not None:
        return n
    if isinstance(value, (list, tuple)):
        total = 0.0
        found = False
        for item in value:
            item_n = _coerce_sum_term(item)
            if item_n is None:
                continue
            total += item_n
            found = True
        return total if found else None
    digits = [int(ch) for ch in str(value) if ch.isdigit()]
    if digits:
        return float(sum(digits))
    return None


def _count_constraint_value(value: Any) -> int:
    """Best-effort count for a single count-constraint argument."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0
        import re

        parts = [p.strip() for p in re.split(r"[,;]\s*", s) if p.strip()]
        return len(parts) if len(parts) > 1 else 1
    return 1


def _constraint_ref_value(
    ref: Optional[str],
    plan: GSWPlan,
    state: dict[str, BlankResult],
) -> tuple[Any, bool]:
    """Resolve a constraint ref from state or a filled entity name."""
    if not ref:
        return None, False
    res = state.get(ref)
    if res is not None:
        return res.value, res.status == "resolved"
    try:
        ent = plan.entity_by_id(ref)
    except ExecutionError:
        return None, False
    if ent.kind == "filled":
        if ent.literal_value is not None:
            return ent.literal_value, True
        return ent.name or ent.id, True
    return None, False


def _derived_arg_refs(constraint: Constraint) -> list[str]:
    """Return derived inputs, preferring the literal-aware field."""
    return list(constraint.args_refs or constraint.args_blanks or [])


def _relational_ref_pair(constraint: Constraint) -> tuple[Optional[str], Optional[str]]:
    if constraint.left_ref and constraint.right_ref:
        return constraint.left_ref, constraint.right_ref
    if constraint.kind == "in_list" and len(constraint.args_blanks) >= 2:
        return constraint.args_blanks[0], constraint.args_blanks[1]
    return None, None


def _constraint_values_equal(left: Any, right: Any) -> bool:
    left_num = _coerce_number(left)
    right_num = _coerce_number(right)
    if left_num is not None and right_num is not None:
        return left_num == right_num
    return str(left).strip().casefold() == str(right).strip().casefold()


def _constraint_contains(member: Any, container: Any) -> bool:
    if isinstance(container, (list, tuple, set)):
        return any(_constraint_values_equal(member, item) for item in container)
    text = str(container or "")
    import re

    parts = [p.strip() for p in re.split(r"[,;\n]\s*", text) if p.strip()]
    if parts:
        return any(_constraint_values_equal(member, part) for part in parts)
    return str(member).strip().casefold() in text.casefold()


# ---------------------------------------------------------------------------
# Per-blank fill strategies
# ---------------------------------------------------------------------------


def _identify_blank(
    plan: GSWPlan,
    blank: Entity,
    state: dict[str, BlankResult],
    retriever: Retriever,
    llm_client: Any,
    top_k: int,
) -> BlankResult:
    """Fill a blank by retrieval + LLM extraction.

    Uses VPs where the blank is one endpoint and the other side is filled
    or already-resolved as search signals.
    """
    t0 = time.time()

    # Collect signals from VPs touching this blank.
    signals: list[str] = []
    for vp in plan.verb_phrases:
        if blank.id not in (vp.subject_id, vp.object_id):
            continue
        other_id = vp.object_id if vp.subject_id == blank.id else vp.subject_id
        other = plan.entity_by_id(other_id)
        other_name: Optional[str] = None
        if other.kind == "filled":
            other_name = other.name or other_id
        elif other.id in state and state[other.id].status == "resolved":
            other_name = _stringify(state[other.id].value)
        if other_name is None:
            continue
        signals.append(f"{other_name} {vp.phrase}".replace("_", " "))
    if not signals:
        # Fall back to the blank's own implied query — use VP phrases directly.
        for vp in plan.verb_phrases:
            if blank.id in (vp.subject_id, vp.object_id):
                signals.append(vp.phrase.replace("_", " "))

    query = "; ".join(signals) if signals else blank.id

    chunks = _retrieve_chunks(retriever, query, top_k=top_k)

    ctx_parts = [
        f"- {e.name}" for e in plan.filled_entities() if e.name
    ]
    for bid, res in state.items():
        if res.status == "resolved":
            ctx_parts.append(f"- {bid} = {_stringify(res.value)}")

    value, evidence, usage = _llm_extract(
        llm_client=llm_client,
        query=query,
        value_type=blank.value_type or "text",
        context="\n".join(ctx_parts),
        chunks=chunks,
    )

    # Basic value-type coercion.
    coerced = _coerce_value(value, blank.value_type)

    res = BlankResult(
        blank_id=blank.id,
        value=coerced,
        evidence_chunk_ids=evidence,
        status="resolved" if coerced is not None else "unknown",
        llm_calls=1,
        wall_time_s=round(time.time() - t0, 3),
    )
    return res


def _compute_constraint(
    constraint: Constraint,
    plan: GSWPlan,
    state: dict[str, BlankResult],
) -> BlankResult:
    """Evaluate a derived / argmax / argmin constraint into its output blank."""
    t0 = time.time()
    out_id = constraint.output_blank_id
    assert out_id, "compute_constraint requires output_blank_id"

    if constraint.kind == "derived":
        arg_refs = _derived_arg_refs(constraint)
        resolved_args = [
            _constraint_ref_value(ref, plan, state)
            for ref in arg_refs
        ]
        if any(not ok or value is None for value, ok in resolved_args):
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )
        values = [value for value, _ok in resolved_args]
        op = constraint.op
        if op == "concat":
            out_val = "; ".join(_stringify(value) for value in values)
            return BlankResult(
                blank_id=out_id,
                value=out_val,
                status="resolved",
                wall_time_s=round(time.time() - t0, 3),
            )
        if op == "count":
            if len(values) == 1:
                val = _count_constraint_value(values[0])
            else:
                val = len(values)
            return BlankResult(
                blank_id=out_id,
                value=val,
                status="resolved",
                wall_time_s=round(time.time() - t0, 3),
            )
        if op == "sum":
            nums = [_coerce_sum_term(value) for value in values]
        else:
            nums = [_coerce_constraint_number(value) for value in values]
        if any(n is None for n in nums):
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )

        if op == "diff":
            val = abs(nums[0] - nums[1]) if len(nums) >= 2 else nums[0]
        elif op == "sum":
            val = sum(nums)
        elif op == "mul":
            val = 1.0
            for n in nums:
                val *= n
        elif op == "div":
            if len(nums) < 2 or nums[1] == 0:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            val = nums[0] / nums[1]
        elif op == "round_nearest":
            step = nums[1] if len(nums) >= 2 else 10.0
            if step == 0:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            import math

            q = nums[0] / step
            rounded_q = math.floor(q + 0.5) if q >= 0 else math.ceil(q - 0.5)
            val = rounded_q * step
        elif op == "avg":
            val = sum(nums) / len(nums)
        elif op == "max":
            val = max(nums)
        elif op == "min":
            val = min(nums)
        else:
            val = nums[0]
        return BlankResult(
            blank_id=out_id,
            value=val,
            status="resolved",
            wall_time_s=round(time.time() - t0, 3),
        )

    if constraint.kind in ("argmax", "argmin"):
        candidates = constraint.candidate_entity_ids
        blanks = constraint.sort_by_blank_ids
        if len(candidates) != len(blanks) or not candidates:
            return BlankResult(blank_id=out_id, status="unknown")
        pairs = []
        for ent_id, b_id in zip(candidates, blanks):
            res = state.get(b_id)
            if res is None or res.status != "resolved" or res.value is None:
                return BlankResult(blank_id=out_id, status="unknown")
            # Use year/number coercion for ordinals.
            key = _coerce_number(res.value)
            if key is None:
                key = _coerce_year(res.value)
            if key is None:
                return BlankResult(blank_id=out_id, status="unknown")
            pairs.append((key, ent_id))
        if constraint.kind == "argmax":
            _, winner = max(pairs)
        else:
            _, winner = min(pairs)
        winner_ent = plan.entity_by_id(winner)
        return BlankResult(
            blank_id=out_id,
            value=winner_ent.name or winner_ent.id,
            status="resolved",
            wall_time_s=round(time.time() - t0, 3),
        )

    if constraint.kind in ("equals", "gt", "lt", "in_list"):
        left_ref, right_ref = _relational_ref_pair(constraint)
        left, left_ok = _constraint_ref_value(left_ref, plan, state)
        right, right_ok = _constraint_ref_value(right_ref, plan, state)
        if not (left_ok and right_ok):
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )

        if constraint.kind == "equals":
            val = _constraint_values_equal(left, right)
        elif constraint.kind == "in_list":
            val = _constraint_contains(left, right)
        else:
            left_num = _coerce_constraint_number(left)
            right_num = _coerce_constraint_number(right)
            if left_num is None or right_num is None:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            val = (
                left_num > right_num
                if constraint.kind == "gt"
                else left_num < right_num
            )
        return BlankResult(
            blank_id=out_id,
            value=val,
            status="resolved",
            wall_time_s=round(time.time() - t0, 3),
        )

    return BlankResult(blank_id=out_id, status="unknown")


# ---------------------------------------------------------------------------
# Top-level execute()
# ---------------------------------------------------------------------------


def execute(
    plan: GSWPlan,
    retriever: Retriever,
    llm_client: Any,
    *,
    top_k: int = 8,
) -> tuple[Any, ExecutionTrace]:
    """Run the GSWPlan end-to-end.

    Returns (final_answer, trace). Raises ExecutionError on structural
    failures (cycle, no target, bad ref). Empty retrievals / failed
    extractions are *not* errors — they cascade as ``unknown``.
    """

    if not plan.entities:
        raise ExecutionError(kind="empty_plan", detail="no entities")
    target = plan.target()  # raises if missing

    order = topological_sort_blanks(plan)
    state: dict[str, BlankResult] = {}
    trace = ExecutionTrace()

    # Index constraints by their output_blank_id.
    constraints_by_output: dict[str, Constraint] = {}
    for c in plan.constraints:
        if c.output_blank_id:
            constraints_by_output[c.output_blank_id] = c

    ent_by_id = {e.id: e for e in plan.entities}

    for blank_id in order:
        blank = ent_by_id[blank_id]
        turn = len(state) + 1

        # Case 1: a constraint produces this blank.
        if blank_id in constraints_by_output:
            res = _compute_constraint(constraints_by_output[blank_id], plan, state)
            state[blank_id] = res
            trace.executed_blanks.append(res)
            trace.wall_times[blank_id] = res.wall_time_s
            trace.tool_calls.append(
                ToolCall(
                    turn=turn,
                    name=f"constraint:{constraints_by_output[blank_id].kind}",
                    args={
                        "op": constraints_by_output[blank_id].op,
                        "output": blank_id,
                    },
                    result_preview=_stringify(res.value)[:200],
                    result_full=json.dumps(
                        {"value": res.value, "status": res.status}, default=str
                    ),
                    duration_s=res.wall_time_s,
                    error=res.error,
                )
            )
            continue

        # Case 2: identify by retrieval + extraction.
        res = _identify_blank(
            plan,
            blank,
            state,
            retriever=retriever,
            llm_client=llm_client,
            top_k=top_k,
        )
        state[blank_id] = res
        trace.executed_blanks.append(res)
        trace.wall_times[blank_id] = res.wall_time_s
        trace.tool_calls.append(
            ToolCall(
                turn=turn,
                name="identify_blank",
                args={
                    "blank_id": blank_id,
                    "value_type": blank.value_type,
                },
                result_preview=_stringify(res.value)[:200],
                result_full=json.dumps(
                    {
                        "value": res.value,
                        "evidence_chunk_ids": res.evidence_chunk_ids,
                        "status": res.status,
                    },
                    default=str,
                ),
                duration_s=res.wall_time_s,
                error=res.error,
            )
        )

    target_res = state.get(target.id)
    final_answer = _stringify(target_res.value) if target_res else ""
    return final_answer, trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        # Integer-like floats should render without trailing .0
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(_stringify(x) for x in v)
    return str(v)


def _coerce_value(value: Any, value_type: Optional[str]) -> Any:
    """Best-effort coerce LLM output into the blank's declared type.

    If the LLM returns the right type already, pass through. We avoid
    raising on mismatch here — the ``_identify_blank`` caller decides
    whether a return value of None means ``unknown``.
    """
    if value is None:
        return None
    if value_type == "number":
        return _coerce_number(value)
    if value_type == "date":
        # Stays as-is; calling code (argmax) will coerce to year when needed.
        return str(value)
    if value_type == "list":
        if isinstance(value, list):
            return [_stringify(x) for x in value]
        # Attempt to split a string list.
        return [x.strip() for x in str(value).split(",") if x.strip()]
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        return None
    return value  # entity / attribute / text
