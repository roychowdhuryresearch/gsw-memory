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
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

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


class VerbPhrase(BaseModel):
    """A typed binary relation between two entities."""

    id: str
    phrase: str
    subject_id: str
    object_id: str


class Constraint(BaseModel):
    """Dependency between blanks.

    Only one of the field groups is populated per instance:
    - ``op`` + ``args_blanks`` for ``kind="derived"``.
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
    def _check_executable_refs(self) -> "GSWPlan":
        """Validate references and constraint shapes the executor can run.

        The planner prompt advertises a small executable constraint
        language. Rejecting malformed uses here prevents bad plans from
        reaching orchestration and then looping around unresolved targets.
        """
        all_ids = [e.id for e in self.entities]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("duplicate entity ids in plan")
        ent_ids = set(all_ids)
        blank_ids = {e.id for e in self.blank_entities()}

        vp_ids = [vp.id for vp in self.verb_phrases]
        if len(vp_ids) != len(set(vp_ids)):
            raise ValueError("duplicate verb-phrase ids in plan")
        for vp in self.verb_phrases:
            missing = [
                ref for ref in (vp.subject_id, vp.object_id)
                if ref not in ent_ids
            ]
            if missing:
                raise ValueError(
                    f"verb-phrase {vp.id!r} references unknown ids: {missing}"
                )

        constraint_ids = [c.id for c in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("duplicate constraint ids in plan")

        output_seen: dict[str, str] = {}
        for c in self.constraints:
            if c.output_blank_id:
                if c.output_blank_id not in blank_ids:
                    raise ValueError(
                        f"constraint {c.id!r} output_blank_id "
                        f"{c.output_blank_id!r} is not a blank id"
                    )
                prior = output_seen.get(c.output_blank_id)
                if prior is not None:
                    raise ValueError(
                        f"multiple constraints output to blank "
                        f"{c.output_blank_id!r}: {prior!r}, {c.id!r}"
                    )
                output_seen[c.output_blank_id] = c.id

            if c.kind == "derived":
                if not c.output_blank_id:
                    raise ValueError(
                        f"derived constraint {c.id!r} requires output_blank_id"
                    )
                if not c.op:
                    raise ValueError(f"derived constraint {c.id!r} requires op")
                if not c.args_blanks:
                    raise ValueError(
                        f"derived constraint {c.id!r} requires args_blanks"
                    )
                bad_args = [b for b in c.args_blanks if b not in blank_ids]
                if bad_args:
                    raise ValueError(
                        f"derived constraint {c.id!r} has non-blank args: {bad_args}"
                    )
                if c.op == "div" and len(c.args_blanks) < 2:
                    raise ValueError(
                        f"div constraint {c.id!r} requires at least 2 args"
                    )
                continue

            if c.kind in ("argmax", "argmin"):
                if not c.output_blank_id:
                    raise ValueError(
                        f"{c.kind} constraint {c.id!r} requires output_blank_id"
                    )
                if not c.candidate_entity_ids or not c.sort_by_blank_ids:
                    raise ValueError(
                        f"{c.kind} constraint {c.id!r} requires candidates "
                        "and sort_by_blank_ids"
                    )
                if len(c.candidate_entity_ids) != len(c.sort_by_blank_ids):
                    raise ValueError(
                        f"{c.kind} constraint {c.id!r} requires aligned "
                        "candidate_entity_ids and sort_by_blank_ids"
                    )
                bad_candidates = [
                    eid for eid in c.candidate_entity_ids if eid not in ent_ids
                ]
                bad_sort = [bid for bid in c.sort_by_blank_ids if bid not in blank_ids]
                if bad_candidates:
                    raise ValueError(
                        f"{c.kind} constraint {c.id!r} has unknown candidates: "
                        f"{bad_candidates}"
                    )
                if bad_sort:
                    raise ValueError(
                        f"{c.kind} constraint {c.id!r} has non-blank sort ids: "
                        f"{bad_sort}"
                    )
                continue

            if c.kind in ("equals", "in_list", "gt", "lt"):
                if not c.output_blank_id:
                    raise ValueError(
                        f"relational constraint {c.id!r} requires output_blank_id"
                    )
                missing_refs = [
                    field for field, ref in (
                        ("left_ref", c.left_ref),
                        ("right_ref", c.right_ref),
                    )
                    if not ref
                ]
                if missing_refs:
                    raise ValueError(
                        f"relational constraint {c.id!r} requires "
                        f"{', '.join(missing_refs)}"
                    )
                bad_refs = [
                    ref for ref in (c.left_ref, c.right_ref)
                    if ref and ref not in ent_ids
                ]
                if bad_refs:
                    raise ValueError(
                        f"relational constraint {c.id!r} references unknown ids: "
                        f"{bad_refs}"
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
            inputs.update(x for x in c.args_blanks if x in blanks)
        elif c.kind in ("argmax", "argmin"):
            inputs.update(x for x in c.sort_by_blank_ids if x in blanks)
        elif c.kind in ("equals", "in_list", "gt", "lt"):
            for x in (c.left_ref, c.right_ref):
                if x and x in blanks:
                    inputs.add(x)

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


def _resolve_constraint_ref(
    ref: Optional[str],
    plan: GSWPlan,
    state: dict[str, BlankResult],
) -> Any:
    """Resolve a constraint ref to either a blank value or filled name."""
    if not ref:
        return None
    ent = plan.entity_by_id(ref)
    if ent.kind == "blank":
        res = state.get(ref)
        if res is None or res.status != "resolved":
            return None
        return res.value
    return ent.name


def _coerce_relational_number(value: Any) -> Optional[float]:
    """Numeric coercion for gt/lt; list values use their first item."""
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return _coerce_constraint_number(value)


def _contains_value(container: Any, needle: Any) -> bool:
    """Best-effort membership with case-insensitive string matching."""
    if container is None or needle is None:
        return False
    if isinstance(container, (list, tuple, set)):
        return any(_contains_value(item, needle) for item in container)
    return str(container).strip().lower() == str(needle).strip().lower()


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
        args = [state.get(b) for b in constraint.args_blanks]
        if any(a is None or a.status != "resolved" or a.value is None for a in args):
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )
        op = constraint.op
        if op == "concat":
            out_val = "; ".join(_stringify(a.value) for a in args)
            return BlankResult(
                blank_id=out_id,
                value=out_val,
                status="resolved",
                wall_time_s=round(time.time() - t0, 3),
            )
        if op == "count":
            if len(args) == 1:
                val = _count_constraint_value(args[0].value)
            else:
                val = len(args)
            return BlankResult(
                blank_id=out_id,
                value=val,
                status="resolved",
                wall_time_s=round(time.time() - t0, 3),
            )
        if op == "sum":
            nums = [_coerce_sum_term(a.value) for a in args]
        else:
            nums = [_coerce_constraint_number(a.value) for a in args]
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
            val = nums[0]
            for denom in nums[1:]:
                if denom == 0:
                    return BlankResult(
                        blank_id=out_id,
                        status="unknown",
                        wall_time_s=round(time.time() - t0, 3),
                    )
                val /= denom
        elif op == "round_nearest":
            base = nums[0]
            nearest = nums[1] if len(nums) > 1 else 10.0
            if nearest == 0:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            # Round halves away from zero for predictable "nearest ten"
            # behavior on positive answer counts.
            import math

            sign = -1 if base < 0 else 1
            val = sign * math.floor(abs(base) / abs(nearest) + 0.5) * abs(nearest)
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

    if constraint.kind in ("equals", "in_list", "gt", "lt"):
        left = _resolve_constraint_ref(constraint.left_ref, plan, state)
        right = _resolve_constraint_ref(constraint.right_ref, plan, state)
        if left is None or right is None:
            return BlankResult(blank_id=out_id, status="unknown")

        result_bool: Optional[bool] = None
        passthrough: Any = None
        if constraint.kind == "equals":
            result_bool = _contains_value(left, right)
            passthrough = left
        elif constraint.kind == "in_list":
            if isinstance(right, (list, tuple, set)):
                result_bool = _contains_value(right, left)
                passthrough = left
            elif isinstance(left, (list, tuple, set)):
                result_bool = _contains_value(left, right)
                passthrough = right
            else:
                result_bool = _contains_value(right, left)
                passthrough = left
        elif constraint.kind in ("gt", "lt"):
            left_num = _coerce_relational_number(left)
            right_num = _coerce_relational_number(right)
            if left_num is None or right_num is None:
                return BlankResult(blank_id=out_id, status="unknown")
            result_bool = (
                left_num > right_num
                if constraint.kind == "gt"
                else left_num < right_num
            )
            passthrough = left

        if result_bool is None:
            return BlankResult(blank_id=out_id, status="unknown")
        out_ent = plan.entity_by_id(out_id)
        if out_ent.value_type == "bool":
            out_val: Any = result_bool
        elif result_bool:
            out_val = passthrough
        else:
            return BlankResult(blank_id=out_id, status="unknown")
        return BlankResult(
            blank_id=out_id,
            value=out_val,
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
