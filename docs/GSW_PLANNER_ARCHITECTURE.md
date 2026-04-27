# GSW-fragment question planner — architecture reference

> **Scope.** Canonical reference for `ours_gsw_planner_v1` (Phase-1, prompt-only) as implemented under `research_agent/src/research_agent/adapters/ours/`. Covers schema, planner prompt, executor algorithm, fallback path, trace shape, and worked examples. Read this before touching the planner code.

> **Version history.**
> - **v1** (2026-04-22): first release — schema, 5 FRAMES few-shots, executor, fallback.
> - **v2** (2026-04-22): `Entity.role` + `Entity.state` fields, `GSWPlan` no-dangling model validator, three-step thinking protocol in the prompt, topological-order visualisation in Streamlit inspectors, shared `_planner_viz.py` helper. 14 schema + 10 few-shot-validation unit tests.
> - **v3** (2026-04-23): three **hard rules** in `PLANNER_SYSTEM` (grounding test, date-anchor rule, collective-expansion escape hatch), tightened role vocabulary, all 5 few-shots replaced with **hand-authored synthetic examples** (the v2 few-shots were sampled from FRAMES dev — test-set leakage), inspector additions for `state` breadcrumb handling and date-coverage soft check.

---

## Table of contents

1. [Motivation](#motivation)
2. [Pipeline overview](#pipeline-overview)
3. [Schema reference](#schema-reference)
4. [Planner prompt](#planner-prompt)
5. [Executor algorithm](#executor-algorithm)
6. [Fallback path](#fallback-path)
7. [Trace shape](#trace-shape)
8. [Worked examples](#worked-examples)
9. [Known limitations](#known-limitations)
10. [File layout](#file-layout)
11. [References](#references)

---

## Motivation

The FRAMES pilot (E1–E13, 15 completed cells, 450 question-runs — `docs/FRAMES_PILOT_FAILURE_ANALYSIS.md`) surfaced a shared failure mode: every competitor system collapses on hard multi-hop questions into either **hallucination** (fabricated single-shot answer) or **loop** (agent never converges). Root cause: every baseline relies on the LLM to carry the plan across turns via reasoning tokens; rich questions overwhelm that budget.

`ours_gsw_planner_v1` externalises the plan as a **structured GSW-fragment DAG** emitted up-front by one LLM call. The DAG is then executed deterministically by Python code, with one LLM extraction call per typed blank. The planner and the executor are decoupled: the planner emits a schema; the executor fills blanks; neither drifts.

The design pulls from:
- **ReWOO** (`arxiv 2305.18323`) — variable substitution via `#E_k` references (adapted here into typed VP + blank edges).
- **LLM Compiler** (`arxiv 2312.04511`) — DAG dependency encoding + parallel fill (sequential in Phase 1, parallel in Phase 2).
- **Plan-over-Graph** (`arxiv 2502.14563`) — strict JSON schema + recipe for SFT + DPO training (Phase 2).
- **ToQD** (COLING 2025) — topology-graph question decomposition.
- **QDMR** (`arxiv 2001.11770`) — operator vocabulary informing blank value-types.

Full 12-paper lit review in `docs/QUESTION_PLANNER_DAG_LITERATURE_REVIEW.md`.

---

## Pipeline overview

```
          ┌──────────────────────────┐
question ─▶│ Planner (1 LLM call)     │
          │  build_planner_messages   │
          │  + 5 few-shots            │
          └──────────────┬───────────┘
                         │ GSWPlan JSON
                         ▼
          ┌──────────────────────────┐
          │ Pydantic validate        │─── parse fail ──▶ 1 repair retry
          │  GSWPlan.model_validate  │           │
          └──────────────┬───────────┘           │ still fail
                         │ valid plan            ▼
                         ▼                  ┌─────────────────┐
          ┌──────────────────────────┐      │ flat ours_gsw_v1│
          │ Executor (pure Python)   │      │ fallback adapter│
          │  topo-sort blanks        │      └─────────────────┘
          │  for each blank:         │            │
          │   • identify (1 LLM)     │            ▼
          │   • project (1 LLM)      │      Trajectory w/
          │   • derive (Python)      │      fallback_flag=True
          └──────────────┬───────────┘
                         │ state[target]
                         ▼
                  final_answer
```

**Key invariants.**
- Planner is called exactly **once** (plus at most one repair retry). No interleaved retrieval.
- Executor makes **at most N LLM calls**, where N = number of blanks that need identification / projection (numeric and argmax constraints are Python-only).
- Failures are structured: parse / validator failure → repair → fallback; execution failure → fallback; empty retrieval → `unknown` value on that blank (cascades through the DAG).
- **"Parse failure"** in the diagram above now includes v2 schema-validator rejections. A plan with a dangling entity triggers `ValidationError`, which the adapter routes through the same repair-retry path as malformed JSON. This gives the LLM a targeted correction signal ("entity X is not referenced by any edge") instead of silently falling back.

---

## Schema reference

Source: `research_agent/src/research_agent/adapters/ours/_planner_exec.py`.

### `Entity`

```python
class Entity(BaseModel):
    id: str
    kind: Literal["filled", "blank"]
    name: Optional[str] = None
    value_type: Optional[
        Literal["date", "number", "entity", "attribute", "list", "text", "bool"]
    ] = None
    is_target: bool = False
    category: bool = False
    # v2 additions — informational metadata (executor ignores both).
    role: Optional[str] = None
    state: Optional[str] = None
```

| field | for | purpose |
|---|---|---|
| `id` | all | short label, usually `"e1"` for filled, `"b1"` for blank |
| `kind` | all | `"filled"` = extracted from question via NER; `"blank"` = to be resolved by retrieval |
| `name` | filled | the surface form — `"Pablo Picasso"`, `"Portland Maine"` |
| `value_type` | blank | the expected shape of the resolved value |
| `is_target` | blank | exactly **one** blank must have `is_target=True`; that blank's resolved value is the final answer |
| `category` | filled | true when the name is a role/category (`"comic book writer"`) rather than a named entity; used as a retrieval hint |
| `role` *(v2)* | all | short role label drawn from a seed vocabulary (see [Role vocabulary](#role-vocabulary)). Surfaces *why* each entity is in the plan. Executor ignores; inspectors + audit rely on it. |
| `state` *(v2, activated v3)* | filled | provenance breadcrumb. Currently one valid form: `"expanded_from_collective:<group-name>"` — declares that this filled entity was expanded from a collective group named in the question (Hard rule 3). |

#### Role vocabulary

Seed vocabulary introduced in v2 and tightened in v3. Not exhaustive — planner may invent new roles when needed, but the seed list covers 80%+ of FRAMES-style questions.

| role | applies to | use when |
|---|---|---|
| `subject` | filled | main named entity the question is about. If it's a category/profession, use `scope-filter` instead. |
| `scope-filter` | filled | categorical predicate narrowing a set (professions, group memberships, temporal boundaries). Usually paired with `category=true`. |
| `year-anchor` / `date-anchor` | filled | date or year token that appears literally in the question text. |
| `as-of-date` | filled | snapshot-in-time anchor (“as of August 2024”). |
| `list-header` | filled | collective name that owns a list-member blank. Always paired with `category=true`. |
| `candidate` | filled | sibling in an argmax / argmin constraint. Never use for a single item. |
| `target` | blank | the final answer (pairs with `is_target=true`). |
| `bridge-entity` / `bridge-date` / `bridge-number` / `bridge-attribute` | blank | intermediate blank the executor resolves to link question to answer. |
| `comparison-output` / `aggregate-output` | blank | blank produced by a constraint rather than retrieval. |
| `list-member` | blank | list-valued blank holding members of a list-header collective. |

### `VerbPhrase`

```python
class VerbPhrase(BaseModel):
    id: str
    phrase: str
    subject_id: str
    object_id: str
```

Binary typed relation between two entities. The **phrase** is a short snake_case predicate (`"died_in"`, `"entered_service_in"`, `"has_population"`). Phrases do double duty:

- `filled → blank` VP: identification signal for the blank (retrieval queries combine `filled.name` + `phrase`).
- `blank_entity → blank` VP: projection edge; the object blank is filled **after** the subject blank resolves, using the subject's value as a retrieval anchor.
- `blank → filled` VP: constraint for identification (the blank must satisfy this relation with the filled endpoint).

### `Constraint`

```python
class Constraint(BaseModel):
    id: str
    kind: Literal["derived", "argmax", "argmin", "equals", "in_list", "gt", "lt"]
    op: Optional[Literal["diff","sum","avg","max","min","count","concat"]] = None
    args_blanks: list[str] = []
    candidate_entity_ids: list[str] = []
    sort_by_blank_ids: list[str] = []
    left_ref: Optional[str] = None
    right_ref: Optional[str] = None
    output_blank_id: Optional[str] = None
```

Only one field group is populated per instance.

| kind | fields populated | behaviour |
|---|---|---|
| `derived` | `op`, `args_blanks`, `output_blank_id` | Compute `op(blanks[args_blanks])` in Python; write the numeric/str result to `output_blank_id` |
| `argmax` / `argmin` | `candidate_entity_ids`, `sort_by_blank_ids`, `output_blank_id` | Select the entity in `candidate_entity_ids[i]` whose aligned blank `sort_by_blank_ids[i]` has the max/min numeric value. Write its name to `output_blank_id`. |
| `equals` / `in_list` / `gt` / `lt` | `left_ref`, `right_ref`, `output_blank_id` | Phase-2 relational constraints; not used by Phase-1 executor. Currently evaluate to `unknown`. |

### `GSWPlan`

```python
class GSWPlan(BaseModel):
    entities: list[Entity]
    verb_phrases: list[VerbPhrase] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
```

Exactly one entity has `is_target=True`. All `subject_id` / `object_id` / `output_blank_id` / `args_blanks` / `candidate_entity_ids` / `sort_by_blank_ids` must resolve to valid entity ids.

#### No-dangling validator (v2)

`GSWPlan` carries a `@model_validator(mode="after")` that rejects any plan where some entity id does **not** appear in at least one verb-phrase endpoint or one constraint reference. A dangling entity is unreachable by the executor — no retrieval signal can use it and no constraint can consume/produce it — so it contributes nothing to answering the question.

```python
@model_validator(mode="after")
def _check_no_dangling(self) -> "GSWPlan":
    referenced: set[str] = set()
    for vp in self.verb_phrases:
        referenced.add(vp.subject_id); referenced.add(vp.object_id)
    for c in self.constraints:
        referenced.update(c.args_blanks)
        referenced.update(c.candidate_entity_ids)
        referenced.update(c.sort_by_blank_ids)
        for ref in (c.left_ref, c.right_ref, c.output_blank_id):
            if ref: referenced.add(ref)
    dangling = [e.id for e in self.entities if e.id not in referenced]
    if dangling:
        raise ValueError(f"dangling entities (no verb-phrase or constraint edge): {dangling}")
    return self
```

The adapter catches `ValidationError` from this and triggers the single-shot repair retry (same path as malformed JSON). Since v2, the ratio of structurally-broken plans reaching the fallback has dropped meaningfully because the error path now gives the LLM a targeted correction signal.

---

## Planner prompt

Source: `research_agent/src/research_agent/adapters/ours/_planner_prompts.py`.

Structure: `[SYSTEM] + [USER (question + 5 few-shots rendered inline)]`. One LLM call. No tools, no JSON-mode (relies on balanced-brace extractor for robustness).

### `PLANNER_SYSTEM` structure (v3)

The system prompt leads with *purpose* ("a downstream executor walks your graph") so the model understands what its output will be used for, then teaches a three-step protocol, then enforces three hard rules. Verbatim text is in the source; here is the structural outline:

1. **Why this matters** — one paragraph explaining that a Python executor walks the plan and that every retrieval signal / constraint input must be drawn on the graph.
2. **How to think (three-step protocol).** Step 1 — entities with roles; Step 2 — verb-phrases; Step 3 — constraints (only when needed).
3. **Role vocabulary.** Seed list for both filled and blank entities (see [Role vocabulary](#role-vocabulary) above).
4. **Standardised predicates for temporal anchors.** `in_year`, `as_of_date`, `signed_in_year`, etc. — the planner should prefer these over ad-hoc predicates so retrieval signals are consistent.
5. **No-dangling rule.** Re-states the `@model_validator` requirement so the LLM self-checks.
6. **Hard rule 1 — Grounding test.** Every filled `entity.name` MUST appear as a substring of the question text (whitespace-normalised, case-insensitive). Exceptions only via the collective-expansion escape hatch.
7. **Hard rule 2 — Date-anchor rule.** Any date, year, month, or temporal-anchor token in the question text MUST become a filled entity with role `year-anchor` / `date-anchor` / `as-of-date` and MUST appear in at least one verb-phrase.
8. **Hard rule 3 — Collective-expansion escape hatch.** When the question names a finite well-known group by collective name, there are two valid patterns:
   - **Pattern 1 (preferred)**: emit the collective as `filled, category=true, role=list-header` + a `blank, role=list-member, value_type=list` + a `members` VP. Executor resolves the list at runtime.
   - **Pattern 2 (escape)**: planner-time expansion is allowed ONLY when downstream reasoning requires argmax/argmin/arithmetic over individual members. Every expanded entity MUST carry `state="expanded_from_collective:<group-name>"` and the collective itself MUST remain in the plan as a `list-header` entity wired via `member_of` VPs.
9. **Field spec (strict)** — `Entity`, `VerbPhrase`, `Constraint`, output-format.

The hard rules correspond to concrete, measurable failure modes observed in the v2 dump audit (see [Known limitations](#known-limitations)).

### Few-shots (5 — hand-authored synthetic, v3)

The v1 and v2 few-shots were drawn from FRAMES dev split (qids `q154, q70, q549, q293, q796`) — each is inside the evaluation set. v3 replaces them with unambiguously-fictional synthetic examples that don't trigger any world-knowledge priors the LLM could use to cheat:

| # | archetype | teaches | question sketch |
|---:|---|---|---|
| 1 | Temporal bridge | date-anchor grounding (`role=year-anchor`) | *"Which song did the band Hollowmoor Drive release in the year the Treaty of Verdane was signed?"* |
| 2 | Attribute arithmetic diff | `role=as-of-date` + `as_of_date` predicate + `derived.diff` | *"As of March 1, 2047, what is the difference in population between the cities Kessington and Raleford?"* |
| 3 | Enumerated argmax | candidates must be named in question text (no priors) | *"Of the following four corporate treasurers, who was appointed most recently — Ada Voss, Petro Milan, Lin Karaczynski, or Sybil Quine?"* |
| 4 | Entity ID via compound scope | three scope-filters identify one target | *"Who was the woman born in Astra Bay who founded the Wren Clinic and authored the novel Tidebound?"* |
| 5 | Collective expansion with state breadcrumb | Hard rule 3 escape hatch | *"Who lived longer — one of the Three Founders of Acme Corp or Dorian Vexley?"* (emits 3 expanded founders each with `state="expanded_from_collective:Three Founders of Acme Corp"`) |

Each few-shot comes bundled with a **"How to think"** block that narrates the three-step protocol in prose before the Plan JSON, teaching the model *how* to derive a plan from a question rather than just what a plan looks like. Every few-shot plan passes `GSWPlan.model_validate` — enforced by `tests/ours/test_planner_prompts.py` (10 shape-validation tests).

The verbatim content of each few-shot (as injected into the user prompt, without the surrounding Python formatting) follows.

#### Few-shot #1 — temporal bridge

**Question.** *Which song did the band Hollowmoor Drive release in the year the Treaty of Verdane was signed?*

**How to think.**

```
Step 1 — Entities + roles (grounding test: both names appear in the question):
  • "Hollowmoor Drive" → filled, role=subject.
  • "Treaty of Verdane" → filled, role=year-anchor  (the
    temporal anchor phrase in the question is "in the year
    the Treaty of Verdane was signed" — the Treaty IS the
    anchor whose year we resolve).
  • Year the Treaty was signed → blank, role=bridge-date, value_type=date.
  • The song → blank, role=target, value_type=entity, is_target=true.
Step 2 — Verb-phrases (every entity participates; none dangle):
  • vp1: (Treaty of Verdane, signed_in_year, ?year) — identifies ?year.
  • vp2: (Hollowmoor Drive, released_in_year, ?year) — ties the song to the bridge.
  • vp3: (Hollowmoor Drive, released, ?song) — identifies ?song.
Step 3 — Constraints: none. Pure VP chain; the bridging year is wired
via vp1 and reused in vp2 as a retrieval signal.
```

**Plan.**

```json
{
  "entities": [
    {"id": "e1", "kind": "filled", "name": "Hollowmoor Drive", "role": "subject"},
    {"id": "e2", "kind": "filled", "name": "Treaty of Verdane", "role": "year-anchor"},
    {"id": "b_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
    {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": true}
  ],
  "verb_phrases": [
    {"id": "vp1", "phrase": "signed_in_year", "subject_id": "e2", "object_id": "b_year"},
    {"id": "vp2", "phrase": "released_in_year", "subject_id": "e1", "object_id": "b_year"},
    {"id": "vp3", "phrase": "released", "subject_id": "e1", "object_id": "t"}
  ],
  "constraints": []
}
```

#### Few-shot #2 — as-of-date arithmetic diff

**Question.** *As of March 1, 2047, what is the difference in population between the cities Kessington and Raleford?*

**How to think.**

```
Step 1 — Entities + roles:
  • "March 1, 2047" → filled, role=as-of-date. (Hard rule 2:
    explicit date token; must become a filled entity.)
  • "Kessington" → filled, role=subject.
  • "Raleford" → filled, role=subject.
  • Kessington's population → blank, role=bridge-number, value_type=number.
  • Raleford's population → blank, role=bridge-number, value_type=number.
  • The difference → blank, role=aggregate-output, value_type=number, is_target=true.
Step 2 — Verb-phrases (the as-of-date entity must be wired;
we attach it to both subjects so it is not dangling):
  • vp1: (Kessington, has_population, ?pop_k) — identify Kessington's population.
  • vp2: (Raleford, has_population, ?pop_r)  — identify Raleford's population.
  • vp3: (Kessington, as_of_date, March 1, 2047) — scope anchor.
  • vp4: (Raleford, as_of_date, March 1, 2047)  — scope anchor.
Step 3 — Constraints:
  The question asks for a numeric difference → derived.diff op.
  c1: derived op=diff, args_blanks=[?pop_k, ?pop_r], output=t.
```

**Plan.**

```json
{
  "entities": [
    {"id": "e_date", "kind": "filled", "name": "March 1, 2047", "role": "as-of-date"},
    {"id": "e1", "kind": "filled", "name": "Kessington", "role": "subject"},
    {"id": "e2", "kind": "filled", "name": "Raleford", "role": "subject"},
    {"id": "b_pop_k", "kind": "blank", "role": "bridge-number", "value_type": "number"},
    {"id": "b_pop_r", "kind": "blank", "role": "bridge-number", "value_type": "number"},
    {"id": "t", "kind": "blank", "role": "aggregate-output", "value_type": "number", "is_target": true}
  ],
  "verb_phrases": [
    {"id": "vp1", "phrase": "has_population", "subject_id": "e1", "object_id": "b_pop_k"},
    {"id": "vp2", "phrase": "has_population", "subject_id": "e2", "object_id": "b_pop_r"},
    {"id": "vp3", "phrase": "as_of_date", "subject_id": "e1", "object_id": "e_date"},
    {"id": "vp4", "phrase": "as_of_date", "subject_id": "e2", "object_id": "e_date"}
  ],
  "constraints": [
    {"id": "c1", "kind": "derived", "op": "diff",
     "args_blanks": ["b_pop_k", "b_pop_r"], "output_blank_id": "t"}
  ]
}
```

#### Few-shot #3 — enumerated argmax

**Question.** *Of the following four corporate treasurers, who was appointed most recently — Ada Voss, Petro Milan, Lin Karaczynski, or Sybil Quine?*

**How to think.**

```
Step 1 — Entities + roles:
  The question names four candidates by surface form. All four
  pass the grounding test. We do NOT need to expand any collective
  — the candidates are spelled out.
  • "Ada Voss" → filled, role=candidate.
  • "Petro Milan" → filled, role=candidate.
  • "Lin Karaczynski" → filled, role=candidate.
  • "Sybil Quine" → filled, role=candidate.
  • Four unknown appointment dates → four blanks, role=bridge-date, value_type=date.
  • The answer → blank, role=target, value_type=entity, is_target=true.
Step 2 — Verb-phrases (four parallel projections):
  • vp1..vp4: (candidate_k, appointed_on, ?date_k).
  The target t does NOT get a VP — it's produced by the argmax constraint.
Step 3 — Constraints:
  "most recently" → argmax. Add c1 with candidate_entity_ids=[e1..e4],
  sort_by_blank_ids=[b1..b4], output_blank_id=t. Executor resolves
  dates in parallel, then picks the candidate whose date is latest.
```

**Plan.**

```json
{
  "entities": [
    {"id": "e1", "kind": "filled", "name": "Ada Voss", "role": "candidate"},
    {"id": "e2", "kind": "filled", "name": "Petro Milan", "role": "candidate"},
    {"id": "e3", "kind": "filled", "name": "Lin Karaczynski", "role": "candidate"},
    {"id": "e4", "kind": "filled", "name": "Sybil Quine", "role": "candidate"},
    {"id": "b1", "kind": "blank", "role": "bridge-date", "value_type": "date"},
    {"id": "b2", "kind": "blank", "role": "bridge-date", "value_type": "date"},
    {"id": "b3", "kind": "blank", "role": "bridge-date", "value_type": "date"},
    {"id": "b4", "kind": "blank", "role": "bridge-date", "value_type": "date"},
    {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": true}
  ],
  "verb_phrases": [
    {"id": "vp1", "phrase": "appointed_on", "subject_id": "e1", "object_id": "b1"},
    {"id": "vp2", "phrase": "appointed_on", "subject_id": "e2", "object_id": "b2"},
    {"id": "vp3", "phrase": "appointed_on", "subject_id": "e3", "object_id": "b3"},
    {"id": "vp4", "phrase": "appointed_on", "subject_id": "e4", "object_id": "b4"}
  ],
  "constraints": [
    {"id": "c1", "kind": "argmax",
     "candidate_entity_ids": ["e1", "e2", "e3", "e4"],
     "sort_by_blank_ids":   ["b1", "b2", "b3", "b4"],
     "output_blank_id": "t"}
  ]
}
```

#### Few-shot #4 — compound scope identification

**Question.** *Who was the woman born in Astra Bay who founded the Wren Clinic and authored the novel Tidebound?*

**How to think.**

```
Step 1 — Entities + roles (only the named things become entities;
generic words like "woman" / "novel" are descriptors, not
first-class entities):
  • "Astra Bay" → filled, role=scope-filter (place-of-birth filter).
  • "Wren Clinic" → filled, role=scope-filter (founder-of filter).
  • "Tidebound" → filled, role=scope-filter (author-of filter).
  • The person → blank, role=target, value_type=entity, is_target=true.
    She is identified by the intersection of the three scope
    filters (born + founded + authored).
Step 2 — Verb-phrases (all three filled entities wire INTO the
target blank; the target resolves via composite retrieval):
  • vp1: (?target, born_in, Astra Bay).
  • vp2: (?target, founded, Wren Clinic).
  • vp3: (?target, authored, Tidebound).
Step 3 — Constraints: none. Entity identification via
intersection of scope filters is a pure VP pattern.
```

**Plan.**

```json
{
  "entities": [
    {"id": "e1", "kind": "filled", "name": "Astra Bay", "role": "scope-filter"},
    {"id": "e2", "kind": "filled", "name": "Wren Clinic", "role": "scope-filter"},
    {"id": "e3", "kind": "filled", "name": "Tidebound", "role": "scope-filter"},
    {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": true}
  ],
  "verb_phrases": [
    {"id": "vp1", "phrase": "born_in", "subject_id": "t", "object_id": "e1"},
    {"id": "vp2", "phrase": "founded", "subject_id": "t", "object_id": "e2"},
    {"id": "vp3", "phrase": "authored", "subject_id": "t", "object_id": "e3"}
  ],
  "constraints": []
}
```

#### Few-shot #5 — collective-expansion escape hatch

**Question.** *Who lived longer — one of the Three Founders of Acme Corp or Dorian Vexley?*

**How to think.**

```
Step 1 — Entities + roles (this question triggers Hard rule 3,
the collective-expansion escape hatch — the comparison requires
argmax over individual lifespans, which Pattern 1 cannot express):
  • "Three Founders of Acme Corp" → filled, role=list-header,
    category=true. The collective itself is grounded in the
    question text.
  • Three expanded founders → each filled, role=candidate,
    with state="expanded_from_collective:Three Founders of Acme Corp".
    These names do NOT need to satisfy the grounding test — the
    state breadcrumb is their provenance declaration.
    (Here we use invented names Mira Tallow, Essen Quorn,
    Varik Opsahl as stand-ins; in a real plan these would be
    the actual founders' names drawn from world knowledge.)
  • "Dorian Vexley" → filled, role=candidate (surface in question).
  • Four lifespans → blanks, role=bridge-number, value_type=number.
  • The longer-lived person → blank, role=target, value_type=entity,
    is_target=true.
Step 2 — Verb-phrases (the collective must be wired or it dangles;
we wire it via member_of edges from each expanded founder):
  • vp1..vp3: (expanded_founder_k, member_of, Three Founders of Acme Corp).
  • vp4..vp7: (person_k, lived_years, ?lifespan_k) — one per
    person (3 expanded + Dorian Vexley).
Step 3 — Constraints:
  c1: argmax over all four candidates,
      candidate_entity_ids=[e_f1,e_f2,e_f3,e_vexley],
      sort_by_blank_ids=[b_lf1,b_lf2,b_lf3,b_lf_vexley],
      output_blank_id=t.
```

**Plan.**

```json
{
  "entities": [
    {"id": "e_collective", "kind": "filled",
     "name": "Three Founders of Acme Corp",
     "role": "list-header", "category": true},
    {"id": "e_f1", "kind": "filled", "name": "Mira Tallow", "role": "candidate",
     "state": "expanded_from_collective:Three Founders of Acme Corp"},
    {"id": "e_f2", "kind": "filled", "name": "Essen Quorn", "role": "candidate",
     "state": "expanded_from_collective:Three Founders of Acme Corp"},
    {"id": "e_f3", "kind": "filled", "name": "Varik Opsahl", "role": "candidate",
     "state": "expanded_from_collective:Three Founders of Acme Corp"},
    {"id": "e_vexley", "kind": "filled", "name": "Dorian Vexley", "role": "candidate"},
    {"id": "b_lf1", "kind": "blank", "role": "bridge-number", "value_type": "number"},
    {"id": "b_lf2", "kind": "blank", "role": "bridge-number", "value_type": "number"},
    {"id": "b_lf3", "kind": "blank", "role": "bridge-number", "value_type": "number"},
    {"id": "b_lf_v", "kind": "blank", "role": "bridge-number", "value_type": "number"},
    {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": true}
  ],
  "verb_phrases": [
    {"id": "vp1", "phrase": "member_of", "subject_id": "e_f1", "object_id": "e_collective"},
    {"id": "vp2", "phrase": "member_of", "subject_id": "e_f2", "object_id": "e_collective"},
    {"id": "vp3", "phrase": "member_of", "subject_id": "e_f3", "object_id": "e_collective"},
    {"id": "vp4", "phrase": "lived_years", "subject_id": "e_f1", "object_id": "b_lf1"},
    {"id": "vp5", "phrase": "lived_years", "subject_id": "e_f2", "object_id": "b_lf2"},
    {"id": "vp6", "phrase": "lived_years", "subject_id": "e_f3", "object_id": "b_lf3"},
    {"id": "vp7", "phrase": "lived_years", "subject_id": "e_vexley", "object_id": "b_lf_v"}
  ],
  "constraints": [
    {"id": "c1", "kind": "argmax",
     "candidate_entity_ids": ["e_f1", "e_f2", "e_f3", "e_vexley"],
     "sort_by_blank_ids":   ["b_lf1", "b_lf2", "b_lf3", "b_lf_v"],
     "output_blank_id": "t"}
  ]
}
```

### Repair prompt (v3)

`build_repair_messages(question, bad_response, error)` appends the original messages + the assistant's bad response + a short system message asking for a corrected JSON. Called **once** on parse / validator failure. The v3 repair system text explicitly names all five rules the corrected plan must honour:

1. Valid `GSWPlan` schema.
2. No dangling entities.
3. `role` on every entity.
4. Grounding test — every filled `name` in the question text unless carrying the `state` breadcrumb.
5. Date-anchor rule — explicit dates promoted to filled entities.

---

## Executor algorithm

Source: `research_agent/src/research_agent/adapters/ours/_planner_exec.py::execute()`.

### 1. Dependency DAG

```python
def build_dependency_graph(plan) -> dict[str, set[str]]:
    # For each blank B, return the set of blanks it depends on.
    deps = {b.id: set() for b in plan.blank_entities()}

    # Constraint edges: output depends on all inputs.
    for c in plan.constraints:
        inputs = _collect_inputs(c)  # args_blanks | sort_by_blank_ids | left/right_ref
        if c.output_blank_id:
            deps[c.output_blank_id] |= inputs - {c.output_blank_id}

    # VP projection edges: if B is an entity-blank and has an outgoing VP
    # to another blank B', then B' depends on B.
    for vp in plan.verb_phrases:
        sub = ent_by_id[vp.subject_id]; obj = ent_by_id[vp.object_id]
        if sub.kind == "blank" and sub.value_type == "entity" and obj.kind == "blank":
            deps[obj.id].add(sub.id)

    return deps
```

### 2. Topological sort (Kahn's)

```python
def topological_sort_blanks(plan) -> list[str]:
    deps = build_dependency_graph(plan)
    indeg = {b: len(ds) for b, ds in deps.items()}
    children = inverse_of(deps)
    q = deque([b for b, d in indeg.items() if d == 0])
    order = []
    while q:
        b = q.popleft(); order.append(b)
        for child in children[b]:
            indeg[child] -= 1
            if indeg[child] == 0: q.append(child)
    if len(order) != len(indeg):
        raise ExecutionError(kind="cyclic_plan")
    return order
```

### 3. Per-blank fill

For each blank in topological order, the executor chooses one of four dispatch paths:

1. **Constraint-produced** (`blank_id` ∈ `constraints_by_output`): Python computation using resolved dependencies.
   - `derived op=diff/sum/avg/max/min/count/concat`
   - `argmax` / `argmin`
2. **Identification** (blank has incoming VP edges from filled or already-resolved entities): BM25 retrieval with composite query + one LLM extraction call asking for a value of `blank.value_type`.
3. **Projection** (blank is the object of an outgoing VP from an already-resolved entity-blank): in Phase 1 this is handled by the identification pass applied to the object blank once the subject resolves. Retrieval signals include the resolved subject's name.
4. **Unknown cascade** (no identification signals available, or retrieval returns zero chunks, or extraction returns null): blank gets `status="unknown"` and any dependents receive `None` as their input. Derived ops over `None` inputs propagate the unknown.

The LLM extraction prompt (`_EXTRACT_SYSTEM` + `_EXTRACT_USER`) is a separate two-message call per identification/projection. It's given the blank's `query`, `value_type`, any resolved-blank context, and the retrieved chunks. It returns `{"value": ..., "evidence_chunk_ids": [...]}`.

### 4. Value-type coercion

After extraction, values are coerced into the declared `value_type`:

- `number` → `float` via `_coerce_number` (strips commas, handles int/float/string).
- `date` → pass-through as string; argmax/argmin constraints extract a 4-digit year via `_coerce_year` at ranking time.
- `list` → split on comma if needed.
- `bool` → `true/yes/1` → True; `false/no/0` → False; else None.
- `entity` / `attribute` / `text` → pass-through as string.

---

## Fallback path

Source: `research_agent/src/research_agent/adapters/ours/gsw_planner_v1.py::_run_fallback`.

Triggered on:
- Planner LLM call exception → `fallback_reason="planner_call_error:..."`
- Parse failure + repair parse failure → `fallback_reason="parse_failure:..."`
- Executor raises `ExecutionError(kind="cyclic_plan" | "no_target" | "bad_ref" | "empty_plan")` → `fallback_reason="execution_error:<kind>"`

Fallback delegates to `research_agent.adapters.baselines.ours_gsw_v1.OursGSWv1Adapter` (flat decomposition pipeline). The resulting `Trajectory` is stamped:

```python
traj.extra["fallback_flag"] = True
traj.extra["fallback_reason"] = reason
traj.system_id = "ours_gsw_planner_v1"  # preserve registry bookkeeping
```

so the downstream aggregator can separate planner runs from fallback runs.

---

## Trace shape

`Trajectory.extra` keys populated by the planner adapter:

| key | type | populated when |
|---|---|---|
| `plan_json` | `dict` | plan successfully parsed |
| `executed_blanks` | `list[dict]` | per blank: `{blank_id, value, status, evidence_chunk_ids, llm_calls, wall_time_s}` |
| `per_blank_wall_times` | `dict[str, float]` | per-blank wall (convenience) |
| `fallback_flag` | `bool` | always present; True when flat fallback was used |
| `fallback_reason` | `str` | present when `fallback_flag=True` |
| `parse_error` | `str` | present when parse (post-repair) failed |
| `raw_planner_output` | `str` | present when parse failed (first 2k chars of raw LLM) |
| `execution_error` | `str` | present when executor raised |
| `stopped_reason` | `str` | `"finished"`, `"finished_unknown"`, `"llm_error"`, `"parse_failure"`, `"execution_error"` |
| `gold_articles` | `list` | echo from harness |

First-class `Trajectory` fields:
- `tool_calls` — one `ToolCall` per executed blank (name = `identify_blank` or `constraint:<kind>`).
- `reasoning` — currently empty (adapter-level summary not synthesised in Phase 1).
- `hidden_reasoning` — empty in Phase 1 because per-blank LLM extraction isn't logging reasoning tokens through the adapter. Phase 2 will thread this.
- `prompt_tokens` / `completion_tokens` — tokens used by the planner call + repair (per-blank extraction tokens aren't currently summed — TODO).
- `wall_time_s` — total from start of `run_question` to return.

---

## Worked examples

> **Note.** The five examples below are real FRAMES dev-split questions — they illustrate *how the executor walks a plan*, not the v3 prompt few-shots. The actual prompt few-shots are synthetic (see [Few-shots (5 — hand-authored synthetic, v3)](#few-shots-5--hand-authored-synthetic-v3)).

### q154 — *"What Pink Floyd album came out the year Pablo Picasso died?"*

```
entities:
  e1 Pablo Picasso            filled
  e2 Pink Floyd               filled
  b1 ?year-of-death           blank:date
  t  ?album                   blank:entity TARGET
verb_phrases:
  vp1 (Pablo Picasso,  died_in,            ?year-of-death)
  vp2 (Pink Floyd,     released_in_year,   ?year-of-death)
  vp3 (Pink Floyd,     released,           ?album)
constraints: []
```

Execution order: `b1 → t`.
- b1: BM25 on `"Pablo Picasso died_in"` → chunks about Picasso; LLM extract `date` → `1973`.
- t: BM25 on `"Pink Floyd released_in_year 1973 Pink Floyd released"` → chunks about DSOTM; LLM extract `entity` → `Dark Side of the Moon`.

### q70 — *"Population difference: Portland ME vs Portland OR"*

```
entities:
  e1 Portland Maine           filled
  e2 Portland Oregon          filled
  b1 ?pop-me                  blank:number
  b2 ?pop-or                  blank:number
  t  ?diff                    blank:number TARGET
verb_phrases:
  vp1 (Portland Maine,  has_population,  ?pop-me)
  vp2 (Portland Oregon, has_population,  ?pop-or)
constraints:
  c1 derived op=diff args_blanks=[b2, b1] output=t
```

Execution order: `b1 ∥ b2 → t`. b1 and b2 are independent — Phase 2 will fill in parallel.
- b1 / b2: BM25 + LLM extract `number`.
- t: `c1` computes `abs(b2 - b1)` in Python. No LLM call.

### q549 — *"Which launcher entered service last: A/B/C/D?"*

```
entities:
  e1..e4 (4 filled, the candidates)
  b1..b4 (4 blanks, value_type=date)
  t      blank:entity TARGET
verb_phrases:
  vp_k (e_k, entered_service_in, b_k)   for k=1..4
constraints:
  c1 argmax
     candidate_entity_ids=[e1,e2,e3,e4]
     sort_by_blank_ids=[b1,b2,b3,b4]
     output=t
```

Execution: b1..b4 in parallel (each is an identification pass). Then c1 picks the entity whose blank has the max year. `t` = entity name.

### q293 — *"Age at death of NYC-born comic book writer who created Catwoman"*

```
entities (filled):
  e1 New York City
  e2 comic book writer  category=true
  e3 Catwoman
entities (blank):
  b_person              blank:entity
  b_birth               blank:date
  b_death               blank:date
  t  ?age               blank:number TARGET
verb_phrases:
  vp1 (b_person, born_in,         e1)
  vp2 (b_person, profession_is,   e2)
  vp3 (b_person, created,         e3)
  vp4 (b_person, born_in_year,    b_birth)
  vp5 (b_person, died_in_year,    b_death)
constraints:
  c1 derived op=diff args_blanks=[b_death, b_birth] output=t
```

Execution order: `b_person → (b_birth, b_death) → t`.
- b_person: incoming VPs are vp1, vp2, vp3 (all from filled entities). Composite retrieval signal → LLM extract `entity` → `Bob Kane`.
- b_birth: incoming signals via vp4 (projection from resolved b_person). Retrieve + extract `date` → `1915`.
- b_death: similar → `1998`.
- t: derived diff → `83`.

### q796 — *"Who won FIFA World Cup the year Falklands War broke out?"*

```
entities:
  e1 Falklands War            filled
  e2 FIFA World Cup           filled
  b_year                      blank:date
  t  ?winner                  blank:entity TARGET
verb_phrases:
  vp1 (Falklands War,   broke_out_in,    ?year)
  vp2 (FIFA World Cup,  held_in_year,    ?year)
  vp3 (FIFA World Cup,  won_by,          ?winner)
constraints: []
```

Execution order: `b_year → t`. Temporal anchor b_year resolves first (`1982`); t's retrieval composes vp2+vp3 signals with year=1982 in context → `Italy`.

---

## Known limitations

### Partially addressed by v3

1. **Temporal scope via date-anchor rule (partial).** v1 had no primitive for "as of X"; v3's Hard rule 2 forces explicit dates to become filled entities with `as-of-date` / `year-anchor` / `date-anchor` roles, standardised `as_of_date` and `in_year` predicates. The executor still does not push the anchor into BM25 retrieval as a hard filter — it flows through as a scope signal in the composite query. A proper temporal primitive (e.g. a `scope` constraint that rejects off-date retrievals) is deferred.
2. **Internal-knowledge leakage (partial).** v2 dump saw filled entities the question didn't name — hallucinated dates, expanded collectives, invented places — in **26%** of 120b plans, **36%** of 20b plans. v3 introduces the grounding test (Hard rule 1) + the collective-expansion escape hatch (Hard rule 3, `state` breadcrumb). The v3 audit (`planner_dump_inspector._surface_ungrounded_filled_entities` + `_expanded_from_collective_entities`) measures residual rate. *Target: ≤10% / ≤20% ungrounded-without-breadcrumb.*
3. **Missing date anchors (partial).** v2 dump had **24%** of 120b plans / **22%** of 20b plans miss explicit date tokens entirely. v3 Hard rule 2 + two standardised predicates (`in_year`, `as_of_date`) aim to close this. *Target: ≤8% / ≤12% date-miss rate.*

### Not yet addressed

4. **One-shot per-blank extraction.** A single LLM call per blank, no intra-blank revision. Ambiguous chunks can fabricate numbers (e.g. q70 gold `84,512` → 20b `128492`).
5. **No parallel fill.** Independent blanks at the same topological level are resolved sequentially. Phase-2 will add `asyncio.gather` over same-level blanks — the executor already tracks levels internally for visualization.
6. **Per-blank LLM call tokens not summed.** `Trajectory.prompt_tokens/completion_tokens` only reflect the planner + repair calls. Wall-time is correct but token accounting undercounts.
7. **Per-blank `hidden_reasoning` not captured.** The executor's LLM calls discard `reasoning_content`. Phase-2 will thread this for inspection.
8. **No `list_map` / `for_each` constraint.** The collective-expansion escape hatch (Hard rule 3 Pattern 2) is the current workaround for argmax-over-collective questions. A proper `list_map` constraint — iterate a `list-member` blank, instantiate a sub-plan per item — would let Pattern 1 (retrieval-resolved list) also cover argmax, removing the escape hatch's dependence on planner priors. Phase-2 candidate.

---

## File layout

### Source

```
research_agent/src/research_agent/adapters/
├── base.py                                   Adapter contract (unchanged)
├── baselines/                                E1–E13 + prior ours variants
│   ├── ours_gsw_v1.py                        ← fallback target
│   └── ... (all other baselines)
└── ours/
    ├── __init__.py
    ├── gsw_planner_v1.py                     Adapter class (registered as "ours_gsw_planner_v1")
    ├── _planner_prompts.py                   PLANNER_SYSTEM + 5 few-shots + build_planner_messages + build_repair_messages
    └── _planner_exec.py                      Entity / VerbPhrase / Constraint / GSWPlan + topo_sort + execute
```

### Tests

```
research_agent/tests/ours/
├── __init__.py
├── test_planner_exec.py      14 tests — topo sort, execute, argmax, cycle/unknown
│                              cascade + v2 additions: role fields accepted,
│                              dangling entity rejected, constraint-only
│                              references accepted.
└── test_planner_prompts.py   10 tests — each v3 synthetic few-shot validates
                              against GSWPlan; breadcrumb hygiene enforced;
                              date-anchor & as-of-date roles asserted.
```

### Tooling

```
research_agent/playground/
├── run_substitution.py             harness CLI (imports ours.gsw_planner_v1 at top)
├── dump_plans.py                   CLI — dump plans for inspection across N models
├── _planner_viz.py                 [v2] shared viz helpers: compute_topo_info,
│                                   plan_to_dot (with topo numbering + rank hints),
│                                   solve_steps_markdown, entities/vps/constraints_df,
│                                   dangling_badge.
├── planner_dump_inspector.py       Streamlit — browse dump_plans artefacts.
│                                   Shows topological order, solve-steps, per-plan
│                                   audit (ungrounded / expanded-from-collective /
│                                   missing-date-anchors / dangling).
└── pilot_run_inspector.py          Streamlit — browse any cell_result.json.
                                    Shows topological graph + side-by-side
                                    planned-vs-actual executed-blanks table.
```

### Docs

```
docs/
├── FRAMES_PILOT_FAILURE_ANALYSIS.md        pilot grid + §12 = this planner's 30-Q run
├── QUESTION_PLANNER_DAG_LITERATURE_REVIEW.md  12 related papers, gap analysis, direction options
└── GSW_PLANNER_ARCHITECTURE.md             ← you are here
```

---

## References

- **ReWOO** (`arxiv 2305.18323`) — variable substitution via `#E_k` (adapted into typed VP + blank edges).
- **LLM Compiler** (`arxiv 2312.04511`) — DAG dependency encoding + parallel fill pattern.
- **Plan-over-Graph** (`arxiv 2502.14563`) — strict JSON schema + SFT + DPO recipe; documents hallucination as the dominant failure (same cluster our pilot showed).
- **ToQD** (COLING 2025, `aclanthology.org/2025.coling-main.191`) — topology-graph question decomposition.
- **QDMR** (`arxiv 2001.11770`) — operator vocabulary informing blank `value_type`s and (future) training supervision.
- **Existing `baselines/ours_gsw_v1`** — flat-decomposition ancestor used as fallback path.
- Full 12-paper comparative review: `docs/QUESTION_PLANNER_DAG_LITERATURE_REVIEW.md`.
- FRAMES pilot grid + failure-mode evidence: `docs/FRAMES_PILOT_FAILURE_ANALYSIS.md`.
