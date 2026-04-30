"""Planner prompt v3 for the GSW-fragment question-planner adapter.

Version 3 deltas vs v2 (2026-04-23):
  - Grounding test: every filled entity.name must appear as a substring of
    the question text (case-insensitive). No exceptions except the
    collective-expansion escape hatch.
  - Date-anchor rule: explicit date / year tokens in the question text
    must be promoted to filled entities with role=year-anchor /
    date-anchor / as-of-date and wired into the plan.
  - Collective-expansion escape hatch: planner-time expansion of a
    well-known finite group is allowed IFF every expanded entity
    carries state="expanded_from_collective:<group-name>" and the
    collective itself remains as a filled entity with category=true,
    role=list-header.
  - All 5 few-shots are now hand-authored synthetic examples (fictional
    entities) — the v2 few-shots were drawn from FRAMES dev split and
    leaked into the evaluation set.
  - Tightened role-vocabulary descriptions: subject vs scope-filter,
    candidate (≥2 siblings only), year-anchor / date-anchor, list-header.

The executor schema did not change in v3. The Entity.state field shipped
in v2 and is now activated by the prompt for collective provenance.
"""

from __future__ import annotations

import json
import textwrap


# ---------------------------------------------------------------------------
# System prompt (v3)
# ---------------------------------------------------------------------------


PLANNER_SYSTEM = textwrap.dedent(
    """
    You are a research planner. Your output is a **GSW-fragment plan** —
    a graph that describes what the question is asking, in a shape a
    downstream executor can walk to find the answer.

    ## Why this matters

    A downstream executor will walk your graph to answer the question:
    it identifies filled entities in the question text, retrieves
    supporting chunks keyed by each verb-phrase, extracts values for
    blank entities one at a time, and computes numeric / selector
    constraints in pure Python. Your job is to make every step of that
    execution *possible to do*. The executor cannot invent retrieval
    signals or infer connections you haven't drawn.

    ## How to think (three-step protocol)

    Before emitting the JSON, work through these three steps in your
    head:

    ### Step 1 — Entities with roles
    List every named thing in the question as a `filled` entity (with
    `name` and a `role` label). Mark categorical filled entities (e.g.
    "comic book writer", "a Greek island") with `category: true`.
    Then add one `blank` entity per unknown value the question asks
    about, each with a `role` and a `value_type`. Exactly one blank
    has `is_target: true` — its resolved value is the final answer.

    **Role vocabulary** (seed — you may invent new roles when a novel
    question demands, but prefer reusing these):

    For FILLED entities:
    - `subject` — the main named entity the question is about. If it's
      a category / profession / role label rather than a named
      individual, use `scope-filter` with `category: true` instead.
    - `scope-filter` — a categorical predicate that narrows the set
      (professions, group memberships, temporal boundaries). Usually
      paired with `category: true`.
    - `year-anchor` / `date-anchor` — a date or year token that
      appears in the question text.
    - `as-of-date` — a snapshot-in-time anchor ("as of August 2024").
    - `list-header` — a collective name that owns a list-member blank.
      Always paired with `category: true`.
    - `candidate` — reserved for siblings compared in an argmax /
      argmin constraint. NEVER use for a single item; if there's only
      one, use `subject` or `scope-filter`.
    - `object`, `attribute-name`, `constraint-value` — specialised.

    For BLANK entities:
    - `target` — the final answer the question asks for.
    - `bridge-entity`, `bridge-date`, `bridge-number`,
      `bridge-attribute` — intermediate blanks the executor must
      resolve to link the question to its answer.
    - `comparison-output`, `aggregate-output` — blank produced by a
      constraint rather than retrieval.
    - `list-member` — a list-valued blank that holds the members of
      a list-header collective.

    ### Step 2 — Verb-phrases (the retrieval edges)
    Wire every entity into the graph via `verb_phrases`. Each VP is a
    directed binary relation `(subject, phrase, object)`. Predicates
    are short snake_case (`died_in`, `has_population`, `won_by`).

    Standardised predicates for temporal anchors:
    - `in_year` — for "in the year X" / "the year that ..." bridges.
    - `as_of_date` — for snapshot-in-time anchors. Wires a subject
      entity to an as-of-date filled entity to tell the executor the
      retrieved attribute value is for that date.
    - `signed_in_year`, `released_in_year`, `founded_in_year`,
      `broke_out_in`, `born_in_year`, `died_in_year` — event-to-year
      predicates.

    A VP serves one of two jobs:
      1. **Identify a blank** — subject or object is a blank; the
         other side plus the predicate become the retrieval signal.
      2. **Project an attribute** — subject is a blank-entity that
         will resolve first; object is a blank that depends on it.

    **NO DANGLING ENTITIES.** Every entity id MUST appear as a
    `subject_id` or `object_id` in at least one verb-phrase, OR in a
    constraint's inputs / outputs. If you emit a filled entity with no
    edge, the executor cannot use it and the plan is invalid.

    ### Step 3 — Constraints (only when needed)
    Add a constraint ONLY when the question demands:
      - **derived**: arithmetic over blank values (`op ∈ {diff, sum,
        avg, max, min, count, concat, mul, div, round_nearest}`,
        `args_blanks = [...]`). For `round_nearest`, one arg rounds to
        the nearest ten; two args use the second blank as the interval.
      - **argmax** / **argmin**: pick the entity among N candidates
        whose ranking blank value is extreme (`candidate_entity_ids`
        and `sort_by_blank_ids` aligned same-length).
      - **equals / in_list / gt / lt**: relational constraints with
        `left_ref`, `right_ref`, and an output bool blank. `in_list`
        may alternatively use `args_blanks=[member_blank, list_blank]`.

    Most 2–3 hop questions need **zero** constraints — the bridging
    dependency is already expressed by verb-phrases. Use constraints
    only when pure Python arithmetic or selection is required to
    produce the target value.

    ## Hard rule 1 — Grounding test (no internal-knowledge leakage)

    Every filled entity's `name` MUST appear as a substring of the
    question text when both are whitespace-normalised and lowercased.
    Before emitting a filled entity, self-apply this test:

        normalise(entity.name) ⊆ normalise(question)

    If the test fails, you must EITHER:
      (a) drop that entity and rethink the plan, or
      (b) rename it to a substring that IS present in the question, or
      (c) — and only if the entity is a member of a collective group
           the question names — use the collective-expansion escape
           hatch (see Hard rule 3).

    Do NOT invent named individuals, dates, places, or facts the
    question doesn't already name. The executor's retrieval grounds
    the blank values; your job is only to scaffold the DAG from the
    question text. When in doubt, promote an unknown to a `blank`
    rather than filling it from your priors.

    ## Hard rule 2 — Date-anchor rule

    Any date, year, month, or temporal-anchor token that appears
    literally in the question text MUST become a filled entity with a
    `year-anchor` / `date-anchor` / `as-of-date` role, and MUST appear
    as `subject_id` or `object_id` of at least one verb-phrase.

    Triggering tokens include (non-exhaustive):
      - 4-digit years: "2010", "1994", "2024".
      - Full dates: "August 3, 2024", "June 12th, 1994", "March 1,
        2047", "1st January 2023".
      - Month + year: "August 2024", "January 2023".
      - Anchor phrases: "as of [date]", "in the year", "the year
        that", "after [date]", "before [date]".

    Use `as_of_date` as the predicate when wiring a snapshot anchor
    to a subject: `(Kessington, as_of_date, March 1 2047)` tells the
    executor to filter retrieved attributes to that date. Use
    `in_year` (or event-to-year predicates like `signed_in_year`)
    when the anchor resolves a bridging year between two entities.

    Do not bake date tokens into predicate names (`died_in_1973`) or
    leave them in prose without a filled entity. Surface them as
    filled entities so the executor can use them as retrieval
    signals and scope filters.

    ## Hard rule 3 — Collective-expansion escape hatch

    When the question names a finite, well-known group by collective
    name (e.g. "the Brontë sisters", "the Jonas Brothers", "the three
    Musketeers"), you have two valid patterns:

    **Pattern 1 — preferred: defer to retrieval.**
    Emit the collective as a single filled entity with
    `category: true, role: "list-header"`. Add a `blank` with
    `role: "list-member", value_type: "list"`. Wire them via a VP
    like `(Brontë sisters, members, ?sisters)`. The executor resolves
    the members at runtime.

    **Pattern 2 — escape hatch: planner-time expansion.**
    Use ONLY when the downstream reasoning needs argmax / argmin /
    arithmetic over the individual members (which Pattern 1 cannot
    currently express — argmax requires fixed `candidate_entity_ids`).
    You may expand the collective into individual filled entities AT
    PLAN TIME if and only if:

      1. Every expanded entity carries
         `state: "expanded_from_collective:<group-name>"` as a
         provenance breadcrumb. The group name is the collective name
         as it appears in the question.
      2. The collective itself ALSO remains in the plan as a separate
         filled entity with `category: true, role: "list-header"`, and
         is wired to the expanded entities via `member_of` VPs (or
         equivalent) so it is not dangling.
      3. The expanded entities' names DO NOT need to satisfy the
         Hard-rule-1 grounding test — the `state` breadcrumb is how
         you legitimately declare their origin.

    Never silently invent members. The `state` breadcrumb is how you
    declare you're using world knowledge and the auditor can count,
    review, or challenge it.

    ## Field spec (strict)

    ### Entity
    - `id` (required): short unique label, e.g. "e1" / "b1" / "t".
    - `kind` (required): "filled" or "blank".
    - `role` (required): short role label (see vocabulary above).
    - For filled: `name` (the surface form from the question).
    - For filled: `category: true` when the name is a role / category
      rather than a named entity.
    - For blank: `value_type` ∈ {date, number, entity, attribute,
      list, text, bool}.
    - Exactly one blank has `is_target: true`.
    - `state` (optional): provenance breadcrumb. Only set when the
      entity originates from collective expansion:
      `"expanded_from_collective:<group-name>"`.

    ### VerbPhrase
    - `id` (required): "vp1", "vp2", ...
    - `phrase` (required): short snake_case predicate.
    - `subject_id`, `object_id` (both required): entity ids.

    ### Constraint
    - `id` (required): "c1", "c2", ...
    - `kind` (required): derived / argmax / argmin / equals / in_list / gt / lt.
    - `output_blank_id` (required): the blank this constraint fills.
    - Kind-specific populated fields:
      - derived: `op`, `args_blanks`.
        Supported ops: diff, sum, avg, max, min, count, concat, mul,
        div, round_nearest.
      - argmax / argmin: `candidate_entity_ids`, `sort_by_blank_ids`
        (same length; position i of one maps to position i of the other).
      - relational: `left_ref`, `right_ref`. For in_list, `args_blanks`
        may be used instead as [member_blank, list_blank].

    ## Output format

    Return **ONE** JSON object matching the `GSWPlan` schema:

    ```
    {
      "entities":     [ ... ],
      "verb_phrases": [ ... ],
      "constraints":  [ ... ]
    }
    ```

    No prose, no Markdown fences, no preamble. Exactly one JSON
    object.
    """
).strip()


# ---------------------------------------------------------------------------
# Few-shot examples (v3 — hand-authored synthetic, to avoid test-set leakage)
#
# Each example uses unambiguously fictional entities so the LLM cannot
# rely on world-knowledge priors to fill them in. Each demonstrates at
# least one v3 rule (grounding, date-anchor, collective-expansion).
# ---------------------------------------------------------------------------


_FEW_SHOT_1_TEMPORAL_BRIDGE = {
    "question": (
        "Which song did the band Hollowmoor Drive release in the year "
        "the Treaty of Verdane was signed?"
    ),
    "how_to_think": textwrap.dedent(
        """
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
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Hollowmoor Drive", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Treaty of Verdane", "role": "year-anchor"},
            {"id": "b_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "signed_in_year", "subject_id": "e2", "object_id": "b_year"},
            {"id": "vp2", "phrase": "released_in_year", "subject_id": "e1", "object_id": "b_year"},
            {"id": "vp3", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    },
}


_FEW_SHOT_2_AS_OF_DATE_DIFF = {
    "question": (
        "As of March 1, 2047, what is the difference in population "
        "between the cities Kessington and Raleford?"
    ),
    "how_to_think": textwrap.dedent(
        """
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
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e_date", "kind": "filled", "name": "March 1, 2047", "role": "as-of-date"},
            {"id": "e1", "kind": "filled", "name": "Kessington", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Raleford", "role": "subject"},
            {"id": "b_pop_k", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_pop_r", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {
                "id": "t",
                "kind": "blank",
                "role": "aggregate-output",
                "value_type": "number",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_population", "subject_id": "e1", "object_id": "b_pop_k"},
            {"id": "vp2", "phrase": "has_population", "subject_id": "e2", "object_id": "b_pop_r"},
            {"id": "vp3", "phrase": "as_of_date", "subject_id": "e1", "object_id": "e_date"},
            {"id": "vp4", "phrase": "as_of_date", "subject_id": "e2", "object_id": "e_date"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "diff",
                "args_blanks": ["b_pop_k", "b_pop_r"],
                "output_blank_id": "t",
            }
        ],
    },
}


_FEW_SHOT_3_ENUMERATED_ARGMAX = {
    "question": (
        "Of the following four corporate treasurers, who was appointed "
        "most recently — Ada Voss, Petro Milan, Lin Karaczynski, or "
        "Sybil Quine?"
    ),
    "how_to_think": textwrap.dedent(
        """
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
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Ada Voss", "role": "candidate"},
            {"id": "e2", "kind": "filled", "name": "Petro Milan", "role": "candidate"},
            {"id": "e3", "kind": "filled", "name": "Lin Karaczynski", "role": "candidate"},
            {"id": "e4", "kind": "filled", "name": "Sybil Quine", "role": "candidate"},
            {"id": "b1", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b2", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b3", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b4", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "appointed_on", "subject_id": "e1", "object_id": "b1"},
            {"id": "vp2", "phrase": "appointed_on", "subject_id": "e2", "object_id": "b2"},
            {"id": "vp3", "phrase": "appointed_on", "subject_id": "e3", "object_id": "b3"},
            {"id": "vp4", "phrase": "appointed_on", "subject_id": "e4", "object_id": "b4"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "argmax",
                "candidate_entity_ids": ["e1", "e2", "e3", "e4"],
                "sort_by_blank_ids": ["b1", "b2", "b3", "b4"],
                "output_blank_id": "t",
            }
        ],
    },
}


_FEW_SHOT_4_COMPOUND_SCOPE = {
    "question": (
        "Who was the woman born in Astra Bay who founded the Wren "
        "Clinic and authored the novel Tidebound?"
    ),
    "how_to_think": textwrap.dedent(
        """
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
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Astra Bay", "role": "scope-filter"},
            {"id": "e2", "kind": "filled", "name": "Wren Clinic", "role": "scope-filter"},
            {"id": "e3", "kind": "filled", "name": "Tidebound", "role": "scope-filter"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "born_in", "subject_id": "t", "object_id": "e1"},
            {"id": "vp2", "phrase": "founded", "subject_id": "t", "object_id": "e2"},
            {"id": "vp3", "phrase": "authored", "subject_id": "t", "object_id": "e3"},
        ],
        "constraints": [],
    },
}


_FEW_SHOT_5_COLLECTIVE_EXPANSION = {
    "question": (
        "Who lived longer — one of the Three Founders of Acme Corp or "
        "Dorian Vexley?"
    ),
    "how_to_think": textwrap.dedent(
        """
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
        """
    ).strip(),
    "plan": {
        "entities": [
            {
                "id": "e_collective",
                "kind": "filled",
                "name": "Three Founders of Acme Corp",
                "role": "list-header",
                "category": True,
            },
            {
                "id": "e_f1",
                "kind": "filled",
                "name": "Mira Tallow",
                "role": "candidate",
                "state": "expanded_from_collective:Three Founders of Acme Corp",
            },
            {
                "id": "e_f2",
                "kind": "filled",
                "name": "Essen Quorn",
                "role": "candidate",
                "state": "expanded_from_collective:Three Founders of Acme Corp",
            },
            {
                "id": "e_f3",
                "kind": "filled",
                "name": "Varik Opsahl",
                "role": "candidate",
                "state": "expanded_from_collective:Three Founders of Acme Corp",
            },
            {"id": "e_vexley", "kind": "filled", "name": "Dorian Vexley", "role": "candidate"},
            {"id": "b_lf1", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_lf2", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_lf3", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_lf_v", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "member_of", "subject_id": "e_f1", "object_id": "e_collective"},
            {"id": "vp2", "phrase": "member_of", "subject_id": "e_f2", "object_id": "e_collective"},
            {"id": "vp3", "phrase": "member_of", "subject_id": "e_f3", "object_id": "e_collective"},
            {"id": "vp4", "phrase": "lived_years", "subject_id": "e_f1", "object_id": "b_lf1"},
            {"id": "vp5", "phrase": "lived_years", "subject_id": "e_f2", "object_id": "b_lf2"},
            {"id": "vp6", "phrase": "lived_years", "subject_id": "e_f3", "object_id": "b_lf3"},
            {
                "id": "vp7",
                "phrase": "lived_years",
                "subject_id": "e_vexley",
                "object_id": "b_lf_v",
            },
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "argmax",
                "candidate_entity_ids": ["e_f1", "e_f2", "e_f3", "e_vexley"],
                "sort_by_blank_ids": ["b_lf1", "b_lf2", "b_lf3", "b_lf_v"],
                "output_blank_id": "t",
            }
        ],
    },
}


_FEW_SHOTS = [
    _FEW_SHOT_1_TEMPORAL_BRIDGE,
    _FEW_SHOT_2_AS_OF_DATE_DIFF,
    _FEW_SHOT_3_ENUMERATED_ARGMAX,
    _FEW_SHOT_4_COMPOUND_SCOPE,
    _FEW_SHOT_5_COLLECTIVE_EXPANSION,
]


# ---------------------------------------------------------------------------
# User prompt rendering
# ---------------------------------------------------------------------------


def _format_few_shot(example: dict) -> str:
    parts = [f"Question: {example['question']}", ""]
    if "how_to_think" in example:
        parts.append("How to think:")
        parts.append(example["how_to_think"])
        parts.append("")
    parts.append("Plan:")
    parts.append(json.dumps(example["plan"], indent=2))
    return "\n".join(parts)


def build_planner_messages(question: str) -> list[dict]:
    """Build the system + user message list for the planner LLM call.

    Each few-shot demonstrates at least one v3 rule: grounding,
    date-anchor handling, or the collective-expansion escape hatch. The
    user prompt reminds the model to self-apply the grounding test
    before emitting JSON.
    """
    few_shots_rendered = "\n\n---\n\n".join(_format_few_shot(ex) for ex in _FEW_SHOTS)
    user = (
        "Here are five worked examples. Each shows the three-step "
        "thinking protocol (How to think) followed by the corresponding "
        "Plan JSON. Pay attention to how each example honours the three "
        "hard rules: grounding (every filled entity.name appears in the "
        "question), date-anchor (dates become filled entities), and the "
        "collective-expansion escape hatch (state breadcrumb when the "
        "planner expands a named group).\n\n"
        "For the new question, walk the same three steps internally, "
        "self-apply the grounding test to each filled entity before "
        "emitting it, then return ONLY the Plan JSON — no prose.\n\n"
        f"{few_shots_rendered}\n\n"
        "---\n\n"
        f"Question: {question}\n"
        "Plan:"
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user},
    ]


REPAIR_SYSTEM = textwrap.dedent(
    """
    Your previous response was not valid JSON matching the GSWPlan
    schema, or it violated a plan rule. Re-emit a single JSON object
    that:

    1. Matches the GSWPlan schema exactly.
    2. Has at least one verb-phrase or constraint edge touching every
       entity id (no dangling entities).
    3. Assigns a `role` label to every entity.
    4. Honours the grounding test: every filled entity's `name` appears
       as a substring of the question text (case-insensitive), UNLESS
       it carries a `state="expanded_from_collective:<group>"` breadcrumb.
    5. Promotes any explicit date / year / month token from the
       question to a filled entity with role=year-anchor,
       date-anchor, or as-of-date, and wires it into at least one
       verb-phrase.

    Do not include any prose or Markdown fences.
    """
).strip()


def build_repair_messages(
    question: str,
    bad_response: str,
    error: str,
    *,
    attempt: int = 1,
    max_attempts: int = 1,
) -> list[dict]:
    """Messages for the repair retry on parse failure.

    ``attempt`` is 1-indexed; ``max_attempts`` is the total repair budget.
    Later attempts get a progressively sharper preface so the LLM knows
    it has already failed earlier rounds and must change its output.
    """
    msgs = build_planner_messages(question)
    msgs.append({"role": "assistant", "content": bad_response})

    if max_attempts <= 1 or attempt <= 1:
        preface = REPAIR_SYSTEM
    elif attempt < max_attempts:
        preface = (
            f"This is repair attempt {attempt} of {max_attempts}. "
            "Your previous repaired output ALSO failed validation. "
            "Read the validation error carefully — the named field "
            "is the one to fix. Do NOT echo your previous response.\n\n"
            f"{REPAIR_SYSTEM}"
        )
    else:
        preface = (
            f"FINAL repair attempt ({attempt}/{max_attempts}). "
            "All earlier repairs have failed. If this attempt does not "
            "validate, the question will be answered without a plan. "
            "Address the SPECIFIC field named in the validation error — "
            "if it complains about empty `args_blanks`, populate them; "
            "if it complains about missing `left_ref`, supply it; if it "
            "complains about a dangling entity, add a verb-phrase that "
            "uses it. Re-emit the entire JSON object from scratch.\n\n"
            f"{REPAIR_SYSTEM}"
        )

    msgs.append(
        {
            "role": "user",
            "content": (
                f"{preface}\n\nValidation error was:\n{error}\n\n"
                "Return the corrected JSON now."
            ),
        }
    )
    return msgs
