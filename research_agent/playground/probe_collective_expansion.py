"""Probe script: how does prompt v3 handle collective questions?

Runs ``emit_plan()`` (via bedrock gpt-oss-120b) on 10 curated questions
spanning three buckets so we can see how often the planner uses
Hard Rule 3 Pattern 1 (defer to retrieval) vs Pattern 2 (expand with
breadcrumb) vs neither.

Buckets:
  A. Collective + superlative — Pattern 2 territory (argmax over members).
  B. Collective + list/count — Pattern 1 territory (no argmax needed).
  C. Non-collective baseline — should NOT trigger Hard Rule 3 at all.

No retrieval, no execution. Just plan emission + structural audit.

Usage::

    .venv/bin/python playground/probe_collective_expansion.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make src importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from research_agent.adapters.ours._planner_emit import PlanEmitError, emit_plan
from research_agent.models.llm_client import LLMClient

try:
    from dotenv import load_dotenv

    load_dotenv()
    _PARENT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
    if _PARENT_ENV.exists():
        load_dotenv(_PARENT_ENV, override=False)
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------


BUCKETS = {
    "A — collective + superlative (Pattern 2 target)": [
        "Who lived longer, one of the Brontë sisters or Jane Austen?",
        "Which Jonas brother had the most Grammy nominations?",
        "Which of the three Musketeers was the youngest?",
    ],
    "B — collective + list/count (Pattern 1 target)": [
        "Who were the Founding Fathers of the United States?",
        "Name all the Brontë sisters and their birth years.",
        "How many members did the Beatles have?",
    ],
    "C — non-collective baseline (neither pattern)": [
        "Who directed The Princess Bride?",
        "What year did World War I end?",
        "Who was the first person to walk on the moon?",
        "What is the population of Tokyo as of 2024?",
    ],
}


_COLLECTIVE_PREFIX = "expanded_from_collective:"


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_plan(plan) -> dict:
    """Compute structural flags for pattern-1 / pattern-2 detection."""
    expanded_entities = []
    list_header_entities = []
    list_member_blanks = []
    collectives_mentioned = set()
    for e in plan.entities:
        if e.kind == "filled":
            st = (e.state or "").strip()
            if st.startswith(_COLLECTIVE_PREFIX):
                expanded_entities.append(e.id)
                collectives_mentioned.add(st[len(_COLLECTIVE_PREFIX):].strip())
            if e.role == "list-header":
                list_header_entities.append(e.id)
        else:  # blank
            if e.role == "list-member":
                list_member_blanks.append(e.id)

    has_pattern_1 = bool(list_header_entities and list_member_blanks)
    has_pattern_2 = bool(expanded_entities)
    if has_pattern_2:
        kind = "pattern_2"
    elif has_pattern_1:
        kind = "pattern_1"
    else:
        kind = "neither"
    return {
        "kind": kind,
        "n_entities": len(plan.entities),
        "n_vps": len(plan.verb_phrases),
        "n_constraints": len(plan.constraints),
        "expanded_entities": expanded_entities,
        "list_header_entities": list_header_entities,
        "list_member_blanks": list_member_blanks,
        "collectives": sorted(collectives_mentioned),
        "has_argmax": any(c.kind in ("argmax", "argmin") for c in plan.constraints),
    }


def render_plan(plan) -> str:
    lines = []
    lines.append(f"  entities: ({len(plan.entities)})")
    for e in plan.entities:
        tag = "filled" if e.kind == "filled" else "blank "
        target = "  TARGET" if e.is_target else ""
        cat = "  category=true" if e.category else ""
        state = f"  state={e.state!r}" if e.state else ""
        name = f" name={e.name!r}" if e.name else ""
        vt = f" vt={e.value_type}" if e.value_type else ""
        lines.append(
            f"    {e.id:<14} [{tag}] role={str(e.role):<18}{name}{vt}{cat}{state}{target}"
        )
    if plan.verb_phrases:
        lines.append(f"  verb_phrases: ({len(plan.verb_phrases)})")
        for v in plan.verb_phrases:
            lines.append(f"    {v.id:<6} ({v.subject_id}) --[{v.phrase}]--> ({v.object_id})")
    if plan.constraints:
        lines.append(f"  constraints: ({len(plan.constraints)})")
        for c in plan.constraints:
            extra = ""
            if c.kind == "derived":
                extra = f" op={c.op} args={c.args_blanks}"
            elif c.kind in ("argmax", "argmin"):
                extra = f" cand={c.candidate_entity_ids} sort={c.sort_by_blank_ids}"
            lines.append(f"    {c.id:<6} kind={c.kind}{extra} → {c.output_blank_id}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    llm = LLMClient(model="bedrock/openai.gpt-oss-120b-1:0")
    all_audits = []
    t_start = time.time()
    for bucket, questions in BUCKETS.items():
        print("=" * 80)
        print(bucket)
        print("=" * 80)
        for q in questions:
            print(f"\n▸ Q: {q}")
            t0 = time.time()
            try:
                plan, meta = emit_plan(q, llm_client=llm, max_tokens=6000)
            except PlanEmitError as exc:
                print(f"  PARSE/LLM FAILURE: {exc.kind}  detail={exc.detail[:200]}")
                continue
            dur = round(time.time() - t0, 1)
            audit = audit_plan(plan)
            audit["bucket"] = bucket[:1]
            audit["question"] = q
            all_audits.append(audit)
            tokens = f"{meta.prompt_tokens}→{meta.completion_tokens}"
            repair = "  +repair" if meta.repair_used else ""
            print(
                f"  [{dur}s tokens={tokens}{repair}]  → classification: {audit['kind']}"
                f"  (argmax={audit['has_argmax']})"
            )
            if audit["expanded_entities"]:
                print(f"  expanded: {audit['expanded_entities']}  collectives: {audit['collectives']}")
            if audit["list_header_entities"] or audit["list_member_blanks"]:
                print(
                    f"  list_header: {audit['list_header_entities']}  "
                    f"list_member: {audit['list_member_blanks']}"
                )
            print(render_plan(plan))

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY ({len(all_audits)} plans in {round(time.time() - t_start, 1)}s)")
    print("=" * 80)
    for bucket_letter in ["A", "B", "C"]:
        xs = [a for a in all_audits if a["bucket"] == bucket_letter]
        kinds = [a["kind"] for a in xs]
        p1 = kinds.count("pattern_1")
        p2 = kinds.count("pattern_2")
        neither = kinds.count("neither")
        print(
            f"  Bucket {bucket_letter}: n={len(xs)}  "
            f"pattern_1={p1}  pattern_2={p2}  neither={neither}"
        )


if __name__ == "__main__":
    main()
