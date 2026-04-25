"""Unit tests for the per-level researcher prompt builder."""

from __future__ import annotations

from research_agent.adapters.ours._planner_exec import BlankResult
from research_agent.adapters.ours._researcher_prompt import (
    RESEARCHER_LEGEND,
    RESEARCHER_RULES,
    RESEARCHER_TASK_BANNER,
    build_researcher_prompt,
)


def _picasso_floyd_plan():
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pablo Picasso", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
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
            {"id": "vp1", "phrase": "died_in", "subject_id": "e1", "object_id": "b_year"},
            {"id": "vp2", "phrase": "released_in_year", "subject_id": "e2", "object_id": "b_year"},
            {"id": "vp3", "phrase": "released", "subject_id": "e2", "object_id": "t"},
        ],
        "constraints": [],
    }


def test_no_finish_contract_language():
    """The researcher legend + rules must never contradict each other
    about the termination protocol — there is no `finish` tool."""
    assert "finish(answer=" not in RESEARCHER_LEGEND
    assert "The `finish` tool" not in RESEARCHER_LEGEND
    assert "no `finish` tool" in RESEARCHER_LEGEND
    assert "no `finish` tool" in RESEARCHER_TASK_BANNER
    assert "no `finish` tool" in RESEARCHER_RULES


def test_commit_budget_and_value_hygiene_language():
    """The per-level researcher prompt should explicitly prevent the
    observed loop/placeholder failures."""
    combined = "\n".join([RESEARCHER_TASK_BANNER, RESEARCHER_RULES])
    assert "after 2 tool calls for a blank" in combined
    assert "Do not spend more than 3 consecutive tool turns on one blank" in combined
    assert "a partial best-effort list is better" in combined
    assert "than no update" in combined
    assert 'value="None"' in combined
    assert "a JSON/tool-call string" in combined


def test_level_0_first_researcher_no_prior_values():
    """At level 0 no blanks are resolved yet; prior-values section
    should render an explicit empty placeholder so the section never
    goes missing."""
    plan = _picasso_floyd_plan()
    state = {
        "b_year": BlankResult(blank_id="b_year", status="unknown"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    prompt = build_researcher_prompt(
        plan, ["b_year", "t"], {"b_year": 0, "t": 1},
        slice_blank_ids=["b_year"],
        state=state,
        level_index=0,
    )
    assert "## Prior resolved values" in prompt
    assert "_(none — you are the first researcher)_" in prompt
    # b_year is in this researcher's slice
    assert "`b_year`" in prompt


def test_level_1_prior_values_included_and_target_visible():
    """At level 1 the researcher's slice is `t`. The prompt must:
    (a) list b_year's resolved value in the upstream section,
    (b) not render b_year as an assignable topo step,
    (c) drop Picasso (he's only wired to b_year, not to t),
    (d) flag the target as TARGET."""
    plan = _picasso_floyd_plan()
    state = {
        "b_year": BlankResult(
            blank_id="b_year", value="1973", status="resolved",
            evidence_chunk_ids=["picasso_wiki__0"],
        ),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    prompt = build_researcher_prompt(
        plan, ["b_year", "t"], {"b_year": 0, "t": 1},
        slice_blank_ids=["t"],
        state=state,
        level_index=1,
    )
    # (a) upstream value visible
    assert "`b_year`" in prompt
    assert "'1973'" in prompt
    # (b) b_year is not an assignable step here
    assert "Step 2" not in prompt
    # (c) Picasso dropped (only wired upstream)
    #     Narrow the check to the "plan for this question" section
    #     so we don't trip on a prompt example elsewhere.
    plan_section = prompt.split("## The plan for this question")[1].split("## Prior resolved values")[0]
    assert "Pablo Picasso" not in plan_section
    assert "Pink Floyd" in plan_section
    # (d) target flagged
    assert "TARGET" in prompt


def test_constraint_output_in_slice_renders_as_auto_computed():
    """If the slice contains a blank that is the output of a
    constraint, the plan brief should tag it as auto-computed — the
    researcher is NOT supposed to call update_blank on it."""
    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "b_b", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "concat",
                "args_blanks": ["b_a", "b_b"],
                "output_blank_id": "t",
            }
        ],
    }
    state = {
        "b_a": BlankResult(blank_id="b_a", value="hello", status="resolved"),
        "b_b": BlankResult(blank_id="b_b", value="world", status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    prompt = build_researcher_prompt(
        plan_dict, ["b_a", "b_b", "t"], {"b_a": 0, "b_b": 0, "t": 1},
        slice_blank_ids=["t"],
        state=state,
        level_index=1,
    )
    # Target step appears as constraint-driven
    assert "`t`" in prompt
    assert "auto-computed" in prompt.lower() or "constraint" in prompt.lower()
    # Orchestrator-will-auto-compute language must be in the legend
    assert "ORCHESTRATOR automatically computes" in prompt


def test_truncates_long_upstream_values():
    """Resolved values >80 chars should be truncated in the upstream
    table to keep the prompt compact."""
    plan = _picasso_floyd_plan()
    long_val = "A" * 200
    state = {
        "b_year": BlankResult(blank_id="b_year", value=long_val, status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    prompt = build_researcher_prompt(
        plan, ["b_year", "t"], {"b_year": 0, "t": 1},
        slice_blank_ids=["t"], state=state, level_index=1,
    )
    assert "..." in prompt  # truncation marker
    assert long_val not in prompt  # the full 200-char version must not appear
