"""Script-Writer: the draft -> sandbox-grade -> repair loop.

This is the only place an LLM is used in the entire decomposer story.
It runs **offline**. The emitted ``.py`` file contains zero LLM imports
and is the only thing the inference path touches.

Loop per iteration:

1. Draft: an LLM writes a candidate decomposer module as raw Python.
2. Load in sandbox (``_sandbox.load_decomposer``) — rejects any module
   that imports banned LLM / network packages.
3. Grade on the dev split (``_sandbox.grade_decomposer``).
4. If this draft beats the incumbent, promote it to best-so-far.
5. Repair: next-iteration prompt is built from the current best module
   + its top-N failing examples + the expected gold decompositions.

Stopping: ``max_iterations`` reached, or no improvement for
``patience`` iterations.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from research_agent.decomposers._sandbox import (
    ForbiddenImportError,
    grade_decomposer,
    load_decomposer,
    scan_source_for_forbidden_imports,
)


def _get_transport_shim():
    """Surgical import that skips gsw_memory.__init__ side effects.

    The top-level ``gsw_memory`` package imports heavy optional deps
    (``bespokelabs``, etc.) at import time — those aren't installed in
    the ``research_agent`` venv. We load the specific transport modules
    by file path so the package __init__ chain never runs.
    """
    import importlib.util
    import os
    import sys

    # _synthesis.py lives at:
    #   <repo>/research_agent/src/research_agent/decomposers/_synthesis.py
    # The gsw_memory source tree is at <repo>/src, which is 4 levels up + /src.
    here = os.path.abspath(os.path.dirname(__file__))
    repo_src = os.path.abspath(os.path.join(here, "..", "..", "..", "..", "src"))
    gsw_root = os.path.join(repo_src, "gsw_memory")
    paths = {
        "gsw_memory.query_then_sleep.transport": os.path.join(
            gsw_root, "query_then_sleep", "transport.py"
        ),
        "gsw_memory.memory.operator_utils.agentic_gsw.transport_shim": os.path.join(
            gsw_root, "memory", "operator_utils", "agentic_gsw", "transport_shim.py"
        ),
    }

    def _load(name: str):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, paths[name])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build spec for {name} at {paths[name]}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        # The transport_shim file does `from ....query_then_sleep.transport import StageTransport`
        # which requires the package names to resolve. Preload the dep.
        if name.endswith("transport_shim"):
            _load("gsw_memory.query_then_sleep.transport")
        spec.loader.exec_module(mod)
        return mod

    mod = _load("gsw_memory.memory.operator_utils.agentic_gsw.transport_shim")
    return mod.TransportShim


# ----- LLM-facing schema -------------------------------------------------


class DraftOutput(BaseModel):
    """What the script-writer LLM returns each iteration.

    ``extra="forbid"`` makes Pydantic emit ``additionalProperties: false``
    in the JSON schema, which is required by OpenAI's structured-output
    mode (otherwise the call 400s).
    """

    # ``extra="forbid"`` emits ``additionalProperties: false``; marking every
    # field required (no defaults) emits them in ``required`` — both are
    # mandatory for OpenAI structured outputs. Without these, every call
    # falls through the shim's repair path, doubling latency and tokens.
    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(
        ...,
        description=(
            "Brief (<=5 sentence) commentary on the strategy the emitted code uses. "
            "Not shipped into the final module."
        ),
    )
    skeletons_implemented: list[str] = Field(
        ...,
        description=(
            "List of distinct skeleton labels your decompose() function "
            "emits branches for. Use the skeleton labels from the preamble "
            "(e.g. 'filter', 'bridged_filter', 'argmax', 'argmin', 'count', "
            "'aggregate_mean', 'aggregate_sum', 'sort_topk', 'intersection', "
            "'temporal_filter_then_argmax', 'deep_composition', "
            "'musique_of_chain'). Must contain at least 5 distinct labels. "
            "Each label must correspond to a code branch that emits a "
            "structurally distinct decomposition (different step count or "
            "different final operator). A module that emits only 3-step "
            "filters regardless of question shape fails this contract."
        ),
    )
    code: str = Field(
        ...,
        description=(
            "Full Python source for the decomposer module. Must define "
            "`def decompose(question: str) -> list[SubQuestion]`. Only `re`, "
            "`string`, stdlib allowed. No LLM / network imports. Each "
            "skeleton you declare above must map to a distinct branch in "
            "the source (different regex, different step count, different "
            "final operator)."
        ),
    )


# ----- Record-keeping ----------------------------------------------------


class InspectorOutput(BaseModel):
    """What the Inspector emits after reading training examples.

    The Inspector's job: read raw training questions + gold
    decompositions, derive the skeleton families that actually appear
    in this dataset, and author the teaching section of the Scriptor's
    system prompt directly from the data. The Scriptor then writes code.

    This replaces the author-by-hand preamble. We do NOT tell the
    Inspector which skeletons to look for — it discovers them from the
    data. That's the point.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_analysis: str = Field(
        ...,
        description=(
            "2-4 sentences summarizing the decomposition style of this "
            "dataset: typical depth, operator vocabulary, how references "
            "work, any dataset-specific markers like '[list]' or 'return "
            "#N where #M ...'."
        ),
    )
    skeleton_families: list[str] = Field(
        ...,
        description=(
            "Named skeleton families observed in training (e.g. 'filter_3', "
            "'bridged_filter_4', 'argmax_5', 'deep_of_chain_7'). The name "
            "should encode both the semantic shape AND the gold step "
            "count, so the Scriptor knows to branch by step count."
        ),
    )
    teaching_prompt: str = Field(
        ...,
        description=(
            "The full instructional section that will be appended to the "
            "Scriptor's system prompt (after the hard contract). Include: "
            "(1) a short taxonomy list, (2) 1-2 worked examples PER "
            "skeleton family picked DIRECTLY from the training examples "
            "you just read (copy the question + gold decomposition "
            "verbatim — don't paraphrase), (3) explicit step-count rules "
            "mapping inner modifiers to skeleton choices, (4) the "
            "step-count discipline warning. Do NOT restate the hard "
            "contract (zero-LLM, imports, SubQuestion schema, quoting "
            "discipline, semantic gate) — it is already included. "
            "Target length: 40-80 lines."
        ),
    )


@dataclass
class IterationRecord:
    iteration: int
    score: dict
    source_path: str
    source_preview: str = ""
    error: str = ""
    samples: list = field(default_factory=list)


@dataclass
class SynthesisResult:
    best_iteration: int
    best_score: dict
    best_source: str
    best_module_path: str
    iterations: List[IterationRecord] = field(default_factory=list)

    def as_markdown_table(self) -> str:
        header = (
            "| iter | coverage | count_match | soft_match | "
            "cm_2hop | cm_3hop | cm_4hop | fail | error |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|"
        rows = [header, sep]
        for rec in self.iterations:
            s = rec.score or {}
            bh = s.get("by_hop") or {}

            def _cm(bucket: str) -> str:
                row = bh.get(bucket) or {}
                if not row.get("n"):
                    return "—"
                return f"{float(row.get('count_match', 0.0)):.2f}"

            rows.append(
                "| {it} | {cov:.3f} | {cm:.3f} | {sm:.3f} | "
                "{h2} | {h3} | {h4} | {fc} | {err} |".format(
                    it=rec.iteration,
                    cov=float(s.get("coverage", 0.0)),
                    cm=float(s.get("count_match", 0.0)),
                    sm=float(s.get("soft_match", 0.0)),
                    h2=_cm("2hop"),
                    h3=_cm("3hop"),
                    h4=_cm("4hop"),
                    fc=int(s.get("fail_count", 0)),
                    err=(rec.error or "")[:60],
                )
            )
        return "\n".join(rows)


# ----- Prompts -----------------------------------------------------------


_HARD_CONTRACT = (
    "You author a DETERMINISTIC question-decomposer Python module. At "
    "inference time the module runs with ZERO LLM calls — regex, rule "
    "tables, and hand-crafted patterns only.\n\n"
    "## Hard contract\n"
    "* Define exactly one public function:\n"
    "    def decompose(question: str) -> list[SubQuestion]\n"
    "* Import SubQuestion via:\n"
    "    from research_agent.decomposers._schema import SubQuestion\n"
    "  Fields: `sub_id`, `question`, `entity_focus`, `relation_hint`, "
    "`hop_type`, `references`. Do NOT redefine locally.\n"
    "* Allowed imports: re, string, dataclasses, typing, functools, "
    "itertools, collections. No LLM / network / subprocess / importlib / "
    "gsw_memory.\n"
    "* Each SubQuestion has unique `sub_id` like 's1' (internal ID). "
    "Populate `references=['s1', 's2', ...]` with the internal sub_ids of "
    "the steps it depends on.\n"
    "* PLACEHOLDER CONVENTION in the question text: use '#N' where N is "
    "the 1-based step index — e.g. 'Is #1 a sorcerer?', 'return #1 where "
    "#2 is true', 'return number of #1'. NEVER use '#s1' or '#sN' — the "
    "gold datasets (MoNaCo QDMR, MuSiQue) use bare '#1, #2, #3' and a "
    "module that emits '#s1' scores 0 on every metric.\n"
    "* Return [] when you cannot confidently split — no hallucinated "
    "decomposition.\n"
    "* QUOTING: regex with apostrophes must use `r\"...\"` (double-quoted "
    "raw string). `r'[A-Z][\\w\\-\\']+'` is a SyntaxError.\n"
    "* SEMANTIC GATE: every emitted sub-question text (with '#N' "
    "placeholders stripped) must consist of whole words present in the "
    "original question. Mid-word regex captures like 'f >> or' or "
    "'#2reator died' are auto-rejected by the grader. Use word-boundary "
    "anchors (`\\bof\\b`) and exact character-span substitution, never "
    "`re.sub(entity, '#1', current)`.\n"
    "* LIFT SPANS, DO NOT TEMPLATE. Build each sub-question's text by "
    "slicing substrings out of the ORIGINAL question string, not from a "
    "prewritten template. Examples the grader REJECTS: "
    "'List all items mentioned in the question' (words 'list', 'all', "
    "'items', 'mentioned' rarely appear in the input); "
    "'What are the itemss?' (hallucinated plural); "
    "'List all nations in south americas?' ('americas' plural not in "
    "'South America'); 'Is #1 sorcerers??' (plural 'sorcerers' vs input "
    "singular 'sorcerer', double '?'). Examples the grader ACCEPTS: "
    "'Who are the characters in HBO\\'s The Witcher? [list]' "
    "(every content word copied from input); 'Is #1 a sorcerer?' "
    "(singular lifted exactly). Pluralization, case, and punctuation "
    "must match the source tokens.\n"
    "* Allowed framing words (whitelisted by the grader): what, who, "
    "whom, whose, where, when, why, how, which, is, are, was, were, be, "
    "the, a, an, of, in, at, on, by, for, to, and, or, return, list, "
    "[list], different, sort, sorted, ascending, descending, filter, "
    "select, project, aggregate, group, compare, count, sum, average, "
    "max, min, same, highest, lowest, number, each, than, that, these, "
    "those, this. Any other word MUST be a substring-match to the "
    "original question — or the prediction is rejected.\n"
)


# A small reference implementation that's safe for 'X of Y' peel chains.
# Kept for MuSiQue-style preambles where of/possessive composition is the
# dominant pattern. NOT included in the MoNaCo path anymore — that path
# is inspector-authored.
_MUSIQUE_COOKBOOK = (
    "## REGEX COOKBOOK (safe reference for 'X of Y' peel chains)\n\n"
    "```python\n"
    "import re\n"
    "from research_agent.decomposers._schema import SubQuestion\n\n"
    "OF_RE = re.compile(\n"
    "    r\"\\b(?P<rel>[\\w\\s\\-]+?)\\s+of\\s+\"\n"
    "    r\"(?P<ent>(?:the\\s+|a\\s+)?[A-Z][\\w\\s\\-.,'()]*?)\"\n"
    "    r\"(?=\\s+(?:of|where|who|that|and|is|was)\\b|[?.!,]|$)\",\n"
    "    re.IGNORECASE,\n"
    ")\n"
    "POSS_RE = re.compile(\n"
    "    r\"\\b(?P<ent>(?:the\\s+)?[A-Z][\\w\\-']*?)'s\\s+\"\n"
    "    r\"(?P<rel>[\\w\\s\\-]+?)(?=[?.!,]|$)\"\n"
    ")\n"
    "# peel rightmost match iteratively; substitute via start/end spans.\n"
    "```\n"
    "Take the RIGHTMOST match (`list(re.finditer(...))[-1]`) so the "
    "INNERMOST hop is emitted first. Substitute with exact spans: "
    "`current[:m.start('ent')] + '#' + sid + current[m.end('ent'):]`.\n"
)


# Back-compat for any external caller that still imports _PREAMBLE_CORE.
# Dataset-specific teaching is now inspector-generated (see inspect_training).
_PREAMBLE_CORE = _HARD_CONTRACT


_PREAMBLE_MUSIQUE_EXAMPLES = (
    "## Target dataset: MuSiQue — 'X of Y' composition chains\n"
    "MuSiQue gold decompositions are 2-4 sequential hops that peel "
    "'of' / possessive layers. Implement iterative peeling with a loop, "
    "not a fixed 2-hop regex.\n"
)


# MoNaCo teaching is now authored by the Inspector stage from training data
# (see inspect_training below). Kept as an empty placeholder so older
# callers using _PREAMBLE_MONACO_EXAMPLES still import something.
_PREAMBLE_MONACO_EXAMPLES = ""


def _preamble_for(mode_label: str, *, inspector_pedagogy: str = "") -> str:
    """Assemble the Script-Writer system prompt.

    For MuSiQue: hard contract + regex cookbook + light hint. For MoNaCo
    and any other dataset: hard contract + inspector-authored pedagogy
    (empty if inspector has not run yet — scriptor will learn from the
    raw training examples in the user prompt).
    """
    lab = mode_label.lower()
    if "monaco" in lab:
        pedagogy = inspector_pedagogy or ""
        return _HARD_CONTRACT + ("\n" + pedagogy if pedagogy else "")
    if "musique" in lab:
        base = _HARD_CONTRACT + "\n" + _MUSIQUE_COOKBOOK + "\n" + _PREAMBLE_MUSIQUE_EXAMPLES
        return base + ("\n" + inspector_pedagogy if inspector_pedagogy else "")
    # Default: minimal contract + inspector pedagogy if provided.
    return _HARD_CONTRACT + ("\n" + inspector_pedagogy if inspector_pedagogy else "")


# Back-compat alias — old code paths reference _PREAMBLE.
_PREAMBLE = _HARD_CONTRACT + "\n" + _MUSIQUE_COOKBOOK + "\n" + _PREAMBLE_MUSIQUE_EXAMPLES


def _format_examples(examples: Iterable[dict], *, max_n: int = 20) -> str:
    blocks: List[str] = []
    for i, ex in enumerate(examples):
        if i >= max_n:
            break
        q = ex.get("question", "")
        gold = ex.get("gold_decomposition", []) or []
        gold_block = json.dumps(gold, ensure_ascii=False, indent=2)
        blocks.append(
            f"Example {i + 1}:\n  question: {q}\n  gold_decomposition:\n{gold_block}"
        )
    return "\n\n".join(blocks)


_INSPECTOR_SYSTEM_PROMPT = (
    "You are the Inspector. You read a set of ACTUAL gold question-"
    "decomposition examples from a target dataset and author the TEACHING "
    "section of a downstream Scriptor's system prompt. The Scriptor will "
    "then write a deterministic Python module (regex + rule tables, zero "
    "LLM calls at inference) that emits decompositions matching this "
    "dataset's gold style.\n\n"
    "## ABSOLUTELY CRITICAL RULES\n"
    "1. You MUST ground the teaching prompt in the training examples shown "
    "to you in the user message. Do NOT invent examples. Do NOT write "
    "SQL-query examples, pseudocode, or synthetic scenarios. Every worked "
    "example you include MUST be copied VERBATIM from the training set: "
    "the exact question string and the exact gold decomposition steps. "
    "If you cannot find a real training example for a skeleton family, "
    "do NOT include that family.\n"
    "2. Derive the operator vocabulary from WHAT YOU SEE. If training uses "
    "'[list]' markers, preserve them. If it uses 'return #N where #M is "
    "highest', preserve that exact phrasing. Do NOT translate into generic "
    "terms like 'filter' or 'aggregate' unless the training text does.\n"
    "3. Name each skeleton family with its gold step count: 'filter_3' "
    "means a 3-step filter. The Scriptor uses the name to branch by step "
    "count.\n"
    "4. You write PROMPT TEXT, not code. Do NOT restate the hard contract "
    "(zero-LLM imports, SubQuestion schema, quoting discipline, semantic "
    "whole-word gate) — that is already in the Scriptor's system prompt.\n\n"
    "## What the teaching_prompt should contain\n"
    "(a) A taxonomy list: <skeleton_name> → one-sentence shape description.\n"
    "(b) 1-2 worked examples per skeleton, COPIED from training (question "
    "    string + numbered list of gold sub-questions). Prefix each block "
    "    with 'From training:' so the Scriptor knows it is ground truth.\n"
    "(c) Step-count rules mapping inner-question markers (e.g. 'since', "
    "    'largest', 'both', 'how many') to the skeleton family whose step "
    "    count they trigger.\n"
    "(d) One line warning: 'emit no decomposition when unsure rather than a "
    "    default 3-step skeleton'."
)


def inspect_training(
    *,
    train_examples: list,
    mode_label: str,
    shim: Any,
    max_examples: int = 40,
    previous_pedagogy: str = "",
    failures: list | None = None,
) -> InspectorOutput:
    """Run the Inspector LLM call.

    Returns an ``InspectorOutput`` whose ``teaching_prompt`` is appended
    to the Scriptor's system prompt. When ``previous_pedagogy`` and
    ``failures`` are provided, the Inspector is asked to REVISE the
    teaching prompt based on what failed last iteration.
    """
    user = (
        f"Target dataset: {mode_label}\n\n"
        f"Training examples (read these carefully — these are the gold "
        f"decompositions you need to teach the Scriptor to reproduce):\n"
        f"{_format_examples(train_examples, max_n=max_examples)}\n\n"
    )
    if previous_pedagogy:
        user += (
            "Your previous teaching prompt (the Scriptor built on this and "
            "graded below target):\n---\n"
            f"{previous_pedagogy}\n---\n\n"
        )
    if failures:
        fmt = []
        for f in failures[:6]:
            q = f.get("question", "")
            gold = json.dumps(f.get("gold_decomposition", []), ensure_ascii=False, indent=2)
            pred = json.dumps(f.get("pred_decomposition", []), ensure_ascii=False, indent=2)
            gn, pn = f.get("gold_hop_count"), f.get("pred_hop_count")
            head = f"- question: {q}\n"
            if gn is not None and pn is not None:
                head += f"  hop count: gold={gn}  scriptor={pn}\n"
            fmt.append(head + f"  gold:\n{gold}\n  scriptor emitted:\n{pred}\n")
        user += (
            "Scriptor failure cases from the previous iteration:\n"
            + "\n".join(fmt)
            + "\n\nRevise the teaching prompt so the Scriptor fixes these. "
            "If the failures show it defaulted to a shallow skeleton, add a "
            "harder worked example from training that forces the right depth.\n"
        )
    else:
        user += (
            "Write the teaching_prompt field such that a non-reasoning "
            "scriptor can produce correct code. Include worked examples "
            "COPIED verbatim from the training set above. Prefer concrete "
            "examples over prose.\n"
        )

    out = shim.call(
        system_prompt=_INSPECTOR_SYSTEM_PROMPT,
        user_prompt=user,
        response_model=InspectorOutput,
        stage="decomposer_inspector",
        max_tokens=15000,
    )
    return out


def _build_initial_user_prompt(
    *,
    mode_label: str,
    train_examples: list,
    dev_examples: list,
) -> str:
    return (
        f"Target mode: {mode_label}\n\n"
        f"Training examples (learn the patterns from these):\n"
        f"{_format_examples(train_examples, max_n=30)}\n\n"
        f"Dev preview (do NOT overfit to these — they will be used to grade your module):\n"
        f"{_format_examples(dev_examples, max_n=6)}\n\n"
        "Emit one Python module that generalises across the training examples. "
        "Prefer a small set of robust patterns over a long chain of brittle "
        "special-cases. Cover comparison ('which X was released first, A or B?'), "
        "composition ('X of Y of Z'), bridge ('place of birth of the director "
        "of film X'), and intersection patterns where they appear."
    )


def _build_repair_user_prompt(
    *,
    mode_label: str,
    previous_code: str,
    failures: list,
    previous_score: Optional[dict] = None,
    previous_error: str = "",
) -> str:
    formatted = []
    for item in failures:
        q = item.get("question", "")
        gold = json.dumps(item.get("gold_decomposition", []), ensure_ascii=False, indent=2)
        pred = json.dumps(item.get("pred_decomposition", []), ensure_ascii=False, indent=2)
        gold_n = item.get("gold_hop_count")
        pred_n = item.get("pred_hop_count")
        err = item.get("error", "")
        header = f"- question: {q}\n"
        if gold_n is not None and pred_n is not None:
            header += f"  hop count: gold={gold_n}  your module={pred_n}"
            if pred_n < gold_n:
                header += "   <-- UNDER-DECOMPOSED (biggest failure mode)"
            header += "\n"
        block = header + f"  gold:\n{gold}\n  your module produced:\n{pred}\n"
        if err:
            block += f"  runtime error: {err}\n"
        formatted.append(block)
    failure_block = "\n".join(formatted) if formatted else "(no failure examples provided)"

    score_block = ""
    if previous_score:
        bh = previous_score.get("by_hop") or {}

        def _fmt(bucket: str) -> str:
            row = bh.get(bucket) or {}
            if not row.get("n"):
                return f"  {bucket}: no examples in dev"
            return (
                f"  {bucket}: n={row['n']} coverage={row['coverage']:.3f} "
                f"count_match={row['count_match']:.3f} soft_match={row['soft_match']:.3f}"
            )

        score_block = (
            "Previous-iteration grade on the dev split:\n"
            f"  overall: coverage={previous_score.get('coverage', 0):.3f} "
            f"count_match={previous_score.get('count_match', 0):.3f} "
            f"soft_match={previous_score.get('soft_match', 0):.3f}\n"
            f"{_fmt('2hop')}\n{_fmt('3hop')}\n{_fmt('4hop')}\n\n"
            "If count_match on 3hop / 4hop is near zero, your decomposer is "
            "stopping too early. Fix the iteration: keep peeling nested "
            "'of X' / possessive layers until the pattern no longer matches, "
            "not just once.\n\n"
        )

    error_block = ""
    if previous_error:
        error_block = (
            "HARD ERROR from the previous draft — fix this FIRST before "
            "anything else:\n"
            f"  {previous_error}\n\n"
            "If this is a SyntaxError, check your string quoting — "
            "regex patterns with apostrophes MUST use `r\"...\"` not "
            "`r'...'`.\n\n"
        )

    return (
        f"Target mode: {mode_label}\n\n"
        f"{error_block}"
        f"{score_block}"
        f"Current module source (it graded below target):\n"
        f"```python\n{previous_code}\n```\n\n"
        f"Failure cases from the dev grade:\n{failure_block}\n\n"
        "Rewrite the module from scratch — keep what worked, fix what didn't. "
        "Preserve coverage where you already had it. Aim to strictly dominate "
        "the previous score, ESPECIALLY count_match on 3-hop and 4-hop."
    )


# ----- Failure sampling --------------------------------------------------


def _safe_subq_dict(s: Any) -> dict:
    """Defensive dict extractor — handles emitted modules that may redefine
    ``SubQuestion`` locally instead of importing ours. Only ``question`` is
    required; everything else is best-effort.
    """
    get = getattr
    out: dict[str, Any] = {}
    for key in ("sub_id", "question", "entity_focus", "relation_hint", "hop_type"):
        val = get(s, key, None)
        if val is not None:
            out[key] = val
    if "question" not in out:
        # Last resort: str() the object so the repair prompt still shows something.
        out["question"] = str(s)
    return out


def _collect_failures(
    decompose_fn: Callable,
    examples: list,
    *,
    max_n: int = 6,
) -> list:
    """Return up to ``max_n`` failure exemplars for the repair prompt."""
    out = []
    for ex in examples:
        q = ex.get("question", "")
        gold = ex.get("gold_decomposition", []) or []
        try:
            pred = decompose_fn(q) or []
        except BaseException as exc:
            out.append(
                {
                    "question": q,
                    "gold_decomposition": gold,
                    "pred_decomposition": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if len(out) >= max_n:
                break
            continue

        # A simple "this one failed" heuristic: count mismatch OR empty pred
        # when gold is non-empty.
        if (not pred and gold) or (len(pred) != len(gold)):
            out.append(
                {
                    "question": q,
                    "gold_hop_count": len(gold),
                    "pred_hop_count": len(pred),
                    "gold_decomposition": gold,
                    "pred_decomposition": [_safe_subq_dict(s) for s in pred],
                    "error": "",
                }
            )
            if len(out) >= max_n:
                break
    return out


# ----- Main entry point --------------------------------------------------


def _score_strictly_better(new: dict, incumbent: Optional[dict]) -> bool:
    if incumbent is None:
        return True
    if new.get("soft_match", 0.0) > incumbent.get("soft_match", 0.0):
        return True
    if new.get("soft_match", 0.0) == incumbent.get("soft_match", 0.0):
        return new.get("count_match", 0.0) > incumbent.get("count_match", 0.0)
    return False


def synthesize_decomposer(
    *,
    mode_label: str,
    train_examples: list,
    dev_examples: list,
    out_path: str | Path,
    model_name: str,
    base_url: Optional[str] = None,
    max_iterations: int = 5,
    patience: int = 2,
    reasoning_effort: str = "medium",
    work_dir: Optional[str | Path] = None,
    seed_module_path: Optional[str | Path] = None,
) -> SynthesisResult:
    """Run the draft -> grade -> repair loop and write the winning module.

    Returns a ``SynthesisResult`` with the best score and iteration table.
    Also writes the best module's source to ``out_path``.
    """

    TransportShim = _get_transport_shim()
    shim = TransportShim(
        model_name=model_name,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )

    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="decomp_synth_"))
    work.mkdir(parents=True, exist_ok=True)

    iterations: List[IterationRecord] = []
    best_score: Optional[dict] = None
    best_source: str = ""
    best_path: str = ""
    stale = 0

    previous_code: Optional[str] = None
    previous_fn: Optional[Callable] = None
    previous_score: Optional[dict] = None
    previous_error: str = ""

    # Inspector stage: teaching prompt is authored from training data.
    # MuSiQue also uses the inspector — it layers on top of the regex
    # cookbook instead of replacing it (see _preamble_for).
    inspector_pedagogy = ""
    if True:
        try:
            inspector_out = inspect_training(
                train_examples=train_examples,
                mode_label=mode_label,
                shim=shim,
            )
            inspector_pedagogy = inspector_out.teaching_prompt or ""
            (work / "inspector_round0.md").write_text(
                f"# Inspector round 0\n\n"
                f"## dataset_analysis\n{inspector_out.dataset_analysis}\n\n"
                f"## skeleton_families\n"
                + "\n".join(f"- {s}" for s in inspector_out.skeleton_families)
                + f"\n\n## teaching_prompt\n{inspector_pedagogy}\n"
            )
        except Exception as exc:
            iterations.append(
                IterationRecord(
                    iteration=-2,
                    score={},
                    source_path="",
                    error=f"inspector_failed: {type(exc).__name__}: {exc}",
                )
            )

    # Seed from an existing module if requested. This lets the loop run
    # as "improve this" instead of "write from scratch" — the first
    # iteration sees the seed's failures and rewrites against them.
    if seed_module_path:
        seed_path = Path(seed_module_path)
        if seed_path.exists():
            try:
                seed_fn = load_decomposer(seed_path)
                seed_score = grade_decomposer(seed_fn, dev_examples)
                previous_code = seed_path.read_text()
                previous_fn = seed_fn
                previous_score = seed_score
                best_score = seed_score
                best_source = previous_code
                best_path = str(seed_path)
                iterations.append(
                    IterationRecord(
                        iteration=-1,
                        score=seed_score,
                        source_path=str(seed_path),
                        source_preview=previous_code[:400],
                        error="seeded",
                    )
                )
            except Exception as exc:
                iterations.append(
                    IterationRecord(
                        iteration=-1,
                        score={},
                        source_path=str(seed_path),
                        error=f"seed_load_failed: {type(exc).__name__}: {exc}",
                    )
                )

    for it in range(max_iterations):
        if (it == 0 and previous_fn is None) or previous_code is None:
            user_prompt = _build_initial_user_prompt(
                mode_label=mode_label,
                train_examples=train_examples,
                dev_examples=dev_examples,
            )
        else:
            failures = (
                _collect_failures(previous_fn, dev_examples, max_n=6)
                if previous_fn is not None
                else []
            )
            user_prompt = _build_repair_user_prompt(
                mode_label=mode_label,
                previous_code=previous_code,
                failures=failures,
                previous_score=previous_score,
                previous_error=previous_error,
            )

        try:
            draft = shim.call(
                system_prompt=_preamble_for(mode_label, inspector_pedagogy=inspector_pedagogy),
                user_prompt=user_prompt,
                response_model=DraftOutput,
                stage=f"decomposer_synth_iter{it}",
                max_tokens=15000,
            )
        except Exception as exc:
            iterations.append(
                IterationRecord(
                    iteration=it,
                    score={},
                    source_path="",
                    error=f"draft_call_failed: {type(exc).__name__}: {exc}",
                )
            )
            stale += 1
            if stale >= patience:
                break
            continue

        draft_path = work / f"draft_iter{it}.py"
        draft_path.write_text(draft.code)

        # Guard: fail fast if the drafted code imports forbidden libs.
        try:
            hits = scan_source_for_forbidden_imports(draft.code)
        except ForbiddenImportError as exc:
            iterations.append(
                IterationRecord(
                    iteration=it,
                    score={},
                    source_path=str(draft_path),
                    source_preview=draft.code[:400],
                    error=f"source_parse_failed: {exc}",
                )
            )
            previous_code = draft.code
            previous_fn = None
            previous_error = f"source_parse_failed: {exc}"
            stale += 1
            if stale >= patience:
                break
            continue
        if hits:
            iterations.append(
                IterationRecord(
                    iteration=it,
                    score={},
                    source_path=str(draft_path),
                    source_preview=draft.code[:400],
                    error=f"forbidden_imports: {sorted(set(hits))}",
                )
            )
            previous_code = draft.code
            previous_fn = None
            previous_error = f"forbidden_imports: {sorted(set(hits))}"
            stale += 1
            if stale >= patience:
                break
            continue

        try:
            fn = load_decomposer(draft_path)
        except Exception as exc:
            iterations.append(
                IterationRecord(
                    iteration=it,
                    score={},
                    source_path=str(draft_path),
                    source_preview=draft.code[:400],
                    error=f"load_failed: {type(exc).__name__}: {exc}",
                )
            )
            previous_code = draft.code
            previous_fn = None
            previous_error = f"load_failed: {type(exc).__name__}: {exc}"
            stale += 1
            if stale >= patience:
                break
            continue

        score = grade_decomposer(fn, dev_examples)
        # Record a handful of concrete decompositions so we can EYE them,
        # not just trust structural scores. Pull one example per hop bucket
        # for variety.
        samples: list = []
        seen_buckets: set[str] = set()
        for ex in dev_examples:
            bucket = ex.get("hop") or ""
            if bucket in seen_buckets:
                continue
            q = ex.get("question", "")
            try:
                pred = fn(q) or []
                pred_dump = [_safe_subq_dict(s) for s in pred]
            except BaseException as exc:
                pred_dump = [{"error": f"{type(exc).__name__}: {exc}"}]
            samples.append(
                {
                    "hop": bucket,
                    "question": q,
                    "gold": [
                        {"question": step.get("question", ""), "answer": step.get("answer", "")}
                        for step in (ex.get("gold_decomposition") or [])
                    ],
                    "pred": pred_dump,
                }
            )
            seen_buckets.add(bucket)
            if len(samples) >= 3:
                break
        iterations.append(
            IterationRecord(
                iteration=it,
                score=score,
                source_path=str(draft_path),
                source_preview=draft.code[:400],
                samples=samples,
            )
        )

        if _score_strictly_better(score, best_score):
            best_score = score
            best_source = draft.code
            best_path = str(draft_path)
            stale = 0
        else:
            stale += 1
            # Inspector revision: ask the Inspector to fix the teaching
            # prompt using the scriptor's concrete failures.
            if stale < patience and it + 1 < max_iterations:
                try:
                    failures_for_inspector = _collect_failures(fn, dev_examples, max_n=6)
                    revised = inspect_training(
                        train_examples=train_examples,
                        mode_label=mode_label,
                        shim=shim,
                        previous_pedagogy=inspector_pedagogy,
                        failures=failures_for_inspector,
                    )
                    if revised.teaching_prompt:
                        inspector_pedagogy = revised.teaching_prompt
                        (work / f"inspector_round{it + 1}.md").write_text(
                            f"# Inspector round {it + 1} (revision)\n\n"
                            f"## dataset_analysis\n{revised.dataset_analysis}\n\n"
                            f"## skeleton_families\n"
                            + "\n".join(f"- {s}" for s in revised.skeleton_families)
                            + f"\n\n## teaching_prompt\n{inspector_pedagogy}\n"
                        )
                except Exception as exc:
                    iterations.append(
                        IterationRecord(
                            iteration=it,
                            score={},
                            source_path="",
                            error=f"inspector_revise_failed: {type(exc).__name__}: {exc}",
                        )
                    )

        previous_code = draft.code
        previous_fn = fn
        previous_score = score
        previous_error = ""

        if stale >= patience:
            break

    # Persist the winning module to the requested path.
    if best_source:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(best_source)

    return SynthesisResult(
        best_iteration=(
            iterations[-1].iteration if not iterations else (
                next(
                    (rec.iteration for rec in iterations if rec.source_path == best_path),
                    -1,
                )
            )
        ),
        best_score=best_score or {},
        best_source=best_source,
        best_module_path=str(out_path) if best_source else "",
        iterations=iterations,
    )
