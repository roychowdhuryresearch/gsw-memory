"""Dump GSW-fragment plans for human inspection.

Runs the Phase-1 planner LLM (``build_planner_messages``) against a FRAMES
split — by default the 100-Q dev split — across one or more models, and
persists per-question artefacts:

- ``logs/planner_dump_<ts>/<model_slug>/q_<id>.json`` — full raw payload
  (question / gold / plan_dict / raw_response / raw_reasoning /
  repair_response / repair_reasoning / tokens / wall_s).
- ``logs/planner_dump_<ts>/<model_slug>/q_<id>.md`` — human-readable render
  with entity / verb-phrase / constraint tables, collapsed raw response,
  collapsed reasoning, collapsed full prompt.
- ``logs/planner_dump_<ts>/<model_slug>/index.md`` — per-model summary table.
- ``logs/planner_dump_<ts>/index.md`` — cross-model comparison.

No retrieval, no execution, no answer — this CLI only emits and persists
plans for visual inspection.

Usage
-----

Smoke (3 Qs, one model):

    .venv/bin/python playground/dump_plans.py --limit 3 \\
        --models bedrock/openai.gpt-oss-120b-1:0

Full run (100 Qs × 2 models):

    .venv/bin/python playground/dump_plans.py \\
        --models bedrock/openai.gpt-oss-120b-1:0,bedrock/openai.gpt-oss-20b-1:0
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Make research_agent importable without PYTHONPATH=src.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import typer
from pydantic import ValidationError

from research_agent.adapters.ours._planner_exec import GSWPlan, _extract_json_object
from research_agent.adapters.ours._planner_prompts import (
    build_planner_messages,
    build_repair_messages,
)
from research_agent.eval.frames_dataset import load_frames
from research_agent.models.llm_client import LLMClient
from research_agent.utils.logging_setup import setup_run_logger, get_logger

try:
    from dotenv import load_dotenv

    load_dotenv()
    _PARENT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
    if _PARENT_ENV.exists():
        load_dotenv(_PARENT_ENV, override=False)
except ImportError:  # pragma: no cover
    pass


app = typer.Typer(add_completion=False)
_log = get_logger("dump_plans")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify_model(m: str) -> str:
    return m.replace("/", "_").replace(":", "_").replace(".", "_")


def _messages_to_text(messages: list[dict]) -> tuple[str, str]:
    """Split planner messages into (system_prompt, user_prompt) text."""
    sys_text = ""
    user_text = ""
    for m in messages:
        if m["role"] == "system":
            sys_text = m["content"]
        elif m["role"] == "user":
            user_text = m["content"]
    return sys_text, user_text


def _call_planner(
    llm: LLMClient, question: str, max_tokens: int
) -> tuple[str, str, int, int, float, list[dict]]:
    """Return (raw_text, reasoning, p_tok, c_tok, wall_s, messages)."""
    messages = build_planner_messages(question)
    t0 = time.time()
    resp = llm.chat(messages=messages, max_tokens=max_tokens)
    wall = time.time() - t0
    return (
        getattr(resp, "text", "") or "",
        getattr(resp, "reasoning_content", "") or "",
        int(getattr(resp, "prompt_tokens", 0) or 0),
        int(getattr(resp, "completion_tokens", 0) or 0),
        wall,
        messages,
    )


def _call_repair(
    llm: LLMClient, question: str, bad: str, error: str, max_tokens: int
) -> tuple[str, str, int, int, float]:
    messages = build_repair_messages(question, bad, error)
    t0 = time.time()
    resp = llm.chat(messages=messages, max_tokens=max_tokens)
    wall = time.time() - t0
    return (
        getattr(resp, "text", "") or "",
        getattr(resp, "reasoning_content", "") or "",
        int(getattr(resp, "prompt_tokens", 0) or 0),
        int(getattr(resp, "completion_tokens", 0) or 0),
        wall,
    )


def _try_parse(text: str) -> tuple[Optional[GSWPlan], Optional[str]]:
    try:
        obj = _extract_json_object(text)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"extract_json: {exc}"
    try:
        plan = GSWPlan.model_validate(obj)
    except ValidationError as exc:
        return None, f"validate: {exc}"
    return plan, None


# ---------------------------------------------------------------------------
# Markdown renderer (per Q)
# ---------------------------------------------------------------------------


def _render_md(record: dict) -> str:
    """Render a per-Q record to Obsidian-ready markdown."""
    q_id = record["question_id"]
    hops = record.get("num_hops", "?")
    model = record["model"]
    question = record["question"]
    gold = record["gold_answer"]
    wall = record.get("wall_s", 0.0)
    p_tok = record.get("prompt_tokens", 0)
    c_tok = record.get("completion_tokens", 0)
    parse_ok = record.get("parse_ok", False)
    parse_error = record.get("parse_error", "")
    repaired = record.get("repair_used", False)
    plan = record.get("plan")
    raw = record.get("raw_response", "")
    reasoning = record.get("raw_reasoning", "")
    repair_raw = record.get("repair_response", "")
    repair_reasoning = record.get("repair_reasoning", "")
    sys_prompt = record.get("system_prompt", "")
    user_prompt = record.get("user_prompt", "")

    status_icon = (
        "✓ (after repair)" if parse_ok and repaired else ("✓" if parse_ok else f"✗ {parse_error}")
    )

    lines: list[str] = []
    lines.append(f"# Q {q_id} ({hops}-hop) — {model}")
    lines.append("")
    lines.append(f"**Question**: {question}")
    lines.append("")
    lines.append(f"**Gold answer**: {gold}")
    lines.append("")
    lines.append(f"**Parse status**: {status_icon}")
    lines.append("")
    lines.append(f"**Wall**: {wall:.1f} s · **Tokens**: in={p_tok} out={c_tok}")
    lines.append("")

    if plan:
        entities = plan.get("entities", [])
        vps = plan.get("verb_phrases", [])
        constraints = plan.get("constraints", [])
        n_filled = sum(1 for e in entities if e.get("kind") == "filled")
        n_blank = sum(1 for e in entities if e.get("kind") == "blank")
        n_target = sum(1 for e in entities if e.get("is_target"))

        lines.append(
            f"## Entities  ({n_filled} filled + {n_blank} blank, {n_target} target)"
        )
        lines.append("")
        lines.append("| id | kind | name | value_type | target? | category? |")
        lines.append("|---|---|---|---|:-:|:-:|")
        for e in entities:
            lines.append(
                f"| `{e.get('id','')}` | {e.get('kind','')} | "
                f"{e.get('name') or '—'} | {e.get('value_type') or '—'} | "
                f"{'✓' if e.get('is_target') else ''} | "
                f"{'✓' if e.get('category') else ''} |"
            )
        lines.append("")

        lines.append(f"## VerbPhrases  ({len(vps)})")
        lines.append("")
        if vps:
            lines.append("| id | subject_id | phrase | object_id |")
            lines.append("|---|---|---|---|")
            for v in vps:
                lines.append(
                    f"| `{v.get('id','')}` | `{v.get('subject_id','')}` | "
                    f"{v.get('phrase','')} | `{v.get('object_id','')}` |"
                )
        else:
            lines.append("_(none)_")
        lines.append("")

        lines.append(f"## Constraints  ({len(constraints)})")
        lines.append("")
        if constraints:
            lines.append("| id | kind | op | inputs | output_blank_id |")
            lines.append("|---|---|---|---|---|")
            for c in constraints:
                inputs_s = ""
                if c.get("kind") == "derived":
                    inputs_s = ", ".join(f"`{x}`" for x in c.get("args_blanks", []))
                elif c.get("kind") in ("argmax", "argmin"):
                    candidates = c.get("candidate_entity_ids", [])
                    sort_by = c.get("sort_by_blank_ids", [])
                    inputs_s = "candidates=[" + ",".join(f"`{x}`" for x in candidates) + "] sort_by=[" + ",".join(f"`{x}`" for x in sort_by) + "]"
                else:
                    inputs_s = (
                        f"left=`{c.get('left_ref','—')}` right=`{c.get('right_ref','—')}`"
                    )
                lines.append(
                    f"| `{c.get('id','')}` | {c.get('kind','')} | "
                    f"{c.get('op') or '—'} | {inputs_s} | `{c.get('output_blank_id','—')}` |"
                )
        else:
            lines.append("_(none)_")
        lines.append("")
    else:
        lines.append("## Plan")
        lines.append("")
        lines.append(f"_Could not parse:_ `{parse_error}`")
        lines.append("")

    # Raw response
    lines.append("<details><summary>Raw LLM response</summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw or "(empty)")
    lines.append("```")
    lines.append("</details>")
    lines.append("")

    # Reasoning
    lines.append("<details><summary>Reasoning (planner call)</summary>")
    lines.append("")
    if reasoning.strip():
        lines.append("```")
        lines.append(reasoning)
        lines.append("```")
    else:
        lines.append("_Not emitted for this model (no reasoning_content returned)._")
    lines.append("</details>")
    lines.append("")

    if repaired:
        lines.append("<details><summary>Repair response + reasoning</summary>")
        lines.append("")
        lines.append("**Repair response:**")
        lines.append("```")
        lines.append(repair_raw or "(empty)")
        lines.append("```")
        lines.append("")
        lines.append("**Repair reasoning:**")
        lines.append("```")
        lines.append(repair_reasoning or "(not emitted)")
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    # Full prompt
    lines.append("<details><summary>Full prompt (system + user)</summary>")
    lines.append("")
    lines.append("**SYSTEM:**")
    lines.append("```")
    lines.append(sys_prompt)
    lines.append("```")
    lines.append("")
    lines.append("**USER:**")
    lines.append("```")
    lines.append(user_prompt)
    lines.append("```")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Index renderer (per model + cross-model)
# ---------------------------------------------------------------------------


def _render_model_index(model: str, records: list[dict], out_dir: Path) -> str:
    ok = sum(1 for r in records if r.get("parse_ok"))
    repaired = sum(1 for r in records if r.get("parse_ok") and r.get("repair_used"))
    failed = sum(1 for r in records if not r.get("parse_ok"))

    n_ents = [len(r.get("plan", {}).get("entities", [])) for r in records if r.get("plan")]
    n_vps = [len(r.get("plan", {}).get("verb_phrases", [])) for r in records if r.get("plan")]
    n_cons = [len(r.get("plan", {}).get("constraints", [])) for r in records if r.get("plan")]

    target_types = Counter()
    for r in records:
        plan = r.get("plan")
        if not plan:
            continue
        for e in plan.get("entities", []):
            if e.get("is_target"):
                target_types[e.get("value_type") or "unknown"] += 1
                break

    def _med(xs: list[int]) -> str:
        if not xs:
            return "—"
        return f"{statistics.median(xs):.1f}"

    lines = [
        f"# Plan dump index — {model}",
        "",
        f"**Total questions**: {len(records)}",
        f"**Parse OK**: {ok}  · after-repair: {repaired}  · **failed**: {failed}",
        f"**Parse validity**: {ok / max(len(records),1):.1%}",
        "",
        "## Shape distribution",
        "",
        f"| metric | median | mean | max |",
        f"|---|---:|---:|---:|",
        f"| entities | {_med(n_ents)} | {sum(n_ents)/max(len(n_ents),1):.1f} | {max(n_ents) if n_ents else 0} |",
        f"| verb_phrases | {_med(n_vps)} | {sum(n_vps)/max(len(n_vps),1):.1f} | {max(n_vps) if n_vps else 0} |",
        f"| constraints | {_med(n_cons)} | {sum(n_cons)/max(len(n_cons),1):.1f} | {max(n_cons) if n_cons else 0} |",
        "",
        "## Target value_type distribution",
        "",
    ]
    if target_types:
        for vt, n in target_types.most_common():
            lines.append(f"- `{vt}`: {n}")
    else:
        lines.append("_(no parsed plans)_")
    lines.append("")

    lines.append("## Per-question index")
    lines.append("")
    lines.append("| # | id | hops | parse | target type | gold | question |")
    lines.append("|---:|---|---:|:-:|---|---|---|")
    for i, r in enumerate(records, start=1):
        q_id = r["question_id"]
        hops = r.get("num_hops", "?")
        status = (
            "✓r"
            if r.get("parse_ok") and r.get("repair_used")
            else ("✓" if r.get("parse_ok") else "✗")
        )
        plan = r.get("plan")
        tv = "—"
        if plan:
            for e in plan.get("entities", []):
                if e.get("is_target"):
                    tv = e.get("value_type") or "—"
                    break
        gold = (r.get("gold_answer", "") or "")[:50]
        q = (r.get("question", "") or "")[:100].replace("|", r"\|")
        md_name = f"q_{q_id}.md"
        lines.append(
            f"| {i} | [`{q_id}`]({md_name}) | {hops} | {status} | `{tv}` | {gold} | {q} |"
        )
    return "\n".join(lines) + "\n"


def _render_crossmodel_index(model_records: dict[str, list[dict]], run_tag: str) -> str:
    lines = [
        f"# Plan dump — cross-model summary ({run_tag})",
        "",
        "| model | n | parse_ok | after-repair | failed | validity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, records in model_records.items():
        ok = sum(1 for r in records if r.get("parse_ok"))
        rep = sum(1 for r in records if r.get("parse_ok") and r.get("repair_used"))
        fail = sum(1 for r in records if not r.get("parse_ok"))
        v = ok / max(len(records), 1)
        slug = _slugify_model(model)
        lines.append(
            f"| [{model}]({slug}/index.md) | {len(records)} | {ok} | {rep} | {fail} | {v:.1%} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    models: str = typer.Option(
        "bedrock/openai.gpt-oss-120b-1:0,bedrock/openai.gpt-oss-20b-1:0",
        "--models",
        help="Comma-separated list of model ids to run the planner on.",
    ),
    split: str = typer.Option("dev", help="FRAMES split: dev (100) or full (824)."),
    limit: int = typer.Option(0, help="If >0, only run the first N questions."),
    max_completion_tokens: int = typer.Option(
        16000, help="Per-call completion cap."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output dir; default logs/planner_dump_<ts>/",
    ),
) -> None:
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        typer.echo("No models given.")
        raise typer.Exit(1)

    questions = load_frames(split=split)
    if limit > 0:
        questions = questions[:limit]

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("logs") / f"planner_dump_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_run_logger(output_dir)
    _log.info("run start | models=%s split=%s n=%d out=%s", model_list, split, len(questions), output_dir)

    model_records: dict[str, list[dict]] = {m: [] for m in model_list}

    for model in model_list:
        _log.info("model start | %s", model)
        slug = _slugify_model(model)
        model_dir = output_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        llm = LLMClient(
            model=model,
            base_url=None,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

        for i, q in enumerate(questions, start=1):
            q_id = q.id
            try:
                raw, reasoning, p_tok, c_tok, wall_1, messages = _call_planner(
                    llm, q.question, max_completion_tokens
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("q%s planner call failed: %s", q_id, exc)
                raw, reasoning, p_tok, c_tok, wall_1, messages = (
                    "",
                    "",
                    0,
                    0,
                    0.0,
                    build_planner_messages(q.question),
                )
                parse_ok = False
                parse_err = f"llm_call_error: {exc}"
                plan_dict = None
                repair_raw = ""
                repair_reasoning = ""
                repair_used = False
            else:
                plan, parse_err = _try_parse(raw)
                parse_ok = plan is not None
                plan_dict = plan.model_dump() if plan else None
                repair_raw = ""
                repair_reasoning = ""
                repair_used = False
                if not parse_ok:
                    # one repair retry
                    try:
                        repair_raw, repair_reasoning, pr, cr, wall_2 = _call_repair(
                            llm, q.question, raw, parse_err or "", max_completion_tokens
                        )
                        p_tok += pr
                        c_tok += cr
                        wall_1 += wall_2
                        plan2, parse_err2 = _try_parse(repair_raw)
                        if plan2 is not None:
                            plan = plan2
                            plan_dict = plan.model_dump()
                            parse_ok = True
                            parse_err = ""
                            repair_used = True
                        else:
                            parse_err = parse_err2 or parse_err
                    except Exception as exc:  # noqa: BLE001
                        parse_err = f"repair_error: {exc}"

            sys_prompt, user_prompt = _messages_to_text(messages)
            record = {
                "question_id": q_id,
                "question": q.question,
                "gold_answer": q.answer,
                "num_hops": q.num_hops,
                "model": model,
                "parse_ok": parse_ok,
                "parse_error": parse_err or "",
                "repair_used": repair_used,
                "plan": plan_dict,
                "raw_response": raw,
                "raw_reasoning": reasoning,
                "repair_response": repair_raw,
                "repair_reasoning": repair_reasoning,
                "wall_s": round(wall_1, 3),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
            }
            (model_dir / f"q_{q_id}.json").write_text(json.dumps(record, indent=2, default=str))
            (model_dir / f"q_{q_id}.md").write_text(_render_md(record))
            model_records[model].append(record)

            status = "✓r" if parse_ok and repair_used else ("✓" if parse_ok else "✗")
            _log.info(
                "q%s %s | %d/%d | hops=%s parse=%s wall=%.1fs tok=%d/%d",
                q_id, status, i, len(questions), q.num_hops,
                "ok" if parse_ok else parse_err[:40],
                wall_1, p_tok, c_tok,
            )

        # Per-model index
        (model_dir / "index.md").write_text(_render_model_index(model, model_records[model], model_dir))
        _log.info("model done | %s saved to %s", model, model_dir)

    # Cross-model index
    (output_dir / "index.md").write_text(_render_crossmodel_index(model_records, run_tag))

    # Terminal summary
    typer.echo("\n=== Plan dump summary ===")
    for model, records in model_records.items():
        ok = sum(1 for r in records if r.get("parse_ok"))
        rep = sum(1 for r in records if r.get("parse_ok") and r.get("repair_used"))
        n_ents = [len(r["plan"]["entities"]) for r in records if r.get("plan")]
        n_vps = [len(r["plan"]["verb_phrases"]) for r in records if r.get("plan")]
        n_cons = [len(r["plan"]["constraints"]) for r in records if r.get("plan")]
        typer.echo(
            f"  {model}: parse_ok={ok}/{len(records)} ({ok/max(len(records),1):.0%}) "
            f"after-repair={rep} "
            f"median(ents/vps/cons)="
            f"{statistics.median(n_ents) if n_ents else 0:.0f}/"
            f"{statistics.median(n_vps) if n_vps else 0:.0f}/"
            f"{statistics.median(n_cons) if n_cons else 0:.0f}"
        )
    typer.echo(f"\nArtefacts: {output_dir}")
    typer.echo(f"Open {output_dir}/index.md")


if __name__ == "__main__":
    app()
