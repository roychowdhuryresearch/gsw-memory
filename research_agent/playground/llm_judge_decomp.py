"""CLI: LLM-as-judge grading for a decomposer module.

Replaces the structural whole-word semantic gate with a model-graded
semantic match. For each question the judge compares the predicted
decomposition against the gold decomposition and returns a per-gold-step
coverage score plus a holistic overall match.

Usage:

    python playground/llm_judge_decomp.py \\
        --module logs/decomp_synth/musique_specialized_v5_cookbook.py \\
        --dataset musique \\
        --data ../playground_data/musique.json \\
        --split all --limit 100 \\
        --judge_model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field

import importlib.util

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from research_agent.decomposers import load_decomposer
from research_agent.decomposers._datasets import load_monaco, load_musique, split_train_dev


def _load_transport_shim():
    """Load TransportShim without importing gsw_memory.__init__ (pulls
    curator + friends not installed in the research_agent venv). Mirrors
    the surgical file-path loader used by _synthesis._get_transport_shim."""
    import os

    gsw_root = (
        Path(__file__).resolve().parent.parent.parent / "src" / "gsw_memory"
    )
    paths = {
        "gsw_memory.query_then_sleep.transport": gsw_root / "query_then_sleep" / "transport.py",
        "gsw_memory.memory.operator_utils.agentic_gsw.transport_shim":
            gsw_root / "memory" / "operator_utils" / "agentic_gsw" / "transport_shim.py",
    }

    def _load(name: str):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, os.fspath(paths[name]))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        if name.endswith("transport_shim"):
            _load("gsw_memory.query_then_sleep.transport")
        spec.loader.exec_module(mod)
        return mod

    return _load("gsw_memory.memory.operator_utils.agentic_gsw.transport_shim").TransportShim


TransportShim = _load_transport_shim()


class JudgeOutput(BaseModel):
    """LLM verdict over one pred/gold pair."""

    model_config = ConfigDict(extra="forbid")

    step_scores: list[float] = Field(
        ...,
        description=(
            "One score per GOLD sub-question in [0.0, 1.0]. Score 1.0 means "
            "some predicted sub-question fully captures this gold step "
            "(same entities, same relation, same operator). 0.5 means "
            "partial overlap (right entity, wrong relation, or vice versa). "
            "0.0 means no predicted step captures this gold step. Must have "
            "exactly as many entries as there are gold steps, in gold order."
        ),
    )
    overall_match: float = Field(
        ...,
        description=(
            "Single 0.0-1.0 score for the pred decomposition as a whole. "
            "Weight: structural fidelity (right number of steps, right "
            "operator vocabulary) AND semantic coverage (each gold step "
            "reflected). A pred that merges two gold steps into one and "
            "skips one should score ~0.5. A pred that captures all gold "
            "steps in a different order should score ~0.9."
        ),
    )
    reasoning: str = Field(
        ...,
        description="One or two sentences explaining the scores.",
    )


_SYSTEM_PROMPT = (
    "You are a grader for question-decomposition systems. You read the "
    "ORIGINAL multi-hop question, the GOLD decomposition (the reference "
    "list of sub-questions), and a PREDICTED decomposition from an "
    "automated system. You rate how well the prediction captures the "
    "gold.\n\n"
    "Rules:\n"
    "1. Score each GOLD sub-question individually on [0.0, 1.0]. Look for "
    "any PREDICTED sub-question that captures the same reasoning step "
    "(entity + relation + operator). Paraphrasing, pluralization, and "
    "minor wording differences should NOT cost points. Missing the step "
    "or merging it into a wrong operator should cost points.\n"
    "2. Placeholder convention: treat '#1', '#2', '#s1', '#s2' as "
    "equivalent references to earlier steps. Different step numbering "
    "between pred and gold is OK as long as the dependency structure is "
    "captured.\n"
    "3. `overall_match` is a holistic 0.0-1.0 — your one-number answer.\n"
    "4. Be decisive. Do not hedge to 0.5 when a clear 0.0 or 1.0 applies."
)


def _format_pair(question: str, gold: list, pred: list) -> str:
    g_str = "\n".join(f"  {i+1}. {g.get('question', str(g)) if isinstance(g, dict) else str(g)}" for i, g in enumerate(gold))
    if pred:
        p_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(pred))
    else:
        p_str = "  (empty — decomposer returned no sub-questions)"
    return (
        f"ORIGINAL QUESTION:\n  {question}\n\n"
        f"GOLD DECOMPOSITION ({len(gold)} steps):\n{g_str}\n\n"
        f"PREDICTED DECOMPOSITION ({len(pred)} steps):\n{p_str}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--dataset", choices=["musique", "monaco"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev_fraction", type=float, default=0.25)
    p.add_argument("--split", choices=["dev", "train", "all"], default="dev")
    p.add_argument("--judge_model", default="gpt-4.1-mini")
    p.add_argument("--base_url", default="")
    p.add_argument("--per_call_timeout_s", type=float, default=2.0)
    p.add_argument("--output_json", default="")
    return p.parse_args()


def _load(dataset: str, path: str, limit: int | None) -> list[dict]:
    if dataset == "musique":
        return load_musique(path, limit=limit)
    return load_monaco(path, limit=limit)


def main() -> None:
    args = _parse_args()
    examples = _load(args.dataset, args.data, args.limit)
    train, dev = split_train_dev(examples, dev_fraction=args.dev_fraction, seed=args.seed)
    subset = {"train": train, "dev": dev, "all": examples}[args.split]

    fn = load_decomposer(args.module, per_call_timeout_s=args.per_call_timeout_s)
    shim = TransportShim(
        model_name=args.judge_model,
        base_url=args.base_url or None,
        reasoning_effort="medium",
    )

    per_example: list[dict] = []
    sums = {"overall": 0.0, "step": 0.0, "step_n": 0, "cov": 0, "n": 0, "fail": 0}
    by_hop: dict[str, dict] = {}

    for i, ex in enumerate(subset, 1):
        q = ex.get("question", "")
        gold = ex.get("gold_decomposition", []) or []
        hop = ex.get("hop") or str(ex.get("id", "") or "").split("_")[0] or "unknown"
        try:
            pred_subs = fn(q) or []
        except Exception as exc:  # noqa: BLE001
            pred_subs = []
            per_example.append({"question": q, "error": f"decompose_failed: {exc}"})
            sums["fail"] += 1
            continue
        pred_texts = [getattr(s, "question", str(s)) for s in pred_subs]

        if not gold:
            continue  # nothing to grade against

        user = _format_pair(q, gold, pred_texts)
        try:
            verdict = shim.call(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user,
                response_model=JudgeOutput,
                stage="llm_judge_decomp",
                max_tokens=2000,
            )
        except Exception as exc:  # noqa: BLE001
            per_example.append({"question": q, "error": f"judge_failed: {exc}"})
            sums["fail"] += 1
            continue

        step_scores = list(verdict.step_scores)[: len(gold)]
        if len(step_scores) < len(gold):
            step_scores += [0.0] * (len(gold) - len(step_scores))
        step_avg = sum(step_scores) / len(step_scores) if step_scores else 0.0

        sums["n"] += 1
        sums["overall"] += verdict.overall_match
        sums["step"] += sum(step_scores)
        sums["step_n"] += len(step_scores)
        if pred_texts:
            sums["cov"] += 1

        row = by_hop.setdefault(hop, {"n": 0, "overall": 0.0, "step": 0.0, "step_n": 0, "cov": 0})
        row["n"] += 1
        row["overall"] += verdict.overall_match
        row["step"] += sum(step_scores)
        row["step_n"] += len(step_scores)
        if pred_texts:
            row["cov"] += 1

        per_example.append({
            "question": q,
            "hop": hop,
            "gold_len": len(gold),
            "pred_len": len(pred_texts),
            "overall_match": verdict.overall_match,
            "step_avg": step_avg,
            "step_scores": step_scores,
            "reasoning": verdict.reasoning,
        })

        if i % 25 == 0:
            print(f"[judge] {i}/{len(subset)} — running overall_mean={sums['overall']/max(sums['n'],1):.3f}")

    n = max(sums["n"], 1)
    summary = {
        "module": args.module,
        "dataset": args.dataset,
        "split": args.split,
        "n": sums["n"],
        "failures": sums["fail"],
        "coverage": sums["cov"] / n,
        "overall_match_mean": sums["overall"] / n,
        "step_match_mean": (sums["step"] / sums["step_n"]) if sums["step_n"] else 0.0,
    }
    per_hop = {}
    for hop, row in sorted(by_hop.items()):
        hn = max(row["n"], 1)
        per_hop[hop] = {
            "n": row["n"],
            "coverage": row["cov"] / hn,
            "overall_match": row["overall"] / hn,
            "step_match": (row["step"] / row["step_n"]) if row["step_n"] else 0.0,
        }
    summary["by_hop"] = per_hop

    print("\n=== LLM-as-judge summary ===")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"summary": summary, "per_example": per_example}, indent=2)
        )
        print(f"\n[judge] wrote per-example verdicts to {args.output_json}")


if __name__ == "__main__":
    main()
