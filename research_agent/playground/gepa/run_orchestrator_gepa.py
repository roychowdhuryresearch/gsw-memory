"""Run GEPA prompt optimization on the orchestrator decision prompt.

Pilot usage:
    python -m playground.gepa.run_orchestrator_gepa \
        --max-metric-calls 50 \
        --train-subset configs/frames_dev_planupdate32.json \
        --reflection-model openai/gpt-4.1
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gepa
import typer
from dotenv import find_dotenv, load_dotenv

from research_agent.eval.frames_dataset import load_frames
from research_agent.eval.llm_judge import LLMJudge
from research_agent.eval.subset import filter_to_subset, load_subset
from research_agent.retrieval import load_frames_corpus
from research_agent.retrieval.dense import build_retriever

from playground.gepa.orchestrator_gepa_adapter import (
    BASELINE_RULES,
    COMPONENT_NAME,
    CostTracker,
    GSWOrchestratorGEPAAdapter,
)


# Load both the local .env (research_agent/) and the parent gsw-memory .env
# which holds OPENAI_API_KEY. find_dotenv walks up from CWD until it finds one.
load_dotenv(find_dotenv(usecwd=True))
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    train_subset: Path = typer.Option(
        Path("configs/frames_dev_planupdate32.json"),
        help="32-Q subset for the GEPA reflective minibatch trainset.",
    ),
    val_subset: Path = typer.Option(
        Path("configs/frames_dev_planupdate32.json"),
        help="Validation subset for Pareto tracking. Same as trainset by default for the pilot.",
    ),
    model: str = typer.Option(
        "bedrock/openai.gpt-oss-120b-1:0", help="Orchestrator/researcher model id."
    ),
    base_url: str = typer.Option("", help="Optional base_url for the orchestrator model."),
    api_key: str = typer.Option("", help="Optional API key (else OPENAI_API_KEY)."),
    judge_model: str = typer.Option("gpt-4.1", help="LLM-as-judge model."),
    reflection_model: str = typer.Option(
        "openai/gpt-4.1",
        help="LiteLLM model name for the reflection LM that proposes new prompts.",
    ),
    max_metric_calls: int = typer.Option(
        50, help="Total rollout budget. Tiny=50, pilot=150, standard=300."
    ),
    reflection_minibatch_size: int = typer.Option(
        3, help="Examples per reflection step."
    ),
    max_concurrent_questions: int = typer.Option(
        4, help="Per-batch question concurrency."
    ),
    max_turns: int = typer.Option(50, help="Per-question orchestrator turn budget."),
    seed: int = typer.Option(0, help="GEPA seed."),
    output_dir: Path = typer.Option(
        Path("logs/gepa"),
        help="Where the optimization run state + best candidate land.",
    ),
):
    typer.echo("=== GEPA optimization run ===")
    typer.echo(f"trainset:       {train_subset}")
    typer.echo(f"valset:         {val_subset}")
    typer.echo(f"model:          {model}")
    typer.echo(f"judge:          {judge_model}")
    typer.echo(f"reflection_lm:  {reflection_model}")
    typer.echo(f"max_metric:     {max_metric_calls}")
    typer.echo(f"concurrency:    {max_concurrent_questions} Qs")

    # --- Load datasets ----------------------------------------------------
    typer.echo("\nLoading FRAMES dev split…")
    all_qs = load_frames(split="dev")
    train_obj = load_subset(train_subset)
    val_obj = load_subset(val_subset)
    train_qs = filter_to_subset(all_qs, train_obj)
    val_qs = filter_to_subset(all_qs, val_obj)
    typer.echo(f"  trainset n={len(train_qs)}  valset n={len(val_qs)}")

    # --- Build retriever + corpus once (shared across rollouts) ----------
    typer.echo("\nBuilding hybrid retriever (BM25 + dense)…")
    corpus = load_frames_corpus()
    retriever = build_retriever("hybrid", corpus)
    typer.echo(f"  corpus chunks: {len(corpus.chunks)}")

    # --- Judge -----------------------------------------------------------
    judge = LLMJudge(model=judge_model)

    # --- Cost tracker ----------------------------------------------------
    cost = CostTracker()

    # --- Adapter ---------------------------------------------------------
    adapter = GSWOrchestratorGEPAAdapter(
        corpus=corpus,
        retriever=retriever,
        judge=judge,
        model_id=model,
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        max_turns=max_turns,
        max_concurrent_questions=max_concurrent_questions,
        cost_tracker=cost,
    )

    # --- Output dir ------------------------------------------------------
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"orchestrator_gepa__{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "train_subset": str(train_subset),
        "val_subset": str(val_subset),
        "model": model,
        "judge_model": judge_model,
        "reflection_model": reflection_model,
        "max_metric_calls": max_metric_calls,
        "reflection_minibatch_size": reflection_minibatch_size,
        "max_concurrent_questions": max_concurrent_questions,
        "max_turns": max_turns,
        "seed": seed,
        "started_at": run_tag,
    }, indent=2))
    typer.echo(f"\nrun_dir: {run_dir}")

    # Wrap evaluate() to dump cost.json + a one-line console update each batch.
    cost_path = run_dir / "cost.json"
    original_evaluate = adapter.evaluate

    def _evaluate_with_cost_logging(batch, candidate, capture_traces=False):
        result = original_evaluate(batch, candidate, capture_traces=capture_traces)
        cost_path.write_text(json.dumps({
            "n_rollouts": cost.n_rollouts,
            "n_judge_calls": cost.n_judge_calls,
            "task_in_tokens": cost.task_in_tokens,
            "task_out_tokens": cost.task_out_tokens,
            "judge_in_tokens": cost.judge_in_tokens,
            "judge_out_tokens": cost.judge_out_tokens,
            "task_cost_usd": round(cost.task_cost, 4),
            "judge_cost_usd": round(cost.judge_cost, 4),
            "total_cost_usd": round(cost.total_cost, 4),
        }, indent=2))
        n_correct = sum(1 for s in result.scores if s >= 1.0)
        typer.echo(
            f"  [batch {cost.n_rollouts // max(len(batch), 1)}] "
            f"n={len(batch)} ✓={n_correct}/{len(batch)} | {cost.summary()}",
            err=True,
        )
        return result

    adapter.evaluate = _evaluate_with_cost_logging

    # --- Optimize --------------------------------------------------------
    typer.echo("\nLaunching gepa.optimize…")
    result = gepa.optimize(
        seed_candidate={COMPONENT_NAME: BASELINE_RULES},
        trainset=train_qs,
        valset=val_qs,
        adapter=adapter,
        reflection_lm=reflection_model,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        seed=seed,
        run_dir=str(run_dir),
        display_progress_bar=True,
    )

    # --- Final cost dump -------------------------------------------------
    typer.echo(f"\nFinal cost: {cost.summary()}")

    # --- Persist best candidate ------------------------------------------
    best_idx = getattr(result, "best_candidate_idx", None)
    best_candidate = getattr(result, "best_candidate", None)
    typer.echo("\n=== GEPA done ===")
    typer.echo(f"best_idx: {best_idx}")
    if best_candidate:
        best_text = best_candidate.get(COMPONENT_NAME, "")
        (run_dir / "best_orchestrator_rules.txt").write_text(best_text)
        typer.echo(f"best ORCHESTRATOR_RULES → {run_dir / 'best_orchestrator_rules.txt'}")
        typer.echo(f"  length: {len(best_text)} chars (baseline {len(BASELINE_RULES)})")

    # Persist Pareto + history if available
    for attr in ("pareto_candidates", "history", "candidates"):
        v = getattr(result, attr, None)
        if v is None:
            continue
        try:
            (run_dir / f"{attr}.json").write_text(json.dumps(v, default=str, indent=2))
            typer.echo(f"  saved {attr}.json")
        except Exception:
            pass


if __name__ == "__main__":
    app()
