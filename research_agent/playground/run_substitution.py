"""Runner CLI — drive one adapter against the FRAMES pilot subset.

Usage
-----
.. code-block:: bash

    # Frontier row: GPT-5 (OpenAI, needs OPENAI_API_KEY)
    python playground/run_substitution.py \\
        --system vanilla_rag_react \\
        --model gpt-5 \\
        --subset configs/pilot_subset.json

    # Small-model swap over vLLM (needs the serve_vllm_*.sh script up)
    python playground/run_substitution.py \\
        --system vanilla_rag_react \\
        --model openai/gpt-oss-20b \\
        --base-url http://127.0.0.1:8001/v1 \\
        --api-key dummy \\
        --subset configs/pilot_subset.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `research_agent` importable without requiring `PYTHONPATH=src` on the CLI.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import typer

# Import-time side effect: loading adapters registers them.
import research_agent.adapters.agentic_reasoning_mindmap  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.asearcher  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.context1  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.eigent_search_q_plus  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.gam  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.graph_r1  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.ours_gsw_v1  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.rule_decomp_gsw  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.search_o1  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.search_r1  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.smtl  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.tongyi_deep_research  # noqa: F401  pylint: disable=unused-import
import research_agent.adapters.vanilla_rag_react  # noqa: F401  pylint: disable=unused-import
from research_agent.adapters.base import AdapterContext, get_adapter
from research_agent.eval import load_frames, load_subset
from research_agent.eval.harness import persist_cell, run_cell
from research_agent.eval.llm_judge import LLMJudge
from research_agent.eval.subset import filter_to_subset
from research_agent.utils import setup_run_logger

try:
    from dotenv import load_dotenv

    # Load research_agent/.env first (wins on overlapping keys).
    load_dotenv()
    # Then fall back to the parent gsw-memory/.env for shared secrets like
    # OPENAI_API_KEY so each sub-workspace doesn't need its own copy.
    _PARENT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
    if _PARENT_ENV.exists():
        load_dotenv(_PARENT_ENV, override=False)
except ImportError:  # pragma: no cover
    pass

app = typer.Typer(add_completion=False)


@app.command()
def main(
    system: str = typer.Option(..., "--system", "-s", help="Adapter id (e.g. vanilla_rag_react)."),
    model: str = typer.Option(..., "--model", "-m", help="Model name sent to the endpoint."),
    subset: Path = typer.Option(
        Path("configs/pilot_subset.json"), "--subset", help="Path to a persisted subset JSON."
    ),
    split: str = typer.Option("dev", help="FRAMES split to load: dev|full."),
    base_url: str = typer.Option("", help="OpenAI-compatible base_url (e.g. vLLM's /v1)."),
    api_key: str = typer.Option("", help="API key (env OPENAI_API_KEY used if empty)."),
    max_turns: int = typer.Option(16, help="Hard turn cap per question."),
    max_completion_tokens: int = typer.Option(
        50000,
        help=(
            "Per-call completion cap. Large default because GPT-5 / o-series "
            "reasoning models consume hidden reasoning tokens inside this budget."
        ),
    ),
    output_dir: Path = typer.Option(
        Path("logs"),
        help="Directory to dump per-cell CellResult JSON and raw trajectories.",
    ),
    limit: int = typer.Option(0, help="If >0, only run the first N questions (smoke)."),
    dry_run: bool = typer.Option(False, help="Print resolved plan without calling the model."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include per-turn tool calls in the terminal log (still always in run.log)."),
    llm_judge: bool = typer.Option(
        True,
        "--llm-judge/--no-llm-judge",
        help="Grade each answer with an LLM judge (default: gpt-4o via OpenAI, "
        "needs OPENAI_API_KEY). Writes per-Q verdicts into the trace and surfaces "
        "a judge_accuracy alongside exact-match accuracy in the cell summary.",
    ),
    judge_model: str = typer.Option("gpt-4o", help="Model id for the LLM judge."),
    judge_base_url: str = typer.Option(
        "", help="Optional OpenAI-compatible base_url for the judge (for vLLM etc.)."
    ),
    reasoner_model: str = typer.Option(
        "", help="Second-stage reasoner model (Context-1 + reasoner adapter). Empty = reuse --model."
    ),
    reasoner_base_url: str = typer.Option(
        "", help="base_url for the reasoner endpoint (empty = reuse --base-url, or OpenAI for gpt-*)."
    ),
    reasoner_api_key: str = typer.Option(
        "", help="API key for the reasoner endpoint (empty = reuse --api-key / OPENAI_API_KEY)."
    ),
) -> None:
    subset_obj = load_subset(subset)
    questions_all = load_frames(split=split)
    questions = filter_to_subset(questions_all, subset_obj)
    if limit > 0:
        questions = questions[:limit]

    typer.echo(f"system={system} model={model} base_url={base_url or '(default openai)'}")
    typer.echo(f"subset={subset_obj.subset_id} split={split} n_questions={len(questions)}")
    typer.echo(f"max_turns={max_turns} max_completion_tokens={max_completion_tokens}")

    if dry_run:
        for q in questions[:5]:
            typer.echo(f"  - id={q.id} hops={q.num_hops} q={q.question[:80]}...")
        return

    adapter_cls = get_adapter(system)
    extra: dict[str, object] = {}
    if reasoner_model:
        extra["reasoner"] = {
            "model_name": reasoner_model,
            "base_url": reasoner_base_url,
            "api_key": reasoner_api_key or os.environ.get("OPENAI_API_KEY", ""),
        }

    ctx = AdapterContext(
        system_id=system,
        model_id=model,
        model_name=model,
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        max_turns=max_turns,
        max_completion_tokens=max_completion_tokens,
        extra=extra,
    )
    adapter = adapter_cls(ctx)

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cell_dir = output_dir / f"{system}__{model.replace('/', '_')}__{run_tag}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = cell_dir / "traces"

    # Tee all research_agent logs to both the terminal (stderr) and run.log.
    # Harness emits q-start / q-done + (at DEBUG) per-turn tool calls.
    setup_run_logger(cell_dir, verbose=verbose)

    # Optional LLM-as-judge. Needs OPENAI_API_KEY (loaded from .env above).
    judge: LLMJudge | None = None
    if llm_judge:
        try:
            judge = LLMJudge(
                model=judge_model,
                base_url=judge_base_url or None,
            )
            typer.echo(f"LLM judge: {judge_model}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"LLM judge disabled ({exc}).")
            judge = None

    cell = run_cell(
        adapter,
        questions,
        subset_id=subset_obj.subset_id,
        benchmark=subset_obj.benchmark,
        raw_trace_dir=trace_dir,
        judge=judge,
    )

    persist_cell(cell, cell_dir / "cell_result.json")
    typer.echo()
    typer.echo(f"=== Cell summary (saved to {cell_dir / 'cell_result.json'}) ===")
    typer.echo(json.dumps(cell.summary_row(), indent=2))


if __name__ == "__main__":
    sys.exit(app() or 0)
