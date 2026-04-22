"""CLI: select a FRAMES pilot subset and persist it to configs/pilot_subset.json.

.. code-block:: bash

    python playground/select_pilot_subset.py \\
        --split dev \\
        --out configs/pilot_subset.json \\
        --subset-id frames_pilot_v1
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Make `research_agent` importable without requiring `PYTHONPATH=src` on the CLI.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import typer

from research_agent.eval import load_frames, select_stratified_subset

app = typer.Typer(add_completion=False, help="Select a FRAMES pilot subset.")


@app.command()
def main(
    split: str = typer.Option("dev", help="'dev' (100 Qs) or 'full' (824)."),
    out: Path = typer.Option(
        Path("configs/pilot_subset.json"),
        "--out",
        help="Output path for the subset JSON.",
    ),
    subset_id: str = typer.Option("frames_pilot_v1", "--subset-id"),
    seed: int = typer.Option(0, "--seed"),
    n_easy: int = typer.Option(10, help="Count for 2-hop bucket."),
    n_medium: int = typer.Option(10, help="Count for 3-hop bucket."),
    n_hard: int = typer.Option(8, help="Count for 4-5-hop bucket."),
    n_long: int = typer.Option(2, help="Count for 6+-hop bucket."),
) -> None:
    questions = load_frames(split=split)
    typer.echo(f"Loaded {len(questions)} FRAMES questions from split={split}")

    targets = {
        "2_hop": n_easy,
        "3_hop": n_medium,
        "4_5_hop": n_hard,
        "6plus_hop": n_long,
    }
    subset = select_stratified_subset(
        questions,
        subset_id=subset_id,
        benchmark="frames",
        split=split,
        targets=targets,
        seed=seed,
    )
    subset.save(out)
    typer.echo(f"Wrote {len(subset.question_ids)} Q ids to {out}")
    typer.echo("Stratification targets: " + json.dumps(targets))
    typer.echo("Stratification actual:  " + json.dumps(subset.stratification))

    id_set = set(subset.question_ids)
    picked = [q for q in questions if q.id in id_set]
    rt = Counter()
    for q in picked:
        for t in q.reasoning_types:
            rt[t] += 1
    typer.echo("Reasoning-type coverage in subset:")
    for k, v in rt.most_common():
        typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
