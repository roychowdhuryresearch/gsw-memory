"""Aggregate cell-result JSONs into the substitution grid + audit report.

Scans ``logs/<system>__<model>__<ts>/cell_result.json`` files, builds
a tidy pandas DataFrame with headline metrics + failure-mode counts per
cell, and writes a human-readable markdown audit.

Usage
-----
.. code-block:: bash

    python playground/aggregate_grid.py \\
        --logs-dir logs \\
        --out logs/grid_summary.md \\
        --csv logs/grid_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `research_agent` importable without requiring `PYTHONPATH=src` on the CLI.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import typer

from research_agent.eval.harness import load_cell
from research_agent.models.failure_modes import FailureMode

app = typer.Typer(add_completion=False)


def _collect_cells(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.glob("*/cell_result.json"))


def _build_dataframe(cell_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in cell_paths:
        cell = load_cell(p)
        row = cell.summary_row()
        row["cell_path"] = str(p.parent)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        for fm in FailureMode:
            col = f"fm_{fm.value}"
            if col not in df.columns:
                df[col] = 0
    return df


def _render_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Substitution grid\n\n_No cell results found._\n"

    headline = df[["system", "model", "n", "accuracy", "mean_f1", "mean_turns", "mean_tools", "mean_wall_s"]]
    fm_cols = [c for c in df.columns if c.startswith("fm_")]
    fm_table = df[["system", "model", *fm_cols]]

    parts: list[str] = []
    parts.append("# Substitution grid — pilot results\n")
    parts.append("## Headline metrics\n")
    parts.append(headline.to_markdown(index=False))
    parts.append("\n\n## Failure-mode histogram per cell\n")
    parts.append(fm_table.to_markdown(index=False))

    # Quick observations
    parts.append("\n\n## Quick observations\n")
    if "accuracy" in df.columns and len(df) > 1:
        best = df.loc[df["accuracy"].idxmax()]
        worst = df.loc[df["accuracy"].idxmin()]
        parts.append(
            f"- Best cell: **{best['system']} + {best['model']}** "
            f"at accuracy={best['accuracy']:.3f} (f1={best['mean_f1']:.3f})"
        )
        parts.append(
            f"- Worst cell: **{worst['system']} + {worst['model']}** "
            f"at accuracy={worst['accuracy']:.3f}"
        )
        parts.append(
            f"- Accuracy spread across the grid: "
            f"{df['accuracy'].max() - df['accuracy'].min():.3f}"
        )
    parts.append("")
    return "\n".join(parts)


@app.command()
def main(
    logs_dir: Path = typer.Option(Path("logs"), "--logs-dir"),
    out: Path = typer.Option(Path("logs/grid_summary.md"), "--out"),
    csv: Path | None = typer.Option(None, "--csv", help="Also write a CSV."),
) -> None:
    cells = _collect_cells(logs_dir)
    typer.echo(f"Found {len(cells)} cell_result.json files under {logs_dir}")
    if not cells:
        typer.echo("No cells to aggregate.")
        raise typer.Exit(code=1)

    df = _build_dataframe(cells)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_markdown(df))
    typer.echo(f"Wrote markdown audit to {out}")

    if csv is not None:
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        typer.echo(f"Wrote CSV to {csv}")

    # Also dump a JSON index for programmatic access.
    index_path = out.with_suffix(".index.json")
    index_path.write_text(
        json.dumps(
            {
                "cells": [
                    {"path": r["cell_path"], "system": r["system"], "model": r["model"], "accuracy": r["accuracy"]}
                    for r in df.to_dict(orient="records")
                ]
            },
            indent=2,
        )
    )
    typer.echo(f"Wrote index to {index_path}")


if __name__ == "__main__":
    app()
