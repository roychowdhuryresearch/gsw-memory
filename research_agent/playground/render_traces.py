"""Regenerate q_*.md readable trace files for a cell that was run before
the markdown renderer existed.

Usage
-----
.. code-block:: bash

    # Regenerate markdown sidecars for one specific cell:
    python playground/render_traces.py logs/vanilla_rag_react__gpt-5__20260417_225858

    # Or for every cell under logs/:
    python playground/render_traces.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap sys.path for CLI invocation.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import typer

from research_agent.eval.harness import load_cell
from research_agent.utils import render_question_markdown

app = typer.Typer(add_completion=False, help="Re-render q_*.md from saved cell_result.json.")


def _render_cell(cell_dir: Path) -> int:
    cell_path = cell_dir / "cell_result.json"
    if not cell_path.exists():
        typer.echo(f"  skip {cell_dir} (no cell_result.json)")
        return 0
    try:
        cell = load_cell(cell_path)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  failed to load {cell_path}: {exc}")
        return 0
    trace_dir = cell_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for qr in cell.questions:
        try:
            (trace_dir / f"q_{qr.question_id}.md").write_text(
                render_question_markdown(qr)
            )
            n += 1
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  failed to render q{qr.question_id}: {exc}")
    typer.echo(f"  rendered {n} md files in {trace_dir}")
    return n


@app.command()
def main(
    cell_dir: Path = typer.Argument(
        None,
        help="Path to a specific cell dir (e.g. logs/<system>__<model>__<ts>/). "
        "Omit when using --all.",
    ),
    all_cells: bool = typer.Option(
        False, "--all", help="Render every cell dir under logs/."
    ),
    logs_root: Path = typer.Option(
        Path("logs"), "--logs-root", help="Root dir scanned by --all."
    ),
) -> None:
    if all_cells:
        n_total = 0
        for cd in sorted(logs_root.iterdir()):
            if cd.is_dir():
                typer.echo(f"cell: {cd.name}")
                n_total += _render_cell(cd)
        typer.echo(f"\nTotal: rendered {n_total} markdown files across {logs_root}/")
        return

    if cell_dir is None:
        typer.echo("Pass a cell dir or --all. See --help.")
        raise typer.Exit(code=1)
    _render_cell(cell_dir)


if __name__ == "__main__":
    app()
