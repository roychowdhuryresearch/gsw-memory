#!/usr/bin/env python3
"""Write a compact Markdown report from the completed instructor run."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

from colab_full_solution import (  # noqa: E402
    decomposition_metrics,
    read_jsonl,
    result_summary,
)
from panini_course import CoursePackage  # noqa: E402


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_No completed records._"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    work_root = HERE / "full_run"
    artifact_root = HERE / "full_run_outputs"
    artifact_root.mkdir(parents=True, exist_ok=True)
    packages = {
        "2wiki": CoursePackage(PROJECT / "release" / "panini_2wiki_100"),
        "musique": CoursePackage(PROJECT / "release" / "panini_musique_100"),
    }
    decomposition_rows = []
    answer_rows = []
    ablation_rows = []
    file_rows = []
    for dataset, package in packages.items():
        cache_root = work_root / "cache" / dataset
        plans = read_jsonl(cache_root / "decompositions.jsonl")
        predicted = {
            row["question_id"]: row["predicted_decomposition"]
            for row in plans
            if row.get("predicted_decomposition")
        }
        decomposition_rows.append(
            {
                "dataset": dataset,
                "all_plans": len(plans),
                **decomposition_metrics(predicted, package.decompositions()),
                "mean_seconds": (
                    sum(row["seconds"] for row in plans) / len(plans) if plans else 0.0
                ),
            }
        )

        results_path = work_root / "submission" / "results" / f"{dataset}_dev.jsonl"
        results = read_jsonl(results_path)
        question_by_id = {
            row["question_id"]: row for row in package.questions("public")
        }
        group = "type" if dataset == "2wiki" else "hop_count"
        enriched = [
            {**row, group: question_by_id[row["question_id"]][group]}
            for row in results
        ]
        for row in result_summary(enriched, group):
            group_value = row.pop(group)
            answer_rows.append({"dataset": dataset, "group": group_value, **row})

        ablations = read_jsonl(cache_root / "ablation_answers.jsonl")
        grouped: dict[str, list[dict]] = {}
        for row in ablations:
            grouped.setdefault(row["configuration"], []).append(row)
        for name, rows in sorted(grouped.items()):
            ablation_rows.append(
                {
                    "dataset": dataset,
                    "configuration": name,
                    "questions": len(rows),
                    "chain_recovery": sum(
                        row["complete_chain_recovery"] for row in rows
                    )
                    / len(rows),
                    "EM": sum(row["exact_match"] for row in rows) / len(rows),
                    "F1": sum(row["token_f1"] for row in rows) / len(rows),
                    "retrieval_seconds": sum(
                        row["retrieval_seconds"] for row in rows
                    )
                    / len(rows),
                }
            )

        for split, path in (
            ("development", results_path),
            (
                "held_out",
                work_root
                / "submission"
                / "predictions"
                / f"{dataset}_heldout.jsonl",
            ),
        ):
            file_rows.append(
                {"dataset": dataset, "split": split, "records": len(read_jsonl(path))}
            )
            shutil.copy2(path, artifact_root / path.name)

    environment_path = work_root / "submission" / "environment.txt"
    if environment_path.exists():
        shutil.copy2(environment_path, artifact_root / environment_path.name)

    report = f"""# PANINI full answer-key run

The run uses the frozen configuration `B=5`, `k=15`, `M=60`, 4-bit Qwen
models, supplied corpus embeddings, and exact-query caches. Neural model speed
was measured on an RTX A6000 and must not be presented as Colab T4 speed.

## Decomposition

{markdown_table(decomposition_rows)}

## Development results

{markdown_table(answer_rows)}

## Ablations

{markdown_table(ablation_rows)}

## Required files

{markdown_table(file_rows)}

The runnable notebook is `Panini_Full_Answer_Key_Colab.ipynb`. The populated
notebook is `Panini_Full_Answer_Key_Executed.ipynb`.
"""
    (HERE / "FULL_RUN_RESULTS.md").write_text(report, encoding="utf-8")
    print(HERE / "FULL_RUN_RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
