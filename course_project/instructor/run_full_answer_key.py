#!/usr/bin/env python3
"""Run one restartable stage of the instructor Colab answer key locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))

from colab_full_solution import (  # noqa: E402
    RunConfig,
    run_answer_stage,
    run_decomposition_stage,
    run_neural_retrieval_stage,
    run_neural_retrieval_stage_low_memory,
    write_environment,
)
from panini_course import CoursePackage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("decompose", "retrieve", "answer", "all")
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument(
        "--dataset",
        choices=("2wiki", "musique"),
        help="Run one package only; omit to run both packages.",
    )
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument(
        "--colab-low-memory",
        action="store_true",
        help="Never keep the 8B encoder and 8B reranker resident together.",
    )
    parser.add_argument("--work-root", type=Path, default=HERE / "full_run")
    args = parser.parse_args()

    packages = {
        "2wiki": CoursePackage(PROJECT / "release" / "panini_2wiki_100"),
        "musique": CoursePackage(PROJECT / "release" / "panini_musique_100"),
    }
    jobs = []
    for dataset, package in packages.items():
        if args.dataset and dataset != args.dataset:
            continue
        questions = package.questions("public") + package.questions("held_out")
        questions = questions[args.start : args.stop]
        cache_root = args.work_root / "cache" / dataset
        cache_root.mkdir(parents=True, exist_ok=True)
        jobs.append((dataset, package, questions, cache_root))

    config = RunConfig()
    if args.stage in {"decompose", "all"}:
        run_decomposition_stage(jobs, config)
    if args.stage in {"retrieve", "all"}:
        retrieval_runner = (
            run_neural_retrieval_stage_low_memory
            if args.colab_low_memory
            else run_neural_retrieval_stage
        )
        retrieval_runner(
            jobs,
            config,
            run_ablations=not args.skip_ablations,
        )
    if args.stage in {"answer", "all"}:
        output_root = args.work_root / "submission"
        run_answer_stage(
            jobs,
            output_root,
            config,
            run_ablations=not args.skip_ablations,
        )
        write_environment(output_root / "environment.txt", config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
