#!/usr/bin/env python3
"""
Bridge Test Runner: Run sleep-time agent on MuSiQue questions.

For each question:
1. Creates a temp directory with symlinks to that question's 20 GSW doc dirs
2. Runs run_sleep_time.py — agent discovers entities itself using discovery tools
3. Collects bridge QAs and tags them with question metadata

Global index mapping: global_doc_idx = question_idx * 20 + paragraph_local_idx

Usage:
    python playground/sleep_time/run_bridge_test.py \
        --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \
        --start 0 --end 10 \
        --model Qwen/Qwen3-235B-A22B-Thinking-2507 \
        --output_dir logs/bridge_test \
        --show-thinking
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
from rich.table import Table
from rich import box

console = Console()

DOCS_PER_QUESTION = 20


def load_musique_questions(path: str, start: int = 0, end: int = None):
    """Load musique questions with start/end range. Returns (original_idx, question) tuples."""
    with open(path) as f:
        data = json.load(f)
    data = data[start:end]
    return [(start + i, d) for i, d in enumerate(data) if d.get("answerable", True)]


def get_global_doc_indices(q_idx: int) -> list:
    """Get global doc indices for a question. Each question has 20 paragraphs."""
    return [q_idx * DOCS_PER_QUESTION + i for i in range(DOCS_PER_QUESTION)]


def get_supporting_global_indices(question: dict, q_idx: int) -> list:
    """Get global doc indices for supporting paragraphs."""
    local_support = [p["idx"] for p in question["paragraphs"] if p["is_supporting"]]
    return [q_idx * DOCS_PER_QUESTION + local for local in local_support]


def setup_temp_gsw_dir(gsw_path: str, global_indices: list, temp_base: str) -> str:
    """Create temp directory with symlinks to the question's GSW doc dirs."""
    temp_dir = os.path.join(temp_base, "networks")
    os.makedirs(temp_dir, exist_ok=True)

    for idx in global_indices:
        source = os.path.join(gsw_path, f"doc_{idx}")
        target = os.path.join(temp_dir, f"doc_{idx}")
        if os.path.exists(source):
            os.symlink(os.path.abspath(source), target)
        else:
            console.print(f"  [yellow]Warning: {source} not found[/yellow]")

    return temp_dir


def run_question(q_idx: int, question: dict, args) -> dict:
    """Run sleep-time exploration for a single question."""
    global_indices = get_global_doc_indices(q_idx)
    support_global = get_supporting_global_indices(question, q_idx)

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold]Question {q_idx}: {question['question']}[/bold]")
    console.print(f"  Answer: {question['answer']}")
    console.print(f"  Supporting docs (global): {support_global}")
    console.print(f"  Hops: {len(question.get('question_decomposition', []))}")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]")

    # Create temp directory for this question's docs
    temp_base = tempfile.mkdtemp(prefix=f"bridge_test_q{q_idx}_")

    try:
        # Setup symlinks
        temp_gsw_dir = setup_temp_gsw_dir(args.gsw_path, global_indices, temp_base)

        # Output dir for this question
        q_output_dir = os.path.join(args.output_dir, f"q_{q_idx}")

        # Build run_sleep_time.py command — agent discovers entities itself
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "run_sleep_time.py"),
            "--gsw_path", temp_gsw_dir,
            "--num_docs", str(DOCS_PER_QUESTION),
            "--model", args.model,
            "--max_iterations", str(args.max_iterations),
            "--output_dir", q_output_dir,
            "--max_tokens", str(args.max_tokens),
            "--reasoning_effort", args.reasoning_effort,
        ]

        if args.base_url:
            cmd.extend(["--base_url", args.base_url])

        if args.verbose or args.show_thinking:
            cmd.append("--verbose")

        if args.show_thinking:
            cmd.append("--show-thinking")

        # Set cache dir inside temp to avoid conflicts
        cache_dir = os.path.join(temp_base, ".gsw_cache")
        cmd.extend(["--cache_dir", cache_dir])

        console.print(f"  Running: {' '.join(cmd[-6:])}")

        # Run the command (cwd=repo root so playground.* imports work)
        repo_root = str(Path(__file__).resolve().parents[2])
        result = subprocess.run(cmd, capture_output=not (args.verbose or args.show_thinking), text=True, cwd=repo_root)

        if result.returncode != 0 and not args.verbose:
            console.print(f"  [red]Error (exit {result.returncode}):[/red]")
            console.print(f"  {result.stderr[-500:] if result.stderr else 'No stderr'}")

        # Find and load results
        # run_sleep_time.py creates output_dir/run_YYYYMMDD_HHMMSS/results.json
        results_json = None
        q_output_path = Path(q_output_dir)
        if q_output_path.exists():
            run_dirs = sorted(q_output_path.glob("run_*"))
            if run_dirs:
                results_file = run_dirs[-1] / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results_json = json.load(f)

        # Build per-question result
        bridges = results_json.get("bridges", []) if results_json else []
        console.print(f"  [green]Generated {len(bridges)} bridges[/green]")

        return {
            "question_idx": q_idx,
            "question_id": question.get("id", ""),
            "question": question["question"],
            "answer": question["answer"],
            "answer_aliases": question.get("answer_aliases", []),
            "num_hops": len(question.get("question_decomposition", [])),
            "decomposition": question.get("question_decomposition", []),
            "supporting_doc_indices_local": [p["idx"] for p in question["paragraphs"] if p["is_supporting"]],
            "supporting_doc_indices_global": support_global,
            "supporting_doc_titles": [p["title"] for p in question["paragraphs"] if p["is_supporting"]],
            "all_doc_indices_global": global_indices,
            "bridges": bridges,
            "num_bridges": len(bridges),
            "run_results": results_json.get("summary", {}) if results_json else {},
        }

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_base, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run bridge test on first N MuSiQue questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--questions", type=str, default="playground_data/musique.json",
                        help="Path to MuSiQue questions file")
    parser.add_argument("--gsw_path", type=str, required=True,
                        help="Path to GSW networks directory (e.g. networks_4_1_mini)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start question index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="End question index (exclusive). Defaults to all.")

    # Exploration
    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="Max iterations per entity")
    parser.add_argument("--max_tokens", type=int, default=500_000,
                        help="Max token budget per question")

    # Model
    parser.add_argument("--model", type=str, default="Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for vLLM/OpenAI-compatible server")
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                        choices=["low", "medium", "high"])

    # Output
    parser.add_argument("--output_dir", type=str, default="logs/bridge_test",
                        help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show run_sleep_time.py output")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Show agent reasoning (implies --verbose)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    console.print("[cyan]Loading questions...[/cyan]")
    questions = load_musique_questions(args.questions, args.start, args.end)
    console.print(f"  Loaded {len(questions)} questions (range: {args.start}:{args.end})")

    # Run each question
    all_results = []
    for q_idx, question in questions:
        try:
            result = run_question(q_idx, question, args)
            all_results.append(result)
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            all_results.append({
                "question_idx": q_idx,
                "question": question["question"],
                "error": str(e),
            })

    # Save combined results
    output_file = os.path.join(args.output_dir, "bridge_test_results.json")
    combined = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "start": args.start,
            "end": args.end,
            "model": args.model,
            "max_iterations": args.max_iterations,
        },
        "results": all_results,
    }
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")

    # Quick summary
    total_bridges = sum(r.get("num_bridges", 0) for r in all_results)
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Questions tested: {len(all_results)}")
    console.print(f"  Total bridges: {total_bridges}")
    console.print(f"  Avg bridges/question: {total_bridges / len(all_results):.1f}")


if __name__ == "__main__":
    main()
