#!/usr/bin/env python3
"""
Bridge Test Runner: run sleep-time agent on bridge-style QA datasets.

For each question:
1. Creates a temp directory with symlinks to that question's GSW doc dirs
2. Runs run_sleep_time.py — agent discovers entities itself using discovery tools
3. Collects bridge QAs and tags them with question metadata

Supported schemas:
- MuSiQue-style: `paragraphs` list with `idx` and `is_supporting`
- 2Wiki/Hotpot-style: `context` + `supporting_facts` with title-based doc mapping

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

DEFAULT_DOCS_PER_QUESTION = 20


def load_questions(path: str, start: int = 0, end: int = None):
    """
    Load questions with start/end range.

    Returns:
        Tuple[List[(original_idx, question)], schema]
        schema is one of: "musique", "context_titles"
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected question file to be a JSON list, got {type(data).__name__}")

    data = data[start:end]
    questions = [(start + i, d) for i, d in enumerate(data) if d.get("answerable", True)]
    if not questions:
        return [], "unknown"

    sample = questions[0][1]
    if "paragraphs" in sample:
        schema = "musique"
    elif "context" in sample and "supporting_facts" in sample:
        schema = "context_titles"
    else:
        schema = "unknown"

    return questions, schema


def infer_corpus_path(questions_path: str) -> str | None:
    """Infer corpus path from question file name if possible."""
    qpath = Path(questions_path)
    if qpath.suffix != ".json":
        return None
    candidate = qpath.with_name(f"{qpath.stem}_corpus.json")
    return str(candidate) if candidate.exists() else None


def build_title_to_doc_index(corpus_path: str) -> dict[str, int]:
    """Build title -> global doc index map from corpus JSON list."""
    with open(corpus_path) as f:
        corpus = json.load(f)
    if not isinstance(corpus, list):
        raise ValueError(f"Expected corpus file to be a JSON list, got {type(corpus).__name__}")

    title_to_idx: dict[str, int] = {}
    for idx, item in enumerate(corpus):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if title:
            title_to_idx[title] = idx
    return title_to_idx


def normalize_question_metadata(
    question: dict,
    q_idx: int,
    schema: str,
    title_to_idx: dict[str, int] | None = None,
):
    """Normalize dataset-specific question fields into a common metadata structure."""
    if schema == "musique":
        paragraphs = question.get("paragraphs", [])
        if not paragraphs:
            raise ValueError("MuSiQue question is missing non-empty 'paragraphs'")

        docs_per_question = len(paragraphs) if paragraphs else DEFAULT_DOCS_PER_QUESTION
        global_indices = [q_idx * docs_per_question + i for i in range(docs_per_question)]
        supporting_local = [p["idx"] for p in paragraphs if p.get("is_supporting")]
        supporting_global = [q_idx * docs_per_question + local for local in supporting_local]
        supporting_titles = [p.get("title", "") for p in paragraphs if p.get("is_supporting")]
        decomposition = question.get("question_decomposition", [])

        return {
            "question_id": question.get("id", ""),
            "question_text": question.get("question", ""),
            "answer": question.get("answer", ""),
            "answer_aliases": question.get("answer_aliases", []),
            "num_hops": len(decomposition),
            "decomposition": decomposition,
            "global_indices": global_indices,
            "supporting_local": supporting_local,
            "supporting_global": supporting_global,
            "supporting_titles": supporting_titles,
        }

    if schema == "context_titles":
        if title_to_idx is None:
            raise ValueError("Title index is required for context_titles schema")

        context = question.get("context", [])
        supporting_facts = question.get("supporting_facts", [])
        if not context:
            raise ValueError("Question is missing non-empty 'context'")

        supporting_title_set = {fact[0] for fact in supporting_facts if isinstance(fact, list) and fact}

        global_indices = []
        missing_titles = []
        supporting_local = []
        supporting_global = []
        supporting_titles = []

        for local_idx, entry in enumerate(context):
            if isinstance(entry, list) and entry:
                title = entry[0]
            else:
                title = None

            if not title:
                continue

            global_idx = title_to_idx.get(title)
            if global_idx is None:
                missing_titles.append(title)
                continue

            global_indices.append(global_idx)
            if title in supporting_title_set:
                supporting_local.append(local_idx)
                supporting_global.append(global_idx)
                supporting_titles.append(title)

        if missing_titles:
            console.print(
                f"  [yellow]Warning: {len(missing_titles)} titles not found in corpus index for q={q_idx}[/yellow]"
            )

        if not global_indices:
            raise ValueError("No mappable context titles found in corpus for this question")

        decomposition = question.get("question_decomposition", question.get("evidences", []))
        answer_aliases = question.get("answer_aliases", [])
        if not answer_aliases and question.get("answer"):
            answer_aliases = [question["answer"]]

        return {
            "question_id": question.get("_id", question.get("id", "")),
            "question_text": question.get("question", ""),
            "answer": question.get("answer", ""),
            "answer_aliases": answer_aliases,
            "num_hops": len(set(supporting_titles)) if supporting_titles else len(supporting_facts),
            "decomposition": decomposition,
            "global_indices": global_indices,
            "supporting_local": supporting_local,
            "supporting_global": supporting_global,
            "supporting_titles": supporting_titles,
        }

    raise ValueError(
        "Unsupported question schema. Expected MuSiQue-style ('paragraphs') "
        "or context-title schema ('context' + 'supporting_facts')."
    )


def setup_temp_gsw_dir(gsw_path: str, global_indices: list, temp_base: str) -> str:
    """Create temp directory with symlinks to the question's GSW doc dirs."""
    temp_dir = os.path.join(temp_base, "networks")
    os.makedirs(temp_dir, exist_ok=True)

    for idx in global_indices:
        source = os.path.join(gsw_path, f"doc_{idx}")
        target = os.path.join(temp_dir, f"doc_{idx}")
        if os.path.exists(target):
            continue
        if os.path.exists(source):
            os.symlink(os.path.abspath(source), target)
        else:
            console.print(f"  [yellow]Warning: {source} not found[/yellow]")

    return temp_dir


def run_question(
    q_idx: int,
    question: dict,
    args,
    schema: str,
    title_to_idx: dict[str, int] | None = None,
) -> dict:
    """Run sleep-time exploration for a single question."""
    meta = normalize_question_metadata(question, q_idx, schema=schema, title_to_idx=title_to_idx)
    global_indices = meta["global_indices"]
    support_global = meta["supporting_global"]
    num_docs = len(global_indices)

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold]Question {q_idx}: {meta['question_text']}[/bold]")
    console.print(f"  Answer: {meta['answer']}")
    console.print(f"  Supporting docs (global): {support_global}")
    console.print(f"  Hops: {meta['num_hops']}")
    console.print(f"  Context docs used: {num_docs}")
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
            "--num_docs", str(num_docs),
            "--model", args.model,
            "--max_iterations", str(args.max_iterations),
            "--output_dir", q_output_dir,
            "--max_tokens", str(args.max_tokens),
            "--reasoning_effort", args.reasoning_effort,
            "--pipeline_mode", args.pipeline_mode,
            "--hybrid_scope", args.hybrid_scope,
        ]

        if args.base_url:
            cmd.extend(["--base_url", args.base_url])

        if args.root_model:
            cmd.extend(["--root_model", args.root_model])

        if args.worker_model:
            cmd.extend(["--worker_model", args.worker_model])

        cmd.extend(
            [
                "--edge_max_depth", str(args.edge_max_depth),
                "--edge_max_calls", str(args.edge_max_calls),
                "--edge_max_tokens", str(args.edge_max_tokens),
                "--max_optional_docs_per_edge", str(args.max_optional_docs_per_edge),
            ]
        )

        if args.verbose or args.show_thinking:
            cmd.append("--verbose")

        if args.show_thinking:
            cmd.append("--show-thinking")

        # Set cache dir inside temp to avoid conflicts
        cache_dir = os.path.join(temp_base, ".gsw_cache")
        cmd.extend(["--cache_dir", cache_dir])

        console.print(
            "  Pipeline knobs: "
            f"depth={args.edge_max_depth}, calls={args.edge_max_calls}, "
            f"edge_tokens={args.edge_max_tokens}, optional_docs={args.max_optional_docs_per_edge}"
        )
        console.print(f"  Running command: {' '.join(cmd)}")

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
            "question_id": meta["question_id"],
            "question": meta["question_text"],
            "answer": meta["answer"],
            "answer_aliases": meta["answer_aliases"],
            "num_hops": meta["num_hops"],
            "decomposition": meta["decomposition"],
            "supporting_doc_indices_local": meta["supporting_local"],
            "supporting_doc_indices_global": meta["supporting_global"],
            "supporting_doc_titles": meta["supporting_titles"],
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
                        help="Path to questions JSON (MuSiQue or 2Wiki/Hotpot-style)")
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="Optional corpus JSON path for context-title datasets (title -> global doc mapping)")
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
    parser.add_argument("--pipeline_mode", type=str, default="legacy",
                        choices=["legacy", "rlm", "hybrid"],
                        help="Exploration pipeline for run_sleep_time.py")
    parser.add_argument("--hybrid_scope", type=str, default="doc_edge",
                        choices=["edge", "doc_edge", "corpus_doc_edge"],
                        help="Hybrid autonomy scope when pipeline_mode=hybrid")
    parser.add_argument("--root_model", type=str, default=None,
                        help="Optional root-stage model in RLM mode")
    parser.add_argument("--worker_model", type=str, default=None,
                        help="Optional worker-stage model in RLM mode")
    parser.add_argument("--edge_max_depth", type=int, default=1,
                        help="Maximum recursion depth per edge in RLM mode")
    parser.add_argument("--edge_max_calls", type=int, default=2,
                        help="Maximum worker calls per edge in RLM mode")
    parser.add_argument("--edge_max_tokens", type=int, default=3000,
                        help="Per-edge token budget in RLM mode")
    parser.add_argument("--max_optional_docs_per_edge", type=int, default=2,
                        help="Maximum optional docs per edge packet in RLM mode")

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
    questions, schema = load_questions(args.questions, args.start, args.end)
    console.print(f"  Loaded {len(questions)} questions (range: {args.start}:{args.end})")
    console.print(f"  Detected schema: {schema}")

    if schema == "unknown" and questions:
        raise ValueError(
            "Unsupported question schema. Expected either MuSiQue ('paragraphs') "
            "or context-title format ('context' + 'supporting_facts')."
        )

    title_to_idx = None
    if schema == "context_titles":
        corpus_path = args.corpus_path or infer_corpus_path(args.questions)
        if not corpus_path:
            raise ValueError(
                "Could not infer corpus path for context-title dataset. "
                "Pass --corpus_path explicitly."
            )
        console.print(f"  Loading corpus index from: {corpus_path}")
        title_to_idx = build_title_to_doc_index(corpus_path)
        console.print(f"  Indexed {len(title_to_idx)} titles")

    # Run each question
    all_results = []
    for q_idx, question in questions:
        try:
            result = run_question(q_idx, question, args, schema=schema, title_to_idx=title_to_idx)
            all_results.append(result)
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            all_results.append({
                "question_idx": q_idx,
                "question": question.get("question", ""),
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
            "schema": schema,
            "model": args.model,
            "max_iterations": args.max_iterations,
            "pipeline_mode": args.pipeline_mode,
            "hybrid_scope": args.hybrid_scope,
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
    avg = (total_bridges / len(all_results)) if all_results else 0.0
    console.print(f"  Avg bridges/question: {avg:.1f}")


if __name__ == "__main__":
    main()
