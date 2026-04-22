#!/usr/bin/env python3
"""
Panini Indexing Benchmark
=========================
Measures EntitySearcher indexing time per phase as num_documents increases.

Usage:
    python playground/sleep_time/benchmark_indexing.py \
        --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \
        --doc_counts 10,20,50,100,200,500,1000 \
        --output_dir logs/benchmark_indexing
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Repo path setup (mirror what simple_entity_search.py does)
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from playground.simple_entity_search import EntitySearcher, load_gsw_files

console = Console()

# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------

_phase_timings: Dict[str, float] = {}


def _timed_wrapper(original_fn, phase_name):
    """Return a wrapper that records wall-clock time for *original_fn*."""

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _phase_timings[phase_name] = elapsed
        return result

    return wrapper


def _instrument_searcher_class():
    """Monkey-patch EntitySearcher methods with timing wrappers.

    Returns the original methods so they can be restored afterward.
    """
    targets = {
        "load_gsw_files": (load_gsw_files, "playground.simple_entity_search"),
        "_extract_entities_from_gsw": None,
        "_build_bm25_index": None,
        "_generate_embeddings": None,
        "_precompute_qa_embeddings": None,
        "_build_qa_embeddings_matrix": None,
    }

    originals = {}

    # Wrap module-level load_gsw_files
    import playground.simple_entity_search as ses_module

    originals["load_gsw_files"] = ses_module.load_gsw_files
    ses_module.load_gsw_files = _timed_wrapper(ses_module.load_gsw_files, "load_gsw_files")

    # Wrap instance methods on the class
    for method_name in [
        "_extract_entities_from_gsw",
        "_build_bm25_index",
        "_generate_embeddings",
        "_precompute_qa_embeddings",
        "_build_qa_embeddings_matrix",
    ]:
        originals[method_name] = getattr(EntitySearcher, method_name)
        setattr(
            EntitySearcher,
            method_name,
            _timed_wrapper(originals[method_name], method_name),
        )

    return originals


def _restore_searcher_class(originals):
    """Undo the monkey-patching."""
    import playground.simple_entity_search as ses_module

    ses_module.load_gsw_files = originals["load_gsw_files"]
    for method_name in [
        "_extract_entities_from_gsw",
        "_build_bm25_index",
        "_generate_embeddings",
        "_precompute_qa_embeddings",
        "_build_qa_embeddings_matrix",
    ]:
        setattr(EntitySearcher, method_name, originals[method_name])


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def _count_qa_pairs(searcher: EntitySearcher) -> int:
    """Count total QA pairs across all loaded GSW structures."""
    count = 0
    for gsw in searcher.gsw_by_doc_id.values():
        for vp in gsw.verb_phrase_nodes:
            if hasattr(vp, "questions") and vp.questions:
                count += len(vp.questions)
    return count


def run_single_benchmark(
    gsw_path: str,
    num_docs: int,
    gpu_device: int,
    embedding_gpu_memory_utilization: float,
) -> Dict[str, Any]:
    """Run one indexing pass and return timing + scale metrics."""
    global _phase_timings
    _phase_timings = {}

    # Use a temp dir for cache so we never hit existing caches
    with tempfile.TemporaryDirectory(prefix="panini_bench_") as tmp_cache:
        t_total_start = time.perf_counter()

        searcher = EntitySearcher(
            num_documents=num_docs,
            path_to_gsw_files=gsw_path,
            cache_dir=tmp_cache,
            rebuild_cache=True,
            verbose=False,
            use_bm25=True,
            gpu_device=gpu_device,
            use_gpu_for_qa_index=False,
            embedding_gpu_memory_utilization=embedding_gpu_memory_utilization,
        )

        t_total = time.perf_counter() - t_total_start

        num_entities = len(searcher.entities)
        num_qa = _count_qa_pairs(searcher)
        num_gsws = len(searcher.gsw_by_doc_id)
        num_qa_cached = len(searcher.qa_embedding_cache)

        # Collect phase times (some may be missing if skipped)
        phase_times = {
            "load_gsw_files": _phase_timings.get("load_gsw_files", 0.0),
            "extract_entities": _phase_timings.get("_extract_entities_from_gsw", 0.0),
            "build_bm25": _phase_timings.get("_build_bm25_index", 0.0),
            "generate_entity_embeddings": _phase_timings.get("_generate_embeddings", 0.0),
            "precompute_qa_embeddings": _phase_timings.get("_precompute_qa_embeddings", 0.0),
            "build_qa_matrix": _phase_timings.get("_build_qa_embeddings_matrix", 0.0),
            "total": t_total,
        }

        # Overhead = total - sum of measured phases
        measured_sum = sum(v for k, v in phase_times.items() if k != "total")
        phase_times["overhead_and_other"] = max(0.0, t_total - measured_sum)

    # Free memory
    del searcher
    gc.collect()

    return {
        "num_docs_requested": num_docs,
        "num_gsws_loaded": num_gsws,
        "num_entities": num_entities,
        "num_qa_pairs": num_qa,
        "num_qa_embeddings": num_qa_cached,
        "phase_times_sec": phase_times,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: List[Dict[str, Any]], output_dir: str):
    """Generate scaling plots from benchmark results."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        console.print("[yellow]matplotlib not available — skipping plots[/yellow]")
        return

    doc_counts = [r["num_gsws_loaded"] for r in results]
    phase_keys = [
        "load_gsw_files",
        "extract_entities",
        "build_bm25",
        "generate_entity_embeddings",
        "precompute_qa_embeddings",
        "build_qa_matrix",
        "overhead_and_other",
    ]
    phase_labels = [
        "Load GSW Files",
        "Extract Entities",
        "Build BM25",
        "Entity Embeddings",
        "QA Embeddings",
        "QA FAISS Matrix",
        "Other / Overhead",
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#9E9E9E"]

    # --- Plot 1: Stacked bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(doc_counts))
    bar_width = 0.6
    bottoms = np.zeros(len(doc_counts))

    for phase_key, label, color in zip(phase_keys, phase_labels, colors):
        values = np.array([r["phase_times_sec"][phase_key] for r in results])
        ax.bar(x, values, bar_width, bottom=bottoms, label=label, color=color)
        bottoms += values

    ax.set_xlabel("Number of Documents", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Panini Indexing Time Breakdown by Phase", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in doc_counts])
    ax.legend(loc="upper left", fontsize=9)

    # Add total time labels on top
    totals = [r["phase_times_sec"]["total"] for r in results]
    for i, total in enumerate(totals):
        if total >= 60:
            label = f"{total / 60:.1f}m"
        else:
            label = f"{total:.1f}s"
        ax.text(i, bottoms[i] + max(totals) * 0.01, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path1 = os.path.join(output_dir, "indexing_time_scaling.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved {path1}[/green]")

    # --- Plot 2: Log-log complexity analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: time vs docs (log-log) per phase
    ax1 = axes[0]
    for phase_key, label, color in zip(phase_keys[:-1], phase_labels[:-1], colors[:-1]):
        values = np.array([r["phase_times_sec"][phase_key] for r in results])
        if np.all(values > 0):
            ax1.plot(doc_counts, values, "o-", label=label, color=color, markersize=5)

            # Fit power law: t = a * n^b => log(t) = log(a) + b*log(n)
            log_n = np.log(np.array(doc_counts, dtype=float))
            log_t = np.log(values)
            coeffs = np.polyfit(log_n, log_t, 1)
            b = coeffs[0]
            ax1.annotate(
                f"O(n^{b:.2f})",
                xy=(doc_counts[-1], values[-1]),
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    # Total
    total_values = np.array([r["phase_times_sec"]["total"] for r in results])
    ax1.plot(doc_counts, total_values, "s--", label="Total", color="black", markersize=6, linewidth=2)
    if np.all(total_values > 0):
        log_n = np.log(np.array(doc_counts, dtype=float))
        log_t = np.log(total_values)
        coeffs = np.polyfit(log_n, log_t, 1)
        ax1.annotate(
            f"O(n^{coeffs[0]:.2f})",
            xy=(doc_counts[-1], total_values[-1]),
            fontsize=9,
            color="black",
            fontweight="bold",
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Documents", fontsize=12)
    ax1.set_ylabel("Time (seconds)", fontsize=12)
    ax1.set_title("Time Complexity (log-log)", fontsize=13)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.grid(True, alpha=0.3)

    # Right: entity/QA count scaling
    ax2 = axes[1]
    entities = [r["num_entities"] for r in results]
    qa_pairs = [r["num_qa_pairs"] for r in results]
    qa_embeds = [r["num_qa_embeddings"] for r in results]

    ax2.plot(doc_counts, entities, "o-", label="Entities", color="#2196F3", markersize=5)
    ax2.plot(doc_counts, qa_pairs, "s-", label="QA Pairs", color="#F44336", markersize=5)
    ax2.plot(doc_counts, qa_embeds, "^-", label="QA Embeddings", color="#9C27B0", markersize=5)

    ax2.set_xlabel("Number of Documents", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Data Scale vs Documents", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Annotate entities/doc and QA/doc ratios
    if len(doc_counts) > 0 and doc_counts[-1] > 0:
        ent_per_doc = entities[-1] / doc_counts[-1]
        qa_per_doc = qa_pairs[-1] / doc_counts[-1]
        ax2.annotate(
            f"~{ent_per_doc:.0f} ent/doc\n~{qa_per_doc:.0f} QA/doc",
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

    plt.tight_layout()
    path2 = os.path.join(output_dir, "complexity_analysis.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved {path2}[/green]")

    # --- Plot 3: Percentage breakdown (area chart) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    phase_pcts = {}
    for phase_key, label in zip(phase_keys, phase_labels):
        values = np.array([r["phase_times_sec"][phase_key] for r in results])
        pcts = values / total_values * 100
        phase_pcts[label] = pcts

    ax.stackplot(
        doc_counts,
        *phase_pcts.values(),
        labels=list(phase_pcts.keys()),
        colors=colors,
        alpha=0.85,
    )
    ax.set_xlabel("Number of Documents", fontsize=12)
    ax.set_ylabel("% of Total Time", fontsize=12)
    ax.set_title("Phase Time Share vs Document Count", fontsize=14)
    ax.legend(loc="center right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path3 = os.path.join(output_dir, "phase_percentage.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved {path3}[/green]")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(results: List[Dict[str, Any]]):
    """Print a rich table summarizing benchmark results."""
    table = Table(title="Panini Indexing Benchmark Results", show_lines=True)
    table.add_column("Docs", justify="right", style="cyan")
    table.add_column("Entities", justify="right")
    table.add_column("QA Pairs", justify="right")
    table.add_column("Load", justify="right")
    table.add_column("Extract", justify="right")
    table.add_column("BM25", justify="right")
    table.add_column("Ent Embed", justify="right", style="red")
    table.add_column("QA Embed", justify="right", style="magenta")
    table.add_column("Total", justify="right", style="bold green")

    def fmt_time(seconds: float) -> str:
        if seconds >= 60:
            return f"{seconds / 60:.1f}m"
        return f"{seconds:.1f}s"

    for r in results:
        pt = r["phase_times_sec"]
        table.add_row(
            str(r["num_gsws_loaded"]),
            str(r["num_entities"]),
            str(r["num_qa_pairs"]),
            fmt_time(pt["load_gsw_files"]),
            fmt_time(pt["extract_entities"]),
            fmt_time(pt["build_bm25"]),
            fmt_time(pt["generate_entity_embeddings"]),
            fmt_time(pt["precompute_qa_embeddings"]),
            fmt_time(pt["total"]),
        )

    console.print(table)

    # Print complexity estimates
    if len(results) >= 3:
        doc_counts = np.array([r["num_gsws_loaded"] for r in results], dtype=float)
        console.print("\n[bold]Estimated Complexity (power-law fit t = a·n^b):[/bold]")
        for phase_key, label in [
            ("generate_entity_embeddings", "Entity Embeddings"),
            ("precompute_qa_embeddings", "QA Embeddings"),
            ("total", "Total"),
        ]:
            values = np.array([r["phase_times_sec"][phase_key] for r in results])
            if np.all(values > 0):
                log_n = np.log(doc_counts)
                log_t = np.log(values)
                b, log_a = np.polyfit(log_n, log_t, 1)
                a = np.exp(log_a)
                console.print(f"  {label:25s}: t = {a:.4f} · n^{b:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Panini EntitySearcher indexing time scaling")
    parser.add_argument("--gsw_path", required=True, help="Path to GSW networks directory")
    parser.add_argument("--output_dir", default="logs/benchmark_indexing", help="Output directory for results")
    parser.add_argument(
        "--doc_counts",
        default="10,20,50,100,200,500,1000",
        help="Comma-separated list of document counts to test (use -1 for all)",
    )
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per doc count (for averaging)")
    parser.add_argument(
        "--embedding_gpu_memory_utilization",
        type=float,
        default=0.5,
        help="Fraction of GPU memory for vLLM embedding model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    doc_counts = [int(x.strip()) for x in args.doc_counts.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    console.print(f"[bold]Panini Indexing Benchmark[/bold]")
    console.print(f"  GSW path:   {args.gsw_path}")
    console.print(f"  Doc counts: {doc_counts}")
    console.print(f"  Runs/count: {args.runs}")
    console.print(f"  GPU device: {args.gpu_device}")
    console.print(f"  Output:     {args.output_dir}")
    console.print()

    # Warm up: initialize the embedding model once so it's shared across runs
    console.print("[cyan]Warming up embedding model (one-time cost)...[/cyan]")
    t_warmup_start = time.perf_counter()
    warmup_searcher = EntitySearcher(
        num_documents=1,
        path_to_gsw_files=args.gsw_path,
        cache_dir=tempfile.mkdtemp(prefix="panini_warmup_"),
        rebuild_cache=True,
        verbose=False,
        use_bm25=True,
        gpu_device=args.gpu_device,
        use_gpu_for_qa_index=False,
        embedding_gpu_memory_utilization=args.embedding_gpu_memory_utilization,
    )
    t_warmup = time.perf_counter() - t_warmup_start
    console.print(f"[green]Model warm-up done in {t_warmup:.1f}s (will be reused)[/green]\n")
    del warmup_searcher
    gc.collect()

    # Instrument
    originals = _instrument_searcher_class()

    all_results = []
    try:
        for doc_count in doc_counts:
            console.print(f"[bold cyan]--- Benchmarking num_docs={doc_count} ---[/bold cyan]")

            run_results = []
            for run_idx in range(args.runs):
                if args.runs > 1:
                    console.print(f"  Run {run_idx + 1}/{args.runs}...")

                result = run_single_benchmark(
                    gsw_path=args.gsw_path,
                    num_docs=doc_count,
                    gpu_device=args.gpu_device,
                    embedding_gpu_memory_utilization=args.embedding_gpu_memory_utilization,
                )
                run_results.append(result)

                total_t = result["phase_times_sec"]["total"]
                console.print(
                    f"  [green]Done:[/green] {result['num_gsws_loaded']} docs, "
                    f"{result['num_entities']} entities, "
                    f"{result['num_qa_pairs']} QA pairs, "
                    f"total={total_t:.1f}s ({total_t / 60:.1f}m)"
                )

            # Average across runs
            if args.runs > 1:
                averaged = {
                    "num_docs_requested": doc_count,
                    "num_gsws_loaded": run_results[0]["num_gsws_loaded"],
                    "num_entities": run_results[0]["num_entities"],
                    "num_qa_pairs": run_results[0]["num_qa_pairs"],
                    "num_qa_embeddings": run_results[0]["num_qa_embeddings"],
                    "num_runs": args.runs,
                    "phase_times_sec": {},
                }
                for phase_key in run_results[0]["phase_times_sec"]:
                    values = [r["phase_times_sec"][phase_key] for r in run_results]
                    averaged["phase_times_sec"][phase_key] = float(np.mean(values))
                    averaged["phase_times_sec"][f"{phase_key}_std"] = float(np.std(values))
                all_results.append(averaged)
            else:
                all_results.append(run_results[0])

            console.print()

    finally:
        _restore_searcher_class(originals)

    # Save results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": {
                    "gsw_path": args.gsw_path,
                    "doc_counts": doc_counts,
                    "runs_per_count": args.runs,
                    "gpu_device": args.gpu_device,
                    "model_warmup_time_sec": t_warmup,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    console.print(f"[green]Saved results to {results_path}[/green]")

    # Summary
    print_summary(all_results)

    # Plots
    plot_results(all_results, args.output_dir)

    # Extrapolation
    if len(all_results) >= 3:
        doc_counts_arr = np.array([r["num_gsws_loaded"] for r in all_results], dtype=float)
        total_times = np.array([r["phase_times_sec"]["total"] for r in all_results])
        if np.all(total_times > 0):
            b, log_a = np.polyfit(np.log(doc_counts_arr), np.log(total_times), 1)
            a = np.exp(log_a)
            max_docs = max(doc_counts_arr)

            console.print(f"\n[bold]Extrapolation (t = {a:.4f} · n^{b:.2f}):[/bold]")
            for target in [500, 1000, 2000, 5000]:
                if target > max_docs:
                    predicted = a * (target ** b)
                    console.print(f"  {target:5d} docs → ~{predicted / 60:.1f} min")


if __name__ == "__main__":
    main()
