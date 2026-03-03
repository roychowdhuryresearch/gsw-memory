#!/usr/bin/env python3
"""
Validate Bridge QAs against MuSiQue question decompositions.

Checks whether the sleep-time agent's bridge QAs correctly connected
the supporting documents for each multi-hop question.

Metrics:
- Bridge hit: does any bridge connect 2+ supporting docs?
- Hop pair coverage: for each decomposition hop pair, was there a bridge?
- Precision: what fraction of bridges involve at least 1 supporting doc?

Usage:
    python playground/sleep_time/misc/validate_bridges.py \
        --results logs/bridge_test/bridge_test_results.json \
        --questions playground_data/musique.json
"""

import argparse
import json
import re
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

DOCS_PER_QUESTION = 20


def parse_doc_id(doc_str: str) -> int:
    """Extract numeric index from doc ID string like 'doc_5'."""
    match = re.search(r"doc_(\d+)", doc_str)
    return int(match.group(1)) if match else -1


def get_decomposition_hop_pairs(result: dict) -> list:
    """
    Extract (global_doc_a, global_doc_b) pairs from question decomposition.
    Each decomposition step has paragraph_support_idx (local).
    Consecutive steps that reference different docs form a hop pair.
    """
    q_idx = result["question_idx"]
    decomp = result.get("decomposition", [])

    # Get unique supporting doc indices in decomposition order
    doc_sequence = []
    for step in decomp:
        local_idx = step.get("paragraph_support_idx")
        if local_idx is not None:
            global_idx = q_idx * DOCS_PER_QUESTION + local_idx
            doc_sequence.append(global_idx)

    # Create pairs of consecutive docs
    pairs = []
    for i in range(len(doc_sequence) - 1):
        if doc_sequence[i] != doc_sequence[i + 1]:
            pairs.append((doc_sequence[i], doc_sequence[i + 1]))
    return pairs


def validate_question(result: dict) -> dict:
    """Validate bridges for a single question."""
    if "error" in result:
        return {"error": result["error"]}

    support_global = set(result.get("supporting_doc_indices_global", []))
    bridges = result.get("bridges", [])
    hop_pairs = get_decomposition_hop_pairs(result)

    # Per-bridge analysis
    bridge_details = []
    bridges_hitting_support = 0
    bridges_connecting_support = 0

    for bridge in bridges:
        source_docs = bridge.get("source_docs", [])
        bridge_doc_indices = {parse_doc_id(d) for d in source_docs}

        # How many supporting docs does this bridge touch?
        support_overlap = bridge_doc_indices & support_global
        hits_support = len(support_overlap) >= 1
        connects_support = len(support_overlap) >= 2

        if hits_support:
            bridges_hitting_support += 1
        if connects_support:
            bridges_connecting_support += 1

        bridge_details.append({
            "question": bridge.get("question", ""),
            "answers": bridge.get("answers", []),
            "reverse_question": bridge.get("reverse_question", ""),
            "reverse_answers": bridge.get("reverse_answers", []),
            "source_docs": source_docs,
            "bridge_doc_indices": sorted(bridge_doc_indices),
            "support_overlap": sorted(support_overlap),
            "hits_support": hits_support,
            "connects_support": connects_support,
            "confidence": bridge.get("confidence", 0),
        })

    # Hop pair coverage: for each consecutive pair in decomposition,
    # check if ANY bridge connects those two docs
    hop_pair_hits = []
    for doc_a, doc_b in hop_pairs:
        covered = False
        for bridge in bridges:
            source_docs = bridge.get("source_docs", [])
            bridge_doc_indices = {parse_doc_id(d) for d in source_docs}
            if doc_a in bridge_doc_indices and doc_b in bridge_doc_indices:
                covered = True
                break
        hop_pair_hits.append({
            "doc_pair": (doc_a, doc_b),
            "covered": covered,
        })

    return {
        "num_bridges": len(bridges),
        "bridges_hitting_support": bridges_hitting_support,
        "bridges_connecting_support": bridges_connecting_support,
        "precision": bridges_hitting_support / len(bridges) if bridges else 0,
        "has_correct_bridge": bridges_connecting_support > 0,
        "hop_pairs": hop_pair_hits,
        "hop_pair_coverage": sum(1 for h in hop_pair_hits if h["covered"]) / len(hop_pair_hits) if hop_pair_hits else 0,
        "bridge_details": bridge_details,
    }


def print_results(results: list, validations: list):
    """Print rich table with per-question results."""
    # Per-question table
    table = Table(
        title="Bridge Validation Results",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
    )
    table.add_column("Q#", style="cyan", width=4)
    table.add_column("Question", max_width=40)
    table.add_column("Hops", justify="center", width=5)
    table.add_column("Bridges", justify="center", width=8)
    table.add_column("Hit Support", justify="center", width=12)
    table.add_column("Connect", justify="center", width=8)
    table.add_column("Hop Coverage", justify="center", width=12)
    table.add_column("Precision", justify="center", width=10)

    for result, validation in zip(results, validations):
        if "error" in validation:
            table.add_row(
                str(result.get("question_idx", "?")),
                result.get("question", "")[:40],
                "-", "-", "-", "-", "-",
                f"[red]ERR[/red]",
            )
            continue

        q_idx = result["question_idx"]
        question = result["question"][:40]
        hops = str(result.get("num_hops", "?"))
        num_bridges = str(validation["num_bridges"])
        hit_support = str(validation["bridges_hitting_support"])
        connects = "[green]YES[/green]" if validation["has_correct_bridge"] else "[red]NO[/red]"
        hop_cov = f"{validation['hop_pair_coverage']:.0%}"
        precision = f"{validation['precision']:.0%}"

        table.add_row(q_idx.__str__(), question, hops, num_bridges, hit_support, connects, hop_cov, precision)

    console.print(table)

    # Aggregate stats
    valid_results = [v for v in validations if "error" not in v]
    if not valid_results:
        console.print("[red]No valid results to aggregate[/red]")
        return

    total_questions = len(valid_results)
    questions_with_correct_bridge = sum(1 for v in valid_results if v["has_correct_bridge"])
    total_bridges = sum(v["num_bridges"] for v in valid_results)
    total_connecting = sum(v["bridges_connecting_support"] for v in valid_results)
    all_hop_pairs = [h for v in valid_results for h in v["hop_pairs"]]
    hop_pairs_covered = sum(1 for h in all_hop_pairs if h["covered"])
    avg_precision = sum(v["precision"] for v in valid_results) / total_questions

    console.print(f"\n[bold]Aggregate Statistics:[/bold]")
    console.print(f"  Questions with correct bridge: {questions_with_correct_bridge}/{total_questions} ({questions_with_correct_bridge/total_questions:.0%})")
    console.print(f"  Total bridges: {total_bridges} ({total_connecting} connecting supporting docs)")
    console.print(f"  Hop pair coverage: {hop_pairs_covered}/{len(all_hop_pairs)} ({hop_pairs_covered/len(all_hop_pairs):.0%})" if all_hop_pairs else "  Hop pairs: none")
    console.print(f"  Average precision: {avg_precision:.0%}")

    # Detailed per-question bridge listing
    console.print(f"\n[bold]Bridge Details:[/bold]")
    for result, validation in zip(results, validations):
        if "error" in validation or not validation.get("bridge_details"):
            continue

        q_idx = result["question_idx"]
        console.print(f"\n  [bold cyan]Q{q_idx}: {result['question']}[/bold cyan]")
        console.print(f"  Answer: {result['answer']}")
        console.print(f"  Supporting docs: {result.get('supporting_doc_titles', [])}")

        for i, bd in enumerate(validation["bridge_details"]):
            status = "[green]CONNECT[/green]" if bd["connects_support"] else (
                "[yellow]PARTIAL[/yellow]" if bd["hits_support"] else "[red]MISS[/red]"
            )
            console.print(f"    Bridge {i+1} {status}: {bd['question']}")
            console.print(f"      docs: {bd['source_docs']} | overlap: {bd['support_overlap']} | conf: {bd['confidence']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate bridge QAs against question decompositions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to bridge_test_results.json from run_bridge_test.py")
    parser.add_argument("--questions", type=str, default="playground_data/musique.json",
                        help="Path to MuSiQue questions (for extra context)")

    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    results = data.get("results", [])
    console.print(f"Loaded {len(results)} question results")
    console.print(f"Model: {data.get('metadata', {}).get('model', 'unknown')}")

    # Validate each question
    validations = [validate_question(r) for r in results]

    # Print
    print_results(results, validations)

    # Save validation output
    output_file = args.results.replace(".json", "_validation.json")
    validation_data = {
        "metadata": data.get("metadata", {}),
        "validation": [
            {**r, "validation": v}
            for r, v in zip(results, validations)
        ],
    }
    with open(output_file, 'w') as f:
        json.dump(validation_data, f, indent=2, default=str)
    console.print(f"\n[green]Validation saved to: {output_file}[/green]")


if __name__ == "__main__":
    main()
