"""Plot learning curves from the continual learning experiment.

Reads learning_curve.json from each arm's output directory and plots:
1. Probe F1 over batches
2. Probe bridges generated per batch
3. Guidance entropy over batches

Usage:
    python playground/sleep_time/plot_learning_curve.py \
        --experiment_dir logs/experiments/2wiki_continual_learning \
        --output learning_curves.png
"""

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_learning_curves(experiment_dir: Path) -> dict[str, list[dict]]:
    """Load learning_curve.json from each arm subdirectory."""
    curves = {}
    for arm_dir in sorted(experiment_dir.iterdir()):
        if not arm_dir.is_dir():
            continue
        lc_path = arm_dir / "learning_curve.json"
        if lc_path.exists():
            with open(lc_path) as f:
                curves[arm_dir.name] = json.load(f)
    return curves


def print_table(curves: dict[str, list[dict]]) -> None:
    """Print learning curves as ASCII table (fallback when matplotlib unavailable)."""
    for arm_name, data in curves.items():
        print(f"\n{'='*70}")
        print(f"Arm: {arm_name} ({len(data)} batches)")
        print(f"{'='*70}")
        print(f"{'Batch':>6} {'Probe F1':>10} {'Probe EM':>10} {'Bridges':>10} {'Relations':>10} {'Entropy':>10} {'Main Br':>10}")
        print("-" * 70)
        for entry in data:
            print(
                f"{entry.get('batch_index', '?'):>6} "
                f"{entry.get('probe_f1', 0):.4f}     "
                f"{entry.get('probe_em', 0):.4f}     "
                f"{entry.get('probe_bridges_generated', 0):>10} "
                f"{entry.get('probe_relation_diversity', 0):>10} "
                f"{entry.get('guidance_entropy', 0):.3f}     "
                f"{entry.get('main_bridges_this_batch', 0):>10}"
            )


def plot_curves(curves: dict[str, list[dict]], output_path: str) -> None:
    """Plot learning curves with matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, printing table instead")
        print_table(curves)
        return

    arm_colors = {
        "guidance_off": "#2196F3",
        "guidance_on": "#F44336",
        "guidance_decay": "#4CAF50",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Continual Learning Experiment — Learning Curves", fontsize=14)

    for arm_name, data in curves.items():
        batches = [d["batch_index"] for d in data]
        color = arm_colors.get(arm_name, "#999999")

        # Probe F1
        axes[0, 0].plot(batches, [d.get("probe_f1", 0) for d in data],
                        marker="o", label=arm_name, color=color, markersize=4)
        axes[0, 0].set_title("Probe F1")
        axes[0, 0].set_ylabel("F1")
        axes[0, 0].set_xlabel("Batch")

        # Probe bridges generated
        axes[0, 1].plot(batches, [d.get("probe_bridges_generated", 0) for d in data],
                        marker="s", label=arm_name, color=color, markersize=4)
        axes[0, 1].set_title("Probe Bridges Generated")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_xlabel("Batch")

        # Guidance entropy
        axes[1, 0].plot(batches, [d.get("guidance_entropy", 0) for d in data],
                        marker="^", label=arm_name, color=color, markersize=4)
        axes[1, 0].set_title("Guidance Entropy")
        axes[1, 0].set_ylabel("Entropy (bits)")
        axes[1, 0].set_xlabel("Batch")

        # Main bridges per batch
        axes[1, 1].plot(batches, [d.get("main_bridges_this_batch", 0) for d in data],
                        marker="d", label=arm_name, color=color, markersize=4)
        axes[1, 1].set_title("Main Bridges per Batch")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_xlabel("Batch")

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from continual learning experiment")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Experiment output directory")
    parser.add_argument("--output", type=str, default="learning_curves.png", help="Output plot path")
    parser.add_argument("--table_only", action="store_true", help="Print ASCII table instead of plotting")
    args = parser.parse_args()

    curves = load_learning_curves(Path(args.experiment_dir))
    if not curves:
        print(f"No learning_curve.json found in {args.experiment_dir}")
        return

    print(f"Found {len(curves)} arms: {list(curves.keys())}")

    if args.table_only or not MATPLOTLIB_AVAILABLE:
        print_table(curves)
    else:
        plot_curves(curves, args.output)
        print_table(curves)


if __name__ == "__main__":
    main()
