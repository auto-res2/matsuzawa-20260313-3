"""Evaluation and comparison across multiple runs."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import wandb


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run data
    """
    api = wandb.Api()

    # Fetch runs with matching display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        print(f"Warning: No runs found for {run_id} in {entity}/{project}")
        return None

    # Get most recent run
    run = runs[0]

    # Extract data
    data = {
        "run_id": run_id,
        "name": run.name,
        "config": run.config,
        "summary": dict(run.summary),
        "history": run.history().to_dict(orient="records")
        if hasattr(run, "history")
        else [],
    }

    return data


def export_run_metrics(run_data: Dict, output_dir: Path):
    """
    Export metrics for a single run.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(run_data["summary"], f, indent=2)

    print(f"Exported metrics: {metrics_file}")

    # Create per-run figures
    if run_data.get("history"):
        create_run_figures(run_data, output_dir)


def create_run_figures(run_data: Dict, output_dir: Path):
    """
    Create visualization figures for a single run.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory
    """
    history = run_data["history"]
    if not history:
        return

    # Plot reasoning steps over examples
    if any("reasoning_steps" in h for h in history):
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = [h.get("reasoning_steps", 0) for h in history]
        ax.plot(steps, marker="o", markersize=3)
        ax.set_xlabel("Example")
        ax.set_ylabel("Reasoning Steps")
        ax.set_title(f"Reasoning Steps - {run_data['run_id']}")
        ax.grid(True, alpha=0.3)

        output_file = output_dir / "reasoning_steps.pdf"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        print(f"Created figure: {output_file}")

    # Plot accuracy over examples
    if any("correct" in h for h in history):
        fig, ax = plt.subplots(figsize=(10, 6))
        correct = [h.get("correct", 0) for h in history]
        cumulative_acc = [sum(correct[: i + 1]) / (i + 1) for i in range(len(correct))]
        ax.plot(cumulative_acc, marker="o", markersize=3)
        ax.set_xlabel("Example")
        ax.set_ylabel("Cumulative Accuracy")
        ax.set_title(f"Cumulative Accuracy - {run_data['run_id']}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        output_file = output_dir / "accuracy.pdf"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        print(f"Created figure: {output_file}")


def create_comparison_figures(all_run_data: List[Dict], output_dir: Path):
    """
    Create comparison figures across multiple runs.

    Args:
        all_run_data: List of run data dictionaries
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparison bar chart for primary metrics
    metrics_to_compare = [
        "accuracy",
        "reasoning_steps",
        "decision_count",
        "uncertain_decisions",
        "support_violations",
    ]

    for metric in metrics_to_compare:
        # Check if all runs have this metric
        values = {}
        for run_data in all_run_data:
            if metric in run_data["summary"]:
                values[run_data["run_id"]] = run_data["summary"][metric]

        if not values:
            continue

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        run_ids = list(values.keys())
        vals = list(values.values())

        bars = ax.bar(
            run_ids,
            vals,
            color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"][: len(run_ids)],
        )
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Comparison: {metric.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x labels if needed
        if len(run_ids) > 3:
            plt.xticks(rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        output_file = output_dir / f"comparison_{metric}.pdf"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        print(f"Created comparison figure: {output_file}")


def compute_aggregated_metrics(all_run_data: List[Dict]) -> Dict:
    """
    Compute aggregated metrics across runs.

    Args:
        all_run_data: List of run data dictionaries

    Returns:
        Aggregated metrics dictionary
    """
    # Collect metrics by run
    metrics_by_run = {}
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        metrics_by_run[run_id] = run_data["summary"]

    # Identify proposed and baseline runs
    proposed_runs = {k: v for k, v in metrics_by_run.items() if "proposed" in k}
    baseline_runs = {k: v for k, v in metrics_by_run.items() if "comparative" in k}

    # Find best runs
    best_proposed = None
    best_proposed_acc = -1
    if proposed_runs:
        for run_id, metrics in proposed_runs.items():
            acc = metrics.get("accuracy", 0)
            if acc > best_proposed_acc:
                best_proposed_acc = acc
                best_proposed = run_id

    best_baseline = None
    best_baseline_acc = -1
    if baseline_runs:
        for run_id, metrics in baseline_runs.items():
            acc = metrics.get("accuracy", 0)
            if acc > best_baseline_acc:
                best_baseline_acc = acc
                best_baseline = run_id

    # Compute gap
    gap = (
        best_proposed_acc - best_baseline_acc
        if best_proposed and best_baseline
        else 0.0
    )

    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_proposed_accuracy": best_proposed_acc,
        "best_baseline": best_baseline,
        "best_baseline_accuracy": best_baseline_acc,
        "gap": gap,
    }

    return aggregated


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare experimental runs"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs"
    )
    args = parser.parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating runs: {run_ids}")

    # Get WandB credentials from environment
    entity = os.getenv("WANDB_ENTITY", "airas")
    project = os.getenv("WANDB_PROJECT", "2026-0313-matsuzawa-3")

    # Fetch data for all runs
    all_run_data = []
    for run_id in run_ids:
        print(f"\nFetching data for: {run_id}")
        run_data = fetch_wandb_run(entity, project, run_id)
        if run_data:
            all_run_data.append(run_data)

            # Export per-run metrics
            run_dir = Path(args.results_dir) / run_id
            export_run_metrics(run_data, run_dir)

    if not all_run_data:
        print("Error: No run data fetched")
        return

    # Create comparison directory
    comparison_dir = Path(args.results_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Compute aggregated metrics
    print("\nComputing aggregated metrics...")
    aggregated = compute_aggregated_metrics(all_run_data)

    # Export aggregated metrics
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Exported aggregated metrics: {agg_file}")

    # Create comparison figures
    print("\nCreating comparison figures...")
    create_comparison_figures(all_run_data, comparison_dir)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Results saved to: {args.results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
