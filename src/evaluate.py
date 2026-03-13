"""Evaluation script for comparing multiple runs using WandB data."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import wandb
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import pandas as pd


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("run_ids", type=str, help="JSON list of run IDs to compare")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--project", type=str, default=None, help="WandB project")

    args = parser.parse_args()

    # Parse run_ids
    run_ids = json.loads(args.run_ids)

    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {args.results_dir}")

    # Get WandB config from environment or args
    entity = args.entity or os.environ.get("WANDB_ENTITY")
    project = args.project or os.environ.get("WANDB_PROJECT")

    if not entity or not project:
        print(
            "ERROR: WandB entity and project must be specified via --entity/--project or environment variables"
        )
        sys.exit(1)

    print(f"Using WandB: {entity}/{project}")

    # Initialize WandB API
    api = wandb.Api()

    # Fetch data for each run
    all_run_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for run: {run_id}")
        run_data = fetch_run_data(api, entity, project, run_id)
        all_run_data[run_id] = run_data

        # Export per-run metrics
        export_run_metrics(args.results_dir, run_id, run_data)

        # Generate per-run figures
        generate_run_figures(args.results_dir, run_id, run_data)

    # Generate comparison
    print("\nGenerating comparison...")
    generate_comparison(args.results_dir, run_ids, all_run_data)

    print("\nEvaluation complete!")


def fetch_run_data(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> Dict[str, Any]:
    """
    Fetch run data from WandB.

    Args:
        api: WandB API instance
        entity: WandB entity
        project: WandB project
        run_id: Run ID (display name)

    Returns:
        Dictionary with run history, summary, and config
    """
    # Search for runs with this display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        print(f"WARNING: No runs found for {run_id}")
        return {
            "history": [],
            "summary": {},
            "config": {},
        }

    # Get the most recent run
    run = runs[0]

    print(f"  Found run: {run.name} (created: {run.created_at})")

    # Fetch history (logged metrics over time)
    history = []
    try:
        history_df = run.history()
        if not history_df.empty:
            history = history_df.to_dict("records")
    except Exception as e:
        print(f"  Warning: Could not fetch history: {e}")

    # Get summary metrics
    summary = dict(run.summary)

    # Get config
    config = dict(run.config)

    return {
        "history": history,
        "summary": summary,
        "config": config,
        "url": run.url,
    }


def export_run_metrics(results_dir: str, run_id: str, run_data: Dict) -> None:
    """
    Export per-run metrics to JSON.

    Args:
        results_dir: Results directory
        run_id: Run ID
        run_data: Fetched run data
    """
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = run_dir / "metrics.json"

    # Export summary metrics
    with open(metrics_file, "w") as f:
        json.dump(run_data["summary"], f, indent=2)

    print(f"  Exported metrics to: {metrics_file}")


def generate_run_figures(results_dir: str, run_id: str, run_data: Dict) -> None:
    """
    Generate per-run figures.

    Args:
        results_dir: Results directory
        run_id: Run ID
        run_data: Fetched run data
    """
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # For inference-only tasks, there's typically no training curve
    # Just create a summary plot if there are multiple logged metrics
    summary = run_data["summary"]

    if not summary:
        print(f"  No metrics to plot for {run_id}")
        return

    # Create a bar chart of key metrics
    metrics_to_plot = {
        k: v
        for k, v in summary.items()
        if isinstance(v, (int, float)) and not k.startswith("_")
    }

    if metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))

        keys = list(metrics_to_plot.keys())
        values = list(metrics_to_plot.values())

        ax.bar(keys, values)
        ax.set_ylabel("Value")
        ax.set_title(f"Metrics Summary: {run_id}")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        fig_file = run_dir / "metrics_summary.pdf"
        plt.savefig(fig_file)
        plt.close()

        print(f"  Generated figure: {fig_file}")


def generate_comparison(
    results_dir: str, run_ids: List[str], all_run_data: Dict[str, Dict]
) -> None:
    """
    Generate comparison figures and aggregated metrics.

    Args:
        results_dir: Results directory
        run_ids: List of run IDs
        all_run_data: Dictionary mapping run_id to run data
    """
    comparison_dir = Path(results_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics across runs
    metrics_by_run = {}

    for run_id in run_ids:
        run_data = all_run_data[run_id]
        summary = run_data["summary"]
        metrics_by_run[run_id] = summary

    # Determine primary metric (accuracy for this task)
    primary_metric = "accuracy"

    # Create aggregated metrics
    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": metrics_by_run,
    }

    # Find best proposed and best baseline
    proposed_runs = [rid for rid in run_ids if "proposed" in rid]
    baseline_runs = [
        rid for rid in run_ids if "comparative" in rid or "baseline" in rid
    ]

    if proposed_runs:
        best_proposed_id = max(
            proposed_runs, key=lambda x: metrics_by_run[x].get(primary_metric, 0)
        )
        aggregated["best_proposed"] = {
            "run_id": best_proposed_id,
            "value": metrics_by_run[best_proposed_id].get(primary_metric, 0),
        }

    if baseline_runs:
        best_baseline_id = max(
            baseline_runs, key=lambda x: metrics_by_run[x].get(primary_metric, 0)
        )
        aggregated["best_baseline"] = {
            "run_id": best_baseline_id,
            "value": metrics_by_run[best_baseline_id].get(primary_metric, 0),
        }

    # Compute gap
    if proposed_runs and baseline_runs:
        gap = (
            aggregated["best_proposed"]["value"] - aggregated["best_baseline"]["value"]
        )
        aggregated["gap"] = gap

    # Export aggregated metrics
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Exported aggregated metrics to: {agg_file}")

    # Generate comparison bar chart
    generate_comparison_bar_chart(
        comparison_dir, run_ids, metrics_by_run, primary_metric
    )

    # If we have additional metrics, create separate plots
    all_metric_keys = set()
    for metrics in metrics_by_run.values():
        all_metric_keys.update(metrics.keys())

    # Filter to numeric metrics
    numeric_metrics = []
    for key in all_metric_keys:
        if key.startswith("_"):
            continue
        # Check if numeric across all runs
        is_numeric = all(
            isinstance(metrics_by_run[rid].get(key), (int, float))
            for rid in run_ids
            if key in metrics_by_run[rid]
        )
        if is_numeric:
            numeric_metrics.append(key)

    # Create comparison plot for each metric
    for metric_name in numeric_metrics:
        if metric_name == primary_metric:
            continue  # Already plotted in main bar chart

        generate_comparison_bar_chart(
            comparison_dir,
            run_ids,
            metrics_by_run,
            metric_name,
            filename=f"comparison_{metric_name}.pdf",
        )


def generate_comparison_bar_chart(
    output_dir: Path,
    run_ids: List[str],
    metrics_by_run: Dict[str, Dict],
    metric_name: str,
    filename: str = "comparison_accuracy.pdf",
) -> None:
    """
    Generate a bar chart comparing a metric across runs.

    Args:
        output_dir: Output directory
        run_ids: List of run IDs
        metrics_by_run: Dictionary of metrics by run
        metric_name: Name of metric to plot
        filename: Output filename
    """
    values = []
    labels = []
    colors = []

    for run_id in run_ids:
        value = metrics_by_run[run_id].get(metric_name, 0)
        values.append(value)
        labels.append(run_id)

        # Color code: proposed = blue, baseline = orange
        if "proposed" in run_id:
            colors.append("steelblue")
        else:
            colors.append("coral")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Comparison: {metric_name.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    fig_file = output_dir / filename
    plt.savefig(fig_file)
    plt.close()

    print(f"Generated comparison figure: {fig_file}")


if __name__ == "__main__":
    main()
