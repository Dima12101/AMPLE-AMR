"""Result export, plotting and LaTeX table generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_directory


RESULT_FILES = {
    "raw_steps": "raw_steps.csv",
    "episodes": "episodes.csv",
    "summary_by_seed": "summary_by_seed.csv",
    "summary": "summary.csv",
    "scalability": "scalability.csv",
    "clustered_vs_global": "clustered_vs_global.csv",
    "sensitivity_operation_modes": "sensitivity_operation_modes.csv",
    "training_history": "training_history.csv",
}


def save_result_frames(
    output_dir: str | Path,
    raw_steps: pd.DataFrame,
    episodes: pd.DataFrame,
    training_history: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write raw and aggregated result frames to disk."""

    result_dir = ensure_directory(output_dir)
    raw_steps.to_csv(result_dir / RESULT_FILES["raw_steps"], index=False)
    episodes.to_csv(result_dir / RESULT_FILES["episodes"], index=False)

    summary_by_seed = (
        episodes.groupby(["scenario", "scenario_size", "method", "mode_profile", "seed"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    summary_by_seed.to_csv(result_dir / RESULT_FILES["summary_by_seed"], index=False)

    summary = (
        summary_by_seed.groupby(["scenario", "scenario_size", "method", "mode_profile"], dropna=False)
        .agg(
            social_welfare=("social_welfare", "mean"),
            completed_tasks=("completed_tasks", "mean"),
            generated_tasks=("generated_tasks", "mean"),
            completion_rate=("completion_rate", "mean"),
            drop_rate=("drop_rate", "mean"),
            deadline_violation_rate=("deadline_violation_rate", "mean"),
            avg_latency_ms=("avg_latency_ms", "mean"),
            p95_latency_ms=("p95_latency_ms", "mean"),
            avg_node_utilization=("avg_node_utilization", "mean"),
            load_imbalance=("load_imbalance", "mean"),
            overload_count=("overload_count", "mean"),
            allocation_overhead_ms=("allocation_overhead_ms", "mean"),
            policy_inference_time_ms=("policy_inference_time_ms", "mean"),
            total_decision_overhead_ms=("total_decision_overhead_ms", "mean"),
            slot_share=("slot_share", "mean"),
            payments_sum=("payments_sum", "mean"),
            payment_robot_sum=("payment_robot_sum", "mean"),
            payment_task_sum=("payment_task_sum", "mean"),
            node_compensation_sum=("node_compensation_sum", "mean"),
            average_externality=("average_externality", "mean"),
            externality_mean=("externality_mean", "mean"),
            externality_p95=("externality_p95", "mean"),
            externality_max=("externality_max", "mean"),
            cluster_count=("cluster_count", "mean"),
            avg_cluster_size=("avg_cluster_size", "mean"),
            graph_cut=("graph_cut", "mean"),
            welfare_loss_vs_global=("welfare_loss_vs_global", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(result_dir / RESULT_FILES["summary"], index=False)

    scalability = summary[summary["scenario"] == "scalability_sweep"].copy()
    scalability.to_csv(result_dir / RESULT_FILES["scalability"], index=False)

    clustered_vs_global = _build_clustered_vs_global(summary_by_seed)
    clustered_vs_global.to_csv(result_dir / RESULT_FILES["clustered_vs_global"], index=False)

    sensitivity = summary[summary["scenario"] == "sensitivity_operation_modes"].copy()
    sensitivity.to_csv(result_dir / RESULT_FILES["sensitivity_operation_modes"], index=False)

    if training_history is None:
        training_history = pd.DataFrame(columns=["size_name", "training_scenario", "episode", "reward", "loss", "epsilon"])
    training_history.to_csv(result_dir / RESULT_FILES["training_history"], index=False)

    return {
        "raw_steps": result_dir / RESULT_FILES["raw_steps"],
        "episodes": result_dir / RESULT_FILES["episodes"],
        "summary_by_seed": result_dir / RESULT_FILES["summary_by_seed"],
        "summary": result_dir / RESULT_FILES["summary"],
        "scalability": result_dir / RESULT_FILES["scalability"],
        "clustered_vs_global": result_dir / RESULT_FILES["clustered_vs_global"],
        "sensitivity_operation_modes": result_dir / RESULT_FILES["sensitivity_operation_modes"],
        "training_history": result_dir / RESULT_FILES["training_history"],
    }


def generate_plots(output_dir: str | Path) -> list[Path]:
    """Generate all requested figures from saved CSV outputs."""

    result_dir = ensure_directory(output_dir)
    summary = pd.read_csv(result_dir / RESULT_FILES["summary"])
    raw_steps = pd.read_csv(result_dir / RESULT_FILES["raw_steps"])
    scalability = pd.read_csv(result_dir / RESULT_FILES["scalability"])
    clustered = pd.read_csv(result_dir / RESULT_FILES["clustered_vs_global"])
    training_history = pd.read_csv(result_dir / RESULT_FILES["training_history"])

    created: list[Path] = []
    created.extend(_plot_summary_metric(summary, "social_welfare", "Welfare by Scenario", result_dir / "welfare_by_scenario"))
    created.extend(_plot_rate_triplet(summary, result_dir / "completion_drop_violation_by_scenario"))
    created.extend(_plot_summary_metric(summary, "p95_latency_ms", "P95 Latency by Scenario", result_dir / "latency_p95_by_scenario"))
    created.extend(_plot_scalability(scalability, result_dir / "overhead_scalability"))
    created.extend(_plot_cluster_tradeoff(clustered, result_dir / "global_vs_clustered_tradeoff"))
    created.extend(_plot_utilization_heatmap(summary, result_dir / "node_utilization_heatmap"))
    created.extend(_plot_learning_curves(training_history, result_dir / "learning_curves"))
    created.extend(_plot_mode_distribution(raw_steps, result_dir / "operation_mode_distribution"))
    return created


def export_latex_tables(output_dir: str | Path, tables_dir: str | Path) -> list[Path]:
    """Export scenario-specific LaTeX tables."""

    result_dir = ensure_directory(output_dir)
    tables_path = ensure_directory(tables_dir)
    summary = pd.read_csv(result_dir / RESULT_FILES["summary"])
    scalability = pd.read_csv(result_dir / RESULT_FILES["scalability"])
    clustered = pd.read_csv(result_dir / RESULT_FILES["clustered_vs_global"])
    sensitivity = pd.read_csv(result_dir / RESULT_FILES["sensitivity_operation_modes"])

    created: list[Path] = []
    scenario_files = {
        "stable_warehouse_load": tables_path / "stable_results.tex",
        "peak_warehouse_load": tables_path / "peak_results.tex",
        "heterogeneous_edge_nodes": tables_path / "heterogeneous_results.tex",
        "network_degradation": tables_path / "network_degradation_results.tex",
        "edge_node_failures": tables_path / "failure_results.tex",
    }
    for scenario_name, table_path in scenario_files.items():
        frame = summary[summary["scenario"] == scenario_name].copy()
        created.append(_write_latex(frame, table_path))

    created.append(_write_latex(scalability.copy(), tables_path / "scalability_results.tex"))
    created.append(_write_latex(clustered.copy(), tables_path / "clustered_results.tex"))
    created.append(_write_latex(sensitivity.copy(), tables_path / "sensitivity_operation_modes.tex"))
    return created


def summarize_generated_files(paths: list[Path]) -> str:
    """Build a concise text summary of generated files."""

    labels = [str(path) for path in sorted(paths)]
    return "\n".join(labels)


def _build_clustered_vs_global(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    ample = summary_by_seed[summary_by_seed["method"] == "ample_amr"].copy()
    clustered = summary_by_seed[summary_by_seed["method"] == "c_ample_amr"].copy()
    if ample.empty or clustered.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "scenario_size",
                "seed",
                "mode_profile",
                "ample_welfare",
                "clustered_welfare",
                "welfare_loss_vs_global",
                "ample_overhead_ms",
                "clustered_overhead_ms",
                "cluster_count",
                "avg_cluster_size",
                "graph_cut",
            ]
        )
    merged = ample.merge(
        clustered,
        on=["scenario", "scenario_size", "seed", "mode_profile"],
        suffixes=("_ample", "_clustered"),
    )
    return pd.DataFrame(
        {
            "scenario": merged["scenario"],
            "scenario_size": merged["scenario_size"],
            "seed": merged["seed"],
            "mode_profile": merged["mode_profile"],
            "ample_welfare": merged["social_welfare_ample"],
            "clustered_welfare": merged["social_welfare_clustered"],
            "welfare_loss_vs_global": merged["social_welfare_ample"] - merged["social_welfare_clustered"],
            "ample_overhead_ms": merged["total_decision_overhead_ms_ample"],
            "clustered_overhead_ms": merged["total_decision_overhead_ms_clustered"],
            "cluster_count": merged["cluster_count_clustered"],
            "avg_cluster_size": merged["avg_cluster_size_clustered"],
            "graph_cut": merged["graph_cut_clustered"],
        }
    )


def _plot_summary_metric(summary: pd.DataFrame, metric: str, title: str, stem: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot = summary.pivot_table(index="scenario", columns="method", values=metric, aggfunc="mean", fill_value=0.0)
    pivot.plot(kind="bar", ax=axis)
    axis.set_title(title)
    axis.set_ylabel(metric)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _save_plot(figure, stem)


def _plot_rate_triplet(summary: pd.DataFrame, stem: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = ["completion_rate", "drop_rate", "deadline_violation_rate"]
    for axis, metric in zip(axes, metrics, strict=True):
        pivot = summary.pivot_table(index="scenario", columns="method", values=metric, aggfunc="mean", fill_value=0.0)
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.legend([], [], frameon=False)
    figure.tight_layout()
    return _save_plot(figure, stem)


def _plot_scalability(scalability: pd.DataFrame, stem: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8, 5))
    if not scalability.empty:
        for method, frame in scalability.groupby("method"):
            ordered = frame.sort_values("scenario_size")
            axis.plot(ordered["scenario_size"], ordered["total_decision_overhead_ms"], marker="o", label=method)
    axis.set_title("Overhead Scalability")
    axis.set_ylabel("total_decision_overhead_ms")
    if axis.has_data():
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _save_plot(figure, stem)


def _plot_cluster_tradeoff(clustered: pd.DataFrame, stem: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8, 5))
    if not clustered.empty:
        axis.scatter(clustered["clustered_overhead_ms"], clustered["clustered_welfare"], label="C-AMPLE-AMR", alpha=0.8)
        axis.scatter(clustered["ample_overhead_ms"], clustered["ample_welfare"], label="AMPLE-AMR", alpha=0.8)
    axis.set_title("Global vs Clustered Tradeoff")
    axis.set_xlabel("decision overhead ms")
    axis.set_ylabel("social welfare")
    if axis.has_data():
        axis.legend(loc="best")
    figure.tight_layout()
    return _save_plot(figure, stem)


def _plot_utilization_heatmap(summary: pd.DataFrame, stem: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot = summary.pivot_table(index="scenario", columns="method", values="avg_node_utilization", aggfunc="mean", fill_value=0.0)
    heatmap = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Node Utilization Heatmap")
    figure.colorbar(heatmap, ax=axis)
    figure.tight_layout()
    return _save_plot(figure, stem)


def _plot_learning_curves(training_history: pd.DataFrame, stem: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not training_history.empty:
        for size_name, frame in training_history.groupby("size_name"):
            ordered = frame.sort_values("episode")
            axes[0].plot(ordered["episode"], ordered["reward"], label=size_name)
            axes[1].plot(ordered["episode"], ordered["loss"], label=size_name)
    axes[0].set_title("QMIX Reward")
    axes[1].set_title("QMIX Loss")
    for axis in axes:
        if axis.has_data():
            axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _save_plot(figure, stem)


def _plot_mode_distribution(raw_steps: pd.DataFrame, stem: Path) -> list[Path]:
    mode_columns = [
        column
        for column in raw_steps.columns
        if column.startswith("mode_") and column != "mode_profile" and pd.api.types.is_numeric_dtype(raw_steps[column])
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    if mode_columns:
        distribution = raw_steps.groupby("method")[mode_columns].mean()
        distribution.plot(kind="bar", stacked=True, ax=axis)
    axis.set_title("Operation Mode Distribution")
    axis.set_ylabel("average node count")
    figure.tight_layout()
    return _save_plot(figure, stem)


def _save_plot(figure: plt.Figure, stem: Path) -> list[Path]:
    png_path = Path(f"{stem}.png")
    pdf_path = Path(f"{stem}.pdf")
    figure.savefig(png_path)
    figure.savefig(pdf_path)
    plt.close(figure)
    return [png_path, pdf_path]


def _write_latex(frame: pd.DataFrame, path: Path) -> Path:
    columns = [
        column
        for column in [
            "scenario",
            "scenario_size",
            "method",
            "mode_profile",
            "social_welfare",
            "completion_rate",
            "drop_rate",
            "deadline_violation_rate",
            "p95_latency_ms",
            "total_decision_overhead_ms",
            "slot_share",
            "cluster_count",
            "avg_cluster_size",
            "graph_cut",
            "welfare_loss_vs_global",
        ]
        if column in frame.columns
    ]
    frame = frame[columns] if columns else frame
    latex = frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}")
    path.write_text(latex, encoding="utf-8")
    return path
