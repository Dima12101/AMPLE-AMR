"""Result export, plotting and LaTeX table generation."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_directory, safe_div


RESULT_FILES = {
    "raw_steps": "raw_steps.csv",
    "episodes": "episodes.csv",
    "summary_by_seed": "summary_by_seed.csv",
    "summary": "summary.csv",
    "scalability": "scalability.csv",
    "clustered_vs_global": "clustered_vs_global.csv",
    "sensitivity_operation_modes": "sensitivity_operation_modes.csv",
    "training_history": "training_history.csv",
    "ch8_overall_summary": "ch8_overall_summary.csv",
    "ch8_externality_diagnostics": "ch8_externality_diagnostics.csv",
    "ch8_payments_by_task_class": "ch8_payments_by_task_class.csv",
    "ch8_mode_distribution": "ch8_mode_distribution.csv",
    "ch8_hypothesis_checks": "ch8_hypothesis_checks.csv",
}

METHOD_ORDER = [
    "fixed_heuristic",
    "fixed_vcg",
    "qmix_heuristic",
    "ample_amr",
    "c_ample_amr",
    "random_modes_heuristic",
]

SCENARIO_ORDER = [
    "stable_warehouse_load",
    "peak_warehouse_load",
    "heterogeneous_edge_nodes",
    "network_degradation",
    "edge_node_failures",
    "scalability_sweep",
    "clustered_vs_global",
    "sensitivity_operation_modes",
]

CORE_CHAPTER_METHODS = [
    "fixed_heuristic",
    "fixed_vcg",
    "qmix_heuristic",
    "ample_amr",
    "c_ample_amr",
]

PRIMARY_CHAPTER_SCENARIOS = [
    "stable_warehouse_load",
    "peak_warehouse_load",
    "heterogeneous_edge_nodes",
    "network_degradation",
    "edge_node_failures",
]

SCALABILITY_CHAPTER_METHODS = ["ample_amr", "c_ample_amr"]
SENSITIVITY_CHAPTER_METHODS = ["fixed_heuristic", "qmix_heuristic", "ample_amr", "random_modes_heuristic"]
AUCTION_CHAPTER_METHODS = ["fixed_vcg", "ample_amr", "c_ample_amr"]
EXTERNALITY_CHAPTER_SCENARIOS = [
    "stable_warehouse_load",
    "peak_warehouse_load",
    "sensitivity_operation_modes",
]

METHOD_LABELS = {
    "fixed_heuristic": "Fixed-H",
    "fixed_vcg": "Fixed-VCG",
    "qmix_heuristic": "QMIX-H",
    "ample_amr": "AMPLE-AMR",
    "c_ample_amr": "C-AMPLE-AMR",
    "random_modes_heuristic": "Random-H",
}

SCENARIO_LABELS = {
    "stable_warehouse_load": "stable",
    "peak_warehouse_load": "peak",
    "heterogeneous_edge_nodes": "heterogeneous",
    "network_degradation": "degradation",
    "edge_node_failures": "failures",
    "scalability_sweep": "scalability",
    "clustered_vs_global": "clustered",
    "sensitivity_operation_modes": "sensitivity",
}

MODE_PROFILE_LABELS = {
    "default": "default",
    "conservative": "conservative",
    "aggressive_bias": "aggressive-bias",
}

MODE_ORDER = ["safe", "normal", "aggressive", "reserve", "critical"]

SUMMARY_METRICS = [
    "social_welfare",
    "completed_tasks",
    "generated_tasks",
    "completion_rate",
    "drop_rate",
    "deadline_violation_rate",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_node_utilization",
    "load_imbalance",
    "overload_count",
    "allocation_overhead_ms",
    "policy_inference_time_ms",
    "total_decision_overhead_ms",
    "slot_share",
    "payments_sum",
    "payment_robot_sum",
    "payment_task_sum",
    "node_compensation_sum",
    "average_externality",
    "externality_mean",
    "externality_p95",
    "externality_max",
    "externality_utilization_corr",
    "externality_latency_corr",
    "externality_queue_corr",
    "cluster_count",
    "avg_cluster_size",
    "graph_cut",
    "welfare_loss_vs_global",
    "relative_welfare_gain_vs_fixed_heuristic",
    "relative_welfare_gain_vs_fixed_vcg",
    "relative_overhead_reduction_c_ample_vs_ample",
]


def save_result_frames(
    output_dir: str | Path,
    raw_steps: pd.DataFrame,
    episodes: pd.DataFrame,
    training_history: pd.DataFrame | None = None,
    task_diagnostics: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write raw and chapter-8 specific result frames to disk."""

    result_dir = ensure_directory(output_dir)
    raw_steps.to_csv(result_dir / RESULT_FILES["raw_steps"], index=False)

    task_diagnostics = _normalize_task_diagnostics(task_diagnostics)
    task_diagnostics.to_csv(result_dir / RESULT_FILES["ch8_externality_diagnostics"], index=False)

    episode_task_metrics = _build_episode_task_metrics(task_diagnostics)
    if not episode_task_metrics.empty:
        episodes = episodes.merge(
            episode_task_metrics,
            on=["scenario", "scenario_size", "seed", "method", "mode_profile"],
            how="left",
        )
    else:
        for column in ("externality_utilization_corr", "externality_latency_corr", "externality_queue_corr"):
            episodes[column] = np.nan
    episodes = _add_relative_metrics(episodes)
    episodes = _add_cluster_relative_metrics(episodes)
    episodes.to_csv(result_dir / RESULT_FILES["episodes"], index=False)

    summary_by_seed = _aggregate_summary_by_seed(episodes)
    summary_by_seed = _add_relative_metrics(summary_by_seed)
    summary_by_seed = _add_cluster_relative_metrics(summary_by_seed)
    summary_by_seed.to_csv(result_dir / RESULT_FILES["summary_by_seed"], index=False)

    summary = _aggregate_summary(summary_by_seed)
    summary = _add_relative_metrics(summary)
    summary = _add_cluster_relative_metrics(summary)
    summary.to_csv(result_dir / RESULT_FILES["summary"], index=False)

    scalability = _ordered_frame(summary[summary["scenario"] == "scalability_sweep"].copy())
    scalability.to_csv(result_dir / RESULT_FILES["scalability"], index=False)

    clustered_vs_global = _build_clustered_vs_global(summary_by_seed)
    clustered_vs_global.to_csv(result_dir / RESULT_FILES["clustered_vs_global"], index=False)

    sensitivity = _ordered_frame(summary[summary["scenario"] == "sensitivity_operation_modes"].copy())
    sensitivity.to_csv(result_dir / RESULT_FILES["sensitivity_operation_modes"], index=False)

    ch8_mode_distribution = _build_mode_distribution(raw_steps)
    ch8_mode_distribution.to_csv(result_dir / RESULT_FILES["ch8_mode_distribution"], index=False)

    ch8_payments_by_task_class = _build_payments_by_task_class(task_diagnostics)
    ch8_payments_by_task_class.to_csv(result_dir / RESULT_FILES["ch8_payments_by_task_class"], index=False)

    ch8_overall_summary = _build_ch8_overall_summary(summary_by_seed)
    ch8_overall_summary.to_csv(result_dir / RESULT_FILES["ch8_overall_summary"], index=False)

    ch8_hypothesis_checks = _build_hypothesis_checks(summary_by_seed, task_diagnostics)
    ch8_hypothesis_checks.to_csv(result_dir / RESULT_FILES["ch8_hypothesis_checks"], index=False)

    if training_history is None:
        training_history = pd.DataFrame(
            columns=["size_name", "training_scenario", "episode", "reward", "loss", "epsilon"]
        )
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
        "ch8_overall_summary": result_dir / RESULT_FILES["ch8_overall_summary"],
        "ch8_externality_diagnostics": result_dir / RESULT_FILES["ch8_externality_diagnostics"],
        "ch8_payments_by_task_class": result_dir / RESULT_FILES["ch8_payments_by_task_class"],
        "ch8_mode_distribution": result_dir / RESULT_FILES["ch8_mode_distribution"],
        "ch8_hypothesis_checks": result_dir / RESULT_FILES["ch8_hypothesis_checks"],
    }


def generate_plots(output_dir: str | Path) -> list[Path]:
    """Generate all requested figures from saved CSV outputs."""

    result_dir = ensure_directory(output_dir)
    summary = pd.read_csv(result_dir / RESULT_FILES["summary"])
    raw_steps = pd.read_csv(result_dir / RESULT_FILES["raw_steps"])
    scalability = pd.read_csv(result_dir / RESULT_FILES["scalability"])
    clustered = pd.read_csv(result_dir / RESULT_FILES["clustered_vs_global"])
    training_history = pd.read_csv(result_dir / RESULT_FILES["training_history"])
    ch8_mode_distribution = pd.read_csv(result_dir / RESULT_FILES["ch8_mode_distribution"])
    ch8_task_diagnostics = pd.read_csv(result_dir / RESULT_FILES["ch8_externality_diagnostics"])
    ch8_payments_by_task_class = pd.read_csv(result_dir / RESULT_FILES["ch8_payments_by_task_class"])

    created: list[Path] = []
    created.extend(_plot_summary_metric(summary, "social_welfare", "Welfare by Scenario", "social_welfare", result_dir / "welfare_by_scenario"))
    created.extend(_plot_rate_triplet(summary, result_dir / "completion_drop_violation_by_scenario"))
    created.extend(_plot_summary_metric(summary, "p95_latency_ms", "P95 Latency by Scenario", "p95_latency_ms", result_dir / "latency_p95_by_scenario"))
    created.extend(_plot_scalability(scalability, result_dir / "overhead_scalability", "Overhead Scalability"))
    created.extend(_plot_cluster_tradeoff(clustered, result_dir / "global_vs_clustered_tradeoff", "Global vs Clustered Tradeoff"))
    created.extend(_plot_utilization_heatmap(summary, result_dir / "node_utilization_heatmap"))
    created.extend(_plot_learning_curves(training_history, result_dir / "learning_curves", "QMIX Reward", "QMIX Loss"))
    created.extend(_plot_mode_distribution(raw_steps, result_dir / "operation_mode_distribution", "Operation Mode Distribution"))

    chapter_image_dir = _resolve_dissertation_images_dir()
    created.extend(
        _plot_mode_distribution(
            ch8_mode_distribution,
            result_dir / "ch8_operation_mode_distribution",
            "Распределение режимов узлов",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_operation_mode_distribution"),
            already_aggregated=True,
        )
    )
    created.extend(
        _plot_learning_curves(
            training_history,
            result_dir / "ch8_learning_curves",
            "Вознаграждение QMIX",
            "Потери QMIX",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_learning_curves"),
        )
    )
    created.extend(
        _plot_single_scenario_metric(
            summary,
            "stable_warehouse_load",
            "social_welfare",
            "Общественное благосостояние при стабильной нагрузке",
            "Благосостояние",
            result_dir / "ch8_stable_welfare",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_stable_welfare"),
        )
    )
    created.extend(
        _plot_single_scenario_rates(
            summary,
            "stable_warehouse_load",
            result_dir / "ch8_stable_quality",
            "Качество обслуживания при стабильной нагрузке",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_stable_quality"),
        )
    )
    created.extend(
        _plot_single_scenario_metric(
            summary,
            "peak_warehouse_load",
            "p95_latency_ms",
            "95-й квантиль задержки при пиковой нагрузке",
            "Задержка p95, мс",
            result_dir / "ch8_peak_latency",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_peak_latency"),
        )
    )
    created.extend(
        _plot_single_scenario_rates(
            summary,
            "peak_warehouse_load",
            result_dir / "ch8_peak_rates",
            "Качество обслуживания при пиковой нагрузке",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_peak_rates"),
        )
    )
    created.extend(
        _plot_single_scenario_metric(
            summary,
            "heterogeneous_edge_nodes",
            "avg_node_utilization",
            "Загрузка узлов в разнородной инфраструктуре",
            "Средняя загрузка",
            result_dir / "ch8_node_utilization",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_node_utilization"),
        )
    )
    created.extend(
        _plot_single_scenario_metric(
            summary,
            "network_degradation",
            "social_welfare",
            "Общественное благосостояние при деградации связи",
            "Благосостояние",
            result_dir / "ch8_degradation_welfare",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_degradation_welfare"),
        )
    )
    created.extend(
        _plot_single_scenario_rates(
            summary,
            "edge_node_failures",
            result_dir / "ch8_failures_quality",
            "Качество обслуживания при отказах узлов",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_failures_quality"),
        )
    )
    created.extend(
        _plot_externality_vs_utilization(
            ch8_task_diagnostics,
            result_dir / "ch8_externality_vs_utilization",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_externality_vs_utilization"),
        )
    )
    created.extend(
        _plot_payments_by_task_class(
            ch8_payments_by_task_class,
            result_dir / "ch8_payments_by_task_class",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_payments_by_task_class"),
        )
    )
    created.extend(
        _plot_scalability(
            scalability,
            result_dir / "ch8_overhead_scalability",
            "Накладные расходы при масштабировании",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_overhead_scalability"),
        )
    )
    created.extend(
        _plot_cluster_tradeoff(
            clustered,
            result_dir / "ch8_clustered_tradeoff",
            "Компромисс кластеризации",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_clustered_tradeoff"),
        )
    )
    created.extend(
        _plot_sensitivity_modes(
            summary,
            result_dir / "ch8_sensitivity_modes",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_sensitivity_modes"),
        )
    )
    created.extend(
        _plot_summary_metric(
            summary,
            "social_welfare",
            "Общественное благосостояние по сценариям",
            "Благосостояние",
            result_dir / "ch8_welfare_by_scenario",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_welfare_by_scenario"),
        )
    )
    created.extend(
        _plot_rate_triplet(
            summary,
            result_dir / "ch8_quality_by_scenario",
            title="Качество обслуживания по сценариям",
            extra_stem=_chapter_stem(chapter_image_dir, "ch8_quality_by_scenario"),
        )
    )
    return created


def export_latex_tables(output_dir: str | Path, tables_dir: str | Path) -> list[Path]:
    """Export scenario-specific LaTeX tables."""

    result_dir = ensure_directory(output_dir)
    tables_path = ensure_directory(tables_dir)
    dissertation_tables_dir = _resolve_dissertation_tables_dir()

    summary = pd.read_csv(result_dir / RESULT_FILES["summary"])
    summary_by_seed = pd.read_csv(result_dir / RESULT_FILES["summary_by_seed"])
    scalability = pd.read_csv(result_dir / RESULT_FILES["scalability"])
    clustered = pd.read_csv(result_dir / RESULT_FILES["clustered_vs_global"])
    sensitivity = pd.read_csv(result_dir / RESULT_FILES["sensitivity_operation_modes"])
    ch8_task_diagnostics = pd.read_csv(result_dir / RESULT_FILES["ch8_externality_diagnostics"])
    ch8_overall_summary = pd.read_csv(result_dir / RESULT_FILES["ch8_overall_summary"])

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
        created.extend(_write_latex(frame, table_path, dissertation_tables_dir))

    created.extend(_write_latex(scalability.copy(), tables_path / "scalability_results.tex", dissertation_tables_dir))
    created.extend(_write_latex(clustered.copy(), tables_path / "clustered_results.tex", dissertation_tables_dir))
    created.extend(_write_latex(sensitivity.copy(), tables_path / "sensitivity_operation_modes.tex", dissertation_tables_dir))

    ch8_tables = {
        "ch8_overall_summary.tex": _build_overall_latex_frame(ch8_overall_summary),
        "ch8_stable_load_results.tex": _build_metric_table(summary_by_seed, "stable_warehouse_load"),
        "ch8_peak_load_results.tex": _build_metric_table(summary_by_seed, "peak_warehouse_load"),
        "ch8_heterogeneous_results.tex": _build_metric_table(summary_by_seed, "heterogeneous_edge_nodes"),
        "ch8_network_degradation_results.tex": _build_metric_table(summary_by_seed, "network_degradation"),
        "ch8_failures_results.tex": _build_metric_table(summary_by_seed, "edge_node_failures"),
        "ch8_scalability_results.tex": _build_scalability_latex_frame(summary_by_seed),
        "ch8_clustered_vs_global_results.tex": _build_clustered_latex_frame(clustered),
        "ch8_sensitivity_operation_modes.tex": _build_sensitivity_latex_frame(summary_by_seed),
        "ch8_externality_diagnostics.tex": _build_externality_latex_frame(ch8_task_diagnostics),
    }
    for filename, frame in ch8_tables.items():
        created.extend(_write_latex(frame, tables_path / filename, dissertation_tables_dir))
    return created


def summarize_generated_files(paths: list[Path]) -> str:
    """Build a concise text summary of generated files."""

    labels = [str(path) for path in sorted(paths)]
    return "\n".join(labels)


def _normalize_task_diagnostics(task_diagnostics: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "scenario",
        "scenario_size",
        "seed",
        "method",
        "mode_profile",
        "task_id",
        "robot_id",
        "task_class",
        "assigned_node_id",
        "task_payment",
        "task_externality",
        "approx_task_payment",
        "approx_task_externality",
        "task_payment_effective",
        "task_externality_effective",
        "task_payment_mode",
        "task_externality_mode",
        "task_utility",
        "task_cost",
        "net_contribution",
        "latency_ms",
        "deadline_ms",
        "deadline_violated",
        "node_utilization_at_assignment",
        "queue_length_at_assignment",
        "status",
    ]
    if task_diagnostics is None or task_diagnostics.empty:
        return pd.DataFrame(columns=columns)
    frame = task_diagnostics.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def _build_episode_task_metrics(task_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if task_diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "scenario_size",
                "seed",
                "method",
                "mode_profile",
                "externality_utilization_corr",
                "externality_latency_corr",
                "externality_queue_corr",
            ]
        )
    rows: list[dict[str, object]] = []
    group_columns = ["scenario", "scenario_size", "seed", "method", "mode_profile"]
    for keys, frame in task_diagnostics.groupby(group_columns, dropna=False):
        rows.append(
            {
                "scenario": keys[0],
                "scenario_size": keys[1],
                "seed": keys[2],
                "method": keys[3],
                "mode_profile": keys[4],
                "externality_utilization_corr": _corr(
                    frame["task_externality_effective"], frame["node_utilization_at_assignment"]
                ),
                "externality_latency_corr": _corr(frame["task_externality_effective"], frame["latency_ms"]),
                "externality_queue_corr": _corr(
                    frame["task_externality_effective"], frame["queue_length_at_assignment"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_summary_by_seed(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = (
        episodes.groupby(["scenario", "scenario_size", "method", "mode_profile", "seed"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    return _ordered_frame(frame)


def _aggregate_summary(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    aggregations = {metric: "mean" for metric in SUMMARY_METRICS if metric in summary_by_seed.columns}
    frame = (
        summary_by_seed.groupby(["scenario", "scenario_size", "method", "mode_profile"], dropna=False)
        .agg(aggregations)
        .reset_index()
    )
    return _ordered_frame(frame)


def _add_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "method" not in frame.columns:
        return frame
    key_columns = [column for column in ("scenario", "scenario_size", "seed", "mode_profile") if column in frame.columns]
    baseline_h = (
        frame[frame["method"] == "fixed_heuristic"][key_columns + ["social_welfare"]]
        .drop_duplicates()
        .rename(columns={"social_welfare": "fixed_heuristic_welfare"})
    )
    baseline_v = (
        frame[frame["method"] == "fixed_vcg"][key_columns + ["social_welfare"]]
        .drop_duplicates()
        .rename(columns={"social_welfare": "fixed_vcg_welfare"})
    )
    merged = frame.merge(baseline_h, on=key_columns, how="left").merge(baseline_v, on=key_columns, how="left")
    merged["relative_welfare_gain_vs_fixed_heuristic"] = merged.apply(
        lambda row: safe_div(row["social_welfare"] - row["fixed_heuristic_welfare"], abs(row["fixed_heuristic_welfare"]))
        if pd.notna(row["fixed_heuristic_welfare"])
        else np.nan,
        axis=1,
    )
    merged["relative_welfare_gain_vs_fixed_vcg"] = merged.apply(
        lambda row: safe_div(row["social_welfare"] - row["fixed_vcg_welfare"], abs(row["fixed_vcg_welfare"]))
        if pd.notna(row["fixed_vcg_welfare"])
        else np.nan,
        axis=1,
    )
    return merged.drop(columns=["fixed_heuristic_welfare", "fixed_vcg_welfare"])


def _add_cluster_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "method" not in frame.columns:
        return frame
    key_columns = [column for column in ("scenario", "scenario_size", "mode_profile", "seed") if column in frame.columns]
    ample = (
        frame[frame["method"] == "ample_amr"][key_columns + ["social_welfare", "total_decision_overhead_ms"]]
        .drop_duplicates()
        .rename(
            columns={
                "social_welfare": "ample_reference_welfare",
                "total_decision_overhead_ms": "ample_reference_overhead_ms",
            }
        )
    )
    merged = frame.merge(ample, on=key_columns, how="left")
    if "welfare_loss_vs_global" not in merged.columns:
        merged["welfare_loss_vs_global"] = 0.0
    if "relative_overhead_reduction_c_ample_vs_ample" not in merged.columns:
        merged["relative_overhead_reduction_c_ample_vs_ample"] = np.nan
    c_mask = merged["method"] == "c_ample_amr"
    merged.loc[c_mask, "welfare_loss_vs_global"] = (
        merged.loc[c_mask, "ample_reference_welfare"] - merged.loc[c_mask, "social_welfare"]
    ).clip(lower=0.0)
    merged.loc[c_mask, "relative_overhead_reduction_c_ample_vs_ample"] = merged.loc[c_mask].apply(
        lambda row: safe_div(
            row["ample_reference_overhead_ms"] - row["total_decision_overhead_ms"],
            abs(row["ample_reference_overhead_ms"]),
        )
        if pd.notna(row["ample_reference_overhead_ms"])
        else np.nan,
        axis=1,
    )
    merged.loc[merged["method"] != "c_ample_amr", "relative_overhead_reduction_c_ample_vs_ample"] = np.nan
    return merged.drop(columns=["ample_reference_welfare", "ample_reference_overhead_ms"])


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
                "relative_overhead_reduction_c_ample_vs_ample",
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
    frame = pd.DataFrame(
        {
            "scenario": merged["scenario"],
            "scenario_size": merged["scenario_size"],
            "seed": merged["seed"],
            "mode_profile": merged["mode_profile"],
            "ample_welfare": merged["social_welfare_ample"],
            "clustered_welfare": merged["social_welfare_clustered"],
            "welfare_loss_vs_global": (merged["social_welfare_ample"] - merged["social_welfare_clustered"]).clip(lower=0.0),
            "ample_overhead_ms": merged["total_decision_overhead_ms_ample"],
            "clustered_overhead_ms": merged["total_decision_overhead_ms_clustered"],
            "relative_overhead_reduction_c_ample_vs_ample": merged.apply(
                lambda row: safe_div(
                    row["total_decision_overhead_ms_ample"] - row["total_decision_overhead_ms_clustered"],
                    abs(row["total_decision_overhead_ms_ample"]),
                ),
                axis=1,
            ),
            "cluster_count": merged["cluster_count_clustered"],
            "avg_cluster_size": merged["avg_cluster_size_clustered"],
            "graph_cut": merged["graph_cut_clustered"],
        }
    )
    return _ordered_frame(frame)


def _build_mode_distribution(raw_steps: pd.DataFrame) -> pd.DataFrame:
    mode_columns = [column for column in raw_steps.columns if column.startswith("mode_") and column != "mode_profile"]
    ordered_mode_columns = [f"mode_{mode}" for mode in MODE_ORDER if f"mode_{mode}" in mode_columns]
    if raw_steps.empty or not ordered_mode_columns:
        return pd.DataFrame(columns=["scenario", "scenario_size", "method", "mode_profile"])
    frame = (
        raw_steps.groupby(["scenario", "scenario_size", "method", "mode_profile"], dropna=False)[ordered_mode_columns]
        .mean()
        .reset_index()
    )
    total = frame[ordered_mode_columns].sum(axis=1)
    for column in ordered_mode_columns:
        frame[f"{column}_share"] = np.where(total > 0.0, frame[column] / total, 0.0)
    return _ordered_frame(frame)


def _build_payments_by_task_class(task_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if task_diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "scenario_size",
                "method",
                "mode_profile",
                "task_class",
                "assigned_task_count",
                "total_task_payment",
                "mean_task_payment",
                "total_task_externality",
                "mean_task_externality",
            ]
        )
    frame = (
        task_diagnostics.groupby(
            ["scenario", "scenario_size", "method", "mode_profile", "task_class"],
            dropna=False,
        )
        .agg(
            assigned_task_count=("task_id", "count"),
            total_task_payment=("task_payment_effective", "sum"),
            mean_task_payment=("task_payment_effective", "mean"),
            total_task_externality=("task_externality_effective", "sum"),
            mean_task_externality=("task_externality_effective", "mean"),
        )
        .reset_index()
    )
    return _ordered_frame(frame)


def _build_ch8_overall_summary(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    if summary_by_seed.empty:
        return pd.DataFrame(columns=["scenario", "scenario_size", "method", "mode_profile"])
    metrics = [
        "social_welfare",
        "completion_rate",
        "drop_rate",
        "deadline_violation_rate",
        "p95_latency_ms",
        "avg_node_utilization",
        "total_decision_overhead_ms",
        "slot_share",
        "externality_mean",
        "payment_task_sum",
        "externality_utilization_corr",
        "externality_latency_corr",
        "externality_queue_corr",
        "relative_welfare_gain_vs_fixed_heuristic",
        "relative_welfare_gain_vs_fixed_vcg",
        "welfare_loss_vs_global",
        "relative_overhead_reduction_c_ample_vs_ample",
    ]
    available_metrics = [metric for metric in metrics if metric in summary_by_seed.columns]
    aggregations: dict[str, list[str]] = {metric: ["mean", "std"] for metric in available_metrics}
    frame = (
        summary_by_seed.groupby(["scenario", "scenario_size", "method", "mode_profile"], dropna=False)
        .agg(aggregations)
        .reset_index()
    )
    frame.columns = [
        column if isinstance(column, str) else "_".join([part for part in column if part]).strip("_")
        for column in frame.columns.to_flat_index()
    ]
    std_columns = [column for column in frame.columns if column.endswith("_std")]
    frame[std_columns] = frame[std_columns].fillna(0.0)
    return _ordered_frame(frame)


def _build_hypothesis_checks(summary_by_seed: pd.DataFrame, task_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _evaluate_h1(summary_by_seed),
        _evaluate_h2(summary_by_seed),
        _evaluate_h3(summary_by_seed),
        _evaluate_h4(task_diagnostics),
        _evaluate_h5(summary_by_seed),
        _evaluate_h6(summary_by_seed),
        _evaluate_h7(summary_by_seed),
    ]
    return pd.DataFrame(rows)


def _plot_summary_metric(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    stem: Path,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot = summary.pivot_table(index="scenario", columns="method", values=metric, aggfunc="mean", fill_value=0.0)
    pivot = _order_pivot(pivot)
    pivot.plot(kind="bar", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Сценарий")
    axis.set_ylabel(ylabel)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_rate_triplet(
    summary: pd.DataFrame,
    stem: Path,
    title: str = "Качество обслуживания по сценариям",
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = [
        ("completion_rate", "Доля завершения"),
        ("drop_rate", "Доля отклонения"),
        ("deadline_violation_rate", "Доля нарушений"),
    ]
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        pivot = summary.pivot_table(index="scenario", columns="method", values=metric, aggfunc="mean", fill_value=0.0)
        pivot = _order_pivot(pivot)
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(label)
        axis.set_xlabel("Сценарий")
        axis.set_ylabel("Доля")
        axis.legend([], [], frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_scalability(
    scalability: pd.DataFrame,
    stem: Path,
    title: str,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8, 5))
    if not scalability.empty:
        for method, frame in _groupby_method(scalability):
            ordered = frame.copy()
            ordered["scenario_size"] = pd.Categorical(
                ordered["scenario_size"],
                categories=["Warehouse-S", "Warehouse-M", "Warehouse-M+", "Warehouse-L", "Warehouse-XL"],
                ordered=True,
            )
            ordered = ordered.sort_values("scenario_size")
            axis.plot(ordered["scenario_size"], ordered["total_decision_overhead_ms"], marker="o", label=method)
    axis.set_title(title)
    axis.set_xlabel("Масштаб")
    axis.set_ylabel("Накладные расходы, мс")
    if axis.has_data():
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_cluster_tradeoff(
    clustered: pd.DataFrame,
    stem: Path,
    title: str,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8, 5))
    if not clustered.empty:
        axis.scatter(clustered["clustered_overhead_ms"], clustered["clustered_welfare"], label="C-AMPLE-AMR", alpha=0.8)
        axis.scatter(clustered["ample_overhead_ms"], clustered["ample_welfare"], label="AMPLE-AMR", alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel("Накладные расходы, мс")
    axis.set_ylabel("Благосостояние")
    if axis.has_data():
        axis.legend(loc="best")
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_utilization_heatmap(summary: pd.DataFrame, stem: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot = summary.pivot_table(index="scenario", columns="method", values="avg_node_utilization", aggfunc="mean", fill_value=0.0)
    pivot = _order_pivot(pivot)
    heatmap = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Тепловая карта загрузки узлов")
    figure.colorbar(heatmap, ax=axis)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem])


def _plot_learning_curves(
    training_history: pd.DataFrame,
    stem: Path,
    reward_title: str,
    loss_title: str,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not training_history.empty:
        for size_name, frame in training_history.groupby("size_name"):
            ordered = frame.sort_values("episode")
            axes[0].plot(ordered["episode"], ordered["reward"], label=size_name)
            axes[1].plot(ordered["episode"], ordered["loss"], label=size_name)
    axes[0].set_title(reward_title)
    axes[1].set_title(loss_title)
    axes[0].set_xlabel("Эпизод")
    axes[1].set_xlabel("Эпизод")
    axes[0].set_ylabel("Вознаграждение")
    axes[1].set_ylabel("Потери")
    for axis in axes:
        if axis.has_data():
            axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_mode_distribution(
    frame: pd.DataFrame,
    stem: Path,
    title: str,
    extra_stem: Path | None = None,
    already_aggregated: bool = False,
) -> list[Path]:
    mode_columns = [
        column
        for column in frame.columns
        if column.startswith("mode_") and column != "mode_profile" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if already_aggregated:
        mode_columns = [column for column in frame.columns if column.endswith("_share") and column.startswith("mode_")]
    figure, axis = plt.subplots(figsize=(12, 6))
    if not frame.empty and mode_columns:
        distribution = frame.copy()
        distribution["label"] = distribution["scenario"] + "\n" + distribution["method"]
        distribution = _ordered_frame(distribution).set_index("label")
        distribution[mode_columns].plot(kind="bar", stacked=True, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Сценарий / метод")
    axis.set_ylabel("Средняя доля режима" if already_aggregated else "Среднее число узлов")
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_single_scenario_metric(
    summary: pd.DataFrame,
    scenario_name: str,
    metric: str,
    title: str,
    ylabel: str,
    stem: Path,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8, 5))
    frame = _ordered_frame(summary[summary["scenario"] == scenario_name].copy())
    if not frame.empty:
        axis.bar(frame["method"], frame[metric])
    axis.set_title(title)
    axis.set_xlabel("Метод")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_single_scenario_rates(
    summary: pd.DataFrame,
    scenario_name: str,
    stem: Path,
    title: str,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    frame = _ordered_frame(summary[summary["scenario"] == scenario_name].copy())
    metrics = [
        ("completion_rate", "Завершение"),
        ("drop_rate", "Отклонение"),
        ("deadline_violation_rate", "Нарушения"),
    ]
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        if not frame.empty:
            axis.bar(frame["method"], frame[metric])
        axis.set_title(label)
        axis.set_ylabel("Доля")
        axis.tick_params(axis="x", rotation=30)
    figure.suptitle(title)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_externality_vs_utilization(
    task_diagnostics: pd.DataFrame,
    stem: Path,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8, 5))
    frame = task_diagnostics.dropna(subset=["task_externality_effective", "node_utilization_at_assignment"]).copy()
    if not frame.empty:
        for method, method_frame in _groupby_method(frame):
            axis.scatter(
                method_frame["node_utilization_at_assignment"],
                method_frame["task_externality_effective"],
                alpha=0.5,
                s=18,
                label=method,
            )
    axis.set_title("Внешний эффект и загрузка узлов")
    axis.set_xlabel("Загрузка узла")
    axis.set_ylabel("Внешний эффект")
    if axis.has_data():
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_payments_by_task_class(
    payments_by_task_class: pd.DataFrame,
    stem: Path,
    extra_stem: Path | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(12, 5))
    if not payments_by_task_class.empty:
        filtered = payments_by_task_class[payments_by_task_class["method"].isin(["fixed_vcg", "ample_amr", "c_ample_amr"])].copy()
        if not filtered.empty:
            pivot = filtered.pivot_table(index="task_class", columns="method", values="mean_task_payment", aggfunc="mean", fill_value=0.0)
            pivot = pivot.reindex(columns=_ordered_methods(list(pivot.columns)))
            if not pivot.empty and len(pivot.columns) > 0:
                pivot.plot(kind="bar", ax=axis)
    axis.set_title("Внутренние цены по классам задач")
    axis.set_xlabel("Класс задачи")
    axis.set_ylabel("Средний платёж")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _plot_sensitivity_modes(summary: pd.DataFrame, stem: Path, extra_stem: Path | None = None) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 5))
    frame = summary[summary["scenario"] == "sensitivity_operation_modes"].copy()
    if not frame.empty:
        pivot = frame.pivot_table(index="mode_profile", columns="method", values="social_welfare", aggfunc="mean", fill_value=0.0)
        pivot = pivot.reindex(columns=_ordered_methods(list(pivot.columns)))
        pivot.plot(kind="bar", ax=axis)
    axis.set_title("Чувствительность к параметрам режимов")
    axis.set_xlabel("Профиль режимов")
    axis.set_ylabel("Благосостояние")
    figure.tight_layout()
    return _save_plot_multi(figure, [stem, extra_stem] if extra_stem else [stem])


def _build_metric_table(summary_by_seed: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    frame = summary_by_seed[summary_by_seed["scenario"] == scenario_name].copy()
    table = _format_mean_std_table(
        frame,
        ["method", "mode_profile"],
        [
            "social_welfare",
            "completion_rate",
            "drop_rate",
            "deadline_violation_rate",
            "p95_latency_ms",
            "total_decision_overhead_ms",
            "slot_share",
        ],
    )
    return _prepare_chapter_table(
        table,
        {
            "method": "method",
            "mode_profile": "profile",
            "social_welfare": "welfare",
            "completion_rate": "completion",
            "drop_rate": "drop",
            "deadline_violation_rate": "deadline",
            "p95_latency_ms": "p95_ms",
            "total_decision_overhead_ms": "overhead_ms",
            "slot_share": "slot_share",
        },
    )


def _build_scalability_latex_frame(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    frame = summary_by_seed[
        (summary_by_seed["scenario"] == "scalability_sweep") & (summary_by_seed["method"].isin(SCALABILITY_CHAPTER_METHODS))
    ].copy()
    table = _format_mean_std_table(
        frame,
        ["scenario_size", "method"],
        ["social_welfare", "total_decision_overhead_ms", "slot_share"],
    )
    return _prepare_chapter_table(
        table,
        {
            "scenario_size": "size",
            "method": "method",
            "social_welfare": "welfare",
            "total_decision_overhead_ms": "overhead_ms",
            "slot_share": "slot_share",
        },
    )


def _build_clustered_latex_frame(clustered: pd.DataFrame) -> pd.DataFrame:
    if clustered.empty:
        return pd.DataFrame()
    table = _format_mean_std_table(
        clustered,
        ["scenario_size", "mode_profile"],
        [
            "cluster_count",
            "avg_cluster_size",
            "ample_overhead_ms",
            "clustered_overhead_ms",
            "ample_welfare",
            "clustered_welfare",
            "welfare_loss_vs_global",
            "relative_overhead_reduction_c_ample_vs_ample",
        ],
    )
    return _prepare_chapter_table(
        table,
        {
            "scenario_size": "size",
            "mode_profile": "profile",
            "cluster_count": "clusters",
            "avg_cluster_size": "avg_cluster",
            "ample_overhead_ms": "ample_overhead_ms",
            "clustered_overhead_ms": "clustered_overhead_ms",
            "ample_welfare": "ample_welfare",
            "clustered_welfare": "clustered_welfare",
            "welfare_loss_vs_global": "welfare_loss",
            "relative_overhead_reduction_c_ample_vs_ample": "overhead_reduction",
        },
    )


def _build_sensitivity_latex_frame(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    frame = summary_by_seed[
        (summary_by_seed["scenario"] == "sensitivity_operation_modes")
        & (summary_by_seed["method"].isin(SENSITIVITY_CHAPTER_METHODS))
    ].copy()
    table = _format_mean_std_table(
        frame,
        ["mode_profile", "method"],
        [
            "social_welfare",
            "completion_rate",
            "drop_rate",
            "p95_latency_ms",
            "total_decision_overhead_ms",
            "slot_share",
        ],
    )
    return _prepare_chapter_table(
        table,
        {
            "mode_profile": "profile",
            "method": "method",
            "social_welfare": "welfare",
            "completion_rate": "completion",
            "drop_rate": "drop",
            "p95_latency_ms": "p95_ms",
            "total_decision_overhead_ms": "overhead_ms",
            "slot_share": "slot_share",
        },
    )


def _build_externality_latex_frame(task_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if task_diagnostics.empty:
        return pd.DataFrame()
    filtered = task_diagnostics[
        task_diagnostics["scenario"].isin(EXTERNALITY_CHAPTER_SCENARIOS)
        & task_diagnostics["method"].isin(AUCTION_CHAPTER_METHODS)
    ].copy()
    if filtered.empty:
        return pd.DataFrame()
    frame = (
        filtered.groupby(["scenario", "method"], dropna=False)
        .agg(
            externality_mean=("task_externality_effective", "mean"),
            payment_task_sum=("task_payment_effective", "sum"),
            externality_utilization_corr=("task_externality_effective", lambda series: np.nan),
            externality_latency_corr=("task_externality_effective", lambda series: np.nan),
            externality_queue_corr=("task_externality_effective", lambda series: np.nan),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        subset = filtered[(filtered["scenario"] == row["scenario"]) & (filtered["method"] == row["method"])]
        rows.append(
            {
                "scenario": row["scenario"],
                "method": row["method"],
                "externality_mean": row["externality_mean"],
                "payment_task_sum": row["payment_task_sum"],
                "externality_utilization_corr": _corr(
                    subset["task_externality_effective"], subset["node_utilization_at_assignment"]
                ),
                "externality_latency_corr": _corr(subset["task_externality_effective"], subset["latency_ms"]),
                "externality_queue_corr": _corr(
                    subset["task_externality_effective"], subset["queue_length_at_assignment"]
                ),
            }
        )
    chapter = _prepare_chapter_table(
        _ordered_frame(pd.DataFrame(rows)),
        {
            "scenario": "scenario",
            "method": "method",
            "externality_mean": "ext_mean",
            "payment_task_sum": "task_payment_sum",
            "externality_utilization_corr": "corr_util",
            "externality_latency_corr": "corr_lat",
            "externality_queue_corr": "corr_queue",
        },
    )
    for column in ("ext_mean", "task_payment_sum", "corr_util", "corr_lat", "corr_queue"):
        if column in chapter.columns:
            chapter[column] = chapter[column].map(_format_compact_float)
    return chapter


def _build_overall_latex_frame(ch8_overall_summary: pd.DataFrame) -> pd.DataFrame:
    if ch8_overall_summary.empty:
        return pd.DataFrame()
    frame = ch8_overall_summary[
        ch8_overall_summary["scenario"].isin(PRIMARY_CHAPTER_SCENARIOS)
        & ch8_overall_summary["method"].isin(CORE_CHAPTER_METHODS)
    ].copy()
    table = pd.DataFrame(
        {
            "scenario": frame["scenario"],
            "scenario_size": frame["scenario_size"],
            "method": frame["method"],
            "social_welfare": _format_series_mean_std(frame, "social_welfare"),
            "completion_rate": _format_series_mean_std(frame, "completion_rate"),
            "drop_rate": _format_series_mean_std(frame, "drop_rate"),
            "deadline_violation_rate": _format_series_mean_std(frame, "deadline_violation_rate"),
            "p95_latency_ms": _format_series_mean_std(frame, "p95_latency_ms"),
            "avg_node_utilization": _format_series_mean_std(frame, "avg_node_utilization"),
            "total_decision_overhead_ms": _format_series_mean_std(frame, "total_decision_overhead_ms"),
        }
    )
    return _prepare_chapter_table(
        _ordered_frame(table),
        {
            "scenario": "scenario",
            "scenario_size": "size",
            "method": "method",
            "social_welfare": "welfare",
            "completion_rate": "completion",
            "drop_rate": "drop",
            "deadline_violation_rate": "deadline",
            "p95_latency_ms": "p95_ms",
            "avg_node_utilization": "util",
            "total_decision_overhead_ms": "overhead_ms",
        },
    )


def _write_latex(frame: pd.DataFrame, path: Path, dissertation_tables_dir: Path | None = None) -> list[Path]:
    if frame.empty:
        latex = pd.DataFrame().to_latex(index=False)
    else:
        latex = frame.to_latex(index=False, escape=True, na_rep="--")
    path.write_text(latex, encoding="utf-8")
    created = [path]
    if dissertation_tables_dir is not None:
        mirrored_path = ensure_directory(dissertation_tables_dir) / path.name
        mirrored_path.write_text(latex, encoding="utf-8")
        created.append(mirrored_path)
    return created


def _format_mean_std_table(frame: pd.DataFrame, key_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=key_columns + metric_columns)
    aggregations = {metric: ["mean", "std"] for metric in metric_columns if metric in frame.columns}
    summary = frame.groupby(key_columns, dropna=False).agg(aggregations).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join([part for part in column if part]).strip("_")
        for column in summary.columns.to_flat_index()
    ]
    data: dict[str, object] = {column: summary[column] for column in key_columns}
    for metric in metric_columns:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in summary.columns:
            data[metric] = [
                f"{mean:.3f} ± {std:.3f}"
                for mean, std in zip(summary[mean_col], summary.get(std_col, pd.Series([0.0] * len(summary))), strict=True)
            ]
    return _ordered_frame(pd.DataFrame(data))


def _format_series_mean_std(frame: pd.DataFrame, metric: str) -> pd.Series:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index)
    std_values = frame[std_col] if std_col in frame.columns else pd.Series([0.0] * len(frame))
    return pd.Series(
        [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(frame[mean_col], std_values, strict=True)],
        index=frame.index,
    )


def _prepare_chapter_table(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return frame.rename(columns=column_map)
    formatted = frame.copy()
    if "method" in formatted.columns:
        formatted["method"] = formatted["method"].map(METHOD_LABELS).fillna(formatted["method"])
    if "scenario" in formatted.columns:
        formatted["scenario"] = formatted["scenario"].map(SCENARIO_LABELS).fillna(formatted["scenario"])
    if "mode_profile" in formatted.columns:
        formatted["mode_profile"] = formatted["mode_profile"].map(MODE_PROFILE_LABELS).fillna(formatted["mode_profile"])
    return formatted.rename(columns=column_map)


def _format_compact_float(value: object) -> str:
    if pd.isna(value):
        return "--"
    numeric = float(value)
    if abs(numeric) < 1e-12:
        return "0"
    if abs(numeric) < 1e-3:
        return f"{numeric:.2e}"
    return f"{numeric:.3f}"


def _save_plot_multi(figure: plt.Figure, stems: list[Path | None]) -> list[Path]:
    created: list[Path] = []
    for stem in stems:
        if stem is None:
            continue
        ensure_directory(stem.parent)
        png_path = Path(f"{stem}.png")
        pdf_path = Path(f"{stem}.pdf")
        figure.savefig(png_path)
        figure.savefig(pdf_path)
        created.extend([png_path, pdf_path])
    plt.close(figure)
    return created


def _order_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    ordered_index = [scenario for scenario in SCENARIO_ORDER if scenario in pivot.index]
    remaining_index = [index for index in pivot.index if index not in ordered_index]
    ordered_columns = _ordered_methods(list(pivot.columns))
    return pivot.reindex(index=ordered_index + remaining_index, columns=ordered_columns)


def _ordered_methods(methods: list[str]) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in methods]
    return ordered + [method for method in methods if method not in ordered]


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.copy()
    if "scenario" in ordered.columns:
        ordered["scenario"] = pd.Categorical(ordered["scenario"], categories=SCENARIO_ORDER, ordered=True)
    if "method" in ordered.columns:
        ordered["method"] = pd.Categorical(ordered["method"], categories=METHOD_ORDER, ordered=True)
    sort_columns = [column for column in ("scenario", "scenario_size", "method", "mode_profile", "seed", "task_class") if column in ordered.columns]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns)
    if "scenario" in ordered.columns:
        ordered["scenario"] = ordered["scenario"].astype(str)
    if "method" in ordered.columns:
        ordered["method"] = ordered["method"].astype(str)
    return ordered.reset_index(drop=True)


def _groupby_method(frame: pd.DataFrame):
    available_methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    for method in available_methods:
        method_frame = frame[frame["method"] == method].copy()
        if not method_frame.empty:
            yield method, method_frame


def _corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 2:
        return float("nan")
    if pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    return float(pair["left"].corr(pair["right"]))


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if pd.notna(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _mean_effect(
    summary_by_seed: pd.DataFrame,
    left_method: str,
    right_method: str,
    metric: str,
    scenarios: list[str] | None = None,
) -> tuple[float, float, int]:
    frame = summary_by_seed.copy()
    if scenarios is not None:
        frame = frame[frame["scenario"].isin(scenarios)]
    key_columns = [column for column in ("scenario", "scenario_size", "seed", "mode_profile") if column in frame.columns]
    left = frame[frame["method"] == left_method][key_columns + [metric]].rename(columns={metric: f"{metric}_left"})
    right = frame[frame["method"] == right_method][key_columns + [metric]].rename(columns={metric: f"{metric}_right"})
    merged = left.merge(right, on=key_columns, how="inner")
    if merged.empty:
        return float("nan"), float("nan"), 0
    effect = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    return float(effect.mean()), float((effect > 0.0).mean()), len(effect)


def _lower_is_better_effect(
    summary_by_seed: pd.DataFrame,
    left_method: str,
    right_method: str,
    metric: str,
    scenarios: list[str] | None = None,
) -> tuple[float, float, int]:
    effect, positive_share, count = _mean_effect(summary_by_seed, right_method, left_method, metric, scenarios=scenarios)
    return effect, positive_share, count


def _status_from_effect(mean_effect: float, positive_share: float, strong_threshold: float = 0.0) -> str:
    if np.isnan(mean_effect):
        return "inconclusive"
    if mean_effect > strong_threshold and positive_share >= 0.7:
        return "confirmed"
    if mean_effect >= strong_threshold and positive_share >= 0.5:
        return "partially_confirmed"
    if positive_share < 0.4:
        return "not_confirmed"
    return "inconclusive"


def _evaluate_h1(summary_by_seed: pd.DataFrame) -> dict[str, object]:
    effect, share, count = _mean_effect(summary_by_seed, "fixed_vcg", "fixed_heuristic", "social_welfare")
    return {
        "hypothesis": "H1",
        "status": _status_from_effect(effect, share),
        "main_metric": "social_welfare",
        "comparison": "fixed_vcg vs fixed_heuristic",
        "observed_effect": effect,
        "comment": f"Положительный эффект в {share:.1%} сопоставлений, n={count}.",
    }


def _evaluate_h2(summary_by_seed: pd.DataFrame) -> dict[str, object]:
    effect, share, count = _mean_effect(summary_by_seed, "qmix_heuristic", "fixed_heuristic", "social_welfare")
    return {
        "hypothesis": "H2",
        "status": _status_from_effect(effect, share),
        "main_metric": "social_welfare",
        "comparison": "qmix_heuristic vs fixed_heuristic",
        "observed_effect": effect,
        "comment": f"Положительный эффект в {share:.1%} сопоставлений, n={count}.",
    }


def _evaluate_h3(summary_by_seed: pd.DataFrame) -> dict[str, object]:
    effect_vcg, share_vcg, _ = _mean_effect(summary_by_seed, "ample_amr", "fixed_vcg", "social_welfare")
    effect_qmix, share_qmix, _ = _mean_effect(summary_by_seed, "ample_amr", "qmix_heuristic", "social_welfare")
    effect = _nanmean([effect_vcg, effect_qmix])
    share = _nanmean([share_vcg, share_qmix])
    return {
        "hypothesis": "H3",
        "status": _status_from_effect(effect, share),
        "main_metric": "social_welfare",
        "comparison": "ample_amr vs {fixed_vcg, qmix_heuristic}",
        "observed_effect": effect,
        "comment": f"Средний выигрыш против одиночных компонентов; доля положительных сравнений {share:.1%}.",
    }


def _evaluate_h4(task_diagnostics: pd.DataFrame) -> dict[str, object]:
    if task_diagnostics.empty:
        return {
            "hypothesis": "H4",
            "status": "inconclusive",
            "main_metric": "externality_utilization_corr",
            "comparison": "auction methods / peak vs stable",
            "observed_effect": np.nan,
            "comment": "Нет task-level diagnostics.",
        }
    auction = task_diagnostics[task_diagnostics["method"].isin(["fixed_vcg", "ample_amr", "c_ample_amr"])].copy()
    corr = _corr(auction["task_externality_effective"], auction["node_utilization_at_assignment"])
    peak = float(
        auction[auction["scenario"] == "peak_warehouse_load"]["task_externality_effective"].mean()
    ) if not auction.empty else float("nan")
    stable = float(
        auction[auction["scenario"] == "stable_warehouse_load"]["task_externality_effective"].mean()
    ) if not auction.empty else float("nan")
    if np.isnan(corr):
        status = "inconclusive"
    elif corr > 0.15 and (np.isnan(peak) or np.isnan(stable) or peak >= stable):
        status = "confirmed"
    elif corr > 0.0:
        status = "partially_confirmed"
    else:
        status = "not_confirmed"
    return {
        "hypothesis": "H4",
        "status": status,
        "main_metric": "externality_utilization_corr",
        "comparison": "auction methods / peak vs stable",
        "observed_effect": corr,
        "comment": f"Корреляция внешнего эффекта с загрузкой {corr:.3f}; peak={peak:.3f}, stable={stable:.3f}.",
    }


def _evaluate_h5(summary_by_seed: pd.DataFrame) -> dict[str, object]:
    scenarios = ["peak_warehouse_load"]
    welfare, welfare_share, _ = _mean_effect(summary_by_seed, "ample_amr", "fixed_heuristic", "social_welfare", scenarios)
    drop_gain, drop_share, _ = _lower_is_better_effect(summary_by_seed, "ample_amr", "fixed_heuristic", "drop_rate", scenarios)
    latency_gain, latency_share, _ = _lower_is_better_effect(summary_by_seed, "ample_amr", "fixed_heuristic", "p95_latency_ms", scenarios)
    score = sum(
        [
            welfare > 0.0 and welfare_share >= 0.5,
            drop_gain > 0.0 and drop_share >= 0.5,
            latency_gain > 0.0 and latency_share >= 0.5,
        ]
    )
    status = "confirmed" if score >= 3 else "partially_confirmed" if score >= 2 else "not_confirmed"
    return {
        "hypothesis": "H5",
        "status": status,
        "main_metric": "drop_rate / p95_latency_ms / social_welfare",
        "comparison": "ample_amr vs fixed_heuristic in peak_warehouse_load",
        "observed_effect": _nanmean([welfare, drop_gain, latency_gain]),
        "comment": f"Улучшения: welfare={welfare:.3f}, drop={drop_gain:.3f}, latency={latency_gain:.3f}.",
    }


def _evaluate_h6(summary_by_seed: pd.DataFrame) -> dict[str, object]:
    scenarios = ["network_degradation", "edge_node_failures"]
    welfare, welfare_share, _ = _mean_effect(summary_by_seed, "ample_amr", "fixed_heuristic", "social_welfare", scenarios)
    completion, completion_share, _ = _mean_effect(summary_by_seed, "ample_amr", "fixed_heuristic", "completion_rate", scenarios)
    latency_gain, latency_share, _ = _lower_is_better_effect(summary_by_seed, "ample_amr", "fixed_heuristic", "p95_latency_ms", scenarios)
    score = sum(
        [
            welfare > 0.0 and welfare_share >= 0.5,
            completion > 0.0 and completion_share >= 0.5,
            latency_gain > 0.0 and latency_share >= 0.5,
        ]
    )
    status = "confirmed" if score >= 3 else "partially_confirmed" if score >= 2 else "not_confirmed"
    return {
        "hypothesis": "H6",
        "status": status,
        "main_metric": "social_welfare / completion_rate / p95_latency_ms",
        "comparison": "ample_amr vs fixed_heuristic in degradation/failures",
        "observed_effect": _nanmean([welfare, completion, latency_gain]),
        "comment": f"Улучшения: welfare={welfare:.3f}, completion={completion:.3f}, latency={latency_gain:.3f}.",
    }


def _evaluate_h7(summary_by_seed: pd.DataFrame) -> dict[str, object]:
    c_frame = _add_cluster_relative_metrics(summary_by_seed.copy())
    c_frame = c_frame[
        (c_frame["method"] == "c_ample_amr")
        & (c_frame["scenario"].isin(["scalability_sweep", "clustered_vs_global"]))
    ].copy()
    if c_frame.empty:
        return {
            "hypothesis": "H7",
            "status": "inconclusive",
            "main_metric": "relative_overhead_reduction_c_ample_vs_ample",
            "comparison": "c_ample_amr vs ample_amr",
            "observed_effect": np.nan,
            "comment": "Нет сопоставимых clustered результатов.",
        }
    overhead_reduction = float(c_frame["relative_overhead_reduction_c_ample_vs_ample"].mean())
    welfare_loss = float(c_frame["welfare_loss_vs_global"].mean())
    status = "confirmed" if overhead_reduction > 0.0 and welfare_loss >= 0.0 else "partially_confirmed"
    if overhead_reduction <= 0.0:
        status = "not_confirmed"
    return {
        "hypothesis": "H7",
        "status": status,
        "main_metric": "relative_overhead_reduction_c_ample_vs_ample",
        "comparison": "c_ample_amr vs ample_amr",
        "observed_effect": overhead_reduction,
        "comment": f"Среднее относительное снижение overhead={overhead_reduction:.3f}, средняя потеря welfare={welfare_loss:.3f}.",
    }


def _resolve_dissertation_root() -> Path | None:
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "диссертация" / "SPbU-Phd-LaTeX-Dissertation"
        if candidate.exists():
            return candidate
    return None


def _resolve_dissertation_images_dir() -> Path | None:
    dissertation_root = _resolve_dissertation_root()
    if dissertation_root is None:
        return None
    return ensure_directory(dissertation_root / "Dissertation" / "images" / "ample_amr")


def _resolve_dissertation_tables_dir() -> Path | None:
    dissertation_root = _resolve_dissertation_root()
    if dissertation_root is None:
        return None
    return ensure_directory(dissertation_root / "Dissertation" / "tables")


def _chapter_stem(chapter_dir: Path | None, name: str) -> Path | None:
    if chapter_dir is None:
        return None
    return chapter_dir / name
