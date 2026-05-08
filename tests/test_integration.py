"""Lightweight integration tests for quick experiment runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ample_amr.runner import ExperimentRunner


def test_warehouse_s_quick_benchmark_produces_expected_csv_outputs(
    experiment_config,
    config_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = ExperimentRunner(experiment_config, config_path)
    runner.run(
        methods=["fixed_heuristic", "fixed_vcg", "qmix_heuristic", "ample_amr", "c_ample_amr"],
        scenarios=["stable_warehouse_load"],
        seeds=[0],
        scenario_size="Warehouse-S",
        quick=True,
    )
    expected = [
        tmp_path / "experiments/results/raw_steps.csv",
        tmp_path / "experiments/results/episodes.csv",
        tmp_path / "experiments/results/summary_by_seed.csv",
        tmp_path / "experiments/results/summary.csv",
        tmp_path / "experiments/results/scalability.csv",
        tmp_path / "experiments/results/clustered_vs_global.csv",
        tmp_path / "experiments/results/sensitivity_operation_modes.csv",
    ]
    for path in expected:
        assert path.exists(), str(path)
    raw_steps = pd.read_csv(tmp_path / "experiments/results/raw_steps.csv")
    episodes = pd.read_csv(tmp_path / "experiments/results/episodes.csv")
    summary = pd.read_csv(tmp_path / "experiments/results/summary.csv")
    required_columns = {
        "externality_mean",
        "externality_p95",
        "externality_max",
        "payment_robot_sum",
        "payment_task_sum",
    }
    assert required_columns.issubset(raw_steps.columns)
    assert required_columns.issubset(episodes.columns)
    assert required_columns.issubset(summary.columns)


def test_scalability_sweep_produces_all_scenario_sizes(
    experiment_config,
    config_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = ExperimentRunner(experiment_config, config_path)
    runner.run(
        methods=["fixed_heuristic"],
        scenarios=["scalability_sweep"],
        seeds=[0],
        quick=True,
    )
    frame = pd.read_csv(tmp_path / "experiments/results/scalability.csv")
    assert set(frame["scenario_size"]) == {"Warehouse-S", "Warehouse-M", "Warehouse-M+", "Warehouse-L", "Warehouse-XL"}
