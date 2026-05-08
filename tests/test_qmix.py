"""Tests for QMIX mode selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ample_amr.domain import Task
from ample_amr.runner import ExperimentRunner
from ample_amr.qmix import QMIXModeController


def test_qmix_action_selection_returns_valid_operation_modes(experiment_config, tmp_path: Path) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    checkpoint = tmp_path / "qmix_test.pt"
    controller = QMIXModeController(scenario, experiment_config.qmix, checkpoint, seed=13)
    env_observations, global_state = controller.scenario.edge_nodes, None
    del env_observations, global_state
    from ample_amr.environment import SimulationEnvironment

    env = SimulationEnvironment(scenario, seed=13)
    observations, state = env.reset(13)
    modes, _ = controller.select_modes(observations, state, [node.id for node in env.nodes], explore=False)
    assert set(modes).issubset({node.id for node in env.nodes})
    assert all(mode in scenario.operation_modes for mode in modes.values())


def test_qmix_training_scenarios_config_is_supported(experiment_config, config_path: Path) -> None:
    runner = ExperimentRunner(experiment_config, config_path)
    assert runner._training_scenario_names() == ["stable_warehouse_load"]

    fallback_config = replace(
        experiment_config,
        qmix=replace(experiment_config.qmix, training_scenarios=[]),
    )
    fallback_runner = ExperimentRunner(fallback_config, config_path)
    assert fallback_runner._training_scenario_names() == ["stable_warehouse_load"]


def test_qmix_still_controls_only_modes(experiment_config, tmp_path: Path) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    checkpoint = tmp_path / "qmix_modes_only.pt"
    controller = QMIXModeController(scenario, experiment_config.qmix, checkpoint, seed=21)

    from ample_amr.environment import SimulationEnvironment

    env = SimulationEnvironment(scenario, seed=21)
    observations, state = env.reset(21)
    task = Task(
        id="mode-only-check",
        robot_id=env.robots[0].id,
        task_class="telemetry_diagnostics",
        arrival_time=0.0,
        input_size_mbit=0.1,
        output_size_mbit=0.02,
        cpu_required=0.05,
        memory_required=0.05,
        gpu_required=0.0,
        deadline_ms=1000.0,
        priority=1,
        hard_deadline=False,
        assigned_node_id="node-keep",
        utility=3.5,
        cost=1.2,
        welfare_contribution=2.3,
        robot_payment=0.7,
        node_compensation=0.4,
        externality_estimate=0.6,
    )
    before = (
        task.assigned_node_id,
        task.utility,
        task.cost,
        task.welfare_contribution,
        task.robot_payment,
        task.node_compensation,
        task.externality_estimate,
    )
    modes, _ = controller.select_modes(observations, state, [node.id for node in env.nodes], explore=False)
    after = (
        task.assigned_node_id,
        task.utility,
        task.cost,
        task.welfare_contribution,
        task.robot_payment,
        task.node_compensation,
        task.externality_estimate,
    )
    assert set(modes) == {node.id for node in env.nodes}
    assert all(mode in scenario.operation_modes for mode in modes.values())
    assert after == before
