"""Tests for QMIX mode selection."""

from __future__ import annotations

from pathlib import Path

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
