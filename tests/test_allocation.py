"""Tests for allocation feasibility and pricing behavior."""

from __future__ import annotations

import math

import pytest

import ample_amr.allocation as allocation_module
from ample_amr.allocation import HeuristicAllocator, VCGLikeAllocator, build_candidates, build_node_budgets
from ample_amr.domain import Task
from ample_amr.environment import SimulationEnvironment
from ample_amr.methods import METHOD_SPECS
from ample_amr.utility import estimate_network_time, estimate_processing_time


def _build_long_task(task_id: str, robot_id: str) -> Task:
    return Task(
        id=task_id,
        robot_id=robot_id,
        task_class="inventory_recognition",
        arrival_time=0.0,
        input_size_mbit=1.0,
        output_size_mbit=0.1,
        cpu_required=0.5,
        memory_required=0.2,
        gpu_required=0.0,
        deadline_ms=2000.0,
        priority=2,
        hard_deadline=False,
        metadata={"base_value": 9.0, "deadline_sensitivity": 0.6, "service_time_ms": 500.0},
    )


def test_hard_deadline_task_cannot_be_assigned_when_latency_exceeds_deadline(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("network_degradation", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=11)
    env.current_step = scenario.degradation_start_step
    env.current_time_ms = env.current_step * scenario.slot_duration_ms
    env._refresh_node_statistics()
    robot = max(env.robots, key=lambda item: item.position[0])
    task = Task(
        id="hard-deadline",
        robot_id=robot.id,
        task_class="obstacle_detection",
        arrival_time=env.current_time_ms,
        input_size_mbit=12.0,
        output_size_mbit=1.0,
        cpu_required=0.2,
        memory_required=0.1,
        gpu_required=0.1,
        deadline_ms=5.0,
        priority=5,
        hard_deadline=True,
        metadata={"base_value": 14.0, "deadline_sensitivity": 2.0, "service_time_ms": 25.0},
    )
    result = HeuristicAllocator().allocate(
        tasks=[task],
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=env.current_step,
    )
    assert task.id not in result.task_to_node


def test_heuristic_allocation_respects_capacity_constraints(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=2)
    env.nodes = [env.nodes[0]]
    env._refresh_node_statistics()
    tasks = [_build_long_task("task-a", env.robots[0].id), _build_long_task("task-b", env.robots[1].id)]
    result = HeuristicAllocator().allocate(
        tasks=tasks,
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=0,
    )
    budgets = build_node_budgets(env.nodes, scenario)
    candidates, _ = build_candidates(tasks, {robot.id: robot for robot in env.robots}, env.nodes, env.graph, scenario, 0)
    assigned = [candidate for task_id, task_candidates in candidates.items() for candidate in task_candidates if result.task_to_node.get(task_id) == candidate.node_id]
    assert sum(candidate.effective_cpu for candidate in assigned) <= budgets[env.nodes[0].id].cpu + 1e-9
    assert sum(candidate.effective_memory for candidate in assigned) <= budgets[env.nodes[0].id].memory + 1e-9
    assert sum(candidate.effective_gpu for candidate in assigned) <= budgets[env.nodes[0].id].gpu + 1e-9


def test_heuristic_assignment_ignores_welfare_during_selection(experiment_config, monkeypatch) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=3)
    env.nodes = [env.nodes[0]]
    env._refresh_node_statistics()
    task = _build_long_task("task-a", env.robots[0].id)

    monkeypatch.setattr(allocation_module, "evaluate_utility", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(allocation_module, "evaluate_cost", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(allocation_module, "estimate_network_time", lambda *args, **kwargs: 5.0)
    monkeypatch.setattr(allocation_module, "estimate_processing_time", lambda *args, **kwargs: 5.0)

    result = HeuristicAllocator().allocate(
        tasks=[task],
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=0,
    )

    assert result.task_to_node == {task.id: env.nodes[0].id}
    assert result.social_welfare == pytest.approx(-1.0)
    assert task.welfare_contribution == pytest.approx(-1.0)


def test_vcg_like_allocation_respects_capacity_constraints(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=5)
    env.nodes = [env.nodes[0]]
    env._refresh_node_statistics()
    tasks = [_build_long_task("task-a", env.robots[0].id), _build_long_task("task-b", env.robots[1].id)]
    result = VCGLikeAllocator().allocate(
        tasks=tasks,
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=0,
    )
    budgets = build_node_budgets(env.nodes, scenario)
    candidates, _ = build_candidates(tasks, {robot.id: robot for robot in env.robots}, env.nodes, env.graph, scenario, 0)
    assigned = [candidate for task_id, task_candidates in candidates.items() for candidate in task_candidates if result.task_to_node.get(task_id) == candidate.node_id]
    assert sum(candidate.effective_cpu for candidate in assigned) <= budgets[env.nodes[0].id].cpu + 1e-9
    assert sum(candidate.effective_memory for candidate in assigned) <= budgets[env.nodes[0].id].memory + 1e-9
    assert sum(candidate.effective_gpu for candidate in assigned) <= budgets[env.nodes[0].id].gpu + 1e-9


def test_vcg_like_payments_are_nonnegative(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=9)
    env._refresh_node_statistics()
    tasks = [_build_long_task("task-a", env.robots[0].id), _build_long_task("task-b", env.robots[1].id)]
    result = VCGLikeAllocator().allocate(
        tasks=tasks,
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=0,
    )
    assert all(payment >= 0.0 for payment in result.payments_by_robot.values())


def test_task_externality_payments_are_recorded_for_vcg(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=17)
    env._refresh_node_statistics()
    tasks = [_build_long_task("task-a", env.robots[0].id), _build_long_task("task-b", env.robots[1].id)]
    result = VCGLikeAllocator().allocate(
        tasks=tasks,
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=0,
    )
    step_row = env._build_step_row(tasks, [], result)
    assert set(result.payments_by_task) == {task.id for task in tasks}
    assert set(result.externality_by_task) == {task.id for task in tasks}
    assigned_task_ids = set(result.task_to_node)
    assert all(result.payments_by_task[task_id] >= 0.0 for task_id in assigned_task_ids)
    assert all(result.externality_by_task[task_id] >= 0.0 for task_id in assigned_task_ids)
    assert step_row["payment_task_sum"] == pytest.approx(sum(result.payments_by_task.values()))
    for task in tasks:
        if task.id in assigned_task_ids:
            assert task.externality_estimate == pytest.approx(result.externality_by_task[task.id])


def test_vcg_like_allocator_falls_back_to_approx_task_externalities_when_exact_mode_is_disabled(
    experiment_config,
    monkeypatch,
) -> None:
    monkeypatch.setattr(allocation_module, "EXACT_TASK_EXTERNALITY_MAX_TASKS", 1)
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=23)
    env._refresh_node_statistics()
    tasks = [_build_long_task(f"task-{index}", env.robots[index].id) for index in range(2)]
    result = VCGLikeAllocator().allocate(
        tasks=tasks,
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=0,
    )
    assert result.auction_stats["task_payment_mode"] == "approx_proportional_robot_payment"
    assert result.auction_stats["task_externality_mode"] == "approx_proportional_robot_payment"
    for task in tasks:
        if task.id not in result.task_to_node:
            continue
        assert task.metadata["task_payment_mode"] == "approx_proportional_robot_payment"
        assert task.metadata["task_externality_mode"] == "approx_proportional_robot_payment"
        assert math.isnan(task.metadata["task_payment"])
        assert math.isnan(task.metadata["task_externality"])
        assert task.metadata["approx_task_payment"] >= 0.0
        assert task.metadata["approx_task_externality"] >= 0.0
        assert task.externality_estimate == pytest.approx(task.metadata["task_externality_effective"])


def test_vcg_like_allocator_can_skip_pricing_during_training(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=29)
    env._refresh_node_statistics()
    tasks = [_build_long_task("task-a", env.robots[0].id), _build_long_task("task-b", env.robots[1].id)]
    result = VCGLikeAllocator().allocate(
        tasks=tasks,
        robots={robot.id: robot for robot in env.robots},
        nodes=env.nodes,
        graph=env.graph,
        scenario=scenario,
        step=0,
        compute_pricing=False,
    )
    assert result.auction_stats["robot_payment_mode"] == "skipped_for_training"
    assert result.auction_stats["task_payment_mode"] == "skipped_for_training"
    assert result.auction_stats["task_externality_mode"] == "skipped_for_training"
    assert set(result.task_to_node).issubset({task.id for task in tasks})
    assert all(payment == 0.0 for payment in result.payments_by_robot.values())
    assert all(payment == 0.0 for payment in result.payments_by_task.values())
    assert all(externality == 0.0 for externality in result.externality_by_task.values())


def test_no_method_uses_qmix_without_allocation_layer() -> None:
    for method_name, spec in METHOD_SPECS.items():
        if spec.uses_qmix:
            assert spec.allocator_key in {"heuristic", "vcg_like", "clustered_vcg_like"}


def test_c_ample_amr_does_not_require_cluster_leader_selection() -> None:
    spec = METHOD_SPECS["c_ample_amr"]
    assert spec.allocator_key == "clustered_vcg_like"
