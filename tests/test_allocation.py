"""Tests for allocation feasibility and pricing behavior."""

from __future__ import annotations

from ample_amr.allocation import HeuristicAllocator, VCGLikeAllocator, build_candidates, build_node_budgets
from ample_amr.environment import SimulationEnvironment
from ample_amr.methods import METHOD_SPECS
from ample_amr.utility import estimate_network_time, estimate_processing_time
from ample_amr.domain import Task


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


def test_no_method_uses_qmix_without_allocation_layer() -> None:
    for method_name, spec in METHOD_SPECS.items():
        if spec.uses_qmix:
            assert spec.allocator_key in {"heuristic", "vcg_like", "clustered_vcg_like"}


def test_c_ample_amr_does_not_require_cluster_leader_selection() -> None:
    spec = METHOD_SPECS["c_ample_amr"]
    assert spec.allocator_key == "clustered_vcg_like"
