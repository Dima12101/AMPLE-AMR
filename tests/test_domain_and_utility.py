"""Tests for workload generation, domain ranges and utility behavior."""

from __future__ import annotations

import math

import numpy as np

from ample_amr.config import OperationModeConfig
from ample_amr.domain import EdgeNode, Robot, Task
from ample_amr.environment import SimulationEnvironment
from ample_amr.network import NetworkGraph
from ample_amr.utility import evaluate_utility
from ample_amr.workload import TaskGenerator


def test_task_generation_is_reproducible_with_seed(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    generator_a = TaskGenerator(
        robot_id="robot-00",
        task_classes=scenario.task_classes,
        task_mix=scenario.task_mix,
        arrival_rate_per_robot=1.0,
        rng=np.random.default_rng(123),
    )
    generator_b = TaskGenerator(
        robot_id="robot-00",
        task_classes=scenario.task_classes,
        task_mix=scenario.task_mix,
        arrival_rate_per_robot=1.0,
        rng=np.random.default_rng(123),
    )
    tasks_a = [generator_a.maybe_generate(now_ms=step * scenario.slot_duration_ms) for step in range(5)]
    tasks_b = [generator_b.maybe_generate(now_ms=step * scenario.slot_duration_ms) for step in range(5)]
    assert [task.task_class for task in tasks_a if task is not None] == [task.task_class for task in tasks_b if task is not None]
    assert [task.input_size_mbit for task in tasks_a if task is not None] == [task.input_size_mbit for task in tasks_b if task is not None]


def test_amr_speed_is_within_configured_warehouse_range(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=7)
    for robot in env.robots:
        assert scenario.robot_speed_range_mps[0] <= robot.max_speed_mps <= scenario.robot_speed_range_mps[1]
        assert scenario.robot_working_speed_range_mps[0] <= robot.working_speed_mps <= scenario.robot_working_speed_range_mps[1]


def test_utility_decreases_with_excessive_latency_for_sensitive_task(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    env = SimulationEnvironment(scenario, seed=3)
    graph: NetworkGraph = env.graph
    assert graph is not None
    robot = env.robots[0]
    node = env.nodes[0]
    low_latency_task = Task(
        id="t-low",
        robot_id=robot.id,
        task_class="obstacle_detection",
        arrival_time=0.0,
        input_size_mbit=0.2,
        output_size_mbit=0.02,
        cpu_required=0.2,
        memory_required=0.1,
        gpu_required=0.1,
        deadline_ms=80.0,
        priority=5,
        hard_deadline=False,
        metadata={"base_value": 14.0, "deadline_sensitivity": 2.0, "service_time_ms": 12.0},
    )
    high_latency_task = Task(
        id="t-high",
        robot_id=robot.id,
        task_class="obstacle_detection",
        arrival_time=0.0,
        input_size_mbit=15.0,
        output_size_mbit=2.0,
        cpu_required=0.2,
        memory_required=0.1,
        gpu_required=0.1,
        deadline_ms=80.0,
        priority=5,
        hard_deadline=False,
        metadata={"base_value": 14.0, "deadline_sensitivity": 2.0, "service_time_ms": 12.0},
    )
    low_utility = evaluate_utility(low_latency_task, robot, node, graph, t=0)
    high_utility = evaluate_utility(high_latency_task, robot, node, graph, t=0)
    assert low_utility > high_utility


def test_exposed_capacity_changes_correctly_by_operation_mode(experiment_config) -> None:
    scenario = experiment_config.resolve_scenario("stable_warehouse_load", size_override="Warehouse-S", quick=True)
    node = EdgeNode.from_class_config(
        node_id="node-test",
        node_class=scenario.edge_node_classes["Edge-S"],
        position=(0.0, 0.0),
        attached_access_point_id=None,
        operation_mode="safe",
    )
    node.current_operation_mode = "safe"
    safe_capacity = node.exposed_capacity(scenario.operation_modes)
    node.current_operation_mode = "normal"
    normal_capacity = node.exposed_capacity(scenario.operation_modes)
    node.current_operation_mode = "aggressive"
    aggressive_capacity = node.exposed_capacity(scenario.operation_modes)
    assert math.isclose(safe_capacity["cpu"], 0.30)
    assert math.isclose(normal_capacity["cpu"], 0.60)
    assert math.isclose(aggressive_capacity["cpu"], 0.90)
