"""Utility, cost and latency estimation functions."""

from __future__ import annotations

import math

from .domain import EdgeNode, Robot, Task
from .network import NetworkGraph
from .utils import clamp, safe_div


def estimate_network_time(robot: Robot, node: EdgeNode, task: Task, graph: NetworkGraph, t: int) -> float:
    """Estimate end-to-end network transfer and propagation time in milliseconds."""

    estimate = graph.estimate_link(robot, node, task, t)
    if not estimate.feasible:
        return float("inf")
    transmission_ms = safe_div(task.total_data_mbit(), estimate.bandwidth_mbps) * 1000.0
    return estimate.latency_ms + transmission_ms


def estimate_queue_time(node: EdgeNode) -> float:
    """Estimate queue waiting time using the latest cached backlog statistic."""

    return float(node.last_auction_stats.get("estimated_backlog_ms", 0.0))


def estimate_processing_time(task: Task, node: EdgeNode) -> float:
    """Estimate compute time for a task on a node under the current operation mode."""

    exposed = node.last_auction_stats.get("exposed_capacity", None)
    if exposed is None:
        exposed_cpu = node.cpu_capacity_norm * 0.6
        exposed_memory = node.memory_capacity_norm * 0.6
        exposed_gpu = node.gpu_capacity_norm * 0.6
    else:
        exposed_cpu = max(float(exposed["cpu"]), 0.05)
        exposed_memory = max(float(exposed["memory"]), 0.05)
        exposed_gpu = max(float(exposed["gpu"]), 0.05)
    weighted_demand = (
        task.cpu_required / exposed_cpu
        + 0.35 * task.memory_required / exposed_memory
        + 0.85 * task.gpu_required / exposed_gpu
    )
    service_base = float(task.metadata.get("service_time_ms", 30.0))
    return max(1.0, service_base * weighted_demand)


def evaluate_utility(task: Task, robot: Robot, node: EdgeNode, graph: NetworkGraph, t: int) -> float:
    """Estimate task utility for a robot-node assignment."""

    network_time = estimate_network_time(robot, node, task, graph, t)
    if math.isinf(network_time):
        return float("-inf")
    queue_time = estimate_queue_time(node)
    processing_time = estimate_processing_time(task, node)
    total_latency = network_time + queue_time + processing_time
    if task.hard_deadline and total_latency > task.deadline_ms:
        return float("-inf")

    base_value = float(task.metadata.get("base_value", 1.0))
    deadline_sensitivity = float(task.metadata.get("deadline_sensitivity", 1.0))
    priority_weight = 1.0 + 0.18 * task.priority
    mission_weight = robot.mission_importance
    latency_factor = math.exp(-deadline_sensitivity * total_latency / max(task.deadline_ms, 1.0))
    network_penalty = 0.08 * network_time + 0.04 * task.total_data_mbit()
    mode_boost = 1.0
    if task.task_class in node.last_auction_stats.get("priority_boost_task_classes", ()):
        mode_boost = float(node.last_auction_stats.get("priority_boost", 1.0))

    utility = base_value * priority_weight * mission_weight * latency_factor * mode_boost - network_penalty
    if total_latency > task.deadline_ms and not task.hard_deadline:
        excess_ratio = (total_latency - task.deadline_ms) / max(task.deadline_ms, 1.0)
        utility -= base_value * clamp(excess_ratio, 0.0, 2.5)
    return utility


def evaluate_cost(task: Task, robot: Robot, node: EdgeNode, graph: NetworkGraph, t: int) -> float:
    """Estimate execution and network cost of an assignment."""

    network_time = estimate_network_time(robot, node, task, graph, t)
    if math.isinf(network_time):
        return float("inf")
    processing_time = estimate_processing_time(task, node)
    mode_exposure = node.last_auction_stats.get("mode_exposure_scalar", 0.6)
    compute_cost = (
        1.8 * task.cpu_required
        + 0.8 * task.memory_required
        + 2.4 * task.gpu_required
    ) * (processing_time / 100.0)
    energy_penalty = (0.7 + mode_exposure) * (task.total_data_mbit() / 10.0)
    urgency_cost = 0.2 * safe_div(network_time + processing_time, max(task.deadline_ms, 1.0))
    return compute_cost + energy_penalty + urgency_cost


def compute_social_welfare(tasks: list[Task]) -> float:
    """Compute the total social welfare of an allocation."""

    return float(sum(task.welfare_contribution for task in tasks if task.assigned_node_id))
