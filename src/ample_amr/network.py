"""Wireless and edge connectivity model for the warehouse."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .config import WarehouseScenarioConfig
from .domain import AccessPoint, EdgeNode, Robot, Task
from .utils import clamp, safe_div


@dataclass
class LinkEstimate:
    """Estimated network characteristics for a robot-node pair."""

    latency_ms: float
    bandwidth_mbps: float
    feasible: bool
    degraded: bool
    network_cost: float


@dataclass
class NetworkGraph:
    """Warehouse network topology including APs, edge links and degradations."""

    scenario: WarehouseScenarioConfig
    access_points: list[AccessPoint]
    rng: np.random.Generator
    robot_to_ap: dict[str, str] = field(default_factory=dict)
    ap_loads: dict[str, int] = field(default_factory=dict)
    robot_interaction_frequency: dict[tuple[str, str], float] = field(default_factory=dict)

    def update_robot_assignments(self, robots: list[Robot]) -> None:
        """Update current AP assignments and AP load statistics."""

        self.robot_to_ap.clear()
        self.ap_loads = {ap.id: 0 for ap in self.access_points}
        for robot in robots:
            ap = self.nearest_access_point(robot.position)
            if ap is None:
                continue
            self.robot_to_ap[robot.id] = ap.id
            self.ap_loads[ap.id] += 1

    def update_interaction_frequency(self, robots: list[Robot]) -> None:
        """Accumulate pairwise interaction frequency based on spatial co-location."""

        for robot_u, robot_v in combinations(sorted(robots, key=lambda item: item.id), 2):
            same_ap = self.robot_to_ap.get(robot_u.id) == self.robot_to_ap.get(robot_v.id)
            distance = self.euclidean_distance(robot_u.position, robot_v.position)
            increment = 1.0 if same_ap else max(0.0, 1.0 - distance / max(self.scenario.width_m, self.scenario.height_m))
            key = (robot_u.id, robot_v.id)
            self.robot_interaction_frequency[key] = self.robot_interaction_frequency.get(key, 0.0) + increment

    def nearest_access_point(self, position: tuple[float, float]) -> AccessPoint | None:
        """Return the nearest AP that covers a given position."""

        best_ap: AccessPoint | None = None
        best_distance = float("inf")
        for ap in self.access_points:
            distance = self.euclidean_distance(position, ap.position)
            if distance <= ap.coverage_radius_m and distance < best_distance:
                best_ap = ap
                best_distance = distance
        return best_ap

    def estimate_link(
        self,
        robot: Robot,
        node: EdgeNode,
        task: Task,
        step: int,
    ) -> LinkEstimate:
        """Estimate latency, bandwidth and feasibility for a robot-node assignment."""

        robot_ap = self.nearest_access_point(robot.position)
        if robot_ap is None or node.attached_access_point_id is None:
            return LinkEstimate(0.0, 0.0, False, False, float("inf"))

        node_ap = next((ap for ap in self.access_points if ap.id == node.attached_access_point_id), None)
        if node_ap is None:
            return LinkEstimate(0.0, 0.0, False, False, float("inf"))

        robot_to_ap_distance = self.euclidean_distance(robot.position, robot_ap.position)
        ap_to_node_distance = self.euclidean_distance(node.position, node_ap.position)
        ap_to_ap_distance = self.euclidean_distance(robot_ap.position, node_ap.position)
        ap_load = self.ap_loads.get(robot_ap.id, 0)
        degraded = self.is_degraded_link(robot, node, step)

        if ap_to_ap_distance > max(self.scenario.width_m, self.scenario.height_m) * 0.85:
            return LinkEstimate(0.0, 0.0, False, degraded, float("inf"))

        if robot_ap.id == node_ap.id:
            base_latency = float(self.rng.uniform(*self.scenario.local_edge_latency_ms))
            base_bandwidth = float(self.rng.uniform(*self.scenario.high_speed_link_mbps))
        else:
            base_latency = float(self.rng.uniform(*self.scenario.local_edge_latency_ms)) + ap_to_ap_distance * 0.15
            base_bandwidth = float(self.rng.uniform(*self.scenario.constrained_link_mbps))

        congestion_penalty = 1.0 + 0.05 * max(0, ap_load - 1)
        latency_ms = base_latency + robot_to_ap_distance * 0.18 + ap_to_node_distance * 0.08
        latency_ms *= congestion_penalty
        bandwidth_mbps = max(1.0, base_bandwidth / congestion_penalty)

        if degraded:
            latency_ms += float(self.rng.uniform(*self.scenario.degraded_latency_ms))
            bandwidth_mbps *= 0.5

        transmission_ms = safe_div(task.total_data_mbit(), bandwidth_mbps) * 1000.0
        network_cost = latency_ms + transmission_ms * (1.0 + safe_div(robot.video_input_rate_mbps, 100.0))
        feasible = not node.failure_flag and bandwidth_mbps > 0.0
        return LinkEstimate(
            latency_ms=latency_ms,
            bandwidth_mbps=bandwidth_mbps,
            feasible=feasible,
            degraded=degraded,
            network_cost=network_cost,
        )

    def local_network_summary(self, node: EdgeNode, robots: list[Robot], step: int) -> dict[str, float]:
        """Aggregate network statistics for QMIX observations."""

        latencies: list[float] = []
        bandwidths: list[float] = []
        degraded_count = 0
        dummy_task = Task(
            id="summary",
            robot_id="summary",
            task_class="telemetry_diagnostics",
            arrival_time=0.0,
            input_size_mbit=0.1,
            output_size_mbit=0.02,
            cpu_required=0.05,
            memory_required=0.05,
            gpu_required=0.0,
            deadline_ms=2000.0,
            priority=1,
            hard_deadline=False,
        )
        for robot in robots:
            estimate = self.estimate_link(robot, node, dummy_task, step)
            if estimate.feasible:
                latencies.append(estimate.latency_ms)
                bandwidths.append(estimate.bandwidth_mbps)
                degraded_count += int(estimate.degraded)
        total = max(1, len(latencies))
        return {
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "avg_bandwidth_mbps": float(np.mean(bandwidths)) if bandwidths else 0.0,
            "degraded_link_fraction": degraded_count / total,
        }

    def robot_network_summary(self, robot: Robot, nodes: list[EdgeNode], step: int) -> dict[str, float]:
        """Compute a robot-side summary of current network quality."""

        dummy_task = Task(
            id="robot-summary",
            robot_id=robot.id,
            task_class="telemetry_diagnostics",
            arrival_time=0.0,
            input_size_mbit=0.1,
            output_size_mbit=0.02,
            cpu_required=0.05,
            memory_required=0.05,
            gpu_required=0.0,
            deadline_ms=2000.0,
            priority=1,
            hard_deadline=False,
        )
        estimates = [self.estimate_link(robot, node, dummy_task, step) for node in nodes]
        feasible = [estimate for estimate in estimates if estimate.feasible]
        return {
            "reachable_nodes": float(len(feasible)),
            "avg_latency_ms": float(np.mean([item.latency_ms for item in feasible])) if feasible else 0.0,
            "avg_bandwidth_mbps": float(np.mean([item.bandwidth_mbps for item in feasible])) if feasible else 0.0,
        }

    def affinity_weight(self, robot_u: Robot, robot_v: Robot, nodes: list[EdgeNode], step: int) -> float:
        """Compute the clustering affinity weight alpha_uv."""

        key = tuple(sorted((robot_u.id, robot_v.id)))
        interaction_frequency = self.robot_interaction_frequency.get(key, 0.0)
        average_network_cost = 0.0
        count = 0
        dummy_task = Task(
            id="affinity",
            robot_id=robot_u.id,
            task_class="telemetry_diagnostics",
            arrival_time=0.0,
            input_size_mbit=0.1,
            output_size_mbit=0.02,
            cpu_required=0.05,
            memory_required=0.05,
            gpu_required=0.0,
            deadline_ms=2000.0,
            priority=1,
            hard_deadline=False,
        )
        for node in nodes:
            estimate_u = self.estimate_link(robot_u, node, dummy_task, step)
            estimate_v = self.estimate_link(robot_v, node, dummy_task, step)
            if estimate_u.feasible and estimate_v.feasible:
                average_network_cost += 0.5 * (estimate_u.network_cost + estimate_v.network_cost)
                count += 1
        network_cost = safe_div(average_network_cost, count) if count else 1.0
        return interaction_frequency / (0.001 + network_cost)

    def is_degraded_link(self, robot: Robot, node: EdgeNode, step: int) -> bool:
        """Determine whether a link is affected by the scenario degradation event."""

        if not self.scenario.is_degraded_step(step):
            return False
        region_threshold = self.scenario.width_m * (1.0 - self.scenario.degraded_region_fraction)
        return robot.position[0] >= region_threshold or node.position[0] >= region_threshold

    @staticmethod
    def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        """Euclidean distance between two points."""

        return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def build_access_points(scenario: WarehouseScenarioConfig) -> list[AccessPoint]:
    """Create a regular AP grid for a warehouse size."""

    aps: list[AccessPoint] = []
    grid_cols = int(np.ceil(np.sqrt(scenario.access_points * scenario.width_m / scenario.height_m)))
    grid_rows = int(np.ceil(scenario.access_points / max(1, grid_cols)))
    spacing_x = scenario.width_m / max(1, grid_cols)
    spacing_y = scenario.height_m / max(1, grid_rows)
    coverage = clamp(max(spacing_x, spacing_y) * 0.9, 6.0, 24.0)
    index = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if index >= scenario.access_points:
                break
            aps.append(
                AccessPoint(
                    id=f"ap-{index:02d}",
                    position=((col + 0.5) * spacing_x, (row + 0.5) * spacing_y),
                    coverage_radius_m=coverage,
                )
            )
            index += 1
    return aps
