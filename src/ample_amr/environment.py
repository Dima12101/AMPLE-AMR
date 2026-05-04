"""Simulation environment for warehouse AMR and edge node experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .allocation import AllocationResult, BaseAllocator
from .config import WarehouseScenarioConfig
from .domain import AccessPoint, EdgeNode, NodeReservation, Robot, Task, default_queue_distribution
from .network import NetworkGraph, build_access_points
from .utility import estimate_processing_time
from .utils import clamp, percentile, safe_div
from .workload import TaskGenerator


class ModePolicy(Protocol):
    """Protocol implemented by mode controllers."""

    def select_modes(
        self,
        observations: list[list[float]],
        global_state: list[float],
        node_ids: list[str],
        explore: bool = False,
    ) -> tuple[dict[str, str], float]:
        """Select operation modes for edge nodes."""


@dataclass
class EnvironmentTransition:
    """Transition returned by a simulation step."""

    next_observations: list[list[float]]
    next_global_state: list[float]
    reward: float
    done: bool
    step_row: dict[str, object]
    allocation_result: AllocationResult


class SimulationEnvironment:
    """Warehouse simulation environment with AMRs, APs, nodes and workloads."""

    def __init__(self, scenario: WarehouseScenarioConfig, seed: int) -> None:
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.access_points: list[AccessPoint] = []
        self.robots: list[Robot] = []
        self.nodes: list[EdgeNode] = []
        self.graph: NetworkGraph | None = None
        self.tasks_by_id: dict[str, Task] = {}
        self.generated_task_ids: list[str] = []
        self.completed_task_ids: list[str] = []
        self.dropped_task_ids: list[str] = []
        self.violated_task_ids: list[str] = []
        self.failed_node_ids: list[str] = []
        self.current_step = 0
        self.current_time_ms = 0.0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> tuple[list[list[float]], list[float]]:
        """Reset the simulation state and return initial observations."""

        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.current_step = 0
        self.current_time_ms = 0.0
        self.tasks_by_id.clear()
        self.generated_task_ids.clear()
        self.completed_task_ids.clear()
        self.dropped_task_ids.clear()
        self.violated_task_ids.clear()
        self.access_points = build_access_points(self.scenario)
        self.nodes = self._build_nodes()
        self.robots = self._build_robots()
        self.graph = NetworkGraph(self.scenario, self.access_points, np.random.default_rng(self.seed + 1))
        self.graph.update_robot_assignments(self.robots)
        self.graph.update_interaction_frequency(self.robots)
        self._refresh_node_statistics()
        self._refresh_robot_summaries()
        observations = self.build_local_observations()
        return observations, self.build_global_state(observations)

    def step(
        self,
        mode_selection: dict[str, str],
        allocator: BaseAllocator,
        policy_time_ms: float = 0.0,
        global_reference_welfare: float | None = None,
    ) -> EnvironmentTransition:
        """Advance the environment by one decision slot."""

        self._apply_failure_events()
        finalized_tasks = self._finalize_completed_tasks(self.current_time_ms)
        self._move_robots()
        assert self.graph is not None
        self.graph.update_robot_assignments(self.robots)
        self.graph.update_interaction_frequency(self.robots)
        self._refresh_robot_summaries()
        self._apply_modes(mode_selection)
        self._refresh_node_statistics()
        new_tasks = self._generate_tasks()
        allocation_result = allocator.allocate(
            tasks=new_tasks,
            robots={robot.id: robot for robot in self.robots},
            nodes=self.nodes,
            graph=self.graph,
            scenario=self.scenario,
            step=self.current_step,
            policy_time_ms=policy_time_ms,
            global_reference_welfare=global_reference_welfare,
        )
        self._schedule_allocated_tasks(new_tasks, allocation_result)
        self._refresh_node_statistics()
        self._update_auction_feedback(new_tasks, allocation_result)
        step_row = self._build_step_row(new_tasks, finalized_tasks, allocation_result)
        reward = float(step_row["social_welfare"]) - 0.7 * float(step_row["drop_rate"]) - 0.5 * float(
            step_row["deadline_violation_rate"]
        ) - 0.1 * float(step_row["overload_count"])
        self.current_step += 1
        self.current_time_ms += self.scenario.slot_duration_ms
        self._refresh_node_statistics()
        next_observations = self.build_local_observations()
        done = self.current_step >= self.scenario.episode_steps
        return EnvironmentTransition(
            next_observations=next_observations,
            next_global_state=self.build_global_state(next_observations),
            reward=reward,
            done=done,
            step_row=step_row,
            allocation_result=allocation_result,
        )

    def run_episode(
        self,
        mode_policy: ModePolicy,
        allocator: BaseAllocator,
        explore: bool = False,
        global_reference_episode_welfare: float | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Run a complete episode under a mode policy and allocator."""

        observations, global_state = self.reset(self.seed)
        step_rows: list[dict[str, object]] = []
        for _ in range(self.scenario.episode_steps):
            node_ids = [node.id for node in self.nodes]
            modes, policy_time_ms = mode_policy.select_modes(observations, global_state, node_ids, explore=explore)
            transition = self.step(
                mode_selection=modes,
                allocator=allocator,
                policy_time_ms=policy_time_ms,
                global_reference_welfare=global_reference_episode_welfare,
            )
            step_rows.append(transition.step_row)
            observations = transition.next_observations
            global_state = transition.next_global_state
            if transition.done:
                break
        self._drain_remaining_work()
        episode_summary = self._build_episode_summary(step_rows)
        return step_rows, episode_summary

    def build_local_observations(self) -> list[list[float]]:
        """Build local observations for all nodes."""

        observations: list[list[float]] = []
        assert self.graph is not None
        for node in self.nodes:
            queued_tasks = [self.tasks_by_id[task_id] for task_id in node.queue if task_id in self.tasks_by_id]
            queue_distribution = default_queue_distribution(queued_tasks)
            network_summary = self.graph.local_network_summary(node, self.robots, self.current_step)
            observations.append(node.local_observation_builder(self.scenario.operation_modes, queue_distribution, network_summary))
        return observations

    def build_global_state(self, observations: list[list[float]]) -> list[float]:
        """Build a centralized global state for QMIX training."""

        node_utilizations = [
            max(node.utilization(self.scenario.operation_modes).values(), default=0.0)
            for node in self.nodes
        ]
        pending_tasks = [
            task for task in self.tasks_by_id.values() if task.status in {"generated", "queued", "running"}
        ]
        state = [
            float(self.current_step),
            float(self.current_time_ms),
            float(len(self.robots)),
            float(len(self.nodes)),
            float(len(pending_tasks)),
            float(np.mean(node_utilizations)) if node_utilizations else 0.0,
            float(np.std(node_utilizations)) if node_utilizations else 0.0,
            safe_div(len(self.completed_task_ids), max(1, len(self.generated_task_ids))),
            safe_div(len(self.dropped_task_ids), max(1, len(self.generated_task_ids))),
        ]
        for observation in observations:
            state.extend(observation)
        return state

    def _build_nodes(self) -> list[EdgeNode]:
        mix = self._expand_node_mix()
        nodes: list[EdgeNode] = []
        ap_indices = np.linspace(0, max(0, len(self.access_points) - 1), self.scenario.edge_nodes, dtype=int)
        for index in range(self.scenario.edge_nodes):
            ap = self.access_points[int(ap_indices[index])]
            node_class_name = mix[index]
            node_class = self.scenario.edge_node_classes[node_class_name]
            offset_x = float(self.rng.uniform(-1.5, 1.5))
            offset_y = float(self.rng.uniform(-1.5, 1.5))
            node = EdgeNode.from_class_config(
                node_id=f"node-{index:02d}",
                node_class=node_class,
                position=(clamp(ap.position[0] + offset_x, 0.0, self.scenario.width_m), clamp(ap.position[1] + offset_y, 0.0, self.scenario.height_m)),
                attached_access_point_id=ap.id,
                operation_mode=self.scenario.fixed_operation_mode,
            )
            nodes.append(node)
        self.failed_node_ids = [
            node.id
            for node in sorted(nodes, key=lambda item: item.id)[: int(round(self.scenario.failure_fraction * len(nodes)))]
        ]
        return nodes

    def _build_robots(self) -> list[Robot]:
        robots: list[Robot] = []
        for index in range(self.scenario.robots):
            position = (
                float(self.rng.uniform(0.0, self.scenario.width_m)),
                float(self.rng.uniform(0.0, self.scenario.height_m)),
            )
            heading = float(self.rng.uniform(0.0, 2.0 * np.pi))
            max_speed = float(self.rng.uniform(*self.scenario.robot_speed_range_mps))
            working_speed = float(self.rng.uniform(*self.scenario.robot_working_speed_range_mps))
            velocity = (working_speed * float(np.cos(heading)), working_speed * float(np.sin(heading)))
            robot_rng = np.random.default_rng(self.seed * 100 + index)
            robot = Robot(
                id=f"robot-{index:02d}",
                position=position,
                velocity=velocity,
                max_speed_mps=max_speed,
                working_speed_mps=working_speed,
                payload_kg=float(self.rng.uniform(*self.scenario.robot_payload_range_kg)),
                mission_importance=float(self.rng.uniform(*self.scenario.mission_importance_range)),
                task_generator=TaskGenerator(
                    robot_id=f"robot-{index:02d}",
                    task_classes=self.scenario.task_classes,
                    task_mix=self.scenario.task_mix,
                    arrival_rate_per_robot=self.scenario.arrival_rate_per_robot,
                    rng=robot_rng,
                ),
                video_input_rate_mbps=float(self.rng.uniform(*self.scenario.video_like_input_rate_mbps)),
            )
            robots.append(robot)
        return robots

    def _expand_node_mix(self) -> list[str]:
        node_types = sorted(self.scenario.heterogeneous_node_mix)
        weights = np.asarray([self.scenario.heterogeneous_node_mix[name] for name in node_types], dtype=float)
        weights = weights / weights.sum()
        counts = [int(np.floor(weight * self.scenario.edge_nodes)) for weight in weights]
        while sum(counts) < self.scenario.edge_nodes:
            index = int(np.argmax(weights - np.asarray(counts) / max(1, self.scenario.edge_nodes)))
            counts[index] += 1
        mix: list[str] = []
        for node_type, count in zip(node_types, counts, strict=True):
            mix.extend([node_type] * count)
        mix = mix[: self.scenario.edge_nodes]
        mix.sort()
        return mix

    def _move_robots(self) -> None:
        slot_seconds = self.scenario.slot_duration_ms / 1000.0
        for robot in self.robots:
            if float(self.rng.random()) < 0.15:
                heading = float(self.rng.uniform(0.0, 2.0 * np.pi))
                speed = min(robot.max_speed_mps, robot.working_speed_mps * float(self.rng.uniform(0.8, 1.1)))
                robot.velocity = (speed * float(np.cos(heading)), speed * float(np.sin(heading)))
            new_position = (
                clamp(robot.position[0] + robot.velocity[0] * slot_seconds, 0.0, self.scenario.width_m),
                clamp(robot.position[1] + robot.velocity[1] * slot_seconds, 0.0, self.scenario.height_m),
            )
            if new_position[0] in {0.0, self.scenario.width_m}:
                robot.velocity = (-robot.velocity[0], robot.velocity[1])
            if new_position[1] in {0.0, self.scenario.height_m}:
                robot.velocity = (robot.velocity[0], -robot.velocity[1])
            robot.position = new_position

    def _generate_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        arrival_multiplier = self.scenario.arrival_rate_multiplier(self.current_step)
        for robot in self.robots:
            task = robot.task_generator.maybe_generate(self.current_time_ms, arrival_multiplier)
            if task is None:
                continue
            self.tasks_by_id[task.id] = task
            self.generated_task_ids.append(task.id)
            tasks.append(task)
        return tasks

    def _apply_modes(self, mode_selection: dict[str, str]) -> None:
        for node in self.nodes:
            node.previous_operation_mode = node.current_operation_mode
            node.current_operation_mode = mode_selection.get(node.id, node.current_operation_mode)

    def _apply_failure_events(self) -> None:
        failure_active = self.scenario.is_failure_step(self.current_step)
        for node in self.nodes:
            should_fail = failure_active and node.id in self.failed_node_ids
            if should_fail and not node.failure_flag:
                node.failure_flag = True
                dropped = node.clear_all_work(self.tasks_by_id, self.current_time_ms)
                self.dropped_task_ids.extend(task.id for task in dropped)
            elif not should_fail and node.failure_flag and self.scenario.enable_recovery:
                node.failure_flag = False

    def _finalize_completed_tasks(self, now_ms: float) -> list[Task]:
        finalized: list[Task] = []
        for node in self.nodes:
            finalized.extend(node.cleanup_finished(self.tasks_by_id, now_ms))
        for task in finalized:
            if task.id in self.completed_task_ids or task.id in self.violated_task_ids:
                continue
            latency = (task.finish_time or now_ms) - task.arrival_time
            if latency > task.deadline_ms and not task.hard_deadline:
                task.status = "violated"
                self.violated_task_ids.append(task.id)
            elif task.status != "dropped":
                task.status = "completed"
                self.completed_task_ids.append(task.id)
        return finalized

    def _schedule_allocated_tasks(self, tasks: list[Task], allocation_result: AllocationResult) -> None:
        node_by_id = {node.id: node for node in self.nodes}
        for task in tasks:
            if task.id not in allocation_result.task_to_node:
                task.status = "dropped"
                task.finish_time = self.current_time_ms
                self.dropped_task_ids.append(task.id)
                continue
            node = node_by_id[allocation_result.task_to_node[task.id]]
            exposed = node.exposed_capacity(self.scenario.operation_modes)
            if (
                task.cpu_required > exposed["cpu"] + 1e-9
                or task.memory_required > exposed["memory"] + 1e-9
                or task.gpu_required > exposed["gpu"] + 1e-9
            ):
                task.status = "dropped"
                task.finish_time = self.current_time_ms
                self.dropped_task_ids.append(task.id)
                continue
            processing_time = estimate_processing_time(task, node)
            start_time = self._find_earliest_start(node, task, processing_time)
            finish_time = start_time + processing_time
            task.assigned_node_id = node.id
            task.start_time = start_time
            task.finish_time = finish_time
            task.status = "running" if start_time <= self.current_time_ms else "queued"
            node.reservations.append(
                NodeReservation(
                    task_id=task.id,
                    start_time=start_time,
                    finish_time=finish_time,
                    cpu_required=task.cpu_required,
                    memory_required=task.memory_required,
                    gpu_required=task.gpu_required,
                )
            )
            node.reservations.sort(key=lambda reservation: (reservation.start_time, reservation.finish_time, reservation.task_id))
            node.update_loads(self.tasks_by_id, self.current_time_ms)

    def _find_earliest_start(self, node: EdgeNode, task: Task, processing_time_ms: float) -> float:
        exposed = node.exposed_capacity(self.scenario.operation_modes)
        candidate_times = {self.current_time_ms}
        for reservation in node.reservations:
            if reservation.finish_time >= self.current_time_ms:
                candidate_times.add(reservation.start_time)
                candidate_times.add(reservation.finish_time)
        sorted_times = sorted(candidate_times)
        if not sorted_times:
            sorted_times = [self.current_time_ms]
        final_fallback = max([self.current_time_ms] + [reservation.finish_time for reservation in node.reservations])
        for candidate_start in sorted_times + [final_fallback]:
            candidate_end = candidate_start + processing_time_ms
            checkpoints = {candidate_start, candidate_end}
            for reservation in node.reservations:
                if reservation.finish_time <= candidate_start or reservation.start_time >= candidate_end:
                    continue
                checkpoints.add(reservation.start_time)
                checkpoints.add(reservation.finish_time)
            feasible = True
            for left, right in zip(sorted(checkpoints)[:-1], sorted(checkpoints)[1:], strict=True):
                midpoint = (left + right) / 2.0
                cpu_load = task.cpu_required
                memory_load = task.memory_required
                gpu_load = task.gpu_required
                for reservation in node.reservations:
                    if reservation.start_time <= midpoint < reservation.finish_time:
                        cpu_load += reservation.cpu_required
                        memory_load += reservation.memory_required
                        gpu_load += reservation.gpu_required
                if cpu_load > exposed["cpu"] + 1e-9 or memory_load > exposed["memory"] + 1e-9 or gpu_load > exposed["gpu"] + 1e-9:
                    feasible = False
                    break
            if feasible:
                return candidate_start
        return final_fallback

    def _refresh_node_statistics(self) -> None:
        for node in self.nodes:
            node.update_loads(self.tasks_by_id, self.current_time_ms)
            exposed = node.exposed_capacity(self.scenario.operation_modes)
            utilization = node.utilization(self.scenario.operation_modes)
            backlog_ms = sum(
                max(0.0, reservation.finish_time - max(self.current_time_ms, reservation.start_time))
                for reservation in node.reservations
                if reservation.finish_time > self.current_time_ms
            )
            mode_config = self.scenario.operation_modes[node.current_operation_mode]
            node.last_auction_stats.update(
                {
                    "estimated_backlog_ms": backlog_ms,
                    "exposed_capacity": exposed,
                    "mode_exposure_scalar": (mode_config.cpu_exposure + mode_config.memory_exposure + mode_config.gpu_exposure) / 3.0,
                    "priority_boost_task_classes": mode_config.priority_boost_task_classes,
                    "priority_boost": mode_config.priority_boost,
                    "exposed_capacity_utilization": float(np.mean(list(utilization.values()))),
                    "scarcity_metric": max(utilization.values()) + 0.1 * len(node.queue),
                    "current_time_ms": self.current_time_ms,
                }
            )

    def _refresh_robot_summaries(self) -> None:
        assert self.graph is not None
        for robot in self.robots:
            robot.local_network_state_summary = self.graph.robot_network_summary(robot, self.nodes, self.current_step)

    def _update_auction_feedback(self, tasks: list[Task], allocation_result: AllocationResult) -> None:
        tasks_by_node: dict[str, list[Task]] = {}
        for task in tasks:
            if task.assigned_node_id:
                tasks_by_node.setdefault(task.assigned_node_id, []).append(task)
        for node in self.nodes:
            node_tasks = tasks_by_node.get(node.id, [])
            node.last_auction_stats.update(
                {
                    "allocated_task_count": float(len(node_tasks)),
                    "average_welfare_contribution": float(np.mean([task.welfare_contribution for task in node_tasks])) if node_tasks else 0.0,
                    "average_payment": float(np.mean([task.robot_payment for task in node_tasks])) if node_tasks else 0.0,
                    "average_externality": float(np.mean([task.externality_estimate for task in node_tasks])) if node_tasks else 0.0,
                }
            )

    def _build_step_row(
        self,
        new_tasks: list[Task],
        finalized_tasks: list[Task],
        allocation_result: AllocationResult,
    ) -> dict[str, object]:
        latencies = [
            (task.finish_time or self.current_time_ms) - task.arrival_time
            for task in finalized_tasks
            if task.status in {"completed", "violated"}
        ]
        node_total_utilization = [
            float(np.mean(list(node.utilization(self.scenario.operation_modes).values())))
            for node in self.nodes
        ]
        overload_count = sum(1 for value in node_total_utilization if value > 1.0)
        payments_sum = sum(allocation_result.payments_by_robot.values())
        node_compensation_sum = sum(allocation_result.compensation_by_node.values())
        mode_counts = {
            f"mode_{mode_name}": float(sum(1 for node in self.nodes if node.current_operation_mode == mode_name))
            for mode_name in self.scenario.operation_modes
        }
        step_row = {
            "scenario": self.scenario.scenario_name,
            "scenario_size": self.scenario.size_name,
            "seed": self.seed,
            "step": self.current_step,
            "social_welfare": allocation_result.social_welfare,
            "completed_tasks": sum(1 for task in finalized_tasks if task.status == "completed"),
            "generated_tasks": len(new_tasks),
            "completion_rate": safe_div(sum(1 for task in finalized_tasks if task.status == "completed"), max(1, len(new_tasks))),
            "drop_rate": safe_div(sum(1 for task in new_tasks if task.status == "dropped"), max(1, len(new_tasks))),
            "deadline_violation_rate": safe_div(sum(1 for task in finalized_tasks if task.status == "violated"), max(1, len(new_tasks))),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": percentile(latencies, 95.0),
            "avg_node_utilization": float(np.mean(node_total_utilization)) if node_total_utilization else 0.0,
            "load_imbalance": float(np.std(node_total_utilization)) if node_total_utilization else 0.0,
            "overload_count": overload_count,
            "allocation_overhead_ms": allocation_result.allocation_time_ms,
            "policy_inference_time_ms": allocation_result.policy_time_ms,
            "total_decision_overhead_ms": allocation_result.total_decision_time_ms,
            "slot_share": allocation_result.slot_share,
            "payments_sum": payments_sum,
            "node_compensation_sum": node_compensation_sum,
            "average_externality": safe_div(
                sum(task.externality_estimate for task in new_tasks if task.assigned_node_id),
                max(1, sum(1 for task in new_tasks if task.assigned_node_id)),
            ),
            "cluster_count": allocation_result.auction_stats.get("cluster_count", 0.0),
            "avg_cluster_size": allocation_result.auction_stats.get("avg_cluster_size", 0.0),
            "graph_cut": allocation_result.auction_stats.get("graph_cut", 0.0),
            "welfare_loss_vs_global": allocation_result.auction_stats.get("welfare_loss_vs_global", 0.0),
        }
        step_row.update(mode_counts)
        return step_row

    def _drain_remaining_work(self) -> None:
        safety_counter = 0
        while any(node.reservations for node in self.nodes) and safety_counter < 1000:
            future_finishes = [
                reservation.finish_time
                for node in self.nodes
                for reservation in node.reservations
                if reservation.finish_time > self.current_time_ms
            ]
            if not future_finishes:
                self._finalize_completed_tasks(self.current_time_ms)
                break
            next_finish = min(future_finishes)
            self.current_time_ms = max(self.current_time_ms, next_finish)
            self._finalize_completed_tasks(self.current_time_ms)
            safety_counter += 1
        self._refresh_node_statistics()

    def _build_episode_summary(self, step_rows: list[dict[str, object]]) -> dict[str, object]:
        completed_tasks = [self.tasks_by_id[task_id] for task_id in self.completed_task_ids]
        violated_tasks = [self.tasks_by_id[task_id] for task_id in self.violated_task_ids]
        all_terminal_tasks = completed_tasks + violated_tasks
        latencies = [(task.finish_time or self.current_time_ms) - task.arrival_time for task in all_terminal_tasks]
        summary = {
            "scenario": self.scenario.scenario_name,
            "scenario_size": self.scenario.size_name,
            "seed": self.seed,
            "episode_steps": len(step_rows),
            "social_welfare": float(np.sum([row["social_welfare"] for row in step_rows])) if step_rows else 0.0,
            "completed_tasks": len(self.completed_task_ids),
            "generated_tasks": len(self.generated_task_ids),
            "completion_rate": safe_div(len(self.completed_task_ids), max(1, len(self.generated_task_ids))),
            "drop_rate": safe_div(len(self.dropped_task_ids), max(1, len(self.generated_task_ids))),
            "deadline_violation_rate": safe_div(len(self.violated_task_ids), max(1, len(self.generated_task_ids))),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": percentile(latencies, 95.0),
            "avg_node_utilization": float(np.mean([row["avg_node_utilization"] for row in step_rows])) if step_rows else 0.0,
            "load_imbalance": float(np.mean([row["load_imbalance"] for row in step_rows])) if step_rows else 0.0,
            "overload_count": int(np.sum([row["overload_count"] for row in step_rows])) if step_rows else 0,
            "allocation_overhead_ms": float(np.mean([row["allocation_overhead_ms"] for row in step_rows])) if step_rows else 0.0,
            "policy_inference_time_ms": float(np.mean([row["policy_inference_time_ms"] for row in step_rows])) if step_rows else 0.0,
            "total_decision_overhead_ms": float(np.mean([row["total_decision_overhead_ms"] for row in step_rows])) if step_rows else 0.0,
            "slot_share": float(np.mean([row["slot_share"] for row in step_rows])) if step_rows else 0.0,
            "payments_sum": float(np.sum([row["payments_sum"] for row in step_rows])) if step_rows else 0.0,
            "node_compensation_sum": float(np.sum([row["node_compensation_sum"] for row in step_rows])) if step_rows else 0.0,
            "average_externality": float(np.mean([row["average_externality"] for row in step_rows])) if step_rows else 0.0,
            "cluster_count": float(np.mean([row["cluster_count"] for row in step_rows])) if step_rows else 0.0,
            "avg_cluster_size": float(np.mean([row["avg_cluster_size"] for row in step_rows])) if step_rows else 0.0,
            "graph_cut": float(np.mean([row["graph_cut"] for row in step_rows])) if step_rows else 0.0,
            "welfare_loss_vs_global": float(np.mean([row["welfare_loss_vs_global"] for row in step_rows])) if step_rows else 0.0,
        }
        return summary
