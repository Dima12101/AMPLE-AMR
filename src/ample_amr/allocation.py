"""Task allocation layers for AMPLE-AMR experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter

import networkx as nx

from .config import WarehouseScenarioConfig
from .domain import EdgeNode, Robot, Task
from .network import NetworkGraph
from .utility import compute_social_welfare, estimate_network_time, estimate_processing_time, evaluate_cost, evaluate_utility
from .utils import safe_div

EXACT_TASK_EXTERNALITY_MAX_TASKS = 10
EXACT_TASK_EXTERNALITY_MAX_NODES = 6


@dataclass
class AllocationResult:
    """Common result object returned by all allocation layers."""

    allocation_matrix: dict[str, str]
    task_to_node: dict[str, str]
    payments_by_robot: dict[str, float]
    payments_by_task: dict[str, float]
    externality_by_task: dict[str, float]
    compensation_by_node: dict[str, float]
    social_welfare: float
    dropped_tasks: list[str]
    auction_stats: dict[str, object]
    allocation_time_ms: float
    policy_time_ms: float
    total_decision_time_ms: float
    slot_share: float


@dataclass(frozen=True)
class CandidateAssignment:
    """Feasible task-to-node candidate evaluated for the auction."""

    task_id: str
    node_id: str
    utility: float
    cost: float
    welfare: float
    total_latency_ms: float
    effective_cpu: float
    effective_memory: float
    effective_gpu: float


@dataclass
class NodeBudget:
    """Allocation-time resource budget for a node."""

    cpu: float
    memory: float
    gpu: float

    def can_fit(self, candidate: CandidateAssignment) -> bool:
        return (
            self.cpu + 1e-9 >= candidate.effective_cpu
            and self.memory + 1e-9 >= candidate.effective_memory
            and self.gpu + 1e-9 >= candidate.effective_gpu
        )

    def apply(self, candidate: CandidateAssignment) -> "NodeBudget":
        return NodeBudget(
            cpu=self.cpu - candidate.effective_cpu,
            memory=self.memory - candidate.effective_memory,
            gpu=self.gpu - candidate.effective_gpu,
        )


class BaseAllocator(ABC):
    """Shared allocator interface."""

    name: str

    @abstractmethod
    def allocate(
        self,
        tasks: list[Task],
        robots: dict[str, Robot],
        nodes: list[EdgeNode],
        graph: NetworkGraph,
        scenario: WarehouseScenarioConfig,
        step: int,
        policy_time_ms: float = 0.0,
        global_reference_welfare: float | None = None,
        compute_pricing: bool = True,
    ) -> AllocationResult:
        """Allocate tasks to nodes."""


def build_node_budgets(nodes: list[EdgeNode], scenario: WarehouseScenarioConfig) -> dict[str, NodeBudget]:
    """Build allocation-time resource budgets for all nodes."""

    budgets: dict[str, NodeBudget] = {}
    for node in nodes:
        exposed = node.last_auction_stats.get("exposed_capacity")
        if exposed is None:
            exposed_cpu = node.cpu_capacity_norm * 0.6
            exposed_memory = node.memory_capacity_norm * 0.6
            exposed_gpu = node.gpu_capacity_norm * 0.6
        else:
            exposed_cpu = float(exposed["cpu"])
            exposed_memory = float(exposed["memory"])
            exposed_gpu = float(exposed["gpu"])
        queue_slack = 1.0 + scenario.queue_reservation_factor
        budgets[node.id] = NodeBudget(
            cpu=max(0.0, exposed_cpu - node.current_cpu_load) * queue_slack,
            memory=max(0.0, exposed_memory - node.current_memory_load) * queue_slack,
            gpu=max(0.0, exposed_gpu - node.current_gpu_load) * queue_slack,
        )
    return budgets


def build_candidates(
    tasks: list[Task],
    robots: dict[str, Robot],
    nodes: list[EdgeNode],
    graph: NetworkGraph,
    scenario: WarehouseScenarioConfig,
    step: int,
    require_positive_welfare: bool = True,
) -> tuple[dict[str, list[CandidateAssignment]], dict[str, float]]:
    """Evaluate feasible candidates for each task."""

    candidates: dict[str, list[CandidateAssignment]] = {}
    max_positive_welfare: dict[str, float] = {}
    for task in tasks:
        robot = robots[task.robot_id]
        task_candidates: list[CandidateAssignment] = []
        for node in nodes:
            if node.failure_flag:
                continue
            exposed = node.last_auction_stats.get("exposed_capacity")
            if exposed is not None and (
                task.cpu_required > float(exposed["cpu"]) + 1e-9
                or task.memory_required > float(exposed["memory"]) + 1e-9
                or task.gpu_required > float(exposed["gpu"]) + 1e-9
            ):
                continue
            utility = evaluate_utility(task, robot, node, graph, step)
            if utility == float("-inf"):
                continue
            cost = evaluate_cost(task, robot, node, graph, step)
            welfare = utility - cost
            if require_positive_welfare and welfare <= 0.0:
                continue
            total_latency_ms = estimate_network_time(robot, node, task, graph, step) + node.last_auction_stats.get(
                "estimated_backlog_ms",
                0.0,
            ) + estimate_processing_time(task, node)
            duration_scale = max(total_latency_ms, scenario.slot_duration_ms) / max(scenario.queue_horizon_ms, 1.0)
            task_candidates.append(
                CandidateAssignment(
                    task_id=task.id,
                    node_id=node.id,
                    utility=utility,
                    cost=cost,
                    welfare=welfare,
                    total_latency_ms=total_latency_ms,
                    effective_cpu=task.cpu_required * duration_scale,
                    effective_memory=task.memory_required * max(1.0, duration_scale * 0.7),
                    effective_gpu=task.gpu_required * duration_scale,
                )
            )
        task_candidates.sort(key=lambda item: (-item.welfare, item.total_latency_ms, item.node_id))
        candidates[task.id] = task_candidates
        max_positive_welfare[task.id] = task_candidates[0].welfare if task_candidates else 0.0
    return candidates, max_positive_welfare


def apply_assignment_to_tasks(
    tasks_by_id: dict[str, Task],
    assignment: dict[str, CandidateAssignment],
    payments_by_robot: dict[str, float],
    payments_by_task: dict[str, float],
    externality_by_task: dict[str, float],
    compensation_by_node: dict[str, float],
    assignment_context_by_node: dict[str, dict[str, float]],
    task_payment_mode: str,
    task_externality_mode: str,
) -> None:
    """Copy allocation outputs into task objects."""

    robot_assigned_totals = defaultdict(float)
    node_assigned_counts = defaultdict(int)
    for candidate in assignment.values():
        robot_id = tasks_by_id[candidate.task_id].robot_id
        robot_assigned_totals[robot_id] += candidate.welfare
        node_assigned_counts[candidate.node_id] += 1
    for task in tasks_by_id.values():
        task.assigned_node_id = None
        task.utility = 0.0
        task.cost = 0.0
        task.welfare_contribution = 0.0
        task.robot_payment = 0.0
        task.node_compensation = 0.0
        task.externality_estimate = 0.0
    for task_id, candidate in assignment.items():
        task = tasks_by_id[task_id]
        task.assigned_node_id = candidate.node_id
        task.utility = candidate.utility
        task.cost = candidate.cost
        task.welfare_contribution = candidate.welfare
        assignment_context = assignment_context_by_node.get(candidate.node_id, {})
        task_payment = payments_by_task.get(task_id, 0.0)
        task_externality = externality_by_task.get(task_id, task_payment)
        payment_total = payments_by_robot.get(task.robot_id, 0.0)
        compensation_total = compensation_by_node.get(candidate.node_id, 0.0)
        task.robot_payment = payment_total * safe_div(candidate.welfare, robot_assigned_totals[task.robot_id])
        task.node_compensation = compensation_total / max(1, node_assigned_counts[candidate.node_id])
        task.externality_estimate = task_externality
        task.metadata["assignment_latency_ms"] = candidate.total_latency_ms
        task.metadata["node_utilization_at_assignment"] = assignment_context.get("node_utilization_at_assignment", 0.0)
        task.metadata["queue_length_at_assignment"] = assignment_context.get("queue_length_at_assignment", 0.0)
        task.metadata["task_payment_mode"] = task_payment_mode
        task.metadata["task_externality_mode"] = task_externality_mode
        task.metadata["task_payment_effective"] = task_payment
        task.metadata["task_externality_effective"] = task_externality
        task.metadata["task_payment"] = task_payment if task_payment_mode == "exact" else float("nan")
        task.metadata["task_externality"] = task_externality if task_externality_mode == "exact" else float("nan")
        task.metadata["approx_task_payment"] = task_payment if task_payment_mode != "exact" else float("nan")
        task.metadata["approx_task_externality"] = task_externality if task_externality_mode != "exact" else float("nan")


def build_assignment_context(nodes: list[EdgeNode]) -> dict[str, dict[str, float]]:
    """Capture node-side diagnostics used for task-level experiment outputs."""

    return {
        node.id: {
            "node_utilization_at_assignment": float(node.last_auction_stats.get("exposed_capacity_utilization", 0.0)),
            "queue_length_at_assignment": float(len(node.queue)),
        }
        for node in nodes
    }


def select_task_externality_mode(tasks: list[Task], nodes: list[EdgeNode]) -> str:
    """Choose whether to use exact or approximate task-level diagnostics."""

    if len(tasks) <= EXACT_TASK_EXTERNALITY_MAX_TASKS and len(nodes) <= EXACT_TASK_EXTERNALITY_MAX_NODES:
        return "exact"
    return "approx_proportional_robot_payment"


class HeuristicAllocator(BaseAllocator):
    """Simple baseline allocator under exposed capacity constraints."""

    def __init__(self, policy: str = "min_latency") -> None:
        self.policy = policy
        self.name = f"heuristic:{policy}"

    def allocate(
        self,
        tasks: list[Task],
        robots: dict[str, Robot],
        nodes: list[EdgeNode],
        graph: NetworkGraph,
        scenario: WarehouseScenarioConfig,
        step: int,
        policy_time_ms: float = 0.0,
        global_reference_welfare: float | None = None,
        compute_pricing: bool = True,
    ) -> AllocationResult:
        start = perf_counter()
        budgets = build_node_budgets(nodes, scenario)
        tasks_by_id = {task.id: task for task in tasks}
        # The heuristic baseline ignores welfare during assignment and only
        # measures welfare after the assignment is fixed.
        candidates, _ = build_candidates(
            tasks,
            robots,
            nodes,
            graph,
            scenario,
            step,
            require_positive_welfare=False,
        )
        assignment_context = build_assignment_context(nodes)
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (-int(task.hard_deadline), task.deadline_ms, -task.priority, task.id),
        )
        assignment: dict[str, CandidateAssignment] = {}
        dropped: list[str] = []
        for task in ordered_tasks:
            task_candidates = [candidate for candidate in candidates[task.id] if budgets[candidate.node_id].can_fit(candidate)]
            if not task_candidates:
                dropped.append(task.id)
                task.status = "dropped"
                continue
            if self.policy == "min_latency":
                chosen = min(task_candidates, key=lambda item: (item.total_latency_ms, item.node_id))
            elif self.policy == "least_loaded":
                chosen = max(
                    task_candidates,
                    key=lambda item: (
                        budgets[item.node_id].cpu + budgets[item.node_id].memory + budgets[item.node_id].gpu,
                        -item.total_latency_ms,
                        item.node_id,
                    ),
                )
            else:
                chosen = max(task_candidates, key=lambda item: (item.welfare, -item.total_latency_ms, item.node_id))
            budgets[chosen.node_id] = budgets[chosen.node_id].apply(chosen)
            assignment[task.id] = chosen
        payments_by_robot = {task.robot_id: 0.0 for task in tasks}
        payments_by_task = {task.id: 0.0 for task in tasks}
        externality_by_task = {task.id: 0.0 for task in tasks}
        compensation_by_node = {node.id: 0.0 for node in nodes}
        apply_assignment_to_tasks(
            tasks_by_id,
            assignment,
            payments_by_robot,
            payments_by_task,
            externality_by_task,
            compensation_by_node,
            assignment_context,
            "heuristic_zero",
            "heuristic_zero",
        )
        social_welfare = compute_social_welfare(list(tasks_by_id.values()))
        allocation_time_ms = (perf_counter() - start) * 1000.0
        return AllocationResult(
            allocation_matrix={task_id: candidate.node_id for task_id, candidate in assignment.items()},
            task_to_node={task_id: candidate.node_id for task_id, candidate in assignment.items()},
            payments_by_robot=payments_by_robot,
            payments_by_task=payments_by_task,
            externality_by_task=externality_by_task,
            compensation_by_node=compensation_by_node,
            social_welfare=social_welfare,
            dropped_tasks=dropped,
            auction_stats={
                "candidate_count": float(sum(len(value) for value in candidates.values())),
                "allocated_task_count": float(len(assignment)),
                "policy": self.policy,
                "robot_payment_mode": "heuristic_zero",
                "task_payment_mode": "heuristic_zero",
                "task_externality_mode": "heuristic_zero",
            },
            allocation_time_ms=allocation_time_ms,
            policy_time_ms=policy_time_ms,
            total_decision_time_ms=allocation_time_ms + policy_time_ms,
            slot_share=(allocation_time_ms + policy_time_ms) / scenario.slot_duration_ms,
        )


@dataclass
class SolverOutcome:
    """Internal exact solver output."""

    welfare: float
    assignment: dict[str, CandidateAssignment]
    explored_states: int
    pruned_states: int


class VCGLikeAllocator(BaseAllocator):
    """Deterministic branch-and-bound allocator with VCG-like pricing."""

    def __init__(self) -> None:
        self.name = "vcg_like"

    def allocate(
        self,
        tasks: list[Task],
        robots: dict[str, Robot],
        nodes: list[EdgeNode],
        graph: NetworkGraph,
        scenario: WarehouseScenarioConfig,
        step: int,
        policy_time_ms: float = 0.0,
        global_reference_welfare: float | None = None,
        compute_pricing: bool = True,
    ) -> AllocationResult:
        start = perf_counter()
        budgets = build_node_budgets(nodes, scenario)
        tasks_by_id = {task.id: task for task in tasks}
        candidates, max_positive_welfare = build_candidates(tasks, robots, nodes, graph, scenario, step)
        assignment_context = build_assignment_context(nodes)
        outcome = self._solve(tasks, budgets, candidates, max_positive_welfare)
        if compute_pricing:
            payments_by_robot = self._compute_payments(
                tasks,
                robots,
                scenario,
                candidates,
                budgets,
                max_positive_welfare,
                outcome,
            )
            task_externality_mode = select_task_externality_mode(tasks, nodes)
            task_payment_mode = task_externality_mode
            if task_externality_mode == "exact":
                payments_by_task, externality_by_task = self._compute_task_externalities(
                    tasks,
                    candidates,
                    budgets,
                    max_positive_welfare,
                    outcome,
                )
            else:
                payments_by_task, externality_by_task = self._approximate_task_externalities(
                    tasks,
                    payments_by_robot,
                    outcome,
                )
            compensation_by_node = self._compute_compensation(
                tasks,
                nodes,
                candidates,
                budgets,
                max_positive_welfare,
                outcome,
            )
        else:
            payments_by_robot = {task.robot_id: 0.0 for task in tasks}
            payments_by_task = {task.id: 0.0 for task in tasks}
            externality_by_task = {task.id: 0.0 for task in tasks}
            compensation_by_node = {node.id: 0.0 for node in nodes}
            task_payment_mode = "skipped_for_training"
            task_externality_mode = "skipped_for_training"
        apply_assignment_to_tasks(
            tasks_by_id,
            outcome.assignment,
            payments_by_robot,
            payments_by_task,
            externality_by_task,
            compensation_by_node,
            assignment_context,
            task_payment_mode,
            task_externality_mode,
        )
        for task in tasks:
            if task.id not in outcome.assignment:
                task.status = "dropped"
        social_welfare = compute_social_welfare(list(tasks_by_id.values()))
        allocation_time_ms = (perf_counter() - start) * 1000.0
        return AllocationResult(
            allocation_matrix={task_id: candidate.node_id for task_id, candidate in outcome.assignment.items()},
            task_to_node={task_id: candidate.node_id for task_id, candidate in outcome.assignment.items()},
            payments_by_robot=payments_by_robot,
            payments_by_task=payments_by_task,
            externality_by_task=externality_by_task,
            compensation_by_node=compensation_by_node,
            social_welfare=social_welfare,
            dropped_tasks=[task.id for task in tasks if task.id not in outcome.assignment],
            auction_stats={
                "candidate_count": float(sum(len(value) for value in candidates.values())),
                "allocated_task_count": float(len(outcome.assignment)),
                "explored_states": float(outcome.explored_states),
                "pruned_states": float(outcome.pruned_states),
                "exact": 1.0,
                "robot_payment_mode": "robot_vcg_like" if compute_pricing else "skipped_for_training",
                "task_payment_mode": task_payment_mode,
                "task_externality_mode": task_externality_mode,
            },
            allocation_time_ms=allocation_time_ms,
            policy_time_ms=policy_time_ms,
            total_decision_time_ms=allocation_time_ms + policy_time_ms,
            slot_share=(allocation_time_ms + policy_time_ms) / scenario.slot_duration_ms,
        )

    def _solve(
        self,
        tasks: list[Task],
        budgets: dict[str, NodeBudget],
        candidates: dict[str, list[CandidateAssignment]],
        max_positive_welfare: dict[str, float],
    ) -> SolverOutcome:
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (-max_positive_welfare.get(task.id, 0.0), -int(task.hard_deadline), task.deadline_ms, task.id),
        )
        best_welfare = 0.0
        best_assignment: dict[str, CandidateAssignment] = {}
        explored_states = 0
        pruned_states = 0
        suffix_bound = [0.0 for _ in range(len(ordered_tasks) + 1)]
        for index in range(len(ordered_tasks) - 1, -1, -1):
            suffix_bound[index] = suffix_bound[index + 1] + max_positive_welfare.get(ordered_tasks[index].id, 0.0)

        def recurse(index: int, current_welfare: float, current_assignment: dict[str, CandidateAssignment], current_budgets: dict[str, NodeBudget]) -> None:
            nonlocal best_welfare, best_assignment, explored_states, pruned_states
            explored_states += 1
            if current_welfare + suffix_bound[index] + 1e-9 < best_welfare:
                pruned_states += 1
                return
            if index >= len(ordered_tasks):
                if current_welfare > best_welfare + 1e-9:
                    best_welfare = current_welfare
                    best_assignment = dict(current_assignment)
                return

            task = ordered_tasks[index]
            recurse(index + 1, current_welfare, current_assignment, current_budgets)
            for candidate in candidates[task.id]:
                budget = current_budgets[candidate.node_id]
                if not budget.can_fit(candidate):
                    continue
                next_assignment = dict(current_assignment)
                next_assignment[task.id] = candidate
                next_budgets = dict(current_budgets)
                next_budgets[candidate.node_id] = budget.apply(candidate)
                recurse(index + 1, current_welfare + candidate.welfare, next_assignment, next_budgets)

        recurse(0, 0.0, {}, budgets)
        return SolverOutcome(
            welfare=best_welfare,
            assignment=best_assignment,
            explored_states=explored_states,
            pruned_states=pruned_states,
        )

    def _compute_payments(
        self,
        tasks: list[Task],
        robots: dict[str, Robot],
        scenario: WarehouseScenarioConfig,
        candidates: dict[str, list[CandidateAssignment]],
        budgets: dict[str, NodeBudget],
        max_positive_welfare: dict[str, float],
        optimal: SolverOutcome,
    ) -> dict[str, float]:
        payments = {task.robot_id: 0.0 for task in tasks}
        task_lookup = tasks_by_id(tasks)
        assigned_welfare_by_robot = defaultdict(float)
        for candidate in optimal.assignment.values():
            assigned_welfare_by_robot[task_lookup[candidate.task_id].robot_id] += candidate.welfare
        for robot_id in sorted({task.robot_id for task in tasks}):
            reduced_tasks = [task for task in tasks if task.robot_id != robot_id]
            reduced_candidates = {task.id: candidates[task.id] for task in reduced_tasks}
            reduced_bound = {task.id: max_positive_welfare.get(task.id, 0.0) for task in reduced_tasks}
            without_robot = self._solve(reduced_tasks, budgets, reduced_candidates, reduced_bound)
            welfare_of_others_with_robot = optimal.welfare - assigned_welfare_by_robot.get(robot_id, 0.0)
            payment = max(0.0, without_robot.welfare - welfare_of_others_with_robot)
            payments[robot_id] = payment
        return payments

    def _approximate_task_externalities(
        self,
        tasks: list[Task],
        payments_by_robot: dict[str, float],
        optimal: SolverOutcome,
    ) -> tuple[dict[str, float], dict[str, float]]:
        payments_by_task = {task.id: 0.0 for task in tasks}
        externality_by_task = {task.id: 0.0 for task in tasks}
        task_lookup = tasks_by_id(tasks)
        assigned_welfare_by_robot = defaultdict(float)
        for candidate in optimal.assignment.values():
            assigned_welfare_by_robot[task_lookup[candidate.task_id].robot_id] += candidate.welfare
        for task_id, candidate in optimal.assignment.items():
            robot_id = task_lookup[task_id].robot_id
            payment = payments_by_robot.get(robot_id, 0.0) * safe_div(candidate.welfare, assigned_welfare_by_robot[robot_id])
            payments_by_task[task_id] = payment
            externality_by_task[task_id] = payment
        return payments_by_task, externality_by_task

    def _compute_task_externalities(
        self,
        tasks: list[Task],
        candidates: dict[str, list[CandidateAssignment]],
        budgets: dict[str, NodeBudget],
        max_positive_welfare: dict[str, float],
        optimal: SolverOutcome,
    ) -> tuple[dict[str, float], dict[str, float]]:
        payments_by_task = {task.id: 0.0 for task in tasks}
        externality_by_task = {task.id: 0.0 for task in tasks}
        for task in tasks:
            assigned = optimal.assignment.get(task.id)
            if assigned is None:
                continue
            reduced_tasks = [candidate_task for candidate_task in tasks if candidate_task.id != task.id]
            reduced_candidates = {candidate_task.id: candidates[candidate_task.id] for candidate_task in reduced_tasks}
            reduced_bound = {candidate_task.id: max_positive_welfare.get(candidate_task.id, 0.0) for candidate_task in reduced_tasks}
            without_task = self._solve(reduced_tasks, budgets, reduced_candidates, reduced_bound)
            payment = max(0.0, without_task.welfare - (optimal.welfare - assigned.welfare))
            payments_by_task[task.id] = payment
            externality_by_task[task.id] = payment
        return payments_by_task, externality_by_task

    def _compute_compensation(
        self,
        tasks: list[Task],
        nodes: list[EdgeNode],
        candidates: dict[str, list[CandidateAssignment]],
        budgets: dict[str, NodeBudget],
        max_positive_welfare: dict[str, float],
        optimal: SolverOutcome,
    ) -> dict[str, float]:
        compensation = {node.id: 0.0 for node in nodes}
        assigned_nodes = sorted({candidate.node_id for candidate in optimal.assignment.values()})
        for node_id in assigned_nodes:
            reduced_nodes = [node for node in nodes if node.id != node_id]
            reduced_budgets = {key: value for key, value in budgets.items() if key != node_id}
            reduced_candidates = {
                task.id: [candidate for candidate in task_candidates if candidate.node_id != node_id]
                for task, task_candidates in ((task, candidates[task.id]) for task in tasks)
            }
            reduced_bound = {task.id: max_positive_welfare.get(task.id, 0.0) for task in tasks}
            without_node = self._solve(tasks, reduced_budgets, reduced_candidates, reduced_bound)
            compensation[node_id] = max(0.0, optimal.welfare - without_node.welfare)
        return compensation


class ClusteredVCGLikeAllocator(BaseAllocator):
    """Clustered local VCG-like allocator without leader selection."""

    def __init__(self) -> None:
        self.name = "clustered_vcg_like"
        self.local_allocator = VCGLikeAllocator()

    def allocate(
        self,
        tasks: list[Task],
        robots: dict[str, Robot],
        nodes: list[EdgeNode],
        graph: NetworkGraph,
        scenario: WarehouseScenarioConfig,
        step: int,
        policy_time_ms: float = 0.0,
        global_reference_welfare: float | None = None,
        compute_pricing: bool = True,
    ) -> AllocationResult:
        start = perf_counter()
        if not tasks:
            return AllocationResult(
                allocation_matrix={},
                task_to_node={},
                payments_by_robot={},
                payments_by_task={},
                externality_by_task={},
                compensation_by_node={node.id: 0.0 for node in nodes},
                social_welfare=0.0,
                dropped_tasks=[],
                auction_stats={
                    "cluster_count": 0.0,
                    "avg_cluster_size": 0.0,
                    "graph_cut": 0.0,
                    "robot_payment_mode": "sum_local_robot_vcg_like",
                    "task_payment_mode": "sum_local_task_diagnostics",
                    "task_externality_mode": "sum_local_task_diagnostics",
                },
                allocation_time_ms=0.0,
                policy_time_ms=policy_time_ms,
                total_decision_time_ms=policy_time_ms,
                slot_share=policy_time_ms / scenario.slot_duration_ms,
            )

        participating_robot_ids = sorted({task.robot_id for task in tasks})
        participating_robots = [robots[robot_id] for robot_id in participating_robot_ids]
        communities, graph_cut = self._build_clusters(participating_robots, nodes, graph, scenario, step)
        cluster_nodes = self._assign_nodes_to_clusters(communities, participating_robots, nodes, graph, step)

        combined_assignment: dict[str, str] = {}
        payments_by_robot: dict[str, float] = {task.robot_id: 0.0 for task in tasks}
        payments_by_task: dict[str, float] = {task.id: 0.0 for task in tasks}
        externality_by_task: dict[str, float] = {task.id: 0.0 for task in tasks}
        compensation_by_node: dict[str, float] = {node.id: 0.0 for node in nodes}
        dropped_tasks: list[str] = []
        cluster_welfare = 0.0
        local_times: list[float] = []

        task_lookup = {task.id: task for task in tasks}
        for cluster_index, community in enumerate(communities):
            cluster_robot_ids = {robot.id for robot in community}
            cluster_tasks = [task for task in tasks if task.robot_id in cluster_robot_ids]
            if not cluster_tasks:
                continue
            local_nodes = cluster_nodes.get(cluster_index, [])
            if not local_nodes:
                for task in cluster_tasks:
                    task.status = "dropped"
                    dropped_tasks.append(task.id)
                continue
            local_result = self.local_allocator.allocate(
                tasks=cluster_tasks,
                robots=robots,
                nodes=local_nodes,
                graph=graph,
                scenario=scenario,
                step=step,
                policy_time_ms=0.0,
                compute_pricing=compute_pricing,
            )
            local_times.append(local_result.allocation_time_ms)
            cluster_welfare += local_result.social_welfare
            combined_assignment.update(local_result.task_to_node)
            dropped_tasks.extend(local_result.dropped_tasks)
            for robot_id, payment in local_result.payments_by_robot.items():
                payments_by_robot[robot_id] = payments_by_robot.get(robot_id, 0.0) + payment
            for task_id, payment in local_result.payments_by_task.items():
                payments_by_task[task_id] = payments_by_task.get(task_id, 0.0) + payment
            for task_id, externality in local_result.externality_by_task.items():
                externality_by_task[task_id] = externality_by_task.get(task_id, 0.0) + externality
            for node_id, compensation in local_result.compensation_by_node.items():
                compensation_by_node[node_id] = compensation_by_node.get(node_id, 0.0) + compensation

        for task_id, node_id in combined_assignment.items():
            task_lookup[task_id].assigned_node_id = node_id
        allocation_time_ms = (perf_counter() - start) * 1000.0
        welfare_loss_vs_global = 0.0
        if global_reference_welfare is not None:
            welfare_loss_vs_global = max(0.0, global_reference_welfare - cluster_welfare)
        return AllocationResult(
            allocation_matrix=dict(combined_assignment),
            task_to_node=dict(combined_assignment),
            payments_by_robot=payments_by_robot,
            payments_by_task=payments_by_task,
            externality_by_task=externality_by_task,
            compensation_by_node=compensation_by_node,
            social_welfare=cluster_welfare,
            dropped_tasks=sorted(set(dropped_tasks)),
            auction_stats={
                "cluster_count": float(len(communities)),
                "avg_cluster_size": safe_div(sum(len(cluster) for cluster in communities), len(communities)),
                "graph_cut": graph_cut,
                "welfare_loss_vs_global": welfare_loss_vs_global,
                "local_allocation_time_mean_ms": sum(local_times) / len(local_times) if local_times else 0.0,
                "robot_payment_mode": "sum_local_robot_vcg_like" if compute_pricing else "skipped_for_training",
                "task_payment_mode": "local_mixed_task_diagnostics" if compute_pricing else "skipped_for_training",
                "task_externality_mode": "local_mixed_task_diagnostics" if compute_pricing else "skipped_for_training",
            },
            allocation_time_ms=allocation_time_ms,
            policy_time_ms=policy_time_ms,
            total_decision_time_ms=allocation_time_ms + policy_time_ms,
            slot_share=(allocation_time_ms + policy_time_ms) / scenario.slot_duration_ms,
        )

    def _build_clusters(
        self,
        robots: list[Robot],
        nodes: list[EdgeNode],
        graph: NetworkGraph,
        scenario: WarehouseScenarioConfig,
        step: int,
    ) -> tuple[list[list[Robot]], float]:
        affinity_graph = nx.Graph()
        for robot in robots:
            affinity_graph.add_node(robot.id, robot=robot)
        for index, robot_u in enumerate(robots):
            for robot_v in robots[index + 1 :]:
                weight = graph.affinity_weight(robot_u, robot_v, nodes, step)
                if weight > 0.0:
                    affinity_graph.add_edge(robot_u.id, robot_v.id, weight=weight)

        if affinity_graph.number_of_edges() > 0:
            communities_ids = [
                sorted(list(community))
                for community in nx.algorithms.community.greedy_modularity_communities(
                    affinity_graph,
                    weight="weight",
                )
            ]
        else:
            communities_ids = []

        if not communities_ids:
            sorted_robots = sorted(robots, key=lambda robot: (robot.position[0], robot.position[1], robot.id))
            chunk_size = max(1, scenario.cluster_target_size)
            communities_ids = [
                [robot.id for robot in sorted_robots[index : index + chunk_size]]
                for index in range(0, len(sorted_robots), chunk_size)
            ]

        robot_lookup = {robot.id: robot for robot in robots}
        communities = [[robot_lookup[robot_id] for robot_id in community] for community in communities_ids]
        graph_cut = 0.0
        membership = {}
        for cluster_index, community in enumerate(communities_ids):
            for robot_id in community:
                membership[robot_id] = cluster_index
        for source, target, data in affinity_graph.edges(data=True):
            if membership.get(source) != membership.get(target):
                graph_cut += float(data.get("weight", 0.0))
        return communities, graph_cut

    def _assign_nodes_to_clusters(
        self,
        communities: list[list[Robot]],
        participating_robots: list[Robot],
        nodes: list[EdgeNode],
        graph: NetworkGraph,
        step: int,
    ) -> dict[int, list[EdgeNode]]:
        cluster_nodes: dict[int, list[EdgeNode]] = {index: [] for index in range(len(communities))}
        dummy_tasks = {
            robot.id: Task(
                id=f"cluster-dummy-{robot.id}",
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
            for robot in participating_robots
        }
        for node in nodes:
            best_cluster = 0
            best_score = float("-inf")
            for cluster_index, community in enumerate(communities):
                feasible_costs: list[float] = []
                for robot in community:
                    estimate = graph.estimate_link(robot, node, dummy_tasks[robot.id], step)
                    if estimate.feasible:
                        feasible_costs.append(-estimate.network_cost)
                score = sum(feasible_costs) / len(feasible_costs) if feasible_costs else float("-inf")
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_index
            cluster_nodes[best_cluster].append(node)
        return cluster_nodes


def tasks_by_id(tasks: list[Task]) -> dict[str, Task]:
    """Build a task lookup dictionary."""

    return {task.id: task for task in tasks}
