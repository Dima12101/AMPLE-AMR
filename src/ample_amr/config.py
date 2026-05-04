"""Configuration loading and scenario resolution for warehouse experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WarehouseSizeSpec:
    """Fixed warehouse scale definition used by all experiments."""

    name: str
    robots: int
    access_points: int
    edge_nodes: int
    width_m: float
    height_m: float


@dataclass(frozen=True)
class EdgeNodeClassConfig:
    """Capacity profile for an edge node class."""

    name: str
    cpu_capacity_norm: float
    memory_capacity_norm: float
    gpu_capacity_norm: float


@dataclass(frozen=True)
class OperationModeConfig:
    """Exposure coefficients for a node operation mode."""

    name: str
    cpu_exposure: float
    memory_exposure: float
    gpu_exposure: float
    priority_boost_task_classes: tuple[str, ...] = ()
    priority_boost: float = 1.0


@dataclass(frozen=True)
class TaskClassConfig:
    """Task class parameter ranges and valuation assumptions."""

    name: str
    arrival_weight: float
    base_value: float
    deadline_ms: tuple[float, float]
    input_size_mbit: tuple[float, float]
    output_size_mbit: tuple[float, float]
    cpu_required: tuple[float, float]
    memory_required: tuple[float, float]
    gpu_required: tuple[float, float]
    hard_deadline_probability: float
    deadline_sensitivity: float
    service_time_ms: tuple[float, float]
    priority_range: tuple[int, int]


@dataclass(frozen=True)
class QMIXConfig:
    """Hyperparameters for the QMIX controller."""

    hidden_dim: int
    mixing_hidden_dim: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    replay_capacity: int
    batch_size: int
    target_update_interval: int
    training_episodes: int
    checkpoint_dir: str


@dataclass(frozen=True)
class SimulationDefaults:
    """Global defaults shared by scenarios."""

    slot_duration_ms: int
    episode_steps: int
    quick_episode_steps: int
    queue_horizon_ms: float
    queue_reservation_factor: float
    robot_speed_range_mps: tuple[float, float]
    robot_working_speed_range_mps: tuple[float, float]
    robot_payload_range_kg: tuple[float, float]
    mission_importance_range: tuple[float, float]
    local_edge_latency_ms: tuple[float, float]
    degraded_latency_ms: tuple[float, float]
    video_like_input_rate_mbps: tuple[float, float]
    high_speed_link_mbps: tuple[float, float]
    constrained_link_mbps: tuple[float, float]
    fixed_operation_mode: str
    random_mode_baseline_seed_offset: int


@dataclass
class WarehouseScenarioConfig:
    """Resolved scenario definition consumed by the simulator."""

    scenario_name: str
    size_name: str
    robots: int
    access_points: int
    edge_nodes: int
    width_m: float
    height_m: float
    slot_duration_ms: int
    episode_steps: int
    queue_horizon_ms: float
    queue_reservation_factor: float
    fixed_operation_mode: str
    robot_speed_range_mps: tuple[float, float]
    robot_working_speed_range_mps: tuple[float, float]
    robot_payload_range_kg: tuple[float, float]
    mission_importance_range: tuple[float, float]
    local_edge_latency_ms: tuple[float, float]
    degraded_latency_ms: tuple[float, float]
    video_like_input_rate_mbps: tuple[float, float]
    high_speed_link_mbps: tuple[float, float]
    constrained_link_mbps: tuple[float, float]
    edge_node_classes: dict[str, EdgeNodeClassConfig]
    operation_modes: dict[str, OperationModeConfig]
    task_classes: dict[str, TaskClassConfig]
    arrival_rate_per_robot: float
    task_mix: dict[str, float]
    heterogeneous_node_mix: dict[str, float]
    peak_spike_multiplier: float = 1.0
    peak_start_step: int = 0
    peak_duration_steps: int = 0
    degraded_region_fraction: float = 0.0
    degradation_start_step: int = 0
    degradation_duration_steps: int = 0
    failure_fraction: float = 0.0
    failure_start_step: int = 0
    failure_duration_steps: int = 0
    enable_recovery: bool = True
    cluster_target_size: int = 3
    scenario_group: str = "default"
    sensitivity_profiles: dict[str, dict[str, float]] = field(default_factory=dict)
    random_mode_baseline_seed_offset: int = 1000

    def arrival_rate_multiplier(self, step: int) -> float:
        """Return the step-dependent task arrival multiplier."""

        in_spike = self.peak_start_step <= step < self.peak_start_step + self.peak_duration_steps
        return self.peak_spike_multiplier if in_spike else 1.0

    def is_degraded_step(self, step: int) -> bool:
        """Check whether network degradation is active for this step."""

        return self.degradation_start_step <= step < self.degradation_start_step + self.degradation_duration_steps

    def is_failure_step(self, step: int) -> bool:
        """Check whether node failures are active for this step."""

        return self.failure_start_step <= step < self.failure_start_step + self.failure_duration_steps


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    defaults: SimulationDefaults
    sizes: dict[str, WarehouseSizeSpec]
    edge_node_classes: dict[str, EdgeNodeClassConfig]
    operation_modes: dict[str, OperationModeConfig]
    task_classes: dict[str, TaskClassConfig]
    scenarios: dict[str, dict[str, Any]]
    qmix: QMIXConfig

    def resolve_scenario(
        self,
        scenario_name: str,
        size_override: str | None = None,
        quick: bool = False,
    ) -> WarehouseScenarioConfig:
        """Resolve a scenario into a concrete warehouse configuration."""

        if scenario_name not in self.scenarios:
            raise KeyError(f"Unknown scenario '{scenario_name}'")
        scenario = dict(self.scenarios[scenario_name])
        size_name = size_override or scenario.get("default_size")
        if size_name is None:
            raise ValueError(f"Scenario '{scenario_name}' does not define a default size")
        if size_name not in self.sizes:
            raise KeyError(f"Unknown warehouse size '{size_name}'")
        size = self.sizes[size_name]
        episode_steps = int(
            self.defaults.quick_episode_steps if quick else scenario.get("episode_steps", self.defaults.episode_steps)
        )
        task_mix = scenario.get(
            "task_mix",
            {name: cfg.arrival_weight for name, cfg in self.task_classes.items()},
        )
        heterogeneous_node_mix = scenario.get(
            "heterogeneous_node_mix",
            {"Edge-S": 1.0},
        )
        return WarehouseScenarioConfig(
            scenario_name=scenario_name,
            size_name=size.name,
            robots=size.robots,
            access_points=size.access_points,
            edge_nodes=size.edge_nodes,
            width_m=size.width_m,
            height_m=size.height_m,
            slot_duration_ms=self.defaults.slot_duration_ms,
            episode_steps=episode_steps,
            queue_horizon_ms=scenario.get("queue_horizon_ms", self.defaults.queue_horizon_ms),
            queue_reservation_factor=scenario.get(
                "queue_reservation_factor",
                self.defaults.queue_reservation_factor,
            ),
            fixed_operation_mode=scenario.get("fixed_operation_mode", self.defaults.fixed_operation_mode),
            robot_speed_range_mps=self.defaults.robot_speed_range_mps,
            robot_working_speed_range_mps=self.defaults.robot_working_speed_range_mps,
            robot_payload_range_kg=self.defaults.robot_payload_range_kg,
            mission_importance_range=self.defaults.mission_importance_range,
            local_edge_latency_ms=self.defaults.local_edge_latency_ms,
            degraded_latency_ms=self.defaults.degraded_latency_ms,
            video_like_input_rate_mbps=self.defaults.video_like_input_rate_mbps,
            high_speed_link_mbps=self.defaults.high_speed_link_mbps,
            constrained_link_mbps=self.defaults.constrained_link_mbps,
            edge_node_classes=self.edge_node_classes,
            operation_modes=self.operation_modes,
            task_classes=self.task_classes,
            arrival_rate_per_robot=float(scenario["arrival_rate_per_robot"]),
            task_mix={str(k): float(v) for k, v in task_mix.items()},
            heterogeneous_node_mix={str(k): float(v) for k, v in heterogeneous_node_mix.items()},
            peak_spike_multiplier=float(scenario.get("peak_spike_multiplier", 1.0)),
            peak_start_step=int(scenario.get("peak_start_step", 0)),
            peak_duration_steps=int(scenario.get("peak_duration_steps", 0)),
            degraded_region_fraction=float(scenario.get("degraded_region_fraction", 0.0)),
            degradation_start_step=int(scenario.get("degradation_start_step", 0)),
            degradation_duration_steps=int(scenario.get("degradation_duration_steps", 0)),
            failure_fraction=float(scenario.get("failure_fraction", 0.0)),
            failure_start_step=int(scenario.get("failure_start_step", 0)),
            failure_duration_steps=int(scenario.get("failure_duration_steps", 0)),
            enable_recovery=bool(scenario.get("enable_recovery", True)),
            cluster_target_size=int(scenario.get("cluster_target_size", 3)),
            scenario_group=str(scenario.get("scenario_group", scenario_name)),
            sensitivity_profiles={
                str(name): {str(k): float(v) for k, v in values.items()}
                for name, values in scenario.get("sensitivity_profiles", {}).items()
            },
            random_mode_baseline_seed_offset=self.defaults.random_mode_baseline_seed_offset,
        )


def _tuple(values: list[float] | tuple[float, float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the experiment configuration from YAML."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = payload["defaults"]
    simulation_defaults = SimulationDefaults(
        slot_duration_ms=int(defaults["slot_duration_ms"]),
        episode_steps=int(defaults["episode_steps"]),
        quick_episode_steps=int(defaults["quick_episode_steps"]),
        queue_horizon_ms=float(defaults["queue_horizon_ms"]),
        queue_reservation_factor=float(defaults["queue_reservation_factor"]),
        robot_speed_range_mps=_tuple(defaults["robot_speed_range_mps"]),
        robot_working_speed_range_mps=_tuple(defaults["robot_working_speed_range_mps"]),
        robot_payload_range_kg=_tuple(defaults["robot_payload_range_kg"]),
        mission_importance_range=_tuple(defaults["mission_importance_range"]),
        local_edge_latency_ms=_tuple(defaults["local_edge_latency_ms"]),
        degraded_latency_ms=_tuple(defaults["degraded_latency_ms"]),
        video_like_input_rate_mbps=_tuple(defaults["video_like_input_rate_mbps"]),
        high_speed_link_mbps=_tuple(defaults["high_speed_link_mbps"]),
        constrained_link_mbps=_tuple(defaults["constrained_link_mbps"]),
        fixed_operation_mode=str(defaults["fixed_operation_mode"]),
        random_mode_baseline_seed_offset=int(defaults.get("random_mode_baseline_seed_offset", 1000)),
    )
    sizes = {
        name: WarehouseSizeSpec(
            name=name,
            robots=int(value["robots"]),
            access_points=int(value["access_points"]),
            edge_nodes=int(value["edge_nodes"]),
            width_m=float(value["width_m"]),
            height_m=float(value["height_m"]),
        )
        for name, value in payload["sizes"].items()
    }
    edge_node_classes = {
        name: EdgeNodeClassConfig(
            name=name,
            cpu_capacity_norm=float(value["cpu_capacity_norm"]),
            memory_capacity_norm=float(value["memory_capacity_norm"]),
            gpu_capacity_norm=float(value["gpu_capacity_norm"]),
        )
        for name, value in payload["edge_node_classes"].items()
    }
    operation_modes = {
        name: OperationModeConfig(
            name=name,
            cpu_exposure=float(value["cpu_exposure"]),
            memory_exposure=float(value["memory_exposure"]),
            gpu_exposure=float(value["gpu_exposure"]),
            priority_boost_task_classes=tuple(value.get("priority_boost_task_classes", [])),
            priority_boost=float(value.get("priority_boost", 1.0)),
        )
        for name, value in payload["operation_modes"].items()
    }
    task_classes = {
        name: TaskClassConfig(
            name=name,
            arrival_weight=float(value["arrival_weight"]),
            base_value=float(value["base_value"]),
            deadline_ms=_tuple(value["deadline_ms"]),
            input_size_mbit=_tuple(value["input_size_mbit"]),
            output_size_mbit=_tuple(value["output_size_mbit"]),
            cpu_required=_tuple(value["cpu_required"]),
            memory_required=_tuple(value["memory_required"]),
            gpu_required=_tuple(value["gpu_required"]),
            hard_deadline_probability=float(value["hard_deadline_probability"]),
            deadline_sensitivity=float(value["deadline_sensitivity"]),
            service_time_ms=_tuple(value["service_time_ms"]),
            priority_range=(int(value["priority_range"][0]), int(value["priority_range"][1])),
        )
        for name, value in payload["task_classes"].items()
    }
    qmix = payload["qmix"]
    qmix_config = QMIXConfig(
        hidden_dim=int(qmix["hidden_dim"]),
        mixing_hidden_dim=int(qmix["mixing_hidden_dim"]),
        learning_rate=float(qmix["learning_rate"]),
        gamma=float(qmix["gamma"]),
        epsilon_start=float(qmix["epsilon_start"]),
        epsilon_end=float(qmix["epsilon_end"]),
        epsilon_decay_steps=int(qmix["epsilon_decay_steps"]),
        replay_capacity=int(qmix["replay_capacity"]),
        batch_size=int(qmix["batch_size"]),
        target_update_interval=int(qmix["target_update_interval"]),
        training_episodes=int(qmix["training_episodes"]),
        checkpoint_dir=str(qmix["checkpoint_dir"]),
    )
    return ExperimentConfig(
        defaults=simulation_defaults,
        sizes=sizes,
        edge_node_classes=edge_node_classes,
        operation_modes=operation_modes,
        task_classes=task_classes,
        scenarios={str(k): dict(v) for k, v in payload["scenarios"].items()},
        qmix=qmix_config,
    )
