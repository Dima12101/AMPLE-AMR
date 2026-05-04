"""Task generation utilities for warehouse AMR workloads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TaskClassConfig
from .domain import Task


@dataclass
class TaskGenerator:
    """Seeded task generator bound to a specific robot."""

    robot_id: str
    task_classes: dict[str, TaskClassConfig]
    task_mix: dict[str, float]
    arrival_rate_per_robot: float
    rng: np.random.Generator
    counter: int = 0

    def maybe_generate(self, now_ms: float, arrival_multiplier: float = 1.0) -> Task | None:
        """Generate at most one task per slot with a seeded Bernoulli process."""

        probability = min(0.98, self.arrival_rate_per_robot * arrival_multiplier)
        if float(self.rng.random()) > probability:
            return None
        task_class_name = self._sample_task_class()
        task_class = self.task_classes[task_class_name]
        self.counter += 1
        task_id = f"{self.robot_id}-task-{self.counter:05d}"
        hard_deadline = bool(self.rng.random() < task_class.hard_deadline_probability)
        priority = int(self.rng.integers(task_class.priority_range[0], task_class.priority_range[1] + 1))
        task = Task(
            id=task_id,
            robot_id=self.robot_id,
            task_class=task_class_name,
            arrival_time=now_ms,
            input_size_mbit=float(self.rng.uniform(*task_class.input_size_mbit)),
            output_size_mbit=float(self.rng.uniform(*task_class.output_size_mbit)),
            cpu_required=float(self.rng.uniform(*task_class.cpu_required)),
            memory_required=float(self.rng.uniform(*task_class.memory_required)),
            gpu_required=float(self.rng.uniform(*task_class.gpu_required)),
            deadline_ms=float(self.rng.uniform(*task_class.deadline_ms)),
            priority=priority,
            hard_deadline=hard_deadline,
            metadata={
                "base_value": task_class.base_value,
                "deadline_sensitivity": task_class.deadline_sensitivity,
                "service_time_ms": float(self.rng.uniform(*task_class.service_time_ms)),
            },
        )
        return task

    def _sample_task_class(self) -> str:
        """Sample a task class from the configured workload mix."""

        names = list(self.task_mix)
        weights = np.asarray([self.task_mix[name] for name in names], dtype=float)
        weights = weights / weights.sum()
        index = int(self.rng.choice(len(names), p=weights))
        return names[index]
