"""Experiment method definitions and construction helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from .allocation import BaseAllocator, ClusteredAuctionAllocator, HeuristicAllocator, AuctionAllocator
from .config import ExperimentConfig, WarehouseScenarioConfig
from .qmix import FixedModeController, QMIXModeController, RandomModeController

AllocatorKey = Literal["heuristic", "auction", "clustered_auction"]


@dataclass(frozen=True)
class MethodSpec:
    """Definition of a benchmark method."""

    key: str
    uses_qmix: bool
    allocator_key: AllocatorKey
    fixed_mode: str | None = None


METHOD_SPECS: dict[str, MethodSpec] = {
    "fixed_heuristic": MethodSpec(key="fixed_heuristic", uses_qmix=False, allocator_key="heuristic", fixed_mode="normal"),
    "fixed_auction": MethodSpec(key="fixed_auction", uses_qmix=False, allocator_key="auction", fixed_mode="normal"),
    "qmix_heuristic": MethodSpec(key="qmix_heuristic", uses_qmix=True, allocator_key="heuristic"),
    "ample_amr": MethodSpec(key="ample_amr", uses_qmix=True, allocator_key="auction"),
    "c_ample_amr": MethodSpec(key="c_ample_amr", uses_qmix=True, allocator_key="clustered_auction"),
    "random_modes_heuristic": MethodSpec(key="random_modes_heuristic", uses_qmix=False, allocator_key="heuristic"),
}

ALLOCATOR_BUILDERS: Final[dict[AllocatorKey, Callable[[], BaseAllocator]]] = {
    "heuristic": lambda: HeuristicAllocator(policy="min_latency"),
    "auction": AuctionAllocator,
    "clustered_auction": ClusteredAuctionAllocator,
}


def get_method_spec(method_key: str) -> MethodSpec:
    """Return the method spec for a canonical method key."""

    try:
        return METHOD_SPECS[method_key]
    except KeyError as exc:
        known_methods = ", ".join(sorted(METHOD_SPECS))
        raise KeyError(f"Unknown method '{method_key}'. Known methods: {known_methods}") from exc


def unique_method_keys(method_keys: Iterable[str]) -> list[str]:
    """Validate method keys and keep only the first occurrence of each method."""

    normalized_keys: list[str] = []
    seen: set[str] = set()
    for method_key in method_keys:
        normalized_key = get_method_spec(method_key).key
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        normalized_keys.append(normalized_key)
    return normalized_keys


def build_allocator(method_key: str) -> BaseAllocator:
    """Build the allocator for an experiment method."""

    spec = get_method_spec(method_key)
    try:
        return ALLOCATOR_BUILDERS[spec.allocator_key]()
    except KeyError as exc:
        raise KeyError(f"Unknown allocator key '{spec.allocator_key}'") from exc


def build_mode_controller(
    method_key: str,
    scenario: WarehouseScenarioConfig,
    experiment_config: ExperimentConfig,
    seed: int,
    allow_missing_checkpoint: bool = False,
) -> FixedModeController | RandomModeController | QMIXModeController:
    """Create the appropriate node mode controller for a method."""

    spec = get_method_spec(method_key)
    if spec.key == "random_modes_heuristic":
        return RandomModeController(list(scenario.operation_modes), seed + scenario.random_mode_baseline_seed_offset)
    if spec.fixed_mode is not None:
        return FixedModeController(spec.fixed_mode)
    checkpoint_path = Path(experiment_config.qmix.checkpoint_dir) / f"qmix_{scenario.size_name}.pt"
    controller = QMIXModeController(scenario, experiment_config.qmix, checkpoint_path, seed)
    if not checkpoint_path.exists() and not allow_missing_checkpoint:
        raise FileNotFoundError(
            f"QMIX checkpoint not found for {scenario.size_name}: {checkpoint_path}. Run with --train first."
        )
    return controller
