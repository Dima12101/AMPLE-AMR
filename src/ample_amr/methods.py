"""Experiment method definitions and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .allocation import BaseAllocator, ClusteredVCGLikeAllocator, HeuristicAllocator, VCGLikeAllocator
from .config import ExperimentConfig, WarehouseScenarioConfig
from .qmix import FixedModeController, QMIXModeController, RandomModeController


@dataclass(frozen=True)
class MethodSpec:
    """Definition of a benchmark method."""

    key: str
    uses_qmix: bool
    allocator_key: str
    fixed_mode: str | None = None


METHOD_SPECS: dict[str, MethodSpec] = {
    "fixed_heuristic": MethodSpec(key="fixed_heuristic", uses_qmix=False, allocator_key="heuristic", fixed_mode="normal"),
    "fixed_auction": MethodSpec(key="fixed_auction", uses_qmix=False, allocator_key="vcg_like", fixed_mode="normal"),
    "qmix_heuristic": MethodSpec(key="qmix_heuristic", uses_qmix=True, allocator_key="heuristic"),
    "ample_amr": MethodSpec(key="ample_amr", uses_qmix=True, allocator_key="vcg_like"),
    "c_ample_amr": MethodSpec(key="c_ample_amr", uses_qmix=True, allocator_key="clustered_vcg_like"),
    "random_modes_heuristic": MethodSpec(key="random_modes_heuristic", uses_qmix=False, allocator_key="heuristic"),
}

METHOD_ALIASES = {
    "fixed_vcg": "fixed_auction",
}


def resolve_method_key(method_key: str) -> str:
    """Resolve deprecated method aliases to their canonical keys."""

    return METHOD_ALIASES.get(method_key, method_key)


def build_allocator(method_key: str) -> BaseAllocator:
    """Build the allocator for an experiment method."""

    method_key = resolve_method_key(method_key)
    spec = METHOD_SPECS[method_key]
    if spec.allocator_key == "heuristic":
        return HeuristicAllocator(policy="min_latency")
    if spec.allocator_key == "vcg_like":
        return VCGLikeAllocator()
    if spec.allocator_key == "clustered_vcg_like":
        return ClusteredVCGLikeAllocator()
    raise KeyError(f"Unknown allocator key '{spec.allocator_key}'")


def build_mode_controller(
    method_key: str,
    scenario: WarehouseScenarioConfig,
    experiment_config: ExperimentConfig,
    seed: int,
    allow_missing_checkpoint: bool = False,
) -> FixedModeController | RandomModeController | QMIXModeController:
    """Create the appropriate node mode controller for a method."""

    method_key = resolve_method_key(method_key)
    spec = METHOD_SPECS[method_key]
    if method_key == "random_modes_heuristic":
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
