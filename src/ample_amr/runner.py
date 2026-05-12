"""Experiment runner orchestration for AMPLE-AMR benchmarks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, OperationModeConfig, WarehouseScenarioConfig
from .environment import SimulationEnvironment
from .methods import MethodSpec, build_allocator, build_mode_controller, get_method_spec, unique_method_keys
from .outputs import export_latex_tables, generate_plots, save_result_frames
from .qmix import QMIXModeController
from .utils import ensure_directory


class ExperimentRunner:
    """High-level driver for training, evaluation and artifact generation."""

    def __init__(self, experiment_config: ExperimentConfig, config_path: str | Path) -> None:
        self.experiment_config = experiment_config
        self.config_path = Path(config_path)
        self.results_dir = ensure_directory("experiments/results")
        self.tables_dir = ensure_directory("tables")

    def run(
        self,
        methods: list[str],
        scenarios: list[str],
        seeds: list[int],
        scenario_size: str | None = None,
        quick: bool = False,
        train: bool = False,
        eval_only: bool = False,
        plot_only: bool = False,
        export_latex_only: bool = False,
    ) -> dict[str, list[Path] | Path]:
        """Run the requested experiment workflow."""

        methods = unique_method_keys(methods)
        method_specs = {method: get_method_spec(method) for method in methods}
        outputs: dict[str, list[Path] | Path] = {}
        if plot_only or export_latex_only:
            if plot_only:
                outputs["plots"] = generate_plots(self.results_dir)
            if export_latex_only:
                outputs["tables"] = export_latex_tables(self.results_dir, self.tables_dir)
            return outputs

        required_sizes = self._required_qmix_sizes(method_specs, scenarios, scenario_size, quick)
        missing_sizes = self._missing_qmix_sizes(required_sizes)
        if train or (missing_sizes and not eval_only):
            training_history = self._train_qmix(required_sizes, quick=quick, seed=seeds[0] if seeds else 0)
        else:
            training_history = pd.DataFrame(
                columns=["size_name", "training_scenario", "episode", "reward", "loss", "epsilon"]
            )
        if missing_sizes and eval_only and any(spec.uses_qmix for spec in method_specs.values()):
            missing_text = ", ".join(sorted(missing_sizes))
            raise FileNotFoundError(f"Missing QMIX checkpoints for sizes: {missing_text}. Run with --train first.")

        step_records: list[dict[str, object]] = []
        episode_records: list[dict[str, object]] = []
        task_records: list[dict[str, object]] = []
        for scenario_name in scenarios:
            size_names = self._scenario_size_names(scenario_name, scenario_size)
            for size_name in size_names:
                resolved = self.experiment_config.resolve_scenario(scenario_name, size_override=size_name, quick=quick)
                for profile_name, scenario in self._iter_scenario_profiles(resolved):
                    for method in methods:
                        spec = method_specs[method]
                        allocator = build_allocator(method)
                        for seed in seeds:
                            controller = build_mode_controller(
                                method,
                                scenario,
                                self.experiment_config,
                                seed,
                                allow_missing_checkpoint=not spec.uses_qmix,
                            )
                            environment = SimulationEnvironment(scenario, seed)
                            raw_steps, episode_summary, task_rows = environment.run_episode(controller, allocator, explore=False)
                            for row in raw_steps:
                                row["method"] = method
                                row["mode_profile"] = profile_name
                                step_records.append(row)
                            episode_summary["method"] = method
                            episode_summary["mode_profile"] = profile_name
                            episode_records.append(episode_summary)
                            for row in task_rows:
                                row["method"] = method
                                row["mode_profile"] = profile_name
                                task_records.append(row)

        raw_steps = pd.DataFrame(step_records)
        episodes = pd.DataFrame(episode_records)
        task_diagnostics = pd.DataFrame(task_records)
        saved = save_result_frames(self.results_dir, raw_steps, episodes, training_history, task_diagnostics)
        outputs["results"] = list(saved.values())
        outputs["plots"] = generate_plots(self.results_dir)
        outputs["tables"] = export_latex_tables(self.results_dir, self.tables_dir)
        return outputs

    def _required_qmix_sizes(
        self,
        method_specs: dict[str, MethodSpec],
        scenarios: list[str],
        scenario_size: str | None,
        quick: bool,
    ) -> set[str]:
        if not any(spec.uses_qmix for spec in method_specs.values()):
            return set()
        sizes: set[str] = set()
        for scenario_name in scenarios:
            sizes.update(self._scenario_size_names(scenario_name, scenario_size))
        return sizes

    def _missing_qmix_sizes(self, size_names: set[str]) -> set[str]:
        missing: set[str] = set()
        for size_name in size_names:
            checkpoint_path = Path(self.experiment_config.qmix.checkpoint_dir) / f"qmix_{size_name}.pt"
            if not checkpoint_path.exists():
                missing.add(size_name)
        return missing

    def _train_qmix(self, size_names: set[str], quick: bool, seed: int) -> pd.DataFrame:
        history_rows: list[dict[str, float | str]] = []
        episodes = min(4, self.experiment_config.qmix.training_episodes) if quick else self.experiment_config.qmix.training_episodes
        allocator = build_allocator("ample_amr")
        training_scenarios = self._training_scenario_names()
        for size_name in sorted(size_names):
            resolved_training_scenarios = [
                self.experiment_config.resolve_scenario(scenario_name, size_override=size_name, quick=quick)
                for scenario_name in training_scenarios
            ]
            controller_scenario = resolved_training_scenarios[0]
            checkpoint_path = Path(self.experiment_config.qmix.checkpoint_dir) / f"qmix_{size_name}.pt"
            controller = QMIXModeController(
                controller_scenario,
                self.experiment_config.qmix,
                checkpoint_path,
                seed,
                load_checkpoint=False,
            )
            history = controller.train(
                allocator=allocator,
                env_factory=lambda episode_seed, scenarios=resolved_training_scenarios, base_seed=seed: SimulationEnvironment(
                    scenarios[(episode_seed - base_seed) % len(scenarios)],
                    episode_seed,
                ),
                episodes=episodes,
            )
            for row in history:
                scenario_index = int(row["episode"]) % len(training_scenarios)
                history_rows.append(
                    {
                        "size_name": size_name,
                        "training_scenario": training_scenarios[scenario_index],
                        **row,
                    }
                )
        return pd.DataFrame(history_rows)

    def _training_scenario_names(self) -> list[str]:
        training_scenarios = [name for name in self.experiment_config.qmix.training_scenarios if name]
        return training_scenarios or ["stable_warehouse_load"]

    def _scenario_size_names(self, scenario_name: str, scenario_size: str | None) -> list[str]:
        if scenario_name == "scalability_sweep":
            return ["Warehouse-S", "Warehouse-M", "Warehouse-M+", "Warehouse-L", "Warehouse-XL"]
        resolved = self.experiment_config.resolve_scenario(scenario_name, size_override=scenario_size)
        return [resolved.size_name]

    def _iter_scenario_profiles(self, scenario: WarehouseScenarioConfig) -> list[tuple[str, WarehouseScenarioConfig]]:
        if scenario.scenario_name != "sensitivity_operation_modes" or not scenario.sensitivity_profiles:
            return [("default", scenario)]
        variants: list[tuple[str, WarehouseScenarioConfig]] = []
        for profile_name, profile_scalars in scenario.sensitivity_profiles.items():
            operation_modes: dict[str, OperationModeConfig] = {}
            for mode_name, mode in scenario.operation_modes.items():
                scalar = profile_scalars.get(mode_name, 1.0)
                operation_modes[mode_name] = replace(
                    mode,
                    cpu_exposure=max(0.05, min(0.95, mode.cpu_exposure * scalar)),
                    memory_exposure=max(0.05, min(0.95, mode.memory_exposure * scalar)),
                    gpu_exposure=max(0.05, min(0.95, mode.gpu_exposure * scalar)),
                )
            variants.append((profile_name, replace(scenario, operation_modes=operation_modes)))
        return variants
