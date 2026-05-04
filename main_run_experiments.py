"""CLI entrypoint for AMPLE-AMR warehouse experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ample_amr.config import load_experiment_config
from ample_amr.runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AMPLE-AMR warehouse experiments.")
    parser.add_argument("--config", default="configs/warehouse_experiments.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fixed_heuristic", "fixed_vcg", "qmix_heuristic", "ample_amr", "c_ample_amr"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "stable_warehouse_load",
            "peak_warehouse_load",
            "heterogeneous_edge_nodes",
            "network_degradation",
            "edge_node_failures",
            "scalability_sweep",
            "clustered_vs_global",
            "sensitivity_operation_modes",
        ],
    )
    parser.add_argument("--quick", action="store_true", help="Run a smoke-test sized benchmark.")
    parser.add_argument("--train", action="store_true", help="Train QMIX policies before evaluation.")
    parser.add_argument("--eval", action="store_true", help="Evaluation only, using existing checkpoints.")
    parser.add_argument("--plot", action="store_true", help="Generate plots from existing CSV outputs.")
    parser.add_argument("--export-latex", action="store_true", help="Generate LaTeX tables from existing CSV outputs.")
    parser.add_argument(
        "--scenario-size",
        choices=["Warehouse-S", "Warehouse-M", "Warehouse-M+", "Warehouse-L", "Warehouse-XL"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(Path(args.config))
    runner = ExperimentRunner(config, args.config)
    outputs = runner.run(
        methods=args.methods,
        scenarios=args.scenarios,
        seeds=args.seeds,
        scenario_size=args.scenario_size,
        quick=args.quick,
        train=args.train,
        eval_only=args.eval,
        plot_only=args.plot and not args.train and not args.eval,
        export_latex_only=args.export_latex and not args.train and not args.eval,
    )
    if args.quick and "results" in outputs:
        print("Generated files:")
        for group_name in ("results", "plots", "tables"):
            for path in outputs.get(group_name, []):
                print(path)


if __name__ == "__main__":
    main()
