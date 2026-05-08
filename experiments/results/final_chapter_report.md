# Final Chapter Artifact Pack

## Run Metadata

- Date: 2026-05-08
- Command:
  `python3 main_run_experiments.py --config configs/warehouse_experiments.yaml --train --seeds 0 1 2 3 4 --methods fixed_heuristic fixed_vcg qmix_heuristic ample_amr c_ample_amr random_modes_heuristic --scenarios stable_warehouse_load peak_warehouse_load heterogeneous_edge_nodes network_degradation edge_node_failures scalability_sweep clustered_vs_global sensitivity_operation_modes`
- End-to-end duration: `828.15 s` (`13.80 min`)
- Seeds: `0 1 2 3 4`
- Methods: `fixed_heuristic`, `fixed_vcg`, `qmix_heuristic`, `ample_amr`, `c_ample_amr`, `random_modes_heuristic`
- Scenarios: all chapter-8 scenarios from the prompt
- QMIX training: `120` rows in `training_history.csv` (`24` episodes for each of `Warehouse-S/M/M+/L/XL`)
- QMIX training scenario set: `stable_warehouse_load`

## Verification

- `pytest -q`: passed (`18 passed`)
- `summary_by_seed.csv`: `420` rows, seeds `0..4`
- `ch8_overall_summary.csv`: `84` rows
- `ch8_hypothesis_checks.csv`: `7` rows
- Dissertation mirrors updated:
  - `32` chapter-8 figures in `Dissertation/images/ample_amr/ch8_*`
  - `10` chapter-8 tables in `Dissertation/tables/ch8_*`

## Hypothesis Status

| Hypothesis | Status | Observed effect | Comment |
| --- | --- | ---: | --- |
| H1 | `not_confirmed` | `0.116` | `fixed_vcg` did not produce a stable welfare gain over `fixed_heuristic`; positive effect only in `5.7%` of pairwise comparisons. |
| H2 | `confirmed` | `100.598` | `qmix_heuristic` consistently outperformed `fixed_heuristic` by welfare across the evaluated scenario set. |
| H3 | `partially_confirmed` | `50.062` | `ample_amr` improved over the single-component baselines on average, but not in a dominant share of pairwise comparisons. |
| H4 | `partially_confirmed` | `0.021` | Externality-to-utilization correlation is positive but weak; diagnostic value exists, yet the signal is not strong in the current parameterization. |
| H5 | `confirmed` | `78.258` | In `peak_warehouse_load`, `ample_amr` increased welfare and reduced both drop rate and p95 latency versus `fixed_heuristic`. |
| H6 | `confirmed` | `36.663` | In `network_degradation` and `edge_node_failures`, `ample_amr` improved welfare and latency relative to `fixed_heuristic`. |
| H7 | `not_confirmed` | `-0.864` | `c_ample_amr` increased, rather than reduced, allocation overhead on the current clustered configurations. |

## Scenario Highlights

- `stable_warehouse_load`: `ample_amr` vs `fixed_heuristic` gave `+123.998` welfare, `-17.470 ms` p95 latency, and `+0.0086` completion rate.
- `peak_warehouse_load`: `ample_amr` vs `fixed_heuristic` gave `+210.516` welfare, `-24.242 ms` p95 latency, and `-0.0143` drop rate.
- `heterogeneous_edge_nodes`: `ample_amr` vs `fixed_heuristic` gave `+95.620` welfare, `+0.0372` completion rate, and `-8.624 ms` p95 latency.
- `network_degradation`: `ample_amr` vs `fixed_heuristic` gave `+94.942` welfare and `-4.288 ms` p95 latency, but completion changed only marginally (`-0.0013`).
- `edge_node_failures`: `ample_amr` vs `fixed_heuristic` gave `+110.733` welfare and `-10.010 ms` p95 latency.
- `clustered_vs_global`: `c_ample_amr` vs `ample_amr` gave `-5.596` welfare and `+4.267 ms` total decision overhead; the scenario-level relative overhead reduction is strongly negative (`-1.505`), which explains the rejection of H7.

## Interpretation Notes for Chapter 8

- The strongest confirmed result is H2: adaptive mode control contributes the largest consistent gain across scenarios.
- H1 is not confirmed because `fixed_vcg` and `fixed_heuristic` remain very close under the current workload and capacity settings; the auction layer alone does not create a robust welfare advantage in these configurations.
- H3 is only partial because, in the current experimental contour, `ample_amr` often matches `qmix_heuristic` numerically on core metrics. This suggests the learned mode policy contributes more than the VCG-like assignment layer in the tested settings.
- H4 is only partial because task-level externality signals are present, but weak. For a stronger chapter argument, it may be worth increasing resource contention or tightening exposed capacity in the congestion-heavy scenarios.
- H7 is the main negative result of the series: the present clustering design does not yet pay off in overhead reduction. This is chapter-worthy and should be stated explicitly rather than hidden.

## Primary Artifacts

- CSV:
  - `experiments/results/ch8_overall_summary.csv`
  - `experiments/results/ch8_externality_diagnostics.csv`
  - `experiments/results/ch8_payments_by_task_class.csv`
  - `experiments/results/ch8_mode_distribution.csv`
  - `experiments/results/ch8_hypothesis_checks.csv`
- LaTeX:
  - `tables/ch8_overall_summary.tex`
  - `tables/ch8_stable_load_results.tex`
  - `tables/ch8_peak_load_results.tex`
  - `tables/ch8_heterogeneous_results.tex`
  - `tables/ch8_network_degradation_results.tex`
  - `tables/ch8_failures_results.tex`
  - `tables/ch8_scalability_results.tex`
  - `tables/ch8_clustered_vs_global_results.tex`
  - `tables/ch8_sensitivity_operation_modes.tex`
  - `tables/ch8_externality_diagnostics.tex`
- Dissertation figures:
  - `Dissertation/images/ample_amr/ch8_*.pdf`
  - `Dissertation/images/ample_amr/ch8_*.png`

## Caveats

- The full run is now reproducible and complete, but the experimental evidence indicates that the clustered variant and the VCG-like layer need stronger stress conditions or revised configuration to support stronger claims in the dissertation text.
- `random_modes_heuristic` was retained as a diagnostic baseline; it supports the interpretation that the observed gains are not caused by arbitrary mode switching alone.
