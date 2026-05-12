# AMPLE-AMR

Reproducible simulation and experiment framework for **AMPLE-AMR** and
**C-AMPLE-AMR** in a warehouse domain with autonomous mobile robots (AMR),
wireless access points, and heterogeneous edge nodes.

The repository implements:

- a configurable warehouse AMR simulation environment;
- a domain model for robots, tasks, edge nodes, access points, and network state;
- three allocation layers: heuristic, VCG-like, and clustered VCG-like;
- robot-level VCG-like payments together with task-level diagnostic externalities;
- a QMIX-based controller for **edge node operation modes only**;
- experiment orchestration, CSV export, plots, and LaTeX tables;
- unit and integration tests for reproducibility and core constraints.

## Scope and method boundaries

This prototype follows the dissertation constraints explicitly:

- **QMIX selects only operation modes** of edge nodes.
- **QMIX does not learn bids, task values, payments, or bidding policies**.
- **Task allocation is always performed by a separate allocation layer**:
  heuristic, VCG-like, or clustered VCG-like.
- There is **no "QMIX only" mode** without an allocation layer.
- The VCG-like layer keeps **robot-level payments** for compatibility and
  exposes **task-level diagnostic externalities** for analysis.
- `C-AMPLE-AMR` uses clustering for localized allocation, but does **not**
  implement cluster leader/orchestrator selection.

## Implemented methods

The benchmark matrix includes:

| Method | Node modes | Allocation layer |
| --- | --- | --- |
| `fixed_heuristic` | fixed (`normal`) | heuristic (`min_latency`) |
| `fixed_auction` | fixed (`normal`) | auction / VCG-like SW optimizer |
| `qmix_heuristic` | QMIX | heuristic (`min_latency`) |
| `ample_amr` | QMIX | VCG-like |
| `c_ample_amr` | QMIX | clustered VCG-like |
| `random_modes_heuristic` | random | heuristic |

## Supported warehouse scales

The scale presets are defined in
[`configs/warehouse_experiments.yaml`](configs/warehouse_experiments.yaml):

| Size | Robots | Access points | Edge nodes |
| --- | ---: | ---: | ---: |
| `Warehouse-S` | 6 | 7 | 4 |
| `Warehouse-M` | 10 | 25 | 5 |
| `Warehouse-M+` | 10 | 25 | 13 |
| `Warehouse-L` | 20 | 40 | 10 |
| `Warehouse-XL` | 40 | 60 | 20 |

## Supported scenarios

- `stable_warehouse_load`
- `peak_warehouse_load`
- `heterogeneous_edge_nodes`
- `network_degradation`
- `edge_node_failures`
- `scalability_sweep`
- `clustered_vs_global`
- `sensitivity_operation_modes`

QMIX training defaults to `stable_warehouse_load`. To train on a scenario
mixture, set `qmix.training_scenarios` in
[`configs/warehouse_experiments.yaml`](configs/warehouse_experiments.yaml).

## Repository layout

```text
.
├── configs/                      # YAML experiment configuration
├── experiments/
│   ├── checkpoints/             # QMIX checkpoints
│   └── results/                 # CSV and plot outputs
├── src/ample_amr/
│   ├── allocation.py            # Heuristic / VCG-like / clustered VCG-like allocators
│   ├── config.py                # Dataclasses and YAML loading
│   ├── domain.py                # Robot / Task / EdgeNode / AccessPoint model
│   ├── environment.py           # SimulationEnvironment
│   ├── methods.py               # Method matrix and controller/allocator builders
│   ├── network.py               # NetworkGraph and link generation
│   ├── outputs.py               # CSV, plots, LaTeX export
│   ├── qmix.py                  # QMIX mode controller
│   ├── runner.py                # Experiment orchestration
│   ├── utility.py               # Utility/cost/welfare model
│   └── workload.py              # Task generation
├── tables/                      # Exported LaTeX tables
├── tests/                       # Unit and integration tests
└── main_run_experiments.py      # Single CLI entrypoint
```

## Requirements

- Python **3.11+**
- `pip`
- CPU or GPU build of PyTorch compatible with your platform

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional editable install:

```bash
python3 -m pip install -e .
```

## Quick start

### 1. Smoke test

Runs a short benchmark, trains missing QMIX checkpoints if needed, and prints
generated files.

```bash
python3 main_run_experiments.py \
  --config configs/warehouse_experiments.yaml \
  --quick \
  --seeds 0 \
  --methods fixed_heuristic fixed_auction qmix_heuristic ample_amr c_ample_amr \
  --scenarios stable_warehouse_load peak_warehouse_load heterogeneous_edge_nodes \
              network_degradation edge_node_failures scalability_sweep \
              clustered_vs_global sensitivity_operation_modes
```

### 2. Full training and evaluation

```bash
python3 main_run_experiments.py \
  --config configs/warehouse_experiments.yaml \
  --train \
  --seeds 0 1 2 3 4 \
  --methods fixed_heuristic fixed_auction qmix_heuristic ample_amr c_ample_amr \
  --scenarios stable_warehouse_load peak_warehouse_load heterogeneous_edge_nodes \
              network_degradation edge_node_failures scalability_sweep \
              clustered_vs_global sensitivity_operation_modes
```

### 3. Evaluation only with existing checkpoints

```bash
python3 main_run_experiments.py \
  --config configs/warehouse_experiments.yaml \
  --eval \
  --seeds 0 1 2 3 4 \
  --methods qmix_heuristic ample_amr c_ample_amr \
  --scenarios stable_warehouse_load peak_warehouse_load
```

### 4. Restrict to one warehouse size

```bash
python3 main_run_experiments.py \
  --config configs/warehouse_experiments.yaml \
  --quick \
  --scenario-size Warehouse-M \
  --scenarios stable_warehouse_load network_degradation
```

### 5. Regenerate plots from existing CSV

```bash
python3 main_run_experiments.py --plot
```

### 6. Regenerate LaTeX tables from existing CSV

```bash
python3 main_run_experiments.py --export-latex
```

## Main outputs

After a full run the framework writes:

### CSV

- `experiments/results/raw_steps.csv`
- `experiments/results/episodes.csv`
- `experiments/results/summary_by_seed.csv`
- `experiments/results/summary.csv`
- `experiments/results/scalability.csv`
- `experiments/results/clustered_vs_global.csv`
- `experiments/results/sensitivity_operation_modes.csv`
- `experiments/results/training_history.csv`

Step-level and episode-level CSV outputs also include task-externality
diagnostics such as `externality_mean`, `externality_p95`,
`externality_max`, `payment_robot_sum`, and `payment_task_sum`.

### Plots

- `experiments/results/welfare_by_scenario.{png,pdf}`
- `experiments/results/completion_drop_violation_by_scenario.{png,pdf}`
- `experiments/results/latency_p95_by_scenario.{png,pdf}`
- `experiments/results/overhead_scalability.{png,pdf}`
- `experiments/results/global_vs_clustered_tradeoff.{png,pdf}`
- `experiments/results/node_utilization_heatmap.{png,pdf}`
- `experiments/results/learning_curves.{png,pdf}`
- `experiments/results/operation_mode_distribution.{png,pdf}`

### LaTeX tables

- `tables/stable_results.tex`
- `tables/peak_results.tex`
- `tables/heterogeneous_results.tex`
- `tables/network_degradation_results.tex`
- `tables/failure_results.tex`
- `tables/scalability_results.tex`
- `tables/clustered_results.tex`
- `tables/sensitivity_operation_modes.tex`

## Reproducibility

- All scenario parameters are stored in YAML.
- Randomness is seed-controlled.
- Tests cover seed reproducibility, capacity constraints, deadline feasibility,
  QMIX action validity, and the absence of unsupported "QMIX only" execution.
- The same CLI is used for training, evaluation, plot generation, and table export.

## Running tests

```bash
pytest -q
```

## Notes

- The framework is oriented toward dissertation validation in a **warehouse AMR**
  domain, while the AMPLE-AMR formulation itself is broader than this domain.
- The default heuristic baseline is `min_latency`, while social welfare is
  evaluated after assignment.
- The VCG-like allocator is deterministic for a fixed seed.
- Task-level externalities are intended for CSV diagnostics and dissertation
  analysis rather than as standalone market payments.
- The clustered allocator is intended as a scalability mechanism, not as a
  universal replacement for global allocation.
