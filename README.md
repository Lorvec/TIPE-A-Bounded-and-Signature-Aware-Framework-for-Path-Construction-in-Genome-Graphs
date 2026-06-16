# TIPE: Bounded and Signature-Aware Path Construction in Genome Graphs

This repository contains a reference implementation of **TIPE** (*Tag-Based Iterative Path Expansion*), a framework for controlled path construction in directed graphs, with a focus on genome graph applications.

## Overview

Graph-based representations are widely used to model genomic variation, but explicit path enumeration quickly becomes infeasible due to combinatorial growth.

TIPE separates **path expansion** from **path retention**. At each iteration, paths are expanded locally, but only a bounded number of representative paths is retained per structural signature. This allows path construction to remain controlled without materializing the full exponential path space.

The code in this repository reproduces the main experiments reported in the manuscript, including synthetic dense-branching experiments, real *E. coli* pangenome graph validation, held-out path recovery, and output-size-matched comparisons with bounded traversal baselines.

## Repository contents

### `run_synthetic_experiment.py`

Runs the synthetic dense layered branching experiment.

This script reproduces the synthetic path-growth, pruning, and runtime experiments corresponding to the dense layered graph setting. The default graph has:

* 12 layers
* 6 nodes per layer
* 73 nodes
* 402 directed edges
* maximum path length `K = 12`

The exhaustive path space is analytically known:

* exact exhaustive paths at `K = 12`: `2,176,782,336`
* cumulative exhaustive paths up to `K = 12`: `2,612,138,802`

The script also reports TIPE retained-path counts under budgets `B = 1`, `B = 5`, and `B = 10`.

Main outputs:

* `synthetic_dense_large/synthetic_dense_large_results.csv`
* `synthetic_dense_large/synthetic_dense_large_summary.csv`
* `synthetic_dense_large/fig_dense_large_paths_vs_k.png`
* `synthetic_dense_large/fig_dense_large_pruning_vs_k.png`
* `synthetic_dense_large/fig_dense_large_runtime_vs_k.png`

### `run_real_ecoli_validation.py`

Runs TIPE on a real *E. coli* pangenome graph in GFA format.

This script reproduces the real-graph validation experiment using a prefix-induced subgraph of the full GFA graph. It reports retained paths and retention-induced pruning behavior under different signature budgets.

Main outputs include CSV files and figures for:

* cumulative retained paths
* retention-induced pruning rate
* real *E. coli* graph validation plots

### `real_ecoli_recovery_and_comparison.py`

Runs the held-out recovery and output-size-matched comparison experiments on high-branching *E. coli* graph neighborhoods.

This script covers:

* real anchor held-out recovery experiments
* output-size-matched comparison between TIPE, random pruning, and beam-style retention
* Table 2-style summary statistics

For the output-size-matched comparison, the default setting uses:

* maximum path length `K = 8`
* TIPE reference budgets `B = 1`, `B = 3`, and `B = 5`
* 20 high-branching graph neighborhoods
* 30 random seeds for random pruning

Main outputs:

* `table2_matched_K8_N20_R30_seed42/table2_matched_raw.csv`
* `table2_matched_K8_N20_R30_seed42/table2_matched_summary.csv`
* `table2_matched_K8_N20_R30_seed42/table2_matched_printable.csv`
* `table2_matched_K8_N20_R30_seed42/table2_debug.csv`

The manuscript Table 2 reports the following metrics from this experiment:

* number of retained paths
* exact recall
* signature diversity

Additional auxiliary metrics may also be saved by the script for debugging or extended analysis.

## Requirements

Python 3.9 or newer is recommended.

Required packages:

```bash
pip install matplotlib pandas
```

The scripts also use Python standard-library modules such as `csv`, `time`, `pathlib`, and `collections`.

## Data

The full *E. coli* GFA graph used in the manuscript is not included in this repository.

To run the real-graph experiments, place the GFA file in the project folder and name it:

```text
EcoliGraph_MGC.gfa
```

The manuscript experiments used the public `EcoliGraph_MGC.gfa` file from the DataSuds dataset associated with the GrAnnoT study.

The real-graph scripts construct a prefix-induced subgraph from the GFA file for computationally controlled validation.

## Usage

Run the scripts directly from the repository root:

```bash
python run_synthetic_experiment.py
python run_real_ecoli_validation.py
python real_ecoli_recovery_and_comparison.py
```

Each script writes its outputs to a separate results folder.

## Notes

* Paths are treated as simple paths, meaning that node revisitation is not allowed.
* Retention is bounded per signature class.
* Synthetic experiments are designed as controlled path-space stress tests.
* Real-graph experiments use a prefix-induced subgraph of the full *E. coli* pangenome GFA.
* Output-size-matched comparisons retain the same number of active paths across methods at each iteration.

## Reproducibility

All experiments use fixed random seeds where stochastic sampling is involved. Results should therefore be reproducible across runs, subject to small differences due to runtime environment and dependency versions.

## Reference

If you use this code, please cite the associated manuscript:

**TIPE: A Bounded and Signature-Aware Framework for Path Construction in Genome Graphs**



