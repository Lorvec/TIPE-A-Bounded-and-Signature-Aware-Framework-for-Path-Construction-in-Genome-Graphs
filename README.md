---

## Overview

Graph-based representations are increasingly used to model genomic variation. However, enumerating paths in such graphs is inherently challenging due to the combinatorial growth of possible traversals.

TIPE addresses this problem by:
- decoupling **path expansion** from **path retention**
- enforcing **bounded retention per signature class**
- enabling controlled exploration without materializing the full exponential set of candidate paths

The framework is designed to provide a principled and interpretable approach to bounded path construction.

This repository contains reproducible experiments corresponding to the synthetic and real-graph evaluations presented in the associated manuscript.

---

## Repository contents

- `run_real_ecoli_validation.py`  
  Runs TIPE on a real genome graph and produces cumulative metrics and scaling behavior (paths, pruning, runtime).

- `run_synthetic_recovery_experiment.py`  
  Synthetic experiment evaluating recovery of held-out paths under controlled branching structure.

- `run_real_anchor_recovery_experiment.py`  
  Real-graph experiment evaluating recovery on high-branching neighborhoods ("hard anchors").

---

## Requirements

- Python 3.9+
- `matplotlib`
- `pandas`

Install dependencies:

```bash
pip install matplotlib pandas
Data

This repository evaluates the TIPE framework on genome graphs in GFA-derived form.

The specific E. coli graph used in our experiments is not distributed with this repository.

To run the real-graph validation script, any GFA-formatted genome graph can be used. Publicly available pangenome graph resources include:

https://github.com/vgteam/vg
https://github.com/pangenome/pggb

Place your GFA file in the repository root and rename it to:

EcoliGraph_MGC.gfa
Usage
1. Real graph validation
python run_real_ecoli_validation.py
2. Synthetic held-out recovery experiment
python run_synthetic_recovery_experiment.py
3. Real hard-anchor recovery experiment
python run_real_anchor_recovery_experiment.py
Output
Real graph validation

Outputs are saved in:

real_ecoli_results/

Includes:

real_ecoli_validation.csv
fig_real_paths_vs_k.png
fig_real_retention_pruning_vs_k.png
fig_real_cycle_reject_vs_k.png
fig_real_runtime_vs_k.png
Synthetic recovery experiment

Outputs are saved in:

synthetic_recovery_results/

Includes:

synthetic_recovery_results.csv
fig_synthetic_read_recall_vs_budget.png
fig_synthetic_relaxed_path_recall_vs_budget.png
synthetic_candidate_paths.txt
Real anchor recovery experiment

Outputs are saved in:

real_anchor_recovery_results/

Includes:

real_anchor_recovery_raw.csv
real_anchor_recovery_summary.csv
real_anchor_debug.csv
fig_real_anchor_read_recall_vs_budget.png
fig_real_anchor_relaxed_path_recall_vs_budget.png
Notes
The real-graph validation experiment operates on a prefix-induced subgraph of fixed size.
All experiments enforce simple paths (no repeated nodes).
Retention is bounded per signature class.
The synthetic and real-anchor experiments illustrate behavior under high branching and partial observation.
Reproducibility

All experiments are deterministic under the default settings provided in the scripts (fixed random seeds).

The repository is intended as a minimal and transparent reference implementation of the TIPE framework used in the associated manuscript.

License

Add a license file (e.g., MIT License) if you plan to distribute or reuse this code.
