TIPE: Bounded and Signature-Aware Path Construction in Genome Graphs

This repository provides a reference implementation of TIPE (Tag-Based Iterative Path Expansion), a framework for controlled and scalable path construction in directed graphs, with applications to genome graph analysis.

Overview

Graph-based representations are increasingly used to model genomic variation. However, enumerating paths in such graphs is challenging due to combinatorial explosion.

TIPE addresses this problem by:

decoupling path expansion from path retention,
enforcing bounded retention per signature class,
enabling controlled exploration without materializing all candidate paths.

This repository contains a reproducible experiment on a real E. coli pangenome graph.

Repository contents
run_real_ecoli_validation.py
Runs TIPE on a real genome graph and produces:
CSV with metrics
figures for retained paths, pruning rates, and runtime
Requirements
Python 3.9+
matplotlib

Install dependencies:

pip install matplotlib

Data

This repository evaluates the TIPE framework on a real genome graph in GFA format.

The specific E. coli graph used in our experiments is not distributed with this repository.

To run the code, any GFA-formatted genome graph can be used.

Users may provide their own GFA file or obtain publicly available pangenome graphs from resources such as:

- https://github.com/vgteam/vg
- https://github.com/pangenome/pggb

Place the GFA file in the repository root and rename it to:

EcoliGraph_MGC.gfa

Usage

Run the experiment:

python run_real_ecoli_validation.py
Output

The script generates:

real_ecoli_validation.csv
→ cumulative metrics per path length
fig_real_paths_vs_k.png
→ retained paths vs K
fig_real_retention_pruning_vs_k.png
→ retention pruning rate
fig_real_cycle_reject_vs_k.png
→ cycle/invalid rejection rate
fig_real_runtime_vs_k.png
→ runtime vs K

All outputs are saved in:

real_ecoli_results/
Notes
The graph is parsed as a prefix-induced subgraph of fixed size.
Path simplicity is enforced (no repeated nodes).
Retention is bounded per signature class (start, end).
Reproducibility

This repository provides a minimal, self-contained implementation of the TIPE framework used in the associated manuscript.
