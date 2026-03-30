# TIPE: Bounded and Signature-Aware Path Construction in Genome Graphs

This repository provides a reference implementation of **TIPE (Tag-Based Iterative Path Expansion)**, a framework for controlled and scalable path construction in directed graphs, with applications to genome graph analysis.

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
