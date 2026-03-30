# TIPE: Bounded and Signature-Aware Path Construction in Genome Graphs

This repository contains a reference implementation of TIPE (Tag-Based Iterative Path Expansion), a framework for controlled path construction in directed graphs, with a focus on genome graph applications.

## Overview

Graph-based representations are widely used to model genomic variation, but path enumeration quickly becomes infeasible due to combinatorial growth.

TIPE is a simple idea: we separate path expansion from path retention, and keep only a limited number of paths per structural signature. This allows exploration to remain bounded without explicitly generating all possible paths.

The code here reproduces the main experiments used in the paper, both on synthetic examples and on real graph data.

## Contents

- run_real_ecoli_validation.py  
  Runs TIPE on a real genome graph and outputs basic scaling metrics (paths, pruning, runtime)

- run_synthetic_recovery_experiment.py  
  Small synthetic experiment showing how TIPE recovers held-out paths under branching

- run_real_anchor_recovery_experiment.py  
  Similar idea, but applied to real graph neighborhoods (high branching regions)

## Requirements

Python 3.9 or newer

Packages:
- matplotlib
- pandas

Install with:
pip install matplotlib pandas

## Data

The E. coli graph used in the paper is not included.

To run the real experiment, you can use any GFA genome graph. For example:
https://github.com/vgteam/vg  
https://github.com/pangenome/pggb  

Put the GFA file in the project folder and rename it to:

EcoliGraph_MGC.gfa

## Usage

Run the scripts directly:

python run_real_ecoli_validation.py  
python run_synthetic_recovery_experiment.py  
python run_real_anchor_recovery_experiment.py  

## Output

Each script writes its results to a separate folder:

- real_ecoli_results/
- synthetic_recovery_results/
- real_anchor_recovery_results/

These include CSV files with metrics and a few simple plots.

## Notes

- Paths are always simple (no repeated nodes)
- Retention is bounded per signature
- The real experiments use a prefix subgraph of the full graph

## Reproducibility

All experiments use fixed random seeds, so results should be consistent across runs.

## Reference


