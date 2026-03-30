"""
Run a synthetic held-out path recovery experiment for TIPE.

This script evaluates bounded, signature-aware path construction on a small
structured toy graph designed to emulate high-branching variation patterns.
It compares:

- haplotype-only traversal
- TIPE with signature (start, end)
- TIPE with signature (start, second, end)

For each setting, the script reports:
- number of candidate complete paths
- read recall on held-out paths
- exact path recall
- relaxed path recall

Outputs:
- synthetic_recovery_results.csv
- fig_synthetic_read_recall_vs_budget.png
- fig_synthetic_relaxed_path_recall_vs_budget.png
- synthetic_candidate_paths.txt
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Experiment configuration
# ============================================================

READ_LEN = 4
ANCHOR = 0
END_NODE = 9
KMAX = 5
BUDGETS = [1, 2, 3, 5, 8, 10]

OUT_DIR = "synthetic_recovery_results"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "synthetic_recovery_results.csv")
FIG_READ_RECALL = os.path.join(OUT_DIR, "fig_synthetic_read_recall_vs_budget.png")
FIG_RELAXED_RECALL = os.path.join(
    OUT_DIR, "fig_synthetic_relaxed_path_recall_vs_budget.png"
)
OUT_CANDIDATES_TXT = os.path.join(OUT_DIR, "synthetic_candidate_paths.txt")


# ============================================================
# Toy graph definition
# ============================================================

NODE_LABELS: Dict[int, str] = {
    0: "L",
    1: "A1",
    2: "B1",
    3: "A2",
    4: "B2",
    5: "A3",
    6: "B3",
    7: "A4",
    8: "B4",
    9: "R",
}

SUCC: Dict[int, List[int]] = {
    0: [1, 2],
    1: [3, 4],
    2: [3, 4],
    3: [5, 6],
    4: [5, 6],
    5: [7, 8],
    6: [7, 8],
    7: [9],
    8: [9],
    9: [],
}

# Structured set of "true" haplotypes (not the full combinatorial space)
TRUE_HAPLOTYPES: List[Tuple[int, ...]] = [
    (0, 1, 3, 5, 7, 9),  # A A A A
    (0, 1, 4, 6, 8, 9),  # A B B B
    (0, 2, 3, 6, 7, 9),  # B A B A
    (0, 2, 4, 5, 8, 9),  # B B A B
    (0, 1, 3, 6, 8, 9),  # A A B B
    (0, 2, 4, 6, 7, 9),  # B B B A
]

# Fixed split for reproducibility
OBSERVED_HAPLOTYPES: List[Tuple[int, ...]] = [
    TRUE_HAPLOTYPES[0],
    TRUE_HAPLOTYPES[4],
    TRUE_HAPLOTYPES[5],
]
HELDOUT_HAPLOTYPES: List[Tuple[int, ...]] = [
    p for p in TRUE_HAPLOTYPES if p not in OBSERVED_HAPLOTYPES
]


# ============================================================
# Type aliases
# ============================================================

Path = Tuple[int, ...]
Signature = Tuple[int, int] | Tuple[int, int, int]


# ============================================================
# Helper functions
# ============================================================

def path_to_sequence(path: Path) -> str:
    """Convert a node path to its symbolic sequence representation."""
    return "".join(NODE_LABELS[node] for node in path)


def generate_reads(sequence: str, read_len: int) -> List[str]:
    """Generate all contiguous reads of length `read_len` from a sequence."""
    if len(sequence) < read_len:
        return []
    return [sequence[i : i + read_len] for i in range(len(sequence) - read_len + 1)]


def is_simple_extension(path: Path, nxt: int) -> bool:
    """Return True if appending `nxt` preserves simplicity of the path."""
    return nxt not in path


def sig_start_end(path: Path) -> Signature:
    """Retention signature based on path start and end."""
    return (path[0], path[-1])


def sig_start_second_end(path: Path) -> Signature:
    """Retention signature based on start, second, and end nodes."""
    if len(path) >= 2:
        return (path[0], path[1], path[-1])
    return (path[0], path[-1])


def bounded_prune(
    candidates: Iterable[Path],
    B: int,
    sig_func: Callable[[Path], Signature],
) -> List[Path]:
    """
    Retain at most B paths per signature class.

    Candidates are kept in insertion order within each signature class.
    This deterministic rule is sufficient for the synthetic experiment.
    """
    buckets: Dict[Signature, List[Path]] = {}

    for path in candidates:
        signature = sig_func(path)

        if signature not in buckets:
            buckets[signature] = [path]
        elif len(buckets[signature]) < B:
            buckets[signature].append(path)

    retained: List[Path] = []
    for bucket in buckets.values():
        retained.extend(bucket)

    return retained


def anchored_tipe_paths(
    succ: Dict[int, List[int]],
    anchor: int,
    Kmax: int,
    B: int,
    sig_func: Callable[[Path], Signature],
) -> Dict[int, List[Path]]:
    """
    Run TIPE from a fixed anchor node up to maximum path length Kmax.

    Returns a dictionary mapping k -> retained paths of length k.
    """
    paths_by_k: Dict[int, List[Path]] = {}

    initial_candidates = [(anchor, v) for v in succ[anchor]]
    Pk = bounded_prune(initial_candidates, B, sig_func)
    paths_by_k[1] = Pk

    for k in range(2, Kmax + 1):
        by_terminal: Dict[int, List[Path]] = defaultdict(list)
        for path in Pk:
            by_terminal[path[-1]].append(path)

        next_candidates: List[Path] = []
        for terminal, path_list in by_terminal.items():
            for path in path_list:
                for nxt in succ[terminal]:
                    if is_simple_extension(path, nxt):
                        next_candidates.append(path + (nxt,))

        Pk = bounded_prune(next_candidates, B, sig_func)
        paths_by_k[k] = Pk

        if not Pk:
            break

    return paths_by_k


def get_complete_paths(paths_by_k: Dict[int, List[Path]], end_node: int) -> List[Path]:
    """
    Collect complete paths ending at `end_node` across all k.

    Duplicate paths are removed while preserving order.
    """
    complete_paths: List[Path] = []
    for path_list in paths_by_k.values():
        for path in path_list:
            if path[-1] == end_node:
                complete_paths.append(path)

    seen = set()
    unique_paths: List[Path] = []
    for path in complete_paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    return unique_paths


def read_recall(
    candidate_sequences: List[str],
    heldout_sequences: List[str],
    read_len: int,
) -> float:
    """Compute held-out read recall based on read coverage by candidate sequences."""
    heldout_reads: List[str] = []
    for sequence in heldout_sequences:
        heldout_reads.extend(generate_reads(sequence, read_len))

    if not heldout_reads:
        return 0.0

    hits = 0
    for read in heldout_reads:
        if any(read in candidate for candidate in candidate_sequences):
            hits += 1

    return hits / len(heldout_reads)


def exact_path_recall(candidate_sequences: List[str], heldout_sequences: List[str]) -> float:
    """Compute exact held-out path recall."""
    if not heldout_sequences:
        return 0.0

    recovered = 0
    for sequence in heldout_sequences:
        if sequence in candidate_sequences:
            recovered += 1

    return recovered / len(heldout_sequences)


def relaxed_path_recall(
    candidate_sequences: List[str],
    heldout_sequences: List[str],
    min_frac: float = 0.8,
    read_len: int = 4,
) -> float:
    """
    Compute relaxed path recall.

    A held-out path is counted as recovered if at least `min_frac` of its
    read windows appear in at least one candidate path.
    """
    if not heldout_sequences:
        return 0.0

    recovered = 0
    for sequence in heldout_sequences:
        reads = generate_reads(sequence, read_len)
        if not reads:
            continue

        covered = sum(
            any(read in candidate for candidate in candidate_sequences)
            for read in reads
        )
        coverage_fraction = covered / len(reads)

        if coverage_fraction >= min_frac:
            recovered += 1

    return recovered / len(heldout_sequences)


# ============================================================
# Experiment logic
# ============================================================

def run_experiment() -> tuple[pd.DataFrame, Dict[str, Dict[int, List[str]]]]:
    """
    Run the synthetic held-out recovery experiment.

    Returns
    -------
    results : pandas.DataFrame
        Table of metrics for all methods and budgets.
    candidate_paths : dict
        Nested mapping: method -> budget -> list of complete candidate sequences.
    """
    observed_sequences = [path_to_sequence(path) for path in OBSERVED_HAPLOTYPES]
    heldout_sequences = [path_to_sequence(path) for path in HELDOUT_HAPLOTYPES]

    rows: List[Dict[str, float | int | str]] = []
    candidate_paths: Dict[str, Dict[int, List[str]]] = {
        "TIPE_start_end": {},
        "TIPE_start_second_end": {},
    }

    # Haplotype-only baseline
    rows.append(
        {
            "method": "haplotype_only",
            "B": 0,
            "num_candidate_paths": len(observed_sequences),
            "read_recall": read_recall(observed_sequences, heldout_sequences, READ_LEN),
            "path_recall_exact": exact_path_recall(observed_sequences, heldout_sequences),
            "path_recall_relaxed": relaxed_path_recall(
                observed_sequences,
                heldout_sequences,
                min_frac=0.8,
                read_len=READ_LEN,
            ),
        }
    )

    # TIPE variants
    for method_name, sig_func in [
        ("TIPE_start_end", sig_start_end),
        ("TIPE_start_second_end", sig_start_second_end),
    ]:
        for budget in BUDGETS:
            paths_by_k = anchored_tipe_paths(SUCC, ANCHOR, KMAX, budget, sig_func)
            complete_paths = get_complete_paths(paths_by_k, END_NODE)
            complete_sequences = [path_to_sequence(path) for path in complete_paths]

            candidate_paths[method_name][budget] = complete_sequences

            rows.append(
                {
                    "method": method_name,
                    "B": budget,
                    "num_candidate_paths": len(complete_sequences),
                    "read_recall": read_recall(
                        complete_sequences,
                        heldout_sequences,
                        READ_LEN,
                    ),
                    "path_recall_exact": exact_path_recall(
                        complete_sequences,
                        heldout_sequences,
                    ),
                    "path_recall_relaxed": relaxed_path_recall(
                        complete_sequences,
                        heldout_sequences,
                        min_frac=0.8,
                        read_len=READ_LEN,
                    ),
                }
            )

    results = pd.DataFrame(rows)
    return results, candidate_paths


# ============================================================
# Output helpers
# ============================================================

def save_candidate_paths(candidate_paths: Dict[str, Dict[int, List[str]]], out_path: str) -> None:
    """Save complete candidate paths for each method and budget to a text file."""
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Observed:\n")
        for path in OBSERVED_HAPLOTYPES:
            handle.write(f"  {path_to_sequence(path)}\n")

        handle.write("\nHeld-out:\n")
        for path in HELDOUT_HAPLOTYPES:
            handle.write(f"  {path_to_sequence(path)}\n")

        handle.write("\n=== Candidate paths per method/budget ===\n")

        for method_name in ["TIPE_start_end", "TIPE_start_second_end"]:
            handle.write(f"\n{method_name}\n")
            for budget in BUDGETS:
                sequences = candidate_paths[method_name][budget]
                handle.write(f"  B={budget}: {sequences}\n")


def plot_read_recall(results: pd.DataFrame, out_png: str) -> None:
    """Plot held-out read recall as a function of budget."""
    plt.figure(figsize=(6.0, 4.0))

    baseline = results.loc[
        results["method"] == "haplotype_only", "read_recall"
    ].iloc[0]
    plt.axhline(baseline, linestyle="--", label="haplotype_only")

    for method_name in ["TIPE_start_end", "TIPE_start_second_end"]:
        subset = results[results["method"] == method_name].sort_values("B")
        plt.plot(
            subset["B"],
            subset["read_recall"],
            marker="o",
            label=method_name,
        )

    plt.xlabel("Budget B")
    plt.ylabel("Read recall")
    plt.title("Synthetic experiment — held-out read recall")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_relaxed_path_recall(results: pd.DataFrame, out_png: str) -> None:
    """Plot relaxed held-out path recall as a function of budget."""
    plt.figure(figsize=(6.0, 4.0))

    baseline = results.loc[
        results["method"] == "haplotype_only", "path_recall_relaxed"
    ].iloc[0]
    plt.axhline(baseline, linestyle="--", label="haplotype_only")

    for method_name in ["TIPE_start_end", "TIPE_start_second_end"]:
        subset = results[results["method"] == method_name].sort_values("B")
        plt.plot(
            subset["B"],
            subset["path_recall_relaxed"],
            marker="o",
            label=method_name,
        )

    plt.xlabel("Budget B")
    plt.ylabel("Relaxed path recall")
    plt.title("Synthetic experiment — held-out path recovery")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run the synthetic TIPE held-out recovery experiment."""
    print("Observed:")
    for path in OBSERVED_HAPLOTYPES:
        print(f"  {path_to_sequence(path)}")

    print("\nHeld-out:")
    for path in HELDOUT_HAPLOTYPES:
        print(f"  {path_to_sequence(path)}")

    results, candidate_paths = run_experiment()

    print("\n=== Results ===")
    print(results.to_string(index=False))

    results.to_csv(OUT_CSV, index=False)
    save_candidate_paths(candidate_paths, OUT_CANDIDATES_TXT)
    plot_read_recall(results, FIG_READ_RECALL)
    plot_relaxed_path_recall(results, FIG_RELAXED_RECALL)

    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Saved figure: {FIG_READ_RECALL}")
    print(f"Saved figure: {FIG_RELAXED_RECALL}")
    print(f"Saved candidate paths: {OUT_CANDIDATES_TXT}")
    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()