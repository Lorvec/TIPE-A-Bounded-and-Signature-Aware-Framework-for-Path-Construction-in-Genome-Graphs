"""
Run a real-graph held-out recovery experiment for TIPE.

This script evaluates TIPE on real high-branching neighborhoods ("hard anchors")
sampled from a genome graph in GFA-derived form. For each anchor, a small set of
simple paths is sampled and split into:

- observed paths (used as the panel / baseline support)
- held-out paths (used for evaluation)

The script compares:
- haplotype-only traversal
- TIPE with signature (start, end)
- TIPE with signature (start, second, end)

For each setting, it reports:
- mean number of candidate paths
- mean read recall
- mean exact path recall
- mean relaxed path recall

Prerequisites
-------------
This script assumes that a real graph has already been loaded and that the
following objects are available:

- succ
- node_seq

For example:
    succ, node_labels, node_seq, seg_id, n, m = load_gfa_prefix(...)

Outputs
-------
- real_anchor_recovery_raw.csv
- real_anchor_recovery_summary.csv
- real_anchor_debug.csv
- fig_real_anchor_read_recall_vs_budget.png
- fig_real_anchor_relaxed_path_recall_vs_budget.png
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Experiment configuration
# ============================================================

RANDOM_SEED = 42

N_EXPERIMENTS = 20
TRUE_PATHS_PER_LOCUS = 3
OBSERVED_SIZE = 2
READ_LEN = 15
KMAX = 5
BUDGETS = [1, 3, 5]
MIN_RELAXED_FRAC = 0.8

ANCHOR_MODE = "high_outdeg"
N_ANCHORS_POOL = 200

OUT_DIR = "real_anchor_recovery_results"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_RAW_CSV = os.path.join(OUT_DIR, "real_anchor_recovery_raw.csv")
OUT_SUMMARY_CSV = os.path.join(OUT_DIR, "real_anchor_recovery_summary.csv")
OUT_DEBUG_CSV = os.path.join(OUT_DIR, "real_anchor_debug.csv")

FIG_READ_RECALL = os.path.join(
    OUT_DIR, "fig_real_anchor_read_recall_vs_budget.png"
)
FIG_RELAXED_RECALL = os.path.join(
    OUT_DIR, "fig_real_anchor_relaxed_path_recall_vs_budget.png"
)


# ============================================================
# Type aliases
# ============================================================

Path = Tuple[int, ...]
Signature = Tuple[int, int] | Tuple[int, int, int]


# ============================================================
# Helper functions
# ============================================================

def normalize_succ(succ: Dict[int, List[int]] | List[List[int]]) -> Dict[int, List[int]]:
    """
    Normalize adjacency structure to a dictionary form.

    Parameters
    ----------
    succ : dict or list
        Graph adjacency structure.

    Returns
    -------
    dict
        Node -> list of outgoing neighbors.
    """
    if isinstance(succ, dict):
        return {int(k): list(v) for k, v in succ.items()}
    if isinstance(succ, list):
        return {i: list(v) for i, v in enumerate(succ)}
    raise TypeError("succ must be either a dict or a list")


def pick_anchors_real(
    succ_dict: Dict[int, List[int]],
    num: int = 100,
    mode: str = "high_outdeg",
    seed: int = 13,
) -> List[int]:
    """
    Select candidate anchors from a real graph.

    Parameters
    ----------
    succ_dict : dict
        Node -> outgoing neighbors.
    num : int
        Number of anchors to return.
    mode : str
        Anchor selection mode: 'random' or 'high_outdeg'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[int]
        Selected anchor nodes.
    """
    rng = random.Random(seed)
    nodes = [u for u, outs in succ_dict.items() if len(outs) > 0]

    if not nodes:
        return []

    if mode == "random":
        rng.shuffle(nodes)
        return nodes[: min(num, len(nodes))]

    ranked = sorted(nodes, key=lambda u: len(succ_dict[u]), reverse=True)
    anchors = ranked[: min(num, len(ranked))]
    rng.shuffle(anchors)
    return anchors


def pick_hard_anchors(succ_dict: Dict[int, List[int]], top_k: int = 20) -> List[int]:
    """
    Select anchors with highest out-degree.

    Parameters
    ----------
    succ_dict : dict
        Node -> outgoing neighbors.
    top_k : int
        Maximum number of anchors to return.

    Returns
    -------
    list[int]
        High-branching anchor nodes.
    """
    candidates = [(u, len(outs)) for u, outs in succ_dict.items() if len(outs) > 0]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in candidates[:top_k]]


def path_to_sequence(path: Path, node_seq: Dict[int, str]) -> str:
    """Convert a node path to its sequence by concatenating node strings."""
    return "".join(node_seq.get(v, "") for v in path)


def generate_reads(sequence: str, read_len: int) -> List[str]:
    """Generate all contiguous reads of length `read_len` from a sequence."""
    if len(sequence) < read_len:
        return []
    return [sequence[i : i + read_len] for i in range(len(sequence) - read_len + 1)]


def read_recall(
    candidate_sequences: List[str],
    heldout_sequences: List[str],
    read_len: int,
) -> float:
    """Compute held-out read recall."""
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


def path_recall_exact(candidate_sequences: List[str], heldout_sequences: List[str]) -> float:
    """Compute exact held-out path recall."""
    if not heldout_sequences:
        return 0.0

    hits = sum(sequence in candidate_sequences for sequence in heldout_sequences)
    return hits / len(heldout_sequences)


def path_recall_relaxed(
    candidate_sequences: List[str],
    heldout_sequences: List[str],
    min_frac: float = 0.8,
    read_len: int = 15,
) -> float:
    """
    Compute relaxed held-out path recall.

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
        frac = covered / len(reads)

        if frac >= min_frac:
            recovered += 1

    return recovered / len(heldout_sequences)


# ============================================================
# Random path sampling on real graph neighborhoods
# ============================================================

def sample_simple_paths_from_anchor(
    succ_dict: Dict[int, List[int]],
    node_seq: Dict[int, str],
    anchor: int,
    n_paths: int = 3,
    max_edges: int = 5,
    min_seq_len: int = 25,
    max_trials: int = 1000,
    seed: int = 0,
) -> List[Path]:
    """
    Sample unique simple paths by random walk from a fixed anchor.

    Variable-length paths are accepted up to `max_edges`, provided that
    the resulting concatenated sequence satisfies the minimum length constraint.
    """
    rng = random.Random(seed)
    found: List[Path] = []
    found_set = set()

    trials = 0
    while len(found) < n_paths and trials < max_trials:
        trials += 1

        path = [anchor]
        visited = {anchor}

        n_steps = rng.randint(2, max_edges)
        for _ in range(n_steps):
            outs = [v for v in succ_dict.get(path[-1], []) if v not in visited]
            if not outs:
                break

            nxt = rng.choice(outs)
            path.append(nxt)
            visited.add(nxt)

        if len(path) < 3:
            continue

        sequence = "".join(node_seq.get(v, "") for v in path)
        if len(sequence) < min_seq_len:
            continue

        tuple_path = tuple(path)
        if tuple_path not in found_set:
            found.append(tuple_path)
            found_set.add(tuple_path)

    return found


# ============================================================
# TIPE core
# ============================================================

def is_simple_extension(path: Path, nxt: int) -> bool:
    """Return True if appending `nxt` preserves path simplicity."""
    return nxt not in path


def sig_start_end(path: Path) -> Signature:
    """Retention signature based on path start and end nodes."""
    return (path[0], path[-1])


def sig_start_second_end(path: Path) -> Signature:
    """Retention signature based on start, second, and end nodes."""
    if len(path) >= 2:
        return (path[0], path[1], path[-1])
    return (path[0], path[-1])


def bounded_prune(
    candidates: List[Path],
    B: int,
    sig_func: Callable[[Path], Signature],
) -> List[Path]:
    """
    Retain at most B paths per signature class.

    Candidates are kept in insertion order within each signature class.
    This deterministic rule is sufficient for the real-anchor experiment.
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
    succ_dict: Dict[int, List[int]],
    anchor: int,
    Kmax: int,
    B: int,
    sig_func: Callable[[Path], Signature],
) -> Dict[int, List[Path]]:
    """
    Run TIPE from a fixed anchor up to maximum path length Kmax.

    Returns a dictionary mapping k -> retained paths of length k.
    """
    paths_by_k: Dict[int, List[Path]] = {}

    initial_candidates = [(anchor, v) for v in succ_dict.get(anchor, [])]
    Pk = bounded_prune(initial_candidates, B, sig_func)
    paths_by_k[1] = Pk

    for k in range(2, Kmax + 1):
        by_terminal: Dict[int, List[Path]] = defaultdict(list)
        for path in Pk:
            by_terminal[path[-1]].append(path)

        next_candidates: List[Path] = []
        for terminal, path_list in by_terminal.items():
            for path in path_list:
                for nxt in succ_dict.get(terminal, []):
                    if is_simple_extension(path, nxt):
                        next_candidates.append(path + (nxt,))

        Pk = bounded_prune(next_candidates, B, sig_func)
        paths_by_k[k] = Pk

        if not Pk:
            break

    return paths_by_k


def get_paths_upto_k(paths_by_k: Dict[int, List[Path]], max_k: int) -> List[Path]:
    """
    Collect retained paths across all lengths up to max_k.

    Duplicate paths are removed while preserving order.
    """
    all_paths: List[Path] = []
    for k in range(1, max_k + 1):
        all_paths.extend(paths_by_k.get(k, []))

    seen = set()
    unique_paths: List[Path] = []
    for path in all_paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    return unique_paths


# ============================================================
# Experiment logic
# ============================================================

def run_real_anchor_experiment(
    succ_dict: Dict[int, List[int]],
    node_seq: Dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the real-graph hard-anchor experiment.

    Returns
    -------
    results : pandas.DataFrame
        Per-locus metrics.
    debug_df : pandas.DataFrame
        Anchor-level diagnostic information.
    """
    if ANCHOR_MODE == "high_outdeg":
        anchors_pool = pick_hard_anchors(succ_dict, top_k=N_ANCHORS_POOL)
    else:
        anchors_pool = pick_anchors_real(
            succ_dict,
            num=N_ANCHORS_POOL,
            mode=ANCHOR_MODE,
            seed=RANDOM_SEED,
        )

    print(f"Anchors selected: {len(anchors_pool)}")
    print(f"Sample anchors: {anchors_pool[:10]}")

    rows: List[Dict[str, float | int | str]] = []
    debug_rows: List[Dict[str, int]] = []
    used_loci = 0

    for locus_idx, anchor in enumerate(anchors_pool):
        true_paths = sample_simple_paths_from_anchor(
            succ_dict,
            node_seq=node_seq,
            anchor=anchor,
            n_paths=TRUE_PATHS_PER_LOCUS,
            max_edges=KMAX,
            min_seq_len=READ_LEN + 10,
            max_trials=1000,
            seed=RANDOM_SEED + locus_idx,
        )

        debug_rows.append(
            {
                "anchor": anchor,
                "n_true_paths_found": len(true_paths),
            }
        )

        if len(true_paths) < TRUE_PATHS_PER_LOCUS:
            continue

        observed_paths = true_paths[:OBSERVED_SIZE]
        heldout_paths = true_paths[OBSERVED_SIZE:]

        if not heldout_paths:
            continue

        observed_sequences = [path_to_sequence(path, node_seq) for path in observed_paths]
        heldout_sequences = [path_to_sequence(path, node_seq) for path in heldout_paths]

        if not all(len(sequence) >= READ_LEN for sequence in heldout_sequences):
            continue

        used_loci += 1

        # Haplotype-only baseline
        rows.append(
            {
                "locus_id": used_loci,
                "anchor": anchor,
                "method": "haplotype_only",
                "B": 0,
                "num_candidate_paths": len(observed_sequences),
                "read_recall": read_recall(
                    observed_sequences,
                    heldout_sequences,
                    READ_LEN,
                ),
                "path_recall_exact": path_recall_exact(
                    observed_sequences,
                    heldout_sequences,
                ),
                "path_recall_relaxed": path_recall_relaxed(
                    observed_sequences,
                    heldout_sequences,
                    min_frac=MIN_RELAXED_FRAC,
                    read_len=READ_LEN,
                ),
            }
        )

        # TIPE settings
        for method_name, sig_func in [
            ("TIPE_start_end", sig_start_end),
            ("TIPE_start_second_end", sig_start_second_end),
        ]:
            for budget in BUDGETS:
                paths_by_k = anchored_tipe_paths(
                    succ_dict,
                    anchor=anchor,
                    Kmax=KMAX,
                    B=budget,
                    sig_func=sig_func,
                )

                candidate_paths = get_paths_upto_k(paths_by_k, KMAX)
                candidate_sequences = [
                    path_to_sequence(path, node_seq) for path in candidate_paths
                ]

                rows.append(
                    {
                        "locus_id": used_loci,
                        "anchor": anchor,
                        "method": method_name,
                        "B": budget,
                        "num_candidate_paths": len(candidate_sequences),
                        "read_recall": read_recall(
                            candidate_sequences,
                            heldout_sequences,
                            READ_LEN,
                        ),
                        "path_recall_exact": path_recall_exact(
                            candidate_sequences,
                            heldout_sequences,
                        ),
                        "path_recall_relaxed": path_recall_relaxed(
                            candidate_sequences,
                            heldout_sequences,
                            min_frac=MIN_RELAXED_FRAC,
                            read_len=READ_LEN,
                        ),
                    }
                )

        if used_loci >= N_EXPERIMENTS:
            break

    results = pd.DataFrame(rows)
    debug_df = pd.DataFrame(debug_rows)

    print(f"\nUsed loci: {used_loci}")
    print(f"Rows collected: {len(rows)}")

    return results, debug_df


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-locus results into mean metrics by method and budget."""
    return (
        results.groupby(["method", "B"], as_index=False)
        .agg(
            num_candidate_paths=("num_candidate_paths", "mean"),
            read_recall=("read_recall", "mean"),
            path_recall_exact=("path_recall_exact", "mean"),
            path_recall_relaxed=("path_recall_relaxed", "mean"),
        )
    )


# ============================================================
# Plot helpers
# ============================================================

def plot_read_recall(summary: pd.DataFrame, out_png: str) -> None:
    """Plot mean held-out read recall as a function of budget."""
    plt.figure(figsize=(6.5, 4.5))

    baseline = summary.loc[
        summary["method"] == "haplotype_only", "read_recall"
    ].iloc[0]
    plt.axhline(baseline, linestyle="--", label="haplotype_only")

    for method_name in ["TIPE_start_end", "TIPE_start_second_end"]:
        subset = summary[summary["method"] == method_name].sort_values("B")
        plt.plot(
            subset["B"],
            subset["read_recall"],
            marker="o",
            label=method_name,
        )

    plt.xlabel("Budget B")
    plt.ylabel("Mean read recall")
    plt.title("Real graph experiment — held-out read recall")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_relaxed_path_recall(summary: pd.DataFrame, out_png: str) -> None:
    """Plot mean relaxed held-out path recall as a function of budget."""
    plt.figure(figsize=(6.5, 4.5))

    baseline = summary.loc[
        summary["method"] == "haplotype_only", "path_recall_relaxed"
    ].iloc[0]
    plt.axhline(baseline, linestyle="--", label="haplotype_only")

    for method_name in ["TIPE_start_end", "TIPE_start_second_end"]:
        subset = summary[summary["method"] == method_name].sort_values("B")
        plt.plot(
            subset["B"],
            subset["path_recall_relaxed"],
            marker="o",
            label=method_name,
        )

    plt.xlabel("Budget B")
    plt.ylabel("Mean relaxed path recall")
    plt.title("Real graph experiment — held-out path recovery")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run the real hard-anchor held-out recovery experiment."""
    required_names = ["succ", "node_seq"]
    missing = [name for name in required_names if name not in globals()]
    if missing:
        raise RuntimeError(
            "Missing required objects: "
            + ", ".join(missing)
            + ".\nPlease load the real graph first (succ, node_seq)."
        )

    random.seed(RANDOM_SEED)

    succ_dict = normalize_succ(succ)

    results, debug_df = run_real_anchor_experiment(succ_dict, node_seq)

    print("\n=== DEBUG (how many true paths were found per anchor) ===")
    print(debug_df.head(20).to_string(index=False))

    if results.empty:
        print("\nNo valid loci collected.")
        print("You may try adjusting:")
        print(f"  TRUE_PATHS_PER_LOCUS = {TRUE_PATHS_PER_LOCUS}")
        print(f"  READ_LEN = {READ_LEN}")
        print(f"  KMAX = {KMAX}")
        return

    print("\n=== RAW RESULTS (head) ===")
    print(results.head(20).to_string(index=False))

    summary = summarize_results(results)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    debug_df.to_csv(OUT_DEBUG_CSV, index=False)
    results.to_csv(OUT_RAW_CSV, index=False)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    plot_read_recall(summary, FIG_READ_RECALL)
    plot_relaxed_path_recall(summary, FIG_RELAXED_RECALL)

    print(f"\nSaved debug CSV: {OUT_DEBUG_CSV}")
    print(f"Saved raw results CSV: {OUT_RAW_CSV}")
    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved figure: {FIG_READ_RECALL}")
    print(f"Saved figure: {FIG_RELAXED_RECALL}")
    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()