"""
Run TIPE validation on a public E. coli pangenome GFA.

This script:
- parses a prefix-induced subgraph from a GFA file,
- runs TIPE with bounded signature-aware retention,
- records cumulative retention and pruning metrics,
- exports results as CSV and publication-ready figures.

Outputs:
  - real_ecoli_validation.csv
  - fig_real_paths_vs_k.png
  - fig_real_retention_pruning_vs_k.png
  - fig_real_cycle_reject_vs_k.png
  - fig_real_runtime_vs_k.png
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


# ============================================================
# Experiment configuration
# ============================================================

GFA_PATH = "EcoliGraph_MGC.gfa"
MAX_SEGMENTS = 50_000
KMAX = 10
BUDGETS = [1, 5, 10]

OUT_DIR = "real_ecoli_results"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "real_ecoli_validation.csv")
FIG_PATHS = os.path.join(OUT_DIR, "fig_real_paths_vs_k.png")
FIG_RETENTION = os.path.join(OUT_DIR, "fig_real_retention_pruning_vs_k.png")
FIG_CYCLE = os.path.join(OUT_DIR, "fig_real_cycle_reject_vs_k.png")
FIG_RUNTIME = os.path.join(OUT_DIR, "fig_real_runtime_vs_k.png")


# ============================================================
# Type aliases
# ============================================================

Path = Tuple[int, ...]
Signature = Tuple[int, int] | Tuple[int, int, int]


# ============================================================
# GFA parsing
# ============================================================

def parse_gfa_to_digraph(
    gfa_path: str,
    max_segments: int,
    verbose: bool = True,
) -> Tuple[List[List[int]], int, int]:
    """
    Parse a GFA file into a directed adjacency list.

    Only the first `max_segments` segment records (S-lines) are retained.
    Directed edges are added from link records (L-lines) when both endpoints
    belong to the retained segment set.

    For this structural validation script, GFA orientations are ignored and
    each link is interpreted as a directed edge from the source segment to
    the target segment.

    Parameters
    ----------
    gfa_path : str
        Path to the GFA file.
    max_segments : int
        Maximum number of segment records (S-lines) to retain.
    verbose : bool, optional
        Whether to print parsing summary information.

    Returns
    -------
    succ : list[list[int]]
        Directed adjacency list.
    n : int
        Number of retained segments.
    m : int
        Number of retained directed edges.
    """
    seg_id: Dict[str, int] = {}
    edges: List[Tuple[int, int]] = []
    kept = 0

    t0 = time.perf_counter()

    with open(gfa_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue

            record_type = parts[0]

            if record_type == "S":
                if len(parts) < 2:
                    continue
                segment_name = parts[1]

                if segment_name in seg_id:
                    continue
                if kept >= max_segments:
                    continue

                seg_id[segment_name] = kept
                kept += 1

            elif record_type == "L":
                if len(parts) < 5:
                    continue

                source_name = parts[1]
                target_name = parts[3]

                if source_name in seg_id and target_name in seg_id:
                    edges.append((seg_id[source_name], seg_id[target_name]))

    n = kept
    succ: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        succ[u].append(v)

    m = len(edges)
    t1 = time.perf_counter()

    if verbose:
        print(f"[GFA] Parsed prefix-induced subgraph: n={n:,} segments, m={m:,} directed edges")
        print(f"[GFA] Parsing time: {t1 - t0:.2f}s (MAX_SEGMENTS={max_segments:,})")

    return succ, n, m


# ============================================================
# TIPE core
# ============================================================

def is_simple_extension(path: Path, nxt: int) -> bool:
    """
    Return True if appending `nxt` preserves simplicity of the path.

    For the path lengths used in this experiment, a linear membership
    check is sufficient.
    """
    return nxt not in path


def lex_key(path: Path) -> Tuple[int, ...]:
    """Deterministic lexicographic key used for tie-breaking."""
    return path


def sig_start_end(path: Path, k_edges: int) -> Signature:
    """
    Signature based on path start and end nodes.

    The `k_edges` argument is included for interface consistency.
    """
    _ = k_edges
    return (path[0], path[-1])


def bounded_prune(
    candidates: Iterable[Path],
    *,
    B: int,
    sig_fn: Callable[[Path, int], Signature],
    k_edges: int,
) -> List[Path]:
    """
    Retain at most B candidate paths per signature class.

    Within each signature class, paths are kept deterministically using
    lexicographic ordering over node identifiers.
    """
    buckets: Dict[Signature, List[Path]] = {}

    for path in candidates:
        signature = sig_fn(path, k_edges)
        bucket = buckets.get(signature)

        if bucket is None:
            buckets[signature] = [path]
            continue

        if len(bucket) < B:
            bucket.append(path)
            continue

        worst_idx = 0
        worst_key = lex_key(bucket[0])

        for i in range(1, len(bucket)):
            candidate_key = lex_key(bucket[i])
            if candidate_key > worst_key:
                worst_key = candidate_key
                worst_idx = i

        if lex_key(path) < worst_key:
            bucket[worst_idx] = path

    retained: List[Path] = []
    for bucket in buckets.values():
        retained.extend(bucket)

    return retained


# ============================================================
# Metrics container
# ============================================================

@dataclass
class RunRow:
    setting: str
    K: int

    exact_retained_k: int
    cumulative_retained_upto_k: int

    attempted_total: int
    candidates_total: int
    retained_total: int

    cycle_reject_rate: float
    retention_prune_rate: float

    runtime_sec: float


# ============================================================
# TIPE runner with cumulative metrics
# ============================================================

def run_tipe_with_metrics(
    succ: List[List[int]],
    *,
    Kmax: int,
    B: int,
    sig_fn: Callable[[Path, int], Signature],
    setting_name: str,
) -> List[RunRow]:
    """
    Run TIPE up to Kmax edges and record cumulative metrics at each K.

    Metrics include:
    - exact number of retained paths at length K,
    - cumulative retained paths up to K,
    - attempted extensions,
    - simple-valid candidates,
    - retained paths after bounded pruning,
    - cumulative cycle/invalid rejection rate,
    - cumulative retention pruning rate,
    - runtime.
    """
    t0 = time.perf_counter()
    rows: List[RunRow] = []

    attempted_total = 0
    candidates_total = 0
    retained_total = 0
    cumulative_retained = 0

    # --------------------------------------------------------
    # K = 1
    # --------------------------------------------------------
    candidates_k: List[Path] = []
    attempted_k = 0
    candidates_k_count = 0

    for u, outs in enumerate(succ):
        for v in outs:
            attempted_k += 1
            candidates_k.append((u, v))
            candidates_k_count += 1

    attempted_total += attempted_k
    candidates_total += candidates_k_count

    retained_k = bounded_prune(candidates_k, B=B, sig_fn=sig_fn, k_edges=1)
    exact_retained_k = len(retained_k)

    retained_total += exact_retained_k
    cumulative_retained += exact_retained_k

    rows.append(
        RunRow(
            setting=setting_name,
            K=1,
            exact_retained_k=exact_retained_k,
            cumulative_retained_upto_k=cumulative_retained,
            attempted_total=attempted_total,
            candidates_total=candidates_total,
            retained_total=retained_total,
            cycle_reject_rate=0.0 if attempted_total == 0 else 1.0 - candidates_total / attempted_total,
            retention_prune_rate=0.0 if candidates_total == 0 else 1.0 - retained_total / candidates_total,
            runtime_sec=time.perf_counter() - t0,
        )
    )

    Pk = retained_k

    # --------------------------------------------------------
    # K >= 2
    # --------------------------------------------------------
    for k in range(1, Kmax):
        by_terminal: Dict[int, List[Path]] = defaultdict(list)
        for path in Pk:
            by_terminal[path[-1]].append(path)

        next_candidates: List[Path] = []
        attempted_k = 0
        candidates_k_count = 0

        for v, paths_ending_at_v in by_terminal.items():
            outs = succ[v]
            if not outs:
                continue

            for path in paths_ending_at_v:
                for nxt in outs:
                    attempted_k += 1
                    if not is_simple_extension(path, nxt):
                        continue
                    next_candidates.append(path + (nxt,))
                    candidates_k_count += 1

        attempted_total += attempted_k
        candidates_total += candidates_k_count

        Pk = bounded_prune(next_candidates, B=B, sig_fn=sig_fn, k_edges=k + 1)
        exact_retained_k = len(Pk)

        retained_total += exact_retained_k
        cumulative_retained += exact_retained_k

        rows.append(
            RunRow(
                setting=setting_name,
                K=k + 1,
                exact_retained_k=exact_retained_k,
                cumulative_retained_upto_k=cumulative_retained,
                attempted_total=attempted_total,
                candidates_total=candidates_total,
                retained_total=retained_total,
                cycle_reject_rate=0.0 if attempted_total == 0 else 1.0 - candidates_total / attempted_total,
                retention_prune_rate=0.0 if candidates_total == 0 else 1.0 - retained_total / candidates_total,
                runtime_sec=time.perf_counter() - t0,
            )
        )

        if not Pk:
            for kk in range(k + 2, Kmax + 1):
                rows.append(
                    RunRow(
                        setting=setting_name,
                        K=kk,
                        exact_retained_k=0,
                        cumulative_retained_upto_k=cumulative_retained,
                        attempted_total=attempted_total,
                        candidates_total=candidates_total,
                        retained_total=retained_total,
                        cycle_reject_rate=0.0 if attempted_total == 0 else 1.0 - candidates_total / attempted_total,
                        retention_prune_rate=0.0 if candidates_total == 0 else 1.0 - retained_total / candidates_total,
                        runtime_sec=time.perf_counter() - t0,
                    )
                )
            break

    return rows


# ============================================================
# Output helpers
# ============================================================

def save_csv(rows: List[RunRow], out_csv: str) -> None:
    """Save run metrics as CSV."""
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "setting",
                "K",
                "exact_retained_k",
                "cumulative_retained_upto_k",
                "attempted_total",
                "candidates_total",
                "retained_total",
                "cycle_reject_rate",
                "retention_prune_rate",
                "runtime_sec",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row.setting,
                    row.K,
                    row.exact_retained_k,
                    row.cumulative_retained_upto_k,
                    row.attempted_total,
                    row.candidates_total,
                    row.retained_total,
                    f"{row.cycle_reject_rate:.6f}",
                    f"{row.retention_prune_rate:.6f}",
                    f"{row.runtime_sec:.6f}",
                ]
            )

    print(f"Saved CSV: {out_csv}")


def group_by_setting(rows: List[RunRow]) -> Dict[str, List[RunRow]]:
    """Group rows by experimental setting and sort each group by K."""
    grouped: Dict[str, List[RunRow]] = defaultdict(list)
    for row in rows:
        grouped[row.setting].append(row)

    for setting in grouped:
        grouped[setting] = sorted(grouped[setting], key=lambda r: r.K)

    return grouped


def plot_paths(rows: List[RunRow], out_png: str, title: str) -> None:
    """Plot cumulative retained paths up to K."""
    grouped = group_by_setting(rows)

    plt.figure(figsize=(7.0, 4.4))
    for setting, setting_rows in sorted(grouped.items()):
        plt.plot(
            [row.K for row in setting_rows],
            [row.cumulative_retained_upto_k for row in setting_rows],
            marker="o",
            label=setting,
        )

    plt.xlabel("K (maximum path length in edges)")
    plt.ylabel("Retained paths (cumulative up to K)")
    plt.title(f"{title} — retained paths")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved figure: {out_png}")


def plot_rate(rows: List[RunRow], out_png: str, title: str, which: str) -> None:
    """Plot cumulative rejection/pruning rates as a function of K."""
    grouped = group_by_setting(rows)

    if which == "retention_prune_rate":
        ylabel = "Retention pruning rate"
        figure_title = f"{title} — retention pruning (1 - retained/candidates), cumulative"
    elif which == "cycle_reject_rate":
        ylabel = "Cycle/invalid rejection rate"
        figure_title = f"{title} — cycle/invalid rejection (1 - candidates/attempted), cumulative"
    else:
        ylabel = which
        figure_title = f"{title} — {which}"

    plt.figure(figsize=(7.0, 4.4))
    for setting, setting_rows in sorted(grouped.items()):
        plt.plot(
            [row.K for row in setting_rows],
            [getattr(row, which) for row in setting_rows],
            marker="o",
            label=setting,
        )

    plt.xlabel("K (maximum path length in edges)")
    plt.ylabel(ylabel)
    plt.title(figure_title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved figure: {out_png}")


def plot_runtime(rows: List[RunRow], out_png: str, title: str) -> None:
    """Plot runtime as a function of K."""
    grouped = group_by_setting(rows)

    plt.figure(figsize=(7.0, 4.4))
    for setting, setting_rows in sorted(grouped.items()):
        plt.plot(
            [row.K for row in setting_rows],
            [row.runtime_sec for row in setting_rows],
            marker="o",
            label=setting,
        )

    plt.xlabel("K (maximum path length in edges)")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"{title} — runtime")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved figure: {out_png}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run the TIPE validation experiment on a real pangenome graph."""

    # --------------------------------------------------------
    # Check input data
    # --------------------------------------------------------
    if not os.path.exists(GFA_PATH):
        raise FileNotFoundError(
            f"GFA file not found: {GFA_PATH}\n"
            "Please download the dataset as described in the README and place it in the repository root."
        )

    # --------------------------------------------------------
    # Parse graph
    # --------------------------------------------------------
    succ, n, m = parse_gfa_to_digraph(GFA_PATH, MAX_SEGMENTS, verbose=True)

    # --------------------------------------------------------
    # Run TIPE experiments
    # --------------------------------------------------------
    all_rows: List[RunRow] = []

    for budget in BUDGETS:
        setting_name = f"TIPE (start,end), B={budget}"
        print(f"\n[RUN] {setting_name}")

        rows = run_tipe_with_metrics(
            succ,
            Kmax=KMAX,
            B=budget,
            sig_fn=sig_start_end,
            setting_name=setting_name,
        )
        all_rows.extend(rows)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    save_csv(all_rows, OUT_CSV)

    figure_title = (
        f"E. coli pangenome graph (prefix-induced subgraph: n={n:,}, m={m:,})"
    )

    plot_paths(all_rows, FIG_PATHS, figure_title)
    plot_rate(all_rows, FIG_RETENTION, figure_title, which="retention_prune_rate")
    plot_rate(all_rows, FIG_CYCLE, figure_title, which="cycle_reject_rate")
    plot_runtime(all_rows, FIG_RUNTIME, figure_title)

    print(f"\nAll outputs saved to: {OUT_DIR}")