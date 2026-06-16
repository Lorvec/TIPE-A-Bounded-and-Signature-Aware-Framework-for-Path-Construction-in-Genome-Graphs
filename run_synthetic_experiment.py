"""
Large dense layered branching graph experiment for TIPE.

Purpose
-------
This script addresses the concern that the previous dense branching synthetic
graph was too small to clearly demonstrate combinatorial path growth.

It constructs a larger directed acyclic layered graph and compares:

1. Exhaustive simple-path count
2. TIPE with bounded signature-aware retention

Graph construction
------------------
- One source node.
- L layers.
- W nodes per layer.
- The source connects to every node in layer 1.
- Every node in layer i connects to every node in layer i+1.
- The graph is acyclic, so all source-to-layer paths are simple.

For this graph, the exact number of exhaustive source-anchored paths of
length K is:

    W^K

and the cumulative number of paths up to K is:

    sum_{i=1}^{K} W^i

Default configuration
---------------------
WIDTH = 6
N_LAYERS = 12
KMAX = 12
BUDGETS = [1, 5, 10]

With these defaults:
- number of nodes = 73
- number of edges = 402
- exhaustive exact paths at K=12 = 2,176,782,336
- exhaustive cumulative paths up to K=12 = 2,612,138,802

Outputs
-------
Saved under OUT_DIR:

- synthetic_dense_large_results.csv
- synthetic_dense_large_summary.csv
- fig_dense_large_paths_vs_k.png
- fig_dense_large_pruning_vs_k.png
- fig_dense_large_runtime_vs_k.png
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# 0) Configuration
# ============================================================

OUT_DIR = Path("synthetic_dense_large")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "synthetic_dense_large_results.csv"
OUT_SUMMARY_CSV = OUT_DIR / "synthetic_dense_large_summary.csv"

FIG_PATHS = OUT_DIR / "fig_dense_large_paths_vs_k.png"
FIG_PRUNING = OUT_DIR / "fig_dense_large_pruning_vs_k.png"
FIG_RUNTIME = OUT_DIR / "fig_dense_large_runtime_vs_k.png"

WIDTH = 6
N_LAYERS = 12
KMAX = 12
BUDGETS = [1, 5, 10]


# ============================================================
# 1) Type aliases
# ============================================================

PathT = Tuple[int, ...]
Signature = Tuple[int, int]


# ============================================================
# 2) Graph construction
# ============================================================

def build_dense_layered_dag(
    width: int,
    n_layers: int,
) -> Tuple[List[List[int]], List[List[int]], int]:
    """
    Build a dense layered directed acyclic graph.

    Node 0 is the source.
    Layers 1..n_layers each contain `width` nodes.
    The source connects to all nodes in layer 1.
    Each node in layer i connects to all nodes in layer i+1.

    Returns
    -------
    succ : list[list[int]]
        Directed adjacency list.
    layers : list[list[int]]
        Node IDs grouped by layer.
    source : int
        Source node ID.
    """
    source = 0
    layers: List[List[int]] = []

    next_node = 1

    for _ in range(n_layers):
        layer = list(range(next_node, next_node + width))
        layers.append(layer)
        next_node += width

    n_nodes = next_node
    succ: List[List[int]] = [[] for _ in range(n_nodes)]

    # Source to first layer
    succ[source].extend(layers[0])

    # Complete bipartite connections between consecutive layers
    for i in range(n_layers - 1):
        for u in layers[i]:
            succ[u].extend(layers[i + 1])

    return succ, layers, source


def count_edges(succ: List[List[int]]) -> int:
    """Count directed edges in adjacency list."""
    return sum(len(neighbors) for neighbors in succ)


# ============================================================
# 3) Exhaustive path count formulas
# ============================================================

def exhaustive_exact_paths(width: int, k: int) -> int:
    """
    Exact number of source-anchored simple paths of length k.

    In this layered graph, each step chooses one of `width` nodes
    in the next layer, therefore exact paths of length k = width^k.
    """
    return width ** k


def exhaustive_cumulative_paths(width: int, k: int) -> int:
    """
    Cumulative number of exhaustive source-anchored paths up to length k.
    """
    return sum(width ** i for i in range(1, k + 1))


# ============================================================
# 4) TIPE retention
# ============================================================

def sig_start_end(path: PathT) -> Signature:
    """
    Retention signature based on path start and end nodes.
    """
    return (path[0], path[-1])


def lex_key(path: PathT) -> Tuple[int, ...]:
    """
    Deterministic lexicographic ranking key.
    """
    return path


def bounded_prune(
    candidates: Iterable[PathT],
    B: int,
    sig_func: Callable[[PathT], Signature],
) -> List[PathT]:
    """
    TIPE bounded retention.

    Retain at most B paths per signature class, using lexicographic
    ranking within each signature class.
    """
    buckets: Dict[Signature, List[PathT]] = defaultdict(list)

    for path in candidates:
        buckets[sig_func(path)].append(path)

    retained: List[PathT] = []

    for signature in sorted(buckets.keys()):
        bucket = sorted(buckets[signature], key=lex_key)
        retained.extend(bucket[:B])

    return retained


def expand_paths_one_step(
    succ: List[List[int]],
    paths: List[PathT],
) -> List[PathT]:
    """
    Expand retained paths by one edge.

    The graph is a layered DAG, so every extension is simple.
    """
    candidates: List[PathT] = []

    by_terminal: Dict[int, List[PathT]] = defaultdict(list)

    for path in paths:
        by_terminal[path[-1]].append(path)

    for terminal, path_list in by_terminal.items():
        for path in path_list:
            for nxt in succ[terminal]:
                candidates.append(path + (nxt,))

    return candidates


def run_tipe_dense(
    succ: List[List[int]],
    source: int,
    Kmax: int,
    B: int,
) -> List[Dict[str, Any]]:
    """
    Run TIPE from the source node up to Kmax.

    Returns one row per K with cumulative metrics.
    """
    t0 = time.perf_counter()

    rows: List[Dict[str, Any]] = []

    attempted_total = 0
    candidates_total = 0
    retained_total = 0
    cumulative_retained = 0

    # -------------------------
    # K = 1
    # -------------------------
    candidates = [(source, v) for v in succ[source]]

    attempted_total += len(candidates)
    candidates_total += len(candidates)

    Pk = bounded_prune(
        candidates=candidates,
        B=B,
        sig_func=sig_start_end,
    )

    retained_k = len(Pk)
    retained_total += retained_k
    cumulative_retained += retained_k

    rows.append({
        "method": f"TIPE B={B}",
        "B": B,
        "K": 1,
        "exact_paths_k": retained_k,
        "cumulative_paths_upto_k": cumulative_retained,
        "attempted_total": attempted_total,
        "candidates_total": candidates_total,
        "retained_total": retained_total,
        "retention_prune_rate": (
            0.0 if candidates_total == 0
            else 1.0 - retained_total / candidates_total
        ),
        "runtime_sec": time.perf_counter() - t0,
    })

    # -------------------------
    # K >= 2
    # -------------------------
    for k in range(2, Kmax + 1):
        candidates = expand_paths_one_step(succ, Pk)

        attempted_total += len(candidates)
        candidates_total += len(candidates)

        Pk = bounded_prune(
            candidates=candidates,
            B=B,
            sig_func=sig_start_end,
        )

        retained_k = len(Pk)
        retained_total += retained_k
        cumulative_retained += retained_k

        rows.append({
            "method": f"TIPE B={B}",
            "B": B,
            "K": k,
            "exact_paths_k": retained_k,
            "cumulative_paths_upto_k": cumulative_retained,
            "attempted_total": attempted_total,
            "candidates_total": candidates_total,
            "retained_total": retained_total,
            "retention_prune_rate": (
                0.0 if candidates_total == 0
                else 1.0 - retained_total / candidates_total
            ),
            "runtime_sec": time.perf_counter() - t0,
        })

    return rows


# ============================================================
# 5) CSV helpers
# ============================================================

def save_rows_csv(
    rows: List[Dict[str, Any]],
    out_csv: Path,
) -> None:
    """Save all experiment rows to CSV."""
    fieldnames = [
        "method",
        "B",
        "K",
        "exact_paths_k",
        "cumulative_paths_upto_k",
        "attempted_total",
        "candidates_total",
        "retained_total",
        "retention_prune_rate",
        "runtime_sec",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {out_csv}")


def save_summary_csv(
    summary_rows: List[Dict[str, Any]],
    out_csv: Path,
) -> None:
    """Save final K summary rows to CSV."""
    fieldnames = [
        "method",
        "B",
        "KMAX",
        "exact_paths_at_KMAX",
        "cumulative_paths_upto_KMAX",
        "retention_prune_rate_at_KMAX",
        "runtime_sec_at_KMAX",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary CSV: {out_csv}")


# ============================================================
# 6) Plot helpers
# ============================================================

def group_rows_by_method(
    rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group rows by method and sort by K."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["method"]].append(row)

    for method in grouped:
        grouped[method] = sorted(grouped[method], key=lambda r: int(r["K"]))

    return grouped


def plot_path_growth(
    grouped: Dict[str, List[Dict[str, Any]]],
    out_png: Path,
) -> None:
    """
    Plot cumulative exhaustive/TIPE paths up to K.

    Uses log-scale y-axis with clean scientific labels.
    """
    plt.figure(figsize=(7.0, 4.6))

    method_order = [
        "Exhaustive simple paths",
        "TIPE B=1",
        "TIPE B=5",
        "TIPE B=10",
    ]

    for method in method_order:
        if method not in grouped:
            continue

        rows = grouped[method]
        xs = [int(row["K"]) for row in rows]
        ys = [float(row["cumulative_paths_upto_k"]) for row in rows]

        if method == "Exhaustive simple paths":
            plt.plot(xs, ys, marker="o", linewidth=2.5, label=method)
        else:
            plt.plot(xs, ys, marker="o", linewidth=2.0, label=method)

    plt.xlabel("K (maximum path length in edges)")
    plt.ylabel("Cumulative paths up to K")
    plt.yscale("log")

    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.get_offset_text().set_visible(False)

    plt.title("Dense layered branching graph — path growth")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved figure: {out_png}")


def plot_retention_pruning(
    grouped: Dict[str, List[Dict[str, Any]]],
    out_png: Path,
) -> None:
    """Plot cumulative retention-induced pruning rate."""
    plt.figure(figsize=(7.0, 4.6))

    for method in ["TIPE B=1", "TIPE B=5", "TIPE B=10"]:
        if method not in grouped:
            continue

        rows = grouped[method]
        xs = [int(row["K"]) for row in rows]
        ys = [float(row["retention_prune_rate"]) for row in rows]

        plt.plot(xs, ys, marker="o", linewidth=2.0, label=method)

    plt.xlabel("K (maximum path length in edges)")
    plt.ylabel("Retention-induced pruning rate")
    plt.title("Dense layered branching graph — retention pruning")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved figure: {out_png}")


def plot_tipe_runtime(
    grouped: Dict[str, List[Dict[str, Any]]],
    out_png: Path,
) -> None:
    """Plot TIPE runtime as a function of K."""
    plt.figure(figsize=(7.0, 4.6))

    for method in ["TIPE B=1", "TIPE B=5", "TIPE B=10"]:
        if method not in grouped:
            continue

        rows = grouped[method]
        xs = [int(row["K"]) for row in rows]
        ys = [float(row["runtime_sec"]) for row in rows]

        plt.plot(xs, ys, marker="o", linewidth=2.0, label=method)

    plt.xlabel("K (maximum path length in edges)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Dense layered branching graph — TIPE runtime")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved figure: {out_png}")


# ============================================================
# 7) Main experiment
# ============================================================

def main() -> None:
    """Run the full synthetic dense branching experiment."""
    succ, layers, source = build_dense_layered_dag(
        width=WIDTH,
        n_layers=N_LAYERS,
    )

    n_nodes = len(succ)
    n_edges = count_edges(succ)

    exhaustive_exact_at_kmax = exhaustive_exact_paths(WIDTH, KMAX)
    exhaustive_cumulative_at_kmax = exhaustive_cumulative_paths(WIDTH, KMAX)

    print("=== Dense layered synthetic graph ===")
    print(f"Width per layer: {WIDTH}")
    print(f"Number of layers: {N_LAYERS}")
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"KMAX: {KMAX}")
    print(f"Exhaustive exact paths at K={KMAX}: {exhaustive_exact_at_kmax:,}")
    print(f"Exhaustive cumulative paths up to K={KMAX}: {exhaustive_cumulative_at_kmax:,}")

    all_rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Exhaustive count rows
    # --------------------------------------------------------
    for k in range(1, KMAX + 1):
        all_rows.append({
            "method": "Exhaustive simple paths",
            "B": "",
            "K": k,
            "exact_paths_k": exhaustive_exact_paths(WIDTH, k),
            "cumulative_paths_upto_k": exhaustive_cumulative_paths(WIDTH, k),
            "attempted_total": "",
            "candidates_total": "",
            "retained_total": "",
            "retention_prune_rate": "",
            "runtime_sec": "",
        })

    # --------------------------------------------------------
    # TIPE rows
    # --------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []

    summary_rows.append({
        "method": "Exhaustive simple paths",
        "B": "",
        "KMAX": KMAX,
        "exact_paths_at_KMAX": exhaustive_exact_at_kmax,
        "cumulative_paths_upto_KMAX": exhaustive_cumulative_at_kmax,
        "retention_prune_rate_at_KMAX": "",
        "runtime_sec_at_KMAX": "",
    })

    for B in BUDGETS:
        print(f"\n[RUN] TIPE B={B}")

        rows = run_tipe_dense(
            succ=succ,
            source=source,
            Kmax=KMAX,
            B=B,
        )

        all_rows.extend(rows)

        final_row = rows[-1]

        print(
            f"  cumulative retained up to K={KMAX}: "
            f"{int(final_row['cumulative_paths_upto_k']):,}, "
            f"retention prune rate: {final_row['retention_prune_rate']:.6f}, "
            f"runtime: {final_row['runtime_sec']:.6f}s"
        )

        summary_rows.append({
            "method": f"TIPE B={B}",
            "B": B,
            "KMAX": KMAX,
            "exact_paths_at_KMAX": int(final_row["exact_paths_k"]),
            "cumulative_paths_upto_KMAX": int(final_row["cumulative_paths_upto_k"]),
            "retention_prune_rate_at_KMAX": float(final_row["retention_prune_rate"]),
            "runtime_sec_at_KMAX": float(final_row["runtime_sec"]),
        })

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    save_rows_csv(all_rows, OUT_CSV)
    save_summary_csv(summary_rows, OUT_SUMMARY_CSV)

    grouped = group_rows_by_method(all_rows)

    plot_path_growth(grouped, FIG_PATHS)
    plot_retention_pruning(grouped, FIG_PRUNING)
    plot_tipe_runtime(grouped, FIG_RUNTIME)

    print("\n=== Final summary ===")
    for row in summary_rows:
        print(row)

    print("\nDone.")
    print(f"All outputs saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
