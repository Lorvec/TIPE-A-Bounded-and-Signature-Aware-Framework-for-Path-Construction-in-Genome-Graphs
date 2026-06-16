# ============================================================
# TABLE 2 — Output-size-matched held-out recovery experiment
# Real E. coli pangenome graph, high-branching anchors
#
# Purpose:
#   Reviewer 3: fair comparison against random/beam with matched retained-set size
#   Reviewer 4: clear K, graph neighborhoods, held-out sampling, #Paths definition
#
# Requires existing objects in memory:
#   - succ
#   - node_seq
#
# Outputs:
#   - table2_matched_raw.csv
#   - table2_matched_summary.csv
#   - table2_matched_printable.csv
#   - table2_debug.csv
# ============================================================

import os
import math
import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Set, Any

import pandas as pd


# ============================================================
# 0) Experiment configuration
# ============================================================

RANDOM_SEED = 42

# Number of valid high-branching loci to collect.
# If the graph does not provide enough valid loci, the script will report the actual number used.
N_EXPERIMENTS = 20

# Pool of candidate high-outdegree anchors searched to find valid loci.
N_ANCHORS_POOL = 2000
ANCHOR_MODE = "high_outdeg"

# Held-out setup
TRUE_PATHS_PER_LOCUS = 3
OBSERVED_SIZE = 2
READ_LEN = 15
MIN_RELAXED_FRAC = 0.8

# Maximum path length in edges
KMAX = 8

# TIPE reference budgets
BUDGETS = [1, 3, 5]

# Random matched baseline repetitions
N_RANDOM_SEEDS = 30

# Random-walk sampling parameters for held-out paths
MIN_SEQ_LEN = READ_LEN + 10
MAX_TRIALS_PER_ANCHOR = 3000

OUT_DIR = f"table2_matched_K{KMAX}_N{N_EXPERIMENTS}_R{N_RANDOM_SEEDS}_seed{RANDOM_SEED}"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_RAW_CSV = os.path.join(OUT_DIR, "table2_matched_raw.csv")
OUT_SUMMARY_CSV = os.path.join(OUT_DIR, "table2_matched_summary.csv")
OUT_PRINTABLE_CSV = os.path.join(OUT_DIR, "table2_matched_printable.csv")
OUT_DEBUG_CSV = os.path.join(OUT_DIR, "table2_debug.csv")


# ============================================================
# 1) Type aliases and graph helpers
# ============================================================

Path = Tuple[int, ...]
Signature = Tuple[int, ...]


def normalize_succ(succ_obj: Any) -> Dict[int, List[int]]:
    """
    Normalize adjacency structure to dict[int, list[int]].
    Accepts either a dict adjacency or a list-of-lists adjacency.
    """
    if isinstance(succ_obj, dict):
        return {int(k): list(v) for k, v in succ_obj.items()}
    if isinstance(succ_obj, list):
        return {i: list(v) for i, v in enumerate(succ_obj)}
    raise TypeError("succ must be either a dict or a list-of-lists.")


def get_node_sequence(node_seq_obj: Any, node: int) -> str:
    """
    Robust node sequence accessor.
    Accepts either dict[int, str] or list[str].
    """
    if isinstance(node_seq_obj, dict):
        return node_seq_obj.get(node, "")
    if isinstance(node_seq_obj, list):
        if 0 <= node < len(node_seq_obj):
            return node_seq_obj[node]
        return ""
    return ""


def path_to_sequence(path: Path, node_seq_obj: Any) -> str:
    """Convert a node path to its sequence by concatenating node sequences."""
    return "".join(get_node_sequence(node_seq_obj, v) for v in path)


def pick_hard_anchors(
    succ_dict: Dict[int, List[int]],
    top_k: int,
) -> List[int]:
    """
    Select high-branching anchors by highest outdegree.
    """
    candidates = [(u, len(outs)) for u, outs in succ_dict.items() if len(outs) > 0]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in candidates[:min(top_k, len(candidates))]]


# ============================================================
# 2) Signatures, ranking, pruning
# ============================================================

def is_simple_extension(path: Path, nxt: int) -> bool:
    """Return True if appending nxt preserves path simplicity."""
    return nxt not in path


def lex_key(path: Path) -> Tuple[int, ...]:
    """Deterministic lexicographic ranking key."""
    return path


def sig_start_end(path: Path) -> Signature:
    """Coarse signature: start and end node."""
    return (path[0], path[-1])


def sig_start_second_end(path: Path) -> Signature:
    """
    More expressive signature: start, second node, and end node.
    This is the TIPE setting used as the reference in Table 2.
    """
    if len(path) >= 2:
        return (path[0], path[1], path[-1])
    return (path[0], path[-1], path[-1])


def bounded_prune_lex(
    candidates: List[Path],
    B: int,
    sig_func: Callable[[Path], Signature],
) -> List[Path]:
    """
    TIPE retention:
    retain at most B paths per signature class using lexicographic ranking.
    """
    buckets: Dict[Signature, List[Path]] = defaultdict(list)

    for path in candidates:
        buckets[sig_func(path)].append(path)

    retained: List[Path] = []
    for signature in sorted(buckets.keys()):
        bucket = sorted(buckets[signature], key=lex_key)
        retained.extend(bucket[:B])

    return retained


def global_beam_select(
    candidates: List[Path],
    M: int,
) -> List[Path]:
    """
    Global beam-style retention:
    retain the top M candidates globally by lexicographic ranking.
    No signature grouping.
    """
    if M <= 0 or not candidates:
        return []
    return sorted(candidates, key=lex_key)[:min(M, len(candidates))]


def global_random_select(
    candidates: List[Path],
    M: int,
    rng: random.Random,
) -> List[Path]:
    """
    Global random retention:
    retain M candidates uniformly at random.
    No signature grouping.
    """
    if M <= 0 or not candidates:
        return []
    if len(candidates) <= M:
        return list(candidates)
    return rng.sample(candidates, M)


# ============================================================
# 3) Path expansion and retained-path collection
# ============================================================

def initial_candidates_from_anchor(
    succ_dict: Dict[int, List[int]],
    anchor: int,
) -> List[Path]:
    """Initial one-edge candidate paths from anchor."""
    return [(anchor, v) for v in succ_dict.get(anchor, [])]


def expand_paths_one_step(
    succ_dict: Dict[int, List[int]],
    Pk: List[Path],
) -> List[Path]:
    """
    Expand current retained paths by one edge under the simple-path constraint.
    """
    candidates: List[Path] = []

    by_terminal: Dict[int, List[Path]] = defaultdict(list)
    for path in Pk:
        by_terminal[path[-1]].append(path)

    for terminal, path_list in by_terminal.items():
        for path in path_list:
            for nxt in succ_dict.get(terminal, []):
                if is_simple_extension(path, nxt):
                    candidates.append(path + (nxt,))

    return candidates


def get_paths_upto_k(
    paths_by_k: Dict[int, List[Path]],
    max_k: int,
) -> List[Path]:
    """
    Collect unique retained paths across all lengths up to max_k.
    """
    out: List[Path] = []
    seen: Set[Path] = set()

    for k in range(1, max_k + 1):
        for path in paths_by_k.get(k, []):
            if path not in seen:
                out.append(path)
                seen.add(path)

    return out


# ============================================================
# 4) TIPE and matched baselines
# ============================================================

def run_tipe_reference(
    succ_dict: Dict[int, List[int]],
    anchor: int,
    Kmax: int,
    B: int,
    sig_func: Callable[[Path], Signature],
) -> Tuple[Dict[int, List[Path]], Dict[int, int]]:
    """
    Run TIPE from a fixed anchor.
    
    Returns:
      paths_by_k: retained paths at each length k
      match_schedule: M_k = number of active paths retained by TIPE at each k
    """
    paths_by_k: Dict[int, List[Path]] = {}
    match_schedule: Dict[int, int] = {}

    candidates = initial_candidates_from_anchor(succ_dict, anchor)
    Pk = bounded_prune_lex(candidates, B=B, sig_func=sig_func)
    paths_by_k[1] = Pk
    match_schedule[1] = len(Pk)

    for k in range(2, Kmax + 1):
        candidates = expand_paths_one_step(succ_dict, Pk)
        Pk = bounded_prune_lex(candidates, B=B, sig_func=sig_func)

        paths_by_k[k] = Pk
        match_schedule[k] = len(Pk)

        if not Pk:
            for kk in range(k + 1, Kmax + 1):
                paths_by_k[kk] = []
                match_schedule[kk] = 0
            break

    return paths_by_k, match_schedule


def run_beam_matched(
    succ_dict: Dict[int, List[int]],
    anchor: int,
    Kmax: int,
    match_schedule: Dict[int, int],
) -> Dict[int, List[Path]]:
    """
    Run global beam-style retention matched to TIPE active retained-set size.
    
    At each iteration k, retain up to M_k paths globally, where
    M_k is the number of paths retained by TIPE at the same k.
    """
    paths_by_k: Dict[int, List[Path]] = {}

    candidates = initial_candidates_from_anchor(succ_dict, anchor)
    Pk = global_beam_select(candidates, match_schedule.get(1, 0))
    paths_by_k[1] = Pk

    for k in range(2, Kmax + 1):
        candidates = expand_paths_one_step(succ_dict, Pk)
        Pk = global_beam_select(candidates, match_schedule.get(k, 0))
        paths_by_k[k] = Pk

        if not Pk:
            for kk in range(k + 1, Kmax + 1):
                paths_by_k[kk] = []
            break

    return paths_by_k


def run_random_matched(
    succ_dict: Dict[int, List[int]],
    anchor: int,
    Kmax: int,
    match_schedule: Dict[int, int],
    rng: random.Random,
) -> Dict[int, List[Path]]:
    """
    Run global random retention matched to TIPE active retained-set size.
    
    At each iteration k, retain up to M_k paths globally, where
    M_k is the number of paths retained by TIPE at the same k.
    """
    paths_by_k: Dict[int, List[Path]] = {}

    candidates = initial_candidates_from_anchor(succ_dict, anchor)
    Pk = global_random_select(candidates, match_schedule.get(1, 0), rng)
    paths_by_k[1] = Pk

    for k in range(2, Kmax + 1):
        candidates = expand_paths_one_step(succ_dict, Pk)
        Pk = global_random_select(candidates, match_schedule.get(k, 0), rng)
        paths_by_k[k] = Pk

        if not Pk:
            for kk in range(k + 1, Kmax + 1):
                paths_by_k[kk] = []
            break

    return paths_by_k


# ============================================================
# 5) Held-out path sampling
# ============================================================

def sample_simple_paths_from_anchor(
    succ_dict: Dict[int, List[int]],
    node_seq_obj: Any,
    anchor: int,
    n_paths: int,
    max_edges: int,
    min_seq_len: int,
    max_trials: int,
    seed: int,
) -> List[Path]:
    """
    Sample unique simple paths by random walks from a fixed anchor.
    Paths are accepted if their sequence length is at least min_seq_len.
    """
    rng = random.Random(seed)
    found: List[Path] = []
    found_set: Set[Path] = set()

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

        sequence = path_to_sequence(tuple(path), node_seq_obj)
        if len(sequence) < min_seq_len:
            continue

        tuple_path = tuple(path)
        if tuple_path not in found_set:
            found.append(tuple_path)
            found_set.add(tuple_path)

    return found


# ============================================================
# 6) Metrics
# ============================================================

def generate_reads(sequence: str, read_len: int) -> List[str]:
    """Generate all contiguous reads of length read_len from a sequence."""
    if len(sequence) < read_len:
        return []
    return [sequence[i:i + read_len] for i in range(len(sequence) - read_len + 1)]


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


def path_recall_exact_nodes(
    candidate_paths: List[Path],
    heldout_paths: List[Path],
) -> float:
    """
    Exact path recall using node-path identity.
    This is the main exact recall metric for Table 2.
    """
    if not heldout_paths:
        return 0.0

    candidate_set = set(candidate_paths)
    hits = sum(path in candidate_set for path in heldout_paths)

    return hits / len(heldout_paths)


def path_recall_exact_sequences(
    candidate_sequences: List[str],
    heldout_sequences: List[str],
) -> float:
    """
    Exact path recall using concatenated sequence identity.
    Kept as an auxiliary metric.
    """
    if not heldout_sequences:
        return 0.0

    candidate_set = set(candidate_sequences)
    hits = sum(sequence in candidate_set for sequence in heldout_sequences)

    return hits / len(heldout_sequences)


def path_recall_relaxed(
    candidate_sequences: List[str],
    heldout_sequences: List[str],
    min_frac: float,
    read_len: int,
) -> float:
    """
    Relaxed held-out path recall.

    A held-out path is counted as recovered if at least min_frac of its
    read windows appear in at least one candidate path.
    """
    if not heldout_sequences:
        return 0.0

    recovered = 0

    for sequence in heldout_sequences:
        reads = generate_reads(sequence, read_len)

        if not reads:
            continue

        covered = 0
        for read in reads:
            if any(read in candidate for candidate in candidate_sequences):
                covered += 1

        frac = covered / len(reads)

        if frac >= min_frac:
            recovered += 1

    return recovered / len(heldout_sequences)


def diversity_signature_classes(
    candidate_paths: List[Path],
    sig_func: Callable[[Path], Signature],
) -> int:
    """
    Diversity metric:
    number of distinct signature classes represented in the final retained set.
    """
    signatures = {sig_func(path) for path in candidate_paths}
    return len(signatures)


def evaluate_candidate_set(
    candidate_paths: List[Path],
    heldout_paths: List[Path],
    node_seq_obj: Any,
) -> Dict[str, float]:
    """
    Compute all metrics for one candidate set.
    """
    candidate_sequences = [
        path_to_sequence(path, node_seq_obj)
        for path in candidate_paths
    ]

    heldout_sequences = [
        path_to_sequence(path, node_seq_obj)
        for path in heldout_paths
    ]

    return {
        "num_candidate_paths": len(candidate_paths),
        "exact_recall": path_recall_exact_nodes(
            candidate_paths,
            heldout_paths,
        ),
        "exact_sequence_recall": path_recall_exact_sequences(
            candidate_sequences,
            heldout_sequences,
        ),
        "read_recall": read_recall(
            candidate_sequences,
            heldout_sequences,
            READ_LEN,
        ),
        "relaxed_recall": path_recall_relaxed(
            candidate_sequences,
            heldout_sequences,
            min_frac=MIN_RELAXED_FRAC,
            read_len=READ_LEN,
        ),
        "diversity": diversity_signature_classes(
            candidate_paths,
            sig_start_second_end,
        ),
    }


# ============================================================
# 7) Main matched experiment
# ============================================================

def run_table2_matched_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the output-size-matched Table 2 experiment.
    """
    required_names = ["succ", "node_seq"]
    missing = [name for name in required_names if name not in globals()]

    if missing:
        raise RuntimeError(
            "Missing required objects: "
            + ", ".join(missing)
            + ". Please load the real graph first so that succ and node_seq exist."
        )

    succ_dict = normalize_succ(globals()["succ"])
    node_seq_obj = globals()["node_seq"]

    anchors_pool = pick_hard_anchors(succ_dict, top_k=N_ANCHORS_POOL)

    print("=== Table 2 matched experiment ===")
    print(f"KMAX = {KMAX}")
    print(f"Reference TIPE budgets = {BUDGETS}")
    print(f"Requested valid loci = {N_EXPERIMENTS}")
    print(f"Anchor mode = {ANCHOR_MODE}")
    print(f"Anchor pool size = {len(anchors_pool)}")
    print(f"Random matched seeds = {N_RANDOM_SEEDS}")
    print(f"Output directory = {OUT_DIR}")

    raw_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    used_loci = 0

    for pool_idx, anchor in enumerate(anchors_pool):
        true_paths = sample_simple_paths_from_anchor(
            succ_dict=succ_dict,
            node_seq_obj=node_seq_obj,
            anchor=anchor,
            n_paths=TRUE_PATHS_PER_LOCUS,
            max_edges=KMAX,
            min_seq_len=MIN_SEQ_LEN,
            max_trials=MAX_TRIALS_PER_ANCHOR,
            seed=RANDOM_SEED + pool_idx,
        )

        debug_rows.append({
            "anchor": anchor,
            "pool_idx": pool_idx,
            "outdegree": len(succ_dict.get(anchor, [])),
            "n_true_paths_found": len(true_paths),
        })

        if len(true_paths) < TRUE_PATHS_PER_LOCUS:
            continue

        # Deterministic split shuffle
        split_rng = random.Random(RANDOM_SEED + 100_000 + pool_idx)
        true_paths = list(true_paths)
        split_rng.shuffle(true_paths)

        observed_paths = true_paths[:OBSERVED_SIZE]
        heldout_paths = true_paths[OBSERVED_SIZE:]

        if not heldout_paths:
            continue

        # Check sequence length of held-out paths
        heldout_sequences = [path_to_sequence(path, node_seq_obj) for path in heldout_paths]
        if not all(len(sequence) >= READ_LEN for sequence in heldout_sequences):
            continue

        used_loci += 1
        locus_id = used_loci

        # Optional haplotype-only baseline, useful for internal checks.
        haplo_metrics = evaluate_candidate_set(
            candidate_paths=observed_paths,
            heldout_paths=heldout_paths,
            node_seq_obj=node_seq_obj,
        )

        raw_rows.append({
            "locus_id": locus_id,
            "anchor": anchor,
            "method": "Haplotype-only",
            "ref_B": 0,
            "seed": -1,
            "K": KMAX,
            "matched_to_tipe": False,
            "match_schedule": "",
            **haplo_metrics,
        })

        for B in BUDGETS:
            # ------------------------------------------------
            # TIPE reference: signature-aware retention
            # ------------------------------------------------
            tipe_paths_by_k, match_schedule = run_tipe_reference(
                succ_dict=succ_dict,
                anchor=anchor,
                Kmax=KMAX,
                B=B,
                sig_func=sig_start_second_end,
            )

            tipe_candidate_paths = get_paths_upto_k(tipe_paths_by_k, KMAX)
            tipe_metrics = evaluate_candidate_set(
                candidate_paths=tipe_candidate_paths,
                heldout_paths=heldout_paths,
                node_seq_obj=node_seq_obj,
            )

            schedule_str = ";".join(
                f"{k}:{match_schedule.get(k, 0)}"
                for k in range(1, KMAX + 1)
            )

            raw_rows.append({
                "locus_id": locus_id,
                "anchor": anchor,
                "method": "TIPE",
                "ref_B": B,
                "seed": -1,
                "K": KMAX,
                "matched_to_tipe": True,
                "match_schedule": schedule_str,
                **tipe_metrics,
            })

            # ------------------------------------------------
            # Beam matched: global lexicographic retention
            # ------------------------------------------------
            beam_paths_by_k = run_beam_matched(
                succ_dict=succ_dict,
                anchor=anchor,
                Kmax=KMAX,
                match_schedule=match_schedule,
            )

            beam_candidate_paths = get_paths_upto_k(beam_paths_by_k, KMAX)
            beam_metrics = evaluate_candidate_set(
                candidate_paths=beam_candidate_paths,
                heldout_paths=heldout_paths,
                node_seq_obj=node_seq_obj,
            )

            raw_rows.append({
                "locus_id": locus_id,
                "anchor": anchor,
                "method": "Beam matched",
                "ref_B": B,
                "seed": -1,
                "K": KMAX,
                "matched_to_tipe": True,
                "match_schedule": schedule_str,
                **beam_metrics,
            })

            # ------------------------------------------------
            # Random matched: repeated global random retention
            # ------------------------------------------------
            for seed_idx in range(N_RANDOM_SEEDS):
                rng = random.Random(
                    RANDOM_SEED
                    + 1_000_000
                    + 10_000 * B
                    + 100 * seed_idx
                    + locus_id
                )

                random_paths_by_k = run_random_matched(
                    succ_dict=succ_dict,
                    anchor=anchor,
                    Kmax=KMAX,
                    match_schedule=match_schedule,
                    rng=rng,
                )

                random_candidate_paths = get_paths_upto_k(random_paths_by_k, KMAX)
                random_metrics = evaluate_candidate_set(
                    candidate_paths=random_candidate_paths,
                    heldout_paths=heldout_paths,
                    node_seq_obj=node_seq_obj,
                )

                raw_rows.append({
                    "locus_id": locus_id,
                    "anchor": anchor,
                    "method": "Random matched",
                    "ref_B": B,
                    "seed": seed_idx,
                    "K": KMAX,
                    "matched_to_tipe": True,
                    "match_schedule": schedule_str,
                    **random_metrics,
                })

        if used_loci >= N_EXPERIMENTS:
            break

    raw_df = pd.DataFrame(raw_rows)
    debug_df = pd.DataFrame(debug_rows)

    print(f"\nUsed valid loci: {used_loci}")
    print(f"Raw rows collected: {len(raw_df)}")

    if used_loci < N_EXPERIMENTS:
        print(
            f"\nWARNING: Only {used_loci} valid loci were collected "
            f"out of requested {N_EXPERIMENTS}. "
            f"Consider increasing N_ANCHORS_POOL or MAX_TRIALS_PER_ANCHOR."
        )

    if raw_df.empty:
        raise RuntimeError("No results collected. Try increasing N_ANCHORS_POOL or MAX_TRIALS_PER_ANCHOR.")

    # Save raw and debug
    raw_df.to_csv(OUT_RAW_CSV, index=False)
    debug_df.to_csv(OUT_DEBUG_CSV, index=False)

    # Summary for matched methods only
    matched_df = raw_df[
        raw_df["method"].isin(["Random matched", "Beam matched", "TIPE"])
    ].copy()

    summary_df = (
        matched_df
        .groupby(["method", "ref_B"], as_index=False)
        .agg(
            n=("exact_recall", "count"),
            num_candidate_paths_mean=("num_candidate_paths", "mean"),
            num_candidate_paths_std=("num_candidate_paths", "std"),
            exact_recall_mean=("exact_recall", "mean"),
            exact_recall_std=("exact_recall", "std"),
            diversity_mean=("diversity", "mean"),
            diversity_std=("diversity", "std"),
            read_recall_mean=("read_recall", "mean"),
            read_recall_std=("read_recall", "std"),
            relaxed_recall_mean=("relaxed_recall", "mean"),
            relaxed_recall_std=("relaxed_recall", "std"),
            exact_sequence_recall_mean=("exact_sequence_recall", "mean"),
            exact_sequence_recall_std=("exact_sequence_recall", "std"),
        )
    )

    # Stable table order
    method_order = {
        "Random matched": 0,
        "Beam matched": 1,
        "TIPE": 2,
    }

    summary_df["method_order"] = summary_df["method"].map(method_order)
    summary_df = summary_df.sort_values(["ref_B", "method_order"]).drop(columns=["method_order"])

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    # Printable Table 2
    def safe_std(x: float) -> float:
        if pd.isna(x):
            return 0.0
        return float(x)

    def fmt_pm(mean_val: float, std_val: float, digits: int = 3) -> str:
        return f"{mean_val:.{digits}f} ± {safe_std(std_val):.{digits}f}"

    printable_rows = []
    for _, row in summary_df.iterrows():
        printable_rows.append({
            "Method": row["method"],
            "Ref. B": int(row["ref_B"]),
            "#Paths": fmt_pm(row["num_candidate_paths_mean"], row["num_candidate_paths_std"], digits=2),
            "Exact recall": fmt_pm(row["exact_recall_mean"], row["exact_recall_std"], digits=3),
            "Diversity": fmt_pm(row["diversity_mean"], row["diversity_std"], digits=2),
            "Read recall": fmt_pm(row["read_recall_mean"], row["read_recall_std"], digits=3),
            "Relaxed recall": fmt_pm(row["relaxed_recall_mean"], row["relaxed_recall_std"], digits=3),
            "n": int(row["n"]),
        })

    printable_df = pd.DataFrame(printable_rows)
    printable_df.to_csv(OUT_PRINTABLE_CSV, index=False)

    print("\n=== Printable Table 2 ===")
    print(printable_df.to_string(index=False))

    print("\nSaved:")
    print(f"  Raw:       {OUT_RAW_CSV}")
    print(f"  Summary:   {OUT_SUMMARY_CSV}")
    print(f"  Printable: {OUT_PRINTABLE_CSV}")
    print(f"  Debug:     {OUT_DEBUG_CSV}")

    return raw_df, summary_df, printable_df


# ============================================================
# 8) Run
# ============================================================

raw_table2, summary_table2, printable_table2 = run_table2_matched_experiment()
