"""Benchmark gdf_to_pyg() on the Liverpool case-study OA graph (paper section 3.2).

Replicates the protocol reported in the paper: full conversion of the OA-level
graph (1,624 nodes; contig + 15_min_walk + 15_min_multi relations, 103,299
undirected edges) with gdf_to_pyg() on a laptop, measured as wall time.

Run from the repository root after notebook 02 has produced
data/processed/graphs/hetero_multi.pt:

    uv run python scripts/benchmark_gdf_to_pyg.py
"""

import platform
import statistics
import subprocess
import time

import torch

import city2graph as c2g

GRAPH_PATH = "data/processed/graphs/hetero_multi.pt"
EDGE_TYPES = [
    ("oa", "contig", "oa"),
    ("oa", "15_min_walk", "oa"),
    ("oa", "15_min_multi", "oa"),
]
EDGE_FEATURE_COLS = {et: ["travel_time_sec"] for et in EDGE_TYPES}
N_REPEATS = 5


def machine_info() -> str:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = platform.processor() or "unknown"
    try:
        mem_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        mem = f"{mem_bytes / 1024**3:.0f}GB"
    except Exception:
        mem = "unknown"
    return f"{chip} {mem} ({platform.platform()})"


def main() -> None:
    data = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    all_nodes, all_edges = c2g.pyg_to_gdf(data)

    nodes = {"oa": all_nodes["oa"]}
    edges = {et: all_edges[et] for et in EDGE_TYPES}

    n_nodes = len(nodes["oa"])
    n_edges_undirected = sum(len(e) for e in edges.values())
    print(f"city2graph {c2g.__version__} | torch {torch.__version__}")
    print(f"machine: {machine_info()}")
    print(f"input: {n_nodes} oa nodes, {n_edges_undirected} undirected edges "
          f"across {len(EDGE_TYPES)} relation types")
    for et, e in edges.items():
        print(f"  {et}: {len(e)}")

    # Single cold run (comparable to the value reported in the paper)
    t0 = time.perf_counter()
    out = c2g.gdf_to_pyg(nodes, edges, edge_feature_cols=EDGE_FEATURE_COLS)
    cold = time.perf_counter() - t0
    print(f"\ncold run: {cold:.3f} s")

    n_directed = sum(out[et].edge_index.shape[1] for et in EDGE_TYPES)
    print(f"resulting edge_index entries (both directions): {n_directed}")

    # Repeated runs for stability
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        c2g.gdf_to_pyg(nodes, edges, edge_feature_cols=EDGE_FEATURE_COLS)
        times.append(time.perf_counter() - t0)
    print(f"median of {N_REPEATS} repeats: {statistics.median(times):.3f} s "
          f"(min {min(times):.3f} s, max {max(times):.3f} s)")


if __name__ == "__main__":
    main()
