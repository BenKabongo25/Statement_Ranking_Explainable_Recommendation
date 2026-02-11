import argparse
import ast
import json
import math
import os
import pandas as pd
import random
import torch

from tqdm import tqdm
from typing import List, Dict, Optional, Tuple


def parse_member_indices(s) -> List[int]:
    if isinstance(s, list):
        return [int(x) for x in s]
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [int(x) for x in v]
        return []
    except Exception:
        return []


def load_embeddings(path: str) -> torch.Tensor:
    E = torch.load(path, map_location="cpu")
    if E.dim() != 2:
        raise ValueError(f"Embeddings must be 2D (N,D). Got {tuple(E.shape)}")
    return E.contiguous()


def dist_from_cos(cos_sim: torch.Tensor) -> torch.Tensor:
    return (1.0 - cos_sim).clamp_min(0.0)


def sample_indices(indices: List[int], k: Optional[int], rng: random.Random) -> List[int]:
    if k is None or k <= 0 or k >= len(indices):
        return indices
    return rng.sample(indices, k)


@torch.no_grad()
def SSE(
    E: torch.Tensor,
    clusters: List[Dict],
    device: torch.device,
    chunk_size: int = 10240,
) -> Tuple[float, float]:
    """
    SSE(C_i) = (1 / (2 m_i)) * sum_{x in C_i} sum_{y in C_i} d(x,y)^2
        with d(x,y) = 1 - cos(x,y).

    Returns:
      sse_sum = sum_i SSE(C_i)
      sse_mean = (1/K) * sum_i SSE(C_i)
    """
    E = E.to(device)
    K = len(clusters)
    sse_sum = 0.0

    for c in clusters:
        members = c["member_indices"]
        m = len(members)
        if m <= 1:
            continue

        X = E[members]  # (m, d),
        total = 0.0
        for i in range(0, m, chunk_size):
            A = X[i : i + chunk_size]         # (a, d)
            cos = A @ X.t()                   # (a, m)
            d = dist_from_cos(cos)            # (a, m)
            total += float((d ** 2).sum().item())

        sse_ci = total / (2.0 * m)
        sse_sum += sse_ci

    sse_mean = sse_sum / float(K) if K > 0 else float("nan")
    return sse_sum, sse_mean


@torch.no_grad()
def SSB(
    E: torch.Tensor,
    clusters: List[Dict],
    device: torch.device,
    chunk_size: int = 10240,
) -> float:
    """
    SSB = (1 / (2K)) * sum_{i=1}^K sum_{j=1}^K d(c_i, c_j)^2
        with d(x,y) = 1 - cos(x,y) and c_i the representative of cluster i.
    """
    E = E.to(device)
    reps = [int(c["representative_index"]) for c in clusters]
    K = len(reps)
    Ni = torch.tensor([len(c["member_indices"]) for c in clusters], device=device)  # (K,)
    N = Ni.sum().item()
    Ni = (Ni / N)
    if K == 0:
        return float("nan")

    R = E[reps]  # (K, d), normalized
    total = 0.0

    for i in range(0, K, chunk_size):
        A = R[i : i + chunk_size]   # (a, d)
        cos = A @ R.t()             # (a, K)
        d = dist_from_cos(cos)      # (a, K)
        total += float( ((d ** 2).sum(dim=0) * Ni).sum().item() )

    ssb = total / (2.0 * K)
    return ssb


@torch.no_grad()
def cluster_diameter(
    E: torch.Tensor,
    members: List[int],
    device: torch.device,
    sample_size: Optional[int],
    rng: random.Random,
    block: int = 10240,
) -> float:
    """
    Δ(C) = max_{x,y in C} d(x,y) with d(x,y) = 1 - cos(x,y)
    """
    if len(members) <= 1:
        return 0.0

    sel = sample_indices(members, sample_size, rng)
    X = E[sel].to(device)

    max_d = 0.0
    n = X.shape[0]
    for i in range(0, n, block):
        A = X[i : i + block]
        cos = A @ X.t()
        d = dist_from_cos(cos)
        max_d = max(max_d, float(d.max().item()))
    return max_d


@torch.no_grad()
def intercluster_min_distance(
    E: torch.Tensor,
    members_a: List[int],
    members_b: List[int],
    device: torch.device,
    sample_size: Optional[int],
    rng: random.Random,
    block_a: int = 10240,
    block_b: int = 10240,
) -> float:
    """
    δ(C_i,C_j) = min_{x in Ci, y in Cj} d(x,y), with d(x,y) = 1 - cos(x,y)
    """
    if not members_a or not members_b:
        return float("inf")

    A_idx = sample_indices(members_a, sample_size, rng)
    B_idx = sample_indices(members_b, sample_size, rng)

    A = E[A_idx].to(device)
    B = E[B_idx].to(device)

    min_d = float("inf")
    nb = B.shape[0]

    for j in range(0, nb, block_b):
        Bblk = B[j : j + block_b]
        na = A.shape[0]
        for i in range(0, na, block_a):
            Ablk = A[i : i + block_a]
            cos = Ablk @ Bblk.t()
            d = dist_from_cos(cos)
            cur = float(d.min().item())
            if cur < min_d:
                min_d = cur

    return min_d


@torch.no_grad()
def dunn_index(
    E: torch.Tensor,
    clusters: List[Dict],
    device: torch.device,
    sample_size: Optional[int],
    seed: int,
) -> Tuple[float, float, float]:
    """
    Dunn Index:
      D = min_{i!=j} δ(C_i,C_j) / max_l Δ(C_l)
    with d = 1 - cos.
    """
    rng = random.Random(seed)

    print("Computing cluster diameters for Dunn Index...")
    max_diam = 0.0
    for c in clusters:
        diam = cluster_diameter(E, c["member_indices"], device, sample_size, rng)
        max_diam = max(max_diam, diam)
    print(f"Max intracluster diameter: {max_diam:.6f}")

    print(f"Computing intercluster distances for Dunn Index")
    min_inter = float("inf")
    for i in tqdm(range(len(clusters))):
        for j in range(i + 1, len(clusters)):
            d = intercluster_min_distance(
                E,
                clusters[i]["member_indices"],
                clusters[j]["member_indices"],
                device,
                sample_size,
                rng,
            )
            min_inter = min(min_inter, d)
    print(f"Min intercluster distance: {min_inter:.6f}")

    if max_diam <= 0.0:
        D = float("inf") if math.isfinite(min_inter) else float("nan")
    else:
        D = min_inter / max_diam

    return D, min_inter, max_diam


def main(args):
    print("Loading clusters and embeddings...")
    df = pd.read_csv(args.cluster_path)

    for col in ["member_indices", "cluster_size", "representative_index"]:
        if col not in df.columns:
            raise ValueError(f"clusters.csv missing required column '{col}'. Found: {list(df.columns)}")

    clusters: List[Dict] = []
    sizes_list: List[int] = []
    for _, row in df.iterrows():
        members = parse_member_indices(row["member_indices"])
        if not members:
            continue
        clusters.append(
            {
                "cluster_size": int(row["cluster_size"]) if int(row["cluster_size"]) > 0 else len(members),
                "member_indices": members,
                "representative_index": int(row["representative_index"]),
            }
        )
        sizes_list.append(clusters[-1]["cluster_size"])

    E = load_embeddings(args.embedding_path)
    K = len(clusters)
    N = E.shape[0]
    reduc = ((N - K) / N) * 100.0 if N > 0 else 0.0
    print(f"Loaded {K} clusters with total {sum(sizes_list)} members, and {N} embeddings.")

    print("Number of clusters: ", K)
    print("Number of points: ", N)
    print("Reduction %:", reduc)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Computing SSE (pairwise, 1-cos)...")
    sse_sum, sse_mean = SSE(E, clusters, device=device, chunk_size=args.block_size)
    print(f"SSE_sum:  {sse_sum:.6f}")
    print(f"SSE_mean: {sse_mean:.6f} (mean over K clusters)")

    print("Computing SSB (pairwise over representatives, 1-cos)...")
    ssb = SSB(E, clusters, device=device, chunk_size=args.block_size)
    print(f"SSB: {ssb:.6f}")

    #dunn_sample = None if args.dunn_sample_size <= 0 else args.dunn_sample_size

    #print("Computing Dunn Index (1-cos)...")
    #D, min_inter, max_diam = dunn_index(
    #    E=E.to(device),
    #    clusters=clusters,
    #    device=device,
    #    sample_size=dunn_sample,
    #    seed=args.seed,
    #)
    #print(f"Dunn Index: {D:.6f} (min inter: {min_inter:.6f}, max diam: {max_diam:.6f})")

    report = {
        "cluster_path": args.cluster_path,
        "embedding_path": args.embedding_path,
        "num_points": int(N),
        "num_clusters": int(K),
        "reduction": reduc,
        "metrics": {
            "SSE_sum": float(sse_sum),
            "SSE_mean_over_K": float(sse_mean),
            "SSB": float(ssb),
        #    "DunnIndex": float(D),
        #    "Dunn_min_intercluster_distance": float(min_inter) if math.isfinite(min_inter) else None,
        #    "Dunn_max_intracluster_diameter": float(max_diam),
        },
        #"settings": {
        #    "dunn_sample_size_per_cluster": dunn_sample,
        #    "seed": args.seed,
        #    "device": str(device),
        #},
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_path",
        type=str,
        required=True,
        help="Path to clusters CSV file with columns: member_indices (list of ints), cluster_size (int), representative_index (int)",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Torch tensor file containing L2-normalized embeddings aligned with statement indices.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write a JSON report with SSE/SSB/Dunn.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=10240,
        help="Block size for pairwise computations (tradeoff speed/memory).",
    )

    #parser.add_argument(
    #    "--dunn_sample_size",
    #    type=int,
    #    default=-1,
    #    help="Sample points per cluster for Dunn computations. Set <=0 for exact (slow).",
    #)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )

    args = parser.parse_args()
    main(args)