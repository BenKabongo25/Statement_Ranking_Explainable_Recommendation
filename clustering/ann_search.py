import argparse
import faiss
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict, List, Tuple

from clustering.utils import load_embeddings, load_statements


def build_index(embeddings: torch.Tensor) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    np_embeddings = norm_embeddings.cpu().numpy().astype("float32")
    index = faiss.IndexFlatL2(np_embeddings.shape[1])
    index.add(np_embeddings)
    return index, np_embeddings


def build_sentiment_indices(
    df: pd.DataFrame, embeddings: torch.Tensor
) -> Dict[str, Tuple[List[int], faiss.IndexFlatL2, np.ndarray, Dict[int, int]]]:
    """Build one FAISS index per sentiment group.

    Returns a dict mapping sentiment -> (ids, index, np_embeddings, id_to_local).
    """
    sentiment_indices: Dict[str, Tuple[List[int], faiss.IndexFlatL2, np.ndarray, Dict[int, int]]] = {}
    for sentiment, group in df.groupby("sentiment"):
        ids = sorted(group.index.tolist())
        group_embeddings = embeddings[ids]
        index, np_embeddings = build_index(group_embeddings)
        id_to_local = {global_id: local_idx for local_idx, global_id in enumerate(ids)}
        sentiment_indices[sentiment] = (ids, index, np_embeddings, id_to_local)
    return sentiment_indices


def compute_full_neighbours(
    df: pd.DataFrame, embeddings: torch.Tensor, search_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top search_k neighbours within each sentiment group for every point.

    Returns:
        neighbours_idx: (N, search_k) int32 global indices (-1 if missing)
        neighbours_sim: (N, search_k) float32 cosine similarities (1 - dist/2, -inf if missing)
    """
    sentiment_indices = build_sentiment_indices(df, embeddings)
    n = len(df)
    neighbours_idx = np.full((n, search_k), -1, dtype=np.int32)
    neighbours_sim = np.full((n, search_k), -np.inf, dtype=np.float32)

    for sentiment, (ids, index, np_embeddings, _) in tqdm(
        sentiment_indices.items(), desc="Computing neighbours by sentiment"
    ):
        # +1 to ensure room to drop self from the results.
        k_group = min(search_k + 1, len(ids))
        distances, neighbours = index.search(np_embeddings, k_group)

        for local_row, global_idx in enumerate(ids):
            filled = 0
            for pos in range(neighbours.shape[1]):
                local_neigh = int(neighbours[local_row, pos])
                neigh_global = ids[local_neigh]
                if neigh_global == global_idx:
                    continue  # skip self
                if filled >= search_k:
                    break
                dist = float(distances[local_row, pos])
                cosine_sim = 1.0 - dist / 2.0
                neighbours_idx[global_idx, filled] = neigh_global
                neighbours_sim[global_idx, filled] = cosine_sim
                filled += 1

    return neighbours_idx, neighbours_sim


def threshold_stats(
    neighbours_idx: np.ndarray, neighbours_sim: np.ndarray, thresholds: List[float]
) -> List[Tuple[float, float, int]]:
    """Compute stats per threshold.

    Returns list of tuples (threshold, mean_similar_per_point, total_unique_pairs).
    """
    n, k = neighbours_idx.shape
    results: List[Tuple[float, float, int]] = []
    for thr in thresholds:
        mask = neighbours_sim >= thr
        counts = mask.sum(axis=1)
        mean_per_point = float(counts.mean())

        pairs = set()
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows.tolist(), cols.tolist()):
            j = int(neighbours_idx[r, c])
            if j < 0:
                continue
            a, b = sorted((r, j))
            pairs.add((a, b))
        results.append((thr, mean_per_point, len(pairs)))
    return results


def format_neighbours(
    indices: np.ndarray,
    distances: np.ndarray,
    df: pd.DataFrame,
    id_lookup: List[int],
    exclude_idx: int,
    k: int,
    reverse: bool = False,
) -> List[str]:
    """Prepare formatted neighbour lines.

    reverse=False -> take closest (smallest distance), reverse=True -> farthest.
    Similarity is derived from the L2 distance assuming normalised embeddings.
    """
    order = range(len(indices) - 1, -1, -1) if reverse else range(len(indices))
    lines: List[str] = []
    added = 0
    for pos in order:
        local_idx = int(indices[pos])
        global_idx = id_lookup[local_idx]
        if global_idx == exclude_idx:
            continue  # skip the query itself
        dist = float(distances[pos])
        cosine_sim = 1.0 - dist / 2.0  # dist from IndexFlatL2 is squared L2
        statement = str(df.loc[global_idx, "statement"])
        lines.append(f"{global_idx:>6} - {statement} (cos={cosine_sim: .4f}, dist={dist: .4f})")
        added += 1
        if added >= k:
            break
    return lines


def pretty_print_sample(
    query_idx: int,
    df: pd.DataFrame,
    top: List[str],
    bottom: List[str],
) -> None:
    statement = str(df.loc[query_idx, "statement"])
    sentiment = df.loc[query_idx, "sentiment"] if "sentiment" in df.columns else None

    print("=" * 80)
    header = f"Sample id={query_idx}"
    details = []
    if sentiment is not None:
        details.append(f"sentiment={sentiment}")
    if details:
        header += " (" + ", ".join(details) + ")"
    print(header)
    print(statement)
    print("\nTop similar:")
    for line in top:
        print("  " + line)
    print("\nBottom similar:")
    for line in bottom:
        print("  " + line)

    input_text = input("\nPress Enter to continue, or 'q' to quit: ")
    if input_text.strip().lower() == "q":
        print("Exiting.")
        exit(0)


def run_sample_mode(args, df: pd.DataFrame, embeddings: torch.Tensor) -> None:
    sentiment_indices = build_sentiment_indices(df, embeddings)

    random.seed(args.seed)
    sample_indices = random.sample(range(len(df)), k=min(args.samples, len(df)))

    for query_idx in sample_indices:
        sentiment = df.loc[query_idx, "sentiment"]
        ids, index, np_embeddings, id_to_local = sentiment_indices[sentiment]
        local_idx = id_to_local[query_idx]

        search_k = min(args.search_k, len(ids))
        distances, neighbours = index.search(np_embeddings[[local_idx]], search_k)
        distances_row, neighbours_row = distances[0], neighbours[0]

        top_lines = format_neighbours(
            neighbours_row, distances_row, df, ids, query_idx, k=args.top_k, reverse=False
        )
        bottom_lines = format_neighbours(
            neighbours_row, distances_row, df, ids, query_idx, k=args.bottom_k, reverse=True
        )
        pretty_print_sample(query_idx, df, top_lines, bottom_lines)


def run_full_mode(args, df: pd.DataFrame, embeddings: torch.Tensor) -> None:
    neighbours_idx, neighbours_sim = compute_full_neighbours(df, embeddings, search_k=args.search_k)
    np.savez(
        args.output_path,
        indices=neighbours_idx,
        similarities=neighbours_sim,
        search_k=args.search_k,
        thresholds=np.array(args.thresholds, dtype=np.float32),
    )
    stats = threshold_stats(neighbours_idx, neighbours_sim, args.thresholds)
    print(f"Saved neighbours to {args.output_path}")
    print("Threshold stats (mean similar per point, unique pairs):")
    for thr, mean_per_point, pair_count in stats:
        print(f"  thr={thr:.2f}: mean_per_point={mean_per_point:.2f}, unique_pairs={pair_count}")


def main(args) -> None:
    df = load_statements(args.statement_path)
    embeddings = load_embeddings(args.embedding_path)

    if len(df) != embeddings.shape[0]:
        raise ValueError(f"Row count mismatch: CSV has {len(df)} rows, embeddings have {embeddings.shape[0]} vectors.")

    if args.mode == "sample":
        run_sample_mode(args, df, embeddings)
    elif args.mode == "full":
        if not args.output_path:
            raise ValueError("Provide --output-path to save neighbours in full mode.")
        run_full_mode(args, df, embeddings)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--statement_path", 
        type=str, 
        required=True, 
        help="CSV file with statements (columns: statement, sentiment, ...)"
    )
    parser.add_argument(
        "--embedding_path", 
        type=str, 
        required=True, 
        help="Torch tensor file containing embeddings aligned with the CSV rows."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["sample", "full"], 
        default="sample", 
        help="sample: qualitative view, full: save neighbours for all points."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        help="Path to save neighbours (npz with indices and similarities) when mode=full."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=100, 
        help="Number of random statements to inspect (sample mode)."
    )
    parser.add_argument(
        "--search_k", 
        type=int, 
        default=128, 
        help="Large K used for the initial nearest-neighbour search."
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=20, 
        help="Number of closest neighbours to show."
    )
    parser.add_argument(
        "--bottom_k", 
        type=int, 
        default=20, 
        help="Number of farthest neighbours to show."
    )
    parser.add_argument(
        "--thresholds", 
        type=float, 
        nargs="+", 
        default=[0.8, 0.85, 0.9, 0.95], 
        help="Thresholds to compute stats on saved neighbours (full mode)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Seed used for the random sampling."
    )
    args = parser.parse_args()
    main(args)
