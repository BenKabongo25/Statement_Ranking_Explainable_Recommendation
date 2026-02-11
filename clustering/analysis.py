import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F

from typing import Dict, List, Sequence, Tuple
from clustering.utils import (
    load_clusters,
    load_embeddings,
    load_statements,
    mlp_config,
    tsne_scatter,
)


def sample_clusters(df_clusters: pd.DataFrame, n: int, min_size: int, seed: int) -> pd.DataFrame:
    eligible = df_clusters[df_clusters["cluster_size"] >= min_size]
    if eligible.empty:
        return df_clusters.head(0)
    weights = eligible["cluster_size"].to_numpy(dtype=float)
    probs = weights / weights.sum()
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(eligible), size=min(n, len(eligible)), replace=False, p=probs)
    return eligible.iloc[chosen_idx]


def nearest_and_farthest(
    rep_idx: int,
    members: Sequence[int],
    norm_embeddings: torch.Tensor,
    top_k: int,
    bottom_k: int,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    member_tensor = norm_embeddings[list(members)]
    rep_vec = norm_embeddings[[rep_idx]]
    sims = (member_tensor @ rep_vec.T).squeeze(1)  # cosine similarities

    pairs = list(zip(members, sims.tolist()))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top = [p for p in pairs_sorted if p[0] != rep_idx][:top_k]
    bottom = [p for p in reversed(pairs_sorted) if p[0] != rep_idx][:bottom_k]
    return top, bottom


def build_statement_cluster_mapping(df_clusters: pd.DataFrame) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for cid, row in df_clusters.iterrows():
        for sid in row["member_indices"]:
            mapping[int(sid)] = int(cid)
    return mapping


def print_cluster_view(
    cluster_row: pd.Series,
    df_statement: pd.DataFrame,
    norm_embeddings: torch.Tensor,
    top_k: int,
    bottom_k: int,
) -> None:
    rep_idx = int(cluster_row["representative_index"])
    members = cluster_row["member_indices"]
    rep_stmt = str(df_statement.loc[rep_idx, "statement"])
    sentiment = cluster_row["sentiment"]

    top, bottom = nearest_and_farthest(rep_idx, members, norm_embeddings, top_k, bottom_k)

    print("=" * 80)
    print(f"Sentiment: {sentiment} | Cluster size: {len(members)} | Representative idx: {rep_idx}")
    print(f"Representative: {rep_stmt}")
    print("\nTop similar:")
    for idx, sim in top:
        print(f"  {idx:>6} | cos={sim: .4f} | {df_statement.loc[idx, 'statement']}")
    print("\nBottom similar:")
    for idx, sim in bottom:
        print(f"  {idx:>6} | cos={sim: .4f} | {df_statement.loc[idx, 'statement']}")
    print()


def sample_tsne_indices(df_statement: pd.DataFrame, target_total: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    total = len(df_statement)
    collected: List[int] = []
    for sentiment, group in df_statement.groupby("sentiment"):
        n_sent = len(group)
        target = max(1, min(n_sent, int(round(target_total * n_sent / total))))
        idxs = group.index.to_list()
        if n_sent > target:
            idxs = rng.choice(idxs, size=target, replace=False).tolist()
        collected.extend(idxs)
    return collected


def tsne_per_sentiment(
    df_statement: pd.DataFrame,
    norm_embeddings: torch.Tensor,
    cluster_ids: np.ndarray,
    output_dir: str,
    per_sentiment_limit: int,
    random_state: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mlp_config()
    rng = np.random.default_rng(random_state)

    for sentiment, group in df_statement.groupby("sentiment"):
        idxs = group.index.to_list()
        if len(idxs) > per_sentiment_limit:
            idxs = rng.choice(idxs, size=per_sentiment_limit, replace=False).tolist()
        X = norm_embeddings[idxs].cpu().numpy()
        global_labels = cluster_ids[idxs]
        unique_clusters = {cid: i for i, cid in enumerate(sorted(set(global_labels.tolist())))}
        local_labels = np.array([unique_clusters[cid] for cid in global_labels], dtype=int)
        title = f"t-SNE sentiment={sentiment}"
        out_path = os.path.join(output_dir, f"tsne_{sentiment}.pdf")
        tsne_scatter(
            X,
            local_labels,
            title,
            out_path,
            perplexity=30.0,
            learning_rate=200.0,
            n_iter=1000,
            early_exaggeration=12.0,
            angle=0.5,
            metric="cosine",
            random_state=random_state,
        )


def tsne_global(
    idxs: List[int],
    df_statement: pd.DataFrame,
    norm_embeddings: torch.Tensor,
    cluster_ids: np.ndarray,
    output_path: str,
    random_state: int,
) -> None:
    if not idxs:
        return
    mlp_config()
    X = norm_embeddings[idxs].cpu().numpy()
    colors = cluster_ids[idxs] % 20  # reuse tab20 palette
    sentiments = df_statement.loc[idxs, "sentiment"].tolist()
    unique_sentiments = sorted(set(sentiments))
    markers = ["o", "s", "^"]
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(unique_sentiments)}

    # Compute t-SNE
    n = X.shape[0]
    if n < 2:
        return
    max_safe = max(2, (n - 1) // 3)
    eff_perp = min(30.0, max(5, max_safe)) if n > 10 else max(2, min(30.0, n - 1))
    if eff_perp >= n:
        eff_perp = max(2, n - 1)
    from sklearn.manifold import TSNE

    X2 = TSNE(
        n_components=2,
        perplexity=eff_perp,
        learning_rate=200.0,
        n_iter=1000,
        init="pca",
        random_state=random_state,
        early_exaggeration=12.0,
        angle=0.5,
        metric="cosine",
        verbose=0,
    ).fit_transform(X)

    plt.figure(figsize=(7.0, 5.5), dpi=1200)
    for sentiment in unique_sentiments:
        mask = [s == sentiment for s in sentiments]
        plt.scatter(
            X2[mask, 0],
            X2[mask, 1],
            c=colors[mask],
            cmap="tab20",
            s=1,
            alpha=0.7,
            marker=marker_map[sentiment],
            label=sentiment,
        )
    plt.title(f"Global t-SNE")
    plt.grid(True)
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close()


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    df_statement = load_statements(args.statement_path)
    embeddings = load_embeddings(args.embedding_path)
    if embeddings.shape[0] != len(df_statement):
        raise ValueError(
            f"Row count mismatch: CSV has {len(df_statement)} rows, embeddings have {embeddings.shape[0]} vectors."
        )
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)

    df_clusters = load_clusters(args.clusters_path)
    df_clusters = df_clusters.reset_index().rename(columns={"index": "cluster_id"})

    # Build statement -> cluster mapping and save JSON.
    stmt_to_cluster = build_statement_cluster_mapping(df_clusters)
    cluster_ids = np.full(len(df_statement), -1, dtype=int)
    for sid, cid in stmt_to_cluster.items():
        if sid < len(cluster_ids):
            cluster_ids[sid] = cid
    if args.statement2cluster_path:
        os.makedirs(os.path.dirname(args.statement2cluster_path) or ".", exist_ok=True)
        with open(args.statement2cluster_path, "w") as f:
            json.dump({str(k): int(v) for k, v in stmt_to_cluster.items()}, f, ensure_ascii=False, indent=2)
        print(f"Saved statement->cluster mapping to {args.statement2cluster_path}")

    sampled = sample_clusters(df_clusters, n=args.num_clusters, min_size=args.min_cluster_size, seed=args.seed)
    if sampled.empty:
        print("No clusters eligible for qualitative analysis.")
    else:
        for _, row in sampled.iterrows():
            print_cluster_view(row, df_statement, norm_embeddings, top_k=args.top_k, bottom_k=args.bottom_k)

    if args.tsne_output_dir:
        # t-SNE per sentiment with local cluster labels.
        tsne_per_sentiment(
            df_statement,
            norm_embeddings,
            cluster_ids=cluster_ids,
            output_dir=args.tsne_output_dir,
            per_sentiment_limit=args.tsne_points_per_sentiment,
            random_state=args.seed,
        )

        # Global t-SNE on a proportional sample (~tsne_total_points).
        sampled_idxs = sample_tsne_indices(df_statement, target_total=args.tsne_total_points, seed=args.seed)
        global_out = os.path.join(args.tsne_output_dir, "tsne_global.pdf")
        tsne_global(
            sampled_idxs,
            df_statement,
            norm_embeddings,
            cluster_ids=cluster_ids,
            output_path=global_out,
            random_state=args.seed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clusters_path", 
        type=str,
        required=True, 
        help="CSV produced by graph_clustering.py."
    )
    parser.add_argument(
        "--statement_path", 
        type=str,
        required=True, 
        help="CSV with columns 'statement' and 'sentiment'."
    )
    parser.add_argument(
        "--embedding_path", 
        type=str,
        required=True, 
        help="Torch tensor with embeddings aligned to statements CSV."
    )
    parser.add_argument(
        "--statement2cluster_path", 
        type=str,
        default=None, 
        help="Path to save the statement->cluster JSON mapping."
    )
    parser.add_argument(
        "--num_clusters", 
        type=int, 
        default=100, 
        help="Number of clusters to sample for qualitative view."
    )
    parser.add_argument(
        "--min_cluster_size", 
        type=int, 
        default=3, 
        help="Minimum size to consider a cluster for sampling."
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Top-k closest members to the representative."
    )
    parser.add_argument(
        "--bottom_k", 
        type=int, 
        default=5, 
        help="Bottom-k farthest members from the representative."
    )
    parser.add_argument(
        "--tsne_output_dir", 
        default="", 
        help="Directory to save t-SNE plots. If empty, skip plotting."
    )
    parser.add_argument(
        "--tsne_points_per_sentiment", 
        type=int, 
        default=10000, 
        help="Max points per sentiment for t-SNE."
    )
    parser.add_argument(
        "--tsne_total_points", 
        type=int, 
        default=10000, 
        help="Approximate total points sampled for global t-SNE."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed for sampling."
    )
    args = parser.parse_args()
    main(args)
