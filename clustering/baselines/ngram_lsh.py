import argparse
import os
import pandas as pd
import random
import torch

from dataclasses import dataclass
from datasketch import MinHash, LeanMinHash, MinHashLSH
from typing import List, Sequence
from tqdm import tqdm


def get_k_shingles(raw_text: str, k: int = 1) -> set:
    text_lower = raw_text.lower()
    words = text_lower.split()
    if k <= 1:
        return set(words)

    shingles = []
    for start in range(len(words) + 1 - k):
        shingles.append(" ".join(words[start : start + k]))
    return set(shingles)


def l2_normalize(E: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return E / (E.norm(dim=1, keepdim=True).clamp_min(eps))


def mean_pairwise_cosine(
    X: torch.Tensor,
    max_pairs: int = -1,
    seed: int = 42,
) -> float:
    m = X.shape[0]
    if m == 0:
        return float("nan")
    if m == 1:
        return 1.0

    if max_pairs is None or max_pairs <= 0:
        # exact: mean of all entries of X @ X^T
        sim = (X @ X.t()).mean().item()
        return float(sim)

    rng = random.Random(seed)
    # sample pairs with replacement
    idx_i = torch.tensor([rng.randrange(m) for _ in range(max_pairs)], device=X.device)
    idx_j = torch.tensor([rng.randrange(m) for _ in range(max_pairs)], device=X.device)
    sim = (X[idx_i] * X[idx_j]).sum(dim=1).mean().item()
    return float(sim)


@dataclass
class Cluster:
    sentiment: str
    representative_index: int                 # global index (row id in input CSV)
    representative_statement: str
    member_indices: List[int]                 # global indices
    representative_embedding: torch.Tensor    # (D,)
    representative_mean_sim: float
    mean_pairwise_sim: float


def clusters_to_dataframe(clusters: Sequence[Cluster]) -> pd.DataFrame:
    records = []
    for c in clusters:
        records.append(
            {
                "sentiment": c.sentiment,
                "representative_index": c.representative_index,
                "representative_statement": c.representative_statement,
                "cluster_size": len(c.member_indices),
                "member_indices": c.member_indices,
                "representative_mean_sim": c.representative_mean_sim,
                "cluster_mean_pairwise_sim": c.mean_pairwise_sim,
            }
        )
    return pd.DataFrame.from_records(records)


def cluster_representative_embeddings(clusters: Sequence[Cluster]) -> torch.Tensor:
    embeddings = [c.representative_embedding for c in clusters]
    return torch.stack(embeddings, dim=0)


def build_minhashes(sentences: List[str], shingle_size: int, num_perm: int) -> List[LeanMinHash]:
    minhashes: List[LeanMinHash] = []
    for sentence in tqdm(sentences, desc="Creating MinHash", total=len(sentences)):
        shingle_set = get_k_shingles(sentence, shingle_size)
        mh = MinHash(num_perm=num_perm)
        for s in shingle_set:
            mh.update(s.encode("utf8"))
        minhashes.append(LeanMinHash(mh))
    return minhashes


def lsh_groups(
    minhashes: List[LeanMinHash],
    sim_threshold: float,
) -> List[List[int]]:
    lsh = MinHashLSH(threshold=sim_threshold)
    for idx, mh in enumerate(minhashes):
        lsh.insert(str(idx), mh)

    groups: List[List[int]] = []
    visited = set()

    for idx, mh in tqdm(list(enumerate(minhashes)), desc="Grouping sentences", total=len(minhashes)):
        if idx in visited:
            continue
        ids_str = lsh.query(mh)  # local ids as strings
        for s in ids_str:
            lsh.remove(s)  # speed-up
        ids_int = [int(s) for s in ids_str]
        for j in ids_int:
            visited.add(j)
        groups.append(ids_int)

    return groups


def main(args):
    print("Loading statements CSV...")
    df = pd.read_csv(args.statement_path).reset_index(drop=True)
    for col in ["statement", "sentiment"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.statement_path}. Found: {list(df.columns)}")

    print("Loading embeddings...")
    E = torch.load(args.embedding_path, map_location="cpu")
    if not isinstance(E, torch.Tensor) or E.dim() != 2:
        raise ValueError(f"Embeddings must be a torch.Tensor of shape (N,D). Got: {type(E)} / {getattr(E,'shape',None)}")
    if E.shape[0] != len(df):
        raise ValueError(f"Embeddings first dim ({E.shape[0]}) must match CSV rows ({len(df)}).")

    # ensure normalized for cosine=dot
    E = l2_normalize(E)
    E_dev = E.to(args.device)

    clusters: List[Cluster] = []
    rng = random.Random(args.seed)

    print("Clustering per sentiment...")
    for sentiment, sdf in df.groupby("sentiment", sort=False):
        local_sentences = sdf["statement"].tolist()
        global_idx = sdf.index.tolist()  # global row ids

        if len(local_sentences) == 0:
            continue

        # build minhashes + LSH groups in LOCAL space
        minhashes = build_minhashes(local_sentences, shingle_size=args.shingle_size, num_perm=args.num_perm)
        groups_local = lsh_groups(minhashes, sim_threshold=args.sim_threshold)

        # convert to global indices + filter by group_size rule
        for g in groups_local:
            members_global = [global_idx[i] for i in g]
            if len(members_global) < args.group_size:
                continue

            rep_global = members_global[0]
            rep_statement = df.loc[rep_global, "statement"]
            rep_emb = E_dev[rep_global]  # (D,)

            # compute representative_mean_sim over members (including itself)
            member_embs = E_dev[members_global]                 # (m, D)
            rep_mean_sim = float((member_embs @ rep_emb).mean().item())

            # compute mean pairwise similarity (exact or sampled)
            pair_mean_sim = mean_pairwise_cosine(
                member_embs,
                max_pairs=args.pairwise_max_pairs,
                seed=rng.randrange(10**9),
            )

            clusters.append(
                Cluster(
                    sentiment=str(sentiment),
                    representative_index=int(rep_global),
                    representative_statement=str(rep_statement),
                    member_indices=[int(x) for x in members_global],
                    representative_embedding=rep_emb.detach().cpu(),
                    representative_mean_sim=rep_mean_sim,
                    mean_pairwise_sim=pair_mean_sim,
                )
            )

        print(f"  sentiment={sentiment} -> kept {sum(1 for c in clusters if c.sentiment==sentiment)} clusters so far")

    print("Creating output dataframe...")
    df_out = clusters_to_dataframe(clusters)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    df_out.to_csv(args.output_path, index=False)
    print(f"Saved {len(df_out)} clusters to {args.output_path}")

    print("Saving cluster representative embeddings...")
    cluster_embeddings = cluster_representative_embeddings(clusters)  # (K, D)
    os.makedirs(os.path.dirname(args.cluster_embedding_path) or ".", exist_ok=True)
    torch.save(cluster_embeddings, args.cluster_embedding_path)
    print(f"Saved cluster embeddings to {args.cluster_embedding_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--statement_path",
        type=str,
        required=True,
        help="CSV with columns 'statement' and 'sentiment'.",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Torch tensor file containing embeddings aligned with the CSV rows.",
    )
    parser.add_argument(
        "--cluster_embedding_path",
        type=str,
        required=True,
        help="Torch tensor file to save cluster representative embeddings.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save the clusters CSV.",
    )

    parser.add_argument("--sim_threshold", type=float, default=0.9)
    parser.add_argument("--shingle_size", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=1, help="Keep groups with size >= group_size (EXTRA behavior).")
    parser.add_argument("--num_perm", type=int, default=128, help="MinHash permutations (tradeoff quality/speed/memory).")

    parser.add_argument(
        "--pairwise_max_pairs",
        type=int,
        default=-1,
        help="If >0, approximate cluster_mean_pairwise_sim by sampling this many random pairs per cluster.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)