import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from clustering.utils import load_embeddings, load_pairs, load_statements


@dataclass
class Cluster:
    sentiment: str
    member_indices: List[int]
    representative_index: int
    representative_statement: str
    representative_mean_sim: float
    representative_embedding: torch.Tensor = None
    mean_pairwise_sim: float = float("-inf")


def build_graphs(
    pairs: pd.DataFrame, bi_threshold: float, cross_threshold: float
) -> Dict[str, Dict[int, Set[int]]]:
    """Return per-sentiment adjacency lists for edges passing both thresholds."""
    graphs: Dict[str, Dict[int, Set[int]]] = {}
    mask = (pairs["bi_encoder_sim"] >= bi_threshold) & (pairs["cross_encoder_sim"] >= cross_threshold)
    filtered = pairs[mask]
    for _, row in filtered.iterrows():
        sentiment = row["sentiment"]
        a = int(row["statement_1_index"])
        b = int(row["statement_2_index"])
        if sentiment not in graphs:
            graphs[sentiment] = {}
        adj = graphs[sentiment]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return graphs


def sentiment_to_indices(df: pd.DataFrame) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        sentiment = row["sentiment"]
        mapping.setdefault(sentiment, []).append(int(idx))
    for key in mapping:
        mapping[key].sort()
    return mapping


def connected_components(nodes: Iterable[int], adjacency: Dict[int, Set[int]]) -> List[List[int]]:
    visited: Set[int] = set()
    comps: List[int] = []
    for start in nodes:
        if start in visited:
            continue
        stack = [start]
        comp: List[int] = []
        visited.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adjacency.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def select_representative_small(
    indices: Sequence[int],
    norm_embeddings: torch.Tensor,
    device: torch.device,
) -> Tuple[int, float, float]:
    """For small clusters (< 1000), use full matrix approach."""
    if len(indices) == 1:
        return indices[0], 1.0, 1.0

    emb = norm_embeddings[indices].to(device)
    sim_matrix = emb @ emb.T

    m = sim_matrix.shape[0]
    mask = torch.ones((m, m), dtype=bool, device=sim_matrix.device)
    mask.fill_diagonal_(False)
    means = (sim_matrix * mask).sum(dim=1) / (m - 1)
    rep_local = int(torch.argmax(means).item())
    rep_idx = int(indices[rep_local])
    rep_mean = float(means[rep_local].item())

    tri_indices = torch.triu_indices(m, m, offset=1, device=sim_matrix.device)
    pairwise_vals = sim_matrix[tri_indices[0], tri_indices[1]]
    cluster_mean = float(pairwise_vals.mean().item())
    return rep_idx, rep_mean, cluster_mean


def select_representative_large(
    indices: Sequence[int],
    norm_embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[int, float, float]:
    """For large clusters, compute similarities in batches to avoid OOM."""
    if len(indices) == 1:
        return indices[0], 1.0, 1.0

    m = len(indices)
    emb_cpu = norm_embeddings[indices]  # stay on CPU to avoid GPU blow-up

    # Mean similarity for each point (exclude self).
    means = torch.zeros(m, device="cpu")

    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        batch = emb_cpu[i:end_i].to(device)
        # accumulate sims across all blocks
        accum = torch.zeros(batch.shape[0], device=device)
        for j in range(0, m, batch_size):
            end_j = min(j + batch_size, m)
            block = emb_cpu[j:end_j].to(device)
            sims = batch @ block.T  # (batch_i, batch_j)
            accum += sims.sum(dim=1)
            if i == j:
                # subtract self-similarity for overlapping positions
                diag_len = min(end_i - i, end_j - j)
                accum[:diag_len] -= 1.0
            del block, sims
        means[i:end_i] = (accum / (m - 1)).cpu()
        del batch, accum

    rep_local = int(torch.argmax(means).item())
    rep_idx = int(indices[rep_local])
    rep_mean = float(means[rep_local].item())

    # Estimate cluster mean using sampling for very large clusters
    if m > 2000:
        sample_size = min(200, m)
        sample_indices = torch.randperm(m)[:sample_size]
        sample_emb = emb_cpu[sample_indices].to(device)
        sample_sims = sample_emb @ sample_emb.T
        tri_indices = torch.triu_indices(sample_size, sample_size, offset=1, device=sample_emb.device)
        cluster_mean = float(sample_sims[tri_indices[0], tri_indices[1]].mean().item())
        del sample_emb, sample_sims
    else:
        # Exact mean using block upper-triangle accumulation
        total_sum = 0.0
        count = 0
        for i in range(0, m, batch_size):
            end_i = min(i + batch_size, m)
            batch_i = emb_cpu[i:end_i].to(device)
            # within block upper triangle
            sims_block = batch_i @ batch_i.T
            tri = torch.triu_indices(end_i - i, end_i - i, offset=1, device=device)
            total_sum += sims_block[tri[0], tri[1]].sum().item()
            count += len(tri[0])
            del sims_block, tri

            # cross blocks
            for j in range(end_i, m, batch_size):
                end_j = min(j + batch_size, m)
                batch_j = emb_cpu[j:end_j].to(device)
                sims = batch_i @ batch_j.T
                total_sum += sims.sum().item()
                count += sims.numel()
                del batch_j, sims
            del batch_i
        cluster_mean = total_sum / count if count > 0 else 0.0

    return rep_idx, rep_mean, cluster_mean


def select_representative(
    indices: Sequence[int], 
    norm_embeddings: torch.Tensor, 
    device: torch.device,
    size_threshold: int = 1000
) -> Tuple[int, float, float, torch.Tensor]:
    """Choose strategy based on cluster size."""
    if len(indices) < size_threshold:
        rep_idx, rep_mean, cluster_mean = select_representative_small(indices, norm_embeddings, device=device)
    else:
        rep_idx, rep_mean, cluster_mean = select_representative_large(indices, norm_embeddings, device=device)
    rep_embedding = norm_embeddings[rep_idx].cpu()
    return rep_idx, rep_mean, cluster_mean, rep_embedding


def _is_cluster_cohesive(
    indices: Sequence[int],
    norm_embeddings: torch.Tensor,
    threshold: float,
    device: torch.device,
    block_size: int,
) -> bool:
    """Return True if all pairwise sims in the cluster are >= threshold."""
    if len(indices) <= 1:
        return True
    emb_cpu = norm_embeddings[indices]
    m = emb_cpu.shape[0]
    for i in range(0, m, block_size):
        end_i = min(i + block_size, m)
        batch_i = emb_cpu[i:end_i].to(device)
        for j in range(i, m, block_size):
            end_j = min(j + block_size, m)
            batch_j = emb_cpu[j:end_j].to(device)
            sims = batch_i @ batch_j.T
            if i == j:
                mask = torch.ones_like(sims, dtype=bool, device=device)
                mask.fill_diagonal_(False)
                sims = sims[mask]
            if torch.any(sims < threshold):
                return False
            del batch_j, sims
        del batch_i
    return True


def refine_component(
    indices: Sequence[int],
    norm_embeddings: torch.Tensor,
    intra_component_threshold: float,
    device: torch.device,
    block_size: int,
    component_adjacency: Dict[int, Set[int]],
) -> List[List[int]]:
    """Split a component into denser sub-clusters using pivot expansion and merge checks."""
    if len(indices) <= 1:
        return [list(indices)]

    if _is_cluster_cohesive(indices, norm_embeddings, intra_component_threshold, device, block_size):
        print(f"    Component of size {len(indices)} already cohesive, skipping refinement.")
        return [sorted(indices)]

    idx_list = [int(i) for i in indices]
    pos_lookup = {idx: pos for pos, idx in enumerate(idx_list)}
    emb_cpu = norm_embeddings[idx_list].cpu()
    remaining: Set[int] = set(idx_list)

    def degree_in_remaining(node: int, remaining_nodes: Set[int]) -> int:
        return len(component_adjacency.get(node, set()) & remaining_nodes)

    def best_pivot(rem: Set[int]) -> int:
        return max(rem, key=lambda n: degree_in_remaining(n, rem))

    def neighbors_in_remaining(node: int, rem: Set[int]) -> Set[int]:
        return component_adjacency.get(node, set()) & rem

    def pivot_similarity_to_current(pivot: int, current_pivot_embs: torch.Tensor) -> float:
        pivot_emb = emb_cpu[pos_lookup[pivot]].to(device)
        sims = (current_pivot_embs @ pivot_emb).squeeze(-1)
        val = float(sims.min().item()) if sims.numel() > 0 else 1.0
        del pivot_emb, sims
        return val

    clusters: List[List[int]] = []
    active_cluster_nodes: Set[int] = set()
    active_pivots: List[int] = []
    active_pivot_embs = torch.empty((0, emb_cpu.shape[1]), device=device)
    merge_threshold = 0.90
    step = 0

    while remaining:
        pivot = best_pivot(remaining)
        pivot_neigh = neighbors_in_remaining(pivot, remaining)
        candidate_nodes = set([pivot]) | pivot_neigh
        pivot_emb = emb_cpu[pos_lookup[pivot]].to(device).unsqueeze(0)

        if not active_cluster_nodes:
            active_cluster_nodes = set(candidate_nodes)
            active_pivots = [pivot]
            active_pivot_embs = pivot_emb
            print(
                f"      Step {step}: seed pivot {pivot} degree {len(pivot_neigh)} "
                f"(cluster size {len(active_cluster_nodes)}, remaining {len(remaining)})"
            )
        else:
            min_sim = pivot_similarity_to_current(pivot, active_pivot_embs)
            if min_sim >= merge_threshold:
                active_cluster_nodes |= candidate_nodes
                active_pivots.append(pivot)
                active_pivot_embs = torch.cat([active_pivot_embs, pivot_emb], dim=0)
                print(
                    f"      Step {step}: merged pivot {pivot} (deg {len(pivot_neigh)}), "
                    f"min sim to pivots {min_sim:.3f}, cluster size {len(active_cluster_nodes)}"
                )
            else:
                clusters.append(sorted(active_cluster_nodes))
                print(
                    f"      Step {step}: could not merge pivot {pivot} (min sim {min_sim:.3f}); "
                    f"finalized cluster size {len(clusters[-1])}"
                )
                active_cluster_nodes = set(candidate_nodes)
                active_pivots = [pivot]
                active_pivot_embs = pivot_emb
                print(
                    f"      Step {step}: new cluster seeded with pivot {pivot} "
                    f"(deg {len(pivot_neigh)}), size {len(active_cluster_nodes)}"
                )

        remaining -= candidate_nodes
        step += 1

    if active_cluster_nodes:
        clusters.append(sorted(active_cluster_nodes))
        print(f"      Final cluster size {len(active_cluster_nodes)}")

    return clusters
    

def cluster_graphs(
    graphs: Dict[str, Dict[int, Set[int]]],
    sentiment_indices: Dict[str, List[int]],
    norm_embeddings: torch.Tensor,
    device: torch.device,
    refine_components: bool,
    intra_component_threshold: float,
    block_size: int,
) -> List[Cluster]:
    clusters: List[Cluster] = []
    for sentiment, nodes in sentiment_indices.items():
        adj = graphs.get(sentiment, {})
        all_nodes: Set[int] = set(nodes)
        comps = connected_components(all_nodes, adj)
        
        print(f"Processing {len(comps)} components for sentiment '{sentiment}'")
        for idx, comp in enumerate(comps):
            if len(comp) > 100 and idx % 10 == 0:
                print(f"  Component {idx+1}/{len(comps)}, size={len(comp)}")
            sub_components = (
                refine_component(
                    comp,
                    norm_embeddings,
                    intra_component_threshold,
                    device=device,
                    block_size=block_size,
                    component_adjacency={
                        node: {nbr for nbr in adj.get(node, set()) if nbr in comp}
                        for node in comp
                    },
                )
                if refine_components and len(comp) > 1
                else [comp]
            )
            for sub in sub_components:
                rep_idx, rep_mean, cluster_mean, rep_embedding = select_representative(
                    sub, norm_embeddings, device=device
                )
                clusters.append(
                    Cluster(
                        sentiment=sentiment,
                        member_indices=sub,
                        representative_index=rep_idx,
                        representative_statement="",
                        representative_mean_sim=rep_mean,
                        mean_pairwise_sim=cluster_mean,
                        representative_embedding=rep_embedding,
                    )
                )
    return clusters


def attach_statements(clusters: List[Cluster], df: pd.DataFrame) -> None:
    for c in clusters:
        c.representative_statement = str(df.loc[c.representative_index, "statement"])


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


def main(args: argparse.Namespace) -> None:
    print("Loading data...")
    df_sts = load_statements(args.statement_path)
    embeddings = load_embeddings(args.embedding_path)
    if embeddings.shape[0] != len(df_sts):
        raise ValueError(
            f"Row count mismatch: CSV has {len(df_sts)} rows, embeddings have {embeddings.shape[0]} vectors."
        )
    
    # Keep base embeddings on CPU to avoid OOM; use compute_device for batched sims.
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {compute_device} (embeddings stay on CPU)")
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)

    print("Building graphs...")
    pairs = load_pairs(args.rerank_path)
    graphs = build_graphs(pairs, bi_threshold=args.bi_threshold, cross_threshold=args.cross_threshold)
    sentiment_indices = sentiment_to_indices(df_sts)
    if not graphs:
        print("No edges satisfied the thresholds; clusters will default to singletons per sentiment.")

    print("Clustering...")
    clusters = cluster_graphs(
        graphs,
        sentiment_indices,
        norm_embeddings,
        device=compute_device,
        refine_components=args.refine_components,
        intra_component_threshold=args.intra_component_threshold,
        block_size=args.block_size,
    )
    
    print("Attaching statements...")
    attach_statements(clusters, df_sts)
    
    print("Creating output dataframe...")
    df_out = clusters_to_dataframe(clusters)
    df_out.to_csv(args.output_path, index=False)
    print(f"Saved {len(df_out)} clusters to {args.output_path}")

    print("Saving cluster representative embeddings...")
    cluster_embeddings = cluster_representative_embeddings(clusters)
    torch.save(cluster_embeddings, args.cluster_embedding_path)
    print(f"Saved cluster embeddings to {args.cluster_embedding_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank_path", 
        type=str,
        required=True, 
        help="CSV produced by rerank_pairs.py."
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
        help="Torch tensor file containing embeddings aligned with the CSV rows."
    )
    parser.add_argument(
        "--cluster_embedding_path", 
        type=str,
        required=True, 
        help="Torch tensor file to save cluster representative embeddings."
    )
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True, 
        help="Where to save the clusters CSV."
    )
    parser.add_argument(
        "--bi_threshold", 
        type=float, 
        default=0.90, 
        help="Minimum bi-encoder similarity to create an edge."
    )
    parser.add_argument(
        "--cross_threshold", 
        type=float, 
        default=0.90, 
        help="Minimum cross-encoder similarity to create an edge."
    )
    parser.add_argument(
        "--refine_components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, refine each connected component using intra-component cosine thresholding to form denser sub-clusters.",
    )
    parser.add_argument(
        "--intra_component_threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold used to refine connected components when --refine-components is enabled.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Block size for batched similarity computations (used during refinement).",
    )
    args = parser.parse_args()
    main(args)
