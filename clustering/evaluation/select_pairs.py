import argparse
import math
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch


BINS: List[Tuple[float, float, float, str]] = [
    (-1.0, 0.50, 0.20, "[-1.0,0.5)"),
    (0.50, 0.60, 0.10, "[0.5,0.6)"),
    (0.60, 0.70, 0.10, "[0.6,0.7)"),
    (0.70, 0.80, 0.10, "[0.7,0.8)"),
    (0.80, 0.90, 0.10, "[0.8,0.9)"),
    (0.90, 1.01, 0.40, "[0.9,1.0]")
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_inputs(statement_path: str, embedding_path: str, device: torch.device):
    df = pd.read_csv(statement_path)
    if "statement" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must contain columns: 'statement' and 'sentiment'.")

    emb = torch.load(embedding_path, map_location="cpu")
    if not isinstance(emb, torch.Tensor):
        raise ValueError("embedding_path must point to a torch Tensor (.pt) aligned with the CSV rows.")
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N,D). Got shape {tuple(emb.shape)}.")

    if len(df) != emb.shape[0]:
        raise ValueError(f"CSV rows ({len(df)}) != embeddings first dim ({emb.shape[0]}).")

    emb = emb.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    return df, emb


def bin_of(sim: float) -> Optional[str]:
    for lo, hi, _, name in BINS:
        if sim >= lo and sim < hi:
            return name
    return None


def allocate_counts(total_pairs: int) -> Dict[str, int]:
    raw = [p * total_pairs for (_, _, p, _) in BINS]
    flo = [int(math.floor(x)) for x in raw]
    remainder = total_pairs - sum(flo)

    frac = [(raw[i] - flo[i], i) for i in range(len(BINS))]
    frac.sort(reverse=True)
    for k in range(remainder):
        flo[frac[k][1]] += 1

    return {BINS[i][3]: flo[i] for i in range(len(BINS))}


def unique_pair_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


@torch.no_grad()
def batch_topk_within_sentiment(
    emb_s: torch.Tensor,
    batch_idx: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute similarities for anchors (batch_idx) vs all emb_s, return topk sims+indices per anchor.
    emb_s is (M,D) on GPU/CPU, normalized.
    """
    anchors = emb_s[batch_idx]  # (B,D)
    sims = anchors @ emb_s.T    # (B,M)

    # remove self matches: set sim to -inf at (row, col=anchor_pos_in_s)
    # anchor_pos_in_s are positions in [0..M-1], i.e., batch_idx itself
    rows = torch.arange(batch_idx.numel(), device=emb_s.device)
    sims[rows, batch_idx] = -1e9

    # topk within sentiment
    k = min(topk, emb_s.shape[0] - 1)
    vals, inds = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
    return vals, inds


@torch.no_grad()
def sample_pairs_for_sentiment(
    df: pd.DataFrame,
    emb: torch.Tensor,
    sent_indices: List[int],
    target_total: int,
    seed: int,
    max_iters: int = 200000,
    anchor_batch: int = 1024,
    topk: int = 64,
) -> List[Dict]:
    """
    Build pairs within one sentiment matching the bin quotas as closely as possible.
    Strategy:
      - For high-sim bins, use nearest-neighbor topk search (fast on GPU).
      - For low-sim bin, use rejection sampling from random pairs.
      - Greedy fill bins; avoid duplicates.
    """
    device = emb.device
    rng = random.Random(seed)

    # subset to sentiment
    s_pos = torch.tensor(sent_indices, device=device, dtype=torch.long)
    emb_s = emb[s_pos]  # (M,D)
    M = emb_s.shape[0]
    if M < 2:
        return []

    # allocate per-bin counts
    targets = allocate_counts(target_total)
    remaining = dict(targets)

    # bookkeeping
    seen_pairs = set()
    out = []

    # Helper to emit a pair (global ids)
    def emit_pair(local_i: int, local_j: int, sim_val: float, bin_name: str):
        gi = int(s_pos[local_i].item())
        gj = int(s_pos[local_j].item())
        key = unique_pair_key(gi, gj)
        if key in seen_pairs:
            return False
        seen_pairs.add(key)
        out.append({
            "sentiment": df.loc[gi, "sentiment"],
            "id1": gi,
            "id2": gj,
            "statement1": df.loc[gi, "statement"],
            "statement2": df.loc[gj, "statement"],
            "similarity": float(sim_val),
            "bin": bin_name,
        })
        remaining[bin_name] -= 1
        return True

    high_bins = [b for b in BINS if b[0] >= 0.60]  # (lo,hi,prop,name)

    # pre-shuffle anchors
    all_locals = list(range(M))
    rng.shuffle(all_locals)

    cur_topk = topk
    anchor_ptr = 0
    rounds = 0
    while any(remaining[b[3]] > 0 for b in high_bins) and rounds < 6:
        rounds += 1
        # process anchors in batches
        while anchor_ptr < M and any(remaining[b[3]] > 0 for b in high_bins):
            batch_local = all_locals[anchor_ptr:anchor_ptr + anchor_batch]
            anchor_ptr += len(batch_local)
            if not batch_local:
                break

            batch_idx = torch.tensor(batch_local, device=device, dtype=torch.long)
            vals, inds = batch_topk_within_sentiment(emb_s, batch_idx, topk=cur_topk)

            # try to assign each anchor to needed bins using its neighbor list
            B = batch_idx.numel()
            K = inds.shape[1]
            vals_cpu = vals.float().cpu()
            inds_cpu = inds.cpu()

            for bi in range(B):
                if not any(remaining[b[3]] > 0 for b in high_bins):
                    break

                local_i = int(batch_idx[bi].item())
                # walk neighbors until we find a needed bin
                for kk in range(K):
                    sim_val = float(vals_cpu[bi, kk].item())
                    local_j = int(inds_cpu[bi, kk].item())

                    # which bin?
                    bname = bin_of(sim_val)
                    if bname is None:
                        continue
                    # only fill bins we consider "high" (>=0.6) here
                    if bname not in remaining or bname == "[-1.0,0.6)":
                        continue
                    if remaining[bname] <= 0:
                        continue

                    if emit_pair(local_i, local_j, sim_val, bname):
                        break

        # If still missing, increase topk and reshuffle anchors to see more candidates
        if any(remaining[b[3]] > 0 for b in high_bins):
            cur_topk = min(cur_topk * 2, max(256, cur_topk))
            anchor_ptr = 0
            rng.shuffle(all_locals)

    # 2) Fill the low bin ([-1.0,0.6)) with rejection sampling from random pairs
    # Also acts as a "catch-all" if high bins are hard to fill (you can relax thresholds later if needed).
    low_bin_name = "[-1.0,0.6)"
    need_low = remaining.get(low_bin_name, 0)

    iters = 0
    while need_low > 0 and iters < max_iters:
        iters += 1
        i = rng.randrange(M)
        j = rng.randrange(M - 1)
        if j >= i:
            j += 1

        gi = int(s_pos[i].item())
        gj = int(s_pos[j].item())
        key = unique_pair_key(gi, gj)
        if key in seen_pairs:
            continue

        # compute similarity quickly
        sim = float((emb_s[i] @ emb_s[j]).float().item())
        if sim < 0.60:  # low bin condition
            if emit_pair(i, j, sim, low_bin_name):
                need_low -= 1

    # 3) If some high bins are still missing, try a more permissive fallback:
    # - sample anchors, compute random candidate sims, and accept if in the desired bin.
    # This is slower but can help if the sentiment slice is small.
    def fallback_fill_bin(lo: float, hi: float, bname: str, attempts: int = 200000):
        nonlocal rng
        need = remaining.get(bname, 0)
        if need <= 0:
            return
        t = 0
        while remaining[bname] > 0 and t < attempts:
            t += 1
            i = rng.randrange(M)
            # sample a few candidates for j
            for _ in range(8):
                j = rng.randrange(M - 1)
                if j >= i:
                    j += 1
                gi = int(s_pos[i].item()); gj = int(s_pos[j].item())
                key = unique_pair_key(gi, gj)
                if key in seen_pairs:
                    continue
                sim = float((emb_s[i] @ emb_s[j]).float().item())
                if sim >= lo and sim < hi:
                    if emit_pair(i, j, sim, bname):
                        break

    for lo, hi, _, bname in BINS:
        if bname == low_bin_name:
            continue
        if remaining.get(bname, 0) > 0:
            fallback_fill_bin(lo, hi, bname)

    return out


def main():
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
        "--output_path",
        type=str,
        required=True,
        help="Where to save the selected pairs CSV."
    )

    parser.add_argument("--num_pairs", type=int, default=500, help="Total number of pairs to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--topk", type=int, default=64, help="Top-k neighbors per anchor for high-sim bins.")
    parser.add_argument("--anchor_batch", type=int, default=1024, help="Anchor batch size for GPU matmul.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device()

    df, emb = load_inputs(args.statement_path, args.embedding_path, device=device)

    sentiments = sorted(df["sentiment"].dropna().unique().tolist())
    if len(sentiments) == 0:
        raise ValueError("No sentiments found in the CSV.")

    base = args.num_pairs // len(sentiments)
    rem = args.num_pairs - base * len(sentiments)
    per_sent_target = {s: base for s in sentiments}
    for i in range(rem):
        per_sent_target[sentiments[i]] += 1

    all_rows = []
    for s in sentiments:
        sent_indices = df.index[df["sentiment"] == s].tolist()
        target = per_sent_target[s]
        rows = sample_pairs_for_sentiment(
            df=df,
            emb=emb,
            sent_indices=sent_indices,
            target_total=target,
            seed=args.seed + hash(s) % 100000,
            anchor_batch=args.anchor_batch,
            topk=args.topk,
        )
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)

    if not out_df.empty:
        out_df = out_df[out_df["sentiment"].isin(sentiments)].reset_index(drop=True)

    out_df.to_csv(args.output_path, index=False)
    print(f"Saved {len(out_df)} pairs to {args.output_path}")

    if len(out_df) > 0:
        print("Achieved bin distribution (overall):")
        print(out_df["bin"].value_counts(normalize=True).sort_index())
        print("\nAchieved sentiment distribution:")
        print(out_df["sentiment"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
