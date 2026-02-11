import math
import numpy as np
import os
import pandas as pd
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple


Axis = Literal["all", "user", "item"]


@dataclass(frozen=True)
class EvalConfig:
    ks: Tuple[int] = (1, 3, 5, 10, 20)
    axes: Tuple[Axis] = ("all", "user", "item")


def _safe_stmt_ids(stmt_ids: torch.LongTensor) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Returns (safe_ids, valid_mask).
    safe_ids has PAD replaced by 0 so gather() is safe.
    """
    valid = stmt_ids >= 0
    safe = torch.where(valid, stmt_ids, torch.zeros_like(stmt_ids))
    return safe, valid


def _scores_from_full_ranking(full_ranking: torch.LongTensor) -> torch.Tensor:
    """
    Convert a full ranking (best->worst) into per-statement scores (higher is better):
      score[id] = -rank_position
    """
    B, S = full_ranking.shape
    device = full_ranking.device
    pos = torch.empty_like(full_ranking)  # (B,S)
    ar = torch.arange(S, device=device, dtype=full_ranking.dtype).unsqueeze(0).expand(B, S)
    pos.scatter_(1, full_ranking, ar)
    return (-pos).float()


def _build_masks_from_offsets(
    offsets_main: torch.LongTensor,  # (B,5)
    K: int,
    valid_mask: torch.BoolTensor,    # (B,K)
) -> Dict[str, torch.BoolTensor]:
    """
    offsets_main = [pos_end, common_end, user_end, item_end, all_end]
    Build:
      mask_pos, mask_common, mask_user_only, mask_item_only
    and axis masks:
      axis_user = POS ∪ COMMON ∪ USER_ONLY
      axis_item = POS ∪ COMMON ∪ ITEM_ONLY
      axis_all  = valid
    """
    B = offsets_main.size(0)
    device = offsets_main.device

    pos_end = offsets_main[:, 0].clamp(0, K)
    common_end = offsets_main[:, 1].clamp(0, K)
    user_end = offsets_main[:, 2].clamp(0, K)
    item_end = offsets_main[:, 3].clamp(0, K)
    all_end = offsets_main[:, 4].clamp(0, K)

    ar = torch.arange(K, device=device).unsqueeze(0).expand(B, K)  # (B,K)

    mask_pos = ar < pos_end.unsqueeze(1)
    mask_common = (ar >= pos_end.unsqueeze(1)) & (ar < common_end.unsqueeze(1))
    mask_user_only = (ar >= common_end.unsqueeze(1)) & (ar < user_end.unsqueeze(1))
    mask_item_only = (ar >= user_end.unsqueeze(1)) & (ar < item_end.unsqueeze(1))
    mask_neg = (ar >= item_end.unsqueeze(1)) & (ar < all_end.unsqueeze(1))

    mask_pos &= valid_mask
    mask_common &= valid_mask
    mask_user_only &= valid_mask
    mask_item_only &= valid_mask
    mask_neg &= valid_mask

    axis_user = mask_pos | mask_common | mask_user_only
    axis_item = mask_pos | mask_common | mask_item_only

    return {
        "mask_pos": mask_pos,
        "mask_common": mask_common,
        "mask_user": axis_user,
        "mask_item": axis_item,
        "mask_neg": mask_neg,
        "valid": valid_mask,
    }


def _masked_topk(scores: torch.Tensor, eligible: torch.BoolTensor, k: int) -> torch.LongTensor:
    masked = scores.masked_fill(~eligible, float("-inf"))
    k_eff = min(k, masked.size(1))
    return torch.topk(masked, k=k_eff, dim=1, largest=True, sorted=True).indices  # (B,k_eff)


def _ndcg(rel_topk: torch.Tensor, total_pos: torch.Tensor) -> torch.Tensor:
    """
    rel_topk: (B,k) binary relevance
    total_pos: (B,) positives available for that axis
    """
    device = rel_topk.device
    B, k = rel_topk.shape

    ranks = torch.arange(1, k + 1, device=device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(ranks + 1.0)  # (k,)
    dcg = (rel_topk.float() * discounts.unsqueeze(0)).sum(dim=1)  # (B,)

    m = torch.minimum(total_pos, torch.tensor(k, device=device, dtype=total_pos.dtype))
    disc_prefix = torch.cumsum(discounts, dim=0)  # (k,)

    idx = (m - 1).clamp_min(0).long()
    idcg = disc_prefix.gather(0, idx)
    idcg = torch.where(m > 0, idcg, torch.zeros_like(idcg))

    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


class StatementRankingMetrics:
    """
    Statement Ranking Evaluation Metrics

    Axes
      - all:   all valid candidates
      - user:  POS ∪ COMMON ∪ USER_ONLY   (default definition from offsets)
      - item:  POS ∪ COMMON ∪ ITEM_ONLY   (default definition from offsets)

    Metrics
      - hitrate@k : 1 if any positive in top-k
      - precision@k : (#pos in top-k) / k
      - recall@k : (#pos in top-k) / (#pos available in axis)
      - ndcg@k : standard NDCG with binary relevance

    Parameters
    ----------
    cfg: EvalConfig
        Configuration for evaluation.
    """

    def __init__(self, cfg: EvalConfig = EvalConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._sum: Dict[str, torch.Tensor] = {}
        self._count: Dict[str, torch.Tensor] = {}

    def _acc(self, key: str, values: torch.Tensor, valid: torch.BoolTensor) -> None:
        # values: (B,)
        v = values[valid]
        s = v.sum()
        c = torch.tensor(v.numel(), device=v.device, dtype=torch.long)
        if key not in self._sum:
            self._sum[key] = s.detach()
            self._count[key] = c.detach()
        else:
            self._sum[key] = self._sum[key] + s.detach()
            self._count[key] = self._count[key] + c.detach()

    def update(
        self,
        stmt_ids: torch.LongTensor,                # (B,K)
        offsets_main: Optional[torch.LongTensor] = None,  # (B,5) if masks not provided

        # optional direct masks (B,K)
        mask_pos: Optional[torch.BoolTensor] = None,
        mask_user: Optional[torch.BoolTensor] = None,
        mask_item: Optional[torch.BoolTensor] = None,

        # one-of model outputs
        cand_scores: Optional[torch.Tensor] = None,       # (B,K)
        full_scores: Optional[torch.Tensor] = None,       # (B,S)
        full_ranking: Optional[torch.LongTensor] = None,  # (B,S)
    ) -> None:
        """
        Update metrics with a new batch.

        Parameters
        ----------
        stmt_ids: torch.LongTensor, (B,K)
            Candidate statement IDs for each batch sample.
        offsets_main: Optional[torch.LongTensor], (B,5)
            Offsets to build masks if masks not provided.
        mask_pos: Optional[torch.BoolTensor], (B,K)
            Mask for positive statements.
        mask_user: Optional[torch.BoolTensor], (B,K)
            Mask for user-related statements.
        mask_item: Optional[torch.BoolTensor], (B,K)
            Mask for item-related statements.
        cand_scores: Optional[torch.Tensor], (B,K)
            Candidate scores directly.
        full_scores: Optional[torch.Tensor], (B,S)
            Full statement scores to gather from.
        full_ranking: Optional[torch.LongTensor], (B,S)
            Full statement ranking to convert to scores.
        """
        modes = [cand_scores is not None, full_scores is not None, full_ranking is not None]
        if sum(modes) != 1:
            raise ValueError("Provide exactly one of cand_scores, full_scores, or full_ranking.")

        safe_ids, valid_mask = _safe_stmt_ids(stmt_ids)
        B, K = safe_ids.shape

        # build candidate scores (B,K)
        if cand_scores is not None:
            if cand_scores.shape != (B, K):
                raise ValueError(f"cand_scores must have shape (B,K)={(B,K)}, got {tuple(cand_scores.shape)}")
            scores = cand_scores
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        else:
            if full_scores is None:
                full_scores = _scores_from_full_ranking(full_ranking)
            # gather
            scores = full_scores.gather(1, safe_ids)
            scores = scores.masked_fill(~valid_mask, float("-inf"))

        # masks: either provided or built from offsets
        if mask_pos is None or mask_user is None or mask_item is None:
            if offsets_main is None:
                raise ValueError("Either provide (mask_pos, mask_user, mask_item) OR offsets_main to build them.")
            m = _build_masks_from_offsets(offsets_main=offsets_main, K=K, valid_mask=valid_mask)
            mask_pos_b = m["mask_pos"]
            mask_user_b = m["mask_user"]
            mask_item_b = m["mask_item"]
        else:
            mask_pos_b = mask_pos.bool() & valid_mask
            mask_user_b = mask_user.bool() & valid_mask
            mask_item_b = mask_item.bool() & valid_mask

        axis_masks: Dict[Axis, torch.BoolTensor] = {
            "all": valid_mask,
            "user": mask_user_b,
            "item": mask_item_b,
        }

        max_k = max(self.cfg.ks)
        for ax in self.cfg.axes:
            eligible = axis_masks[ax]
            y_true = mask_pos_b & eligible  # positives within this axis
            total_pos = y_true.sum(dim=1)   # (B,)
            valid_samples = total_pos > 0

            top_idx = _masked_topk(scores, eligible, k=max_k)  # (B,k_eff)
            rel_top = y_true.gather(1, top_idx)                # (B,k_eff)

            for k in self.cfg.ks:
                k_eff = min(k, rel_top.size(1))
                rel_k = rel_top[:, :k_eff]

                hits = (rel_k.sum(dim=1) > 0).float()
                prec = rel_k.float().sum(dim=1) / float(k)
                rec = rel_k.float().sum(dim=1) / total_pos.clamp_min(1).float()

                ndcg = _ndcg(rel_k, total_pos)

                self._acc(f"{ax}/hitrate@{k}", hits, valid_samples)
                self._acc(f"{ax}/precision@{k}", prec, valid_samples)
                self._acc(f"{ax}/recall@{k}", rec, valid_samples)
                self._acc(f"{ax}/ndcg@{k}", ndcg, valid_samples)

    def compute(self) -> Dict[str, float]:
        """
        Compute averaged metrics.

        Returns
        -------
        out: Dict[str, float]
            Averaged metrics results.
        """
        out: Dict[str, float] = {}
        for k, s in self._sum.items():
            c = int(self._count[k].item()) if k in self._count else 0
            out[k] = float((s / c).item()) if c > 0 else 0.0
        return out
    


@dataclass(frozen=True)
class BenchmarkArtifacts:
    stmt_ids: np.ndarray        # (N, K) int32
    offsets_main: np.ndarray    # (N, 5) int32
    offsets_neg: Optional[np.ndarray] = None  # (N, 4) int32 or None

    @property
    def n_interactions(self) -> int:
        return int(self.stmt_ids.shape[0])

    @property
    def K(self) -> int:
        return int(self.stmt_ids.shape[1])


def load_benchmark_artifacts(
    benchmark_dir: str,
    n_interactions: int,
    K: int,
) -> BenchmarkArtifacts:
    """
    Load benchmark sampling artifacts from a directory.

    Expected files:
      - stmt_ids.int32.mmap          (N, K) int32 (raw memmap)
      - offsets_main.int32.npy       (N, 5) int32
      - offsets_neg.int32.npy        (N, 4) int32 (optional)

    Parameters
    ----------
    benchmark_dir : str
        Path to benchmark directory.

    Returns
    -------
    BenchmarkArtifacts
    """
    benchmark_dir = Path(benchmark_dir)
    stmt_path = benchmark_dir / "stmt_ids.int32.mmap"
    offm_path = benchmark_dir / "offsets_main.int32.npy"
    offn_path = benchmark_dir / "offsets_neg.int32.npy"

    if not stmt_path.exists():
        raise FileNotFoundError(f"Missing {stmt_path}")
    if not offm_path.exists():
        raise FileNotFoundError(f"Missing {offm_path}")

    N = int(n_interactions)
    K = int(K)

    stmt_ids = np.memmap(
        str(stmt_path),
        dtype=np.int32,
        mode="r",
        shape=(N, K),
        order="C",
    )

    offsets_main = np.load(str(offm_path))
    offsets_main = np.asarray(offsets_main, dtype=np.int32)

    if offsets_main.shape != (N, 5):
        raise ValueError(f"offsets_main has shape {offsets_main.shape}, expected {(N, 5)}")

    offsets_neg = None
    if offn_path.exists():
        offsets_neg = np.load(str(offn_path))
        offsets_neg = np.asarray(offsets_neg, dtype=np.int32)
        if offsets_neg.shape != (N, 4):
            raise ValueError(f"offsets_neg has shape {offsets_neg.shape}, expected {(N, 4)}")

    return BenchmarkArtifacts(stmt_ids=stmt_ids, offsets_main=offsets_main, offsets_neg=offsets_neg)


def get_benchmark_split(
    artifacts: BenchmarkArtifacts,
    interaction_idx: np.ndarray,          # (B,) int64/int32 indices into [0..N-1]
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Get a benchmark split by indexing into the artifacts.

    Parameters
    ----------
    artifacts : BenchmarkArtifacts
        Loaded benchmark artifacts.
    interaction_idx : np.ndarray
        Indices of interactions to select.
    device : Optional[torch.device]
        Device to put tensors on.

    Returns
    -------
    Dict[str, torch.Tensor]
        Selected benchmark tensors:
          - stmt_ids: (B, K) long
          - offsets_main: (B, 5) long
          - offsets_neg: (B, 4) long (if available)
    """
    idx = np.asarray(interaction_idx)

    # slice numpy (fast, especially with memmap)
    stmt = artifacts.stmt_ids[idx]           # (B,K) int32
    offm = artifacts.offsets_main[idx]       # (B,5) int32

    out = {
        "stmt_ids": torch.as_tensor(stmt, dtype=torch.long, device=device),
        "offsets_main": torch.as_tensor(offm, dtype=torch.long, device=device),
    }

    if artifacts.offsets_neg is not None:
        offn = artifacts.offsets_neg[idx]    # (B,4)
        out["offsets_neg"] = torch.as_tensor(offn, dtype=torch.long, device=device)

    return out
