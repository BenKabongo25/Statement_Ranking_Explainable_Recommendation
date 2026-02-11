import pandas as pd
import torch
from typing import Dict, Tuple


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

    Metrics
      - hitrate@k : 1 if any positive in top-k
      - precision@k : (#pos in top-k) / k
      - recall@k : (#pos in top-k) / (#pos)
      - ndcg@k : standard NDCG with binary relevance

    Parameters
    ----------
    ks: Tuple[int]
        Cutoff values for evaluation metrics.
    """

    def __init__(self, ks: Tuple[int] = (1, 3, 5, 10, 20), save_everything: bool = False) -> None:
        super().__init__()
        self.ks = ks
        self.save_everything = save_everything
        self.reset()

    def reset(self) -> None:
        self._sum: Dict[str, torch.Tensor] = {}
        self._count: Dict[str, torch.Tensor] = {}

        self.data_df = None
        if self.save_everything:
            data = {
                "user_id": [],
                "item_id": [],
                "top_10": [],
            }
            for k in self.ks:
                data[f"hitrate@{k}"] = []
                data[f"precision@{k}"] = []
                data[f"recall@{k}"] = []
                data[f"ndcg@{k}"] = []
            self.data_df = pd.DataFrame(data)

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
        stmt_ids: torch.LongTensor,             # (B,K)
        stmt_scores: torch.FloatTensor,         # (B,K)
        mask_pos: torch.BoolTensor,             # (B,K)
        user_ids: torch.LongTensor = None,      # (B,)
        item_ids: torch.LongTensor = None,      # (B,)
    ) -> None:
        """
        Update metrics with a new batch.

        Parameters
        ----------
        stmt_ids: torch.LongTensor
            Statement ids, -100 for padding. Shape (B,K)
        stmt_scores: torch.FloatTensor
            Statement scores. Shape (B,K)
        mask_pos: torch.BoolTensor
            Mask for positive statements. Shape (B,K)
        user_ids: torch.LongTensor, optional
            User ids for each sample. Shape (B,)
        item_ids: torch.LongTensor, optional
            Item ids for each sample. Shape (B,)
        """
        valid_mask = stmt_ids >= 0
        safe_ids = torch.where(valid_mask, stmt_ids, torch.zeros_like(stmt_ids))
        B, K = safe_ids.shape

        max_k = max(self.ks)
        total_pos = mask_pos.sum(dim=1)   # (B,)
        valid_samples = total_pos > 0

        scores = stmt_scores.masked_fill(~valid_mask, float("-inf"))
        top_idx = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True).indices  # (B,max_k)
        rel_top = mask_pos.gather(1, top_idx) # (B,max_k)

        rows = None
        if self.save_everything:
            rows = []
            for i in range(B):
                if not valid_samples[i]:
                    continue

                row = {
                    "user_id": user_ids[i].item() if user_ids is not None else -1,
                    "item_id": item_ids[i].item() if item_ids is not None else -1,
                    "top_10": stmt_ids[i][top_idx[i]].tolist()[:10],
                }
                rows.append(row)

        for k in self.ks:
            rel_k = rel_top[:, :k]
            num_pos = rel_k.float().sum(dim=1) 

            hits = (num_pos > 0).float()
            prec = num_pos / float(k)
            rec = num_pos / total_pos.clamp_min(1).float()
            ndcg = _ndcg(rel_k, total_pos)

            self._acc(f"hitrate@{k}", hits, valid_samples)
            self._acc(f"precision@{k}", prec, valid_samples)
            self._acc(f"recall@{k}", rec, valid_samples)
            self._acc(f"ndcg@{k}", ndcg, valid_samples)

            if self.save_everything:
                for i in range(B):
                    if not valid_samples[i]:
                        continue
                    rows[i][f"hitrate@{k}"] = hits[i].item()
                    rows[i][f"precision@{k}"] = prec[i].item()
                    rows[i][f"recall@{k}"] = rec[i].item()
                    rows[i][f"ndcg@{k}"] = ndcg[i].item()

        if self.save_everything:
            batch_df = pd.DataFrame(rows)
            self.data_df = pd.concat([self.data_df, batch_df], ignore_index=True)

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