import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union


def _parse_int_list(x) -> List[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str) and x:
        import ast
        return [int(v) for v in ast.literal_eval(x)]
    return []


def _stable_hash64(x: np.ndarray, seed: int) -> np.ndarray:
    z = x.astype(np.uint64, copy=False) + np.uint64(seed)
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


@dataclass(frozen=True)
class PopStats:
    """
    Statment popularity statistics.

    - global_counts: (S,) int32
    - user_counts: dict user_idx -> (sid, count) sorted by sid
    - item_counts: dict item_idx -> (sid, count) sorted by sid
    """
    n_statements: int
    global_counts: np.ndarray
    user_counts: Dict[int, Tuple[np.ndarray, np.ndarray]]
    item_counts: Dict[int, Tuple[np.ndarray, np.ndarray]]


def build_pop_stats_from_train(
    train_df: pd.DataFrame,
    user_id2index: Dict[str, int],
    item_id2index: Dict[str, int],
    n_statements: int,
    statement_col: str = "statement_ids",
) -> PopStats:
    """
    Build global/user/item statement frequency statistics from train data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions dataframe.
    user_id2index : Dict[str, int]
        Mapping from user_id to user index.
    item_id2index : Dict[str, int]
        Mapping from item_id to item index.
    n_statements : int
        Total number of statements S.
    statement_col : str, optional
        Column name in train_df containing statement id lists, by default "statement_ids".

    Returns
    -------
    PopStats
        Train popularity statistics.
    """
    S = int(n_statements)
    global_counts = np.zeros(S, dtype=np.int32)

    # accumulate sparse dicts first (python), then compress to sorted arrays
    user_dict: Dict[int, Dict[int, int]] = {}
    item_dict: Dict[int, Dict[int, int]] = {}

    u_col = train_df["user_id"].astype(str).to_numpy()
    i_col = train_df["item_id"].astype(str).to_numpy()
    s_col = train_df[statement_col].to_numpy()

    for u_str, i_str, s_raw in zip(u_col, i_col, s_col):
        if u_str not in user_id2index or i_str not in item_id2index:
            continue
        u = user_id2index[u_str]
        i = item_id2index[i_str]

        sids = _parse_int_list(s_raw)
        if not sids:
            continue

        # optional: de-dup within interaction to not overcount repeated statements in same review
        sids = list(dict.fromkeys([sid for sid in sids if 0 <= sid < S]))
        if not sids:
            continue

        # global
        global_counts[np.array(sids, dtype=np.int64)] += 1

        # user
        ud = user_dict.get(u)
        if ud is None:
            ud = {}
            user_dict[u] = ud
        for sid in sids:
            ud[sid] = ud.get(sid, 0) + 1

        # item
        idd = item_dict.get(i)
        if idd is None:
            idd = {}
            item_dict[i] = idd
        for sid in sids:
            idd[sid] = idd.get(sid, 0) + 1

    def compress(d: Dict[int, Dict[int, int]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for key, m in d.items():
            if not m:
                continue
            sids = np.fromiter(m.keys(), dtype=np.int32)
            cnts = np.fromiter(m.values(), dtype=np.int32)
            order = np.argsort(sids)
            out[key] = (sids[order], cnts[order])
        return out

    return PopStats(
        n_statements=S,
        global_counts=global_counts,
        user_counts=compress(user_dict),
        item_counts=compress(item_dict),
    )


def _gather_sparse_counts(
    sid_sorted: np.ndarray,
    cnt_sorted: np.ndarray,
    query_sids: np.ndarray,
) -> np.ndarray:
    """
    Given sparse counts stored as (sid_sorted, cnt_sorted),
    return counts for query_sids (vectorized) using searchsorted.
    """
    # positions where each query would be inserted
    pos = np.searchsorted(sid_sorted, query_sids)
    in_bounds = pos < sid_sorted.size
    hit = np.zeros_like(query_sids, dtype=np.int32)
    if sid_sorted.size == 0:
        return hit
    # exact match
    ok = in_bounds & (sid_sorted[pos.clip(max=sid_sorted.size - 1)] == query_sids)
    hit[ok] = cnt_sorted[pos[ok]]
    return hit


class BaseStatementRanker:
    """
    Base class for statement rankers.
    """

    def rank(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return ranked stmt_ids (B,K).
        """
        raise NotImplementedError()

    def rank_scores(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return scores aligned with stmt_ids. Higher = better.
        """
        raise NotImplementedError()


class RandomBaseline(BaseStatementRanker):
    """
    Random statement ranker.

    For each sample (row), returns a random permutation of the candidate stmt_ids.
    Deterministic given `seed` and optionally a per-row `seed_ui`.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = int(seed)

    def rank(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(stmt_ids, torch.Tensor)
        if is_torch:
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            s = np.asarray(stmt_ids)

        B, K = s.shape
        out = np.empty_like(s)

        for b in range(B):
            keys = _stable_hash64(s[b].astype(np.int64), self.seed)
            out[b] = s[b][np.argsort(keys)]

        if is_torch:
            return torch.as_tensor(out, dtype=stmt_ids.dtype, device=device)
        return out
    
    def rank_scores(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return random scores aligned with stmt_ids. Higher = better.
        score = -random rank
        """
        is_torch = isinstance(stmt_ids, torch.Tensor)
        if is_torch:
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            s = np.asarray(stmt_ids)

        B, K = s.shape
        scores = np.empty_like(s, dtype=np.float32)

        for b in range(B):
            keys = _stable_hash64(s[b].astype(np.int64), self.seed)
            order = np.argsort(keys)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(K)
            scores[b] = -ranks.astype(np.float32)

        if is_torch:
            return torch.as_tensor(scores, dtype=torch.float32, device=device)
        return scores.astype(np.float32)


class GlobalPopBaseline(BaseStatementRanker):
    """
    Global popularity statement ranker.

    Ranks candidate stmt_ids by global frequency in train (descending).
    """

    def __init__(self, stats: PopStats) -> None:
        super().__init__()
        self.global_counts = stats.global_counts  # (S,)

    def rank_scores(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],  # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return scores aligned with stmt_ids. Higher = better.
        score = global_count
        """
        is_torch = isinstance(stmt_ids, torch.Tensor)
        if is_torch:
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            s = np.asarray(stmt_ids)

        # handle PAD=-100 by giving score -1
        scores = np.full(s.shape, -1, dtype=np.int32)
        valid = s >= 0
        scores[valid] = self.global_counts[s[valid].astype(np.int64)]

        if is_torch:
            return torch.as_tensor(scores, dtype=torch.float32, device=device)
        return scores.astype(np.float32)

    def rank(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],  # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return ranked stmt_ids (B,K) sorted by global popularity desc, then sid asc.
        """
        is_torch = isinstance(stmt_ids, torch.Tensor)
        if is_torch:
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            s = np.asarray(stmt_ids)

        scores = self.rank_scores(s)  # numpy float32
        scores = np.asarray(scores, dtype=np.float32)

        # lexsort uses last key as primary; we want:
        # primary: -score (desc), secondary: sid (asc)
        out = np.empty_like(s)
        for b in range(s.shape[0]):
            sid = s[b]
            sc = scores[b]
            # PAD to end: set score very low
            sc = np.where(sid >= 0, sc, -1e9)
            order = np.lexsort((sid, -sc))
            out[b] = sid[order]

        if is_torch:
            return torch.as_tensor(out, dtype=stmt_ids.dtype, device=device)
        return out


class UserPopBaseline(BaseStatementRanker):
    """
    User popularity statement ranker.

    For each user u, ranks candidate stmt_ids by frequency in train interactions of that user (descending).
    Unseen statements get score 0.
    """

    def __init__(self, stats: PopStats) -> None:
        super().__init__()
        self.user_counts = stats.user_counts

    def rank_scores(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(stmt_ids, torch.Tensor) or isinstance(user_ids, torch.Tensor)
        if isinstance(stmt_ids, torch.Tensor):
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            device = None
            s = np.asarray(stmt_ids)

        u = user_ids.detach().cpu().numpy() if isinstance(user_ids, torch.Tensor) else np.asarray(user_ids)
        u = u.astype(np.int64, copy=False)

        B, K = s.shape
        scores = np.zeros((B, K), dtype=np.int32)

        for b in range(B):
            sid = s[b].astype(np.int32, copy=False)
            valid = sid >= 0
            if not valid.any():
                continue
            pack = self.user_counts.get(int(u[b]))
            if pack is None:
                continue
            sid_sorted, cnt_sorted = pack
            q = sid[valid]
            scores[b, valid] = _gather_sparse_counts(sid_sorted, cnt_sorted, q)

        if is_torch:
            return torch.as_tensor(scores, dtype=torch.float32, device=device)
        return scores.astype(np.float32)

    def rank(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(stmt_ids, torch.Tensor)
        if is_torch:
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            device = None
            s = np.asarray(stmt_ids)

        scores = np.asarray(self.rank_scores(user_ids, s), dtype=np.float32)
        out = np.empty_like(s)
        for b in range(s.shape[0]):
            sid = s[b]
            sc = scores[b]
            sc = np.where(sid >= 0, sc, -1e9)
            order = np.lexsort((sid, -sc))
            out[b] = sid[order]

        if is_torch:
            return torch.as_tensor(out, dtype=stmt_ids.dtype, device=device)
        return out


class ItemPopBaseline(BaseStatementRanker):
    """
    Item popularity statement ranker.

    For each item i, ranks candidate stmt_ids by frequency in TRAIN interactions of that item (descending).
    Unseen statements get score 0.
    """

    def __init__(self, stats: PopStats) -> None:
        super().__init__()
        self.item_counts = stats.item_counts

    def rank_scores(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(stmt_ids, torch.Tensor) or isinstance(item_ids, torch.Tensor)
        if isinstance(stmt_ids, torch.Tensor):
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            device = None
            s = np.asarray(stmt_ids)

        i = item_ids.detach().cpu().numpy() if isinstance(item_ids, torch.Tensor) else np.asarray(item_ids)
        i = i.astype(np.int64, copy=False)

        B, K = s.shape
        scores = np.zeros((B, K), dtype=np.int32)

        for b in range(B):
            sid = s[b].astype(np.int32, copy=False)
            valid = sid >= 0
            if not valid.any():
                continue
            pack = self.item_counts.get(int(i[b]))
            if pack is None:
                continue
            sid_sorted, cnt_sorted = pack
            q = sid[valid]
            scores[b, valid] = _gather_sparse_counts(sid_sorted, cnt_sorted, q)

        if is_torch:
            return torch.as_tensor(scores, dtype=torch.float32, device=device)
        return scores.astype(np.float32)

    def rank(
        self,
        user_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        item_ids: Union[np.ndarray, torch.Tensor],   # (B,)
        stmt_ids: Union[np.ndarray, torch.Tensor],   # (B,K)
    ) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(stmt_ids, torch.Tensor)
        if is_torch:
            device = stmt_ids.device
            s = stmt_ids.detach().cpu().numpy()
        else:
            device = None
            s = np.asarray(stmt_ids)

        scores = np.asarray(self.rank_scores(item_ids, s), dtype=np.float32)
        out = np.empty_like(s)
        for b in range(s.shape[0]):
            sid = s[b]
            sc = scores[b]
            sc = np.where(sid >= 0, sc, -1e9)
            order = np.lexsort((sid, -sc))
            out[b] = sid[order]

        if is_torch:
            return torch.as_tensor(out, dtype=stmt_ids.dtype, device=device)
        return out
    