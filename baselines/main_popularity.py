import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from baselines.popularity import (
    BaseStatementRanker,
    RandomBaseline,
    GlobalPopBaseline,
    UserPopBaseline,
    ItemPopBaseline,
    build_pop_stats_from_train,
    _parse_int_list,
)
from dataset.utils import (
    read_dataset_csv,
    read_statements_csv,
)
from evaluation.metrics import StatementRankingMetrics


def build_id_maps_from_train(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build user and item ID to index mappings from training data."""
    users = train_df["user_id"].astype(str).unique().tolist()
    items = train_df["item_id"].astype(str).unique().tolist()
    user_id2index = {u: i for i, u in enumerate(users)}
    item_id2index = {it: i for i, it in enumerate(items)}
    return user_id2index, item_id2index


def build_user_statements(
    data_df: pd.DataFrame,
    user_id2index: Dict[str, int],
    n_statements: int,
    statement_col: str = "statement_ids",
) -> Dict[int, np.ndarray]:
    """
    Build mapping from user_idx to all statement IDs seen by that user in data.
    
    Returns
    -------
    user_stmts: Dict[int, np.ndarray]
        Map user_idx -> sorted unique statement IDs
    """
    user_stmts: Dict[int, set] = {}
    
    u_col = data_df["user_id"].astype(str).to_numpy()
    s_col = data_df[statement_col].to_numpy()
    
    for u_str, s_raw in zip(u_col, s_col):
        if u_str not in user_id2index:
            continue
        u_idx = user_id2index[u_str]
        
        sids = _parse_int_list(s_raw)
        valid_sids = [sid for sid in sids if 0 <= sid < n_statements]
        
        if u_idx not in user_stmts:
            user_stmts[u_idx] = set()
        user_stmts[u_idx].update(valid_sids)
    
    return {u: np.array(sorted(sids), dtype=np.int32) 
            for u, sids in user_stmts.items()}


def build_item_statements(
    data_df: pd.DataFrame,
    item_id2index: Dict[str, int],
    n_statements: int,
    statement_col: str = "statement_ids",
) -> Dict[int, np.ndarray]:
    """
    Build mapping from item_idx to all statement IDs seen for that item in data.
    
    Returns
    -------
    item_stmts: Dict[int, np.ndarray]
        Map item_idx -> sorted unique statement IDs
    """
    item_stmts: Dict[int, set] = {}
    
    i_col = data_df["item_id"].astype(str).to_numpy()
    s_col = data_df[statement_col].to_numpy()
    
    for i_str, s_raw in zip(i_col, s_col):
        if i_str not in item_id2index:
            continue
        i_idx = item_id2index[i_str]
        
        sids = _parse_int_list(s_raw)
        valid_sids = [sid for sid in sids if 0 <= sid < n_statements]
        
        if i_idx not in item_stmts:
            item_stmts[i_idx] = set()
        item_stmts[i_idx].update(valid_sids)
    
    return {i: np.array(sorted(sids), dtype=np.int32) 
            for i, sids in item_stmts.items()}


def get_test_ground_truth(
    test_df: pd.DataFrame,
    n_statements: int,
    statement_col: str = "statement_ids",
) -> List[np.ndarray]:
    """
    Extract ground truth statement IDs for each test interaction.
    
    Returns
    -------
    gt_list: List[np.ndarray]
        List of ground truth statement arrays, one per test interaction
    """
    gt_list = []
    s_col = test_df[statement_col].to_numpy()
    
    for s_raw in s_col:
        sids = _parse_int_list(s_raw)
        valid_sids = np.array([sid for sid in sids if 0 <= sid < n_statements], 
                              dtype=np.int32)
        gt_list.append(valid_sids)
    
    return gt_list


@torch.no_grad()
def eval_baseline_paradigm(
    baseline: BaseStatementRanker,
    paradigm: str,
    test_df: pd.DataFrame,
    user_ids_test: np.ndarray,
    item_ids_test: np.ndarray,
    gt_list: List[np.ndarray],
    all_stmt_ids: np.ndarray,
    user_stmts: Dict[int, np.ndarray],
    item_stmts: Dict[int, np.ndarray],
    batch_size: int,
    device: torch.device,
    ks: Tuple[int, ...],
    save_everything: bool = False,
) -> Dict[str, float]:
    """
    Evaluate baseline under a specific ranking paradigm.
    
    Parameters
    ----------
    paradigm: str
        One of 'all', 'item'
    """
    metrics = StatementRankingMetrics(ks=ks, save_everything=save_everything)
    metrics.reset()
    
    N = len(gt_list)
    steps = int(math.ceil(N / batch_size))
    
    for step in range(steps):
        s = step * batch_size
        e = min(N, s + batch_size)
        B = e - s
        
        u_batch = user_ids_test[s:e]  # (B,)
        i_batch = item_ids_test[s:e]  # (B,)
        
        if paradigm == "all":
            # ALL: all statements for every interaction
            K = len(all_stmt_ids)
            stmt_ids_batch = np.tile(all_stmt_ids, (B, 1))  # (B, K)
            
        elif paradigm == "user":
            # USER: all statements seen by each user
            max_K = max(len(user_stmts.get(u, [])) for u in u_batch)
            if max_K == 0:
                max_K = 1  # Handle case with no user history
            stmt_ids_batch = np.full((B, max_K), -100, dtype=np.int32)
            
            for b, u in enumerate(u_batch):
                u_stmts = user_stmts.get(u, np.array([], dtype=np.int32))
                if len(u_stmts) > 0:
                    stmt_ids_batch[b, :len(u_stmts)] = u_stmts
                    
        elif paradigm == "item":
            # ITEM: all statements seen for each item
            max_K = max(len(item_stmts.get(i, [])) for i in i_batch)
            if max_K == 0:
                max_K = 1  # Handle case with no item history
            stmt_ids_batch = np.full((B, max_K), -100, dtype=np.int32)
            
            for b, i in enumerate(i_batch):
                i_stmts = item_stmts.get(i, np.array([], dtype=np.int32))
                if len(i_stmts) > 0:
                    stmt_ids_batch[b, :len(i_stmts)] = i_stmts
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")
        
        stmt_ids_t = torch.as_tensor(stmt_ids_batch, dtype=torch.long, device=device)
        u_t = torch.as_tensor(u_batch, dtype=torch.long, device=device)
        i_t = torch.as_tensor(i_batch, dtype=torch.long, device=device)
        
        scores = baseline.rank_scores(u_t, i_t, stmt_ids_t)  # (B, K)
        
        mask_pos = torch.zeros_like(stmt_ids_t, dtype=torch.bool)
        for b in range(B):
            gt = gt_list[s + b]
            if len(gt) == 0:
                continue
            for sid in gt:
                mask = (stmt_ids_t[b] == sid)
                mask_pos[b] |= mask
        
        metrics.update(
            stmt_ids=stmt_ids_t,
            stmt_scores=scores,
            mask_pos=mask_pos,
            user_ids=u_t,
            item_ids=i_t,
        )

    if save_everything:
        return metrics.compute(), metrics.data_df
    return metrics.compute()


def main(args) -> None:
    dataset_dir = Path(args.dataset_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Loading dataset from {dataset_dir}...")
    df = read_dataset_csv(str(dataset_dir / args.dataset_csv))
    stmt_df = read_statements_csv(str(dataset_dir / args.statements_csv))
    
    n_statements = int(stmt_df.shape[0])
    all_stmt_ids = np.arange(n_statements, dtype=np.int32)
    
    train_idx = np.load(str(dataset_dir / args.train_idx)).astype(np.int64)
    test_idx = np.load(str(dataset_dir / args.test_idx)).astype(np.int64)
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"[info] Train: {len(train_df)} interactions, Test: {len(test_df)} interactions")
    print(f"[info] Total statements: {n_statements}")
    
    user_id2index, item_id2index = build_id_maps_from_train(train_df)
    print(f"[info] Users: {len(user_id2index)}, Items: {len(item_id2index)}")
    
    test_users = test_df["user_id"].astype(str).to_numpy()
    test_items = test_df["item_id"].astype(str).to_numpy()
    
    user_ok = np.array([u in user_id2index for u in test_users], dtype=bool)
    item_ok = np.array([i in item_id2index for i in test_items], dtype=bool)
    ok = user_ok & item_ok
    
    if not ok.all():
        print(f"[info] Filtering test set: {ok.sum()}/{len(ok)} interactions have known user+item")
        test_df = test_df.iloc[ok].reset_index(drop=True)
        test_users = test_df["user_id"].astype(str).to_numpy()
        test_items = test_df["item_id"].astype(str).to_numpy()
    
    user_ids_test = np.array([user_id2index[u] for u in test_users], dtype=np.int64)
    item_ids_test = np.array([item_id2index[i] for i in test_items], dtype=np.int64)
    
    print("[info] Extracting ground truth...")
    gt_list = get_test_ground_truth(test_df, n_statements, args.statement_col)
    
    print("[info] Building statistics from train...")
    stats = build_pop_stats_from_train(
        train_df=train_df,
        user_id2index=user_id2index,
        item_id2index=item_id2index,
        n_statements=n_statements,
        statement_col=args.statement_col,
    )
    
    print("[info] Building user/item statement sets...")
    user_stmts = build_user_statements(df, user_id2index, n_statements, args.statement_col)
    item_stmts = build_item_statements(df, item_id2index, n_statements, args.statement_col)
    
    avg_user_stmts = np.mean([len(s) for s in user_stmts.values()]) if user_stmts else 0
    avg_item_stmts = np.mean([len(s) for s in item_stmts.values()]) if item_stmts else 0
    print(f"[info] Avg statements per user: {avg_user_stmts:.1f}")
    print(f"[info] Avg statements per item: {avg_item_stmts:.1f}")
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[info] Using device: {device}")
    
    baselines = {
        "Random": RandomBaseline(seed=args.random_seed),
        "GlobalPop": GlobalPopBaseline(stats),
        "UserPop": UserPopBaseline(stats),
        "ItemPop": ItemPopBaseline(stats),
    }
    
    paradigms = ["all", "user", "item"]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    for baseline_name, baseline in baselines.items():
        print(f"\n[{baseline_name}] Evaluating...")
        results[baseline_name] = {}
        
        for paradigm in paradigms:
            print(f"  Paradigm: {paradigm.upper()}...", end=" ", flush=True)
            
            res0 = eval_baseline_paradigm(
                baseline=baseline,
                paradigm=paradigm,
                test_df=test_df,
                user_ids_test=user_ids_test,
                item_ids_test=item_ids_test,
                gt_list=gt_list,
                all_stmt_ids=all_stmt_ids,
                user_stmts=user_stmts,
                item_stmts=item_stmts,
                batch_size=args.batch_size,
                device=device,
                ks=tuple(args.ks),
                save_everything=args.save_everything,
            )

            if args.save_everything:
                res, data_df = res0
                reverse_user_id2index = {v: k for k, v in user_id2index.items()}
                reverse_item_id2index = {v: k for k, v in item_id2index.items()}
                data_df["user_id"] = data_df["user_id"].map(reverse_user_id2index)
                data_df["item_id"] = data_df["item_id"].map(reverse_item_id2index)
            else:
                res = res0
                data_df = None
            
            results[baseline_name][paradigm] = res
            
            key_metrics = [f"ndcg@{k}" for k in [1, 5, 10] if k in args.ks]
            metric_str = ", ".join([f"{m}: {res.get(m, 0.0):.4f}" for m in key_metrics])
            print(f"✓ [{metric_str}]")
            
            out_path = save_dir / f"{baseline_name}_{paradigm}.metrics.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, sort_keys=True)
            if data_df is not None:
                data_path = save_dir / f"{baseline_name}_{paradigm}.data.csv"
                data_df.to_csv(data_path, index=False)
    
    summary_path = save_dir / "all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    for k in args.ks:
        print(f"\nNDCG@{k}:")
        print(f"{'Baseline':<15} {'ALL':<12} {'USER':<12} {'ITEM':<12}")
        print("-" * 51)
        for baseline_name in baselines.keys():
            all_score = results[baseline_name]["all"].get(f"ndcg@{k}", 0.0)
            user_score = results[baseline_name]["user"].get(f"ndcg@{k}", 0.0)
            item_score = results[baseline_name]["item"].get(f"ndcg@{k}", 0.0)
            print(f"{baseline_name:<15} {all_score:<12.4f} {user_score:<12.4f} {item_score:<12.4f}")
    
    print(f"\n[done] Results saved to {save_dir}")
    print(f"[done] Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate statistical baselines under ALL/USER/ITEM ranking paradigms"
    )
    
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Directory containing dataset CSV and split files")
    parser.add_argument("--dataset_csv", type=str, default="dataset_vC.csv")
    parser.add_argument("--statements_csv", type=str, default="statements_vC.csv")
    parser.add_argument("--statement_col", type=str, default="statement_ids",
                       help="Column name for statement IDs in dataset")
    parser.add_argument("--train_idx", type=str, default="train_idx.npy")
    parser.add_argument("--test_idx", type=str, default="test_idx.npy")
    
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation (lower for ALL paradigm)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                       help="Cutoff values for metrics")
    
    parser.add_argument("--random_seed", type=int, default=0)
    
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--save_everything", action=argparse.BooleanOptionalAction, default=True,
                       help="Whether to save per-interaction data for analysis")
    
    args = parser.parse_args()
    main(args)