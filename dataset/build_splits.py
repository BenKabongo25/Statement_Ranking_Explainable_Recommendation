import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


def _as_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").astype(str)


def build_leave_last_2_split_ids(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    time_col: str,
    min_interactions: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      train_idx, eval_idx, test_idx (int32)
      train_users (str array)
      train_items (str array)

    Warm-start filtering is applied to eval/test (remove cold-start items).
    Only users with >= min_interactions are included in train/eval/test.
    """
    if user_col not in df.columns or item_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Missing one of required columns: {user_col}, {item_col}, {time_col}")

    df = df.copy()
    df = df.reset_index(drop=True)

    df[user_col] = _as_str_series(df[user_col])
    df[item_col] = _as_str_series(df[item_col])

    df = df.dropna(subset=[user_col, item_col, time_col]).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values([user_col, time_col, "_row_id"], ascending=True).reset_index(drop=True)

    train_idx: List[int] = []
    eval_idx: List[int] = []
    test_idx: List[int] = []

    for _, g in df.groupby(user_col, sort=False):
        n = len(g)
        if n < min_interactions:
            continue

        idxs = g.index.to_numpy(dtype=np.int64) 
        train_idx.extend(idxs[:-2].tolist())
        eval_idx.append(int(idxs[-2]))
        test_idx.append(int(idxs[-1]))

    train_idx = np.array(train_idx, dtype=np.int32)
    eval_idx = np.array(eval_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    train_items_set = set(df.loc[train_idx, item_col].astype(str).tolist())

    eval_mask = df.loc[eval_idx, item_col].astype(str).isin(train_items_set).to_numpy()
    test_mask = df.loc[test_idx, item_col].astype(str).isin(train_items_set).to_numpy()
    eval_idx = eval_idx[eval_mask]
    test_idx = test_idx[test_mask]

    train_users = np.array(sorted(set(df.loc[train_idx, user_col].astype(str).tolist())), dtype=str)
    train_items = np.array(sorted(train_items_set), dtype=str)

    return train_idx, eval_idx, test_idx, train_users, train_items


def main(args) -> None:
    dataset_dir = Path(args.dataset_dir)
    df = pd.read_csv(dataset_dir / "dataset_vC.csv")

    train_idx, eval_idx, test_idx, train_users, train_items = build_leave_last_2_split_ids(
        df=df,
        user_col=args.user_col,
        item_col=args.item_col,
        time_col=args.time_col,
        min_interactions=args.min_interactions,
    )

    np.save(dataset_dir / "train_idx.npy", train_idx)
    np.save(dataset_dir / "eval_idx.npy", eval_idx)
    np.save(dataset_dir / "test_idx.npy", test_idx)
    np.save(dataset_dir / "train_users.npy", train_users)
    np.save(dataset_dir / "train_items.npy", train_items)

    meta = {
        "min_interactions": int(args.min_interactions),
        "train": int(train_idx.size),
        "eval": int(eval_idx.size),
        "test": int(test_idx.size),
        "users": int(train_users.size),
        "items": int(train_items.size),
    }
    with open(dataset_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[split] saved to:", args.dataset_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--user_col", type=str, default="user_id")
    parser.add_argument("--item_col", type=str, default="item_id")
    parser.add_argument("--time_col", type=str, default="timestamp")
    parser.add_argument("--min_interactions", type=int, default=3, help="Keep users with >= this count (default=3)")
    args = parser.parse_args()
    main(args)
