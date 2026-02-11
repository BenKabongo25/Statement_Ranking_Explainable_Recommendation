import ast
import datetime
import math
import numpy as np
import os
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def get_now_time():
    """a string of current time"""
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def evaluate_exp(test_tuple_list, test_tuple_predict):
    top_k = len(test_tuple_predict[0])
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    precision_sum = 0
    recall_sum = 0  # it is also named hit ratio
    f1_sum = 0
    for x, rank_list in zip(test_tuple_list, test_tuple_predict):
        exps = x[2]
        hits = 0
        for idx, e in enumerate(rank_list):
            if e in exps:
                ndcg += dcgs[idx]
                hits += 1

        pre = hits / top_k
        rec = hits / len(exps)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    ndcg = ndcg / (sum(dcgs) * len(test_tuple_list))
    precision = precision_sum / len(test_tuple_list)
    recall = recall_sum / len(test_tuple_list)
    f1 = f1_sum / len(test_tuple_list)

    return ndcg, precision, recall, f1


def evaluate_item(user2items_test, user2items_top):
    top_k = len(list(user2items_top.values())[0])
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    precision_sum = 0
    recall_sum = 0  # it is also named hit ratio
    f1_sum = 0
    for u, test_items in user2items_test.items():
        rank_list = user2items_top[u]
        hits = 0
        for idx, item in enumerate(rank_list):
            if item in test_items:
                ndcg += dcgs[idx]
                hits += 1

        pre = hits / top_k
        rec = hits / len(test_items)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    ndcg = ndcg / (sum(dcgs) * len(user2items_test))
    precision = precision_sum / len(user2items_test)
    recall = recall_sum / len(user2items_test)
    f1 = f1_sum / len(user2items_test)

    return ndcg, precision, recall, f1


def read_dataset_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "item_id", "rating", "statement_ids", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset csv: {sorted(missing)}")

    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    return df


def read_statements_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "statement" not in df.columns:
        raise ValueError("statements csv must have a 'statement' column.")
    # optionally: sentiment/frequency columns are ignored here
    return df


def parse_int_list(x) -> List[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    if not isinstance(x, str) or len(x) == 0:
        return []
    return [int(v) for v in ast.literal_eval(x)]


def load_split_ids(dataset_dir: str):
    dataset_dir = os.path.abspath(dataset_dir)
    train_idx = np.load(os.path.join(dataset_dir, "train_idx.npy"))
    eval_idx = np.load(os.path.join(dataset_dir, "eval_idx.npy"))
    test_idx = np.load(os.path.join(dataset_dir, "test_idx.npy"))

    train_idx = train_idx.astype(np.int64, copy=False)
    eval_idx = eval_idx.astype(np.int64, copy=False)
    test_idx = test_idx.astype(np.int64, copy=False)

    return train_idx, eval_idx, test_idx


def build_user_item_map(
    data_df: pd.DataFrame,
    idx: np.ndarray,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    data_df = data_df.iloc[idx].copy()

    data_df["user_id"] = data_df["user_id"].astype(str)
    data_df["item_id"] = data_df["item_id"].astype(str)

    user_map = {user: int(idx) for idx, user in enumerate(data_df["user_id"].unique())}
    item_map = {item: int(idx) for idx, item in enumerate(data_df["item_id"].unique())}

    return user_map, item_map


@dataclass
class Data:
    train_tuple_list: List[Tuple[int, int, List[int]]]
    eval_tuple_list: List[Tuple[int, int, List[int]]]
    test_tuple_list: List[Tuple[int, int, List[int]]]

    user2items_test: Dict[int, set]
    user_id2index: Dict[str, int]
    item_id2index: Dict[str, int]
    text_list: List[str]

    train_idx: Optional[np.ndarray] = None
    eval_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None

    n_interactions: Optional[int] = None
    data_df: pd.DataFrame = None
    stmt_df: pd.DataFrame = None


def load_data(
    dataset_dir: str,
) -> Data:
    df = read_dataset_csv(os.path.join(dataset_dir, "dataset_vC.csv"))
    stmt_df = read_statements_csv(os.path.join(dataset_dir, "statements_vC.csv"))

    text_list = stmt_df["statement"].astype(str).tolist()
    print(f'Number of statements: {len(text_list)}')

    train_idx, eval_idx, test_idx = load_split_ids(dataset_dir)
    all_idx = np.concatenate([train_idx, eval_idx, test_idx])
    n_interactions = df.shape[0]
    user_id2index, item_id2index = build_user_item_map(df, all_idx)

    train_df = df.iloc[train_idx].copy()
    eval_df = df.iloc[eval_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(f'Number of training interactions: {train_df.shape[0]}')
    print(f'Number of evaluation interactions: {eval_df.shape[0]}')
    print(f'Number of test interactions: {test_df.shape[0]}')

    print("Samples from training set:")
    print(train_df.head(3))

    def format_split(split_df: pd.DataFrame) -> List[Tuple[int, int, List[int]]]:
        out: List[Tuple[int, int, List[int]]] = []
        for row in split_df.itertuples(index=False):
            u_str = str(row.user_id)
            i_str = str(row.item_id)
            u = user_id2index[u_str]
            i = item_id2index[i_str]

            sids = parse_int_list(row.statement_ids)
            sids = [int(s) for s in sids]

            out.append((u, i, sids))
        return out

    train_tuple_list = format_split(train_df)
    eval_tuple_list = format_split(eval_df)
    test_tuple_list = format_split(test_df)

    user2items_test: Dict[int, set] = {}
    for u, i, _ in test_tuple_list:
        user2items_test.setdefault(u, set()).add(i)

    return Data(
        train_tuple_list=train_tuple_list,
        eval_tuple_list=eval_tuple_list,
        test_tuple_list=test_tuple_list,
        user2items_test=user2items_test,
        user_id2index=user_id2index,
        item_id2index=item_id2index,
        text_list=text_list,
        train_idx=train_idx,
        eval_idx=eval_idx,
        test_idx=test_idx,
        n_interactions=n_interactions,
        data_df=df,
        stmt_df=stmt_df,
    )