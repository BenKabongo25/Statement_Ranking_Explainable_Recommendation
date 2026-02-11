import ast
import numpy as np
import pandas as pd

from typing import List


def read_dataset_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "item_id", "rating", "statement_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset csv: {sorted(missing)}")
    return df


def read_statements_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "statement" not in df.columns:
        raise ValueError("statements csv must have a 'statement' column.")
    return df


def load_embeddings(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        embeds = np.load(path)
    elif path.endswith(".pt"):
        import torch
        embeds = torch.load(path, map_location="cpu").numpy()
    else:
        raise ValueError("Embeddings file must be .npy or .pt")
    return embeds


def parse_int_list(x) -> List[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if not isinstance(x, str) or len(x) == 0:
        return []
    return [int(v) for v in ast.literal_eval(x)]


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)