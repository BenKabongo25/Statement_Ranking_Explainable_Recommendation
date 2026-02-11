import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings

from sklearn.manifold import TSNE
from typing import List, Tuple


def load_statements(sts_path: str) -> pd.DataFrame:
    df = pd.read_csv(sts_path)
    df = df.reset_index(drop=True)
    if "statement" not in df.columns:
        raise ValueError("Expected a 'statement' column in the CSV.")
    if "sentiment" not in df.columns:
        raise ValueError("Expected a 'sentiment' column in the CSV for sentiment-grouped search.")
    return df


def load_embeddings(embedding_path: str) -> torch.Tensor:
    embeddings = torch.load(embedding_path, map_location="cpu")
    if isinstance(embeddings, dict):
        tensors = [v for v in embeddings.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            raise ValueError("Embedding checkpoint dict does not contain any torch.Tensor.")
        embeddings = tensors[0]
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("Loaded embeddings are not a torch.Tensor.")
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D embeddings (N x D), got shape {tuple(embeddings.shape)}.")
    return embeddings


def load_neighbours(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if "indices" not in data or "similarities" not in data:
        raise ValueError("NPZ must contain 'indices' and 'similarities' arrays.")
    indices = data["indices"]
    similarities = data["similarities"]
    return indices, similarities


def load_pairs(rerank_path: str) -> pd.DataFrame:
    df = pd.read_csv(rerank_path)
    required = {
        "sentiment",
        "statement_1_index",
        "statement_2_index",
        "bi_encoder_sim",
        "cross_encoder_sim",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in rerank CSV: {sorted(missing)}")
    return df


def parse_member_indices(val: str) -> List[int]:
    if isinstance(val, list):
        return [int(x) for x in val]
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
    except Exception as exc:
        raise ValueError(f"Could not parse member_indices value: {val}") from exc
    raise ValueError(f"Unsupported member_indices type: {type(val)}")


def load_clusters(clusters_path: str) -> pd.DataFrame:
    df = pd.read_csv(clusters_path)
    required = {
        "sentiment",
        "representative_index",
        "representative_statement",
        "cluster_size",
        "member_indices",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in clusters CSV: {sorted(missing)}")
    df["member_indices"] = df["member_indices"].apply(parse_member_indices)
    return df


def mlp_config():
    mpl.rcParams['figure.dpi'] = 1200
    mpl.rcParams['grid.linestyle'] = ":"
    mpl.rcParams['grid.linewidth'] = 0.6
    mpl.rcParams['grid.alpha'] = 0.7
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False


def tsne_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    *,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    early_exaggeration: float,
    angle: float,
    metric: str,
    random_state: int
) -> None:
    """
    2D t-SNE scatter plot.
    Applies a safe perplexity when group size is small.
    """
    n = X.shape[0]
    if n < 2:
        return
    try:
        # safe perplexity: must be < n, typical range [5,50]
        max_safe = max(2, (n - 1) // 3)  # heuristic upper bound
        eff_perp = min(perplexity, max(5, max_safe)) if n > 10 else max(2, min(perplexity, n - 1))
        # If still invalid (e.g., n very small), bail out gracefully
        if eff_perp >= n:
            eff_perp = max(2, n - 1)

        tsne = TSNE(
            n_components=2,
            perplexity=eff_perp,
            learning_rate=learning_rate,   # use numeric for broad sklearn compatibility
            n_iter=n_iter, # max_iter in next versions
            init="pca",
            random_state=random_state,
            early_exaggeration=early_exaggeration,
            angle=angle,
            metric=metric,
            verbose=0
        )
        X2 = tsne.fit_transform(X)
        plt.figure(figsize=(6.5, 5.0), dpi=1200)
        plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=1, alpha=0.7, cmap='tab20')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=1200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        warnings.warn(f"t-SNE scatter failed for {title}: {e}")