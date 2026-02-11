import argparse
import ast
import json
import os
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple



def parse_sequence(value) -> Optional[List]:
    """Safely parse a list-like cell coming from a CSV."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan"}:
            return None
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            return None
    return None


def parse_int_list(value) -> Optional[List[int]]:
    seq = parse_sequence(value)
    if not seq:
        return None
    out = []
    for x in seq:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out or None


def load_clusters(clusters_path: str) -> Tuple[pd.DataFrame, Dict[int, Dict[str, str]]]:
    df = pd.read_csv(clusters_path)
    required = {"sentiment", "representative_statement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in clusters CSV: {sorted(missing)}")
    df = df.reset_index().rename(columns={"index": "cluster_id"})
    lookup: Dict[int, Dict[str, str]] = {}
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        lookup[cid] = {
            "statement": row["representative_statement"],
            "sentiment": row["sentiment"],
        }
    return df, lookup


def transform_dataset(
    df: pd.DataFrame,
    stmt_to_cluster: Dict[int, int],
    cluster_lookup: Dict[int, Dict[str, str]],
    deduplicate: bool = True,
) -> Tuple[pd.DataFrame, Counter]:
    new_statements: List[Optional[List[Dict[str, str]]]] = []
    new_statement_ids: List[Optional[List[int]]] = []
    new_sentiments: List[Optional[List]] = []
    freq_counter: Counter = Counter()

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    has_sentiments_col = "sentiments" in df.columns

    for _, row in df.iterrows():
        stmt_ids = parse_int_list(row.get("statement_ids"))
        if not stmt_ids:
            new_statements.append(None)
            new_statement_ids.append(None)
            if has_sentiments_col:
                new_sentiments.append(None)
            continue

        seen_clusters = set()
        row_statements: List[Dict[str, str]] = []
        row_clusters: List[int] = []
        row_sentiments: List = []

        for sid in stmt_ids:
            cid = stmt_to_cluster.get(int(sid))
            if cid is None:
                continue

            if deduplicate and cid in seen_clusters:
                continue
            meta = cluster_lookup.get(cid)
            if not meta:
                continue
        
            freq_counter[cid] += 1

            row_clusters.append(cid)
            row_statements.append({"statement": meta["statement"], "sentiment": meta["sentiment"]})
            if has_sentiments_col:
                row_sentiments.append(sentiment_map.get(str(meta["sentiment"]).lower(), meta["sentiment"]))
            seen_clusters.add(cid)

        if not row_statements:
            row_statements = None
            row_clusters = None
            row_sentiments = None if has_sentiments_col else []

        new_statements.append(row_statements)
        new_statement_ids.append(row_clusters)
        if has_sentiments_col:
            new_sentiments.append(row_sentiments if row_statements is not None else None)

    df_out = pd.DataFrame(df)
    df_out["statements"] = new_statements
    df_out["statement_ids"] = new_statement_ids
    if has_sentiments_col:
        df_out["sentiments"] = new_sentiments

    df_out = df_out.dropna(subset=["statements", "statement_ids"])
    return df_out, freq_counter


def build_statements_df(df_clusters: pd.DataFrame, freq_counter: Counter) -> pd.DataFrame:
    records = []
    for _, row in df_clusters.iterrows():
        cid = int(row["cluster_id"])
        records.append(
            {
                "statement": row["representative_statement"],
                "sentiment": row["sentiment"],
                "frequency": int(freq_counter.get(cid, 0)),
            }
        )
    return pd.DataFrame(records)


def main(args: argparse.Namespace) -> None:
    dataset_path = os.path.join(args.dataset_dir, "dataset_vS.csv")
    cluster_dir = args.cluster_dir or os.path.join(args.dataset_dir, "clusters")
    clusters_path = os.path.join(cluster_dir, "clusters.csv")
    stmt2cluster_path = os.path.join(cluster_dir, "statement2cluster.json")

    output_dataset_path = args.output_dataset_path or os.path.join(args.dataset_dir, "dataset_vC.csv")
    output_statements_path = args.output_statements_path or os.path.join(args.dataset_dir, "statements_vC.csv")

    df_clusters, cluster_lookup = load_clusters(clusters_path)
    with open(stmt2cluster_path, "r") as f:
        raw_map = json.load(f)
    stmt_to_cluster = {int(k): int(v) for k, v in raw_map.items()}

    df_dataset = pd.read_csv(dataset_path, index_col=0)
    df_c, freq_counter = transform_dataset(
        df_dataset, stmt_to_cluster, cluster_lookup, deduplicate=True
    )

    df_statements_c = build_statements_df(df_clusters, freq_counter)

    os.makedirs(os.path.dirname(output_dataset_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_statements_path) or ".", exist_ok=True)

    df_c.to_csv(output_dataset_path)
    df_statements_c.to_csv(output_statements_path)

    print(f"Saved clustered dataset to {output_dataset_path} (rows: {len(df_c)})")
    print(f"Saved clustered statements to {output_statements_path} (clusters: {len(df_statements_c)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing dataset_vS.csv.",
    )
    parser.add_argument(
        "--cluster_dir",
        type=str,
        default=None,
        help="Directory containing clusters.csv and statement2cluster.json. Defaults to <dataset_dir>/clusters.",
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default=None,
        help="Output path for dataset_vC.csv. Defaults to <dataset_dir>/dataset_vC.csv.",
    )
    parser.add_argument(
        "--output_statements_path",
        type=str,
        default=None,
        help="Output path for statements_vC.csv. Defaults to <dataset_dir>/statements_vC.csv.",
    )
    args = parser.parse_args()
    main(args)
