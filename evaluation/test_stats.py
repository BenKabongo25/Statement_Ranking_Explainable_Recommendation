import argparse
import json
import math
import numpy as np
import os
import pandas as pd

from dataclasses import dataclass
from scipy import stats
from typing import List, Tuple


def cohen_d_paired(diff: np.ndarray) -> float:
    # Cohen's d for paired samples: mean(diff) / std(diff, ddof=1)
    diff = diff.astype(float)
    if diff.size < 2:
        return float("nan")
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(diff) / sd)


def cliffs_delta_paired(diff: np.ndarray) -> float:
    """
    For paired setup, Cliff's delta on differences vs 0:
    delta = P(diff > 0) - P(diff < 0)
    Range [-1,1]
    """
    if diff.size == 0:
        return float("nan")
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    n = diff.size
    return float((gt - lt) / n)


def significance_label(p: float, thresholds_desc: List[float]) -> str:
    """
    thresholds_desc: e.g. [0.05, 0.01, 0.001] (descending)
    Returns a compact label like:
      p<0.001, p<0.01, p<0.05, ns
    """
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return "na"
    for t in sorted(thresholds_desc):  # smallest first
        if p < t:
            return f"p<{t:g}"
    return "ns"


@dataclass
class TestResult:
    metric_at_k: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    median_diff: float
    cohen_d: float
    cliffs_delta: float
    ttest_p: float
    wilcoxon_p: float


def compute_tests(
    a: pd.Series,
    b: pd.Series,
    run_ttest: bool,
    run_wilcoxon: bool,
    min_pairs: int = 5,
) -> Tuple[int, float, float, float, float, float, float, float, float]:
    """
    Returns:
    n, mean_a, mean_b, mean_diff, median_diff, d, cliffs, t_p, w_p
    """
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.shape[0] < min_pairs:
        return (df.shape[0], float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"), float("nan"))

    aa = df["a"].astype(float).to_numpy()
    bb = df["b"].astype(float).to_numpy()
    diff = bb - aa  # positive means method B better

    mean_a = float(np.mean(aa))
    mean_b = float(np.mean(bb))
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    d = cohen_d_paired(diff)
    cd = cliffs_delta_paired(diff)

    t_p = float("nan")
    w_p = float("nan")

    if run_ttest:
        try:
            t_res = stats.ttest_rel(bb, aa, nan_policy="omit")
            t_p = float(t_res.pvalue)
        except Exception:
            t_p = float("nan")

    if run_wilcoxon:
        try:
            if np.allclose(diff, 0):
                w_p = 1.0
            else:
                w_res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                w_p = float(w_res.pvalue)
        except Exception:
            w_p = float("nan")

    return df.shape[0], mean_a, mean_b, mean_diff, median_diff, d, cd, t_p, w_p



def main(args):
    method_a = args.method_a
    method_b = args.method_b

    metrics = args.metrics
    ks = args.ks
    thresholds = args.p_thresholds
    thresholds = sorted(thresholds, reverse=True)

    run_ttest = args.tests in ("ttest", "both")
    run_wilcoxon = args.tests in ("wilcoxon", "both")

    out_subdir = f"{method_a}_vs_{method_b}"
    out_path = os.path.join(args.out_dir, out_subdir)
    os.makedirs(out_path, exist_ok=True)

    df_a = pd.read_csv(args.a_csv)
    df_b = pd.read_csv(args.b_csv)

    metric_cols = []
    for m in metrics:
        for k in ks:
            col = f"{m}@{k}"
            metric_cols.append(col)

    missing_a = [c for c in metric_cols if c not in df_a.columns]
    missing_b = [c for c in metric_cols if c not in df_b.columns]
    if missing_a or missing_b:
        msg = []
        if missing_a:
            msg.append(f"Missing in A: {missing_a[:10]}{'...' if len(missing_a)>10 else ''}")
        if missing_b:
            msg.append(f"Missing in B: {missing_b[:10]}{'...' if len(missing_b)>10 else ''}")
        raise ValueError("Some requested metric columns are missing.\n" + "\n".join(msg))

    results: List[TestResult] = []

    for col in metric_cols:
        n, mean_a, mean_b, mean_diff, median_diff, d, cd, t_p, w_p = compute_tests(
            df_a[col], df_b[col], run_ttest, run_wilcoxon, min_pairs=args.min_pairs
        )
        results.append(
            TestResult(
                metric_at_k=col,
                n=n,
                mean_a=mean_a,
                mean_b=mean_b,
                mean_diff=mean_diff,
                median_diff=median_diff,
                cohen_d=d,
                cliffs_delta=cd,
                ttest_p=t_p,
                wilcoxon_p=w_p,
            )
        )

    res_df = pd.DataFrame([r.__dict__ for r in results])

    if run_ttest:
        res_df["ttest_sig"] = [significance_label(x, thresholds) for x in res_df["ttest_p"].to_numpy()]
    if run_wilcoxon:
        res_df["wilcoxon_sig"] = [significance_label(x, thresholds) for x in res_df["wilcoxon_p"].to_numpy()]

    res_df["direction"] = np.where(res_df["mean_diff"] > 0, f"{method_b} better",
                            np.where(res_df["mean_diff"] < 0, f"{method_a} better", "tie"))

    summary_csv = os.path.join(out_path, "stats_summary.csv")
    res_df.to_csv(summary_csv, index=False)

    metadata = {
        "method_a": method_a,
        "method_b": method_b,
        "path_a": args.a_csv,
        "path_b": args.b_csv,
        "metrics": metrics,
        "ks": ks,
        "tests": args.tests,
        "p_thresholds": thresholds,
        "min_pairs": args.min_pairs,
    }
    with open(os.path.join(out_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    preview_cols = [
        "metric_at_k", "n", 
        "mean_a", "mean_b", "mean_diff", 
        "cohen_d", "cliffs_delta", "direction",
        "ttest_p", "ttest_sig",
        "wilcoxon_p", "wilcoxon_sig"
    ]

    print(f"\nSaved results to: {out_path}")
    print(res_df[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_csv", required=True, help="Path to CSV of method A (baseline).")
    parser.add_argument("--b_csv", required=True, help="Path to CSV of method B (candidate).")
    parser.add_argument("--method_a", required=True, help="Name for method A.")
    parser.add_argument("--method_b", required=True, help="Name for method B.")
    parser.add_argument("--out_dir", required=True, help="Base output directory.")
    parser.add_argument("--metrics", type=str, nargs="*",default=["precision", "recall", "ndcg"],
                    help="Metric base names (without @k).")
    parser.add_argument("--ks", type=int, nargs="*", default=[5, 10], 
                    help="k values.")
    parser.add_argument("--tests", choices=["ttest", "wilcoxon", "both"], default="both",
                    help="Which paired tests to run.")
    parser.add_argument("--p_thresholds", type=float, nargs="*", default=[0.05, 0.01, 0.001],
                    help="Significance thresholds (e.g. 0.05, 0.01, 0.001).")
    parser.add_argument("--min_pairs", type=int, default=5, help="Minimum non-NaN paired samples required to test.")

    args = parser.parse_args()
    main(args)
