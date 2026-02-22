# Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation

This repository contains the code and data artifacts for *StaR* (Statement Ranking), a benchmark and pipeline for statement-level explanation ranking in recommender systems.

**Contributions**:
- LLM-based statement extraction with verification to enforce **explanatoriness** and **atomicity**.
- Scalable semantic clustering to enforce **uniqueness** of statements.
- The **StaR** benchmark built from four Amazon Reviews 2014 categories.

## Data

The prepared datasets are available here: [dataset link](https://drive.google.com/drive/folders/1YGoSapnCSo2VxjuII6X78FEG9qhCHpRr?usp=sharing).

Categories:
- `Toys`, `Clothes`, `Beauty`, `Sports`

Files per category:
- `dataset_vS.csv`, `statements_vS.csv`
- `dataset_vC.csv`, `statements_vC.csv`
- Splits: `train_idx.npy`, `eval_idx.npy`, `test_idx.npy`, `split_meta.json`

Naming conventions:
- `vS`: statements **before** clustering
- `vC`: statements **after** clustering

## Dataset Stats (StaR)

Summary after clustering:

| Dataset | Users | Items | Interactions | Unique statements | Triplets |
| --- | --- | --- | --- | --- | --- |
| Toys | 18,594 | 10,130 | 155,992 | 281,664 | 712,963 |
| Clothes | 39,371 | 22,940 | 274,635 | 260,477 | 1,142,256 |
| Beauty | 22,361 | 12,086 | 198,225 | 229,448 | 850,332 |
| Sports | 35,594 | 18,322 | 294,513 | 556,209 | 1,361,389 |

Full pre/post clustering statistics are in the paper.

## Main Results (NDCG@10)

NDCG@10 for global-level and item-level ranking (higher is better). These are the headline results; full P/R/NDCG@{5,10} tables are in the paper.

| Dataset | Setting | Random | UserPop | ItemPop | GlobalPop | BPER+ | ExpGCN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Toys | Global | 0.00439 | 0.07890 | 0.09410 | 0.07172 | 0.07984 | **0.10054** |
| Toys | Item | 0.12951 | **0.26999** | 0.09928 | 0.18156 | 0.18545 | 0.18618 |
| Clothes | Global | 0.00005 | 0.08464 | 0.12000 | 0.14494 | 0.10118 | **0.15976** |
| Clothes | Item | 0.15632 | 0.25325 | 0.12947 | 0.25389 | 0.24685 | **0.25646** |
| Beauty | Global | 0.00392 | 0.04358 | 0.09264 | 0.05478 | 0.06910 | **0.10001** |
| Beauty | Item | 0.11619 | **0.18171** | 0.09731 | 0.16637 | 0.17112 | 0.18159 |
| Sports | Global | 0.00001 | 0.03434 | 0.07921 | 0.06309 | 0.02486 | **0.08639** |
| Sports | Item | 0.11322 | **0.16326** | 0.08252 | 0.15370 | 0.14371 | 0.15978 |

Key takeaway:
- ExpGCN is strongest in **global-level** ranking.
- User history signals (UserPop) are very strong in **item-level** ranking, often matching or surpassing ExpGCN.

## Pipeline Overview

1. **Extraction**: LLM-based candidate extraction + verification. See [`extraction/README.md`](extraction/README.md).
2. **Clustering**: embeddings + ANN + pairwise filtering + graph refinement. See [`clustering/README.md`](clustering/README.md).
3. **Dataset build + splits**: dataset formatting and leave-last-2 splits. See [`dataset/README.md`](dataset/README.md).

## Repository Structure

- `extraction/`: LLM prompts + extraction/verification scripts
- `clustering/`: semantic clustering pipeline and post-process
- `dataset/`: dataset builder and split scripts
- `baselines/`: popularity + SOTA baselines
- `evaluation/`: ranking metrics and evaluation helpers
- `assets/`: figures used in docs/paper

## Citation

If you use this code or data, please cite the paper:

```bibtex
@inproceedings{xxx,
  title     = {Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation},
  author    = {Anonymous Author},
  year      = {2025}
}
```
