# Dataset Building (StaR)

Utilities to go from statement extraction outputs to **statement-aware** datasets and train/eval/test splits.

## Overview
1. **Build dataset + statements** from the extraction/verification output (after clustering).
2. **Split the dataset** leave-last-2 (before clustering).

## Build Dataset

Takes the output of the extraction + verification pipeline and:
- keeps only well-formed statements,
- normalizes surface forms,
- formats the dataset with statement IDs.

**Script**:
```bash
PYTHONPATH=. python dataset/build_dataset.py \
  --dataset_dir /path/to/dataset_dir
```

**Expected input**:
- `dataset.csv` from extraction/verification (`cleaned` column required).

**Outputs**:
- `dataset_vS.csv`
- `statements_vS.csv`

## Build Splits

Leave-last-2 splits (per user), with warm-start filtering for eval/test.

**Script**:
```bash
PYTHONPATH=. python dataset/build_splits.py \
  --dataset_dir /path/to/dataset_dir \
  --user_col user_id \
  --item_col item_id \
  --time_col timestamp \
  --min_interactions 3
```

**Expected input**:
- `dataset_vC.csv` in `dataset_dir` (after clustering dataset).

**Outputs**:
- `train_idx.npy`, `eval_idx.npy`, `test_idx.npy`: splits indexes
- `train_users.npy`, `train_items.npy`
- `split_meta.json`
