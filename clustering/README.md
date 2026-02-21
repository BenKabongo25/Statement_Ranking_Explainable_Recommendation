# Statement Clustering (StaR)


![](../assets/statement_clustering.drawio.svg)
Scalable and semantic-oriented pipeline to consolidate paraphrased statements and enforce **uniqueness** after extraction.

## Overview
- **Statement embeddings**
- (1) **ANN search** (per sentiment)
- (2) **Pairwise filtering** (cross-encoder)
- (3) **Graph refinement**
- **Post-process to final dataset**

## Statement Embeddings

**Script**:
```bash
PYTHONPATH=. python sentence_encode/embed.py \
  --input_path /path/to/statements_vS.csv \
  --sentence_column statement \
  --output_path /path/to/embeddings_vS.pt \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --batch_size 32
```

**Output**:
- `embeddings_vS.pt` aligned with `statements_vS.csv`.

## 1) Approximate Nearest-Neighbor Search

**Script**:
```bash
PYTHONPATH=. python clustering/ann_search.py \
  --statement_path /path/to/statements_vS.csv \
  --embedding_path /path/to/embeddings_vS.pt \
  --output_path /path/to/ann_results.npz \
  --mode full \
  --search_k 128
```

**Input**:
- CSV with `statement` and `sentiment`.
- Embeddings aligned with CSV rows.

**Output**:
- `ann_results.npz` with `indices` and `similarities` (`N x search_k`).

Tip:
- `--mode sample` prints nearest/farthest neighbors for quick inspection.

## 2) Pairwise Filtering (Cross-Encoder)

**Prompt**:
```
Given two short statements, decide if they are perfect paraphrases or synonyms (same meaning). 
Answer "yes" only when they clearly express the same idea, otherwise answer "no".
```

**Script**:
```bash
PYTHONPATH=. accelerate launch clustering/pairwise_filtering.py \
  --model_name Qwen/Qwen3-Reranker-0.6B \
  --statement_path /path/to/statements_vS.csv \
  --neighbours_path /path/to/ann_results.npz \
  --output_path /path/to/rerank_pairs.csv \
  --threshold 0.90 \
  --batch_size 128 \
  --max_length 8192
```

**Output**:
- `rerank_pairs.csv` with `bi_encoder_sim`, `bi_encoder_rank`, `cross_encoder_sim`.

## 3) Refinement (Graph Clustering)

**Script**:
```bash
PYTHONPATH=. python clustering/graph_refinement.py \
  --statement_path /path/to/statements_vS.csv \
  --rerank_path /path/to/rerank_pairs.csv \
  --embedding_path /path/to/embeddings_vS.pt \
  --output_path /path/to/clusters.csv \
  --cluster_embedding_path /path/to/embeddings_vC.pt \
  --bi_threshold 0.90 \
  --cross_threshold 0.90 \
  --refine_components \
  --intra_component_threshold 0.85
```

**Output**:
- `clusters.csv` with `sentiment`, `representative_index`, `representative_statement`, `cluster_size`, `member_indices`, etc.
- `embeddings_vC.pt` (representative embeddings).

## Post-Process

Use clusters to build an aggregated dataset and consolidated statements.

**Script**:
```bash
PYTHONPATH=. python clustering/post_process.py \
  --dataset_dir /path/to/dataset_dir \
  --cluster_dir /path/to/dataset_dir/clusters \
  --output_dataset_path /path/to/dataset_vC.csv \
  --output_statements_path /path/to/statements_vC.csv
```

**Expected inputs**:
- `dataset_vS.csv` in `dataset_dir`.
- `clusters.csv` + `statement2cluster.json` in `cluster_dir`.

**Outputs**:
- `dataset_vC.csv` and `statements_vC.csv`.

## Qualitative Analysis (Optional)

**Script**:
```bash
PYTHONPATH=. python clustering/analysis.py \
  --clusters_path /path/to/clusters.csv \
  --statement_path /path/to/statements_vS.csv \
  --embedding_path /path/to/embeddings_vS.pt \
  --statement2cluster_path /path/to/statement2cluster.json \
  --num_clusters 100 \
  --min_cluster_size 3 \
  --top_k 10 \
  --bottom_k 5 \
  --tsne_output_dir /path/to/tsne_plots \
  --tsne_points_per_sentiment 10000 \
  --tsne_total_points 10000 \
  --seed 42
```