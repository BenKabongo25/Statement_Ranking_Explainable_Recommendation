## Clustering pipeline

This folder contains a 4-step pipeline to cluster statements by sentiment using fast FAISS search and a cross-encoder reranker before graph clustering.

### 1) Build nearest neighbours
- Script: `clustering/similarity_search.py` (see `scripts/similarity_search.sh` for an example).
- Inputs: statements CSV with columns `statement,sentiment`, torch embeddings aligned to the CSV rows.
- Outputs: `npz` containing `indices` and `similarities` arrays (shape `N x search_k`) with cosine scores within each sentiment group. Use `mode=sample` for quick inspection or `mode=full` to save neighbours.

### 2) Rerank pairs with a cross-encoder
- Script: `clustering/rerank_pairs.py` (`scripts/rerank_pairs.sh` provides a template).
- Inputs: statements CSV, neighbour `npz`, similarity threshold to keep candidate pairs, cross-encoder model id.
- Outputs: CSV with pairs, bi-encoder similarity, rank, and cross-encoder similarity (`cross_encoder_sim`).

### 3) Graph clustering
- Script: `clustering/graph_clustering.py` (`scripts/graph_clustering.sh`).
- Inputs: statements CSV, embeddings, reranked pairs CSV.
- Process: build per-sentiment graphs using bi/cross similarity thresholds, optionally refine large components, pick representatives by mean cosine similarity.
- Outputs: clusters CSV (`sentiment,representative_index,cluster_size,member_indices,...`) and torch tensor of representative embeddings.

### 4) Qualitative analysis / visualization
- Script: `clustering/analysis.py`.
- Inputs: clusters CSV, statements CSV, embeddings.
- Functions: sample clusters to print nearest/farthest members and optionally generate per-sentiment t-SNE plots of embeddings.

### Quickstart
1. Run neighbour search: `python clustering/similarity_search.py --sts_path data/sts.csv --embedding_path data/emb.pt --mode full --output_path data/neighbours.npz`
2. Rerank: `python clustering/rerank_pairs.py --sts_path data/sts.csv --neighbours_path data/neighbours.npz --output_path data/pairs.csv --threshold 0.9`
3. Cluster: `python clustering/graph_clustering.py --sts_path data/sts.csv --embedding_path data/emb.pt --rerank_path data/pairs.csv --output_path data/clusters.csv --cluster_embedding_path data/cluster_emb.pt`
4. Inspect: `python clustering/analysis.py --clusters_path data/clusters.csv --sts_path data/sts.csv --embedding_path data/emb.pt --num_clusters 20 --top_k 5 --bottom_k 3`
