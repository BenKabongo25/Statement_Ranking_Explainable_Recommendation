#!/bin/bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common
DATASET_DIR=${ROOT}/RecommendationDatasets/StatementDatasets/${DATASET_NAME}${VERSION}
CLUSTER_DIR=${DATASET_DIR}/clusters

PYTHONPATH=. python clustering/analysis.py \
  --clusters_path  ${CLUSTER_DIR}/clusters.csv  \
  --statement_path ${DATASET_DIR}/statements_vS.csv \
  --embedding_path ${DATASET_DIR}/embeddings_vS.pt \
  --statement2cluster_path ${CLUSTER_DIR}/statement2cluster.json \
  --num_clusters 100 \
  --min_cluster_size 3 \
  --top_k 10 \
  --bottom_k 5 \
  --tsne_output_dir ${DATASET_DIR}/tmp/tsne_plots \
  --tsne_points_per_sentiment 10000 \
  --tsne_total_points 10000 \
  --seed 42
