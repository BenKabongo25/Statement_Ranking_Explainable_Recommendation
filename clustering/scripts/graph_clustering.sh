#!/bin/bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common
DATASET_DIR=${ROOT}/RecommendationDatasets/StatementDatasets/${DATASET_NAME}${VERSION}
CLUSTER_DIR=${DATASET_DIR}/clusters

mkdir -p ${CLUSTER_DIR}

PYTHONPATH=. python clustering/graph_clustering.py \
    --statement_path ${DATASET_DIR}/statements_vS.csv \
    --rerank_path ${DATASET_DIR}/tmp/rerank_pairs.csv \
    --embedding_path ${DATASET_DIR}/embeddings_vS.pt \
    --output_path ${CLUSTER_DIR}/clusters.csv \
    --cluster_embedding_path ${DATASET_DIR}/embeddings_vC.pt \
    --bi_threshold 0.90 \
    --cross_threshold 0.90 \
    --refine_components \
    --intra_component_threshold 0.85
    