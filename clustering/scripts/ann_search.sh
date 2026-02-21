#!/bin/bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common
DATASET_DIR=${ROOT}/RecommendationDatasets/StatementDatasets/${DATASET_NAME}${VERSION}
OUTPUT_DIR=${DATASET_DIR}/tmp

mkdir -p ${OUTPUT_DIR}

PYTHONPATH=. python clustering/ann_search.py \
    --statement_path ${DATASET_DIR}/statements_vS.csv \
    --embedding_path ${DATASET_DIR}/embeddings_vS.pt \
    --output_path ${OUTPUT_DIR}/ann_results.npz \
    --mode full \
    --search_k 128
