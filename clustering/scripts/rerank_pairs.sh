#!/bin/bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cell
VERSION=14

ROOT=data/common
DATASET_DIR=${ROOT}/RecommendationDatasets/StatementDatasets/${DATASET_NAME}${VERSION}
OUTPUT_DIR=${DATASET_DIR}/tmp
CROSS_ENCODER=Qwen/Qwen3-Reranker-0.6B

PYTHONPATH=. accelerate launch clustering/rerank_pairs.py \
    --model_name ${CROSS_ENCODER} \
    --statement_path ${DATASET_DIR}/statements_vS.csv \
    --neighbours_path ${OUTPUT_DIR}/ann_results.npz \
    --output_path ${OUTPUT_DIR}/rerank_pairs.csv \
    --threshold 0.90 \
    --batch_size 128 \
    --max_length 8192