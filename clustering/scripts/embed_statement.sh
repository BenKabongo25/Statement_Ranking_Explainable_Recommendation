#!/bin/bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common
DATASET_DIR=${ROOT}/RecommendationDatasets/StatementDatasets/${DATASET_NAME}${VERSION}
ENCODER=Qwen/Qwen3-Embedding-0.6B

PYTHONPATH=. python sentence_encode/embed.py \
    --input_path ${DATASET_DIR}/statements_vS.csv \
    --sentence_column statement \
    --output_path ${DATASET_DIR}/embeddings_vS.pt \
    --model_name ${ENCODER} \
    --batch_size 32
