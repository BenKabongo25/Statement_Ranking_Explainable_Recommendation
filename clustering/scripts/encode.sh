#!/bin/bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common
DATASET_DIR=${ROOT}/RecommendationDatasets/StatementDatasets/${DATASET_NAME}${VERSION}
#ENCODER=Qwen/Qwen3-Embedding-0.6B
ENCODER=tencent/KaLM-Embedding-Gemma3-12B-2511

PYTHONPATH=. python clustering/encode.py \
    --input_path ${DATASET_DIR}/statements_vS.csv \
    --sentence_column statement \
    --output_path ${DATASET_DIR}/eval_embeddings.pt \
    --model_name ${ENCODER} \
    --batch_size 1024
