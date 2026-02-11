#!/bin/bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common
DATASET_DIR=${ROOT}/RecommendationDatasets/StatementDatasets/${DATASET_NAME}${VERSION}

PYTHONPATH=. python3 statement_benchmark/dataset/build_splits.py \
  --dataset_dir ${DATASET_DIR} \
  --user_col user_id \
  --item_col item_id \
  --time_col timestamp \
  --min_interactions 3