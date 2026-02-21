#!/usr/bin/env bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common/RecommendationDatasets/StatementDatasets/ # adjust this path to your local setup
DATASET_DIR=${ROOT}${DATASET_NAME}${VERSION}
DATASET_PATH=${DATASET_DIR}/dataset_0.csv       # output of candidate extraction stage, adjust if needed
OUTPUT_DIR=${DATASET_DIR}

MODEL="Qwen/Qwen3-8B"
PROMPT_TEXT_FILE="extraction/prompts/verification.txt"

DO_SAMPLE="--do_sample"              # use "--no-do_sample" to disable
SKIP_EXISTING="--skip_existing"       # use "--no-skip_existing" to disable

BATCH_SIZE=64
MAX_NEW_TOKENS=512
TEMPERATURE=0.1
TOP_P=0.95
MAX_STATEMENTS=128
STATEMENTS_COLUMN="statements"

PYTHONPATH=. accelerate launch extraction/verification.py \
  --model "$MODEL" \
  --prompt_text_file "$PROMPT_TEXT_FILE" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  $DO_SAMPLE \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  $SKIP_EXISTING \
  --statements_column "$STATEMENTS_COLUMN" \
  --max_statements "$MAX_STATEMENTS"
