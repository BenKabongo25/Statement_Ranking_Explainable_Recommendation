#!/usr/bin/env bash

DATASET_NAME=${1:-Toys}   # Toys, Clothes, Beauty, Sports, Cellphones
VERSION=14

ROOT=/data/common/RecommendationDatasets/StatementDatasets/ # adjust this path to your local setup
DATASET_DIR=${ROOT}${DATASET_NAME}${VERSION}
DATASET_PATH=${DATASET_DIR}/reviews.jsonl # https://jmcauley.ucsd.edu/data/amazon/links.html
OUTPUT_DIR=${DATASET_DIR}

MODEL="Qwen/Qwen3-14B"
PROMPT_TEXT_FILE="extraction/prompts/candidate_extraction.txt"
FORMAT="amz14"

JSON_FORMAT="--json_format"          # use "--no-json_format" to disable
DO_SAMPLE="--do_sample"              # use "--no-do_sample" to disable
SKIP_EXISTING="--no-skip_existing"   # use "--skip_existing" to enable

BATCH_SIZE=64
MAX_NEW_TOKENS=768
TEMPERATURE=0.5
TOP_P=0.95
MAX_LENGTH=128

PYTHONPATH=. accelerate launch extraction/candidate_extraction.py \
  --model "$MODEL" \
  --prompt_text_file "$PROMPT_TEXT_FILE" \
  --dataset_path "$DATASET_PATH" \
  $JSON_FORMAT \
  --format "$FORMAT" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  $DO_SAMPLE \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  $SKIP_EXISTING \
  --max_length "$MAX_LENGTH"
