# Statement Extraction (StaR)

![](../assets/statement_extraction.drawio.svg)

Two-stage LLM pipeline that converts raw reviews into **explanatory**, **atomic** statements.

## Overview
1. **Candidate extraction**: generate statement candidates + sentiment.
2. **Verification**: filter to keep only explanatory and atomic statements.

Key notions used in prompts:
- **Explanatoriness**: product-focused facts that justify user–item interaction.
- **Atomicity**: one fact/opinion per statement.

## Candidate Extraction

**Prompt**:
- File: [`extraction/prompts/candidate_extraction.txt`](../extraction/prompts/candidate_extraction.txt)
- Defines: explainatoriness rules, atomicity rules, output JSON format, few-shot examples.

**Script**:
```bash
PYTHONPATH=. accelerate launch extraction/candidate_extraction.py \
  --model Qwen/Qwen3-14B \
  --prompt_text_file extraction/prompts/candidate_extraction.txt \
  --dataset_path /path/to/reviews.jsonl \
  --format amz14 \
  --output_dir /path/to/output \
  --batch_size 64 \
  --max_new_tokens 768 \
  --temperature 0.5 \
  --top_p 0.95 \
  --max_length 128
```

**Output**:
- `dataset_0.csv`
- `statements`: JSON array of extracted candidates
- original fields preserved

**Apply to your data**:
Default expected keys (Amazon 2014):
- `overall -> rating`
- `unixReviewTime -> timestamp`
- `reviewText -> review`
- `reviewerName -> user_name`
- `reviewerID -> user_id`
- `asin -> item_id`
- `summary -> review_title`

If your schema differs:
- edit the mapping in [`extraction/candidate_extraction.py`](../extraction/candidate_extraction.py) (`AMAZON_2014_ENTRIES`), or
- preprocess to the Amazon-style keys.

## Verification

**Prompt**:
- File: [`extraction/prompts/verification.txt`](../extraction/prompts/verification.txt)
- Filters non-explanatory, non-atomic, redundant statements.

**Script**:
```bash
PYTHONPATH=. accelerate launch extraction/verification.py \
  --model Qwen/Qwen3-8B \
  --prompt_text_file extraction/prompts/verification.txt \
  --dataset_path /path/to/output/dataset_0.csv \ # candidate extraction output
  --output_dir /path/to/output \
  --batch_size 64 \
  --max_new_tokens 512 \
  --temperature 0.1 \
  --top_p 0.95 \
  --statements_column statements \
  --max_statements 128
```

Output:
- `dataset.csv`
- `cleaned`: JSON array of verified statements
- `statements` preserved
