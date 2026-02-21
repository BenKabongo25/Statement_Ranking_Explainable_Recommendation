import argparse
import datasets
import json
import logging
import os
import pandas as pd
import re
import time
import torch
import ast

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_message_from_statements(example, prompt, statements_column: str, max_items: int = 50):
    raw = example.get(statements_column, "")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raw = ""

    parsed = None
    if isinstance(raw, (dict, list)):
        parsed = raw
    elif isinstance(raw, str) and raw.strip():
        s = raw.strip()

        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None

    if isinstance(parsed, list):
        parsed = {"statements": parsed}
    elif isinstance(parsed, dict):
        if "statements" not in parsed and "statement" in parsed:
            parsed = {"statements": [parsed]}
    else:
        parsed = {"statements": []}

    if isinstance(parsed.get("statements"), list) and max_items is not None:
        parsed["statements"] = parsed["statements"][:max_items]

    example["messages"] = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Input JSON:\n" + json.dumps(parsed, ensure_ascii=False) + "\nOutput:"
        },
    ]
    return example


def extract_json_object_from_output(text: str):
    m = re.search(r'\{\s*"statements"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    m2 = re.search(r"\{.*\}", text, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if isinstance(obj, dict) and "statements" in obj:
                return obj
        except json.JSONDecodeError:
            return None
    return None


def empty_cache():
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()


def messages_to_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main(config):
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world = accelerator.num_processes
    logging.info(f"[rank {rank}/{world}] device={device}")

    os.makedirs(config.output_dir, exist_ok=True)

    data_set = os.path.basename(config.dataset_path).rsplit(".", 1)[0]
    print(data_set) 
    path = os.path.join(config.output_dir, f"clean_statements_{data_set}_{rank}.csv")

    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    dataset = datasets.load_dataset(
        "csv",
        data_files={"train": config.dataset_path},
        split="train",
        streaming=True,
    )

    dataset = dataset.filter(lambda ex, idx: (idx % world) == rank, with_indices=True)

    if os.path.exists(path) and config.skip_existing:
        df = pd.read_csv(path)
        length = len(df)
        logging.info(f"[rank {rank}] Skipping {length} already-done rows from {path}")
        dataset = dataset.skip(length)

    prompt = open(config.prompt_text_file, "r").read().strip()

    dataset = dataset.map(
        lambda x: format_message_from_statements(
            x,
            prompt,
            statements_column=config.statements_column,
            max_items=config.max_statements,
        )
    )

    batch_dataset = dataset.batch(batch_size=config.batch_size)

    for batch in batch_dataset:
        start = time.time()

        prompts = [messages_to_prompt(tokenizer, m) for m in batch["messages"]]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        gen_ids = out_ids[:, prompt_len:]
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        json_outputs = [extract_json_object_from_output(t) for t in gen_texts]
        batch["cleaned"] = json_outputs

        del batch["messages"]

        pd.DataFrame(batch).to_csv(
            path,
            mode="a",
            header=not os.path.exists(path),
            index=False,
            escapechar="\\",
        )

        logging.info(f"[rank {rank}] Batch processed in {time.time() - start:.2f}s")
        empty_cache()

    if rank == 0:
        all_dfs = []
        for r in range(world):
            df_path = os.path.join(config.output_dir, f"clean_statements_{data_set}_{r}.csv")
            if os.path.exists(df_path):
                all_dfs.append(pd.read_csv(df_path))
            else:
                logging.warning(f"Expected file not found: {df_path}")
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(os.path.join(config.output_dir, f"dataset.csv"), index=False)
            logging.info(f"Combined CSV saved with {len(combined_df)} rows.")
        else:
            logging.warning("No individual CSV files found to combine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt_text_file", type=str, default="extraction/prompts/verification.txt")
    parser.add_argument("--dataset_path", type=str, default="data/RecommendationDatasets/StatementDatasets/Toys14/dataset_0.csv")
    parser.add_argument("--output_dir", type=str, default="data/RecommendationDatasets/StatementDatasets/Toys14")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--statements_column", type=str, default="statements")
    parser.add_argument("--max_statements", type=int, default=128)

    config = parser.parse_args()
    main(config)
