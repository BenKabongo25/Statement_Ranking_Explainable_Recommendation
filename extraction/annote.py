import argparse
import datasets
import json
import logging
import os
import pandas as pd
import re
import time
import torch

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM


AMAZON_2014_ENTRIES = {
    "overall": "rating",
    "unixReviewTime": "timestamp",
    "reviewText": "review",
    "reviewerName": "user_name",
    "reviewerID": "user_id",
    "asin": "item_id",
    "summary": "review_title",
}


def format_message(example, prompt, max_length: int):
    review = example["review"]
    review_title = example["review_title"]
    if len(review.split()) > max_length:
        review = " ".join(review.split()[:max_length])

    example["messages"] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Title: "+review_title+ " - Review: " + review + "\nOutput:"},
    ]
    return example


def extract_json_from_output(text):
    #print("text:",text)
    #print("------------------------------------")
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def empty_cache():
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()


def messages_to_prompt(tokenizer, messages):
    """
    Converts your list-of-dicts chat messages into a single text prompt.
    Uses the model's chat template if available (best for Qwen).
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False 
    )



def main(config):
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    print("ranks",rank)
    world = accelerator.num_processes
    logging.info(f"[rank {rank}/{world}] device={device}")

    os.makedirs(config.output_dir, exist_ok=True)

    # IMPORTANT: separate outputs per GPU/process to avoid corrupting one CSV
    path = os.path.join(config.output_dir, f"statement_rank{rank}.csv")

    # Separate streaming resume state per process too
    state_dict_dir = os.path.join(config.output_dir, "state_dicts")
    os.makedirs(state_dict_dir, exist_ok=True)
    state_dict_path = os.path.join(state_dict_dir, f"state_dict.rank{rank}.json")

    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    terminators = [t for t in terminators if isinstance(t, int)]

    fmt = "json" if config.json_format else "csv"
    dataset = datasets.load_dataset(
        fmt,
        data_files={"train": config.dataset_path},
        split="train",
        streaming=True
    )

    if config.format == "amz14":
        for key, value in AMAZON_2014_ENTRIES.items():
            dataset = dataset.rename_column(key, value)

    # Split dataset across GPUs/processes (rank0 gets 0,2,4... rank1 gets 1,3,5...)
    dataset = dataset.filter(lambda ex, idx: (idx % world) == rank, with_indices=True)


    # If resuming: skip already-produced rows for THIS rank only
    if os.path.exists(path) and config.skip_existing:
        df = pd.read_csv(path)
        length = len(df)
        logging.info(f"[rank {rank}] Skipping {length} already-done rows from {path}")
        dataset = dataset.skip(length)
    else:
        logging.info(f"[rank {rank}] No existing results found in {path}")

    prompt = open(config.prompt_text_file, "r").read().strip()
    dataset = dataset.map(lambda x: format_message(x, prompt, config.max_length))

    batch_dataset = dataset.batch(batch_size=config.batch_size)

    #if os.path.exists(state_dict_path):
    #    with open(state_dict_path, "r") as f:
    #        batch_dataset.load_state_dict(json.load(f))
    #    logging.info(f"[rank {rank}] Loaded state dict from {state_dict_path}")
    #else:
    #    logging.info(f"[rank {rank}] No state dict found!")

    for batch in batch_dataset:
        start = time.time()

        # Build prompts from your messages
        prompts = [messages_to_prompt(tokenizer, m) for m in batch["messages"]]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # lengths per sample (so we can slice generated tokens cleanly)
        #lengths = inputs["attention_mask"].sum(dim=1).tolist()
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                #eos_token_id=terminators,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # Extract only the generated continuation (not the prompt)
        #gen_texts = []
        #for i in range(out_ids.shape[0]):
        gen_ids = out_ids[:, prompt_len:]
        print("gen_ids",gen_ids.shape)
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        json_outputs = [extract_json_from_output(t) for t in gen_texts]

        batch["statements"] = json_outputs
        del batch["messages"]

        pd.DataFrame(batch).to_csv(
            path,
            mode="a",
            header=not os.path.exists(path),
            index=False,
            escapechar="\\",
        )

        with open(state_dict_path, "w") as f:
            json.dump(batch_dataset.state_dict(), f)

        logging.info(f"[rank {rank}] Batch processed in {time.time() - start:.2f}s")
        empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen3-14B")
    parser.add_argument("--prompt_text_file", type=str, default="prompts/prompt5.txt")
    parser.add_argument("--dataset_path", type=str, default="data/RecommendationDatasets/reviews.json")
    parser.add_argument("--json_format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--format", type=str, default="amz14", choices=["amz14"])
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_length", type=int, default=128)

    config = parser.parse_args()
    main(config)
