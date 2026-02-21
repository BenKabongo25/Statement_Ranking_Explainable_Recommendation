import argparse
import numpy as np
import os
import pandas as pd
import torch

from accelerate import Accelerator
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Sequence, Tuple

from clustering.utils import load_statements, load_neighbours


DEFAULT_INSTRUCTION = (
    "Given two short statements, decide if they are perfect paraphrases or synonyms (same meaning). "
    'Answer "yes" only when they clearly express the same idea, otherwise answer "no".'
)


@dataclass
class PairCandidate:
    sentiment: str
    a_idx: int
    b_idx: int
    statement_a: str
    statement_b: str
    bi_encoder_sim: float
    bi_encoder_rank: int
    threshold: float


def collect_unique_pairs(
    df: pd.DataFrame, neighbours: np.ndarray, similarities: np.ndarray, threshold: float
) -> List[PairCandidate]:
    if neighbours.shape != similarities.shape:
        raise ValueError("indices and similarities must have the same shape.")
    n, k = neighbours.shape
    if len(df) != n:
        raise ValueError(f"Row count mismatch: CSV has {len(df)} rows, neighbours have {n}.")

    pairs: Dict[Tuple[int, int], PairCandidate] = {}
    for i in tqdm(range(n), desc="Collecting unique pairs"):
        sentiment = df.loc[i, "sentiment"]
        for rank, (j, sim) in enumerate(zip(neighbours[i], similarities[i]), start=1):
            j = int(j)
            if j < 0 or sim < threshold:
                continue
            a, b = sorted((i, j))
            key = (a, b)
            if key not in pairs:
                pairs[key] = PairCandidate(
                    sentiment=sentiment,
                    a_idx=a,
                    b_idx=b,
                    statement_a=str(df.loc[a, "statement"]),
                    statement_b=str(df.loc[b, "statement"]),
                    bi_encoder_sim=float(sim),
                    bi_encoder_rank=rank,
                    threshold=threshold,
                )
            else:
                if sim > pairs[key].bi_encoder_sim:
                    pairs[key].bi_encoder_sim = float(sim)
                if rank < pairs[key].bi_encoder_rank:
                    pairs[key].bi_encoder_rank = rank
                if pairs[key].sentiment != sentiment:
                    raise ValueError(f"Sentiment mismatch for pair {key}: {pairs[key].sentiment} vs {sentiment}")
    return list(pairs.values())


def format_instruction(instruction: str, stmt_a: str, stmt_b: str) -> str:
    instruction = instruction or DEFAULT_INSTRUCTION
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {stmt_a}\n"
        f"<Document>: {stmt_b}"
    )


def build_prompts(pairs: Sequence[PairCandidate], instruction: str) -> List[str]:
    return [format_instruction(instruction, p.statement_a, p.statement_b) for p in pairs]


def prepare_inputs(
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    prefix_tokens: List[int],
    suffix_tokens: List[int],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(
        texts,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    return inputs


@torch.no_grad()
def score_batches(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    accelerator: Accelerator,
    pairs: Sequence[PairCandidate],
    output_path: str,
    verbose: bool = True,
) -> None:
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    batches_per_process = (total_batches + num_processes - 1) // num_processes
    start_batch = process_index * batches_per_process
    end_batch = min(start_batch + batches_per_process, total_batches)
    
    start_idx = start_batch * batch_size
    end_idx = min(end_batch * batch_size, len(texts))
    
    print(f"Rank {process_index}/{num_processes}: processing indices {start_idx}-{end_idx}")
    
    output_dir = os.path.dirname(output_path) or "."
    output_basename = os.path.basename(output_path)
    output_name, output_ext = os.path.splitext(output_basename)
    rank_output_path = os.path.join(output_dir, f"{output_name}_rank{process_index}{output_ext}")
    
    if os.path.exists(rank_output_path):
        os.remove(rank_output_path)
    
    total_saved = 0
    
    for start in tqdm(
        range(start_idx, end_idx, batch_size), 
        desc=f"Scoring batches (Rank {process_index})",
        disable=process_index != 0
    ):
        chunk = texts[start : min(start + batch_size, end_idx)]
        inputs = prepare_inputs(tokenizer, chunk, prefix_tokens, suffix_tokens, max_length)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        stacked = torch.nn.functional.log_softmax(stacked, dim=1)
        probs_true = stacked[:, 1].exp().cpu().tolist()
        
        batch_records = []
        for i, score in enumerate(probs_true):
            idx = start + i
            pair = pairs[idx]
            batch_records.append({
                "rank": process_index,
                "sentiment": pair.sentiment,
                "statement_1_index": pair.a_idx,
                "statement_2_index": pair.b_idx,
                "statement_1": pair.statement_a,
                "statement_2": pair.statement_b,
                "bi_encoder_sim": pair.bi_encoder_sim,
                "bi_encoder_rank": pair.bi_encoder_rank,
                "cross_encoder_sim": float(score),
                "threshold": pair.threshold,
            })
        
        batch_df = pd.DataFrame.from_records(batch_records)
        write_header = not os.path.exists(rank_output_path)
        batch_df.to_csv(rank_output_path, mode='a', header=write_header, index=False)
        total_saved += len(batch_records)

        if verbose and start % (100 * batch_size) == 0 and process_index == 0:
            print(f"Sample scores at index {start}: {probs_true[:5]}")
    
    print(f"Rank {process_index}: Saved {total_saved} pairs to {rank_output_path}")
    accelerator.wait_for_everyone()


def rerank_pairs(
    pairs: Sequence[PairCandidate],
    model_name: str,
    batch_size: int,
    max_length: int,
    output_path: str,
    accelerator: Accelerator,
    verbose: bool = True,
) -> None:
    if accelerator.is_local_main_process:
        print(f"Using {accelerator.num_processes} process(es)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(accelerator.device).eval()
    
    texts = build_prompts(pairs, DEFAULT_INSTRUCTION)
    
    score_batches(
        model, 
        tokenizer, 
        texts, 
        batch_size=batch_size, 
        max_length=max_length, 
        accelerator=accelerator,
        pairs=pairs,
        output_path=output_path,
        verbose=verbose
    )


def merge_rank_csvs(output_path: str, num_processes: int) -> None:
    output_dir = os.path.dirname(output_path) or "."
    output_basename = os.path.basename(output_path)
    output_name, output_ext = os.path.splitext(output_basename)
    
    rank_files = []
    for rank in range(num_processes):
        rank_path = os.path.join(output_dir, f"{output_name}_rank{rank}{output_ext}")
        if os.path.exists(rank_path):
            rank_files.append(rank_path)
    
    if not rank_files:
        print("No rank files found to merge.")
        return
    
    print(f"\nMerging {len(rank_files)} rank files...")
    dfs = []
    for rank_file in sorted(rank_files):
        df = pd.read_csv(rank_file)
        dfs.append(df)
        print(f"  - Loaded {len(df)} rows from {os.path.basename(rank_file)}")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    if 'rank' in merged_df.columns:
        merged_df = merged_df.drop(columns=['rank'])
    
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged {len(merged_df)} total pairs to {output_path}")
    
    #print("\nCleaning up temporary rank files...")
    #for rank_file in rank_files:
    #    try:
    #        os.remove(rank_file)
    #        print(f"  - Removed {os.path.basename(rank_file)}")
    #    except Exception as e:
    #        print(f"  ! Warning: Could not remove {rank_file}: {e}")


def main(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        print("Loading statements and neighbours...")
    
    df = load_statements(args.statement_path)
    neighbours, similarities = load_neighbours(args.neighbours_path)
    pairs = collect_unique_pairs(df, neighbours, similarities, threshold=args.threshold)
    
    if not pairs:
        if accelerator.is_local_main_process:
            print(f"No pairs found with similarity >= {args.threshold}. Nothing to rerank.")
        return

    if accelerator.is_local_main_process:
        print(f"Found {len(pairs)} pairs to rerank")
    
    rerank_pairs(
        pairs,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_path=args.output_path,
        accelerator=accelerator,
        verbose=args.verbose,
    )
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
        merge_rank_csvs(args.output_path, accelerator.num_processes)
        print(f"\nAll done! Final results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--statement_path", 
        type=str,
        required=True, 
        help="CSV with columns 'statement' and 'sentiment'."
    )
    parser.add_argument(
        "--neighbours_path", 
        type=str,
        required=True, 
        help="NPZ produced by similarity_search.py (indices & similarities).")
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True, 
        help="Path to save reranked pairs as CSV.")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.90, 
        help="Minimum bi-encoder similarity to consider.")
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128, 
        help="Batch size for the reranker.")
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=8192, 
        help="Max sequence length for the reranker."
    )
    parser.add_argument(
        "--model_name", 
        default="Qwen/Qwen3-Reranker-0.6B", 
        help="HuggingFace model id for the reranker."
    )
    parser.add_argument(
        "--verbose", 
        action=argparse.BooleanOptionalAction, 
        default=True, 
        help="Enable verbose output."
    )
    args = parser.parse_args()
    main(args)