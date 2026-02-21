import argparse
import json
import math
import os
import pandas as pd

from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams


def make_prompt(statement: str,tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Statement: {statement.strip()}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

SYSTEM_PROMPT = """
You are a verifier for extracted product statements.

You will receive ONE statement (already rewritten from a review).
Your task is ONLY to check two binary properties:

A) Atomicity:
- atomicity = 1 if it expresses ONE main fact about the product (minor qualifiers are ok).
- atomicity = 0 if it contains multiple independent facts/clauses that could be split
  (often signaled by "and/but/or", "because", "while", ";", multiple sentences).

B) Explainatoriness (product-descriptive & impersonal):
- explainatoriness = 1 if it is a general, product-focused description: a product property, behavior,
  performance, or useful outcome that could apply to any user.
- explainatoriness = 0 if it includes personal life details, user-specific context, or preference framing
  that makes it about the reviewer rather than the product (e.g., mentions my/I/our, my child, for my trip).

OUTPUT RULE (STRICT):
- If both atomicity = 1 AND explainatoriness = 1, return:
  {"atomicity": 1, "explainatoriness": 1}
- If not atomicity but explainatoriness:
  {"atomicity": 0, "explainatoriness": 1}
- and so on.

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON.

No extra keys. No rationale. No markdown.
""".strip()

USER_TEMPLATE = "Statement: {statement}\n"


def extract_json_object(text: str) -> List[str]:
    """
    Extract all JSON objects from a string by scanning with JSONDecoder.
    This is more reliable than regex when the model mixes JSON + other tokens.
    """
    if not text:
        return []

    s = text.strip()
    dec = json.JSONDecoder()
    objs = []

    i = 0
    n = len(s)
    while i < n:
        # find next '{'
        j = s.find("{", i)
        if j == -1:
            break
        try:
            obj, end = dec.raw_decode(s[j:])
            # Only keep dict objects (we expect dict)
            if isinstance(obj, dict):
                objs.append(s[j:j+end])
            i = j + end
        except json.JSONDecodeError:
            i = j + 1

    return objs


def parse_verdict(raw: str) -> Optional[Tuple[int, int]]:
    """
    Expect JSON with exactly keys: atomicity, explainatoriness.
    Values must be 0 or 1 (ints). Accept "0"/"1" strings too, but normalize to int.
    """
    try:
        obj = json.loads(raw)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    # strict keys
    if set(obj.keys()) != {"atomicity", "explainatoriness"}:
        return None

    def norm01(v: Any) -> Optional[int]:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int) and v in (0, 1):
            return v
        if isinstance(v, str):
            v2 = v.strip()
            if v2 in ("0", "1"):
                return int(v2)
        return None

    a = norm01(obj.get("atomicity"))
    e = norm01(obj.get("explainatoriness"))
    if a is None or e is None:
        return None
    return a, e


def run_llm_batch(llm: LLM, prompts: List[str], sampling: SamplingParams) -> List[str]:
    outs = llm.generate(prompts, sampling)
    texts: List[str] = []
    for o in outs:
        if o.outputs:
            texts.append(o.outputs[0].text)
        else:
            texts.append("")
    return texts


@dataclass
class InferenceConfig:
    model: str
    tp: int
    batch_size: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    max_model_len: int
    gpu_mem_util: float
    retries: int


def infer_labels_for_texts(
    llm: LLM,
    texts: List[str],
    cfg: InferenceConfig,
    tokenizer,
) -> Dict[str, Tuple[int, int]]:
    """
    Batch infer labels for a list of unique texts.
    Returns mapping: text -> (atomicity, explainatoriness)
    """
    sampling = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        seed=cfg.seed,
    )

    results: Dict[str, Tuple[int, int]] = {}
    total = len(texts)

    for i in tqdm(range(0, total, cfg.batch_size), desc="LLM verify", unit="stmt"):
        batch_texts = texts[i:i + cfg.batch_size]
        prompts = [make_prompt(t,tokenizer) for t in batch_texts]
        raw_outs = run_llm_batch(llm, prompts, sampling)

        for t, raw in zip(batch_texts, raw_outs):
            verdict: Optional[Tuple[int, int]] = None

            # attempt 0: parse directly (or extracted JSON)
            #json_str = extract_json_object(raw)
            #if json_str:
            verdict = parse_verdict(raw)

            # retries with stricter reminder if invalid
            attempt = 0
            while verdict is None and attempt < cfg.retries:
                print(raw)
                print("\n")
                #print(json_str)
                print("is not valid")
                print("\n")
                attempt += 1
                retry_prompt = (
                    f"<|system|>\n{SYSTEM_PROMPT}\n"
                    f"<|user|>\n{USER_TEMPLATE.format(statement=t)}"
                    f"<|assistant|>\n"
                    "Return ONLY JSON with exactly keys atomicity and explainatoriness and values 0 or 1.\n"
                )
                raw2 = run_llm_batch(llm, [retry_prompt], sampling)[0]
                json_str2 = extract_json_object(raw2)
                if json_str2:
                    verdict = parse_verdict(json_str2)

            

            results[t] = verdict

    return results


def is_missing(x: Any) -> bool:
    # pandas NaN / None handling
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip() == "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallel size (#GPUs)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--statements-col", default="statements")
    ap.add_argument("--cleaned-col", default="cleaned")
    args = ap.parse_args()

    cfg = InferenceConfig(
        model=args.model,
        tp=args.tp,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
        retries=args.retries,
    )

    tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    use_fast=True,
    )

    df = pd.read_csv(args.input)

    st_col = args.statements_col
    cl_col = args.cleaned_col
    if st_col not in df.columns or cl_col not in df.columns:
        raise ValueError(f"Missing required columns. Need '{st_col}' and '{cl_col}' in CSV.")

    # Collect unique non-missing texts for each column separately (independent evaluation).
    statements_texts = sorted({str(x).strip() for x in df[st_col].tolist() if not is_missing(x)})
    cleaned_texts = sorted({str(x).strip() for x in df[cl_col].tolist() if not is_missing(x)})

    # Init LLM once
    llm = LLM(
        model=cfg.model,
        tensor_parallel_size=cfg.tp,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_mem_util,
        trust_remote_code=True,
    )

    # Infer for each column independently (so the model sees the exact statement being verified).
    print(f"Unique '{st_col}': {len(statements_texts)}")
    map_statements = infer_labels_for_texts(llm, statements_texts, cfg,tokenizer) if statements_texts else {}

    print(f"Unique '{cl_col}': {len(cleaned_texts)}")
    map_cleaned = infer_labels_for_texts(llm, cleaned_texts, cfg,tokenizer) if cleaned_texts else {}

    # Write new columns
    df[f"{st_col}_atomicity"] = df[st_col].apply(lambda x: map_statements.get(str(x).strip(), (None, None))[0] if not is_missing(x) else None)
    df[f"{st_col}_explainatoriness"] = df[st_col].apply(lambda x: map_statements.get(str(x).strip(), (None, None))[1] if not is_missing(x) else None)

    df[f"{cl_col}_atomicity"] = df[cl_col].apply(lambda x: map_cleaned.get(str(x).strip(), (None, None))[0] if not is_missing(x) else None)
    df[f"{cl_col}_explainatoriness"] = df[cl_col].apply(lambda x: map_cleaned.get(str(x).strip(), (None, None))[1] if not is_missing(x) else None)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
    