import argparse
import json
import math
import numpy as np
import torch

from pathlib import Path
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict, Tuple

from bperp import BPERp, Batchify
from utils import now_time, load_data

from baselines.main_popularity import (
    get_test_ground_truth,
    build_user_statements,
    build_item_statements,
)
from evaluation.metrics import StatementRankingMetrics

parser = argparse.ArgumentParser(description='Bayesian Personalized Explanation Ranking enhanced by BERT (BPER+)')
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Path to the dataset directory containing data.")
parser.add_argument('--save_dir', type=str, required=True,
                    help='directory to save results')
parser.add_argument("--save_everything", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to save per-interaction data for analysis")
parser.add_argument('--dimension', type=int, default=20,
                    help='number of latent factors')
parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                    help='BERT type or folder to load pre-downloaded, see https://huggingface.co/transformers/pretrained_models.html')
parser.add_argument('--hidden_size', type=int, default=768,
                    help='hidden size of BERT (see the above webpage)')
parser.add_argument('--seq_max_len', type=int, default=10,
                    help='number of words to use for each explanation')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=4096,
                    help='batch size')
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True,
                    help='use CUDA')
parser.add_argument('--top_k', type=int, default=20,
                    help='select top k to evaluate')
parser.add_argument('--mu_on_user', type=float, default=0.7,
                    help='ratio on user for score prediction (-1 means from 0, 0.1, ..., 1)')
args = parser.parse_args()

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

print(now_time() + 'Loading data')
ldata = load_data(args.dataset_dir)
train_tuple_list = ldata.train_tuple_list
eval_tuple_list = ldata.eval_tuple_list
test_tuple_list = ldata.test_tuple_list
user2items_test = ldata.user2items_test
user2index = ldata.user_id2index
item2index = ldata.item_id2index
text_list = ldata.text_list
text_idx = ldata.test_idx
n_interactions = ldata.n_interactions
exp2index = {cid: cid for cid in range(len(text_list))}
S = len(text_list)  # total number of explanations

data = Batchify(train_tuple_list, text_list, args.model_name, args.seq_max_len, args.batch_size, device)
user_test, item_test, exp_test, text_test, mask_test = data.prediction_batch(test_tuple_list)
model = BPERp(args.batch_size, len(user2index), len(item2index), len(exp2index), args.model_name, args.hidden_size, args.dimension).to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

print(now_time() + 'Training')
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.
    total_sample = 0
    while True:
        user, item, exp, text, mask = data.next_batch()
        batch_size = user.size(0)
        optimizer.zero_grad()
        loss = model(user, item, exp, text, mask)
        loss.backward()
        optimizer.step()

        total_loss += batch_size * loss.item()
        total_sample += batch_size
        if data.step == data.total_step:
            break
    print(now_time() + 'epoch {} loss: {:4.4f}'.format(epoch, total_loss / total_sample))

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# Model saving/loading
model_path = Path(args.save_dir) / "bperp_model.pth"

if args.epochs > 0:
    torch.save(model.state_dict(), model_path)
    print(f"[info] Model saved to {model_path}")
else:
    if not model_path.exists():
        print(f"[error] Model file not found at {model_path}")
    else:
        print(f"[info] Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"[info] Model loaded from {model_path}")

model.eval()

print(now_time() + 'Testing')

n_statements = int(ldata.stmt_df.shape[0])
all_stmt_ids = np.arange(n_statements, dtype=np.int32)
    
test_df = ldata.data_df.iloc[ldata.test_idx].reset_index(drop=True)
print(f"[info] Test: {len(test_df)} interactions")
print(f"[info] Total statements: {n_statements}")
    
print("[info] Extracting ground truth...")
gt_list = get_test_ground_truth(test_df, n_statements, "statement_ids")
    
print("[info] Building user/item statement sets...")
user_stmts = build_user_statements(ldata.data_df, ldata.user_id2index, n_statements, "statement_ids")
item_stmts = build_item_statements(ldata.data_df, ldata.item_id2index, n_statements, "statement_ids")

reverse_user_id2index = {v: k for k, v in user2index.items()}
reverse_item_id2index = {v: k for k, v in item2index.items()}

avg_user_stmts = np.mean([len(s) for s in user_stmts.values()]) if user_stmts else 0
avg_item_stmts = np.mean([len(s) for s in item_stmts.values()]) if item_stmts else 0
print(f"[info] Avg statements per user: {avg_user_stmts:.1f}")
print(f"[info] Avg statements per item: {avg_item_stmts:.1f}")

min_user_stmts = np.min([len(s) for s in user_stmts.values()]) if user_stmts else 0
min_item_stmts = np.min([len(s) for s in item_stmts.values()]) if item_stmts else 0
print(f"[info] Min statements per user: {min_user_stmts}")
print(f"[info] Min statements per item: {min_item_stmts}")

max_user_stmts = np.max([len(s) for s in user_stmts.values()]) if user_stmts else 0
max_item_stmts = np.max([len(s) for s in item_stmts.values()]) if item_stmts else 0
print(f"[info] Max statements per user: {max_user_stmts}")
print(f"[info] Max statements per item: {max_item_stmts}")

print("[info] Precomputing text embeddings...")
with torch.no_grad():
    exp_emb_cache = model.get_all_text_embeddings(text_test, mask_test) # (all_exp, dimension)
    exp_emb_cache = exp_emb_cache.cpu()  # Stocker sur CPU
    print(f"[info] Text embeddings cached: {exp_emb_cache.shape}")

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

ks = [1, 3, 5, 10, 20]
#mus = list([x / 10.0 for x in range(0, 11)])
mus = [args.mu_on_user]
paradigms = ["all", "item"]
results: Dict[float, Dict[str, Dict[str, float]]] = {}

metrics: Dict[float, Dict[str, StatementRankingMetrics]] = {}
for mu in mus:
    metrics[mu] = {}
    for paradigm in paradigms:
        metrics[mu][paradigm] = StatementRankingMetrics(ks=ks, save_everything=args.save_everything)
        metrics[mu][paradigm].reset()

N = len(gt_list)
K = len(all_stmt_ids)
steps = int(math.ceil(N / args.batch_size))

for step in tqdm(range(steps), desc=f"Evaluating"):
    s = step * args.batch_size
    e = min(N, s + args.batch_size)
    B = e - s
        
    u_batch = user_test[s:e]  # (B,)
    i_batch = item_test[s:e]  # (B,)

    with torch.no_grad():
        u_scores, i_scores = model(u_batch, i_batch, exp_test, text_test, mask_test, False, exp_emb_cache)
        u_scores = u_scores.detach().cpu()
        i_scores = i_scores.detach().cpu()
        
    scores = {}
    for mu in mus:
        scores[mu] = u_scores * mu + i_scores * (1 - mu)

    for paradigm in paradigms:
        if paradigm == "all":
            # ALL: all statements for every interaction
            stmt_ids_batch = np.tile(all_stmt_ids, (B, 1))  # (B, K)
            
        elif paradigm == "user":
            # USER: all statements seen by each user
            max_K = max(len(user_stmts.get(int(u), [])) for u in u_batch)
            if max_K == 0:
                max_K = 20  # Handle case with no user history
            stmt_ids_batch = np.full((B, max_K), -100, dtype=np.int32)
            
            for b, u in enumerate(u_batch):
                u_stmts = user_stmts.get(int(u), np.array([], dtype=np.int32))
                if len(u_stmts) > 0:
                    stmt_ids_batch[b, :len(u_stmts)] = u_stmts
                    
        elif paradigm == "item":
            # ITEM: all statements seen for each item
            max_K = max(len(item_stmts.get(int(i), [])) for i in i_batch)
            if max_K == 0:
                max_K = 20  # Handle case with no item history
            stmt_ids_batch = np.full((B, max_K), -100, dtype=np.int32)
            
            for b, i in enumerate(i_batch):
                i_stmts = item_stmts.get(int(i), np.array([], dtype=np.int32))
                if step == 0 and b < 5:
                    print(f"[debug] Item {i} statements: {i_stmts}")
                if len(i_stmts) > 0:
                    stmt_ids_batch[b, :len(i_stmts)] = i_stmts

        stmt_ids_t = torch.as_tensor(stmt_ids_batch, dtype=torch.long)
        mask_pos = torch.zeros_like(stmt_ids_t, dtype=torch.bool)
        for b in range(B):
            gt = gt_list[s + b]
            if len(gt) == 0:
                continue
            for sid in gt:
                mask = (stmt_ids_t[b] == sid)
                mask_pos[b] |= mask

        for mu in mus:
            # gather scores
            stmt_scores = scores[mu]
            safe_stmt_ids = stmt_ids_t.clone()
            safe_stmt_ids[safe_stmt_ids < 0] = 0
            stmt_scores = torch.gather(stmt_scores, 1, safe_stmt_ids)  # (B, K)
            stmt_scores[stmt_ids_t < 0] = -1e9  # mask padding

            metrics[mu][paradigm].update(
                stmt_ids=stmt_ids_t,
                stmt_scores=stmt_scores,
                mask_pos=mask_pos,
                user_ids=u_batch,
                item_ids=i_batch,
            )

for mu in mus:
    results[mu] = {}
    for paradigm in paradigms:
        print(f"\n[info] Results for mu={mu}, paradigm={paradigm}:")
        res = metrics[mu][paradigm].compute()
        results[mu][paradigm] = res
            
        out_path = save_dir / f"BPERp_{mu}_{paradigm}.metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, sort_keys=True)

        if args.save_everything:
            data_path = save_dir / f"BPERp_{mu}_{paradigm}.data.csv"
            data_df = metrics[mu][paradigm].data_df
            data_df["user_id"] = data_df["user_idx"].map(reverse_user_id2index)
            data_df["item_id"] = data_df["item_idx"].map(reverse_item_id2index)
            data_df.to_csv(data_path, index=False)


print(f"\n[done] Results saved to {save_dir}")