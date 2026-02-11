import argparse
import json
import math
import numpy as np
import torch

from pathlib import Path
from typing import Dict, List, Tuple

from recbole.config import Config
from recbole.data.interaction import Interaction
from recbole.utils import init_seed

from Custom.dataset import CustomDataset
from Custom.trainer import JointTagTrainer
from Custom.utils import custom_get_model, custom_data_preparation

from evaluation.metrics import StatementRankingMetrics


def _unique_positive_tags(tag_row: np.ndarray) -> np.ndarray:
    tag_ids = tag_row[tag_row > 0]
    if tag_ids.size == 0:
        return np.array([], dtype=np.int32)
    return np.unique(tag_ids).astype(np.int32)


def _build_statement_maps(
    datasets,
    uid_field: str,
    iid_field: str,
    tag_field: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    user_stmts: Dict[int, set] = {}
    item_stmts: Dict[int, set] = {}

    for ds in datasets:
        inter = ds.inter_feat
        users = inter[uid_field].cpu().numpy()
        items = inter[iid_field].cpu().numpy()
        tags = inter[tag_field].cpu().numpy()

        for idx in range(users.shape[0]):
            tag_ids = _unique_positive_tags(tags[idx])
            if tag_ids.size == 0:
                continue
            u_idx = int(users[idx])
            i_idx = int(items[idx])
            user_stmts.setdefault(u_idx, set()).update(tag_ids.tolist())
            item_stmts.setdefault(i_idx, set()).update(tag_ids.tolist())

    user_out = {u: np.array(sorted(sids), dtype=np.int32) for u, sids in user_stmts.items()}
    item_out = {i: np.array(sorted(sids), dtype=np.int32) for i, sids in item_stmts.items()}
    return user_out, item_out


def _build_gt_list(test_dataset, tag_field: str) -> List[np.ndarray]:
    tags = test_dataset.inter_feat[tag_field].cpu().numpy()
    return [_unique_positive_tags(row) for row in tags]


def _pad_stmt_ids(stmt_ids: np.ndarray, min_k: int) -> np.ndarray:
    if stmt_ids.shape[1] >= min_k:
        return stmt_ids
    pad_width = min_k - stmt_ids.shape[1]
    pad = np.full((stmt_ids.shape[0], pad_width), -100, dtype=np.int32)
    return np.concatenate([stmt_ids, pad], axis=1)


@torch.no_grad()
def evaluate_statement_ranking(
    model,
    device: torch.device,
    user_ids_test: np.ndarray,
    item_ids_test: np.ndarray,
    gt_list: List[np.ndarray],
    all_stmt_ids: np.ndarray,
    item_stmts: Dict[int, np.ndarray],
    batch_size: int,
    ks: Tuple[int, ...],
    paradigms: Tuple[str, ...],
    uid_field: str,
    iid_field: str,
    save_everything: bool = False,
) ->  Dict[str, StatementRankingMetrics]:
    metrics: Dict[str, StatementRankingMetrics] = {}
    for paradigm in paradigms:
        metrics[paradigm] = StatementRankingMetrics(ks=ks, save_everything=save_everything)
        metrics[paradigm].reset()

    max_k = max(ks)
    N = len(gt_list)
    steps = int(math.ceil(N / batch_size))

    for step in range(steps):
        s = step * batch_size
        e = min(N, s + batch_size)
        B = e - s

        u_batch = user_ids_test[s:e]
        i_batch = item_ids_test[s:e]

        interaction = Interaction({
            uid_field: torch.as_tensor(u_batch, dtype=torch.long, device=device),
            iid_field: torch.as_tensor(i_batch, dtype=torch.long, device=device),
        })
        scores = model.tag_predict(interaction)

        for paradigm in paradigms:
            if paradigm == "all":
                stmt_ids_batch = np.tile(all_stmt_ids, (B, 1))
            elif paradigm == "item":
                max_K = 0
                for i in i_batch:
                    max_K = max(max_K, len(item_stmts.get(int(i), [])))
                max_K = max(max_K, max_k)
                stmt_ids_batch = np.full((B, max_K), -100, dtype=np.int32)

                for b, i in enumerate(i_batch):
                    i_stmts = item_stmts.get(int(i), np.array([], dtype=np.int32))
                    if i_stmts.size > 0:
                        stmt_ids_batch[b, :i_stmts.size] = i_stmts
            else:
                raise ValueError(f"Unknown paradigm: {paradigm}")

            stmt_ids_batch = _pad_stmt_ids(stmt_ids_batch, max_k)
            stmt_ids_t = torch.as_tensor(stmt_ids_batch, dtype=torch.long, device=device)

            mask_pos = torch.zeros_like(stmt_ids_t, dtype=torch.bool)
            for b in range(B):
                gt = gt_list[s + b]
                if gt.size == 0:
                    continue
                for sid in gt:
                    mask_pos[b] |= (stmt_ids_t[b] == int(sid))

            safe_stmt_ids = stmt_ids_t.clone()
            safe_stmt_ids[safe_stmt_ids < 0] = 0
            stmt_scores = torch.gather(scores, 1, safe_stmt_ids)
            stmt_scores[stmt_ids_t < 0] = -1e9

            metrics[paradigm].update(
                stmt_ids=stmt_ids_t,
                stmt_scores=stmt_scores,
                mask_pos=mask_pos,
            )

    #results: Dict[str, Dict[str, float]] = {}
    #for paradigm in paradigms:
    #    results[paradigm] = metrics[paradigm].compute()
    #return results
    return metrics


def _load_checkpoint(model, model_file: Path, device: torch.device) -> None:
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ExpGCN statement-ranking evaluation")
    parser.add_argument("--dataset", "-d", type=str, default="AmazonMTV",
                        help="Dataset name (used to pick Params/{dataset}.yaml)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save evaluation metrics")
    parser.add_argument("--model_file", type=str, default=None,
                        help="Optional checkpoint file to load (skips training)")
    parser.add_argument("--train", type=int, default=1,
                        help="Train the model before evaluation")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                        help="Cutoff values for metrics")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for statement evaluation (default: eval_batch_size)")
    parser.add_argument("--paradigms", type=str, nargs="+", default=["all", "item"],
                        help="Ranking paradigms to evaluate: all, item")
    args = parser.parse_args()

    config_files = [
        "Params/Overall.yaml",
        f"Params/{args.dataset}.yaml",
    ]
    model_class = custom_get_model(config_files)
    config = Config(model=model_class, config_file_list=config_files)
    init_seed(config["seed"], config["reproducibility"])

    dataset = CustomDataset(config)
    train_data, valid_data, test_data = custom_data_preparation(config, dataset)
    train_dataset = train_data.dataset
    valid_dataset = valid_data[0].dataset
    test_dataset = test_data[0].dataset

    model = model_class(config, train_dataset).to(config["device"])
    trainer = JointTagTrainer(config, model)

    model_file = None
    if args.train:
        trainer.fit(train_data, valid_data, verbose=True, show_progress=True)
        model_file = Path(trainer.saved_model_file)
    elif args.model_file:
        model_file = Path(args.model_file)
    else:
        raise ValueError("Either enable --train or provide --model_file")

    if model_file is not None:
        _load_checkpoint(model, model_file, config["device"])

    model.eval()

    tag_field = test_dataset.tag_field
    uid_field = test_dataset.uid_field
    iid_field = test_dataset.iid_field

    user_ids_test = test_dataset.inter_feat[uid_field].cpu().numpy().astype(np.int64)
    item_ids_test = test_dataset.inter_feat[iid_field].cpu().numpy().astype(np.int64)
    gt_list = _build_gt_list(test_dataset, tag_field)

    _, item_stmts = _build_statement_maps(
        datasets=[train_dataset, valid_dataset, test_dataset],
        uid_field=uid_field,
        iid_field=iid_field,
        tag_field=tag_field,
    )

    n_statements = int(test_dataset.num(tag_field))
    all_stmt_ids = np.arange(1, n_statements, dtype=np.int32)
    if all_stmt_ids.size == 0:
        raise ValueError("No statements found in dataset.")

    eval_batch_size = args.batch_size if args.batch_size is not None else config["eval_batch_size"]
    metrics = evaluate_statement_ranking(
        model=model,
        device=config["device"],
        user_ids_test=user_ids_test,
        item_ids_test=item_ids_test,
        gt_list=gt_list,
        all_stmt_ids=all_stmt_ids,
        item_stmts=item_stmts,
        batch_size=eval_batch_size,
        ks=tuple(args.ks),
        paradigms=tuple(args.paradigms),
        uid_field=uid_field,
        iid_field=iid_field,
        save_everything=True,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for paradigm, metrics in metrics.items():
        res = metrics.compute()
        out_path = save_dir / f"ExpGCN_{paradigm}.metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, sort_keys=True)

        data_path = save_dir / f"ExpGCN_{paradigm}.data.csv"
        data_df = metrics.data_df
        #data_df["user_id"] = data_df["user_idx"].map(reverse_user_id2index)
        #data_df["item_id"] = data_df["item_idx"].map(reverse_item_id2index)
        data_df.to_csv(data_path, index=False)

    print(f"[done] Results saved to {save_dir}")

if __name__ == "__main__":
    main()
