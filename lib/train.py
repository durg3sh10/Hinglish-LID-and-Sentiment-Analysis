"""Training / evaluation loop for the switch-aware sentiment model.

Outputs (per run, in <results_dir>):
    <run_name>.csv           id, predictions        -> consumed by the shared evaluate_all.py
    ground.csv               id, sentiment          -> the shared ground-truth file
    <run_name>_metrics.json  dev/test accuracy, macro P/R/F1 (same formula as evaluate_all.py)
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from evaluate_all import macro_metrics                      # shared metric, do not modify   # noqa: E402
from lib.data import ID2LABEL, load_predicted_tags, load_split  # noqa: E402
from lib.model import SwitchAwareSentimentModel               # noqa: E402
from lib.switch import align_to_subwords, switch_distances, switch_points  # noqa: E402

DIST_PAD = 3   # "3+" bucket for special tokens / padding
BIN_PAD = 0


class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, args, pred_tags=None):
        self.rows = df.to_dict("records")
        self.tok = tokenizer
        self.args = args
        self.pred_tags = pred_tags

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        tags = self.pred_tags[r["id"]] if self.pred_tags is not None else r["tags"]
        sw = switch_points(tags, ignore_other=not self.args.naive_switch)
        if self.args.pooling == "distance":
            feat, pad = switch_distances(sw, max_bucket=DIST_PAD), DIST_PAD
        else:
            feat, pad = sw, BIN_PAD
        enc = self.tok(r["tokens"], is_split_into_words=True, truncation=True, max_length=self.args.max_length)
        word_ids = enc.word_ids()
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "pool_mask": [0 if w is None else 1 for w in word_ids],
                "switch_feat": align_to_subwords(feat, word_ids, pad, mode=self.args.switch_subword),
                "label": r["label"], "id": r["id"], "n_switches": sum(sw)}


def make_collate(pad_id):
    def collate(batch):
        T = max(len(b["input_ids"]) for b in batch)
        def pad(key, val):
            return torch.tensor([b[key] + [val] * (T - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
                "pool_mask": pad("pool_mask", 0), "switch_feat": pad("switch_feat", 0),
                "labels": torch.tensor([b["label"] for b in batch]),
                "ids": [b["id"] for b in batch], "n_switches": [b["n_switches"] for b in batch]}
    return collate


@torch.no_grad()
def run_eval(model, loader, device, use_amp):
    model.eval()
    ids, preds, golds, nsw = [], [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                        batch["pool_mask"].to(device), batch["switch_feat"].to(device))
        preds += out["logits"].argmax(-1).cpu().tolist()
        golds += batch["labels"].tolist(); ids += batch["ids"]; nsw += batch["n_switches"]
    y_true = pd.Series([ID2LABEL[g] for g in golds]); y_pred = pd.Series([ID2LABEL[p] for p in preds])
    metrics = macro_metrics(y_true, y_pred)
    has_sw = pd.Series(nsw) > 0                         # breakdown: sentences with vs. without a switch
    if has_sw.any():
        metrics["with_switch"] = {"n": int(has_sw.sum()), **macro_metrics(y_true[has_sw], y_pred[has_sw])}
    if (~has_sw).any():
        metrics["no_switch"] = {"n": int((~has_sw).sum()), **macro_metrics(y_true[~has_sw], y_pred[~has_sw])}
    return metrics, pd.DataFrame({"id": ids, "predictions": y_pred})


def trainable_state(model):
    """State dict restricted to trainable tensors (LoRA adapters + heads), so 4-bit quantized
    encoder weights are never snapshotted or re-loaded."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def train_and_evaluate(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.model_name.split('/')[-1]}_{args.pooling}_{args.switch_source}"

    splits = {s: load_split(s) for s in ["train", "dev", "test"]}
    if args.max_train_samples:
        splits["train"] = splits["train"].sample(n=args.max_train_samples, random_state=args.seed)
    if args.max_eval_samples:
        for s in ["dev", "test"]:
            splits[s] = splits[s].sample(n=min(args.max_eval_samples, len(splits[s])), random_state=args.seed)
    pred_tags = None
    if args.switch_source == "predicted":
        pred_tags = {s: load_predicted_tags(s, args.pred_tags_dir) for s in splits}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:                         # decoder-only models (Llama/Mistral/Qwen)
        tokenizer.pad_token = tokenizer.eos_token
    collate = make_collate(tokenizer.pad_token_id)
    loaders = {s: DataLoader(SentimentDataset(df, tokenizer, args, pred_tags[s] if pred_tags else None),
                             batch_size=args.batch_size if s == "train" else 2 * args.batch_size,
                             shuffle=(s == "train"), collate_fn=collate, num_workers=2)
               for s, df in splits.items()}

    model = SwitchAwareSentimentModel(args.model_name, pooling=args.pooling, switch_dim=args.switch_dim,
                                      scorer=args.scorer, lora=args.lora, lora_r=args.lora_r,
                                      load_in_4bit=args.load_in_4bit, lora_targets=args.lora_targets)
    if args.load_in_4bit:        # quantized encoder is already placed by device_map; move only the heads
        model.pooling.to(device); model.dropout.to(device); model.classifier.to(device)
    else:
        model.to(device)
    model.encoder.config.pad_token_id = tokenizer.pad_token_id
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    head_params = list(model.pooling.parameters()) + list(model.classifier.parameters())
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    total = len(loaders["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    print(f"[+] run={run_name} device={device} trainable={sum(p.numel() for p in enc_params + head_params):,}")

    best_f1, best_state, history = -1.0, None, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); losses = []
        for batch in tqdm(loaders["train"], desc=f"epoch {ep + 1}/{args.epochs}"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["pool_mask"].to(device), batch["switch_feat"].to(device),
                            labels=batch["labels"].to(device))
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(out["loss"].item())
        dev_metrics, _ = run_eval(model, loaders["dev"], device, use_amp)
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "dev": dev_metrics})
        print(f"[epoch {ep + 1}] loss={np.mean(losses):.4f} dev_macro_f1={dev_metrics['macro_f1']:.4f} "
              f"dev_acc={dev_metrics['accuracy']:.4f}")
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            best_state = trainable_state(model)

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    dev_metrics, _ = run_eval(model, loaders["dev"], device, use_amp)
    test_metrics, test_pred = run_eval(model, loaders["test"], device, use_amp)

    test_pred.to_csv(results_dir / f"{run_name}.csv", index=False)
    splits["test"][["id", "sentiment"]].to_csv(results_dir / "ground.csv", index=False)
    summary = {"run_name": run_name, "args": vars(args), "best_dev_macro_f1": best_f1,
               "dev": dev_metrics, "test": test_metrics, "history": history,
               "train_minutes": (time.time() - t0) / 60}
    (results_dir / f"{run_name}_metrics.json").write_text(json.dumps(summary, indent=2))
    if args.save_model:
        ckpt = Path("checkpoints") / run_name; ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(trainable_state(model), ckpt / "model.pt")
    print(f"[+] TEST  acc={test_metrics['accuracy']:.4f}  P={test_metrics['macro_precision']:.4f}  "
          f"R={test_metrics['macro_recall']:.4f}  F1={test_metrics['macro_f1']:.4f}")
    print(f"[+] predictions -> {results_dir / (run_name + '.csv')}   ground -> {results_dir / 'ground.csv'}")
    return summary
