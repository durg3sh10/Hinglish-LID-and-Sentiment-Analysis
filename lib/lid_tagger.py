"""Task 1 - token-level LID tagging (EN/HI/O) with a small fine-tuned encoder.

Trains an AutoModelForTokenClassification (default xlm-roberta-base) on the
SentiMix word-level tags (hin / eng / o), reports token-level accuracy and
macro-F1 on dev/test, and writes predicted tags for every split to
    <out_dir>/{split}_pred_tags.csv   (columns: id, pred_tags)
which `main.py --switch_source predicted` consumes to derive switch points
without gold LID. The tagger is trained once and then FROZEN: the sentiment
model never back-propagates into it.

Usage:
    python3 lib/lid_tagger.py --model_name xlm-roberta-base --out_dir checkpoints/lid_xlmr
    python3 lib/lid_tagger.py --predict_only --out_dir checkpoints/lid_xlmr
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import LID2ID, LID_TAGS, load_split  # noqa: E402


class TagDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.rows = df.to_dict("records")
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["tokens"], is_split_into_words=True, truncation=True, max_length=self.max_length)
        word_ids = enc.word_ids()
        labels, prev = [], None
        for wid in word_ids:                       # label only the first sub-word of every word
            if wid is None or wid == prev:
                labels.append(-100)
            else:
                labels.append(LID2ID[r["tags"][wid].lower()])
            prev = wid
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels, "word_ids": word_ids, "id": r["id"], "n_words": len(r["tokens"])}


def collate(batch, pad_id):
    T = max(len(b["input_ids"]) for b in batch)
    def pad(key, val):
        return torch.tensor([b[key] + [val] * (T - len(b[key])) for b in batch])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100), "meta": batch}


@torch.no_grad()
def predict(model, loader, device):
    """Return dict id -> list of predicted tags (one per word; truncated words default to 'o')."""
    model.eval()
    out = {}
    for batch in tqdm(loader, desc="predict", leave=False):
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        pred = logits.argmax(-1).cpu().tolist()
        for b, p in zip(batch["meta"], pred):
            tags = ["o"] * b["n_words"]
            prev = None
            for j, wid in enumerate(b["word_ids"]):
                if wid is not None and wid != prev:
                    tags[wid] = LID_TAGS[p[j]]
                prev = wid
            out[b["id"]] = tags
    return out


def score(pred: dict, df) -> dict:
    y_true, y_pred = [], []
    for _, r in df.iterrows():
        y_true += [t.lower() for t in r["tags"]]
        y_pred += pred[r["id"]]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    f1s = []
    for c in LID_TAGS:
        tp = ((y_pred == c) & (y_true == c)).sum(); fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0; r_ = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r_ / (p + r_) if p + r_ else 0.0)
    return {"token_accuracy": float((y_true == y_pred).mean()), "macro_f1": float(np.mean(f1s)),
            **{f"f1_{c}": float(f) for c, f in zip(LID_TAGS, f1s)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--out_dir", default="checkpoints/lid_xlmr")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_train_samples", type=int, default=None, help="for smoke tests")
    ap.add_argument("--predict_only", action="store_true", help="load <out_dir> and only write predictions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(s) for s in ["train", "dev", "test"]}
    if args.max_train_samples:
        splits["train"] = splits["train"].sample(n=args.max_train_samples, random_state=args.seed)

    src = str(out_dir) if args.predict_only else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModelForTokenClassification.from_pretrained(
        src, num_labels=len(LID_TAGS), id2label=dict(enumerate(LID_TAGS)), label2id=LID2ID).to(device)
    pad_id = tokenizer.pad_token_id
    loaders = {s: DataLoader(TagDataset(df, tokenizer, args.max_length), batch_size=args.batch_size,
                             shuffle=(s == "train" and not args.predict_only),
                             collate_fn=lambda b: collate(b, pad_id)) for s, df in splits.items()}

    if not args.predict_only:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total = len(loaders["train"]) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        best = -1.0
        for ep in range(args.epochs):
            model.train()
            for batch in tqdm(loaders["train"], desc=f"LID epoch {ep + 1}/{args.epochs}"):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                                 labels=batch["labels"].to(device)).loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            dev_scores = score(predict(model, loaders["dev"], device), splits["dev"])
            print(f"[epoch {ep + 1}] dev: {json.dumps(dev_scores)}")
            if dev_scores["macro_f1"] > best:
                best = dev_scores["macro_f1"]
                model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
        model = AutoModelForTokenClassification.from_pretrained(out_dir).to(device)

    metrics = {}
    for s, df in splits.items():
        pred = predict(model, loaders[s], device)
        metrics[s] = score(pred, df)
        pd.DataFrame({"id": list(pred), "pred_tags": [str(v) for v in pred.values()]}).to_csv(
            out_dir / f"{s}_pred_tags.csv", index=False)
        print(f"[LID {s:5s}] {json.dumps(metrics[s])}")
    (out_dir / "lid_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[+] predicted tags written to {out_dir}/{{train,dev,test}}_pred_tags.csv")


if __name__ == "__main__":
    main()
