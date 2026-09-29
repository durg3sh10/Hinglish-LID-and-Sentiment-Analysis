"""Aggregate Experiment 2 runs: mean +- std over seeds per (pooling, switch_source, scorer).

    python3 lib/summarize.py --results_dir results/exp2
Writes <results_dir>/summary.csv and prints a markdown table.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/exp2")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.results_dir).glob("*_metrics.json")):
        m = json.load(open(f)); a = m["args"]; t = m["test"]
        rows.append({"model": a["model_name"].split("/")[-1], "pooling": a["pooling"],
                     "switch_source": a["switch_source"] if a["pooling"] in ("binary", "distance") else "-",
                     "scorer": a["scorer"], "seed": a["seed"], "dev_f1": m["best_dev_macro_f1"],
                     "test_acc": t["accuracy"], "test_P": t["macro_precision"], "test_R": t["macro_recall"],
                     "test_F1": t["macro_f1"]})
    df = pd.DataFrame(rows)
    keys = ["model", "pooling", "switch_source", "scorer"]
    # `none` pooling ignores the switch source, so its gold/predicted runs are identical: keep one per seed
    df = df.drop_duplicates(subset=keys + ["seed"], keep="first")
    g = df.groupby(keys)
    summary = g.agg(seeds=("seed", "count"), dev_f1=("dev_f1", "mean"),
                    test_acc=("test_acc", "mean"), test_P=("test_P", "mean"), test_R=("test_R", "mean"),
                    test_F1_mean=("test_F1", "mean"), test_F1_std=("test_F1", "std")).reset_index()
    summary["test_F1_std"] = summary["test_F1_std"].fillna(0.0)
    order = {"none": 0, "cls": 1, "mean": 2, "binary": 3, "distance": 4}
    summary = summary.sort_values(["model", "scorer", "pooling", "switch_source"],
                                  key=lambda c: c.map(order) if c.name == "pooling" else c).reset_index(drop=True)
    summary.to_csv(Path(args.results_dir) / "summary.csv", index=False)

    print("| model | pooling | switch source | scorer | seeds | dev F1 | test acc | test P | test R | test F1 (mean +- std) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        print(f"| {r.model} | {r.pooling} | {r.switch_source} | {r.scorer} | {r.seeds} | {r.dev_f1:.4f} | "
              f"{r.test_acc:.4f} | {r.test_P:.4f} | {r.test_R:.4f} | {r.test_F1_mean:.4f} +- {r.test_F1_std:.4f} |")
    print(f"\nper-run detail:\n{df.sort_values(keys + ['seed']).to_string(index=False, float_format=lambda x: f'{x:.4f}')}")


if __name__ == "__main__":
    main()
