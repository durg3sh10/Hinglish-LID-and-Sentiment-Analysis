"""Post-hoc analysis of Experiment 2 predictions.

Reads every prediction CSV in <results_dir> (id, predictions) plus the test split and reports
macro-F1 (shared formula from evaluate_all.py) overall and per switch-count bucket, so we can
see WHERE switch information helps. Every SentiMix test sentence contains >= 1 switch, so the
buckets are on the number of switches per sentence.

    python3 lib/analyze.py --results_dir results/exp2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from evaluate_all import macro_metrics          # noqa: E402
from lib.data import load_split                 # noqa: E402
from lib.switch import switch_points            # noqa: E402

BUCKETS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 10**9, "10+")]


def bucket(n):
    for lo, hi, name in BUCKETS:
        if lo <= n <= hi:
            return name
    return "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/exp2")
    ap.add_argument("--naive_switch", action="store_true")
    args = ap.parse_args()

    test = load_split("test")
    test["n_switches"] = test["tags"].apply(lambda t: sum(switch_points(t, ignore_other=not args.naive_switch)))
    test["bucket"] = test["n_switches"].apply(bucket)
    print("test sentences per switch-count bucket:", test["bucket"].value_counts().sort_index().to_dict())
    print(f"mean switches / sentence: {test['n_switches'].mean():.2f}\n")

    rows = []
    for f in sorted(Path(args.results_dir).glob("*.csv")):
        if f.name in ("ground.csv",) or "evaluation" in f.name:
            continue
        pred = pd.read_csv(f)
        if "id" not in pred.columns or "predictions" not in pred.columns:   # summary / analysis tables
            continue
        pred = pred.rename(columns={"predictions": "pred"})
        m = test.merge(pred, on="id", how="inner")
        row = {"model": f.stem, "n": len(m), "macro_f1": macro_metrics(m["sentiment"], m["pred"])["macro_f1"],
               "accuracy": macro_metrics(m["sentiment"], m["pred"])["accuracy"]}
        for _, _, name in BUCKETS:
            sub = m[m["bucket"] == name]
            row[f"f1@{name}"] = macro_metrics(sub["sentiment"], sub["pred"])["macro_f1"] if len(sub) else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out = Path(args.results_dir) / "switch_bucket_analysis.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
