"""Build and load the SentiMix splits used by Experiment 2.

The shared `get_sentimix.py` (master branch) only builds `train.csv`. This
module re-uses its `process_dataset` function, unchanged, to build the `dev`
and `test` splits in exactly the same CSV format:

    id, sentence (python-list of tokens), tags (python-list of LID tags), sentiment

The SentiMix test file on HuggingFace is unlabelled; its sentiment labels live
in `test_labels_hinglish.txt` (columns: Uid, Sentiment). We join them by id.
"""

import ast
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))          # so `import get_sentimix` works from anywhere

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LID_TAGS = ["hin", "eng", "o"]              # SentiMix word-level language tags (lower-cased by get_sentimix.py)
LID2ID = {t: i for i, t in enumerate(LID_TAGS)}
# SentiMix also contains a rare `emt` (emoticon) tag (~0.1% of tokens); we fold it into `o`
# so the tagger and the switch logic work with the 3-way EN / HI / O scheme of the task.
TAG_NORMALISE = {"emt": "o"}


def dataset_dir() -> Path:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    data_home = os.getenv("DATA_HOME")
    datacard = os.getenv("HG_DATACARD")
    if data_home is None or datacard is None:
        raise ValueError("DATA_HOME / HG_DATACARD must be defined in .env (see .env.example)")
    return REPO_ROOT / data_home / datacard.split("/")[-1]


def build_splits(force: bool = False) -> None:
    """Create train.csv / dev.csv / test.csv via the shared processing script."""
    import get_sentimix                                   # shared, do not modify
    ddir = dataset_dir()

    if force or not (ddir / "train.csv").exists():
        get_sentimix.process_dataset("HG_DATACARD", "train", is_roman=True)
    if force or not (ddir / "dev.csv").exists():
        get_sentimix.process_dataset("HG_DATACARD", "dev", is_roman=True)
    if force or not (ddir / "test.csv").exists():
        # raw file is `Hindi_test_unalbelled_conll_updated.txt` -> prefix "Hindi_test_"
        get_sentimix.process_dataset("HG_DATACARD", "Hindi_test", is_roman=True)
        unlabelled = pd.read_csv(ddir / "Hindi_test.csv")
        labels = pd.read_csv(ddir / "test_labels_hinglish.txt").rename(
            columns={"Uid": "id", "Sentiment": "sentiment"})
        labels["sentiment"] = labels["sentiment"].str.strip().str.lower()
        merged = unlabelled.drop(columns=["sentiment"]).merge(labels, on="id", how="inner")
        merged = merged[["id", "sentence", "tags", "sentiment"]]
        merged.to_csv(ddir / "test.csv", index=False)
        print(f"[SUCCESS] Dataset saved to {ddir / 'test.csv'}  ({len(merged)} labelled rows)")


def load_split(split: str) -> pd.DataFrame:
    """Return a DataFrame with columns id, tokens (list[str]), tags (list[str]), label (int)."""
    path = dataset_dir() / f"{split}.csv"
    if not path.exists():
        build_splits()
    df = pd.read_csv(path)
    df["tokens"] = df["sentence"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(lambda ts: [TAG_NORMALISE.get(t.lower(), t.lower()) for t in ast.literal_eval(ts)])
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[df["sentiment"].isin(LABEL2ID)].reset_index(drop=True)
    df["label"] = df["sentiment"].map(LABEL2ID)
    # drop empty / malformed rows; the shared parser emits a few rows with a non-numeric id
    # (a token literally spelled "meta") and one exact duplicate per training split
    df = df[df["tokens"].apply(len) > 0]
    df = df[pd.to_numeric(df["id"], errors="coerce").notna()]
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    assert (df["tokens"].apply(len) == df["tags"].apply(len)).all(), "token/tag length mismatch"
    return df[["id", "tokens", "tags", "sentiment", "label"]]


def load_predicted_tags(split: str, pred_dir: Path) -> dict:
    """Map id -> list[str] of LID tags predicted by lib/lid_tagger.py."""
    df = pd.read_csv(Path(pred_dir) / f"{split}_pred_tags.csv")
    return {row["id"]: ast.literal_eval(row["pred_tags"]) for _, row in df.iterrows()}


if __name__ == "__main__":
    build_splits(force="--force" in sys.argv)
    for s in ["train", "dev", "test"]:
        d = load_split(s)
        print(f"{s:5s}: {len(d):6d} rows | label dist: {d['sentiment'].value_counts().to_dict()}")
