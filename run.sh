#!/bin/bash
# Experiment 2 (Switch Point + Sentiment) - full comparison on SentiMix Hi-En.
# Usage: bash run.sh [MODEL_NAME]      (default: xlm-roberta-base; any HF encoder id works)
# For SLURM clusters, add the usual #SBATCH header and `source venv/bin/activate` here.
set -e
MODEL=${1:-xlm-roberta-base}
TAG=$(basename "$MODEL")
mkdir -p logs results/exp2

# 0. dataset splits (train/dev/test) via the shared get_sentimix.py
python3 lib/data.py

# 1. token-level LID tagger (trained once, then frozen) -> checkpoints/lid_${TAG}/{split}_pred_tags.csv
python3 lib/lid_tagger.py --model_name "$MODEL" --out_dir "checkpoints/lid_${TAG}" 2>&1 | tee "logs/lid_${TAG}.log"

# 2. sentiment: baseline pooling vs binary switch embedding vs distance-to-switch embedding
for SRC in gold predicted; do
  for POOL in none binary distance; do
    python3 main.py --model_name "$MODEL" --pooling "$POOL" --switch_source "$SRC" \
      --pred_tags_dir "checkpoints/lid_${TAG}" --results_dir results/exp2 \
      2>&1 | tee "logs/exp2_${TAG}_${POOL}_${SRC}.log"
  done
done

# 3. shared evaluation (macro P / R / F1 + accuracy) over every prediction file
cd results/exp2 && python3 ../../evaluate_all.py --ground ground.csv --pred-glob "*.csv" --out evaluation_results.csv
