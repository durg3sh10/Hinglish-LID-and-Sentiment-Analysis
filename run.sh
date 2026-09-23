#!/bin/bash
#SBATCH --job-name=anecdotal_experiment_0
#SBATCH --partition=priogp
#SBATCH --nodelist=scn83-10g
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
cd $SLURM_SUBMIT_DIR
source ~/research/sentiment_analysis/venv/bin/activate
cd ~/research/sentiment_analysis
python -u anecdotal0_inference.py --sent "very beautiful eyes yarrrrr"