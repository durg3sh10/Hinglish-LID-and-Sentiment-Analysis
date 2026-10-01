# PROGRAM GATEWAY - Experiment 2 (Switch Point + Sentiment)
#
# Compare sentence-level sentiment on SentiMix Hi-En with attention pooling that is
#   --pooling none      : plain attention pooling            (baseline, no switch information)
#   --pooling binary    : + learned embedding of the binary switch flag
#   --pooling distance  : + learned embedding of bucketed distance-to-nearest-switch (0,1,2,3+)
# Switch points come from gold SentiMix LID tags (--switch_source gold) or from the
# frozen tagger trained by lib/lid_tagger.py (--switch_source predicted).
import argparse

from lib.model import POOLING_MODES
from lib.train import train_and_evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2: switch-aware sentiment classification")
    p.add_argument("--model_name", default="xlm-roberta-base", help="HF encoder id or local path")
    p.add_argument("--pooling", default="binary", choices=POOLING_MODES)
    p.add_argument("--switch_source", default="gold", choices=["gold", "predicted"])
    p.add_argument("--pred_tags_dir", default="checkpoints/lid_xlmr", help="output dir of lib/lid_tagger.py")
    p.add_argument("--naive_switch", action="store_true", help="treat 'o' tokens as a language when detecting switches")
    p.add_argument("--switch_subword", default="all", choices=["all", "first"], help="how a word's switch feature is spread over its sub-words")
    p.add_argument("--switch_dim", type=int, default=16, help="size of e_i^sw (8-16 recommended)")
    p.add_argument("--scorer", default="linear", choices=["linear", "mlp"], help="attention score function")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5, help="encoder learning rate")
    p.add_argument("--head_lr", type=float, default=1e-3, help="pooling + classifier learning rate")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--lora", action="store_true", help="LoRA on the encoder (for 7B-class models)")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
                   help="comma-separated LoRA target modules (decoder naming; same set as exp1)")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--max_eval_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--results_dir", default="results/exp2")
    p.add_argument("--run_name", default=None)
    p.add_argument("--save_model", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
