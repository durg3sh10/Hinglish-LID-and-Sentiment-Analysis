"""Switch-aware sentiment model.

    h_i        = encoder(x)_i                              (any HF AutoModel)
    a_i        = softmax_i( W_h h_i + W_s e_i^sw + b )      (switch-biased attention pooling)
    s          = sum_i a_i h_i                              (sentiment vector)
    logits     = W_c s + b_c

e_i^sw is a small learned embedding (default 16-d) of either
  * binary   : switch flag in {0, 1}
  * distance : bucketed distance to the nearest switch in {0, 1, 2, 3+}
The `none` mode drops the W_s term (plain attention pooling baseline).
`cls` and `mean` pooling are included as extra reference points.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

POOLING_MODES = ["none", "binary", "distance", "cls", "mean"]


class SwitchAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, mode: str, switch_dim: int = 16,
                 n_distance_buckets: int = 4, scorer: str = "linear", scorer_hidden: int = 128):
        super().__init__()
        assert mode in POOLING_MODES, mode
        self.mode = mode
        if mode == "binary":
            self.switch_emb = nn.Embedding(2, switch_dim)
        elif mode == "distance":
            self.switch_emb = nn.Embedding(n_distance_buckets, switch_dim)
        else:
            self.switch_emb = None

        if scorer == "linear":                       # exactly a_i = softmax(W_h h_i + W_s e_i + b)
            self.w_h = nn.Linear(hidden_size, 1)
            self.w_s = nn.Linear(switch_dim, 1, bias=False) if self.switch_emb is not None else None
            self.v = None
        elif scorer == "mlp":                        # a_i = softmax(v^T tanh(W_h h_i + W_s e_i + b))
            self.w_h = nn.Linear(hidden_size, scorer_hidden)
            self.w_s = nn.Linear(switch_dim, scorer_hidden, bias=False) if self.switch_emb is not None else None
            self.v = nn.Linear(scorer_hidden, 1, bias=False)
        else:
            raise ValueError(scorer)

    def forward(self, h: torch.Tensor, pool_mask: torch.Tensor, switch_feat: torch.Tensor):
        """h: [B,T,H]; pool_mask: [B,T] (1 = pool over this token); switch_feat: [B,T] long."""
        if self.mode == "cls":
            return h[:, 0], None
        if self.mode == "mean":
            m = pool_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0), None

        scores = self.w_h(h)                                        # [B,T,1] or [B,T,S]
        if self.switch_emb is not None:
            scores = scores + self.w_s(self.switch_emb(switch_feat))
        if self.v is not None:
            scores = self.v(torch.tanh(scores))
        scores = scores.squeeze(-1).float()                          # [B,T]
        scores = scores.masked_fill(pool_mask == 0, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1).to(h.dtype)             # [B,T]
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)          # [B,H]
        return pooled, attn


class SwitchAwareSentimentModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, pooling: str = "binary",
                 switch_dim: int = 16, scorer: str = "linear", dropout: float = 0.1,
                 lora: bool = False, lora_r: int = 8, load_in_4bit: bool = False):
        super().__init__()
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            kwargs["device_map"] = {"": 0}
        self.encoder = AutoModel.from_pretrained(model_name, **kwargs)
        if lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if load_in_4bit:
                self.encoder = prepare_model_for_kbit_training(self.encoder)
            self.encoder = get_peft_model(self.encoder, LoraConfig(
                r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
                task_type="FEATURE_EXTRACTION"))
            self.encoder.print_trainable_parameters()
        hidden = self.encoder.config.hidden_size
        self.pooling = SwitchAttentionPooling(hidden, pooling, switch_dim=switch_dim, scorer=scorer)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, pool_mask, switch_feat, labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled, attn = self.pooling(h, pool_mask, switch_feat)
        logits = self.classifier(self.dropout(pooled)).float()
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "attention": attn}
