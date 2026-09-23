import os, torch, math, argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.transformer import Transformer

parser = argparse.ArgumentParser()
parser.add_argument("--sent", type=str, required=True, help="Input sentence for sentiment prediction")
args = parser.parse_args()

checkpoint_base_url = "/nlsasfs/home/aidrive/dassuv/models"
model_name = "mistralai-7B-instruct-v0.3"

MODEL_PATH = os.path.join(checkpoint_base_url, model_name)

FINETUNED_CHECKPOINT = ("checkpoints/mistral_sentiment_last2.pt")
MAX_LENGTH = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = MistralTokenizer.from_file(os.path.join(MODEL_PATH, "tokenizer.model.v3"))

print("Loading base model...")
model = Transformer.from_folder(MODEL_PATH)
model = model.to(device)
print("Base model loaded.")
hidden_size = model.args.dim

model_dtype = next(model.parameters()).dtype

print("Hidden size:", hidden_size)
print("Model dtype:", model_dtype)

classifier = nn.Linear(hidden_size, 3).to(device=device, dtype=model_dtype)
print("Loading fine-tuned checkpoint...")

checkpoint = torch.load(FINETUNED_CHECKPOINT, map_location=device, weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])

classifier.load_state_dict(checkpoint["classifier_state_dict"])

label2id = checkpoint["label2id"]
id2label = checkpoint["id2label"]

id2label = {int(k): v for k, v in id2label.items()}
print("Fine-tuned checkpoint loaded.")
print("Labels:", id2label)

model.eval()
classifier.eval()

def tokenize(sentence):

    tokens = (tokenizer .instruct_tokenizer .tokenizer .encode(sentence, bos=True, eos=False))
    tokens = tokens[:MAX_LENGTH]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    return input_ids, tokens

def get_hidden_representation(input_ids):
    seqlens = [input_ids.shape[0]]
    hidden_states = model.forward_partial(input_ids=input_ids, seqlens=seqlens)
    return hidden_states

def predict(sentence):
    input_ids, token_ids = tokenize(sentence)
    with torch.no_grad():
        hidden_states = (get_hidden_representation(input_ids))
        sentence_representation = (hidden_states[-1])
        sentence_representation = (sentence_representation.unsqueeze(0))
        logits = classifier(sentence_representation)
        probabilities = torch.softmax(logits.float(), dim=-1)
        prediction_id = (torch.argmax(probabilities, dim=-1).item())
    prediction = id2label[prediction_id]
    return (prediction, probabilities[0].cpu(), token_ids)

def get_last_layer_attention(sentence):
    input_ids, token_ids = tokenize(sentence)
    last_layer = list(model.layers.values())[-1]
    captured = {}
    def capture_last_layer_input(module, args):
        captured["hidden"] = args[0].detach()
    hook = last_layer.register_forward_pre_hook(capture_last_layer_input)
    with torch.no_grad():
        _ = get_hidden_representation(input_ids)
    hook.remove()
    hidden = captured["hidden"]
    if hidden.dim() == 3:
        hidden = hidden.squeeze(0)
    seq_len = len(token_ids)
    hidden_norm = last_layer.attention_norm(hidden)
    attention = last_layer.attention
    q = attention.wq(hidden_norm)
    k = attention.wk(hidden_norm)
    n_heads = model.args.n_heads
    n_kv_heads = getattr(model.args, "n_kv_heads", n_heads)
    head_dim = model.args.dim // n_heads
    q = q.view(seq_len, n_heads, head_dim)
    k = k.view(seq_len, n_kv_heads, head_dim)
    freqs_cis = model.freqs_cis[:seq_len]

    def apply_rope(x, freqs):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = freqs[:, None, :]
        x_rotated = x_complex * freqs
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)

    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    if n_heads != n_kv_heads:
        repeat_factor = (n_heads // n_kv_heads)
        k = torch.repeat_interleave(k, repeats=repeat_factor, dim=1)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    return token_ids, attention_weights.cpu()

def get_token_labels(token_ids):
    labels = []
    for token_id in token_ids:
        text = (tokenizer.instruct_tokenizer.tokenizer.decode([token_id]))
        if text == "":
            text = "<BOS>"
        text = text.replace(" ", "▁")
        labels.append(text)
    return labels

def plot_token_attention(sentence):
    token_ids, attention_weights = get_last_layer_attention(sentence)
    token_labels = get_token_labels(token_ids)
    attention_weights = (attention_weights.detach().float().cpu())
    num_heads = attention_weights.shape[0]
    seq_len = attention_weights.shape[1]
    print("\nNumber of attention heads:", num_heads)
    print("Sequence length:", seq_len)
    os.makedirs("plots", exist_ok=True)
    ncols = 4
    nrows = math.ceil(num_heads / ncols)
    subplot_width = max(5, seq_len * 0.35)
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_width * ncols, 3 * nrows), constrained_layout=True)
    if num_heads == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for head in range(num_heads):
        ax = axes[head]
        token_attention = attention_weights[head, -1]
        token_attention = (token_attention.numpy())
        print(f"\n===== Attention Head {head} =====")
        for token, score in zip(token_labels, token_attention):
            print(f"{token:20s} -> {score:.6f}")
        heatmap = token_attention.reshape(1,-1)
        image = ax.imshow(heatmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"Head {head}", fontsize=11)
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([0])
        ax.set_yticklabels(["Final token"], fontsize=8)

    for i in range(num_heads, len(axes)):
        axes[i].axis("off")

    cbar = fig.colorbar(image, ax=axes[:num_heads], orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=11)
    fig.suptitle("Token-wise Attention of Final Token " "— Last Mistral Layer", fontsize=16)
    save_path = ("plots/last_layer_token_attention_all_heads.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nAttention heatmap saved to: {save_path}")

sentence = args.sent
prediction, probabilities, token_ids = predict(sentence)

print("\nSentence:")
print(sentence)

print("\nPrediction:")
print(prediction)

print("\nProbabilities:")

for i, probability in enumerate(probabilities):
    print(f"{id2label[i]:10s}: {probability.item():.4f}")

plot_token_attention(sentence)