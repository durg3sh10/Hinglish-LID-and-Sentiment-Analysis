# imports 
import os, torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.transformer import Transformer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

print("working 1")

# configurations
checkpoint_base_url = "/nlsasfs/home/aidrive/dassuv/models"
model_name = "mistralai-7B-instruct-v0.3"
csv_path="/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/dataset/SentiMix/train.csv"
layers_to_tune = 2
text_column = "sentence"
label_column = "sentiment"
max_length = 256
batch_size = 1
epochs = 3
learning_rate = 2e-5
PAD_TOKEN_ID = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load the local checkpoints
MODEL_NAME = "mistralai-7B-instruct-v0.3"
MODEL_PATH = os.path.join(checkpoint_base_url, MODEL_NAME)
tokenizer = MistralTokenizer.from_file(os.path.join(MODEL_PATH, "tokenizer.model.v3"))
model = Transformer.from_folder(MODEL_PATH)

print("working 2")

# freeze every layer
for param in model.parameters():
    param.requires_grad = False

# unfreeze last few layers for train
for layer in list(model.layers.values())[-layers_to_tune:]:
    for param in layer.parameters():
        param.requires_grad = True

# show number of trainable parameters
trainable, total = 0, 0
for name, param in model.named_parameters():
    total += param.numel()
    if param.requires_grad:
        trainable += param.numel()
        
print(f"\nTrainable parameters: {trainable:,}")
print(f"Total parameters: {total:,}")
print(f"Trainable: {100 * trainable / total:.2f}%")

# tokenization experiments
sentence = "Tumhara exam khatam ho gaya? kal project start kare."
tokens = tokenizer.instruct_tokenizer.tokenizer.encode(sentence, bos=True, eos=False)
print("Token IDs:")
print(tokens)
print("\nIndividual tokens:")
for token_id in tokens:
    token_text = tokenizer.instruct_tokenizer.tokenizer.decode([token_id])
    print(f"{token_id:8d} -> {repr(token_text)}")


df = pd.read_csv(csv_path)
print("\nDataset shape:", df.shape)
print(df.head())
df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}

df[label_column] = (df[label_column] .str.lower() .str.strip() .map(label2id))
df = df.dropna(subset=[label_column]).reset_index(drop=True)
df[label_column] = (df[label_column].astype(int))
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df[label_column])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df[label_column])

print("\nDataset splits")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

class SentimentDataset(Dataset):
    def __init__( self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentence = str(row[text_column])
        label = int(row[label_column])
        tokens = (self.tokenizer .instruct_tokenizer .tokenizer .encode(sentence, bos=True, eos=False))
        tokens = tokens[:self.max_length]
        return {"input_ids": torch.tensor(tokens, dtype=torch.long),"label": torch.tensor(label, dtype=torch.long)}


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.tensor([len(x) for x in input_ids], dtype=torch.long)
    max_len = max(len(x) for x in input_ids)
    padded = torch.full((len(input_ids), max_len), PAD_TOKEN_ID, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded[i, :len(ids)] = ids

    return {
        "input_ids": padded,
        "lengths": lengths,
        "labels": labels
    }

train_dataset = SentimentDataset(train_df, tokenizer, max_length)
val_dataset = SentimentDataset(val_df, tokenizer, max_length)
test_dataset = SentimentDataset(test_df, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print("\nModel args:")
print(model.args)
hidden_size = model.args.dim
model_dtype = next(model.parameters()).dtype
print("Mistral dtype:", model_dtype)
print("Hidden dimension:", hidden_size)
classifier = nn.Linear(hidden_size, 3).to(device=device, dtype=model_dtype)
print("Classifier dtype:", classifier.weight.dtype)

model = model.to(device)
trainable_model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_model_parameters + list(classifier.parameters()), lr=learning_rate, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

def get_hidden_representation(model, input_ids, lengths):
    token_sequences = []
    for i, length in enumerate(lengths.tolist()):
        token_sequences.append(input_ids[i, :length])

    flat_tokens = torch.cat(token_sequences, dim=0)
    seqlens = lengths.tolist()
    hidden_states = model.forward_partial(input_ids=flat_tokens, seqlens=seqlens,)
    return hidden_states

def forward_classifier(input_ids, lengths):
    hidden_states = get_hidden_representation(model, input_ids, lengths)
    last_indices = torch.cumsum(lengths, dim=0) - 1
    last_indices = last_indices.to(hidden_states.device)
    sentence_representation = hidden_states[last_indices]
    logits = classifier(sentence_representation)
    return logits

# validity check
batch = next(iter(train_loader))
input_ids = batch["input_ids"].to(device)
lengths = batch["lengths"].to(device)
hidden = get_hidden_representation(model, input_ids, lengths)
print(hidden.shape)
print(hidden.requires_grad)
print(hidden.grad_fn)

# training
for epoch in range(epochs):
    model.train()
    classifier.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = forward_classifier(input_ids, lengths) # Forward
        loss = criterion(logits, labels) # Loss
        loss.backward() # Backpropagation
        torch.nn.utils.clip_grad_norm_(trainable_model_parameters + list(classifier.parameters()), max_norm=1.0) # Gradient clipping (optional)
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = (total_loss / len(train_loader))
    print( f"\nEpoch {epoch + 1} - Training Loss: - {avg_loss:.4f}") # print average loss

os.makedirs("checkpoints", exist_ok=True)
torch.save({
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "label2id": label2id,
        "id2label": id2label},
    "checkpoints/"
    "mistral_sentiment_last2.pt")

print("\nModel saved.")