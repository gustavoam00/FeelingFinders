import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)


tqdm.pandas()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
#data
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

train_df = pd.read_csv("./data/training.csv")
test_df = pd.read_csv("./data/test.csv")

train_df["label_id"] = train_df["label"].map(label2id)
train_df['clean_sentence'] = train_df['sentence'].astype(str).apply(preprocess)
test_df['clean_sentence'] = test_df['sentence'].astype(str).apply(preprocess)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["clean_sentence"].tolist(),
    train_df["label_id"].tolist(),
    test_size=0.1,
    stratify=train_df["label_id"],
    random_state=42
)
test_text = test_df['clean_sentence'].tolist()

#parameters
freeze_num = 9
lr = 5e-6
epochs = 3
model_name = "vinai/bertweet-large" #"siebert/sentiment-roberta-large-english", "microsoft/deberta-v3-large"

#model
MODEL = model_name
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
).to(device)

base_model = getattr(model, model.base_model_prefix, model.base_model)
if freeze_num != 0:
    for name, param in base_model.embeddings.named_parameters():
        param.requires_grad = False

    for i in range(freeze_num):
        for param in base_model.encoder.layer[i].parameters():
            param.requires_grad = False

train_encodings = tokenizer(train_texts, truncation=True, max_length=128, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, max_length=128, padding=True)
test_encodings = tokenizer(test_text, truncation=True, max_length=128, padding=True, return_tensors='pt')

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)
test_dataset = torch.utils.data.TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask']
)

optimizer = AdamW(model.parameters(), lr=lr)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

#train
model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

#evaluate
model.eval()
abs_errors = []
total = 0
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)

        pred_labels = preds.cpu().tolist()
        true_labels = batch["labels"].cpu().tolist()

        for pred, true in zip(pred_labels, true_labels):
            mapped_pred = pred - 1
            mapped_true = true - 1
            abs_errors.append(abs(mapped_pred - mapped_true))
            total += 1

mean_abs_error = sum(abs_errors) / total
val_score = 0.5 * (2 - mean_abs_error)

#results
model.eval()
predicted_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predicted_labels.extend(preds.cpu().numpy())
        

label_preds = [id2label[int(i)] for i in predicted_labels]

test_output = test_df[['id']].copy()
test_output['label'] = label_preds
lr_str = f"{lr:.0e}"
filename = f"results/{model_name}_f{freeze_num}_lr{lr_str}_e{epochs}_predictions.csv"
test_output.to_csv(filename, index=False)
