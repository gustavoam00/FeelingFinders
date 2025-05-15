import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import time
import os

start_time = time.time()

model_options = ["vinai/bertweet-large", "microsoft/deberta-v3-large", "facebook/bart-large"]
model_names = ["bertweet", "deberta", "bart"]
MODEL = model_options[0]
NAME = model_names[0]

#31039 pos and 21910 neg on og data
BACKTRANSLATED_POS = "./data/backtranslated_pos.csv"
NUM_POS = 0
BACKTRANSLATED_NEG = "./data/backtranslated_neg.csv"
NUM_NEG = 0
TRAINING_DATA = "./data/training.csv"
TEST_DATA = "./data/test.csv"

OUTPUT_NAME = f"./results/{NAME}_predictions.csv"
LOG_FILE = f".results/metrics_log.txt"
SAVE_DIR = f"./saved_models/{NAME}"
LOGITS_DIR = f"./saved_logits/{NAME}"

#config
SEED = 42
EVAL_SIZE = 0.1
FREEZE_NUM = 9
LR =  5e-6
EPOCHS = 3

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOGITS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_NAME), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

#--------------------------------------------------------------------------#

def normalize(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def add_backtranslated_data(train_df, pos_num=0, neg_num=0):
    pos_df = pd.read_csv(BACKTRANSLATED_POS)
    neg_df = pd.read_csv(BACKTRANSLATED_NEG)

    pos_sampled = pos_df.sample(n=pos_num, random_state=42)
    neg_sampled = neg_df.sample(n=neg_num, random_state=42)

    aug_df = pd.concat([pos_sampled, neg_sampled], ignore_index=True)

    aug_df['clean_sentence'] = aug_df['sentence'].astype(str).apply(normalize)
    aug_df['label_id'] = aug_df['label'].map(label2id)

    if not aug_df.empty:
        aug_df['clean_sentence'] = aug_df['sentence'].astype(str).apply(normalize)
        aug_df['label_id'] = aug_df['label'].map(label2id)
        combined_df = pd.concat([train_df, aug_df], ignore_index=True)
    else:
        combined_df = train_df
        
    return combined_df

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

train_df = pd.read_csv(TRAINING_DATA)
test_df = pd.read_csv(TEST_DATA)

train_df["label_id"] = train_df["label"].map(label2id)
train_df['clean_sentence'] = train_df['sentence'].astype(str).apply(normalize)

test_df['clean_sentence'] = test_df['sentence'].astype(str).apply(normalize)

train_df = add_backtranslated_data(train_df, pos_num=NUM_POS, neg_num=NUM_NEG)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["clean_sentence"].tolist(),
    train_df["label_id"].tolist(),
    test_size=EVAL_SIZE,
    stratify=train_df["label_id"],
    random_state=SEED
)
test_text = test_df['clean_sentence'].tolist()

val_df = pd.DataFrame({
    "id": [f"val_{i}" for i in range(len(val_texts))],
    "text": val_texts,
    "label": val_labels
})


#model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
).to(device)

if NAME == "bart":
    if FREEZE_NUM != 0:
        for param in model.model.encoder.embed_tokens.parameters():
            param.requires_grad = False

        for i in range(FREEZE_NUM):
            for param in model.model.encoder.layers[i].parameters():
                param.requires_grad = False
else:            
    base_model = getattr(model, model.base_model_prefix, model.base_model)
    if FREEZE_NUM != 0:
        for name, param in base_model.embeddings.named_parameters():
            param.requires_grad = False

        for i in range(FREEZE_NUM):
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

optimizer = AdamW(model.parameters(), lr=LR)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

#train
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights) 
        
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        loss = loss_fn(outputs.logits, batch["labels"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
 
 
# Evaluate and save validation logits, labels, IDs, and texts
model.eval()
abs_errors = []
total = 0
val_logits = []
val_labels_all = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)

        pred_labels = preds.cpu().tolist()
        true_labels = batch["labels"].cpu().tolist()

        val_logits.append(outputs.logits.cpu().numpy())
        val_labels_all.append(batch["labels"].cpu().numpy())

        for pred, true in zip(pred_labels, true_labels):
            mapped_pred = pred - 1
            mapped_true = true - 1
            abs_errors.append(abs(mapped_pred - mapped_true))
            total += 1

mean_abs_error = sum(abs_errors) / total
val_score = 0.5 * (2 - mean_abs_error)

val_logits = np.concatenate(val_logits, axis=0)
val_labels_all = np.concatenate(val_labels_all, axis=0)

np.savez(f"{LOGITS_DIR}/eval_logits.npz",
         logits=val_logits,
         labels=val_labels_all,
         ids=val_df['id'].values,
         texts=val_df['text'].values)

val_df_out = pd.DataFrame({
    "id": val_df['id'].values,
    "text": val_df['text'].values,
    "label": val_labels_all
})
val_df_out.to_csv(f"{LOGITS_DIR}/eval_reference.csv", index=False)


# Predict test labels and save logits and IDs
model.eval()
predicted_labels = []
test_logits = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        test_logits.append(outputs.logits.cpu().numpy())
        preds = torch.argmax(outputs.logits, dim=1)
        predicted_labels.extend(preds.cpu().numpy())

test_logits = np.concatenate(test_logits, axis=0)
test_ids = test_df['id'].values

np.savez(f"{LOGITS_DIR}/test_logits.npz", logits=test_logits, ids=test_ids)


label_preds = [id2label[int(i)] for i in predicted_labels]

test_output = test_df[['id']].copy()
test_output['label'] = label_preds
filename = OUTPUT_NAME
test_output.to_csv(filename, index=False)

end_time = time.time()

with open(LOG_FILE, "a") as f:
    f.write(f"-----------------------------------------\n")
    f.write(f"{MODEL}\n")
    f.write(f"Validation Score: {val_score:.4f}\n")
    f.write(f"Mean Absolute Error: {mean_abs_error:.4f}\n")
    f.write(f"Total time: {end_time-start_time}\n")
    
torch.cuda.empty_cache()

# # Load val data
# val_data = np.load(f"{SAVE_DIR}/val_logits.npz")
# val_logits = val_data['logits']
# val_labels = val_data['labels']
# val_ids = val_data['ids']

# # Load test data
# test_data = np.load(f"{SAVE_DIR}/test_logits.npz")
# test_logits = test_data['logits']
# test_ids = test_data['ids']

# # Load saved model
# model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
# tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
# model.eval()