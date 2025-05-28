import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

start_time = time.time()

model_options = ["vinai/bertweet-large", "microsoft/deberta-v3-large", "facebook/bart-large", "bert-base-uncased"]
model_names = ["bertweet", "deberta", "bart", "bert"]
MODEL = model_options[1]
NAME = model_names[1]

BACKTRANSLATED_POS = "./data/backtranslated_pos.csv"
BACKTRANSLATED_NEG = "./data/backtranslated_neg.csv"
CONTEXT_DATA = "./data/contextual_aug_1.csv"

training_sets = ["./data/training.csv", "./data/TRAIN_P_SOFT.csv", "./data/TRAIN_P_SOFTPLUS.csv", "./data/TRAIN_P_HARD.csv"]
TRAINING_DATA = training_sets[0]
test_sets = ["./data/test.csv", "./data/TEST_P_SOFT.csv", "./data/TEST_P_SOFTPLUS.csv", "./data/TEST_P_HARD.csv"]
TEST_DATA = test_sets[0]

OUTPUT_NAME = f"./results/{NAME}_predictions.csv"
LOG_FILE = f"./results/metrics_log.txt"
LOGITS_DIR = f"./saved_logits/{NAME}"

#config
SEED = 42
FREEZE_NUM = 9
LR =  5e-6
EPOCHS = 3

os.makedirs(LOGITS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_NAME), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

tqdm.pandas()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
#--------------------------------------------------------------------------#

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

#--------------------------------------------------------------------------#

#data
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

train_df = pd.read_csv(TRAINING_DATA)
test_df = pd.read_csv(TEST_DATA)

pos_df = pd.read_csv(BACKTRANSLATED_POS)
neg_df = pd.read_csv(BACKTRANSLATED_NEG)
context_df = pd.read_csv(CONTEXT_DATA)

train_df["label_id"] = train_df["label"].map(label2id)
pos_df['label_id'] = pos_df['label'].map(label2id)
neg_df['label_id'] = neg_df['label'].map(label2id)
context_df['label_id'] = context_df['label'].map(label2id)

train_texts = train_df['sentence'].tolist()
train_labels = train_df['label_id'].tolist()
test_text = test_df['sentence'].tolist()

#--------------------------------------------------------------------------#

#model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
).to(device)

# freeze a num of layers sequentially 
if MODEL != "facebook/bart-large":
    base_model = getattr(model, model.base_model_prefix, model.base_model)
    if FREEZE_NUM != 0:
        for name, param in base_model.embeddings.named_parameters():
            param.requires_grad = False

        for i in range(FREEZE_NUM):
            for param in base_model.encoder.layer[i].parameters():
                param.requires_grad = False
else:
    if FREEZE_NUM != 0:
        for param in model.model.encoder.embed_tokens.parameters():
            param.requires_grad = False

        for i in range(FREEZE_NUM):
            for param in model.model.encoder.layers[i].parameters():
                param.requires_grad = False

#tokenizes data and loads it
train_encodings = tokenizer(train_texts, truncation=True, max_length=128, padding=True)
test_encodings = tokenizer(test_text, truncation=True, max_length=128, padding=True, return_tensors='pt')

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = torch.utils.data.TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask']
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

#--------------------------------------------------------------------------#

#train
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights) #balances classes using weighted loss
optimizer = AdamW(model.parameters(), lr=LR)

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

#--------------------------------------------------------------------------#

# Predict test
model.eval()
predicted_labels = []

# Runs inference on the test set
with torch.no_grad():
     for batch in tqdm(test_loader, desc="Running inference on test set"):
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = torch.argmax(outputs.logits, dim=1)
        predicted_labels.extend(preds.cpu().numpy())

label_preds = [id2label[int(i)] for i in predicted_labels]
test_output = test_df[['id']].copy()
test_output['label'] = label_preds
filename = OUTPUT_NAME
test_output.to_csv(filename, index=False)

end_time = time.time()

with open(LOG_FILE, "a") as f:
    f.write(f"-----------------------------------------\n")
    f.write(f"{NAME}\n")
    f.write(f"Total time: {end_time-start_time}\n")
    
torch.cuda.empty_cache()
