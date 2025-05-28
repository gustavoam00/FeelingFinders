import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


model_options = ["vinai/bertweet-large", "microsoft/deberta-v3-large", "facebook/bart-large"]
model_names = ["bertweet", "deberta", "bart"]
MODEL = "bert-base-uncased" #model_options[0]
NAME = "bert" #model_names[0]

TRAINING_DATA = "./data/training.csv"
TEST_DATA = "./data/test.csv"

OUTPUT_NAME = f"./results/{NAME}_predictions.csv"
LOG_FILE = f".results/metrics_log.txt"
SAVE_DIR = f"./saved_models/{NAME}"
LOGITS_DIR = f"./saved_logits/{NAME}"

#config
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOGITS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_NAME), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

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
train_df['clean_sentence'] = train_df['sentence'].astype(str)

test_df['clean_sentence'] = test_df['sentence'].astype(str)

train_texts = train_df["clean_sentence"].tolist()
train_labels = train_df["label_id"].tolist()
test_text = test_df['clean_sentence'].tolist()

#model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
).to(device)

test_encodings = tokenizer(test_text, truncation=True, max_length=128, padding=True, return_tensors='pt')

test_dataset = torch.utils.data.TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask']
)
test_loader = DataLoader(test_dataset, batch_size=64)

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

label_preds = [id2label[int(i)] for i in predicted_labels]

test_output = test_df[['id']].copy()
test_output['label'] = label_preds
filename = OUTPUT_NAME
test_output.to_csv(filename, index=False)
