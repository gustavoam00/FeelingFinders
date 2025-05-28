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


#CONFIG
start_time = time.time()

ORIG_LABEL_NEG = 0
ORIG_LABEL_NEU = 1
ORIG_LABEL_POS = 2

#Model experts
EXPERT_CONFIGS = [
    {
        "name": "Embedder_NeuVsNegVsPos",
        "original_labels_to_keep": [ORIG_LABEL_NEG, ORIG_LABEL_NEU, ORIG_LABEL_POS],
        "relabel_map": {ORIG_LABEL_NEU: 1, ORIG_LABEL_NEG: 0, ORIG_LABEL_POS: 2},
        "num_classes": 3,
        "id2label_new": {0: "Negative", 1: "Neutral", 2: "Positive"},
        "label2id_new": {"Negative": 0, "Neutral": 1, "Positive": 2},
        "add_pos": False,
        "add_neg": False,
    },
    {
        "name": "Embedder_PosVsNotPos",
        "original_labels_to_keep": [ORIG_LABEL_NEG, ORIG_LABEL_NEU, ORIG_LABEL_POS],
        "relabel_map": {ORIG_LABEL_POS: 1, ORIG_LABEL_NEG: 0, ORIG_LABEL_NEU: 0},
        "num_classes": 2,
        "id2label_new": {0: "Not_Positive", 1: "Positive"},
        "label2id_new": {"Not_Positive": 0, "Positive": 1},
        "add_pos": False,
        "add_neg": False,
    },
    {
        "name": "Embedder_NegVsNotNeg",
        "original_labels_to_keep": [ORIG_LABEL_NEG, ORIG_LABEL_NEU, ORIG_LABEL_POS],
        "relabel_map": {ORIG_LABEL_NEG: 1, ORIG_LABEL_POS: 0, ORIG_LABEL_NEU: 0},
        "num_classes": 2,
        "id2label_new": {0: "Not_Negative", 1: "Negative"},
        "label2id_new": {"Not_Negative": 0, "Negative": 1},
        "add_pos": False,
        "add_neg": False,
    },
    {
        "name": "Embedder_NeuVsNotNeu",
        "original_labels_to_keep": [ORIG_LABEL_NEG, ORIG_LABEL_NEU, ORIG_LABEL_POS],
        "relabel_map": {ORIG_LABEL_NEU: 1, ORIG_LABEL_NEG: 0, ORIG_LABEL_POS: 0},
        "num_classes": 2,
        "id2label_new": {0: "Not_Neutral", 1: "Neutral"},
        "label2id_new": {"Not_Neutral": 0, "Neutral": 1},
        "add_pos": False,
        "add_neg": False,
    },
    {
        "name": "Embedder_extraPosVsNotPos",
        "original_labels_to_keep": [ORIG_LABEL_NEG, ORIG_LABEL_NEU, ORIG_LABEL_POS],
        "relabel_map": {ORIG_LABEL_POS: 1, ORIG_LABEL_NEG: 0, ORIG_LABEL_NEU: 0},
        "num_classes": 2,
        "id2label_new": {0: "Not_Positive", 1: "Positive"},
        "label2id_new": {"Not_Positive": 0, "Positive": 1},
        "add_pos": True,
        "add_neg": False,
    },
    {
        "name": "Embedder_extraNegVsNotNeg",
        "original_labels_to_keep": [ORIG_LABEL_NEG, ORIG_LABEL_NEU, ORIG_LABEL_POS],
        "relabel_map": {ORIG_LABEL_NEG: 1, ORIG_LABEL_POS: 0, ORIG_LABEL_NEU: 0},
        "num_classes": 2,
        "id2label_new": {0: "Not_Negative", 1: "Negative"},
        "label2id_new": {"Not_Negative": 0, "Negative": 1},
        "add_pos": False,
        "add_neg": True,
    },
]

IDX = 0
EXPERT = EXPERT_CONFIGS[IDX]

MODEL = "microsoft/deberta-v3-large"
NAME = EXPERT["name"]

SAVE_LOGITS_TEST = True
SAVE_EMBEDDINGS_TEST = False
SAVE_LOGITS_TRAIN = False
SAVE_EMBEDDINGS_TRAIN = False

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
EVAL_SIZE = 0
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

def prepare_expert_data(df_orig, expert_config):
    """
    Helper function that relabels data based on whihc expert it is training
    """
    df = df_orig.copy()
    df = df[df['label_id'].isin(expert_config['original_labels_to_keep'])]

    if 'relabel_map' in expert_config:
        df['label_id'] = df['label_id'].map(expert_config['relabel_map'])
        if df['label_id'].isnull().any():
            print(f"Warning: Null labels found after mapping for expert {expert_config['name']}. Original labels present: {df_orig['label'].unique()}. Dropping affected rows.")
            df.dropna(subset=['label_id'], inplace=True)
        df['label_id'] = df['label_id'].astype(int)
    return df

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

if EXPERT["add_pos"]: #augmented data postive
    context_pos_df = context_df[context_df['label_id'] == 2].copy()
    train_df = pd.concat([train_df, pos_df, context_pos_df], ignore_index=True)
    
if EXPERT["add_neg"]: #augmented data negative
    context_neg_df = context_df[context_df['label_id'] == 0].copy()
    train_df = pd.concat([train_df, neg_df, context_neg_df], ignore_index=True)

new_train_df = prepare_expert_data(train_df, EXPERT)

train_texts = new_train_df['sentence'].tolist()
train_labels = new_train_df['label_id'].tolist()
test_text = test_df['sentence'].tolist()

#--------------------------------------------------------------------------#

#model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL,
#     num_labels=3,
#     id2label=id2label,
#     label2id=label2id
# ).to(device)
config = AutoConfig.from_pretrained(
    MODEL,
    num_labels=EXPERT["num_classes"],
    id2label=EXPERT["id2label_new"],
    label2id=EXPERT["label2id_new"],
    output_hidden_states=True
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=config).to(device)

# freeze a num of layers sequentially 
base_model = getattr(model, model.base_model_prefix, model.base_model)
if FREEZE_NUM != 0:
    for name, param in base_model.embeddings.named_parameters():
        param.requires_grad = False

    for i in range(FREEZE_NUM):
        for param in base_model.encoder.layer[i].parameters():
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

# Predict test labels and save logits and IDs
model.eval()
predicted_labels = []
test_embeddings = []
test_logits = []

# Runs inference on the test set
with torch.no_grad():
     for batch in tqdm(test_loader, desc="Running inference on test set"):
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        test_logits.append(outputs.logits.cpu().numpy())

        cls_embeddings = outputs.hidden_states[-1][:, 0, :]
        test_embeddings.append(cls_embeddings.cpu().numpy())

        preds = torch.argmax(outputs.logits, dim=1)
        predicted_labels.extend(preds.cpu().numpy())

# Save logits and embeddings of the test set if requested
save_dict = {"ids": test_df['id'].values}
if SAVE_LOGITS_TEST:
    test_logits = np.concatenate(test_logits, axis=0)
    save_dict["logits"] = test_logits
if SAVE_EMBEDDINGS_TEST:
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    save_dict["embeddings"] = test_embeddings
if SAVE_LOGITS_TEST or SAVE_EMBEDDINGS_TEST:
    np.savez(f"{LOGITS_DIR}/test_outputs.npz", **save_dict)
    

# Save logits and embeddings of the training set if requested
if SAVE_LOGITS_TRAIN or SAVE_EMBEDDINGS_TRAIN:
    train_embeddings = []
    train_logits = []
    saved_train_labels = []
    
    #Reloads original original training data to run inference on it
    train2_df = pd.read_csv(TRAINING_DATA)
    train2_df["label_id"] = train2_df["label"].map(label2id)
    new_train2_df = prepare_expert_data(train2_df, EXPERT)
    train2_texts = new_train2_df['sentence'].tolist()
    train2_labels = new_train2_df['label_id'].tolist()
    train2_encodings = tokenizer(train2_texts, truncation=True, max_length=128, padding=True)
    train2_dataset = SentimentDataset(train2_encodings, train2_labels)
    train2_loader = DataLoader(train2_dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(train2_loader, desc="Running inference on train set"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
            train_logits.append(outputs.logits.cpu().numpy())
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            train_embeddings.append(cls_embeddings.cpu().numpy())
            saved_train_labels.extend(batch["labels"].cpu().numpy())
    
    save_dict = {"ids": new_train2_df['id'].values, "labels": np.array(saved_train_labels)}

    if SAVE_LOGITS_TRAIN:
        train_logits = np.concatenate(train_logits, axis=0)
        save_dict["logits"] = train_logits

    if SAVE_EMBEDDINGS_TRAIN:
        train_embeddings = np.concatenate(train_embeddings, axis=0)
        save_dict["embeddings"] = train_embeddings

    np.savez(f"{LOGITS_DIR}/train_outputs.npz", **save_dict)


# Model prediciton on the test data
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