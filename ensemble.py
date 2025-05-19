import torch
import numpy as np
import pandas as pd
import os

model_names = ["bertweet", "deberta", "bart"]
weights = [0.3, 0.45, 0.25]
LOGITS_DIR_BASE = "./saved_logits"
OUTPUT_NAME = "./results/test4_predictions.csv"


label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

logits_list = []
ids_reference = None

for name in model_names:
    path = os.path.join(LOGITS_DIR_BASE, name, "test_logits.npz")
    data = np.load(path)
    logits = data['logits']
    ids = data['ids']

    if ids_reference is None:
        ids_reference = ids

    logits_list.append(logits)

probs_stack = np.stack(logits_list, axis=0)
mean_probs = np.average(probs_stack, axis=0, weights=weights)


predicted_indices = np.argmax(mean_probs, axis=1)

predicted_labels = [id2label[int(i)] for i in predicted_indices]

test_output = pd.DataFrame({
    'id': ids_reference,
    'label': predicted_labels
})
test_output.to_csv(OUTPUT_NAME, index=False)