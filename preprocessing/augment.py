IN_CSV   = "data/training.csv"
OUT_CSV  = "data/contextual_aug.csv"
TEXT_COL = "sentence"
LABEL_COL= "label"

AUG_RATIO = 0.25
N_AUG     = 1
DEVICE    = "cuda" 

import pandas as pd, torch, random
import nlpaug.augmenter.word as naw
from tqdm import tqdm

df = pd.read_csv(IN_CSV)

ctx_aug = naw.ContextualWordEmbsAug(
    model_path="microsoft/deberta-v3-base",
    action="substitute",
    top_k=10,
    aug_p=0.10,
    device=DEVICE if torch.cuda.is_available() else "cpu",
    stopwords=["not","no","never"]
)

rows = []
for r in tqdm(df.itertuples(index=False), total=len(df)):
    if random.random() < AUG_RATIO:
        for _ in range(N_AUG):
            augmented = ctx_aug.augment(getattr(r, TEXT_COL))
            if isinstance(augmented, list):
                augmented = augmented[0]
            rows.append({
                TEXT_COL:  augmented,
                LABEL_COL: getattr(r, LABEL_COL)
            })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"wrote {len(rows)} augmented rows → {OUT_CSV}")
