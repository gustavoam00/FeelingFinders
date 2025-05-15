
TRAIN_CSV_FILES = [
    "data/training.csv",
    "data/backtranslated_neg.csv",
    "data/backtranslated_pos.csv",
    "data/contextual_aug.csv",
]

TEST_CSV       = "data/test.csv"
      
TRAIN_SOFT_OUT = "data/train_soft.csv"
TRAIN_HARD_OUT = "data/train_hard.csv"
TEST_SOFT_OUT  = "data/test_soft.csv"
TEST_HARD_OUT  = "data/test_hard.csv"


import re, emoji, torch, pandas as pd
from emot.emo_unicode import EMOTICONS_EMO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer

URL_RE  = re.compile(r'http\S+|www\.\S+', re.I)
USER_RE = re.compile(r'@\w+')
BUT_RE  = re.compile(r"\b(but|however|although|though)\b", re.I)
EMO_RE  = re.compile("|".join(map(re.escape, EMOTICONS_EMO.keys())))

def pre_soft(text: str) -> str:
    text = URL_RE.sub("HTTPURL", text)
    text = USER_RE.sub("@USER", text)
    return emoji.demojize(text, delimiters=("", "")).strip()

ek = TextPreProcessor(
        normalize=['url', 'user'],
        fix_html=True,
        segmenter=None,
        corrector="twitter",
        unpack_hashtags=False,
        tokenizer=SocialTokenizer(lowercase=False).tokenize)

def pre_hard(text: str) -> str:
    text = pre_soft(text)
    text = EMO_RE.sub(lambda m: EMOTICONS_EMO[m.group()], text)
    text = emoji.demojize(text, delimiters=("", ""))
    return " ".join(ek.pre_process_doc(text)).strip()

SARC_MODEL = "helinivan/english-sarcasm-detector"
tokenizer  = AutoTokenizer.from_pretrained(SARC_MODEL)
model  = AutoModelForSequenceClassification.from_pretrained(SARC_MODEL)
model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

@torch.inference_mode()
def sarcasm_probs(texts, bs: int = 32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i+bs], truncation=True,
                   padding=True, return_tensors="pt").to(model.device)
        out.extend(torch.softmax(model(**enc).logits, -1)[:, 1].cpu())
    return out


def preprocess_df(df: pd.DataFrame, text_col="sentence"):
    # tag <SARC>/<AMBIG>
    sarc_prob = sarcasm_probs(df[text_col].tolist())
    tags  = []
    for sent, p in zip(df[text_col], sarc_prob):
        tkns = []
        if p > 0.8:            tkns.append("<SARC>")
        if BUT_RE.search(sent): tkns.append("<AMBIG>")
        tags.append((" ".join(tkns) + " ") if tkns else "")

    soft_sent = [tags[i] + pre_soft(s) for i, s in enumerate(df[text_col])]
    hard_sent = [tags[i] + pre_hard(s) for i, s in enumerate(df[text_col])]

    meta_cols = df.columns.difference([text_col])
    soft_df = pd.concat([df[meta_cols].reset_index(drop=True),
                         pd.Series(soft_sent, name=text_col)], axis=1)
    hard_df = pd.concat([df[meta_cols].reset_index(drop=True),
                         pd.Series(hard_sent, name=text_col)], axis=1)
    return soft_df, hard_df


if TRAIN_CSV_FILES:
    print(f"→ Reading {len(TRAIN_CSV_FILES)} training file(s)…")
    train_df = pd.concat([pd.read_csv(p) for p in TRAIN_CSV_FILES],
                         ignore_index=True)
    
    trpre_soft, trpre_hard = preprocess_df(train_df, "sentence")
    trpre_soft.to_csv(TRAIN_SOFT_OUT, index=False)
    trpre_hard.to_csv(TRAIN_HARD_OUT, index=False)
    print(f"Wrote {TRAIN_SOFT_OUT} and {TRAIN_HARD_OUT}")


if TEST_CSV is not None:
    print("→ Reading test file…")
    test_df = pd.read_csv(TEST_CSV)
    tepre_soft, tepre_hard = preprocess_df(test_df, "sentence")
    tepre_soft.to_csv(TEST_SOFT_OUT, index=False)
    tepre_hard.to_csv(TEST_HARD_OUT, index=False)
    print(f"Wrote {TEST_SOFT_OUT} and {TEST_HARD_OUT}")

