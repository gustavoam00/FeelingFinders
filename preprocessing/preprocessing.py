import os, re, emoji, torch, pandas as pd
from emot.emo_unicode import EMOTICONS_EMO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from deep_translator import GoogleTranslator
from langdetect import detect

TRAIN_CSV_FILES = [
    "data/training.csv",
    "data/backtranslated_neg.csv",
    "data/backtranslated_pos.csv",
    "data/contextual_aug_05.csv",
]

TEST_CSV       = "data/test.csv"
      
URL_RE  = re.compile(r'http\S+|www\.\S+', re.I)
USER_RE = re.compile(r'@\w+')
BUT_RE  = re.compile(r"\b(but|however|although|though)\b", re.I)
EMO_RE  = re.compile("|".join(map(re.escape, EMOTICONS_EMO.keys())))

def translate(sentence):
    lang = 'en'
    try:
        lang = detect(sentence)
    finally:
        if lang != 'en':
            sentence = GoogleTranslator(source='auto', target='en').translate(sentence)
        return sentence

def pre_soft(text):
    #text = translate(text)
    text = URL_RE.sub("HTTPURL", text)
    text = USER_RE.sub("@USER", text)
    text = EMO_RE.sub(lambda m: EMOTICONS_EMO[m.group()], text)
    return emoji.demojize(text, delimiters=("", "")).strip()

ek = TextPreProcessor(
        normalize=['url', 'user'],
        fix_html=True,
        segmenter=None,
        corrector="twitter",
        unpack_hashtags=False,
        tokenizer=SocialTokenizer(lowercase=False).tokenize)

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
    #SOFT
    soft_sent = [pre_soft(s) for s in df[text_col]]

    # tag <SARC>/<AMBIG>
    sarc_prob = sarcasm_probs(df[text_col].tolist())
    tags  = []
    for sent, p in zip(df[text_col], sarc_prob):
        tkns = []
        if p > 0.8:            tkns.append("<SARC>")
        if BUT_RE.search(sent): tkns.append("<AMBIG>")
        tags.append((" ".join(tkns) + " ") if tkns else "")

    #SOFT PLUS
    soft_plus_sent = [f"{tag}{s}" for tag, s in zip(tags, soft_sent)]

    #HARD
    hard_sent = [" ".join(ek.pre_process_doc(s)).strip() for s in soft_plus_sent]

    soft_df       = df.copy().reset_index(drop=True)
    soft_plus_df  = df.copy().reset_index(drop=True)
    hard_df       = df.copy().reset_index(drop=True)

    soft_df[text_col]       = soft_sent
    soft_plus_df[text_col]  = soft_plus_sent
    hard_df[text_col]       = hard_sent
    return soft_df, soft_plus_df, hard_df


for csv_path in TRAIN_CSV_FILES:
    print(f"-> Reading training file {csv_path} …")
    df_soft, df_soft_plus, df_hard = preprocess_df(pd.read_csv(csv_path), "sentence")

    base, _ = os.path.splitext(csv_path)
    soft_out       = f"{base}_prepped_soft.csv"
    soft_plus_out  = f"{base}_prepped_softplus.csv"
    hard_out       = f"{base}_prepped_hard.csv"

    df_soft.to_csv(soft_out, index=False)
    df_soft_plus.to_csv(soft_plus_out, index=False)
    df_hard.to_csv(hard_out, index=False)
    print(f"Wrote {soft_out}, {soft_plus_out}, and {hard_out}")


if TEST_CSV:
    print(f"-> Reading test file {TEST_CSV} …")
    df_soft, df_soft_plus, df_hard = preprocess_df(pd.read_csv(TEST_CSV), "sentence")

    base, _ = os.path.splitext(TEST_CSV)
    soft_out       = f"{base}_prepped_soft.csv"
    soft_plus_out  = f"{base}_prepped_softplus.csv"
    hard_out       = f"{base}_prepped_hard.csv"

    df_soft.to_csv(soft_out, index=False)
    df_soft_plus.to_csv(soft_plus_out, index=False)
    df_hard.to_csv(hard_out, index=False)
    print(f"Wrote {soft_out}, {soft_plus_out}, and {hard_out}")

