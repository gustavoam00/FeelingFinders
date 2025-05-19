import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict
from deep_translator import GoogleTranslator
from langdetect import detect
from tqdm import tqdm


"""
Define all preprocessing steps
"""

def translate(sentence):
    lang = 'en'
    try:
        lang = detect(sentence)
    finally:
        if lang != 'en':
            sentence = GoogleTranslator(source='auto', target='en').translate(sentence)
        return sentence
    
text_processor_soft = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'user'],
    # terms that will be annotated
    annotate={"emoticons"},
    fix_html=True,  # fix HTML tokens
    
    # Don't do anything else
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=False,
    unpack_contractions=False,
    spell_correct_elong=False,
    spell_correction=False,

    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # Dictionary for emoticons
    dicts=[emoticons]
)

text_processor_hard = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    spell_correction=True,
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons, slangdict]
)
    
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

BUT_RE  = re.compile(r"\b(but|however|although|though)\b", re.I)


"""
Build different preprocessing pipelines: SOFT, SOFTPLUS, HARD
"""

def process_soft(sentence):
    sentence = translate(sentence)
    try:
        sentence = " ".join(text_processor_soft.pre_process_doc(sentence))
    finally:
        return sentence
    
def process_softplus(sentence):
    sentence = translate(sentence)
    try:
        sentence = " ".join(text_processor_soft.pre_process_doc(sentence))
    finally:
        if sarcasm_probs(sentence)[0] > 0.5:
            sentence = "<SARC> " + sentence
        if BUT_RE.search(sentence):
            sentence = "<AMBIG> " + sentence
        return sentence

def process_hard(sentence):
    sentence = translate(sentence)
    try:
        sentence = " ".join(text_processor_hard.pre_process_doc(sentence))
    finally:
        if sarcasm_probs(sentence)[0] > 0.7:
            sentence = "<SARC> " + sentence
        if BUT_RE.search(sentence):
            sentence = "<AMBIG> " + sentence
        return sentence
    

"""
Apply all three pipelines
"""

tqdm.pandas()

df_train = pd.read_csv("training.csv")
df_test = pd.read_csv("test.csv")

df_train_soft = df_train['sentence'].progress_apply(process_soft)
df_train_soft.to_csv("TRAIN_P_SOFT.csv", index=False)
print("TRAIN_P_SOFT completed")
df_train_softplus = df_train['sentence'].progress_apply(process_softplus)
df_train_softplus.to_csv("TRAIN_P_SOFTPLUS.csv", index=False)
print("TRAIN_P_SOFTPLUS completed")
df_train_hard = df_train['sentence'].progress_apply(process_hard)
df_train_hard.to_csv("TRAIN_P_HARD.csv", index=False)
print("TRAIN_P_HARD completed")

df_test_soft = df_test['sentence'].progress_apply(process_soft)
df_test_soft.to_csv("TEST_P_SOFT.csv", index=False)
print("TEST_P_SOFT completed")
df_test_softplus = df_test['sentence'].progress_apply(process_softplus)
df_test_softplus.to_csv("TEST_P_SOFTPLUS.csv", index=False)
print("TEST_P_SOFTPLUS completed")
df_test_hard = df_test['sentence'].progress_apply(process_hard)
df_test_hard.to_csv("TEST_P_HARD.csv", index=False)
print("TEST_P_HARD completed")
