import torch
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

en_to_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_to_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

fr_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
en_to_fr_model = en_to_fr_model.to(device)
fr_to_en_model = fr_to_en_model.to(device)

def batch_backtranslate(sentences, batch_size=8):
    backtranslated = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Backtranslating"):
        batch = sentences[i:i+batch_size]

        # EN → FR
        fr_tokens = en_to_fr_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        fr_translation = en_to_fr_model.generate(**fr_tokens)
        fr_texts = en_to_fr_tokenizer.batch_decode(fr_translation, skip_special_tokens=True)

        # FR → EN
        en_tokens = fr_to_en_tokenizer(fr_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        en_translation = fr_to_en_model.generate(**en_tokens)
        en_texts = fr_to_en_tokenizer.batch_decode(en_translation, skip_special_tokens=True)

        backtranslated.extend(en_texts)
    return backtranslated

def process_and_save(df, target_label, output_filename):
    subset = df[df["label"] == target_label].copy()
    subset['clean_sentence'] = subset['sentence'].astype(str)
    original_sentences = subset['clean_sentence'].tolist()

    backtranslated_sentences = batch_backtranslate(original_sentences)

    out_df = pd.DataFrame({
        'id': subset['id'].tolist(),
        'sentence': backtranslated_sentences,
        'label': [target_label] * len(backtranslated_sentences)
    })

    out_df.to_csv(output_filename, index=False)


df = pd.read_csv("../data/training.csv")

process_and_save(df, target_label="positive", output_filename="../data/backtranslated_pos.csv")
process_and_save(df, target_label="negative", output_filename="../data/backtranslated_neg.csv")
