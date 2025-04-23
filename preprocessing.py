import numpy as np
import pandas as pd
from tqdm import tqdm
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from deep_translator import GoogleTranslator
from langdetect import detect
from textblob import TextBlob


"""
Define all preprocessing methods
"""

def translate(sentence):
    lang = 'en'
    try:
        lang = detect(sentence)
    finally:
        if lang != 'en':
            sentence = GoogleTranslator(source='auto', target='en').translate(sentence)
        return sentence
    

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
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
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def process_ekphrasis(sentence):
    try:
        sentence = " ".join(text_processor.pre_process_doc(sentence))
    finally:
        return sentence


def correct_spelling(sentence):
    try:
        text = TextBlob(sentence)
    finally:
        return sentence
    

"""
Select and perform the above preprocessing methods
"""

df = pd.read_csv("./data/training.csv")

tqdm.pandas()

processed_df = df
processed_df['sentence'] = processed_df['sentence'].progress_apply(translate)
processed_df['sentence'] = processed_df['sentence'].progress_apply(process_ekphrasis)
processed_df['sentence'] = processed_df['sentence'].progress_apply(correct_spelling)

processed_df.to_csv("train_prepped.csv", index=False)