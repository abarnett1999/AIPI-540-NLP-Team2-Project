import os
import numpy as np
import pandas as pd
import string
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import nltk
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

nlp = spacy.load('en_core_web_sm')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')


"""### **Pre-process text**"""


# Tokenize text on white space and punctuation (using NLTK)
# Then lemmatize the text

def tokenize(sentence, method):
    # Tokenize and lemmatize text, remove stopwords and punctuation

    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

    if method == 'nltk':
        # Tokenize
        tokens = nltk.word_tokenize(sentence, preserve_line=True)
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        # Lemmatize
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
        tokens = " ".join([i for i in tokens])
    else:
        # Tokenize
        with nlp.select_pipes(enable=['tokenizer', 'lemmatizer']):
            tokens = nlp(sentence)
        # Lemmatize
        tokens = [word.lemma_.lower().strip() for word in tokens]
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        tokens = " ".join([i for i in tokens])
    return tokens
