import numpy as np
import pandas as pd
import string
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import nltk
import warnings
from sklearn.metrics import recall_score

# importing sys
import sys

# adding Folder_2 to the system path
from scripts import make_dataset, model, text_preprocessing, create_features

sys.path.insert(0, 'scripts')


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
