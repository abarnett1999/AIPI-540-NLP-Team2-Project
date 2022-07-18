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


def train_model(X, y):
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X, y)
    return logreg_model
