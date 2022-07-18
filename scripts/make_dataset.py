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


def make_dataset():

    # Load data from csv to pandas dataframe
    data_df = pd.read_csv('labeled_data.csv')

    # Clean data up a bit more
    data_df = data_df[['filename', 'impression', 'label']]

    # Split into training and test sets - 80/20
    X = data_df['impression']
    y = data_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    return X_train, X_test, y_train, y_test
