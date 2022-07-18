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


"""## **Create features**"""
def build_features(train_data, test_data, ngram_range, method):
    if method == 'tfidf':
        # Create features using TFIDF
        vec = TfidfVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_data)
        X_test = vec.transform(test_data)

    elif method == 'count':
        # Create features using word counts
        vec = CountVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_data)
        X_test = vec.transform(test_data)

    return X_train, X_test