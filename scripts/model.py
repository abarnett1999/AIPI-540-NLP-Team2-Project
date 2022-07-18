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


def train_model(X, y):
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X, y)
    return logreg_model
