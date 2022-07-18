import numpy as np
import pandas as pd
import string
import time
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
import warnings
from sklearn.metrics import recall_score

from scripts/make_dataset import make_dataset
from scripts/text_preprocessing import tokenize
from scripts/model import train_model
from scripts/create_features import build_features

nlp = spacy.load('en_core_web_sm')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

""" Get data """
X_train, X_test, y_train, y_test = make_dataset()


""" Tokenize and lemmatize the data """
# Process the training set
tqdm.pandas()
X_train_processed = X_train.progress_apply(lambda x: tokenize(x, method='nltk'))

# Process the test set text
tqdm.pandas()
X_test_processed = X_test.progress_apply(lambda x: tokenize(x, method='nltk'))

""" Create features using  Count Vectorization """
method = 'count'
ngram_range = (1, 2)
X_train, X_test = build_features(X_train_processed, X_test_processed, ngram_range, method)


""" Train model on training set """
logreg_model = train_model(X_train, y_train)
preds = logreg_model.predict(X_train)
acc = sum(preds == y_train) / len(y_train)
recall = recall_score(y_train, preds)
print('Accuracy on the training set is {:.3f}'.format(acc))
print('Recall on the training set is {:.3f}'.format(recall))

""" Evaluate model on test set """
test_preds = logreg_model.predict(X_test)
test_acc = sum(test_preds == y_test) / len(y_test)
test_recall = recall_score(y_test, test_preds)
print('Accuracy on the test set is {:.3f}'.format(test_acc))
print('Recall on the test set is {:.3f}'.format(test_recall))


