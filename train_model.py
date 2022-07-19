import warnings
import pickle

import nltk
from sklearn.metrics import recall_score
from tqdm import tqdm

import sys

# adding Folder_2 to the system path
sys.path.insert(0, 'myscripts')
from myscripts import make_dataset, model, text_preprocessing, create_features


nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')



""" Get data """
X_train, X_test, y_train, y_test = make_dataset.make_dataset()

""" Tokenize and lemmatize the data """
# Process the training set
tqdm.pandas()
X_train_processed = X_train.progress_apply(lambda x: text_preprocessing.tokenize(x, method='nltk'))

# Process the test set text
tqdm.pandas()
X_test_processed = X_test.progress_apply(lambda x: text_preprocessing.tokenize(x, method='nltk'))

""" Create features using  Count Vectorization """
method = 'count'
ngram_range = (1, 2)
X_train = create_features.build_features(X_train_processed, ngram_range, method)
# X_test = create_features.build_features(X_test_processed, True, ngram_range, method)

print("-------Commencing training------")

""" Train model on training set """
logreg_model = model.create_model(X_train, y_train)
preds = logreg_model.predict(X_train)
acc = sum(preds == y_train) / len(y_train)
recall = recall_score(y_train, preds)
print('Accuracy on the training set is {:.3f}'.format(acc))
print('Recall on the training set is {:.3f}'.format(recall))

pickle.dump(logreg_model, open('models/model.pkl', 'wb'))
