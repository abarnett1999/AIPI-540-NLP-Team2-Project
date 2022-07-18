from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
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
