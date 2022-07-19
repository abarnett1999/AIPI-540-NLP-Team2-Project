import string
import nltk
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
import warnings

nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')


"""Pre-process text"""


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
    return tokens
