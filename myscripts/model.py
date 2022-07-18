from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


def train_model(X, y):
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X, y)
    return logreg_model
