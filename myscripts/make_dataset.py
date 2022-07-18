import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


def make_dataset():

    # Load data from csv to pandas dataframe
    data_df = pd.read_csv('data/labeled_data.csv')

    # Clean data up a bit more
    data_df = data_df[['filename', 'impression', 'label']]

    # Split into training and test sets - 80/20
    X = data_df['impression']
    y = data_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    return X_train, X_test, y_train, y_test
