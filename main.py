import nltk
import warnings
import streamlit as st
import pandas as pd
import pickle

from myscripts import text_preprocessing

nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

# Headings for Web Application
st.title("Chest X-Rays-Normal or Abnormal")

# Textbox for text user is entering
st.subheader("Enter the text to analyze.")
text = st.text_input('Enter text')  # text is stored in this variable

text_list = [text]
# Create the pandas DataFrame with column name is provided explicitly
input_df = pd.DataFrame(text_list, columns=['impression'])
input_df = input_df['impression']

# Process the input text
processed = input_df.apply(lambda x: text_preprocessing.tokenize(x, method='nltk'))

# Load model and vectorizer
vect = pickle.load(open('./models/vec.pkl','rb'))
model = pickle.load(open('models/model.pkl', 'rb'))

# Create features using  Count Vectorization
model_input = vect.transform(processed)

# Use model to predict from input
preds = model.predict(model_input)

print('pred: ', str(preds[0]))

if preds[0] == 0:
    result = 'Normal'
else:
    result = 'Abnormal'


# Display results of the NLP task
st.header("Results")
st.subheader(result)
