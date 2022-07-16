#Necessary imports
import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt



#Headings for Web Application
st.title("Chest X-Rays-Normal or Abnormal")
#st.subheader("What type of NLP service would you like to use?")

#Picking what NLP task you want to do
#option = st.selectbox('NLP Service',('Sentiment Analysis', 'Entity Extraction', 'Text Summarization')) #option is stored in this variable

#Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze.")
text = st.text_input('Enter text') #text is stored in this variable

#Display results of the NLP task
st.header("Results")

# @st.cache
# def load_model():
#     # model = path/to/model:object
#     return model

# def make_prediction(text_input, model):
#     pred = model.predict(text_input)
#     prob = model.predict_proba(text_input)
#     return pred, prob

# model = load_model()
# prediction, probability = make_prediction(text, model)

# put some streamlit displays for the prediction and probability
# preds_text = {
#     0 : "Normal",
#     1 : "Refer to Radiologist"
# }

# st.text(f"The Results are: {preds_text[prediction]}")