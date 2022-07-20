# AIPI-540-NLP-Team2-Project

# Chest X-ray: abnormal or normal, An Innovative approach using NLP

#### Category: Health, wellness and fitness
#### Type: Natural Language Processing

### Background & Situation 
-Chest X-rays: most common imaging procedures in medicine: images of our heart, lungs, blood vessels, airways, and the bones of our chest and spine. These images are used to diagnose conditons and complied in a report as impressions to determine whether a patient has heart problems, a collapsed lung, pneumonia, broken ribs, emphysema, cancer or any of several other conditions. 

-To minimize time it takes for emergent findings to be communicated, there needs to be a tool that could triage the abnormal cases to the head of the queue to be read  by a radiologist. 

### Links to original datasets
- https://academictorrents.com/details/66450ba52ba3f83fbf82ef9c91f2bde0e845aba9

### Links to papers
-https://rohansoni-jssaten2019.medium.com/indiana-university-chest-x-rays-automated-report-generation-38f928e6bfc2

## Project Structure

### main.py
Evaluate text input from streamlit app on the model and display results.

### model.py
Train model based on features created by count vectorization model. 

### data
- labeled_data.csv - impressions extracted from medical report xml files, labeled with 0/1
- xml_data - raw data, medical report xml files

### models
- model.pkl - trained logistic regression model
- vec.pkl - trained Count Vectorization model

### myscripts
- make_dataset.py - clean data and split into training/test sets
- text_preprocessing.py - tokenize and lemmatize text
- create_features.py - create features using count vectorization/TFIDF
- create_model.py - create logistic regression model

## Instructions to run Streamlit app. 
### Use Streamlit to classify an impression from medical report.
1. Install the requirements needed to use Streamlit
```
pip install -r requirements.txt
```
2. Start the Streamlit app
```
make run
```

