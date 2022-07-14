import csv
import pandas as pd

""" prelabel some of the data with normal = 0, abnormal = 1 """

# reading the csv file into pandas dataframe
df = pd.read_csv('./data/impressions.csv', header=0)

df["impression"] = df["impression"].str.strip()

# if No active disease, label as normal
df.loc[df['impression'] == "No active disease.", 'label'] = 0
df.loc[df['impression'] == "No evidence of active disease.", 'label'] = 0
df.loc[df['impression'] == "No acute cardiopulmonary disease.", 'label'] = 0
df.loc[df['impression'] == "No acute cardiopulmonary abnormality.", 'label'] = 0


print(df.count())

# convert back to csv
df.to_csv('./data/labeled_data.csv')

