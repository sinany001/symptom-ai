import pandas as pd

data = pd.read_csv("data/symptom2disease.csv")

print(data.head())
print(data.shape)
print(data.columns)
print(data["label"].nunique())
print(data["label"].value_counts())
print(data.isnull().sum())
print(data["text"].iloc[0])
