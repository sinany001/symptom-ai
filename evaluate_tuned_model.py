
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

# 1. Load and clean dataset
data = pd.read_csv("data/symptom2disease.csv")
data = data.drop(columns=["Unnamed: 0"], errors="ignore")
data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

X = data["text"]
y = data["label"]

# 2. Recreate the original train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create a new model with the selected C value
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LinearSVC(C=2.0))
])

# 4. Train only on the training set
model.fit(X_train, y_train)

# 5. Evaluate on the held-out test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)
weighted_f1 = f1_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

errors = sum(
    actual != predicted
    for actual, predicted in zip(y_test, predictions)
)

print("\n" + "=" * 55)
print("TUNED MODEL: LINEARSVC (C=2.0)")
print("=" * 55)

print(f"Test samples: {len(y_test)}")
print(f"Accuracy:     {accuracy:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")
print(f"Incorrect predictions: {errors}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
print("\nINCORRECT PREDICTIONS")

for text, actual, predicted in zip(X_test, y_test, predictions):
    if actual != predicted:
        print("\nText:", text)
        print("Actual:", actual)
        print("Predicted:", predicted)
        print("-" * 60)

print("Existing application model files were not changed.")
