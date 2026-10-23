
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and clean the dataset
data = pd.read_csv("data/symptom2disease.csv")

# Remove unnecessary column if it exists
data = data.drop(columns=["Unnamed: 0"], errors="ignore")

# Remove duplicate symptom descriptions
data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

X = data["text"]
y = data["label"]

# 2. Recreate the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load the saved model AND its matching TF-IDF vectorizer
model = joblib.load("models/symptom_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# 4. Transform test data and make predictions
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# 5. Classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, y_pred, zero_division=0))

# 6. Confusion matrix
labels = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(16, 13))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("SymptomAI - SVM Confusion Matrix")

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion matrix saved as confusion_matrix.png")

# 7. Display incorrect predictions
results = pd.DataFrame({
    "Text": X_test.reset_index(drop=True),
    "Actual": y_test.reset_index(drop=True),
    "Predicted": y_pred
})

errors = results[results["Actual"] != results["Predicted"]]

print("\n" + "=" * 60)
print("INCORRECT PREDICTIONS")
print("=" * 60)

print("Total test samples:", len(y_test))
print("Total incorrect predictions:", len(errors))
print("Accuracy:", round((len(y_test) - len(errors)) / len(y_test), 4))

for _, row in errors.iterrows():
    print("\nText:", row["Text"])
    print("Actual:", row["Actual"])
    print("Predicted:", row["Predicted"])
    print("-" * 60)
