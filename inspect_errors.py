
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load and clean the dataset
data = pd.read_csv("data/symptom2disease.csv")

if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

X = data["text"]
y = data["label"]

# 2. Use the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train the SVM model
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LinearSVC()
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

# 4. Find incorrect predictions
errors = []

for i, (actual, predicted) in enumerate(
    zip(y_test.tolist(), predictions)
):
    if actual != predicted:
        errors.append(i)

print("Total incorrect predictions:", len(errors))

# 5. Analyze each incorrect prediction
for i in errors:
    test_vector = X_test_tfidf[i]

    similarities = cosine_similarity(
        test_vector,
        X_train_tfidf
    ).flatten()

    nearest_indices = similarities.argsort()[-3:][::-1]

    print("\n" + "=" * 70)
    print("TEST TEXT:", X_test.iloc[i])
    print("ACTUAL:", y_test.iloc[i])
    print("PREDICTED:", predictions[i])

    # SVM decision scores (not probabilities)
    decision_scores = model.decision_function(test_vector)[0]

    top_indices = decision_scores.argsort()[-3:][::-1]

    print("\nTOP 3 MODEL SCORES:")

    for class_index in top_indices:
        print(
            f"{model.classes_[class_index]}: "
            f"{decision_scores[class_index]:.4f}"
        )

    print("\nMOST SIMILAR TRAINING EXAMPLES:")

    for index in nearest_indices:
        print("\nSimilarity:", round(similarities[index], 3))
        print("Label:", y_train.iloc[index])
        print("Text:", X_train.iloc[index])
