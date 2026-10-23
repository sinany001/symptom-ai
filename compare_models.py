
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Load and clean the dataset
data = pd.read_csv("data/symptom2disease.csv")

data = data.drop(columns=["Unnamed: 0"], errors="ignore")
data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

X = data["text"]
y = data["label"]

print("Dataset size after cleaning:", len(data))
print("Number of classes:", y.nunique())


# 2. Use the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Define three experiments
experiments = {
    "1. Word TF-IDF + LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LinearSVC())
    ]),

    "2. Word + Character TF-IDF + LinearSVC": Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True
            ))
        ])),
        ("classifier", LinearSVC())
    ]),

    "3. Word TF-IDF + Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=2000))
    ])
}


# 4. Train and evaluate each experiment
results = []

for name, pipeline in experiments.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    errors = sum(
        actual != predicted
        for actual, predicted in zip(y_test, predictions)
    )

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted Prec.: {precision:.4f}")
    print(f"Weighted Recall:{recall:.4f}")
    print(f"Incorrect:      {errors}/{len(y_test)}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Weighted F1": weighted_f1,
        "Macro F1": macro_f1,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Incorrect Predictions": errors
    })


# 5. Compare results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to model_comparison.csv")
print("Existing saved model files were not changed.")
