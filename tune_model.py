
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer, f1_score

# 1. Load and clean dataset
data = pd.read_csv("data/symptom2disease.csv")
data = data.drop(columns=["Unnamed: 0"], errors="ignore")
data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

X = data["text"]
y = data["label"]

print("Dataset size:", len(data))
print("Number of classes:", y.nunique())

# 2. Keep the test set separate
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Held-out test samples:", len(X_test))

# 3. Define cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Try different C values
c_values = [0.1, 0.5, 1.0, 2.0, 5.0]
results = []

for c_value in c_values:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LinearSVC(C=c_value))
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "macro_f1": make_scorer(
                f1_score,
                average="macro",
                zero_division=0
            ),
            "weighted_f1": make_scorer(
                f1_score,
                average="weighted",
                zero_division=0
            )
        },
        n_jobs=-1
    )

    result = {
        "C": c_value,
        "Accuracy": scores["test_accuracy"].mean(),
        "Macro F1": scores["test_macro_f1"].mean(),
        "Weighted F1": scores["test_weighted_f1"].mean()
    }

    results.append(result)

    print(
        f"C={c_value:<3} | "
        f"Accuracy={result['Accuracy']:.4f} | "
        f"Macro F1={result['Macro F1']:.4f} | "
        f"Weighted F1={result['Weighted F1']:.4f}"
    )

# 5. Save and display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

results_df.to_csv("tuning_results.csv", index=False)

print("\n" + "=" * 55)
print("RESULTS SORTED BY MACRO F1")
print("=" * 55)
print(results_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

print("\nResults saved to tuning_results.csv")
print("Existing application model files were not changed.")
