import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Verileri yükle
old = pd.read_csv("data/symptom2disease.csv")[["text", "label"]]
expanded = pd.read_csv("data/combined_experiment_300.csv")[["text", "label"]]

for df in (old, expanded):
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

old = old[old["text"] != ""].drop_duplicates(subset=["text"])

# 2. Test kümesini yalnızca eski veriden ayır
X_train, X_test, y_train, y_test = train_test_split(
    old["text"],
    old["label"],
    test_size=0.2,
    random_state=42,
    stratify=old["label"],
)

# 3. Eğitim kümelerini oluştur
old_train = pd.DataFrame({"text": X_train, "label": y_train})
old_texts = set(old_train["text"])

# Genişletilmiş dosyadan eski eğitim örneklerini ve yeni adayları ayır.
# Test kümesindeki eski örneklerin eğitimde kullanılmasını önle.
new_only = expanded[~expanded["text"].isin(set(old["text"]))].copy()

expanded_train = pd.concat(
    [old_train, new_only],
    ignore_index=True,
).drop_duplicates(subset=["text"])

print("Eski eğitim örneği:", len(old_train))
print("Yeni aday eğitim örneği:", len(new_only))
print("Genişletilmiş eğitim örneği:", len(expanded_train))
print("Ortak test örneği:", len(X_test))

# 4. İki modeli aynı test kümesinde değerlendir
for name, train_data in [
    ("Model A - Eski veri", old_train),
    ("Model B - Genişletilmiş veri", expanded_train),
]:
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svm", LinearSVC(C=10.0)),
    ])

    model.fit(train_data["text"], train_data["label"])
    predictions = model.predict(X_test)

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)
    print(f"Accuracy:    {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Macro F1:    "
        f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Weighted F1: "
        f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print("\nSınıf bazında sonuçlar:")
    print(classification_report(y_test, predictions, zero_division=0))

print("\nNot: Kayıtlı model dosyaları değiştirilmedi.")
