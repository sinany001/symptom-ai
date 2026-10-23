import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# 1. VERİ SETİNİ YÜKLE
# ==========================================

data = pd.read_csv("data/symptom2disease.csv")

print("Original dataset size:", len(data))


# ==========================================
# 2. GEREKSİZ SÜTUNU KALDIR
# ==========================================

data = data.drop(columns=["Unnamed: 0"])


# ==========================================
# 3. DUPLICATE METİNLERİ TEMİZLE
# ==========================================

data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Dataset after removing duplicates:", len(data))


# ==========================================
# 4. GİRDİ VE HEDEFİ BELİRLE
# ==========================================

X = data["text"]
y = data["label"]


# ==========================================
# 5. EĞİTİM / TEST AYIR
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 7. MODELLER
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "SVM": LinearSVC(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ==========================================
# 8. MODELLERİ EĞİT VE KARŞILAŞTIR
# ==========================================

for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ==========================================
# 9. FINAL MODEL: SVM
# ==========================================

final_model = LinearSVC()

final_model.fit(X_train_tfidf, y_train)


# ==========================================
# 10. MODELİ KAYDET
# ==========================================

joblib.dump(
    final_model,
    "models/symptom_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


print("\n================================")
print("Final SVM model saved successfully!")
print("================================")