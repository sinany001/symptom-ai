import joblib
import pandas as pd

# Model ve vectorizer'ı yükle
model = joblib.load("models/symptom_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# TF-IDF'teki kelimeler
features = vectorizer.get_feature_names_out()

# Sadece incelemek istediğimiz sınıflar
classes_to_check = [
    "Bronchial Asthma",
    "Pneumonia"
]

for class_name in classes_to_check:

    # Modelde bu sınıfın index'i
    class_index = list(model.classes_).index(class_name)

    # O sınıfa ait kelime ağırlıkları
    coefficients = model.coef_[class_index]

    # En güçlü 15 kelimeyi bul
    top_indices = coefficients.argsort()[-15:][::-1]

    print("\n========================================")
    print(class_name)
    print("========================================")

    for index in top_indices:
        print(f"{features[index]:25} {coefficients[index]:.4f}")