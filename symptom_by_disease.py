import pandas as pd
import re
from collections import Counter

# Dataset'i yükle
data = pd.read_csv("data/symptom2disease.csv")

# Duplicate kayıtları temizle
data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

# İncelemek istediğimiz temel semptomlar
symptoms = [
    "fever",
    "high fever",
    "cough",
    "persistent cough",
    "dry cough",
    "difficulty breathing",
    "trouble breathing",
    "shortness of breath",
    "chest pain",
    "headache",
    "dizziness",
    "nausea",
    "vomiting",
    "stomach pain",
    "abdominal pain",
    "back pain",
    "neck pain",
    "sore throat",
    "rash",
    "skin rash",
    "itching",
    "severe itching",
    "chills",
    "fatigue",
    "weakness",
    "muscle pain",
    "muscle ache",
    "joint pain",
    "swollen joints",
    "loss of appetite",
    "weight loss",
    "blurred vision",
    "visual disturbance",
    "acid reflux",
    "indigestion",
    "yellow skin",
    "yellow urine"
]

print("\n========================================")
print("SYMPTOM FREQUENCY BY DISEASE")
print("========================================")

# Her hastalık için
for disease in sorted(data["label"].unique()):

    disease_data = data[data["label"] == disease]

    # Hastalığa ait bütün metinleri birleştir
    texts = " ".join(
        disease_data["text"].astype(str).tolist()
    ).lower()

    found_symptoms = []

    # Semptomları ara
    for symptom in symptoms:
        count = len(
            re.findall(
                r"\b" + re.escape(symptom) + r"\b",
                texts
            )
        )

        if count > 0:
            found_symptoms.append(
                (symptom, count)
            )

    print(f"\n--- {disease} ---")

    if found_symptoms:
        found_symptoms.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for symptom, count in found_symptoms:
            print(f"{symptom:25} {count}")
    else:
        print("No listed symptoms found.")