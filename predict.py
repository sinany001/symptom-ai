import joblib

from symptom_extractor import extract_symptoms
from risk_assessment import assess_risk


# ========================================
# MODEL VE VECTORIZER'I YÜKLE
# ========================================

model = joblib.load("models/symptom_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ========================================
# BAŞLANGIÇ
# ========================================

print("========================================")
print("          SYMPTOM AI")
print("   PRELIMINARY ASSESSMENT SYSTEM")
print("========================================")


# Kullanıcıdan semptomları al
user_input = input("\nDescribe your symptoms: ")


# ========================================
# 1. SYMPTOM EXTRACTION
# ========================================

detected_symptoms = extract_symptoms(user_input)


# ========================================
# 2. SVM CLASSIFICATION
# ========================================

symptom_tfidf = vectorizer.transform([user_input])

prediction = model.predict(symptom_tfidf)[0]

scores = model.decision_function(symptom_tfidf)[0]

top_3_indices = scores.argsort()[-3:][::-1]


# ========================================
# 3. RISK ASSESSMENT
# ========================================

risk_result = assess_risk(detected_symptoms)


# ========================================
# 4. DETECTED SYMPTOMS
# ========================================

print("\n========================================")
print("          DETECTED SYMPTOMS")
print("========================================")

if detected_symptoms:

    for symptom in detected_symptoms:
        print(f"- {symptom}")

else:

    print("No known symptoms detected.")


# ========================================
# 5. PRELIMINARY ASSESSMENT
# ========================================

print("\n========================================")
print("       PRELIMINARY ASSESSMENT")
print("========================================")

print(f"\nMost likely dataset classification: {prediction}")


# ========================================
# 6. TOP 3 CLASSIFICATIONS
# ========================================

print("\nTop 3 model classifications:")

for index in top_3_indices:

    disease = model.classes_[index]
    score = scores[index]

    print(f"- {disease}: {score:.4f}")


# ========================================
# 7. RISK ASSESSMENT RESULT
# ========================================

print("\n========================================")
print("          RISK ASSESSMENT")
print("========================================")

print(f"Risk Level: {risk_result['level']}")

print(f"Message: {risk_result['message']}")


# ========================================
# 8. SAFETY NOTICE
# ========================================

print("\n----------------------------------------")
print("IMPORTANT:")
print("This system provides a preliminary")
print("assessment based on the dataset.")
print("It is NOT a medical diagnosis.")
print("----------------------------------------")