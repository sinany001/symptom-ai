import re
from symptom_dictionary import SYMPTOM_DICTIONARY


def extract_symptoms(text):
    """
    Kullanıcının yazdığı cümleden
    bilinen semptomları çıkarır.
    """

    text = text.lower()

    detected_symptoms = []

    for symptom, phrases in SYMPTOM_DICTIONARY.items():

        for phrase in phrases:

            # Kelimenin/ifadenin tam olarak geçmesini kontrol et
            pattern = r"\b" + re.escape(phrase) + r"\b"

            if re.search(pattern, text):
                detected_symptoms.append(symptom)
                break

    return detected_symptoms


