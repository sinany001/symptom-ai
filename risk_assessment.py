# ========================================
# RISK ASSESSMENT
# ========================================

HIGH_RISK_SYMPTOMS = {
    "breathing_difficulty",
    "chest_pain"
}

MEDIUM_RISK_SYMPTOMS = {
    "high fever",
    "fever",
    "severe itching",
    "vomiting",
    "dizziness"
}


def assess_risk(detected_symptoms):

    detected = set(detected_symptoms)

    # Öncelikle yüksek riskli belirtileri kontrol et
    if detected.intersection(HIGH_RISK_SYMPTOMS):

        return {
            "level": "HIGH",
            "message": (
                "Some detected symptoms may require "
                "prompt professional medical evaluation."
            )
        }

    # Daha sonra orta riskli belirtileri kontrol et
    if detected.intersection(MEDIUM_RISK_SYMPTOMS):

        return {
            "level": "MEDIUM",
            "message": (
                "The detected symptoms may require "
                "medical attention, especially if they "
                "persist or become more severe."
            )
        }

    # Diğer durumlar
    return {
        "level": "LOW",
        "message": (
            "No high-risk symptom was detected. "
            "Consider professional medical advice "
            "if symptoms persist or worsen."
        )
    }