from risk_assessment import assess_risk


test_symptoms = [
    "fever",
    "cough",
    "breathing_difficulty"
]

result = assess_risk(test_symptoms)

print("\n========================================")
print("RISK ASSESSMENT")
print("========================================")

print(f"Risk Level: {result['level']}")
print(f"Message: {result['message']}")