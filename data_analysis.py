import pandas as pd


# ==========================================
# 1. VERİ SETİNİ YÜKLE
# ==========================================

data = pd.read_csv("data/symptom2disease.csv")


# ==========================================
# 2. TEMEL BİLGİLER
# ==========================================

print("\n========== DATASET INFORMATION ==========\n")

print("Total samples:", len(data))
print("Total columns:", len(data.columns))

print("\nColumns:")
print(data.columns.tolist())


# ==========================================
# 3. HASTALIK SAYISI
# ==========================================

print("\n========== DISEASE INFORMATION ==========\n")

print("Number of diseases:", data["label"].nunique())

print("\nSamples per disease:")
print(data["label"].value_counts())


# ==========================================
# 4. EKSİK VERİ KONTROLÜ
# ==========================================

print("\n========== MISSING VALUES ==========\n")

print(data.isnull().sum())


# ==========================================
# 5. DUPLICATE KONTROLÜ
# ==========================================

print("\n========== DUPLICATE DATA ==========\n")

print("Duplicate rows:", data.duplicated().sum())

print("Duplicate texts:", data["text"].duplicated().sum())


# ==========================================
# 6. BOŞ METİNLER
# ==========================================

print("\n========== EMPTY TEXT ==========\n")

empty_texts = data["text"].isna().sum()

print("Empty texts:", empty_texts)


# ==========================================
# 7. METİN UZUNLUKLARI
# ==========================================

data["text_length"] = data["text"].str.len()

print("\n========== TEXT LENGTH ==========\n")

print("Average text length:", data["text_length"].mean())
print("Shortest text:", data["text_length"].min())
print("Longest text:", data["text_length"].max())


# ==========================================
# 8. EN KISA VE EN UZUN ÖRNEKLER
# ==========================================

print("\n========== SHORTEST TEXT ==========\n")

shortest = data.loc[data["text_length"].idxmin()]

print("Disease:", shortest["label"])
print("Text:", shortest["text"])


print("\n========== LONGEST TEXT ==========\n")

longest = data.loc[data["text_length"].idxmax()]

print("Disease:", longest["label"])
print("Text:", longest["text"])