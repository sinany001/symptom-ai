import pandas as pd
import re
from collections import Counter

# Dataset'i yükle
data = pd.read_csv("data/symptom2disease.csv")

# Duplicate kayıtları temizle
data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Bütün metinleri birleştir
all_text = " ".join(data["text"].astype(str).tolist()).lower()

# Türkçe "ve" gibi dataset içindeki gereksiz kelimeleri de temizle
stop_words = {
    "and", "i", "my", "a", "an", "the", "been", "have",
    "has", "had", "is", "are", "am", "to", "of", "in",
    "on", "with", "for", "that", "this", "it", "me",
    "also", "very", "really", "quite", "some", "there",
    "when", "as", "from", "can", "all", "these", "those",
    "or", "so", "up", "out", "over", "like", "get", "got",
    "having", "feel", "feeling", "experiencing", "recently",
    "often", "frequently", "occasionally", "ve"
}

# Metni kelimelere ayır
words = re.findall(r"\b[a-z]+\b", all_text)

# Stop-word'leri kaldır
filtered_words = [
    word for word in words
    if word not in stop_words
]

# Bigram ve trigram oluştur
bigrams = []
trigrams = []

for i in range(len(filtered_words) - 1):
    bigrams.append(
        filtered_words[i] + " " + filtered_words[i + 1]
    )

for i in range(len(filtered_words) - 2):
    trigrams.append(
        filtered_words[i]
        + " "
        + filtered_words[i + 1]
        + " "
        + filtered_words[i + 2]
    )

bigram_counts = Counter(bigrams)
trigram_counts = Counter(trigrams)

print("\n========================================")
print("TOP 50 TWO-WORD PHRASES")
print("========================================")

for phrase, count in bigram_counts.most_common(50):
    print(f"{phrase:35} {count}")

print("\n========================================")
print("TOP 50 THREE-WORD PHRASES")
print("========================================")

for phrase, count in trigram_counts.most_common(50):
    print(f"{phrase:35} {count}")