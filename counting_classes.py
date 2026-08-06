from collections import Counter

def compute_class_counts(csv_path):
    """"zählen der counts für den weighted loss
    (geht wahrscheinlich eleganter weil wir oben die daten schonmal eingelesen haben aber egal jetzt)"""
    sentiments = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentiments.append(int(row["sentiment"]))

    counts = Counter(sentiments)

    # Reihenfolge 0,1,2,3,4 sicherstellen
    class_counts = torch.tensor([counts[i] for i in range(5)], dtype=torch.float)

    return class_counts
import csv
import torch
from collections import Counter

def compute_class_counts(csv_path):
    sentiments = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentiments.append(int(row["sentiment"]))

    counts = Counter(sentiments)

    # Reihenfolge 0,1,2,3,4 sicherstellen
    class_counts = torch.tensor([counts[i] for i in range(5)], dtype=torch.float)

    return class_counts

# Beispiel:
csv_path = "./data/sst-sentiment-train.csv"
class_counts = compute_class_counts(csv_path)
print("Class counts:", class_counts)
weights = class_counts.max() / class_counts
#weights = weights.to(device)
print("Class weights:", weights)

