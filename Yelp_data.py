import pandas as pd
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Yelp Review Full Dataset herunterladen
# ---------------------------------------------------------

url = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz"
filename = "data/yelp_review_full_csv.tgz"

print("Downloading Yelp Review Full Dataset...")
urllib.request.urlretrieve(url, filename)

print("Extracting...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()

# Jetzt liegen train.csv und test.csv im Ordner yelp_review_full_csv/

# ---------------------------------------------------------
# 2. CSVs laden
# ---------------------------------------------------------

train_df = pd.read_csv("data/yelp_review_full_csv/train.csv", header=None, names=["sentiment", "sentence"])
test_df  = pd.read_csv("data/yelp_review_full_csv/test.csv",  header=None, names=["sentiment", "sentence"])

# IDs hinzufügen
train_df["id"] = range(len(train_df))
test_df["id"]  = range(len(test_df))

# Spalten sortieren
train_df = train_df[["id", "sentence", "sentiment"]]
test_df  = test_df[["id", "sentence", "sentiment"]]

# ---------------------------------------------------------
# 3. Train → 80% train, 20% dev
# ---------------------------------------------------------

train_split, dev_split = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# ---------------------------------------------------------
# 4. Kleine Splits (je 5%)
# ---------------------------------------------------------

train_small = test_df.sample(frac=0.05, random_state=42)
dev_small   = dev_split.sample(frac=0.05, random_state=42)

# ---------------------------------------------------------
# 5. CSV-Dateien speichern
# ---------------------------------------------------------

train_split.to_csv("data/yelp-train.csv", index=False)
dev_split.to_csv("data/yelp-dev.csv", index=False)
train_small.to_csv("data/yelp-train-small.csv", index=False)
dev_small.to_csv("data/yelp-dev-small.csv", index=False)

print("Fertig! Alle CSV-Dateien wurden erzeugt.")
