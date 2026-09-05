import pandas as pd
import zipfile
import urllib.request
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. SST-5 Augmented Dataset herunterladen
# ---------------------------------------------------------

url = "https://huggingface.co/datasets/SetFit/sst5/resolve/main/data.zip"
filename = "data/sst5_augmented.zip"

print("Downloading SST-5 Augmented Dataset...")
urllib.request.urlretrieve(url, filename)

print("Extracting ZIP...")
with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall("data/sst5_augmented/")

# Die extrahierten Dateien liegen nun in data/sst5_augmented/data/

# ---------------------------------------------------------
# 2. CSVs laden
# ---------------------------------------------------------

train_df = pd.read_csv("data/sst5_augmented/data/train.csv")

# SetFit hat Format: sentence, label
train_df = train_df.rename(columns={"label": "sentiment"})

# IDs hinzufügen
train_df["id"] = range(len(train_df))

# Spalten sortieren
train_df = train_df[["id", "sentence", "sentiment"]]
test_df  = test_df[["id", "sentence", "sentiment"]]

# ---------------------------------------------------------
# 3. Original DEV/TEST laden (diese dürfen NICHT ins Training!)
# ---------------------------------------------------------

dev_original  = pd.read_csv("data/sst-sentiment-dev.csv")
test_original = pd.read_csv("data/sst-sentiment-test-student.csv")
