import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Yelp Review Full Dataset laden (Labels 0–4)
# ---------------------------------------------------------

print("Lade Yelp Review Full Dataset...")
ds = load_dataset("yelp_review_full")

train_ds = ds["train"]
test_ds = ds["test"]

# ---------------------------------------------------------
# 2. In DataFrames konvertieren
# ---------------------------------------------------------

train_df = pd.DataFrame({
    "id": range(len(train_ds)),
    "sentence": train_ds["text"],
    "sentiment": train_ds["label"]
})

test_df = pd.DataFrame({
    "id": range(len(test_ds)),
    "sentence": test_ds["text"],
    "sentiment": test_ds["label"]
})

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
dev_small = dev_split.sample(frac=0.05, random_state=42)

# ---------------------------------------------------------
# 5. CSV-Dateien speichern
# ---------------------------------------------------------

train_split.to_csv("data/yelp-train.csv", index=False)
dev_split.to_csv("data/yelp-dev.csv", index=False)
train_small.to_csv("data/yelp-train-small.csv", index=False)
dev_small.to_csv("data/yelp-dev-small.csv", index=False)

print("Fertig! CSV-Dateien wurden erzeugt.")
