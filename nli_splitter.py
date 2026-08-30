import csv
import random

def split_allnli(input_path, train_path, dev_path, dev_ratio=0.1, seed=11711):
    random.seed(seed)

    # AllNLI laden
    with open(input_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Shuffle
    random.shuffle(reader)

    # Split
    split_idx = int(len(reader) * (1 - dev_ratio))
    train_data = reader[:split_idx]
    dev_data = reader[split_idx:]

    # Header übernehmen
    fieldnames = reader[0].keys()

    # Train speichern
    with open(train_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_data)

    # Dev speichern
    with open(dev_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dev_data)

    print(f"Gesamt: {len(reader)}")
    print(f"Train: {len(train_data)}")
    print(f"Dev:   {len(dev_data)}")


if __name__ == "__main__":
    split_allnli(
        input_path="data/allnli.csv",
        train_path="data/allnli-train.csv",
        dev_path="data/allnli-dev.csv",
        dev_ratio=0.2
    )
