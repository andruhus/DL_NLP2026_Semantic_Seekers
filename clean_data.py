import csv

INPUT_FILE = "input.csv"
OUTPUT_FILE = "cleaned.csv"

VALID_LABELS = {0, 1, 2, 3, 4}

def is_valid(row):
    try:
        label = int(row["label"])
        return label in VALID_LABELS
    except:
        return False

def clean_csv():
    good = 0
    bad = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

        writer.writeheader()

        for row in reader:
            if is_valid(row):
                writer.writerow(row)
                good += 1
            else:
                bad += 1

    print(f"✓ Fertig! {good} gültige Zeilen behalten.")
    print(f"✗ {bad} ungültige Zeilen entfernt.")

if __name__ == "__main__":
    clean_csv()
