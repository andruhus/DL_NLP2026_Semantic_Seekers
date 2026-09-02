import csv
import argparse

VALID_LABELS = {0, 1, 2, 3, 4}

def is_valid(row):
    try:
        label = int(row["sentiment"])
        return label in VALID_LABELS
    except:
        return False

def clean_csv(input_path, output_path):
    good = 0
    bad = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8", newline="") as outfile:

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

def main():
    parser = argparse.ArgumentParser(description="CSV Cleaner für SST-5 Labels (0–4).")
    parser.add_argument("--input", required=True, help="Pfad zur Eingabe-CSV")
    parser.add_argument("--output", required=True, help="Pfad zur Ausgabe-CSV")

    args = parser.parse_args()
    clean_csv(args.input, args.output)

if __name__ == "__main__":
    main()
