import re
from pathlib import Path
from tabulate import tabulate

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+).*?train loss :: ([0-9.]+), train :: ([0-9.]+), dev :: ([0-9.]+)"
)

def parse_logfile(path):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epoch = int(m.group(1))
                results[epoch] = {
                    "train": float(m.group(3)),
                    "dev": float(m.group(4)),
                }
    return results

def choose_two_files(logfiles):
    print("\nLogfiles:")
    for i, lf in enumerate(logfiles):
        print(f"[{i}] {lf.name}")

    a = input("\nErstes Logfile (Enter = 0): ").strip()
    b = input("Zweites Logfile (Enter = 1): ").strip()

    idx1 = int(a) if a else 0
    idx2 = int(b) if b else 1

    return logfiles[idx1], logfiles[idx2]

def short(name):
    return name[:15]

def print_diff_table(log1, log2, name1, name2):
    name1s = short(name1)
    name2s = short(name2)

    print(f"\nDifferenz = {name2s} - {name1s}\n")

    epochs = sorted(set(log1.keys()) | set(log2.keys()))
    rows = []

    for ep in epochs:
        a = log1.get(ep, {"train": None, "dev": None})
        b = log2.get(ep, {"train": None, "dev": None})

        if a["train"] is None or b["train"] is None:
            rows.append([ep, "-", "-"])
        else:
            rows.append([
                ep,
                b["train"] - a["train"],
                b["dev"] - a["dev"]
            ])

    headers = ["Epoch", "Δ train", "Δ dev"]

    print(tabulate(rows, headers=headers, floatfmt=".3f"))
    print()

def main():
    log_dir = Path("logs")
    logfiles = sorted(log_dir.glob("*.log"))

    lf1, lf2 = choose_two_files(logfiles)

    log1 = parse_logfile(lf1)
    log2 = parse_logfile(lf2)

    print_diff_table(log1, log2, lf1.name, lf2.name)

if __name__ == "__main__":
    main()
