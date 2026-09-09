#!/usr/bin/env bash
# Build the SNLI and PAWS triplet caches used by the STS transfer-pretraining
# experiments (--nli_pretrain_epochs / --paws_pretrain_epochs).
#
# Why a script: the download must run from OUTSIDE the repository directory.
# This project ships its own datasets.py, which shadows the HuggingFace `datasets`
# package on the import path — importing it from the repo root silently fails.
# The script cd's to a temp directory, writes plain JSON caches into
# data/nli_cache/, and training then reads those with no HuggingFace import at all.
#
# Usage:   bash scripts/build_nli_cache.sh
# Needs:   pip install datasets     (allowed for Part 2), network access
# Output:  data/nli_cache/snli_triplets.json   (~149,145 triplets)
#          data/nli_cache/paws_pairs.json      (~49,401 pairs)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$REPO/data/nli_cache"
mkdir -p "$CACHE"

if ! python -c "import datasets" 2>/dev/null; then
    echo "The 'datasets' package is required:  pip install datasets" >&2
    exit 1
fi

echo "Building triplet caches in $CACHE"
cd "$(mktemp -d)"          # leave the repo so datasets.py cannot shadow the package

REPO="$REPO" CACHE="$CACHE" python - <<'PY'
import json, os
from collections import defaultdict
from datasets import load_dataset

cache = os.environ["CACHE"]

snli_out = os.path.join(cache, "snli_triplets.json")
if os.path.exists(snli_out):
    print(f"  SNLI cache already present, skipping: {snli_out}")
else:
    ds = load_dataset("snli", cache_dir=cache, split="train")
    by = defaultdict(lambda: {"pos": [], "neg": []})
    for ex in ds:
        if   ex["label"] == 0: by[ex["premise"]]["pos"].append(ex["hypothesis"])
        elif ex["label"] == 2: by[ex["premise"]]["neg"].append(ex["hypothesis"])
    trip = [(p, v["pos"][0], v["neg"][0]) for p, v in by.items() if v["pos"] and v["neg"]]
    json.dump(trip, open(snli_out, "w"))
    print(f"  SNLI: {len(trip):,} triplets -> {snli_out}")

paws_out = os.path.join(cache, "paws_pairs.json")
if os.path.exists(paws_out):
    print(f"  PAWS cache already present, skipping: {paws_out}")
else:
    ds = load_dataset("paws", "labeled_final", cache_dir=cache, split="train")
    pairs = [(e["sentence1"], e["sentence2"], int(e["label"])) for e in ds]
    json.dump(pairs, open(paws_out, "w"))
    print(f"  PAWS: {len(pairs):,} pairs -> {paws_out}")
PY

echo "Done."
