import subprocess
import json
import os
import re
from datetime import datetime

#############################################
# Helper: Run training and extract dev acc
#############################################

def run_job(params):
    cmd = [
        "python", "multitask_classifier.py",
        "--task", "sst",
        "--option", "finetune",
        "--lr", str(params["lr"]),
        "--batch_size", str(params["batch_size"]),
        "--warmup_ratio", str(params["warmup_ratio"]),
        "--weight_decay", str(params["weight_decay"]),
        "--label_smoothing", str(params["label_smoothing"]),
        "--classifier_dropout", str(params["classifier_dropout"]),
        "--use_gpu",
        "--local_files_only"
    ]

    print("\n>>> Starte Job:", " ".join(cmd))

    os.makedirs("hyperopt_logs", exist_ok=True)
    logfile = f"hyperopt_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(logfile, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=f)
        process.wait()

    with open(logfile, "r") as f:
        content = f.read()

    match = re.findall(r"dev :: ([0-9.]+)", content)
    if match:
        dev_accs = [float(m) for m in match]
        best_dev = max(dev_accs)
        print(f"✓ Beste Dev-Accuracy im Lauf: {best_dev}")
        return best_dev
    return -1.0


#############################################
# Hyperparameter Search Space
#############################################

SEARCH_SPACE = {
    "lr": [1e-5, 2e-5, 3e-5],
    "batch_size": [12, 16, 20],
    "warmup_ratio": [0.0, 0.05, 0.1],
    "weight_decay": [0.0, 0.001, 0.005],
    "label_smoothing": [0.0, 0.05, 0.01],
    "classifier_dropout": [0.1, 0.2, 0.3]
}

#############################################
# Sequential Hyperparameter Optimization
#############################################

def sequential_search(
    do_lr=True,
    do_batch=True,
    do_warmup=True,
    do_weight_decay=True,
    do_dropout=True,
    do_label_smoothing=True
):
    best = {}
    best_score = -1

    # 1) LR
    if do_lr:
        print("\n=== Schritt 1: Learning Rate ===")
        for lr in SEARCH_SPACE["lr"]:
            params = {
                "lr": lr,
                "batch_size": 16,
                "warmup_ratio": 0.1,
                "weight_decay": 0.00,
                "label_smoothing": 0.00,
                "classifier_dropout": 0.3
            }
            score = run_job(params)
            if score > best_score:
                best_score = score
                best = params.copy()
    else:
        # Default wie im Original
        best = {
            "lr": 2e-5,
            "batch_size": 16,
            "warmup_ratio": 0.1,
            "weight_decay": 0.00,
            "label_smoothing": 0.00,
            "classifier_dropout": 0.3
        }

    # 2) Batch Size
    if do_batch:
        print("\n=== Schritt 2: Batch Size ===")
        for bs in SEARCH_SPACE["batch_size"]:
            params = best.copy()
            params["batch_size"] = bs
            score = run_job(params)
            if score > best_score:
                best_score = score
                best = params.copy()

    # 3) Warmup Ratio
    if do_warmup:
        print("\n=== Schritt 3: Warmup Ratio ===")
        for wr in SEARCH_SPACE["warmup_ratio"]:
            params = best.copy()
            params["warmup_ratio"] = wr
            score = run_job(params)
            if score > best_score:
                best_score = score
                best = params.copy()

    # 4) Weight Decay
    if do_weight_decay:
        print("\n=== Schritt 4: Weight Decay ===")
        for wd in SEARCH_SPACE["weight_decay"]:
            params = best.copy()
            params["weight_decay"] = wd
            score = run_job(params)
            if score > best_score:
                best_score = score
                best = params.copy()

    # 5) Dropout
    if do_dropout:
        print("\n=== Schritt 5: Classifier Dropout ===")
        for dp in SEARCH_SPACE["classifier_dropout"]:
            params = best.copy()
            params["classifier_dropout"] = dp
            score = run_job(params)
            if score > best_score:
                best_score = score
                best = params.copy()

    # 6) Label Smoothing
    if do_label_smoothing:
        print("\n=== Schritt 6: Label Smoothing ===")
        for ls in SEARCH_SPACE["label_smoothing"]:
            params = best.copy()
            params["label_smoothing"] = ls
            score = run_job(params)
            if score > best_score:
                best_score = score
                best = params.copy()

    print("\n\n=== BESTE KONFIGURATION GEFUNDEN ===")
    print(json.dumps(best, indent=4))
    print("Beste Dev-Accuracy:", best_score)

    with open("best_hyperparams.json", "w") as f:
        json.dump(best, f, indent=4)


#############################################
# Run
#############################################

sequential_search(
    do_lr=False,
    do_batch=False,
    do_warmup=False,
    do_weight_decay=True,
    do_dropout=True,
    do_label_smoothing=False
)
