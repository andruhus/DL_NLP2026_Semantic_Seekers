import subprocess
import json
import os
import re
from datetime import datetime

#############################################
# Helper: Run training and extract dev acc
#############################################

def run_job(params):
    """
    Runs your training script with given hyperparameters.
    Returns the dev accuracy parsed from the log output.
    """

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
        "--use_gpu"
        "--local_files_only"
    ]

    print("\n>>> Starte Job:", " ".join(cmd))

    os.makedirs("hyperopt_logs", exist_ok=True)
    logfile = f"hyperopt_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(logfile, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=f)
        process.wait()

    # Parse dev accuracy
    with open(logfile, "r") as f:
        content = f.read()

    match = re.findall(r"dev :: ([0-9.]+)", content)
    if match:
        dev_acc = float(match[-1])
        print(f"✓ Dev-Accuracy gefunden: {dev_acc}")
        return dev_acc

    print("⚠ Keine Dev-Accuracy gefunden!")
    return -1.0


#############################################
# Hyperparameter Search Space
#############################################

SEARCH_SPACE = {
    #"lr": [1e-5, 2e-5, 3e-5],#2
    #"batch_size": [8, 16, 32],#16
    "warmup_ratio": [0.0, 0.05, 0.1],#
    "weight_decay": [0.0, 0.001, 0.005],
    "label_smoothing": [0.0, 0.05, 0.01],
    "classifier_dropout": [0.1, 0.2, 0.3]
}

#############################################
# Sequential Hyperparameter Optimization
#############################################

def sequential_search(LR=True,BATCH=True):
    best = {}
    best_score = -1
    params = {
        "lr": 2e-5,
        "batch_size": 16,
        "warmup_ratio": 0.1,
        "weight_decay": 0.00,
        "label_smoothing": 0.00,
        "classifier_dropout": 0.3
    }
    if LR:
        # 1) LR + Epochs
        print("\n=== Schritt 1: Learning Rate ===")
        for lr in SEARCH_SPACE["lr"]:
            params["lr"] = lr
            score = run_job(params)
            if score > best_score:
                best_score = score
                best.update(params)
    if BATCH:
        # 2) Batch Size
        print("\n=== Schritt 2: Batch Size ===")
        for bs in SEARCH_SPACE["batch_size"]:
            params = best.copy()
            params["batch_size"] = bs
            score = run_job(params)
            if score > best_score:
                best_score = score
                best.update(params)
    # 3) warmup ratio
    print("\n=== Schritt 3: Warmup Ratio ===")
    for wr in SEARCH_SPACE["warmup_ratio"]:
        params["warmup_ratio"] = wr
        score = run_job(params)
        if score > best_score:
            best_score = score
            best.update(params)


    # 4) Weight Decay
    print("\n=== Schritt 4: Weight Decay ===")
    for wd in SEARCH_SPACE["weight_decay"]:
        params = best.copy()
        params["weight_decay"] = wd
        score = run_job(params)
        if score > best_score:
            best_score = score
            best.update(params)

    # 5) Classifier Dropout
    print("\n=== Schritt 5: Classifier Dropout ===")
    for dp in SEARCH_SPACE["classifier_dropout"]:
        params = best.copy()
        params["classifier_dropout"] = dp
        score = run_job(params)
        if score > best_score:
            best_score = score
            best.update(params)

    # 6) Label Smoothing
    print("\n=== Schritt 6: Label Smoothing ===")
    for ls in SEARCH_SPACE["label_smoothing"]:
        params = best.copy()
        params["label_smoothing"] = ls
        score = run_job(params)
        if score > best_score:
            best_score = score
            best.update(params)

    print("\n\n=== BESTE KONFIGURATION GEFUNDEN ===")
    print(json.dumps(best, indent=4))
    print("Beste Dev-Accuracy:", best_score)

    with open("best_hyperparams.json", "w") as f:
        json.dump(best, f, indent=4)


sequential_search(False,False)