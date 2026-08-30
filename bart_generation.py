import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from paraphrase_generation.lr_schedulers import (
    ConstantLearningRate,
    CosineDecay,
    InverseSquareRoot,
    LinearDecay,
    MetricDependent,
    StepDecay,
)
from optimizer import AdamW


TQDM_DISABLE = False


def remove_dev_overlap(train_dataset, dev_dataset):
    """Remove training rows whose ETPC ID occurs in the development split."""
    if "id" not in train_dataset.columns or "id" not in dev_dataset.columns:
        raise ValueError("Both ETPC datasets must contain an 'id' column.")

    train_ids = train_dataset["id"].astype("string").str.strip().str.lower()
    dev_ids = dev_dataset["id"].astype("string").str.strip().str.lower()
    overlap_mask = train_ids.notna() & train_ids.isin(dev_ids.dropna())

    removed_rows = int(overlap_mask.sum())
    removed_unique_ids = int(train_ids[overlap_mask].nunique())
    filtered_train = train_dataset.loc[~overlap_mask].reset_index(drop=True)

    remaining_train_ids = (
        filtered_train["id"].astype("string").str.strip().str.lower().dropna()
    )
    remaining_overlap = set(remaining_train_ids) & set(dev_ids.dropna())
    assert not remaining_overlap, "ETPC train/dev ID leakage remains after filtering."

    print(
        f"Removed {removed_rows} training rows "
        f"({removed_unique_ids} unique IDs) overlapping with the dev set."
    )
    return filtered_train


def transform_data(dataset, max_length=256, batch_size=8, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    inputs = [
        str(r["sentence1"]) + " </s> " +
        str(r["sentence1_segment_location"]) + " </s> " +
        str(r["paraphrase_type_ids"])
        for _, r in dataset.iterrows()
    ]
    input_enc = tokenizer(
        inputs, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = input_enc["input_ids"]
    attention_mask = input_enc["attention_mask"]

    has_targets = "sentence2" in dataset.columns
    if has_targets:
        targets = dataset["sentence2"].tolist()
        target_enc = tokenizer(
            targets, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        labels = target_enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
    else:
        labels = torch.full((input_ids.size(0),), -100, dtype=torch.long)

    ds = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model,
    train_data,
    dev_dataset,
    device,
    tokenizer,
    lr_sched,
    checkpoint_path,
    epochs=5,
    batch_size=8,
):
    optimizer = AdamW(
        model.parameters(), lr=lr_sched.lr(), lr_sched=lr_sched,
    )
    best_bleu = -float("inf")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if not lr_sched.requires_metric:
                lr_sched.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        bleu = evaluate_model(
            model,
            dev_dataset,
            device,
            tokenizer,
            batch_size=batch_size,
        )
        if lr_sched.requires_metric:
            lr_sched.step(bleu)
        print(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
            f"penalized_BLEU={bleu:.3f}, next_lr={lr_sched.lr():.8g}"
        )

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing"):
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model.generate(
                input_ids, attention_mask=attention_mask,
                max_length=128, num_beams=5, early_stopping=True,
            )
            decoded = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            predictions.extend(decoded)
    return pd.DataFrame({"id": test_ids.tolist(), "Generated_sentence2": predictions})


def evaluate_model(model, test_data, device, tokenizer, batch_size=8):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and 
    the column "sentence2" contains all target sentences
    """
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(
        test_data, batch_size=batch_size, shuffle=False,
    )
    with torch.no_grad():
        for batch in dataloader: 
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )
            
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            
            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    model.train()
    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")
    

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--scheduler_mode",
        choices=(
            "constant",
            "step",
            "cosine",
            "linear",
            "inverse_sqrt",
            "metric",
            "compare",
        ),
        default="compare",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--step_decay_epochs", type=int, default=1)
    parser.add_argument("--step_gamma", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--metric_factor", type=float, default=0.5)
    parser.add_argument("--metric_patience", type=int, default=1)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning_rate must be a finite, positive number")
    if not math.isfinite(args.min_lr) or args.min_lr < 0.0:
        parser.error("--min_lr must be a finite, non-negative number")
    if args.min_lr > args.learning_rate:
        parser.error("--min_lr cannot exceed --learning_rate")
    if args.step_decay_epochs < 1:
        parser.error("--step_decay_epochs must be at least 1")
    if not math.isfinite(args.step_gamma) or not 0.0 < args.step_gamma <= 1.0:
        parser.error("--step_gamma must be in the interval (0, 1]")
    if args.warmup_steps < 0:
        parser.error("--warmup_steps cannot be negative")
    if (
        not math.isfinite(args.metric_factor)
        or not 0.0 < args.metric_factor < 1.0
    ):
        parser.error("--metric_factor must be in the interval (0, 1)")
    if args.metric_patience < 0:
        parser.error("--metric_patience cannot be negative")
    return args


def create_experiments(
    scheduler_mode,
    learning_rate,
    min_lr,
    total_steps,
    steps_per_epoch,
    step_decay_epochs,
    step_gamma,
    warmup_steps,
    metric_factor,
    metric_patience,
):
    all_experiments = (
        (
            "Constant learning rate",
            "constant",
            ConstantLearningRate(learning_rate),
        ),
        (
            "Step decay",
            "step",
            StepDecay(
                learning_rate,
                step_size=steps_per_epoch * step_decay_epochs,
                gamma=step_gamma,
                min_lr=min_lr,
            ),
        ),
        (
            "Cosine decay",
            "cosine",
            CosineDecay(learning_rate, total_steps=total_steps, min_lr=min_lr),
        ),
        (
            "Linear decay",
            "linear",
            LinearDecay(learning_rate, total_steps=total_steps, min_lr=min_lr),
        ),
        (
            "Inverse square root",
            "inverse_sqrt",
            InverseSquareRoot(
                learning_rate, warmup_steps=warmup_steps, min_lr=min_lr,
            ),
        ),
        (
            "Metric dependent",
            "metric",
            MetricDependent(
                learning_rate,
                factor=metric_factor,
                patience=metric_patience,
                min_lr=min_lr,
            ),
        ),
    )

    selected = (
        all_experiments
        if scheduler_mode == "compare"
        else tuple(
            experiment
            for experiment in all_experiments
            if experiment[1] == scheduler_mode
        )
    )
    return [
        (
            name,
            lr_sched,
            f"models/bart_generation_{mode}_best.pt",
        )
        for name, mode, lr_sched in selected
    ]


def run_experiment(
    name,
    lr_sched,
    checkpoint_path,
    train_data,
    dev_dataset,
    device,
    tokenizer,
    seed,
    epochs,
    batch_size,
):
    seed_everything(seed)
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", local_files_only=True,
    )
    model.to(device)

    print(f"\nTraining {name}...")
    model = train_model(
        model,
        train_data,
        dev_dataset,
        device,
        tokenizer,
        lr_sched,
        checkpoint_path,
        epochs=epochs,
        batch_size=batch_size,
    )
    bleu = evaluate_model(
        model,
        dev_dataset,
        device,
        tokenizer,
        batch_size=batch_size,
    )
    result = {
        "scheduler": name,
        "dev_penalized_bleu": bleu,
        "checkpoint_path": checkpoint_path,
    }
    return model, result


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large", local_files_only=True,
    )

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv")
    train_dataset = remove_dev_overlap(train_dataset, dev_dataset)

    train_data = transform_data(train_dataset, batch_size=args.batch_size)
    test_data = transform_data(
        test_dataset, batch_size=args.batch_size, shuffle=False,
    )
    steps_per_epoch = len(train_data)
    total_steps = steps_per_epoch * args.epochs
    experiments = create_experiments(
        args.scheduler_mode,
        args.learning_rate,
        args.min_lr,
        total_steps,
        steps_per_epoch,
        args.step_decay_epochs,
        args.step_gamma,
        args.warmup_steps,
        args.metric_factor,
        args.metric_patience,
    )

    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Optimizer updates per experiment: {total_steps}")

    results = []
    for name, lr_sched, checkpoint_path in experiments:
        model, result = run_experiment(
            name,
            lr_sched,
            checkpoint_path,
            train_data,
            dev_dataset,
            device,
            tokenizer,
            args.seed,
            args.epochs,
            args.batch_size,
        )
        results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = pd.DataFrame(results)
    displayed_comparison = comparison[["scheduler", "dev_penalized_bleu"]]
    print("\nDevelopment-set scheduler comparison:")
    print(
        displayed_comparison.to_string(
            index=False, float_format=lambda value: f"{value:.4f}",
        )
    )
    comparison_path = Path(
        "predictions/bart/bart-generation-lr-scheduler-comparison.csv"
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    displayed_comparison.to_csv(comparison_path, index=False)

    best_result = max(results, key=lambda result: result["dev_penalized_bleu"])
    print(
        f"Using {best_result['scheduler']} for test prediction "
        f"(dev penalized BLEU={best_result['dev_penalized_bleu']:.3f})."
    )
    prediction_model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", local_files_only=True,
    )
    prediction_model.load_state_dict(
        torch.load(best_result["checkpoint_path"], map_location=device)
    )
    prediction_model.to(device)

    test_ids = test_dataset["id"]
    test_results = test_model(
        test_data, test_ids, device, prediction_model, tokenizer,
    )
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
