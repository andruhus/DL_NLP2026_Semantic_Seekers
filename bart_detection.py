import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel

from optimizer import AdamW
from paraphrase_detection.weighted_bce import (
    compute_pos_weights,
    count_label_examples,
    create_weighted_bce_loss,
    encode_paraphrase_labels,
    print_label_statistics,
    verify_positive_training_examples,
)


TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained(
            "facebook/bart-large", local_files_only=True,
        )
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)
        return logits


def transform_data(dataset, max_length=512, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large", local_files_only=True,
    )

    combined = [
        str(r["sentence1"]) + " </s> " + str(r["sentence2"])
        for _, r in dataset.iterrows()
    ]
    encoding = tokenizer(
        combined, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    has_labels = "paraphrase_type_ids" in dataset.columns
    if has_labels:
        labels_tensor = encode_paraphrase_labels(dataset)
        ds = TensorDataset(input_ids, attention_mask, labels_tensor)
    else:
        ds = TensorDataset(input_ids, attention_mask)

    return DataLoader(ds, batch_size=16, shuffle=shuffle)


def train_model(
    model,
    train_data,
    dev_data,
    criterion,
    device,
    checkpoint_path,
    epochs=5,
):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = criterion.to(device)
    best_acc = -float("inf")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}", disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        acc, mcc = evaluate_model(model, dev_data, device)
        print(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
            f"dev_acc={acc:.3f}, MCC={mcc:.3f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def test_model(model, test_data, test_ids, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing", disable=TQDM_DISABLE):
            if len(batch) == 3:
                input_ids, attention_mask, _ = batch
            else:
                input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy().tolist()
            all_preds.extend(preds)
    return pd.DataFrame({
        "id": test_ids.tolist(),
        "Predicted_Paraphrase_Types": all_preds,
    })


def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the 26 paraphrase types the model has predicted correctly for each data point..
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (torch.sigmoid(logits) > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        # compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(
            true_labels_np[:, label_idx], predicted_labels_np[:, label_idx]
        )
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    matthews_coefficient = np.mean(matthews_coefficients)
    model.train()
    return accuracy, matthews_coefficient


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
    parser.add_argument(
        "--loss_mode",
        choices=("unweighted", "weighted", "compare"),
        default="compare",
        help="Train with unweighted BCE, weighted BCE, or both for comparison.",
    )
    args = parser.parse_args()
    return args


def create_experiments(loss_mode, pos_weights):
    experiments = []
    if loss_mode in {"unweighted", "compare"}:
        experiments.append(
            (
                "Unweighted BCE",
                nn.BCEWithLogitsLoss(),
                "models/bart_detection_unweighted_best.pt",
            )
        )
    if loss_mode in {"weighted", "compare"}:
        experiments.append(
            (
                "Weighted BCE",
                create_weighted_bce_loss(pos_weights),
                "models/bart_detection_weighted_best.pt",
            )
        )
    return experiments


def run_experiment(
    name, criterion, checkpoint_path, train_data, dev_data, device, seed,
):
    # Reset the seed so compared models start from the same initialization and
    # see the same shuffled training order.
    seed_everything(seed)
    model = BartWithClassifier()
    model.to(device)
    print(f"\nTraining {name}...")
    model = train_model(
        model,
        train_data,
        dev_data,
        criterion,
        device,
        checkpoint_path,
    )
    accuracy, matthews_corr = evaluate_model(model, dev_data, device)
    result = {
        "loss": name,
        "dev_accuracy": accuracy,
        "dev_mcc": matthews_corr,
    }
    return model, result


def finetune_paraphrase_detection(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset, shuffle=False)
    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    # These statistics are computed exclusively from the training split.
    train_labels = encode_paraphrase_labels(train_dataset)
    positive_counts, negative_counts = count_label_examples(train_labels)
    verify_positive_training_examples(positive_counts)
    pos_weights = compute_pos_weights(positive_counts, negative_counts)
    print_label_statistics(positive_counts, negative_counts, pos_weights)

    experiments = create_experiments(args.loss_mode, pos_weights)
    results = []
    prediction_model = None
    for experiment_index, experiment in enumerate(experiments):
        name, criterion, checkpoint_path = experiment
        model, result = run_experiment(
            name,
            criterion,
            checkpoint_path,
            train_data,
            dev_data,
            device,
            args.seed,
        )
        results.append(result)

        # The weighted experiment is last in comparison mode, so earlier large
        # BART models can be released before the next one is constructed.
        if experiment_index == len(experiments) - 1:
            prediction_model = model
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    comparison = pd.DataFrame(results)
    print("\nDevelopment-set comparison:")
    print(
        comparison.to_string(
            index=False, float_format=lambda value: f"{value:.4f}",
        )
    )

    test_ids = test_dataset["id"]
    test_results = test_model(prediction_model, test_data, test_ids, device)
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
