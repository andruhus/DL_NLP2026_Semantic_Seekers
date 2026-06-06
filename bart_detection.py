import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW


TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        probabilities = self.sigmoid(logits)
        return probabilities


def transform_data(dataset, max_length=512, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

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
        unused = {12, 19, 20, 23, 27}
        valid_ids = sorted(set(range(1, 32)) - unused)
        labels = []
        for type_str in dataset["paraphrase_type_ids"]:
            type_set = set(eval(str(type_str)))
            labels.append([1 if t in type_set else 0 for t in valid_ids])
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        ds = TensorDataset(input_ids, attention_mask, labels_tensor)
    else:
        ds = TensorDataset(input_ids, attention_mask)

    return DataLoader(ds, batch_size=16, shuffle=shuffle)


def train_model(model, train_data, dev_data, device):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()
    best_acc = 0.0

    for epoch in range(5):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            probs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        acc, mcc = evaluate_model(model, dev_data, device)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, dev_acc={acc:.3f}, MCC={mcc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/bart_detection_best.pt")

    model.load_state_dict(torch.load("models/bart_detection_best.pt"))
    return model



def test_model(model, test_data, test_ids, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing"):
            if len(batch) == 3:
                input_ids, attention_mask, _ = batch
            else:
                input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            probs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (probs > 0.5).int().cpu().numpy().tolist()
            all_preds.extend(preds)
    return pd.DataFrame({"id": test_ids.tolist(), "Predicted_Paraphrase_Types": all_preds})


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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > 0.5).int()

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
        correct_predictions = np.sum(true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx])
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        # compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:, label_idx], predicted_labels_np[:, label_idx])
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
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset, shuffle=False)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device)

    print("Training finished.")

    accuracy, matthews_corr = evaluate_model(model, dev_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
