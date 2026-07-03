import argparse
import ast
import random
from pathlib import Path

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

# ETPC assigns IDs 1--31, but these five IDs are not used by the dataset.
PARAPHRASE_TYPE_IDS = tuple(
    type_id for type_id in range(1, 32) if type_id not in {12, 19, 20, 23, 27}
)


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


def transform_data(dataset, max_length=512, shuffle=False, batch_size=16):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [6, 6, 6, 25, 25, 29]. This means that
    the sentence pair contains type 6, 25, and 29. Turn this into a binary form, where the
    label becomes [0, 0, 0, 0, 0, 1, ..., 1, 0, 0, 1, 0, 0].
    IMPORTANT: You will find that the dataset contains types up to 31, but some are not
    assigned. You need to drop 12, 19, 20, 23 and 27 when creating the binary labels.
    This way you should end up with a binary label of size 26.
    Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    encodings = tokenizer(
        dataset["sentence1"].astype(str).tolist(),
        dataset["sentence2"].astype(str).tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if "paraphrase_type_ids" not in dataset.columns:
        tensor_dataset = TensorDataset(input_ids, attention_mask)
    else:
        binary_labels = []
        for row_number, value in enumerate(dataset["paraphrase_type_ids"]):
            try:
                type_ids = ast.literal_eval(value) if isinstance(value, str) else value
                type_ids = {int(type_id) for type_id in type_ids}
            except (TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"Invalid paraphrase_type_ids value in row {row_number}: {value!r}"
                ) from exc

            binary_labels.append(
                [float(type_id in type_ids) for type_id in PARAPHRASE_TYPE_IDS]
            )

        labels = torch.tensor(binary_labels, dtype=torch.float32)
        tensor_dataset = TensorDataset(input_ids, attention_mask, labels)

    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_data, dev_data, device):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.BCELoss()
    checkpoint_path = Path("models/bart_detection_best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_dev_accuracy = float("-inf")

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        prediction_count = 0

        for batch in tqdm(
            train_data, desc=f"Detection epoch {epoch + 1}/5", disable=TQDM_DISABLE
        ):
            input_ids, attention_mask, labels = (tensor.to(device) for tensor in batch)

            optimizer.zero_grad()
            probabilities = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(probabilities, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (probabilities.detach() > 0.5).to(labels.dtype)
            correct_predictions += (predictions == labels).sum().item()
            prediction_count += labels.numel()

        train_loss = total_loss / max(len(train_data), 1)
        train_accuracy = correct_predictions / max(prediction_count, 1)
        dev_accuracy, dev_mcc = evaluate_model(model, dev_data, device)
        print(
            f"Epoch {epoch + 1}: loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.3f}, dev_acc={dev_accuracy:.3f}, "
            f"dev_MCC={dev_mcc:.3f}"
        )

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model



def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    was_training = model.training
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Detection test", disable=TQDM_DISABLE):
            input_ids, attention_mask = batch[:2]
            probabilities = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            )
            predictions.extend((probabilities > 0.5).int().cpu().tolist())

    model.train(was_training)
    ids = test_ids.tolist() if hasattr(test_ids, "tolist") else list(test_ids)
    if len(ids) != len(predictions):
        raise ValueError(
            f"Received {len(ids)} test IDs but generated {len(predictions)} predictions."
        )

    return pd.DataFrame(
        {"id": ids, "Predicted_Paraphrase_Types": predictions},
        columns=["id", "Predicted_Paraphrase_Types"],
    )


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
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # Use the 80/20 split from the project baseline and keep the held-out rows out
    # of training so that the reported development metrics are meaningful.
    shuffled_dataset = train_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_index = int(0.8 * len(shuffled_dataset))
    train_dataset = shuffled_dataset.iloc[:split_index].reset_index(drop=True)
    dev_dataset = shuffled_dataset.iloc[split_index:].reset_index(drop=True)

    train_data = transform_data(train_dataset, shuffle=True)
    dev_data = transform_data(dev_dataset, shuffle=False)
    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device)

    print("Training finished.")

    accuracy, matthews_corr = evaluate_model(model, dev_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    Path("predictions/bart").mkdir(parents=True, exist_ok=True)
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
