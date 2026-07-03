import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW


TQDM_DISABLE = False


def transform_data(dataset, max_length=256, shuffle=False, batch_size=8):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase_type_ids.
    Return Data Loader.
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    separator = tokenizer.sep_token or "</s>"
    model_inputs = [
        f"{sentence} {separator} {segment_location} {separator} {type_ids}"
        for sentence, segment_location, type_ids in zip(
            dataset["sentence1"].astype(str),
            dataset["sentence1_segment_location"].astype(str),
            dataset["paraphrase_type_ids"].astype(str),
        )
    ]
    encodings = tokenizer(
        model_inputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if "sentence2" in dataset.columns:
        target_encodings = tokenizer(
            dataset["sentence2"].astype(str).tolist(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encodings["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
    else:
        # Keep the same three-tensor batch interface for labeled and unlabeled data.
        # These placeholders are ignored by test_model.
        labels = torch.full_like(input_ids, -100)

    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    optimizer = AdamW(model.parameters(), lr=1e-5)
    checkpoint_path = Path("models/bart_generation_best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_dev_bleu = float("-inf")

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for batch in tqdm(
            train_data, desc=f"Generation epoch {epoch + 1}/5", disable=TQDM_DISABLE
        ):
            input_ids, attention_mask, labels = (tensor.to(device) for tensor in batch)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_data), 1)
        dev_bleu = evaluate_model(model, dev_data, device, tokenizer)
        print(
            f"Epoch {epoch + 1}: loss={train_loss:.4f}, "
            f"penalized_BLEU={dev_bleu:.3f}"
        )

        if dev_bleu > best_dev_bleu:
            best_dev_bleu = dev_bleu
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    was_training = model.training
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generation test", disable=TQDM_DISABLE):
            input_ids, attention_mask = batch[:2]
            generated_ids = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )
            predictions.extend(
                tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for generated in generated_ids
            )

    model.train(was_training)
    ids = test_ids.tolist() if hasattr(test_ids, "tolist") else list(test_ids)
    if len(ids) != len(predictions):
        raise ValueError(
            f"Received {len(ids)} test IDs but generated {len(predictions)} sentences."
        )

    return pd.DataFrame(
        {"id": ids, "Generated_sentence2": predictions},
        columns=["id", "Generated_sentence2"],
    )


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and
    the column "sentence2" contains all target sentences
    """
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, shuffle=False)
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
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")

    shuffled_dataset = train_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_index = int(0.8 * len(shuffled_dataset))
    train_dataset = shuffled_dataset.iloc[:split_index].reset_index(drop=True)
    dev_dataset = shuffled_dataset.iloc[split_index:].reset_index(drop=True)

    train_data = transform_data(train_dataset, shuffle=True)
    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_dataset, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    Path("predictions/bart").mkdir(parents=True, exist_ok=True)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
