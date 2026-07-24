import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW


TQDM_DISABLE = False


def transform_data(dataset, max_length=256, shuffle=True):
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
    return DataLoader(ds, batch_size=8, shuffle=shuffle)


def train_model(model, train_data, dev_data, device, tokenizer):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    best_bleu = 0.0

    for epoch in range(5):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        bleu = evaluate_model(model, dev_data, device, tokenizer)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, penalized_BLEU={bleu:.3f}")

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), "models/bart_generation_best.pt")

    model.load_state_dict(torch.load("models/bart_generation_best.pt"))
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
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv")

    # You might do a split of the train data into train/validation set here
    # ...

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_dataset, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
