#training möglichst wie im multitask file:

import argparse
import os
from pprint import pformat
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel

from datasets import load_allnli_data, AllNLIDataset


from evaluation import model_eval_multitask, test_model_multitask, model_eval_nli
from optimizer import AdamW

TQDM_DISABLE = True
import logging
from datetime import datetime
import os
from transformers import get_linear_schedule_with_warmup
#LOGFILE
def setup_logging(model_name):
    # Ordner für Logs anlegen
    os.makedirs("logs", exist_ok=True)

    # Zeitstempel für eindeutige Logfiles
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Log-Dateiname
    basename = os.path.basename(model_name)
    logfile = f"logs/{basename}_{timestamp}.log"

    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()  # optional: weiterhin auf der Konsole ausgeben
        ]
    )

    logging.info(f"Logfile gestartet für Modell: {model_name}")
    return logfile

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for these tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        self.nli_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE*2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 Klassen für NLI
        )
    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        output = self.bert(input_ids, attention_mask)
        return output['pooler_output']

    def predict_nli(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        emb1 = self.forward(input_ids1, attention_mask1)
        emb2 = self.forward(input_ids2, attention_mask2)

        # Klassisch: concat beider Embeddings
        pair_emb = torch.cat([emb1, emb2], dim=1)

        return self.nli_classifier(pair_emb)


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    logging.info(f"Saving the model to {filepath}.")


# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    nli_train_data = load_allnli_data(args.nli_train)  # musst du definieren
    nli_dev_data = load_allnli_data(args.nli_dev)

    # --- Debug: Label-Datentyp prüfen ---
    print("Train label example:", nli_train_data[0][2], type(nli_train_data[0][2]))
    print("Unique train labels:", {x[2] for x in nli_train_data[:50]})

    nli_train_dataloader = None
    nli_dev_dataloader = None
    nli_train_data = AllNLIDataset(nli_train_data, args)
    nli_dev_data = AllNLIDataset(nli_dev_data, args)
    nli_train_dataloader = DataLoader(
        nli_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=nli_train_data.collate_fn,
    )
    nli_dev_dataloader = DataLoader(
        nli_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=nli_dev_data.collate_fn,
    )
    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }

    config = SimpleNamespace(**config)

    separator = "-" * 30
    logging.info(separator)
    logging.info("    BERT Model Configuration")
    logging.info(separator)
    logging.info(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    logging.info(separator)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
###
### Hier kommt learning rate scheduler
###

    # Anzahl Trainingsschritte berechnen
    total_steps = args.epochs * len(nli_train_dataloader)

    # Warmup: 10% der Trainingsschritte
    warmup_steps = int(0.1 * total_steps)

    # Scheduler: Warmup + Linear Decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
###
###
###
    best_dev_acc = float("-inf")
    from collections import Counter


    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(nli_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE):
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                batch["token_ids_1"], batch["attention_mask_1"],
                batch["token_ids_2"], batch["attention_mask_2"],
                batch["labels"],
            )
            b_ids1, b_mask1 = b_ids1.to(device), b_mask1.to(device)
            b_ids2, b_mask2 = b_ids2.to(device), b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_nli(b_ids1, b_mask1, b_ids2, b_mask2)
            loss = F.cross_entropy(logits, b_labels, label_smoothing=0.0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / num_batches

        nli_train_acc, _, _ = (
            model_eval_nli(
                nli_train_dataloader,
                model=model,
                device=device,
            )
        )

        nli_dev_acc, _, _ = (
            model_eval_nli(
                nli_dev_dataloader,
                model=model,
                device=device
            )
        )

        train_acc, dev_acc = {
            "nli": (nli_train_acc, nli_dev_acc)
        }["nli"]

        logging.info(
            f"Epoch {epoch+1:02} (nli): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)



def get_args():
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )

    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()

    # Dataset paths
    parser.add_argument("--nli_train", type=str, default="data/allnli-train.csv")
    parser.add_argument("--nli_dev", type=str, default="data/allnli-dev.csv")



    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/pretrain-allnli.pt"  # save path
    # Logging aktivieren
    logfile = setup_logging(args.filepath)
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)

