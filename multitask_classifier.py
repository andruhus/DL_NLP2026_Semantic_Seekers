import argparse
import json
import math
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW
from tokenizer import BertTokenizer


# =============================================================================
# STS (Part 2) — Uwaish
# -----------------------------------------------------------------------------
# Helpers for the semantic textual similarity task: triplet datasets and loaders
# for contrastive pretraining, TF-IDF hard-negative mining, and two auxiliary
# losses (AnglE, SMART). None of this is used by SST / QQP / ETPC.
#
# Triplet caches are built on the login node rather than here — this file's own
# `datasets` import resolves to the project's datasets.py, which shadows the
# HuggingFace package. See the README for the cache-building snippet.
# =============================================================================


class NLITripletDataset(torch.utils.data.Dataset):
    """Anchor / positive / hard-negative triplets for contrastive pretraining."""
    def __init__(self, triplets, args):
        self.triplets = triplets
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

    def collate_fn(self, batch):
        anchors   = [x[0] for x in batch]
        positives = [x[1] for x in batch]
        negatives = [x[2] for x in batch]
        enc_a = self.tokenizer(anchors,   return_tensors="pt", padding=True, truncation=True)
        enc_p = self.tokenizer(positives, return_tensors="pt", padding=True, truncation=True)
        enc_n = self.tokenizer(negatives, return_tensors="pt", padding=True, truncation=True)
        return {
            "anchor_ids":  torch.LongTensor(enc_a["input_ids"]),
            "anchor_mask": torch.LongTensor(enc_a["attention_mask"]),
            "pos_ids":     torch.LongTensor(enc_p["input_ids"]),
            "pos_mask":    torch.LongTensor(enc_p["attention_mask"]),
            "neg_ids":     torch.LongTensor(enc_n["input_ids"]),
            "neg_mask":    torch.LongTensor(enc_n["attention_mask"]),
        }


def load_cached_triplets(path, kind):
    """Load (anchor, positive, hard_negative) triplets from a pre-built JSON cache.

    The cache is built on the login node (see docs/sts_strategy.md), NOT here: the
    project's own datasets.py shadows the HuggingFace `datasets` package and is already
    in sys.modules by the time this runs, which silently broke the original Exp 8.
    Failing loudly is deliberate — a silent fallback produced a result that tested nothing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{kind} cache missing at {path}. Build it on the login node before submitting."
        )
    with open(path) as fp:
        data = json.load(fp)
    print(f"Loaded {len(data):,} {kind} triplets from {path}.")
    return [tuple(t) for t in data]


def load_paws_triplets(path="data/nli_cache/paws_pairs.json"):
    """PAWS ships as (s1, s2, label) pairs; regroup into anchor/positive/hard-negative.

    PAWS negatives are adversarial near-duplicates — high lexical overlap, different
    meaning — so they are far harder negatives than random in-batch pairs.
    """
    from collections import defaultdict
    if not os.path.exists(path):
        raise FileNotFoundError(f"PAWS cache missing at {path}. Build it on the login node.")
    with open(path) as fp:
        pairs = json.load(fp)
    by = defaultdict(lambda: {"pos": [], "neg": []})
    for s1, s2, lab in pairs:
        by[s1]["pos" if int(lab) == 1 else "neg"].append(s2)
    trip = [(k, v["pos"][0], v["neg"][0]) for k, v in by.items() if v["pos"] and v["neg"]]
    print(f"Loaded {len(trip):,} PAWS triplets from {len(pairs):,} pairs.")
    return trip


def mine_sts_hard_negatives(sts_data, top_k=1):
    """Mine hard negatives from the STS training set itself using TF-IDF overlap.

    For each pair (s1, s2, label), find the sentence elsewhere in the training set that
    is most lexically similar to s1 but belongs to a pair whose gold label is low.
    High surface overlap with low semantic similarity = the hardest kind of negative,
    and it needs no external data.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    sents = [d[0] for d in sts_data] + [d[1] for d in sts_data]
    labels = [float(d[2]) for d in sts_data] * 2
    vec = TfidfVectorizer(min_df=1, stop_words=None).fit(sents)
    X = vec.transform(sents)
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, -1.0)
    n = len(sts_data)
    # only sentences from low-similarity pairs are eligible as negatives
    eligible = np.array([lab < 3.0 for lab in labels])
    sim[:, ~eligible] = -1.0
    trip = []
    for i in range(n):
        sim[i, i] = -1.0
        sim[i, i + n] = -1.0          # never pick this pair's own partner
        j = int(np.argmax(sim[i]))
        if sim[i, j] > 0:
            trip.append((sts_data[i][0], sts_data[i][1], sents[j]))
    print(f"Mined {len(trip):,} TF-IDF hard-negative triplets from the STS train set.")
    return trip


def angle_loss(emb1, emb2, labels, tau=1.0):
    """AnglE loss (Li & Li, ACL 2024).

    Treats each embedding as a complex vector (first half real, second half imaginary)
    and optimises the *angle* between them rather than the cosine. Cosine saturates near
    +/-1 where its gradient vanishes; the angle formulation does not, which matters most
    for the extreme-similarity pairs cosine handles worst.
    """
    a, b = torch.chunk(emb1, 2, dim=1)
    c, d = torch.chunk(emb2, 2, dim=1)
    z = torch.sum(c ** 2 + d ** 2, dim=1, keepdim=True) + 1e-8
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z
    dz = torch.sqrt(torch.sum(a ** 2 + b ** 2, dim=1, keepdim=True) + 1e-8)
    dw = torch.sqrt(torch.sum(c ** 2 + d ** 2, dim=1, keepdim=True) + 1e-8)
    scale = dz / dw
    re, im = re / scale, im / scale
    pred = torch.abs(torch.sum(torch.cat([re, im], dim=1), dim=1)) * tau   # larger = more similar
    lab = labels.flatten().float()
    mask = (lab[:, None] < lab[None, :]).float()        # pairs where i is less similar than j
    diff = pred[:, None] - pred[None, :]
    diff = (diff - (1.0 - mask) * 1e12).view(-1)
    diff = torch.cat([torch.zeros(1, device=diff.device), diff], dim=0)
    return torch.logsumexp(diff, dim=0)


def smart_loss(emb1, emb2, cos_clean, sigma=1e-5, eta=1e-3):
    """SMART-style adversarial smoothness regularisation (Jiang et al., 2020).

    Adapted in two ways: the perturbation is applied to the pooled sentence embeddings
    rather than the input embedding layer (bert.py is off-limits), and the symmetrised
    KL of the original — defined for classification — is replaced by MSE between the
    clean and perturbed similarity predictions, which is the standard regression analogue.
    """
    # eta is a RELATIVE perturbation size: a fraction of each embedding's own norm.
    # An absolute eta is meaningless here — mean-pooled BERT embeddings have norms of
    # order 10, so a fixed 1e-3 step is numerically inert (cf. the weight-decay finding
    # in Exp 22, where an lr-scaled penalty was too small to have any effect).
    n1 = (torch.randn_like(emb1) * sigma).requires_grad_(True)
    n2 = (torch.randn_like(emb2) * sigma).requires_grad_(True)
    cos_adv = F.cosine_similarity(emb1 + n1, emb2 + n2, dim=-1)
    adv = F.mse_loss(cos_adv, cos_clean.detach())
    g1, g2 = torch.autograd.grad(adv, [n1, n2], retain_graph=True, create_graph=False)
    g1 = g1 / (g1.norm(dim=-1, keepdim=True) + 1e-8)
    g2 = g2 / (g2.norm(dim=-1, keepdim=True) + 1e-8)
    p1 = (eta * emb1.norm(dim=-1, keepdim=True) * g1).detach()
    p2 = (eta * emb2.norm(dim=-1, keepdim=True) * g2).detach()
    cos_pert = F.cosine_similarity(emb1 + p1, emb2 + p2, dim=-1)
    return F.mse_loss(cos_pert, cos_clean)


# ========================= end STS (Part 2) helpers ==========================


TQDM_DISABLE = True


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
    - Semantic Textual Similarity (predict_similarity_sts)
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
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.similarity_regressor = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.paraphrase_type_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 26)
        # --- STS (Part 2): cross-attention interaction layer -----------------
        # Off unless --cross_attn is passed, so other tasks build an identical model.
        self.use_cross_attn = getattr(config, 'cross_attn', False)
        if self.use_cross_attn:
            self.cross_attn_layer = nn.MultiheadAttention(
                BERT_HIDDEN_SIZE, num_heads=8,
                dropout=getattr(config, 'cross_attn_dropout', 0.1),
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(BERT_HIDDEN_SIZE)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        output = self.bert(input_ids, attention_mask)
        return output['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST
        """
        embedding = self.forward(input_ids, attention_mask)
        return self.sentiment_classifier(embedding)

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)
        return self.paraphrase_classifier(torch.cat([emb1, emb2], dim=1))

    def predict_similarity_sts(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0,5], your output should be normalized to the interval [0,5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
        """
        if self.use_cross_attn:
            emb1, emb2 = self.encode_pair(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        else:
            emb1 = self.encode(input_ids_1, attention_mask_1)
            emb2 = self.encode(input_ids_2, attention_mask_2)
        cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)   # [B], range [-1, 1]
        return ((cos_sim + 1) / 2 * 5).unsqueeze(-1)         # [B, 1], range [0, 5]

    def predict_paraphrase_types(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for detecting the paraphrase types.
        There are 26 different types of paraphrases.
        Thus, your output should contain 26 unnormalized logits for each sentence. It will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: ETPC
        """
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)
        return self.paraphrase_type_classifier(torch.cat([emb1, emb2], dim=1))

    def encode(self, input_ids, attention_mask):
        # STS (Part 2). Mean pooling over non-padding tokens.
        hidden = self.bert(input_ids, attention_mask)['last_hidden_state']  # [B, L, 768]
        mask = attention_mask.unsqueeze(-1).float()                          # [B, L, 1]
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)                 # [B, 768]

    def encode_pair(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """STS (Part 2): cross-attention encoding.

        Each sentence attends to the other's token sequence before pooling.
        """
        h1 = self.bert(input_ids_1, attention_mask_1)['last_hidden_state']  # [B, L1, 768]
        h2 = self.bert(input_ids_2, attention_mask_2)['last_hidden_state']  # [B, L2, 768]
        # True where padding (MultiheadAttention ignores these keys)
        pad1 = (attention_mask_1 == 0)
        pad2 = (attention_mask_2 == 0)
        # h1 attends to h2, h2 attends to h1
        h1_cross, _ = self.cross_attn_layer(h1, h2, h2, key_padding_mask=pad2)
        h1_cross = self.cross_attn_norm(h1 + h1_cross)
        h2_cross, _ = self.cross_attn_layer(h2, h1, h1, key_padding_mask=pad1)
        h2_cross = self.cross_attn_norm(h2 + h2_cross)
        # Mean pool over non-padding tokens
        mask1 = attention_mask_1.unsqueeze(-1).float()
        mask2 = attention_mask_2.unsqueeze(-1).float()
        emb1 = (h1_cross * mask1).sum(dim=1) / mask1.sum(dim=1)
        emb2 = (h2_cross * mask2).sum(dim=1) / mask2.sum(dim=1)
        return emb1, emb2


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
    print(f"Saving the model to {filepath}.")


# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    # SST dataset
    if args.task == "sst" or args.task == "multitask":
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
        )

    nli_dataloader = None
    if args.task == "sts" or args.task == "multitask":
        # --- STS (Part 2): symmetry augmentation -----------------------------
        if args.sts_symmetry:
            # STS similarity is symmetric: sim(A,B) == sim(B,A). Doubles training pairs.
            sts_train_data = sts_train_data + [
                (s2, s1, label, f"{sid}_rev") for s1, s2, label, sid in sts_train_data
            ]
            print(f"Symmetry augmentation: {len(sts_train_data)} STS training pairs.")
        sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
        sts_train_dataloader = DataLoader(
            sts_train_data, shuffle=True, batch_size=args.batch_size,
            collate_fn=sts_train_data.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data, shuffle=False, batch_size=args.batch_size,
            collate_fn=sts_dev_data.collate_fn,
        )

    if args.task == "qqp" or args.task == "multitask":
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)
        quora_train_dataloader = DataLoader(
            quora_train_data, shuffle=True, batch_size=args.batch_size,
            collate_fn=quora_train_data.collate_fn,
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_data, shuffle=False, batch_size=args.batch_size,
            collate_fn=quora_dev_data.collate_fn,
        )

    if args.task == "etpc" or args.task == "multitask":
        etpc_train_data = SentencePairDataset(etpc_train_data, args)
        etpc_dev_data = SentencePairDataset(etpc_dev_data, args)
        etpc_train_dataloader = DataLoader(
            etpc_train_data, shuffle=True, batch_size=args.batch_size,
            collate_fn=etpc_train_data.collate_fn,
        )
        etpc_dev_dataloader = DataLoader(
            etpc_dev_data, shuffle=False, batch_size=args.batch_size,
            collate_fn=etpc_dev_data.collate_fn,
        )

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "cross_attn": args.cross_attn,
        "cross_attn_dropout": args.cross_attn_dropout,
    }

    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config)

    # --- STS (Part 2): optional encoder warm-start ---------------------------
    # Optionally warm-start the shared BERT encoder from an existing checkpoint
    # (e.g. a teammate's QQP-trained model). Only encoder weights are taken —
    # task heads and any architecture-specific layers are left at their init.
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        if not bert_sd:
            raise ValueError(
                f"No 'bert.*' keys found in {args.init_checkpoint}. "
                f"Top-level keys present: {list(sd.keys())[:10]}"
            )
        missing, unexpected = model.bert.load_state_dict(bert_sd, strict=False)
        print(f"Warm-start from {args.init_checkpoint}: "
              f"loaded {len(bert_sd)} encoder tensors, "
              f"{len(missing)} missing, {len(unexpected)} unexpected.")
        if missing:
            print(f"  missing (kept at init): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  unexpected (ignored): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    best_dev_acc = float("-inf")

    # --- STS (Part 2): optional LR warmup + cosine decay ---------------------
    scheduler = None
    if args.warmup_ratio > 0 and sts_train_dataloader is not None:
        total_steps = len(sts_train_dataloader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

    # =========================================================================
    # STS (Part 2): transfer pretraining phases.
    # All are no-ops unless the corresponding --*_pretrain_epochs flag is set,
    # and all are additionally gated on args.task == "sts".
    # =========================================================================

    def _triplet_pretrain(triplets, tag, n_epochs):
        """Contrastive pretraining with explicit hard negatives.

        Each anchor is scored against every positive AND every hard negative in the
        batch, so the hard negatives act as extra columns in the MNRL similarity matrix
        rather than merely as in-batch noise.
        """
        ds = NLITripletDataset(triplets, args)
        loader = DataLoader(ds, shuffle=True, batch_size=args.batch_size,
                            collate_fn=ds.collate_fn)
        for ep in range(n_epochs):
            model.train()
            tot, nb = 0.0, 0
            for batch in tqdm(loader, desc=f"{tag}-{ep+1:02}", disable=TQDM_DISABLE):
                a_ids, a_mask = batch["anchor_ids"].to(device), batch["anchor_mask"].to(device)
                p_ids, p_mask = batch["pos_ids"].to(device),    batch["pos_mask"].to(device)
                n_ids, n_mask = batch["neg_ids"].to(device),    batch["neg_mask"].to(device)
                optimizer.zero_grad()
                emb_a = F.normalize(model.encode(a_ids, a_mask), dim=-1)
                emb_p = F.normalize(model.encode(p_ids, p_mask), dim=-1)
                emb_n = F.normalize(model.encode(n_ids, n_mask), dim=-1)
                keys = torch.cat([emb_p, emb_n], dim=0)               # [2B, H]
                sim = torch.mm(emb_a, keys.T) / args.mnrl_tau         # [B, 2B]
                tgt = torch.arange(emb_a.size(0), device=device)
                loss = F.cross_entropy(sim, tgt)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                tot += loss.item(); nb += 1
            print(f"{tag} pretrain epoch {ep+1:02}: loss :: {tot/max(1,nb):.4f}", flush=True)

    if args.task == "sts":
        if args.nli_pretrain_epochs > 0:
            _triplet_pretrain(load_cached_triplets("data/nli_cache/snli_triplets.json", "SNLI"),
                              "snli", args.nli_pretrain_epochs)
        if args.paws_pretrain_epochs > 0:
            _triplet_pretrain(load_paws_triplets(), "paws", args.paws_pretrain_epochs)
        if args.mine_hard_negatives > 0:
            _raw = load_multitask_data(args.sst_train, args.quora_train,
                                       args.sts_train, args.etpc_train, split="train")[3]
            _triplet_pretrain(mine_sts_hard_negatives(_raw), "stsmined",
                              args.mine_hard_negatives)

    # Optional: pretrain the encoder on Quora paraphrase pairs before STS fine-tuning.
    # Quora provides 135K pairs vs 5.7K for STS — positives give MNRL a far richer
    # contrastive signal to shape the embedding space before the small STS fine-tune.
    if args.quora_pretrain_epochs > 0 and args.task == "sts":
        _, _, quora_pt_raw, _, _ = load_multitask_data(
            args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
        )
        positives = [d for d in quora_pt_raw if float(d[2]) == 1.0]
        print(f"Quora pretraining: {len(positives)} positive pairs of {len(quora_pt_raw)} total.")
        quora_pt_ds = SentencePairDataset(positives, args)
        quora_pt_loader = DataLoader(
            quora_pt_ds, shuffle=True, batch_size=args.batch_size,
            collate_fn=quora_pt_ds.collate_fn,
        )
        for pt_epoch in range(args.quora_pretrain_epochs):
            model.train()
            pt_loss, pt_batches = 0, 0
            for batch in tqdm(quora_pt_loader, desc=f"quora-pt-{pt_epoch+1:02}", disable=TQDM_DISABLE):
                b_ids1, b_mask1 = batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device)
                b_ids2, b_mask2 = batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device)

                optimizer.zero_grad()
                e1 = F.normalize(model.encode(b_ids1, b_mask1), dim=-1)
                e2 = F.normalize(model.encode(b_ids2, b_mask2), dim=-1)
                sim = torch.mm(e1, e2.T) / args.mnrl_tau
                tgt = torch.arange(sim.size(0), device=device)
                loss = F.cross_entropy(sim, tgt)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                pt_loss += loss.item()
                pt_batches += 1
            print(f"Quora pretrain epoch {pt_epoch+1:02}: loss :: {pt_loss/max(1,pt_batches):.3f}")

    # ==================== end STS (Part 2) pretraining =======================

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask":
            # Train the model on the sst dataset.

            for batch in tqdm(
                sst_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask":
            for batch in tqdm(sts_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE):
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch["token_ids_1"], batch["attention_mask_1"],
                    batch["token_ids_2"], batch["attention_mask_2"],
                    batch["labels"],
                )
                b_ids1, b_mask1 = b_ids1.to(device), b_mask1.to(device)
                b_ids2, b_mask2 = b_ids2.to(device), b_mask2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                emb1 = model.encode(b_ids1, b_mask1)
                emb2 = model.encode(b_ids2, b_mask2)
                if args.cross_attn:
                    emb_s1, emb_s2 = model.encode_pair(b_ids1, b_mask1, b_ids2, b_mask2)
                else:
                    emb_s1, emb_s2 = emb1, emb2
                cos_sim = F.cosine_similarity(emb_s1, emb_s2, dim=-1)
                logits = ((cos_sim + 1) / 2 * 5).unsqueeze(-1)
                mse_loss = F.mse_loss(logits.flatten().float(), b_labels.flatten().float())

                if args.cosent:
                    # Cosine embedding loss uses cross-attended embeddings when enabled
                    cos_target = torch.where(b_labels.flatten() >= 2.5,
                                             torch.ones_like(b_labels.flatten()),
                                             -torch.ones_like(b_labels.flatten()))
                    cos_emb_loss = F.cosine_embedding_loss(emb_s1, emb_s2, cos_target)
                    # MNRL using STS pairs as positives (same as Exp 5)
                    e1 = F.normalize(emb1, dim=-1)
                    e2 = F.normalize(emb2, dim=-1)
                    mnrl_sim = torch.mm(e1, e2.T) / args.mnrl_tau
                    mnrl_labels_t = torch.arange(mnrl_sim.size(0), device=mnrl_sim.device)
                    mnrl_val = F.cross_entropy(mnrl_sim, mnrl_labels_t)
                    # CoSENT ranking loss (optional — zero weight = Exp 5 replica)
                    labels_norm = b_labels.flatten().float() / 5.0
                    diff_cos   = cos_sim.unsqueeze(0) - cos_sim.unsqueeze(1)
                    diff_label = labels_norm.unsqueeze(0) - labels_norm.unsqueeze(1)
                    cosent_mask = (diff_label > 0).float()
                    cosent_val = torch.log1p(
                        torch.exp(diff_cos / args.cosent_tau) * cosent_mask
                    ).sum() / (cosent_mask.sum() + 1e-8)
                    loss = mse_loss + 0.5 * cos_emb_loss + args.cosent_weight * cosent_val + args.mnrl_weight * mnrl_val
                    if args.angle_weight > 0:
                        loss = loss + args.angle_weight * angle_loss(
                            emb_s1, emb_s2, b_labels, tau=args.angle_tau)
                    if args.smart_weight > 0:
                        loss = loss + args.smart_weight * smart_loss(
                            emb_s1, emb_s2, cos_sim,
                            sigma=args.smart_sigma, eta=args.smart_eta)
                else:
                    # SimCSE: encode each batch twice for different dropout masks
                    z1_b = model.encode(b_ids1, b_mask1)
                    z2_b = model.encode(b_ids2, b_mask2)
                    cos_target = torch.where(b_labels.flatten() >= 2.5,
                                             torch.ones_like(b_labels.flatten()),
                                             -torch.ones_like(b_labels.flatten()))
                    cos_loss = F.cosine_embedding_loss(emb1, emb2, cos_target)
                    z_a = torch.cat([emb1, emb2], dim=0)
                    z_b = torch.cat([z1_b, z2_b], dim=0)
                    z_a_norm = F.normalize(z_a, dim=-1)
                    z_b_norm = F.normalize(z_b, dim=-1)
                    sim_matrix = torch.mm(z_a_norm, z_b_norm.T) / 0.05
                    simcse_labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                    simcse_loss = F.cross_entropy(sim_matrix, simcse_labels)
                    if args.simcse_only:
                        loss = args.mnrl_weight * simcse_loss
                    elif args.nli_simcse:
                        loss = mse_loss + 0.5 * cos_loss
                    else:
                        loss = mse_loss + 0.5 * cos_loss + args.mnrl_weight * simcse_loss

                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "qqp" or args.task == "multitask":
            for batch in tqdm(quora_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE):
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch["token_ids_1"], batch["attention_mask_1"],
                    batch["token_ids_2"], batch["attention_mask_2"],
                    batch["labels"],
                )
                b_ids1, b_mask1 = b_ids1.to(device), b_mask1.to(device)
                b_ids2, b_mask2 = b_ids2.to(device), b_mask2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(
                    logits.flatten().float(), b_labels.flatten().float()
                )
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "etpc" or args.task == "multitask":
            for batch in tqdm(etpc_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE):
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch["token_ids_1"], batch["attention_mask_1"],
                    batch["token_ids_2"], batch["attention_mask_2"],
                    batch["labels"],
                )
                b_ids1, b_mask1 = b_ids1.to(device), b_mask1.to(device)
                b_ids2, b_mask2 = b_ids2.to(device), b_mask2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), b_labels.float()
                )
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0, 0),  # TODO
        }[args.task]

        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)


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
    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")

    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="data/quora-paraphrase-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-similarity-test-student.csv")

    # TODO
    # You should split the train data into a train and dev set first and change the
    # default path of the --etpc_dev argument to your dev set.
    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train.csv")
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv"
    )

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
        ),
    )

    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    # =========================================================================
    # STS (Part 2) — Uwaish.  Flags below affect the STS task only; every one
    # defaults to a no-op so SST / QQP / ETPC runs are unchanged.
    # Best configuration and full results are documented in the README.
    # =========================================================================

    # --- Loss composition ----------------------------------------------------
    parser.add_argument("--mnrl_weight", type=float, default=0.5,
                        help="weight of the MNRL contrastive term (best: 0.5)")
    parser.add_argument("--mnrl_tau", type=float, default=0.05,
                        help="MNRL temperature (best: 0.05; 0.01 diverges)")
    parser.add_argument("--cosent", action="store_true", default=False,
                        help="select the MSE + cosine + MNRL loss branch")
    parser.add_argument("--cosent_weight", type=float, default=1.0,
                        help="CoSENT ranking-loss weight (0.0 disables; found redundant)")
    parser.add_argument("--cosent_tau", type=float, default=0.05,
                        help="CoSENT temperature (0.05 overflows; use 0.5 if enabled)")
    parser.add_argument("--angle_weight", type=float, default=0.0,
                        help="AnglE loss weight (evaluated, no gain)")
    parser.add_argument("--angle_tau", type=float, default=1.0)
    parser.add_argument("--smart_weight", type=float, default=0.0,
                        help="SMART adversarial smoothness weight (evaluated, no effect)")
    parser.add_argument("--smart_sigma", type=float, default=1e-5,
                        help="SMART noise scale for the initial perturbation")
    parser.add_argument("--smart_eta", type=float, default=1e-3,
                        help="SMART step size, as a FRACTION of embedding norm")
    parser.add_argument("--simcse_only", action="store_true", default=False,
                        help="unsupervised SimCSE only (evaluated, much worse)")
    parser.add_argument("--nli_simcse", action="store_true", default=False,
                        help="legacy flag from the original NLI attempt; superseded "
                             "by --nli_pretrain_epochs")

    # --- Architecture --------------------------------------------------------
    parser.add_argument("--cross_attn", action="store_true", default=False,
                        help="cross-attention interaction layer before pooling "
                             "(largest architectural gain)")
    parser.add_argument("--cross_attn_dropout", type=float, default=0.1,
                        help="dropout inside the cross-attention layer")

    # --- Data augmentation and transfer pretraining --------------------------
    parser.add_argument("--sts_symmetry", action="store_true", default=False,
                        help="append reversed pairs: sim(A,B)==sim(B,A) doubles the "
                             "train set and halves time-to-peak")
    parser.add_argument("--nli_pretrain_epochs", type=int, default=0,
                        help="SNLI triplet contrastive pretraining epochs (best gain; "
                             "requires data/nli_cache/snli_triplets.json)")
    parser.add_argument("--paws_pretrain_epochs", type=int, default=0,
                        help="PAWS hard-negative pretraining epochs")
    parser.add_argument("--quora_pretrain_epochs", type=int, default=0,
                        help="MNRL pretraining epochs on Quora positive pairs")
    parser.add_argument("--mine_hard_negatives", type=int, default=0,
                        help="TF-IDF-mined STS hard-negative pretraining epochs")

    # --- Optimisation --------------------------------------------------------
    # NOTE: --warmup_ratio and --weight_decay are also registered on the SST branch
    # (simon_testbranch) with identical types and defaults. Whoever merges second should
    # delete one copy — argparse raises "conflicting option string" on a duplicate.
    # Defaults are kept identical to theirs so either copy behaves the same.
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="fraction of STS steps used for linear LR warmup (0=disabled)")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="max grad norm for STS steps (0=disabled)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="decoupled weight decay for AdamW. NOTE: optimizer.py scales "
                             "this by lr, so conventional values are inert here")

    # --- Checkpoint plumbing (useful to all tasks, not STS-specific) ---------
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="warm-start the BERT encoder from an existing checkpoint; "
                             "only bert.* tensors are loaded")
    parser.add_argument("--filepath", type=str, default=None,
                        help="explicit model save path. Set this when running jobs in "
                             "parallel, or they overwrite each other's checkpoints")
    # ======================= end STS (Part 2) flags ==========================

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
    if args.filepath is None:
        args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
