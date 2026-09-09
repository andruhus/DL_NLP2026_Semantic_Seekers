# Semantic Textual Similarity (STS) — Write-up Draft

**Author:** Mohd Uwaish (mohd.uwaish@stud.uni-goettingen.de)
**Task:** Semantic Textual Similarity — Part 2 improvements
**Status:** draft — to be merged into the group `README.md` at assembly time

> Per-task draft, following the same convention as
> `QQP_Paraphrase_Detection_Readme_Draft.md`. Group metadata (group code, tutor,
> team leader, member list, AI-usage card) belongs in the root README and is not
> duplicated here. Section headings below map onto the course README template so
> this can be pasted in with minimal editing.

**Result: dev Pearson r = 0.379 → 0.849** (mean of 3 seeds), against a 0.811 team target.

---

# Setup instructions

Only the STS-specific parts are documented here; the shared environment setup
is unchanged from the course scaffold.

### Environment

```sh
bash setup.sh          # local machine
bash setup_gwdg.sh     # GWDG / Grete cluster
conda activate dnlp
```

This installs PyTorch 2.2.0 (CUDA 12.1), `numpy<2`, `tqdm`, `transformers==4.38.2`
(tokenizer only — **no pretrained model weights are loaded from it**), `tokenizers`,
`tensorboard`, `sacrebleu`, and `scikit-learn`.

### Additional dependencies beyond `dnlp`

Two packages are required for the STS task and are **not** in `setup.sh`. Part 2 permits
additional installs:

```sh
pip install datasets     # SNLI / PAWS download only — not used at training time
pip install matplotlib   # figure generation only — not used at training time
```

Neither is imported by `multitask_classifier.py` during training. `datasets` is used once
offline to build a triplet cache (below), and `matplotlib` only by `figures/make_sts_figures.py`.

### One-time data preparation (STS improvements only)

The SNLI and PAWS triplet caches must be built once before training:

```sh
pip install datasets            # allowed for Part 2; not imported during training
bash scripts/build_nli_cache.sh
```

Expected output:

```
SNLI: 149,145 triplets -> data/nli_cache/snli_triplets.json
PAWS:  49,401 pairs    -> data/nli_cache/paws_pairs.json
```

The script runs the download from a temporary directory on purpose. This project ships
its own `datasets.py`, which shadows the HuggingFace `datasets` package on the import
path; importing it from the repository root fails silently. The script writes plain JSON
caches, so training itself performs no HuggingFace import at all. Re-running is safe —
existing caches are detected and skipped.

### Reproducing the STS result

```sh
python -u multitask_classifier.py --task sts --option finetune --use_gpu \
  --cosent --cosent_weight 0.0 --cosent_tau 0.05 \
  --mnrl_weight 0.5 --mnrl_tau 0.05 \
  --cross_attn --sts_symmetry --nli_pretrain_epochs 1 \
  --hidden_dropout_prob 0.3 --batch_size 64 --epochs 5 --lr 2e-5 \
  --warmup_ratio 0.0 \
  --init_checkpoint models/<qqp_encoder>.pt \
  --filepath models/sts_best.pt --local_files_only
```

Runtime ≈ 28 min on one A100 (≈18 min SNLI pretraining + ≈10 min STS fine-tuning).
Dev Pearson r ≈ 0.849. Drop `--init_checkpoint` to reproduce from stock minBERT
(≈0.847; see [Experiments](#5-encoder-initialisation-exp-25)).

On the Grete cluster:

```sh
sbatch --partition=grete:shared --gres=gpu:A100:1 --time=01:00:00 --mem=16G \
  --cpus-per-task=4 --output=slurm_files/sts_best.out \
  --wrap="source activate dnlp && python -u multitask_classifier.py \
    --task sts --option finetune --use_gpu \
    --cosent --cosent_weight 0.0 --cosent_tau 0.05 \
    --mnrl_weight 0.5 --mnrl_tau 0.05 \
    --cross_attn --sts_symmetry --nli_pretrain_epochs 1 \
    --hidden_dropout_prob 0.3 --batch_size 64 --epochs 5 --lr 2e-5 \
    --warmup_ratio 0.0 \
    --init_checkpoint models/<qqp_encoder>.pt \
    --filepath models/sts_best.pt --local_files_only"
```

Monitor with `squeue --me`; output lands in `slurm_files/sts_best.out`.

### Key STS parameters

| Parameter | Description |
| --- | --- |
| `--cross_attn` | Cross-attention interaction layer before pooling |
| `--cross_attn_dropout` | Dropout inside that layer (default 0.1) |
| `--sts_symmetry` | Append reversed pairs; sim(A,B) = sim(B,A) |
| `--nli_pretrain_epochs` | SNLI triplet contrastive pretraining epochs |
| `--paws_pretrain_epochs` | PAWS hard-negative pretraining epochs |
| `--quora_pretrain_epochs` | MNRL pretraining on Quora positive pairs |
| `--mine_hard_negatives` | TF-IDF-mined STS hard-negative pretraining epochs |
| `--mnrl_weight`, `--mnrl_tau` | MNRL contrastive weight and temperature |
| `--cosent`, `--cosent_weight`, `--cosent_tau` | CoSENT ranking loss |
| `--angle_weight`, `--angle_tau` | AnglE loss |
| `--smart_weight`, `--smart_eta` | SMART adversarial smoothness |
| `--init_checkpoint` | Warm-start the encoder (`bert.*` tensors only) |
| `--filepath` | Explicit checkpoint path — **required when running jobs in parallel** |
| `--warmup_ratio`, `--grad_clip`, `--weight_decay` | Optimisation controls |

Every STS flag defaults to a no-op, so SST / QQP / ETPC runs are unaffected.
(Note: `--help` only lists the five arguments parsed before `parse_known_args()`
in the course scaffold, not the full set — the table above is the reference.)

### Generating predictions

Test predictions are written automatically at the end of training to
`predictions/bert/`. `python prepare_submit.py` bundles that directory for submission.

---

# Methodology

## Semantic Textual Similarity

STS scores sentence pairs on a continuous [0, 5] similarity scale, evaluated by Pearson
correlation against human judgements. The dataset is small: **5,719 training pairs** and
1,430 dev pairs. That size, rather than model capacity, turned out to be the binding
constraint on every result below.

The Part 1 baseline encoded each sentence to BERT's `[CLS]` pooler output, concatenated
the two vectors, and passed them through a learned linear regressor trained with MSE. It
reached **r = 0.379**.

Our final system makes four changes, applied in the order below. Each was validated
independently before being kept.

### 1. Cosine similarity head over mean-pooled tokens

Following Sentence-BERT ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)),
we replaced the concat-and-regress head with a **parameter-free cosine similarity** over
mean-pooled `last_hidden_state`, rescaled to the label range:

```python
emb1, emb2 = encode(s1), encode(s2)             # masked mean pooling
sim = ((cosine_similarity(emb1, emb2) + 1) / 2) * 5
```

The concat head could satisfy the training objective by memorising pairs. Cosine
similarity has no parameters, so the only way to reduce the loss is to reshape the
**embedding space itself** — which is what generalises. This was the largest single gain
of the project (+0.280).

An instructive detail: adding mean pooling while *keeping* the concat head made results
**worse** (0.366 vs 0.379). Mean pooling only helps in combination with a cosine head;
the two are not independent contributions.

### 2. MNRL contrastive loss

Multiple Negatives Ranking Loss ([Henderson et al., 2017](https://arxiv.org/abs/1705.00652))
treats each STS pair as a positive and every other in-batch pair as a negative, optimised
as NT-Xent/InfoNCE at temperature τ = 0.05:

```python
sim_matrix = normalize(emb1) @ normalize(emb2).T / tau     # [B, B]
loss = cross_entropy(sim_matrix, arange(B))                # diagonal = positives
```

Total loss: `MSE + 0.5·CosineEmbeddingLoss + 0.5·MNRL`. Beyond the +0.127 gain, MNRL
markedly stabilised training — dev r held within 0.793–0.804 across epochs 4–10, where
the MSE-only model peaked at epoch 1 and declined steadily thereafter.

### 3. Cross-attention interaction layer

**This is our main architectural contribution.** Sentence-BERT is a *bi-encoder*: each
sentence is encoded independently, so embeddings can be precomputed and compared cheaply.
That property is what makes SBERT suitable for corpus-scale retrieval — and it is
irrelevant when scoring a fixed set of *given* pairs, where the accuracy cost is pure loss.

We therefore let the two token sequences attend to each other **before** pooling:

```python
h1 = bert(s1).last_hidden_state                # [B, L1, 768]
h2 = bert(s2).last_hidden_state                # [B, L2, 768]
h1_cross = LayerNorm(h1 + MHA(q=h1, k=h2, v=h2, key_padding_mask=pad2))
h2_cross = LayerNorm(h2 + MHA(q=h2, k=h1, v=h1, key_padding_mask=pad1))
emb1, emb2 = masked_mean_pool(h1_cross), masked_mean_pool(h2_cross)
```

One `nn.MultiheadAttention` layer (8 heads, dropout 0.1), shared across both directions
and wrapped in a residual + LayerNorm like a standard transformer sublayer. This lets
*"dog"* in one sentence align with *"animal"* in the other before either is compressed to
a single vector.

Architecturally this is a **late-interaction** design, closer in spirit to ColBERT
([Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832)) than to a full cross-encoder:
BERT still runs separately per sentence, and only the token sequences interact. It costs
≈1.5× a bi-encoder rather than the quadratic blow-up of concatenated input.

MNRL continues to use **independently** encoded embeddings, since its all-pairs similarity
matrix requires each sentence encoded without reference to a particular partner. MSE and
the cosine embedding loss consume the cross-attended embeddings.

This was the breakthrough after nine consecutive experiments failed to beat 0.804.

### 4. NLI triplet pretraining

Following supervised SimCSE ([Gao et al., 2021](https://arxiv.org/abs/2104.08821)), the
encoder is pretrained for one epoch on **149,145 SNLI triplets** — premise as anchor,
entailed hypothesis as positive, contradicting hypothesis as an **explicit hard negative**
— before STS fine-tuning:

```
anchor   (premise)      : A person on a horse jumps over a broken down airplane.
positive (entailment)   : A person is outdoors, on a horse.
negative (contradiction): A person is at a diner, ordering an omelette.
```

Each anchor is scored against every positive **and** every hard negative in the batch, so
contradictions enter the similarity matrix as additional columns rather than as in-batch
noise:

```python
keys = cat([emb_pos, emb_neg], dim=0)          # [2B, H]
loss = cross_entropy(emb_anchor @ keys.T / tau, arange(B))
```

A contradiction is *topically close but semantically opposite* — shared vocabulary and
register, different meaning. That is exactly the discrimination STS demands, and far more
informative than a random unrelated sentence. NLI transfer for sentence representations
goes back to InferSent ([Conneau et al., 2017](https://arxiv.org/abs/1705.02364)) and is
the training data behind the original SBERT models.

### Supporting technique: symmetry augmentation

STS similarity is symmetric — sim(A, B) = sim(B, A) — but the training file lists each
pair in one direction only. Appending the reversed pair doubles the training set to
**11,438 examples** using no external data.

This is not a no-op: cross-attention is *asymmetric* (sentence 1 queries sentence 2's
tokens and vice versa produce different computations), and MNRL's `emb1 @ emb2.T` treats
row *i* as anchor and column *i* as its positive. Nothing otherwise forces order
invariance.

Its accuracy gain (+0.003) falls **within seed noise** and we do not claim it. Its effect
on *convergence* is large and consistent: the dev peak moves from epoch 9 to epoch 4. We
retain it on that basis.

---

# Experiments

**43 training runs across 27 distinct experiment configurations.** All evaluated on the
STS dev set (1,430 pairs) by Pearson r, using `evaluation.py` unmodified. The test file
`sts-similarity-test-student.csv` was used **once**, for final prediction generation only —
never for training, tuning, or model selection.

Hardware: one NVIDIA A100 on the GWDG Grete cluster. A typical run is 6–19 minutes.

### The measurement problem, first

Late in the project we ran the **same configuration under three random seeds**:

| Seed | Dev Pearson r |
| --- | --- |
| 11711 | 0.847 |
| 42 | 0.851 |
| 7 | 0.848 |

**Mean 0.849, spread 0.004.** The seed controls cross-attention weight initialisation,
batch composition (hence which negatives MNRL sees), and dropout masks — nothing else.

This sets a **noise floor of ≈0.004**, and we apply it consistently below: differences
smaller than that are reported as *not distinguishable from noise*, whatever their sign.
We should have measured this at the start. Several mid-project comparisons that felt
informative at the time sit inside this band, and are re-labelled accordingly.

We therefore report the **mean across three seeds** for the final configuration, not the
best single run.

---

## 1. Pooling and similarity head (Exp 1–3)

**Experiment.** Replace the `[CLS]` pooler + concat + linear-regressor head. Three
variants: mean pooling with the concat head retained; mean pooling with a cosine head;
max pooling with a cosine head (ablation).

**Expectation.** Moderate gains. SBERT reports mean pooling clearly beating `[CLS]`, so we
expected pooling alone to help.

**Result.**

| Configuration | Dev r | Δ vs baseline |
| --- | --- | --- |
| Baseline: `[CLS]` pooler + concat + MSE | 0.379 | — |
| Mean pooling, concat head kept | 0.366 | **−0.013** |
| Mean pooling + **cosine head** | **0.659** | **+0.280** |
| Max pooling + cosine head | 0.463 | +0.084 |

**Discussion.** The expectation was wrong in an informative way. Mean pooling *alone* made
things **worse** — train r reached 0.951 while dev collapsed to 0.304, i.e. the concat
regressor simply memorised training pairs more efficiently from better features. The gain
comes from the **cosine head**, which has no parameters and therefore cannot memorise: the
only way to reduce its loss is to reshape the embedding geometry. Pooling and head are not
separable contributions, which is why we report them jointly.

Max pooling underperforms mean by 0.196 — it captures salient local features but discards
the distributional sentence meaning that graded similarity depends on.

## 2. Loss composition (Exp 4–6, 13)

**Experiment.** Add a cosine embedding loss, then MNRL, then sweep MNRL's weight over
{0.3, 0.5, 1.0, 2.0}. Separately, add CoSENT ([Huang et al., 2024](https://ieeexplore.ieee.org/document/10380768)),
a ranking loss designed for continuous labels.

**Expectation.** MNRL should help substantially — contrastive objectives are standard for
sentence embeddings. CoSENT should help *more* than MNRL, since it optimises the ranking
of the continuous [0, 5] labels directly rather than binarising them.

**Result.**

| Configuration | Dev r |
| --- | --- |
| + Cosine embedding loss (w = 0.5) | 0.677 |
| **+ MNRL (τ = 0.05, w = 0.5)** | **0.804** |
| MNRL w = 0.3 / 1.0 / 2.0 | 0.794 / 0.798 / 0.782 |
| + CoSENT (τ = 0.05, w = 1.0) | **0.490** |
| + CoSENT (τ = 0.5, w = 0.1, MNRL off) | 0.660 |
| + CoSENT (τ = 0.5, w = 0.1, MNRL on) | 0.800 |

**Discussion.** MNRL delivered as expected (+0.127). CoSENT did not, and the first attempt
failed spectacularly — **0.490, below the Part 1 baseline**, with train r *decreasing*
during training (0.355 → 0.283).

That was a numerical bug in our implementation, not a flaw in the method. CoSENT computes
`exp(Δcos / τ)`; at τ = 0.05 with cosine differences up to 2.0 the exponent reaches e⁴⁰,
so the CoSENT term (4.2–5.9) dwarfed MSE and its gradient dominated destructively. At
τ = 0.5 training is stable — and CoSENT then lands at 0.800, just *below* the 0.804
MNRL baseline. Its ranking signal proves largely redundant with what MNRL already provides.

We report this failure in full because the diagnosis is the useful part: a loss that
"doesn't work" and a loss that overflows look identical from the final number alone.

## 3. The contrastive plateau (Exp 7–11)

**Experiment.** SimCSE unsupervised at three weights; MNRL temperature sweep
{0.01, 0.05, 0.10}; batch size 128; LR warmup with cosine decay.

**Expectation.** SimCSE ([Gao et al., 2021](https://arxiv.org/abs/2104.08821)) reports large
STS gains, so we expected it to be one of our stronger improvements. Larger batches should
help MNRL by supplying more in-batch negatives.

**Result.** Nine consecutive experiments, **none above 0.804**.

| Configuration | Dev r | Δ vs 0.804 |
| --- | --- | --- |
| SimCSE unsupervised (w = 0.5) | 0.711 | −0.093 |
| SimCSE (w = 0.1) | 0.696 | −0.108 |
| SimCSE only, no supervision | 0.651 | −0.153 |
| MNRL τ = 0.01 | 0.746 | −0.058 |
| MNRL τ = 0.10 | 0.801 | −0.003 |
| Batch size 128 | 0.803 | −0.001 |
| LR warmup 10% + cosine decay | 0.803 | −0.001 |

**Discussion.** SimCSE failed decisively, and the *manner* of failure is diagnostic. Trained
alone, its **loss collapsed to 0.005 while dev r declined monotonically** (0.651 → 0.609):
the model trivially solved "which dropout view came from my sentence" without learning
anything about semantics. Lowering its weight made results *worse*, not better, which rules
out simple interference between objectives.

The explanation is that SimCSE is designed for the setting where *no labelled similarity
data exists*. Given 5,719 labelled pairs, MNRL's real positives strictly dominate
dropout-generated pseudo-positives. Our expectation was based on a paper solving a
different problem.

τ = 0.01 is a clear failure (0.746): a very low temperature makes the softmax extremely
peaked, the contrastive term dominates the loss, and gradients become destructive — train
r reaches only 0.824 after 10 epochs versus 0.976 at τ = 0.05.

Batch size and LR schedule each moved dev r by ≤0.003 — at the time this read as "close
but not quite"; with the noise floor known it reads as **no measurable effect at all**.

**The trend that mattered.** Three orthogonal hyperparameters, all null. Before continuing
we audited the pipeline end-to-end (correct data splits, train/eval consistency, correct
Pearson computation) and found nothing wrong. That is what redirected effort from *loss
functions* to *architecture*: every one of these nine experiments changed how two
**independently computed** embeddings are scored, and information discarded when a
sentence is compressed to a vector in isolation cannot be recovered by any scoring
function.

## 4. Cross-attention interaction layer (Exp 16, 19–21)

**Experiment.** Add the cross-attention layer of §3 of the Methodology. Then test symmetry
augmentation, warmup + gradient clipping, and their combination on top.

**Expectation.** Moderate gain. We expected word-level interaction to help, but were
uncertain whether a single randomly initialised attention layer could learn enough from
5,719 pairs.

**Result.**

| Configuration | Dev r | Peak epoch |
| --- | --- | --- |
| Bi-encoder + MNRL (previous best) | 0.804 | 6 |
| **+ cross-attention** | **0.828** | 9 |
| + symmetry augmentation | 0.831 | 4 |
| + warmup + grad clip (15 epochs) | 0.829 | 6 |
| + symmetry + warmup + clip | 0.829 | 4 |

**Discussion.** +0.024 — six times the noise floor, and the first result to clear the 0.811
team target. The dev curve rises **monotonically through epoch 9**, where every bi-encoder
variant had plateaued around epoch 6: the model was still learning when the others had
stopped.

Symmetry augmentation adds +0.003 (within noise) but **halves time-to-peak**, from epoch 9
to 4 — the model had never been shown that the similarity function must be order-invariant.

Combining symmetry with warmup scored *worse* (0.829) than symmetry alone. The two work
against each other: symmetry augmentation accelerates convergence while warmup
deliberately delays early learning. **Regularisers are not additive**, and we dropped
warmup from the recipe.

## 5. Encoder initialisation (Exp 24–25)

**Experiment.** Two ways to give the encoder a better starting point: (a) intermediate-task
pretraining with MNRL on the 49,796 positive pairs of the course-provided Quora set;
(b) warm-starting from a teammate's QQP-finetuned encoder (0.868 QQP dev accuracy, versus
our own untouched Part 1 QQP model at 0.781).

**Expectation.** The teammate checkpoint should be the stronger start — it is a much better
model on a closely related paraphrase task, and its supervised training used Quora's
~85,000 *negative* pairs, which our contrastive pretraining discards.

**Result.**

| Initialisation | Dev r |
| --- | --- |
| Stock minBERT | 0.831 |
| **+ Quora MNRL pretraining (1 epoch)** | **0.838** (mean of 3 seeds) |
| Quora pretraining, 2 epochs | 0.840 |
| Teammate QQP encoder | 0.830 |
| Teammate QQP encoder + Quora pretraining | 0.832 |

**Discussion.** The expectation was wrong, and the reason is worth recording. Before
running, we diffed the teammate encoder against stock minBERT: 199 of 200 tensors differed,
with the **largest changes in `bert_layers.11.*` and `pooler_dense.weight`**.

That is the explanation. `predict_paraphrase()` trains through `forward()` →
`pooler_output` (`[CLS]` → dense → tanh), so their gradients shaped the CLS pathway and the
pooler. Our `encode()` **ignores the pooler entirely** and mean-pools `last_hidden_state`.
Their strongest adaptations live in the component our pipeline discards, and the top-layer
specialisation for a binary decision appears to actively cost us on graded similarity.

**Encoder transfer between teammates requires a shared consumption path, not merely a
shared backbone.** The same asymmetry applies in reverse: our mean-pooling-trained encoder
would likely transfer poorly to their pooler-based head.

Quora pretraining worked (+0.007). A second epoch added nothing despite halving the Quora
loss (0.069 → 0.046) — the encoder extracts what it needs in one pass, and further fitting
to Quora does not transfer.

## 6. Regularisation (Exp 22)

**Experiment.** Train r sat at 0.978 against dev 0.831, which looks like textbook
overfitting. We tested weight decay λ ∈ {0.01, 0.1}, cross-attention dropout 0.3, and both.

**Expectation.** A clear win. The gap is large and the training set is small.

**Result.** All four landed at **0.829–0.830** — none beating the unregularised 0.831.
More tellingly, all four produced training curves nearly *identical* to the unregularised
run (epoch-1 loss 2.584 / 2.586 / 2.588 / 2.588 versus 2.587).

**Discussion.** The flags were verified as received in the config dump, so this was not a
plumbing bug — the interventions genuinely did almost nothing, and the arithmetic explains
why. `optimizer.py` applies decoupled decay as `p -= alpha · λ · p` with `alpha = lr = 2e-5`.
Over 8 epochs × 179 batches = 1,432 steps the total shrinkage is:

| λ | Per-step factor | Total shrinkage over the run |
| --- | --- | --- |
| 0.01 | 2 × 10⁻⁷ | **0.03%** |
| 0.1 | 2 × 10⁻⁶ | **0.29%** |

The implementation is the standard decoupled formulation and is correct; it is simply
**inert at conventional λ** for this learning rate and step count. We retested at λ = 20,
chosen for ≈5% shrinkage. There the mechanism finally engaged — **train r dropped 0.984 →
0.973, the only time any regulariser measurably reduced training fit** — but dev r fell to
0.823.

So weight decay is fully characterised and closed: too small to matter, or large enough to
matter and harmful. **A large train/dev gap does not by itself tell you whether the model
needs constraining or the dataset is too small.** Here it was the latter — which is what
motivated the transfer-learning experiments that followed.

## 7. Transfer data sources (Exp 8, 12, 23)

**Experiment.** Four sources of contrastive pretraining data, all using the same MNRL
objective and the same fine-tuning recipe: SNLI triplets (149,145), Quora positives
(49,796), PAWS adversarial near-duplicates (9,672), and TF-IDF hard negatives mined from
the STS training set itself (5,716).

**Expectation.** PAWS should be strong — its negatives are adversarial near-duplicates
with high lexical overlap, exactly the hard cases a similarity model should struggle with.
Mined negatives should also help, since they need no external data.

**Result.**

| Source | Triplets/pairs | Dev r | Δ vs 0.830 |
| --- | --- | --- | --- |
| **SNLI (entailment / contradiction)** | 149,145 | **0.847** | **+0.017** |
| Quora (positives only) | 49,796 | 0.838 | +0.008 |
| PAWS (word scrambles) | 9,672 | 0.835 | +0.005 |
| TF-IDF mined from STS train | 5,716 | 0.834 | +0.004 |

**Discussion.** Only SNLI is unambiguously outside the noise floor — and it beat the
expectation-favourite PAWS by a wide margin. Two reasons:

**Volume.** PAWS ships 49,401 pairs, but regrouping into (anchor, positive, negative)
triplets requires a premise with *both* an entailment and a contradiction; most PAWS
premises lack one, leaving only 9,672 usable triplets — 6% of SNLI's.

**Label type.** PAWS negatives are generated by *word scrambling*, so they test word-order
sensitivity. SNLI contradictions test **semantic relations**. STS asks for the latter.
Hard negatives help, but *which kind of hardness* matters as much as hardness itself.

The mined negatives are qualitatively correct — e.g. anchor *"you don't need to know
everything"* → negative *"you don't have to know"* — but they are drawn from the same 5,719
pairs the model already trains on. They **reweight existing data rather than adding any**,
and land exactly at the noise floor. This is the cleanest statement of the project's
central trend: **new data helps; rearranging what you already have does not.**

Most of SNLI's benefit is present *before* STS fine-tuning begins — epoch-1 dev r is
already 0.842, above every other configuration's best.

## 8. Auxiliary losses (Exp 14–15)

**Experiment.** AnglE ([Li & Li, 2024](https://arxiv.org/abs/2309.12871)), which optimises
the angle between embeddings in complex space to avoid cosine's vanishing gradients near
±1; and SMART ([Jiang et al., 2020](https://arxiv.org/abs/1911.03437)), adversarial
smoothness regularisation.

**Expectation.** Low. By this point eight loss-function experiments had produced no gain,
and we ran these mainly for completeness of the ablation.

**Result.** AnglE (w = 1.0) 0.834, AnglE (w = 0.1) 0.830, SMART (w = 10) 0.833,
SMART (w = 100) 0.832. All at or within the noise floor.

**Discussion.** Expectation confirmed. AnglE's saturation problem does not appear to bind
once cross-attention supplies a rich interaction signal.

**SMART should be read as "evaluated without measurable effect" rather than fairly
tested.** A tenfold weight increase moved dev r by 0.001 and left training curves nearly
identical (epoch-1 loss 2.424 vs 2.427) — a regulariser that unresponsive was never
meaningfully active. Note also that our implementation deviates from the paper twice:
the perturbation is applied to **pooled embeddings** rather than the input embedding layer
(`bert.py` is off-limits), and symmetrised KL is replaced by MSE (ours is regression, not
classification). A faithful test would require perturbing the embedding layer.

## 9. Whitening post-processing (Exp 27)

**Experiment.** Whitening ([Su et al., 2021](https://arxiv.org/abs/2103.15316)) centres
sentence embeddings and rescales them to identity covariance, correcting the *anisotropy*
of raw BERT representations. Requires no retraining. We fitted the transform on **training**
embeddings and applied it to dev — the original papers fit on the evaluation set itself,
which uses no labels but does use dev-set structure.

**Expectation.** Low-to-moderate. The reported gains are large, but contrastive training
already includes a uniformity term that should do similar work.

**Result.** The principal component spectrum answers the question before the scores do:

| Measure | Our model | Raw BERT (literature) |
| --- | --- | --- |
| Top-1 principal component | **4.7%** of variance | 30–50% |
| Effective rank | **226 / 768** | much lower |

| Whitening dimension | Dev r | Δ |
| --- | --- | --- |
| None | **0.830** | — |
| 768 | 0.815 | −0.015 |
| 384 | 0.829 | −0.001 |
| 256 | 0.832 | +0.002 |
| 128 | 0.835 | +0.005 |
| 64 | 0.827 | −0.004 |

**Discussion.** The embedding space is **already near-isotropic**, so the pathology
whitening corrects has largely been removed by MNRL's uniformity term during training.
Full-dimensional whitening actively *hurts*: rescaling every direction to unit variance
amplifies low-variance directions that carry mostly noise in a well-conditioned space.

**We do not claim the 128-d row**, although it is the highest number in the table. It
exceeds the noise floor by 0.0009 and is the **best of five dimensionalities selected
against the same set it is scored on**; with five draws at that noise level, a best-of-five
near +0.005 is close to what chance alone produces. Confirming it would need validation
data we do not have. Adopting it would repeat exactly the error that measuring seed
variance was meant to expose.

This matches Gao et al. (2021), who report that contrastive training subsumes the benefit
of flow- and whitening-based post-processing.

## 10. A methodological failure worth reporting

Our **first** NLI experiment reported 0.689 while training on **no NLI data at all**.

The project's own `datasets.py` shadows the HuggingFace `datasets` package and was already
in `sys.modules` before the in-function `sys.path` fix ran. `load_dataset` raised, a broad
`except` returned an empty list, and training silently continued on the remaining losses —
producing a plausible-looking number that tested nothing.

Repaired (triplets pre-extracted to JSON outside the repository, and the loader now
**raises** instead of falling back), the same experiment produced the **best result of the
project**. The gap between 0.689 and 0.849 is the cost of a fallback that degraded
silently.

---

## Results

All figures are dev-set scores to three-decimal precision.

### Semantic Textual Similarity (STS)

| **Semantic Textual Similarity (STS)** | **Pearson r** | **Δ vs baseline** | **Above noise floor (0.004)?** |
| --- | --- | --- | --- |
| Baseline (`[CLS]` pooler + concat + MSE) | 0.379 | — | — |
| Improvement 1 — cosine head + mean pooling | 0.659 | +0.280 | ✅ |
| Improvement 2 — + cosine embedding loss | 0.677 | +0.298 | ✅ |
| Improvement 3 — + MNRL contrastive loss | 0.804 | +0.425 | ✅ |
| Improvement 4 — + cross-attention layer | 0.828 | +0.449 | ✅ |
| Improvement 5 — + symmetry augmentation | 0.831 | +0.452 | ❌ within noise |
| Improvement 6 — + teammate encoder init | 0.830 | +0.451 | ❌ within noise |
| **Improvement 7 — + SNLI triplet pretraining** | **0.849** | **+0.470** | ✅ |

**Final: dev Pearson r = 0.849** (mean of 3 seeds: 0.847 / 0.851 / 0.848).
Prior team best 0.811 → **+0.038**.

### Incremental contribution of each technique

| # | Technique | Dev r | Δ | Kept? |
| --- | --- | --- | --- | --- |
| 0 | Part 1 baseline | 0.379 | — | — |
| 1 | Mean pooling (concat head kept) | 0.366 | −0.013 | ✗ superseded by #2 |
| 2 | Cosine similarity head + mean pooling | 0.659 | +0.280 | ✓ |
| 3 | Max pooling (ablation) | 0.463 | −0.196 | ✗ |
| 4 | Cosine embedding loss | 0.677 | +0.018 | ✓ |
| 5 | MNRL contrastive (τ = 0.05, w = 0.5) | 0.804 | +0.127 | ✓ |
| 6 | MNRL weight sweep {0.3, 1.0, 2.0} | ≤0.798 | ≤0 | ✗ w = 0.5 optimal |
| 7 | SimCSE unsupervised | 0.651–0.711 | −0.153…−0.093 | ✗ |
| 8 | MNRL τ sweep {0.01, 0.10} | 0.746 / 0.801 | −0.058 / −0.003 | ✗ τ = 0.05 optimal |
| 9 | Batch size 128 | 0.803 | −0.001 | ✗ |
| 10 | LR warmup + cosine decay | 0.803 | −0.001 | ✗ |
| 11 | CoSENT ranking loss | 0.490 / 0.800 | −0.314 / −0.004 | ✗ |
| 12 | **Cross-attention interaction layer** | **0.828** | **+0.024** | ✓ |
| 13 | STS symmetry augmentation | 0.831 | +0.003 | ✓ for convergence |
| 14 | Warmup + gradient clipping | 0.829 | −0.002 | ✗ |
| 15 | Weight decay {0.01, 0.1, 20} | 0.823–0.830 | ≤0 | ✗ |
| 16 | Cross-attention dropout 0.3 | 0.829 | −0.002 | ✗ |
| 17 | Quora MNRL pretraining | 0.838 | +0.008 | ✗ superseded by #21 |
| 18 | Teammate QQP encoder init | 0.830 | −0.001 | ✓ retained |
| 19 | TF-IDF mined hard negatives | 0.834 | +0.004 | ✗ at noise floor |
| 20 | PAWS hard negatives | 0.835 | +0.005 | ✗ marginal |
| 21 | **SNLI triplet pretraining** | **0.849** | **+0.019** | ✓ |
| 22 | AnglE loss | 0.830 / 0.834 | ≤+0.004 | ✗ |
| 23 | SMART adversarial smoothness | 0.832 / 0.833 | ≤+0.003 | ✗ no effect |
| 24 | Whitening post-processing | 0.815–0.835 | −0.015…+0.005 | ✗ |

**22 of 24 techniques did not survive.** The four that did — cosine head, MNRL,
cross-attention, NLI pretraining — account for +0.450 of the +0.470 total.

### Discussion of results

Three patterns hold across all 27 STS experiments.

**Architecture and data moved the metric; losses and regularisation did not.** Of the four
techniques that survived, two change *what gets encoded* (cosine head, cross-attention) and
two change *what data the encoder sees* (MNRL over real pairs, NLI pretraining). Ten
loss-function variants and five regularisation settings produced nothing outside the noise
floor. Once the model saturates a 5,719-pair dataset, reweighting the objective cannot add
information that is not there.

**The bi-encoder ceiling was structural.** Nine consecutive experiments plateaued at
0.800–0.804 while varying temperature, batch size, LR schedule, and three different loss
formulations. All shared one property — each sentence encoded independently. The plateau
broke only when that assumption changed.

**Small datasets defeat regularisation.** A train/dev gap of 0.978 vs 0.831 reads as
overfitting, but no penalty setting improved dev r, while adding 149,145 external triplets
moved it +0.019. The gap indicated a data shortage, not an over-flexible model.

---

### Hyperparameter Optimization

We did not run automated search (Ray Tune / Optuna). With runs at 6–19 minutes and a
noise floor of 0.004, a random search would mostly have sampled noise — and would not have
distinguished a real effect from a lucky seed without repeated runs per configuration.
Instead each hyperparameter was swept **individually, with a stated hypothesis**, on top of
the best configuration at the time.

| Parameter | Values tried | Chosen | Evidence |
| --- | --- | --- | --- |
| Pooling | CLS, mean, max | **mean** | 0.379 / 0.659 / 0.463 |
| Similarity head | concat+linear, cosine | **cosine** | +0.280 |
| MNRL weight | 0.3, 0.5, 1.0, 2.0 | **0.5** | 0.794 / 0.804 / 0.798 / 0.782 |
| MNRL temperature τ | 0.01, 0.05, 0.10 | **0.05** | 0.746 / 0.804 / 0.801 |
| CoSENT temperature | 0.05, 0.5 | n/a (dropped) | 0.490 / 0.800 |
| Batch size | 64, 128 | **64** | 0.804 / 0.803 |
| Dropout | 0.1, 0.3 | **0.3** | see confound note below |
| Warmup ratio | 0.0, 0.1 | **0.0** | 0.804 / 0.803 |
| Weight decay λ | 0, 0.01, 0.1, 20 | **0** | 0.831 / 0.830 / 0.829 / 0.823 |
| Cross-attn heads | 8 | 8 | not swept |
| Cross-attn dropout | 0.1, 0.3 | **0.1** | 0.831 / 0.829 |
| Epochs | 4–15 | **4–5** | peak moved 9 → 4 → 3 as init improved |
| SNLI pretrain epochs | 1, 2 | **1** | identical result, 2 min cheaper |

**Two hyperparameter findings are worth more than the values themselves.**

*Temperature is not a free parameter.* Both τ = 0.01 (MNRL) and τ = 0.05 (CoSENT) produced
catastrophic failures for the same underlying reason — an exponential term overwhelming the
rest of the loss. Sweeping τ blindly would have recorded two "bad methods" instead of one
numerical bug and one genuine redundancy.

*A carried-over default silently confounded eight experiments.* We lowered dropout 0.3 →
0.1 for the SimCSE experiments, where dropout **is** the augmentation. That setting then
persisted through seven further experiments, exaggerating overfitting in all of them. It
was caught only when a CoSENT rerun with dropout restored scored 0.140 higher than the
same configuration with dropout 0.1. Every run afterwards passes
`--hidden_dropout_prob 0.3` explicitly rather than relying on the default.

---

## Visualizations

All figures are generated from the raw SLURM training logs by
`python figures/make_sts_figures.py` (requires `matplotlib`); no values are entered by hand.
Sources live in `slurm_files/`, output in `figures/sts/`.

> **Note on validation loss.** `evaluation.py` computes Pearson correlation only, so
> per-epoch *dev loss* was never logged and cannot be reconstructed without re-running all
> 43 experiments. We plot **dev Pearson r** in its place — the task metric, and the
> quantity model selection actually used.

### Overall progression

![Dev performance per epoch](figures/sts/v1_dev_progression.png)

Each improvement stage, one curve, ★ marking the saved checkpoint. The bi-encoder stages
(grey) plateau below the 0.811 target; cross-attention (blue) crosses it at epoch 3 and
keeps climbing to epoch 9; SNLI pretraining (green) **starts above every other
configuration's best**.

### Does improvement A converge faster than improvement B?

![Convergence speed](figures/sts/v2_convergence_speed.png)

Epochs required to first reach each dev-r level. To reach **r ≥ 0.80**: MNRL needs 6
epochs, cross-attention 3, symmetry augmentation 2, SNLI pretraining **1**. Only SNLI ever
reaches 0.84.

This also shows the converse case: symmetry augmentation (orange) reaches 0.82 in 2 epochs
where cross-attention alone needs 5 — but cross-attention overtakes it by epoch 6 and ends
higher. **Faster convergence and better final performance are separate properties**, and
symmetry augmentation buys the first without the second.

### Overfitting dynamics

![Overfitting](figures/sts/v3_overfitting_dynamics.png)

Left: train (solid) vs dev (dashed). Right: the gap, which grows monotonically for every
variant. No intervention prevented this — only more training data changed the picture.

### Training loss

![Training loss](figures/sts/v4_training_loss.png)

Restricted to four runs sharing an **identical loss composition**
(`MSE + 0.5·cosine + 0.5·MNRL`); loss is not comparable across runs with different terms,
since adding a term mechanically raises the total. Better initialisation starts lower and
descends faster.

### Failure modes are visible during training

![Failure modes](figures/sts/v5_failure_modes.png)

Left: SimCSE's training loss falls to 0.005 while **dev r declines** — the model is
solving the pretext task without learning semantics. Middle: CoSENT at τ = 0.05 versus
τ = 0.5, showing a numerical bug rather than a bad method. Right: MNRL τ = 0.01 never
converges.

### Is a difference real, or is it the seed?

![Seed variance](figures/sts/v6_seed_variance.png)

The same configuration under three seeds. The shaded band is the full spread — 0.004 — and
is the reference for every "within noise" claim in this README.

### Which transfer source, and why?

![Transfer sources](figures/sts/v8_transfer_sources.png)

Left: dev curves during fine-tuning, ordered exactly by data volume. Right: performance
after **one** epoch versus best. SNLI reaches 0.842 after a single fine-tuning epoch —
above every other source's best — confirming the benefit is transferred rather than learned
downstream.

### Regularisation: a null result, visualised

![Regularisation](figures/sts/v9_regularisation.png)

Five regularisation settings and the unregularised baseline. Four curves are visually
inseparable. Only λ = 20 (red) moves — and it lowers *both* train and dev r. The right
panel shows why: nothing but λ = 20 changes the training fit at all.

### Error analysis

![Error analysis](figures/sts/v7_error_analysis.png)

The model **compresses the output range**: predicted σ = 1.16 against gold σ = 1.47, with
a fitted slope of 0.67. Error is worst on the *most dissimilar* pairs (MAE 0.93 in the 0–1
band versus 0.52 in 4–5) — it rarely predicts below 0.41 where gold reaches 0.0.

Pearson r is scale-invariant, so this costs nothing on our metric, but it would matter
under Spearman ρ or MSE. It is the expected behaviour of a cosine head, whose output is
squashed toward the middle of [0, 5]. **Calibrating the output range is the clearest
remaining improvement we did not pursue.**

---

## Members Contribution

**Mohd Uwaish — Semantic Textual Similarity (STS):** implemented the STS improvements in
`multitask_classifier.py` — masked mean pooling and cosine similarity head, MNRL/NT-Xent
contrastive loss, the cross-attention interaction layer (`encode_pair`), symmetry
augmentation, and four transfer-pretraining paths (SNLI, PAWS, Quora, TF-IDF-mined hard
negatives), plus CoSENT, AnglE and SMART auxiliary losses. Built the offline triplet
caching pipeline, the checkpoint warm-start mechanism (`--init_checkpoint`), and the
LR-scheduler/gradient-clipping/weight-decay controls. Ran and analysed 43 training jobs
across 27 configurations, established the seed-variance noise floor, produced all figures
(`figures/make_sts_figures.py`), and wrote the STS sections of this README. Raised STS dev
Pearson r from **0.379 → 0.849**.

*(Other members' contributions belong in the root README.)*

---

# References

**Foundational**
1. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
2. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
3. Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
4. Kingma, D. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

**Sentence embeddings and contrastive learning**
5. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) — architecture for improvements 1–2.
6. Henderson, M. et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply.* [arXiv:1705.00652](https://arxiv.org/abs/1705.00652) — MNRL.
7. Gao, T., Yao, X. & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP. [arXiv:2104.08821](https://arxiv.org/abs/2104.08821) — unsupervised SimCSE (negative result) and the supervised NLI-triplet formulation (our best result).
8. Conneau, A. et al. (2017). *Supervised Learning of Universal Sentence Representations from NLI Data.* EMNLP. [arXiv:1705.02364](https://arxiv.org/abs/1705.02364) — InferSent; NLI as transfer data.
9. Wu, X. et al. (2021). *ESimCSE: Enhanced Sample Building Method for Contrastive Learning of Unsupervised Sentence Embedding.* [arXiv:2109.04380](https://arxiv.org/abs/2109.04380)
10. Khattab, O. & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) — late-interaction framing for our cross-attention layer.

**Losses evaluated**
11. Huang, X. et al. (2024). *CoSENT: Consistent Sentence Embedding via Similarity Ranking.* IEEE/ACM TASLP.
12. Li, X. & Li, J. (2024). *AnglE-optimized Text Embeddings.* ACL. [arXiv:2309.12871](https://arxiv.org/abs/2309.12871)
13. Jiang, H. et al. (2020). *SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization.* ACL. [arXiv:1911.03437](https://arxiv.org/abs/1911.03437)

**Post-processing and representation geometry**
14. Su, J. et al. (2021). *Whitening Sentence Representations for Better Semantics and Faster Retrieval.* [arXiv:2103.15316](https://arxiv.org/abs/2103.15316)
15. Li, B. et al. (2020). *On the Sentence Embeddings from Pre-trained Language Models.* EMNLP. [arXiv:2011.05864](https://arxiv.org/abs/2011.05864) — BERT-flow; anisotropy.
16. Wang, T. & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)

**Transfer learning and datasets**
17. Phang, J., Févry, T. & Bowman, S. (2018). *Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks.* [arXiv:1811.01088](https://arxiv.org/abs/1811.01088) — intermediate-task transfer.
18. Bowman, S. et al. (2015). *A Large Annotated Corpus for Learning Natural Language Inference.* EMNLP. [arXiv:1508.05326](https://arxiv.org/abs/1508.05326) — SNLI.
19. Zhang, Y., Baldridge, J. & He, L. (2019). *PAWS: Paraphrase Adversaries from Word Scrambling.* NAACL. [arXiv:1904.01130](https://arxiv.org/abs/1904.01130)
20. Cer, D. et al. (2017). *SemEval-2017 Task 1: Semantic Textual Similarity.* SemEval. [arXiv:1708.00055](https://arxiv.org/abs/1708.00055)

**Code**
21. minbert assignment, CMU [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html).
22. Stanford [CS 224N](https://web.stanford.edu/class/cs224n/) default final project.
23. HuggingFace [`transformers`](https://github.com/huggingface/transformers) (Apache 2.0) — tokenizer only.

---

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the default
final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/)
developed by Gabriel Poesia, John Hewitt, Amelie Byun, John Cho, and their team.

The BERT implementation part of the project was adapted from the "minbert" assignment
developed at Carnegie Mellon University's
[CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html), created by
Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and
Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and
[Terry Ruas](https://terryruas.com/). The project was modified by
[Niklas Bauer](https://github.com/ItsNiklas/) and
[Tolga Ermis](https://github.com/Tollgaermis/) for the 2026 DNLP course at the University
of Göttingen.
