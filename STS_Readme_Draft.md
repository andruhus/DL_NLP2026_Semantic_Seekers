# Semantic Textual Similarity (STS) — Write-up Draft

**Author:** Mohd Uwaish (mohd.uwaish@stud.uni-goettingen.de)
**Task:** Semantic Textual Similarity
---

# Setup instructions

### Environment

```sh
bash setup.sh          # local machine
bash setup_gwdg.sh     # GWDG / Grete cluster
conda activate dnlp
```

This installs PyTorch 2.2.0 (CUDA 12.1), `numpy<2`, `tqdm`, `transformers==4.38.2`
(tokenizer only; no pretrained model weights are loaded from it), `tokenizers`,
`tensorboard`, `sacrebleu`, and `scikit-learn`.

# Semantic Textual Similarity Setup

### Additional dependencies

Two packages are required for the STS task and are not in `setup.sh`. Part 2 permits
additional installs.

```sh
pip install datasets     # SNLI / PAWS download only — not used at training time
pip install matplotlib   # figure generation only — not used at training time
```

Neither is imported by `multitask_classifier.py` during training. `datasets` is used
once, offline, to build a triplet cache (below); `matplotlib` is used to create plots.

### One-time data preparation

The SNLI and PAWS triplet caches must be built once before training:

```sh
pip install datasets          
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
caches, so training itself performs no HuggingFace import at all. Re-running is safe:
existing caches are detected and skipped.

### Reproducing the STS result

Locally, on one GPU:

```sh
python -u multitask_classifier.py \
  --task sts --option finetune --use_gpu \
  --cosent --cosent_weight 0.0 --cosent_tau 0.05 \
  --mnrl_weight 0.5 --mnrl_tau 0.05 \
  --cross_attn --sts_symmetry --nli_pretrain_epochs 1 \
  --hidden_dropout_prob 0.3 --batch_size 64 --epochs 5 --lr 2e-5 \
  --warmup_ratio 0.0 \
  --init_checkpoint models/qqp_encoder.pt \
  --filepath models/sts_best.pt --local_files_only
```

On the Grete cluster, the same run as a single copy-pasteable job:

```sh
mkdir -p slurm_files models

sbatch --partition=grete:shared --gres=gpu:A100:1 --time=01:00:00 --mem=16G \
  --cpus-per-task=4 --job-name=sts_best --output=slurm_files/sts_best.out \
  --wrap="source activate dnlp && python -u multitask_classifier.py \
    --task sts --option finetune --use_gpu \
    --cosent --cosent_weight 0.0 --cosent_tau 0.05 \
    --mnrl_weight 0.5 --mnrl_tau 0.05 \
    --cross_attn --sts_symmetry --nli_pretrain_epochs 1 \
    --hidden_dropout_prob 0.3 --batch_size 64 --epochs 5 --lr 2e-5 \
    --warmup_ratio 0.0 \
    --init_checkpoint models/qqp_encoder.pt \
    --filepath models/sts_best.pt --local_files_only"
```

### Checkpoints

No model weights are committed to the repository; each `.pt` file is several hundred
megabytes. Two paths appear in the commands above:

- `models/sts_best.pt` is written by the run itself, so nothing needs to be present
  beforehand.




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
| `--filepath` | Explicit checkpoint path; required when running jobs in parallel |
| `--warmup_ratio`, `--grad_clip`, `--weight_decay` | Optimisation controls |


`--help` lists only the five arguments parsed before `parse_known_args()` in the course
scaffold, not the full set; the table above is the reference.

---

# Methodology

STS scores sentence pairs on a continuous [0, 5] similarity scale, evaluated by Pearson
correlation against human judgements. The dataset is small: 5,719 training pairs and
1,430 dev pairs. That size, rather than model capacity, turned out to be the binding
constraint on every result below.

The Part 1 baseline encoded each sentence to BERT's `[CLS]` pooler output, concatenated
the two vectors, and passed them through a learned linear regressor trained with MSE. It
reached r = 0.379.

Our final system makes four changes, applied in the order below. Each was validated
independently before being kept.

### 1. Cosine similarity head over mean-pooled tokens

Following Sentence-BERT ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)),
we replaced the concat-and-regress head with a parameter-free cosine similarity over
mean-pooled `last_hidden_state`, rescaled to the label range:

```python
emb1, emb2 = encode(s1), encode(s2)             
sim = ((cosine_similarity(emb1, emb2) + 1) / 2) * 5
```

The concat head can satisfy the training objective by memorising pairs. Cosine similarity
has no parameters, so the only way to reduce the loss is to reshape the embedding space
itself, which is what generalises. This was the largest single gain of the project.

Adding mean pooling while keeping the concat head made results worse (0.366 against
0.379). Mean pooling helps only in combination with a cosine head; the two are not
independent contributions.

### 2. MNRL contrastive loss

Multiple Negatives Ranking Loss ([Henderson et al., 2017](https://arxiv.org/abs/1705.00652))
treats each STS pair as a positive and every other in-batch pair as a negative, optimised
as NT-Xent/InfoNCE at temperature τ = 0.05:

```python
sim_matrix = normalize(emb1) @ normalize(emb2).T / tau     
loss = cross_entropy(sim_matrix, arange(B))                
```

The total objective is `MSE + 0.5·CosineEmbeddingLoss + 0.5·MNRL`. Besides raising dev r,
MNRL stabilised training: dev r held within 0.793–0.804 across epochs 4–10, whereas the
MSE-only model peaked at epoch 1 and declined thereafter.

### 3. Cross-attention interaction layer

This is our main architectural contribution. Sentence-BERT is a bi-encoder: each sentence
is encoded independently, so embeddings can be precomputed and compared cheaply. That
property is what makes SBERT suitable for corpus-scale retrieval, and it is irrelevant
when scoring a fixed set of given pairs, where the accuracy cost buys nothing.

We therefore let the two token sequences attend to each other before pooling:

```python
h1 = bert(s1).last_hidden_state                # [B, L1, 768]
h2 = bert(s2).last_hidden_state                # [B, L2, 768]
h1_cross = LayerNorm(h1 + MHA(q=h1, k=h2, v=h2, key_padding_mask=pad2))
h2_cross = LayerNorm(h2 + MHA(q=h2, k=h1, v=h1, key_padding_mask=pad1))
emb1, emb2 = masked_mean_pool(h1_cross), masked_mean_pool(h2_cross)
```

One `nn.MultiheadAttention` layer (8 heads, dropout 0.1) is shared across both directions
and wrapped in a residual plus LayerNorm, like a standard transformer sublayer. The design
intent was to let *dog* in one sentence align with *animal* in the other before either is
compressed to a single vector. Inspecting the trained weights shows the layer does not do
this, and the mechanism it does implement is characterised in Visualizations under
"Cross-attention weights".

This is not a full cross-encoder: BERT still runs separately on each sentence, and only
the token sequences interact afterwards. The layer therefore costs roughly 1.5× a
bi-encoder rather than incurring the quadratic blow-up of encoding the two sentences
concatenated.

MNRL continues to use independently encoded embeddings, since its all-pairs similarity
matrix requires each sentence to be encoded without reference to a particular partner.
MSE and the cosine embedding loss consume the cross-attended embeddings.

### 4. NLI triplet pretraining

Following supervised SimCSE ([Gao et al., 2021](https://arxiv.org/abs/2104.08821)), the
encoder is pretrained for one epoch on 149,145 SNLI triplets — premise as anchor,
entailed hypothesis as positive, contradicting hypothesis as an explicit hard negative —
before STS fine-tuning:

```
anchor   (premise)      : A person on a horse jumps over a broken down airplane.
positive (entailment)   : A person is outdoors, on a horse.
negative (contradiction): A person is at a diner, ordering an omelette.
```

Each anchor is scored against every positive and every hard negative in the batch, so
contradictions enter the similarity matrix as additional columns rather than as in-batch
noise:

```python
keys = cat([emb_pos, emb_neg], dim=0)       
loss = cross_entropy(emb_anchor @ keys.T / tau, arange(B))
```

A contradiction is topically close but semantically opposite: shared vocabulary and
register, different meaning. That is the discrimination STS demands, and it is more
informative than a random unrelated sentence. SNLI
([Bowman et al., 2015](https://arxiv.org/abs/1508.05326)) is also the training data behind
the original SBERT models, so this is a well-trodden transfer path rather than a novel
one.

### Supporting technique: symmetry augmentation

STS similarity is symmetric, sim(A, B) = sim(B, A), but the training file lists each pair
in one direction only. Appending the reversed pair doubles the training set to 11,438
examples using no external data.

This is not a no-op. Cross-attention is asymmetric: sentence 1 querying sentence 2's
tokens and the reverse are different computations. MNRL's `emb1 @ emb2.T` likewise treats
row *i* as anchor and column *i* as its positive. Nothing otherwise forces order
invariance.

Its accuracy gain falls within seed noise and we do not claim it. Its effect on
convergence is large and consistent, moving the dev peak from epoch 9 to epoch 4, and we
retain it on that basis.

---

# Experiments

We ran 43 training runs on the STS task, one per file in `slurm_files/sts_exp*.out`, all
evaluated on the STS dev
set (1,430 pairs) by Pearson r using `evaluation.py`. The test file
`sts-similarity-test-student.csv` was used once, for final prediction generation only,
never for training, tuning or model selection.


### Seed variance and the noise floor

Late in the project we ran the same configuration under three random seeds. The seed
controls cross-attention weight initialisation, batch composition (hence which negatives
MNRL sees), and dropout masks; nothing else.

| Seed | Dev Pearson r |
| --- | --- |
| 11711 | 0.847 |
| 42 | 0.851 |
| 7 | 0.848 |

The mean is 0.849 and the spread 0.004. We treat 0.004 as a noise floor and apply it
consistently below: differences smaller than that are reported as not distinguishable
from noise, whatever their sign. We should have measured this at the start. Several
mid-project comparisons that felt informative at the time sit inside this band and are
relabelled accordingly.

For the final configuration we report the mean across three seeds rather than the best
single run.

---

## 1. Pooling and similarity head (Exp 1–3)

**Experiment.** Replace the `[CLS]` pooler, concatenation and linear-regressor head.
Three variants: mean pooling with the concat head retained; mean pooling with a cosine
head; max pooling with a cosine head as an ablation.

**Expectation.** Moderate gains. SBERT reports mean pooling clearly beating `[CLS]`, so
we expected pooling alone to help.

**Result.**

| Configuration | Dev r |
| --- | --- |
| Baseline: `[CLS]` pooler + concat + MSE | 0.379 |
| Mean pooling, concat head kept | 0.366 |
| Mean pooling + cosine head | **0.659** |
| Max pooling + cosine head | 0.463 |

**Observation.** Mean pooling on its own lowered dev r below the baseline, and did so
while train r reached 0.951 against a dev r of 0.304. Swapping the concat head for cosine
similarity raised dev r by 0.280, the largest single movement recorded in the project.
Max pooling with the same cosine head reached 0.463.

**Discussion.** The expectation was wrong in an informative way. Better features fed to
the concat regressor let it memorise training pairs more efficiently, which is what the
0.951/0.304 split shows. The gain therefore comes from the head, not the pooling: cosine
similarity has no parameters and cannot memorise, so the only route to a lower loss is to
reshape the embedding geometry. Pooling and head are not separable contributions, which
is why we report them jointly. Max pooling keeps only the strongest activation per
dimension, discarding the distributional sentence meaning that graded similarity depends
on.

## 2. Loss composition (Exp 4–6, 13)

**Experiment.** Add a cosine embedding loss, then MNRL, then sweep MNRL's weight over
{0.3, 0.5, 1.0, 2.0}. Separately, add CoSENT
([Huang et al., 2024](https://ieeexplore.ieee.org/document/10380768)), a ranking loss
designed for continuous labels.

**Expectation.** MNRL should help substantially, since contrastive objectives are standard
for sentence embeddings. CoSENT should help more than MNRL, because it optimises the
ranking of the continuous [0, 5] labels directly rather than binarising them.

**Result.**

| Configuration | Dev r |
| --- | --- |
| + Cosine embedding loss (w = 0.5) | 0.677 |
| + MNRL (τ = 0.05, w = 0.5) | **0.804** |
| MNRL w = 0.3 / 1.0 / 2.0 | 0.794 / 0.798 / 0.782 |
| + CoSENT (τ = 0.05, w = 1.0) | 0.490 |
| + CoSENT (τ = 0.5, w = 0.1, MNRL off) | 0.660 |
| + CoSENT (τ = 0.5, w = 0.1, MNRL on) | 0.800 |

**Observation.** MNRL added 0.127 over the cosine-embedding configuration and was best at
w = 0.5, with performance falling away monotonically as the weight rose to 2.0. CoSENT at
τ = 0.05 scored 0.490, below the Part 1 baseline, and its train r *decreased* over
training, from 0.355 to 0.283. At τ = 0.5 training was stable and CoSENT reached 0.800
with MNRL active and 0.660 without.

**Discussion.** The τ = 0.05 collapse was a numerical bug in our implementation, not a
property of the method. CoSENT computes `exp(Δcos / τ)`; at τ = 0.05 with cosine
differences up to 2.0 the exponent reaches e⁴⁰, so the CoSENT term (4.2–5.9 in the logs)
dwarfed MSE and its gradient dominated destructively. A train r that falls during training
is the signature: the model is being actively pushed away from the data, not merely
failing to fit it. Once stabilised, CoSENT lands just below the MNRL baseline, so its
ranking signal is largely redundant with what MNRL already supplies. We report the failure
in full because a loss that does not work and a loss that overflows are indistinguishable
from the final number alone.

## 3. The contrastive plateau (Exp 7–11)

**Experiment.** SimCSE unsupervised at three weights; an MNRL temperature sweep over
{0.01, 0.05, 0.10}; batch size 128; LR warmup with cosine decay.

**Expectation.** SimCSE ([Gao et al., 2021](https://arxiv.org/abs/2104.08821)) reports
large STS gains, so we expected it to be one of our stronger improvements. Larger batches
should help MNRL by supplying more in-batch negatives.

**Result.** Nine consecutive experiments, none above the 0.804 MNRL configuration.

| Configuration | Dev r |
| --- | --- |
| **Previous best (MNRL, τ = 0.05, w = 0.5)** | **0.804** |
| SimCSE unsupervised (w = 0.5) | 0.711 |
| SimCSE (w = 0.1) | 0.696 |
| SimCSE only, no supervision | 0.651 |
| MNRL τ = 0.01 | 0.746 |
| MNRL τ = 0.10 | 0.801 |
| Batch size 128 | 0.803 |
| LR warmup 10% + cosine decay | 0.803 |

**Observation.** Trained alone, SimCSE's loss fell to 0.005 while dev r declined
monotonically from 0.651 to 0.609. Lowering its weight made results worse, not better.
MNRL at τ = 0.01 reached train r of only 0.824 after 10 epochs, against 0.976 at τ = 0.05.
Batch size and LR schedule each moved dev r by at most 0.003.

**Discussion.** SimCSE's loss collapsing while dev r falls means the model solved the
pretext task — identifying which dropout view came from which sentence — without learning
anything transferable about semantics. That the lower weight was also worse rules out
simple interference between objectives. The method is designed for the setting where no
labelled similarity data exists; given 5,719 labelled pairs, MNRL's real positives
dominate dropout-generated pseudo-positives. Our expectation came from a paper solving a
different problem.

The τ = 0.01 failure is mechanical: a very low temperature makes the softmax extremely
peaked, the contrastive term dominates the total loss, and gradients become destructive,
which is why train r never approaches the τ = 0.05 level. Batch size and LR schedule read
at the time as close but not quite; with the noise floor known they read as no measurable
effect at all.

### Why we stopped tuning losses

Three orthogonal hyperparameters returning null results is itself a result. Before
continuing we audited the pipeline end to end — data splits, train/eval consistency, the
Pearson computation — and found nothing wrong. What the nine experiments had in common was
that every one of them changed how two independently computed embeddings are *scored*.
Information discarded when a sentence is compressed to a single vector in isolation cannot
be recovered by any scoring function applied afterwards. That reasoning is what redirected
the remaining effort from loss functions to architecture.

## 4. Cross-attention interaction layer (Exp 16, 19–21)

**Experiment.** Add the cross-attention layer described in Methodology §3, then test
symmetry augmentation, warmup with gradient clipping, and their combination on top.

**Expectation.** A moderate gain. We expected word-level interaction to help but were
uncertain whether a single randomly initialised attention layer could learn enough from
5,719 pairs.

**Result.**

| Configuration | Dev r | Peak epoch |
| --- | --- | --- |
| Bi-encoder + MNRL (previous best) | 0.804 | 6 |
| + cross-attention | **0.828** | 9 |
| + symmetry augmentation | 0.831 | 4 |
| + warmup + grad clip (15 epochs) | 0.829 | 6 |
| + symmetry + warmup + clip | 0.829 | 4 |

**Observation.** Cross-attention added 0.024, six times the noise floor, and was the first
configuration to break the 0.804 plateau that nine preceding experiments had failed to
clear. It passed 0.804 at epoch 4 and its dev curve then rose monotonically through epoch
9, whereas every bi-encoder variant had flattened by epoch 6. Symmetry augmentation added 0.003 and moved the peak from epoch 9 to epoch 4.
Combining symmetry with warmup scored 0.829, below symmetry alone.

**Discussion.** The value of cross-attention is visible in the shape of the curve as much
as in the endpoint: the model was still improving at epoch 9, where the bi-encoder
variants had stopped. Symmetry augmentation's 0.003 sits inside the noise floor and we do
not claim it as an accuracy gain, but halving the time to peak is well outside anything
seed variance produces, and it is explicable — the model had never been shown that the
similarity function must be order-invariant. The symmetry-plus-warmup result is a useful
reminder that regularisers are not additive: symmetry accelerates early convergence while
warmup deliberately suppresses it, so the two work against each other. We dropped warmup
from the recipe.

The gain is robust, but our initial account of why it works was not. Inspecting the
trained attention weights (Visualizations, "Cross-attention weights") shows the layer does
not learn token alignment; what it supplies is a global summary of the partner sentence.
The measurement changes the explanation, not the result.

## 5. Encoder initialisation (Exp 24–25)

**Experiment.** Two ways to give the encoder a better starting point: intermediate-task
pretraining with MNRL on the 49,796 positive pairs of the course-provided Quora set, and
warm-starting from a teammate's QQP-finetuned encoder (0.868 QQP dev accuracy, against our
own untouched Part 1 QQP model at 0.781).

**Expectation.** The teammate checkpoint should be the stronger start. It is a much better
model on a closely related paraphrase task, and its supervised training used Quora's
roughly 85,000 negative pairs, which our contrastive pretraining discards.

**Result.**

| Initialisation | Dev r |
| --- | --- |
| Stock minBERT | 0.831 |
| + Quora MNRL pretraining (1 epoch) | **0.838** (mean of 3 seeds) |
| Quora pretraining, 2 epochs | 0.840 |
| Teammate QQP encoder | 0.830 |
| Teammate QQP encoder + Quora pretraining | 0.832 |

**Observation.** The teammate checkpoint performed no better than stock minBERT. Diffing
it against stock minBERT before the run showed 199 of 200 tensors differing, with the
largest changes concentrated in `bert_layers.11.*` and `pooler_dense.weight`. Quora
pretraining added 0.007; a second epoch added nothing further despite halving the Quora
loss, from 0.069 to 0.046.

**Discussion.** The teammate result is explained by where their gradients went.
`predict_paraphrase()` trains through `forward()` to `pooler_output` (`[CLS]`, dense,
tanh), so their adaptations shaped the CLS pathway and the pooler. Our `encode()` ignores
the pooler entirely and mean-pools `last_hidden_state`. Their strongest adaptations live
in exactly the components our pipeline discards, and the top-layer specialisation for a
binary decision appears to cost us slightly on graded similarity. Encoder transfer between
teammates requires a shared consumption path, not merely a shared backbone; the same
asymmetry would apply in reverse if they warm-started from our mean-pooling-trained
encoder.

That a second Quora epoch improved the Quora loss without improving dev r indicates the
encoder extracts what transfers in a single pass, and further fitting to Quora is fitting
to Quora.

Note on the final recipe: the reported 0.849 run uses `--init_checkpoint`, but the same
configuration from stock minBERT reaches 0.847. The difference is inside the noise floor,
and we retain the flag only so the reported run reproduces exactly.

## 6. Regularisation (Exp 22)

**Experiment.** Train r sat at 0.978 against dev 0.831, which looks like textbook
overfitting. We tested weight decay λ ∈ {0.01, 0.1}, cross-attention dropout 0.3, and both
together.

**Expectation.** A clear win. The gap is large and the training set is small.

**Result.** All four settings landed at 0.829–0.830, none beating the unregularised 0.831.
Epoch-1 training losses were 2.584, 2.586, 2.588 and 2.588, against 2.587 unregularised.

**Observation.** The interventions changed neither the dev score nor the training
trajectory. The flags were verified as received in the config dump, so this was not a
plumbing failure. The arithmetic accounts for it: `optimizer.py` applies decoupled decay as
`p -= alpha · λ · p` with `alpha = lr = 2e-5`, and over 8 epochs × 179 batches = 1,432
steps the cumulative shrinkage is negligible.

| λ | Per-step factor | Total shrinkage over the run |
| --- | --- | --- |
| 0.01 | 2 × 10⁻⁷ | 0.03% |
| 0.1 | 2 × 10⁻⁶ | 0.29% |

Retesting at λ = 20, chosen to give roughly 5% shrinkage, finally engaged the mechanism:
train r dropped from 0.984 to 0.973, the only time any regulariser measurably reduced
training fit, while dev r fell to 0.823.

**Discussion.** The implementation is the standard decoupled formulation and is correct;
it is simply inert at conventional λ for this learning rate and step count. Weight decay
is therefore fully characterised and closed for this setting: either too small to matter,
or large enough to matter and harmful. The wider lesson is that a large train/dev gap does
not by itself distinguish a model that needs constraining from a dataset that is too
small. Here it was the latter, which is what motivated the transfer-learning experiments
that follow.

## 7. Transfer data sources (Exp 8, 12, 23)

**Experiment.** Four sources of contrastive pretraining data, all using the same MNRL
objective and the same fine-tuning recipe: SNLI triplets (149,145), Quora positives
(49,796), PAWS adversarial near-duplicates (9,672 usable triplets), and TF-IDF hard
negatives mined from the STS training set itself (5,716).

**Expectation.** PAWS should be strong, since its negatives are adversarial
near-duplicates with high lexical overlap, exactly the hard cases a similarity model
should struggle with. Mined negatives should also help, as they need no external data.

**Result.** The no-transfer reference for this comparison is 0.830.

| Source | Triplets/pairs | Dev r |
| --- | --- | --- |
| SNLI (entailment / contradiction) | 149,145 | **0.847** |
| Quora (positives only) | 49,796 | 0.838 |
| PAWS (word scrambles) | 9,672 | 0.835 |
| TF-IDF mined from STS train | 5,716 | 0.834 |
| No transfer pretraining | — | 0.830 |

**Observation.** Only SNLI is unambiguously outside the noise floor. Dev r after the first
fine-tuning epoch orders the four sources exactly by corpus size: 0.816 (mined), 0.824
(PAWS), 0.836 (Quora), 0.842 (SNLI), against 0.815 with no transfer. SNLI's epoch-1 score
already exceeds every other source's best. The gap between epoch-1 and best score also
narrows as the source grows, from 0.015 for no transfer to 0.005 for SNLI. Regrouping PAWS
into (anchor, positive, negative) triplets requires a premise carrying both an entailment
and a contradiction; most PAWS premises lack one, so 49,401 pairs yield only 9,672 usable
triplets, 6% of SNLI's.

**Discussion.** Two factors plausibly separate SNLI from the rest:

1. **Volume.** SNLI supplies fifteen times as many triplets as PAWS after the regrouping
   loss described above.
2. **Label type.** PAWS negatives are generated by word scrambling, so they probe
   word-order sensitivity. SNLI contradictions encode semantic relations. STS asks for the
   latter, which suggests that which kind of hardness a negative provides matters as much
   as whether it is hard.

These two explanations are confounded in our comparison: SNLI is simultaneously the
largest source and the only one with semantic-relation negatives, so this experiment
cannot separate them. Subsampling SNLI to 9,672 triplets and rerunning would isolate the
volume term, and we did not do it.

The mined negatives are qualitatively correct — for example, anchor *"you don't need to
know everything"* paired with negative *"you don't have to know"* — but they are drawn from
the same 5,719 pairs the model already trains on. They reweight existing data rather than
adding any, and land exactly at the noise floor. New data helped; rearranging data we
already had did not.

### A silent failure in the first NLI run

Our first NLI experiment reported 0.689 while training on no NLI data at all. The
project's own `datasets.py` shadows the HuggingFace `datasets` package and was already in
`sys.modules` before the in-function `sys.path` fix ran. `load_dataset` raised, a broad
`except` returned an empty list, and training continued on the remaining losses, producing
a plausible-looking number that tested nothing.

After the repair — triplets pre-extracted to JSON outside the repository, and the loader
raising instead of falling back — the same experiment produced the best result of the
project. The distance between 0.689 and 0.849 is the cost of a fallback that degraded
silently, and it is the reason the loader now has no fallback path.

## 8. Auxiliary losses (Exp 14–15)

**Experiment.** AnglE ([Li & Li, 2024](https://arxiv.org/abs/2309.12871)), which optimises
the angle between embeddings in complex space to avoid cosine's vanishing gradients near
±1, and SMART ([Jiang et al., 2020](https://arxiv.org/abs/1911.03437)), adversarial
smoothness regularisation.

**Expectation.** Low. By this point eight loss-function experiments had produced no gain,
and we ran these mainly for completeness of the ablation.

**Result.** AnglE (w = 1.0) 0.834, AnglE (w = 0.1) 0.830, SMART (w = 10) 0.833, SMART
(w = 100) 0.832. All at or within the noise floor of the 0.830 reference.

**Observation.** A tenfold increase in the SMART weight moved dev r by 0.001 and left the
training curves nearly identical (epoch-1 loss 2.424 against 2.427).

**Discussion.** AnglE's saturation problem does not appear to bind once cross-attention
supplies a rich interaction signal, which is consistent with the expectation.

SMART should be read as evaluated without measurable effect rather than fairly tested. A
regulariser that unresponsive to a tenfold weight change was never meaningfully active.
Our implementation also deviates from the paper in two ways: the perturbation is applied
to pooled embeddings rather than the input embedding layer, because `bert.py` is
off-limits, and symmetrised KL is replaced by MSE, because ours is regression rather than
classification. A faithful test would require perturbing the embedding layer.

## 9. Whitening post-processing (post-hoc analysis)

**Experiment.** Whitening ([Su et al., 2021](https://arxiv.org/abs/2103.15316)) centres
sentence embeddings and rescales them to identity covariance, correcting the anisotropy of
raw BERT representations. It requires no retraining, so this section reports post-hoc
analysis of embeddings extracted from an already-trained checkpoint rather than any
training run. We fitted the transform on training embeddings and applied it to dev. The
original papers fit on the evaluation set itself, which uses no labels but does use
dev-set structure.

**Expectation.** Low to moderate. The reported gains are large, but contrastive training
already includes a uniformity term that should do similar work.

**Result.**

| Whitening dimension | Dev r |
| --- | --- |
| None | 0.830 |
| 768 | 0.815 |
| 384 | 0.829 |
| 256 | 0.832 |
| 128 | 0.835 |
| 64 | 0.827 |

**Observation.** The principal component spectrum answers the question before the dev
scores do. In our model the top principal component accounts for 4.7% of embedding
variance and the effective rank is 226 of 768. Published measurements of raw BERT put the
top component at 30–50% of variance with a correspondingly lower effective rank, so our
embedding space is far closer to isotropic than the representations whitening was designed
to fix. Full-dimensional whitening lowered dev r; only the 128-dimensional setting scored
above the untransformed baseline.

**Discussion.** The pathology whitening corrects has largely been removed already by
MNRL's uniformity term during training. Full-dimensional whitening actively hurts because
rescaling every direction to unit variance amplifies low-variance directions that carry
mostly noise in a well-conditioned space.

We do not claim the 128-dimensional row, despite it being the highest number in the table.
It exceeds the noise floor by 0.0009 and is the best of five dimensionalities selected
against the same set it is scored on; with five draws at that noise level, a best-of-five
near +0.005 is close to what chance alone produces. Confirming it would need validation
data we do not have, and adopting it would repeat the error that measuring seed variance
was meant to expose. The overall result matches Gao et al. (2021), who report that
contrastive training subsumes the benefit of flow- and whitening-based post-processing.

---

## Results

All figures are dev-set scores to three-decimal precision.

### Semantic Textual Similarity (STS)


### All runs
 
Every configuration evaluated, in experiment order. Sweeps are broken out one row per
setting rather than collapsed into ranges. Rows marked with a dagger are reference points
shared with an adjacent experiment and were not rerun.
 
| Exp | Configuration | Dev r |
| --- | --- | --- |
| — | Part 1 baseline: `[CLS]` pooler + concat + MSE | 0.379 |
| 1 | Mean pooling, concat head retained | 0.366 |
| 2 | Mean pooling + cosine head | 0.659 |
| 3 | Max pooling + cosine head | 0.463 |
| 4 | + cosine embedding loss (w = 0.5) | 0.677 |
| 5 | + MNRL (τ = 0.05, w = 0.5) | 0.804 |
| 6a | MNRL w = 0.3 | 0.794 |
| 6b | MNRL w = 1.0 | 0.798 |
| 6c | MNRL w = 2.0 | 0.782 |
| 7a | SimCSE unsupervised (w = 0.5) | 0.711 |
| 7b | SimCSE unsupervised (w = 0.1) | 0.696 |
| 7c | SimCSE only, no supervision | 0.651 |
| 8 | NLI pretraining — invalid, trained without NLI data | 0.689 |
| 8a | SNLI triplet pretraining, seed 11711 | 0.847 |
| 8b | SNLI triplet pretraining, seed 42 | 0.851 |
| 8c | SNLI triplet pretraining, seed 7 | 0.848 |
| 9a | MNRL τ = 0.01 | 0.746 |
| 9b | MNRL τ = 0.10 | 0.801 |
| 10 | Batch size 128 | 0.803 |
| 11 | LR warmup 10% + cosine decay | 0.803 |
| 12 | TF-IDF hard negatives mined from STS train | 0.834 |
| 13a | CoSENT (τ = 0.05, w = 1.0) | 0.490 |
| 13b | CoSENT (τ = 0.5, w = 0.1, MNRL off) | 0.660 |
| 13c | CoSENT (τ = 0.5, w = 0.1, MNRL on) | 0.800 |
| 14a | AnglE (w = 1.0) | 0.834 |
| 14b | AnglE (w = 0.1) | 0.830 |
| 15a | SMART (w = 10) | 0.833 |
| 15b | SMART (w = 100) | 0.832 |
| 16 | Cross-attention interaction layer | 0.828 |
| 19 | + symmetry augmentation | 0.831 |
| 20 | + warmup + gradient clipping (15 epochs) | 0.829 |
| 21 | + symmetry + warmup + clipping | 0.829 |
| 22† | Unregularised reference (= Exp 19) | 0.831 |
| 22a | Weight decay λ = 0.01 | 0.830 |
| 22b | Weight decay λ = 0.1 | 0.829 |
| 22c | Cross-attention dropout 0.3 | 0.829 |
| 22d | Weight decay λ = 0.01 + dropout 0.3 | 0.829 |
| 22e | Weight decay λ = 20 | 0.823 |
| 23 | PAWS hard-negative pretraining | 0.835 |
| 24† | Stock minBERT reference (= Exp 19) | 0.831 |
| 24a | Quora MNRL pretraining, 1 epoch | 0.840 |
| 24b | Quora MNRL pretraining, 2 epochs | 0.840 |
| 25a | QQP encoder initialisation | 0.830 |
| 25b | QQP encoder + Quora pretraining | 0.832 |
| 26a | Quora pretraining, seed 42 | 0.836 |
| 26b | Quora pretraining, seed 7 | 0.837 |
 
Two groups of rows are not training runs of their own. The daggered rows are reference
scores carried over from Exp 19. The whitening results below are transformations applied
to embeddings from a saved checkpoint, with the 0.830 of Exp 25a as their reference point.
 
| Configuration | Dev r |
| --- | --- |
| Whitening to 768 dimensions | 0.815 |
| Whitening to 384 dimensions | 0.829 |
| Whitening to 256 dimensions | 0.832 |
| Whitening to 128 dimensions | 0.835 |
| Whitening to 64 dimensions | 0.827 |

Overall techniques that added the improvement


| Configuration | Pearson r |
| --- | --- |
| Baseline (`[CLS]` pooler + concat + MSE) | 0.379 |
| Improvement 1 — cosine head + mean pooling | 0.659 |
| Improvement 2 — + cosine embedding loss | 0.677 |
| Improvement 3 — + MNRL contrastive loss | 0.804 |
| Improvement 4 — + cross-attention layer | 0.828 |
| Improvement 5 — + symmetry augmentation | 0.831 |
| Improvement 6 — + teammate encoder init | 0.830 |
| **Improvement 7 — + SNLI triplet pretraining** | **0.849** |

Final dev Pearson r is 0.849, the mean of three seeds (0.847 / 0.851 / 0.848), an increase
of 0.470 over the Part 1 baseline.

Improvements 5 and 6 are inside the noise floor and are retained for the reasons given in
Experiments §4 and §5, not for their scores.




### Techniques retained

| Technique | Dev r |
| --- | --- |
| Cosine similarity head + mean pooling | 0.659 |
| Cosine embedding loss | 0.677 |
| MNRL contrastive (τ = 0.05, w = 0.5) | 0.804 |
| Cross-attention interaction layer | 0.828 |
| STS symmetry augmentation | 0.831 |
| Teammate QQP encoder initialisation | 0.830 |
| SNLI triplet pretraining | 0.849 |

Retained does not mean each of these improved dev r. Symmetry augmentation is kept for
convergence speed and the teammate encoder initialisation only so the reported run
reproduces exactly; both sit inside the noise floor on score. Four of the seven — the
cosine head, MNRL, cross-attention and SNLI pretraining — account for 0.450 of the 0.470
total improvement.

### Techniques evaluated and discarded

| Technique | Dev r |
| --- | --- |
| Part 1 baseline | 0.379 |
| Mean pooling, concat head kept | 0.366 |
| Max pooling (ablation) | 0.463 |
| MNRL weight sweep {0.3, 1.0, 2.0} | ≤ 0.798 |
| SimCSE unsupervised | 0.651–0.711 |
| MNRL τ sweep {0.01, 0.10} | 0.746 / 0.801 |
| Batch size 128 | 0.803 |
| LR warmup + cosine decay | 0.803 |
| CoSENT ranking loss | 0.490 / 0.800 |
| Warmup + gradient clipping | 0.829 |
| Weight decay {0.01, 0.1, 20} | 0.823–0.830 |
| Cross-attention dropout 0.3 | 0.829 |
| Quora MNRL pretraining (superseded by SNLI) | 0.838 |
| TF-IDF mined hard negatives | 0.834 |
| PAWS hard negatives | 0.835 |
| AnglE loss | 0.830 / 0.834 |
| SMART adversarial smoothness | 0.832 / 0.833 |
| Whitening post-processing | 0.815–0.835 |

Seventeen of the twenty-four techniques evaluated were discarded.

### Discussion of results

Three patterns hold across the 43 runs.

**Architecture and data moved the metric; losses and regularisation did not.** Of the four
techniques carrying the gain, two change what gets encoded (cosine head, cross-attention)
and two change what data the encoder sees (MNRL over real pairs, NLI pretraining). Ten
loss-function variants and five regularisation settings produced nothing outside the noise
floor. Once the model saturates a 5,719-pair dataset, reweighting the objective cannot add
information that is not present.

**The bi-encoder ceiling was structural, not a tuning problem.** Nine consecutive
experiments plateaued at 0.800–0.804 while varying temperature, batch size, LR schedule
and three different loss formulations. All shared the property that each sentence was
encoded independently, and the plateau broke only when that assumption changed.

**Small datasets defeat regularisation.** A train/dev gap of 0.978 against 0.831 reads as
overfitting, but no penalty setting improved dev r, while adding 149,145 external triplets
improved it by 0.019. The gap indicated a data shortage rather than an over-flexible
model.

---

### Hyperparameter Optimization

We did not run automated search (Ray Tune, Optuna). With runs at 6–19 minutes and a noise
floor of 0.004, a random search would mostly have sampled noise, and could not have
distinguished a real effect from a lucky seed without repeated runs per configuration.
Instead each hyperparameter was swept individually, with a stated hypothesis, on top of
the best configuration available at the time.

| Parameter | Values tried | Chosen | Result |
| --- | --- | --- | --- |
| Pooling | CLS, mean, max | mean | 0.379 / 0.659 / 0.463 |
| Similarity head | concat+linear, cosine | cosine | 0.379 / 0.659 |
| MNRL weight | 0.3, 0.5, 1.0, 2.0 | 0.5 | 0.794 / 0.804 / 0.798 / 0.782 |
| MNRL temperature τ | 0.01, 0.05, 0.10 | 0.05 | 0.746 / 0.804 / 0.801 |
| CoSENT temperature | 0.05, 0.5 | dropped | 0.490 / 0.800 |
| Batch size | 64, 128 | 64 | 0.804 / 0.803 |
| Hidden dropout | 0.1, 0.3 | 0.3 | see confound note below |
| Warmup ratio | 0.0, 0.1 | 0.0 | 0.804 / 0.803 |
| Weight decay λ | 0, 0.01, 0.1, 20 | 0 | 0.831 / 0.830 / 0.829 / 0.823 |
| Cross-attention heads | 8 | 8 | not swept |
| Cross-attention dropout | 0.1, 0.3 | 0.1 | 0.831 / 0.829 |
| Epochs | 4–15 | 4–5 | peak moved 9 → 4 → 3 as initialisation improved |
| SNLI pretrain epochs | 1, 2 | 1 | same result, two minutes cheaper |

Two findings from this sweep are worth more than the values themselves.

**Temperature is not a free parameter.** Both τ = 0.01 in MNRL and τ = 0.05 in CoSENT
produced catastrophic failures for the same underlying reason, an exponential term
overwhelming the rest of the loss. Sweeping τ blindly and reading only the final scores
would have recorded two bad methods instead of one numerical bug and one genuine
redundancy.

**A carried-over default silently confounded eight experiments.** We lowered dropout from
0.3 to 0.1 for the SimCSE experiments, where dropout is the augmentation. That setting
then persisted through seven further experiments, exaggerating overfitting in all of them.
It was caught only when a CoSENT rerun with dropout restored scored 0.140 higher than the
same configuration at dropout 0.1. Every run afterwards passes `--hidden_dropout_prob 0.3`
explicitly rather than relying on the default.

---

## Visualizations
Two conventions apply throughout. First, `evaluation.py` computes Pearson correlation
only, so per-epoch dev *loss* was never logged and cannot be reconstructed without
rerunning all 43 experiments; we plot dev Pearson r in its place, which is the task metric
and the quantity model selection actually used. Second, where a figure shows a derived
quantity rather than a logged one, the transformation is stated in the text below it.

### Dev performance per epoch

![Dev Pearson r per epoch for each improvement stage](figures/sts/v1_dev_progression.png)

One curve per improvement stage; the star marks the epoch whose checkpoint was saved.

The cosine-head run (grey, dotted) peaks at 0.657 in epoch 3 and then falls steadily to
0.572 by epoch 10, the only configuration whose dev score degrades over training. Adding
MNRL (grey, solid) removes that decline and holds 0.79–0.80 from epoch 4 onward without
ever exceeding 0.804. Cross-attention (blue) passes that level at epoch 4 and continues
rising to 0.828 at epoch 9. Symmetry augmentation (orange) reaches a comparable level four epochs
earlier and then drifts down. SNLI pretraining (green) opens at 0.843 in epoch 1, above
every other configuration's best score at any epoch, and needs only five epochs in total.

### Convergence speed

![Epochs required to reach each dev r level](figures/sts/v2_convergence_speed.png)

For each configuration and each dev-r threshold, the bar height is the first epoch at which
that threshold was reached; "never" means the run finished below it. Lower is faster.

Reaching r ≥ 0.80 takes MNRL six epochs, cross-attention three, symmetry augmentation two,
and SNLI pretraining one. At r ≥ 0.82 the bi-encoder never arrives, and at r ≥ 0.84 only
SNLI does.

The 0.82 column also shows the converse case: symmetry augmentation gets there in two
epochs where cross-attention alone needs five, yet cross-attention overtakes it by epoch 6
and finishes higher (v1). Faster convergence and better final performance are separate
properties, and symmetry augmentation buys the first without the second.

### Overfitting dynamics

![Train and dev Pearson r, and the gap between them](figures/sts/v3_overfitting_dynamics.png)

Left: train (solid) against dev (dashed) Pearson r. Right: the generalisation gap, computed
as train r minus dev r at the same epoch.

Every variant's gap grows monotonically, from roughly 0.08–0.11 at epoch 1 to 0.14–0.17 by
the end. No intervention flattened it. Cross-attention holds the smallest gap of the
ten-epoch runs at every epoch, so its dev gain does not come at the cost of extra
memorisation. SNLI pretraining starts at the highest train r of any run, 0.945 in epoch 1,
yet its gap grows more slowly than the others over the epochs it shares with them,
consistent with the interpretation that the additional data, rather than any penalty term,
is what constrains the model.

### Training loss

![Training loss per epoch for four runs with identical loss composition](figures/sts/v4_training_loss.png)

Restricted to four runs sharing an identical loss composition
(`MSE + 0.5·cosine + 0.5·MNRL`). Loss is not comparable across runs with different terms,
since adding a term mechanically raises the total.

Epoch-1 loss orders exactly by quality of initialisation: 2.99 for cross-attention from
stock weights, 2.58 with symmetry augmentation, 2.43 with the teammate encoder, 2.19 after
SNLI pretraining. Better-initialised runs also descend faster. The symmetry run reaches
the lowest final loss of the four while scoring 0.018 below the SNLI run on dev, a
reminder that within this family the training loss ranks the runs differently from the
metric we select on.

### Failure modes visible during training

![SimCSE, CoSENT and MNRL temperature failure modes](figures/sts/v5_failure_modes.png)

Left: SimCSE's training loss falls from 0.033 to 0.005 while dev r declines from 0.651 to
0.608, the signature of a model solving the pretext task rather than learning semantics.
Middle: CoSENT at τ = 0.05 oscillates between 0.39 and 0.49 with no upward trend, while
τ = 0.5 climbs smoothly to just under the MNRL baseline; the difference between a bad
method and an overflowing one is visible here but not in the final scores. Right: MNRL at
τ = 0.01 never exceeds 0.75, whereas τ = 0.05 and τ = 0.10 both converge into the 0.80
band and are nearly indistinguishable from each other.

### Seed variance

![Dev Pearson r under three random seeds](figures/sts/v6_seed_variance.png)

The same configuration under three seeds; the shaded band is the full spread across seeds
at each epoch, and the heavy line is their mean.

Best scores span 0.847 to 0.851, a spread of 0.004, which is the reference for every
"within noise" claim in this write-up. The spread is narrowest at epoch 1 and widens as
training proceeds, so late-epoch comparisons are the least reliable ones. All three seeds
peak between epochs 2 and 4 and decline afterwards, which is the direct evidence behind
the 4–5 epoch setting in the final recipe.

### Transfer sources

![Dev performance by transfer pretraining source](figures/sts/v8_transfer_sources.png)

Left: dev r during STS fine-tuning for each pretraining source. Right: dev r after one
fine-tuning epoch (faded) against the best epoch (solid).

The epoch-1 scores order the sources exactly by corpus size, and SNLI's epoch-1 score of
0.842 already exceeds every other source's best. The distance between the faded and solid
bars also shrinks as the source grows, from 0.015 with no transfer to 0.005 for SNLI:
better-pretrained encoders have less left to learn from the 5,719 STS pairs. Note that
corpus size and negative-label type covary across these four sources, so this figure
cannot attribute SNLI's advantage to either alone.

### Regularisation

![Dev and train Pearson r for five regularisation settings](figures/sts/v9_regularisation.png)

Five regularisation settings against the unregularised baseline. Left: dev r. Right: train
r, which is what a regulariser is supposed to move first.

Four of the five dev curves are visually inseparable from the baseline across all eight
epochs. Only λ = 20 separates, and it lowers both panels: its dev peak is 0.823 at epoch 3
against the baseline's 0.831 at epoch 4, and it then collapses to 0.791 by epoch 8. The
right panel is the more informative one, since it shows that four settings left the
training fit completely untouched. An intervention that does not change how well a model
fits its training data has not been tested, whatever its dev score says.

### Cross-attention weights

![Cross-attention weights for two dev pairs](figures/sts/v10_cross_attention.png)

Attention weights extracted from the trained model (`models/sts_exp08_seed42.pt`) for two
dev pairs, one with a high gold score and one unrelated. Rows are sentence 1 tokens acting
as queries, columns are sentence 2 tokens acting as keys, each row sums to one, and the
map is averaged over the eight heads. White squares mark the per-row argmax.

Cross-attention was the largest architectural gain in the project, so we inspected what
the layer had learned. It has not learned token alignment. Both sentences here have nine
tokens, so the uniform weight is 1/9 = 0.111, and the observed weights span 0.071–0.130.
KL divergence from uniform is 0.0022 averaged over heads and 0.0096 for the most selective
single head, against a maximum of ln 9 = 2.20 for a nine-way distribution: the learned
distributions sit at roughly a tenth of a percent of the way from uniform to fully peaked.
No head is selective. The token *guitar* shows no preference for *instrument*, and within
each panel every content-token row places its argmax in the same key column, so the
distribution barely depends on which token is querying. The argmax column does differ
between the two pairs, which is the only sense in which the weights respond to content.

If the weights are near-uniform then the attention output is
`Σⱼ aᵢⱼ·W_V·h2ⱼ ≈ W_V·mean(h2)` for every query position i, and after the output
projection the sublayer computes `h1_cross ≈ LayerNorm(h1 + W·mean(h2))` with
`W = W_O W_V` a learned linear map. The layer therefore conditions each sentence's
representation on a single global summary of its partner before pooling, rather than
aligning tokens between them.

The performance gain remains real and reproducible at 0.024, six times the seed-noise
floor. What this measurement rules out is our original explanation of it. The natural
follow-up, which we did not have time to run, is to replace the attention sublayer with
`LayerNorm(h1 + W·mean(h2))` directly: if that matches 0.828, the eight heads are
redundant and the same benefit is available at a fraction of the cost. Note that the
ablation must keep the learned projection, since the collapse is to a projected mean and
not to the raw mean.

Two limitations apply. This is two hand-picked pairs from a single seed's checkpoint, so
the KL figures characterise these examples rather than the model in general; computing the
same statistic over all 1,430 dev pairs would settle it and requires only a forward pass.
Separately, the left pair carries a high gold score but is predicted at 2.32 out of 5,
which is the range compression documented in the error analysis below appearing in a
single worked example.

### Error analysis

![Predicted against gold similarity, error by band, and score distributions](figures/sts/v7_error_analysis.png)

Left: predicted against gold similarity for all 1,430 dev pairs, with an ordinary
least-squares fit. Middle: mean absolute error within each gold-similarity band. Right:
distributions of gold and predicted scores.

The model compresses the output range. The fitted slope is 0.67 rather than 1.0, predicted
σ is 1.16 against gold σ of 1.47, and predictions essentially never fall below 0.4 even
though gold scores reach 0.0. Error is correspondingly worst at the dissimilar end, with a
mean absolute error of 0.93 in the 0–1 band against 0.52 in the 4–5 band, and the middle
bands sitting near 0.55.

Pearson r is invariant to affine rescaling, so this costs nothing on our metric, but it
would matter under Spearman ρ or MSE, and it would matter for any downstream use of the
raw scores. It is expected behaviour for a cosine head, whose output is squashed toward the
middle of the rescaled range. Calibrating the output range, by fitting a linear correction
on the training set or by training against standardised targets, is the clearest remaining
improvement we did not pursue.

---

## Members Contribution

**Mohd Uwaish — Semantic Textual Similarity (STS):** implemented the STS improvements in
`multitask_classifier.py`: masked mean pooling and cosine similarity head, MNRL/NT-Xent
contrastive loss, the cross-attention interaction layer (`encode_pair`), symmetry
augmentation, and four transfer-pretraining paths (SNLI, PAWS, Quora, TF-IDF-mined hard
negatives), plus the CoSENT, AnglE and SMART auxiliary losses. Built the offline triplet
caching pipeline, the checkpoint warm-start mechanism (`--init_checkpoint`), and the
LR-scheduler, gradient-clipping and weight-decay controls. Ran and analysed 43 training
runs, established the seed-variance noise floor, produced all
figures (`figures/make_sts_figures.py`), and wrote the STS sections of this README. Raised
STS dev Pearson r from 0.379 to 0.849.

---

# References

1. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) — mean pooling and the cosine head (Methodology §1).
2. Henderson, M. et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply.* [arXiv:1705.00652](https://arxiv.org/abs/1705.00652) — MNRL (Methodology §2).
3. Gao, T., Yao, X. & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP. [arXiv:2104.08821](https://arxiv.org/abs/2104.08821) — the unsupervised variant we could not reproduce a gain from (Experiments §3), the supervised NLI-triplet formulation behind our best result (Methodology §4), and the finding that contrastive training subsumes whitening (Experiments §9).
4. Bowman, S. et al. (2015). *A Large Annotated Corpus for Learning Natural Language Inference.* EMNLP. [arXiv:1508.05326](https://arxiv.org/abs/1508.05326) — SNLI, the source of the 149,145 pretraining triplets.
5. Huang, X. et al. (2024). *CoSENT: Consistent Sentence Embedding via Similarity Ranking.* IEEE/ACM TASLP. — evaluated in Experiments §2.
6. Li, X. & Li, J. (2024). *AnglE-optimized Text Embeddings.* ACL. [arXiv:2309.12871](https://arxiv.org/abs/2309.12871) — evaluated in Experiments §8.
7. Jiang, H. et al. (2020). *SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization.* ACL. [arXiv:1911.03437](https://arxiv.org/abs/1911.03437) — evaluated in Experiments §8, with the two implementation deviations noted there.
8. Su, J. et al. (2021). *Whitening Sentence Representations for Better Semantics and Faster Retrieval.* [arXiv:2103.15316](https://arxiv.org/abs/2103.15316) — the post-processing analysed in Experiments §9.