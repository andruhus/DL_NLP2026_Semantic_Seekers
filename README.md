# Semantic Seekers

- **Group name:** Semantic Seekers
- **Group code:** _To be added_
- **Group repository:** [DL_NLP2026_Semantic_Seekers](https://github.com/andruhus/DL_NLP2026_Semantic_Seekers)
- **Tutor responsible:** _To be added_
- **Group team leader:** _To be added_
- **Group members:** _To be added_

# Setup instructions

## Environment and dependencies

Make sure that Anaconda or Miniconda is installed. From the repository root, create the `dnlp` environment and install the required dependencies with:

```sh
source setup.sh
conda activate dnlp
```

On the GWDG cluster, use the GWDG-specific setup script instead:

```sh
source setup_gwdg.sh
conda activate dnlp
```

The setup scripts install the project dependencies in the `dnlp` environment. The GWDG setup also downloads `bert-base-uncased` and `facebook/bart-large` before compute jobs are started. Local QQP and PTD runs require the corresponding model to be available in the Hugging Face cache; `bart_detection.py` loads `facebook/bart-large` with local-files-only mode.

All commands below should be executed from the repository root. Add `--use_gpu` when a CUDA-capable GPU is available; the reported experiments were run with `--use_gpu` on the GWDG cluster.

## Quora Question Pairs (QQP)

The QQP train, development, and test data are included at:

```text
data/quora-paraphrase-train.csv
data/quora-paraphrase-dev.csv
data/quora-paraphrase-test-student.csv
```

The model is evaluated after every epoch. Whenever development accuracy improves, the best checkpoint is written to `models/`. After training, the selected checkpoint is loaded and prediction files are written to `predictions/bert/`.

The linear and MLP heads were implemented in different source revisions rather than selected through a command-line option. The current `main` branch implements the MLP head, so its runs can be reproduced directly below. Reproducing the linear-head variant requires its recorded revision.

### Improvement 1: Pair-Feature Linear Head Run

```sh
python multitask_classifier.py \
  --task qqp \
  --seed 11711 \
  --option finetune \
  --use_gpu \
  --local_files_only \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-5
```

The corresponding archived SLURM script, log, and predictions are in `experiments/qqp_improvement_1_only_sentence_pair_repr/`.

### Improvement 2: MLP Head

#### MLP Head with Default Dropout

This run used the same pair interaction representation and the MLP head with the default dropout value.

```sh
python multitask_classifier.py \
  --task qqp \
  --seed 11711 \
  --option finetune \
  --use_gpu \
  --local_files_only \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-5
```

The corresponding SLURM script is `slurm_scripts/run_qqp_mlp_head_full.sh`, and the archived predictions are in `experiments/qqp_mlp_head/full_run_predictions/`.

#### MLP Head with Lower Dropout

This run used the same pair interaction representation and the same MLP head, but reduced dropout and increased the epoch budget.

```sh
python multitask_classifier.py \
  --task qqp \
  --seed 11711 \
  --option finetune \
  --use_gpu \
  --local_files_only \
  --epochs 7 \
  --batch_size 8 \
  --lr 1e-5 \
  --hidden_dropout_prob 0.1 \
  --early_stopping \
  --patience 2
```

The corresponding SLURM script is `slurm_scripts/run_qqp_mlp_head_dropout01_7ep_es.sh`.

The final prediction files from the best run were copied to:

```text
experiments/qqp_mlp_head/dropout01_7ep_es_predictions/quora_mlp_head_dropout01_7ep_es_job15591323_dev.csv
experiments/qqp_mlp_head/dropout01_7ep_es_predictions/quora_mlp_head_dropout01_7ep_es_job15591323_test.csv
```

The corresponding checkpoint was also used to initialize experimental training runs for the STS (Semantic Textual Similarity) task, since both tasks involve sentence-pair comparison. However, this transfer is only expected to be potentially useful for the shared BERT encoder; the QQP-specific classification head is not directly applicable to STS because STS predicts a graded similarity score rather than a binary paraphrase label.

## Stanford Sentiment Treebank (SST)

_Setup and reproduction instructions are to be added._

## Semantic Textual Similarity (STS)

_Setup and reproduction instructions are to be added._

## Paraphrase Type Detection (PTD)

The PTD train, development, and test data are included at:

```text
data/etpc-paraphrase-train.csv
data/etpc-paraphrase-dev.csv
data/etpc-paraphrase-detection-test-student.csv
```

`run_bart_detection.sh` is the SLURM wrapper that submits the requested run and invokes `bart_detection.py`. The Python script evaluates the model after each epoch and saves the checkpoint with the highest development accuracy; an earlier checkpoint wins a tie. Each method reloads its own accuracy-selected checkpoint for reporting.

In a multi-method run, the script currently writes test predictions from the final method in command order rather than automatically selecting the strongest method. The output model is therefore aggressive Weighted BCE for the BCE comparison, capped Weighted BCE for the smoothed comparison, $\gamma=4$ for the exploratory focal sweep, and $\gamma=1.25$ for the confirmatory focal run. Predictions are written to `predictions/bart/etpc-paraphrase-detection-test-output.csv`.

### Unweighted BCE vs. Aggressive Weighted BCE

We compared unweighted BCE with the aggressive inverse-frequency Weighted BCE for 25 epochs using a batch size of 16. The current comparison is rerun with:

```sh
sbatch run_bart_detection.sh 25 compare \
    --compare_bce_only \
    --batch_size 16 \
    --use_gpu
```

### Smoothed Weighted BCE Variants

We trained the three smoothed Weighted BCE variants for 25 epochs with a batch size of 16. The unweighted BCE results from the aggressive Weighted BCE experiment are reused as the common reference; `compare_non_aggressive_weighted` does not rerun that baseline. The current experiment can be rerun with:

```sh
sbatch run_bart_detection.sh 25 compare_non_aggressive_weighted \
    --batch_size 16 \
    --weighted_bce_cap 20 \
    --use_gpu
```

This command trains square-root, logarithmic, and capped weighting in that order.

### Exploratory 5-Epoch Focal Sweep

The exploratory sweep evaluated 16 focusing parameters for 5 epochs with a batch size of 16:

```sh
sbatch run_bart_detection.sh 5 focal \
    --batch_size 16 \
    --focal_gamma 0.25 \
    --focal_gamma 0.5 \
    --focal_gamma 0.62 \
    --focal_gamma 0.75 \
    --focal_gamma 0.8 \
    --focal_gamma 0.87 \
    --focal_gamma 0.9 \
    --focal_gamma 0.95 \
    --focal_gamma 1 \
    --focal_gamma 1.05 \
    --focal_gamma 1.1 \
    --focal_gamma 1.15 \
    --focal_gamma 1.25 \
    --focal_gamma 1.5 \
    --focal_gamma 2 \
    --focal_gamma 4 \
    --use_gpu
```

The dedicated `focal` mode runs only the focal-loss experiments and does **not** rerun unweighted BCE.

### Confirmatory 25-Epoch Focal Run

Based on the exploratory scores, we selected four moderate values for a 25-epoch confirmatory run:

```sh
sbatch run_bart_detection.sh 25 focal \
    --batch_size 16 \
    --focal_gamma 0.87 \
    --focal_gamma 1.1 \
    --focal_gamma 1.15 \
    --focal_gamma 1.25 \
    --use_gpu
```

## Paraphrase Type Generation (PTG)

_Setup and reproduction instructions are to be added._

# Methodology

## Quora Question Pairs (QQP)

### Task, Baseline, and Contribution

Quora Question Pairs (QQP) paraphrase detection is a binary sentence-pair classification task. Given two questions, the model predicts whether both questions express the same meaning. The output is a single logit, which is converted to a probability with the sigmoid function during evaluation and then thresholded for binary classification.

The original BERT-based classifier represents a question pair by concatenating the pooled BERT embeddings of both questions:

$$
u = \mathrm{BERT}(q_1), \qquad v = \mathrm{BERT}(q_2)
$$

$$
h_{\mathrm{base}} = [u, v].
$$

This representation gives the classifier access to both question embeddings, but it does not explicitly encode how the embeddings differ or where they overlap dimension-wise.

As a reference point, the course project description reports a QQP development accuracy of approximately **0.765** for the default BERT-based setup with simple task-specific heads. In an earlier local baseline run with the original pair representation $[u, v]$, we observed a QQP development accuracy of **0.785**.

Our work focuses on improving the QQP paraphrase classifier in three steps. First, we change the sentence-pair representation. Second, we replace the linear classification head with a small Multi-Layer Perceptron (MLP) classifier head. Third, we adjust the training configuration for the MLP head by reducing dropout and using development-based checkpoint selection with optional early stopping.

The final selected QQP model reaches a development accuracy of **0.868**.

### Loss Function

For each label $y \in \{0,1\}$, model logit $z$, and sigmoid probability $\sigma(z)$, the loss is:

$$\mathcal{L}=- y \log \sigma(z)- (1-y)\log(1-\sigma(z)).$$

The implementation uses binary cross entropy with logits combining the sigmoid transformation and binary cross-entropy loss in a numerically stable way.

### Improvement 1: Sentence-Pair Interaction Features

The first improvement changes the sentence pair representation used for the paraphrase detection.

Instead of only passing the two pooled BERT embeddings to the classifier,

$$
h_{\mathrm{base}} = [u, v],
$$

we construct an enriched sentence-pair representation:

$$
h_{\mathrm{pair}} = [u, v, \lvert u-v \rvert, u \odot v].
$$

Here,

$$
u = \mathrm{BERT}(q_1), \qquad v = \mathrm{BERT}(q_2),
$$

where $q_1$ and $q_2$ are the two input questions.

The term $\lvert u-v \rvert$ is the element-wise absolute difference between both sentence embeddings. It provides a direct dimension-wise distance signal. This is related to distance-based similarity measures such as Manhattan / L1 distance, but instead of reducing the difference to one scalar, we keep the full vector so that the classifier can learn which dimensions are informative.

The term $u \odot v$ is the Hadamard product, i.e. the element-wise product of both embeddings. It highlights dimensions where both question embeddings are jointly active and provides a complementary overlap signal.

Since BERT-base produces 768-dimensional pooled embeddings, the enriched pair representation has dimensionality:

$$
4 \times 768 = 3072.
$$

This improvement is related to established sentence-pair matching representations. Sentence-BERT (Sentence Embeddings using Siamese BERT-Networks) uses Siamese BERT encoders and pair-combination features such as $[u, v, \lvert u-v \rvert]$ for classification objectives.[^1] InferSent-style sentence-pair models for Natural Language Inference use a closely related representation containing $u$, $v$, $\lvert u-v \rvert$, and $u*v$ before fully connected classification layers.[^2]

We therefore do **not** claim this representation as a novel architectural contribution. Instead, we apply this established sentence-pair matching idea to our initial architecture and evaluate whether it improves over the simpler baseline representation.

### Improvement 2: MLP Classification Head & Dropout

The second improvement keeps the enriched sentence-pair representation from Improvement 1, but replaces the single linear classification head with a small MLP head.

The linear-head variant predicts the paraphrase logit as:

$$
z = W h_{\mathrm{pair}} + b,
$$

where $z$ is the unnormalized paraphrase logit.

The MLP variant instead uses:

$$
z =
W_2\,
\mathrm{Dropout}
\left(
\mathrm{ReLU}
\left(
W_1\,\mathrm{Dropout}(h_{\mathrm{pair}}) + b_1
\right)
\right)
+b_2.
$$

In the implementation, the first linear layer maps the 3072-dimensional pair representation back to the BERT hidden size of 768:

$$
3072
\xrightarrow{\mathrm{Dropout}}
3072
\xrightarrow{\mathrm{Linear}}
768
\xrightarrow{\mathrm{ReLU}}
768
\xrightarrow{\mathrm{Dropout}}
768
\xrightarrow{\mathrm{Linear}}
1.
$$

The choice of 768 as the hidden size is motivated by the BERT-base embedding size. The enriched pair representation consists of four 768-dimensional parts, and the MLP head compresses this representation back to the natural hidden size of the BERT backbone before producing the final logit.

### Altering Training Configuration: Changing Dropout Rate, Longer Training, and Early Stopping

The third step does not change the pair representation or the classifier head and its MLP architecture. Instead, it changes the training configuration.

The first MLP experiment used the original default dropout value:

```text
hidden_dropout_prob = 0.3
```

with 3 training epochs. This run reached **0.849** QQP development accuracy and therefore did not improve over the simpler pair-feature linear-head model, which reached **0.850**.

We therefore ran a targeted follow-up experiment using the same sentence-pair interaction representation and the same MLP head, but with:

```text
hidden_dropout_prob = 0.1
epochs = 7
early_stopping = True
patience = 2
```

Dropout is a standard regularization method for neural networks that randomly disables units during training to reduce co-adaptation and overfitting.[^3] Since the added MLP head introduces additional trainable parameters, the default dropout value of `0.3` may have regularized the classifier head too strongly. Reducing dropout to `0.1` was therefore tested as a targeted follow-up.

The value `0.1` was **not** selected by an exhaustive hyperparameter sweep. We did not test additional values such as `0.01`, `0.05`, `0.15`, or `0.2`. Therefore, this experiment should be interpreted as a targeted training-configuration improvement, not as proof that `0.1` is the globally optimal dropout value. Since the MLP head adds extra trainable parameters on top of the sentence-pair representation, but the `0.3` dropout run did not outperform the linear head, we hypothesized that this dropout rate might be too aggressive for the newly added classifier head and evaluated `0.1` as a less restrictive regularization setting.

Early stopping is a validation-based model-selection and regularization technique where validation performance is used to stop training or select a checkpoint before overfitting becomes worse.[^4] In our setup, early stopping was primarily introduced to avoid wasting compute budget on additional epochs after development accuracy stopped improving. Since the training loop already saves the best checkpoint whenever development accuracy improves, early stopping mainly reduces unnecessary training time rather than changing the checkpoint-selection criterion.

## Stanford Sentiment Treebank (SST)

_Methodology is to be added._

## Semantic Textual Similarity (STS)

_Methodology is to be added._

## Paraphrase Type Detection (PTD)

### Task, Baseline, and Contribution

ETPC paraphrase-type detection is formulated as a multi-label classification problem with 26 output labels. Following the course-provided task setup, the baseline fine-tunes `facebook/bart-large`[^6] with a 26-output linear classification head and binary cross-entropy (BCE). The course-provided task setup explicitly specifies `facebook/bart-large` as the starting point, and `setup_gwdg.sh` downloads that checkpoint. This extension retains that provided model and changes only its training objective; it does not introduce another pretrained model or external embedding.

The ETPC labels are strongly imbalanced. In the archived, unfiltered 2,730-row training data used for the preliminary results, only 11,648 of the 70,980 binary label assignments are positive (16.410%). Individual-label positive counts range from 3 to 2,711. Plain BCE can therefore obtain high accuracy by favoring negative decisions while learning little about rare positive labels. We investigate whether inverse-frequency Weighted BCE, smoothed Weighted BCE, or focal loss can improve minority-label behavior and/or reach a useful solution faster than BCE.

### Research Questions and Hypotheses

1. **Development-set performance:** Does an imbalance-aware objective improve the course's primary ETPC-detection metric, development accuracy, over the unweighted BCE baseline under the same training and checkpoint-selection protocol?
2. **Minority-label behavior:** Does it improve mean per-label MCC and per-label precision/recall, especially for rare paraphrase types, without an unacceptable reduction in accuracy?
3. **Training efficiency:** Under the same fixed epoch budget, does focal loss or smoothed Weighted BCE reach a strong development score earlier than ordinary BCE?

| Method | Change from baseline | Hypothesis |
| --- | --- | --- |
| Unweighted BCE | None | Strong aggregate accuracy, but majority-negative bias for rare labels |
| Aggressive Weighted BCE | Positive term for label $c$ multiplied by $N_c^-/N_c^+$ | Better rare-positive recall and MCC, potentially at the cost of accuracy and precision |
| Smoothed Weighted BCE | Compress or cap $N_c^-/N_c^+$ | Retain some minority-label benefit while avoiding unstable extreme weights |
| Focal loss | Down-weight easy decisions by $(1-p_t)^\gamma$ | Focus on difficult decisions without applying fixed class-level weights, improving early convergence and MCC |

### Baseline: Unweighted BCE

The baseline concatenates the two sentences with `</s>`, tokenizes to a maximum length of 512, and passes the result through BART-large. The classifier uses the hidden state of the first token to produce one logit for each of the 26 paraphrase types. Every output is treated as an independent binary decision and optimized with `BCEWithLogitsLoss`[^7]; at evaluation time, sigmoid probabilities greater than 0.5 are mapped to positive predictions.

### Proposed Loss Functions

The following preliminary class counts and weights come from the archived, overlapping 2,730-row training data. The current code removes development IDs before computing these values, so the weights must be regenerated for current reruns.

#### Method 1: Aggressive Weighted BCE

For each paraphrase type $c$, we computed a positive-class weight from the training data:

$$
w_c = \frac{N_c^-}{N_c^+},
$$

where $N_c^+$ and $N_c^-$ are the numbers of positive and negative training examples for label $c$. The resulting vector is passed to PyTorch's `BCEWithLogitsLoss` as `pos_weight`, yielding

$$
\mathcal{L}_{i,c} = -w_c y_{i,c}\log\sigma(z_{i,c}) - (1-y_{i,c})\log(1-\sigma(z_{i,c})).
$$

This calculation is implemented in `paraphrase_detection/weighted_bce.py` and is called on `train_labels` in `bart_detection.py`.

##### Archived Training-Set Class Weights

| Label ID | Positive | Negative | `pos_weight` |
| ---: | ---: | ---: | ---: |
| 1 | 366 | 2,364 | 6.4590 |
| 2 | 126 | 2,604 | 20.6667 |
| 3 | 124 | 2,606 | 21.0161 |
| 4 | 369 | 2,361 | 6.3984 |
| 5 | 479 | 2,251 | 4.6994 |
| 6 | 1,753 | 977 | 0.5573 |
| 7 | 316 | 2,414 | 7.6392 |
| 8 | 145 | 2,585 | 17.8276 |
| 9 | 3 | 2,727 | 909.0000 |
| 10 | 8 | 2,722 | 340.2500 |
| 11 | 560 | 2,170 | 3.8750 |
| 13 | 28 | 2,702 | 96.5000 |
| 14 | 115 | 2,615 | 22.7391 |
| 15 | 13 | 2,717 | 209.0000 |
| 16 | 47 | 2,683 | 57.0851 |
| 17 | 34 | 2,696 | 79.2941 |
| 18 | 302 | 2,428 | 8.0397 |
| 21 | 534 | 2,196 | 4.1124 |
| 22 | 49 | 2,681 | 54.7143 |
| 24 | 218 | 2,512 | 11.5229 |
| 25 | 2,096 | 634 | 0.3025 |
| 26 | 521 | 2,209 | 4.2399 |
| 28 | 240 | 2,490 | 10.3750 |
| 29 | 2,711 | 19 | 0.0070 |
| 30 | 431 | 2,299 | 5.3341 |
| 31 | 60 | 2,670 | 44.5000 |

#### Method 2: Smoothed Weighted BCE

The raw inverse-frequency ratio used in Method 1 can produce extremely large positive weights. To retain its imbalance-aware behavior while reducing the influence of a few rare examples, we define the raw ratio

$$
r_c = \frac{N_c^-}{N_c^+},
$$

and evaluate three smoothed alternatives:

1. **Square-root weighting**

$$
w_c^{\mathrm{sqrt}} = \sqrt{r_c}.
$$

2. **Logarithmic weighting**

$$
w_c^{\mathrm{log}} = \log(1 + r_c).
$$

3. **Capped inverse-frequency weighting**

$$
w_c^{\mathrm{cap}} = \min(r_c, w_{\max}), \qquad w_{\max}=20.
$$

All three vectors are passed to `BCEWithLogitsLoss` as `pos_weight`. Square-root and logarithmic transformations compress large ratios smoothly, while the capped variant preserves the original ratio up to 20 and clips every larger value.

##### Archived Smoothed Class Weights

| Label ID | Positive | Negative | Square-root weight | Logarithmic weight | Capped weight |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 366 | 2,364 | 2.5415 | 2.0094 | 6.4590 |
| 2 | 126 | 2,604 | 4.5461 | 3.0758 | 20.0000 |
| 3 | 124 | 2,606 | 4.5843 | 3.0918 | 20.0000 |
| 4 | 369 | 2,361 | 2.5295 | 2.0013 | 6.3984 |
| 5 | 479 | 2,251 | 2.1678 | 1.7404 | 4.6994 |
| 6 | 1,753 | 977 | 0.7465 | 0.4430 | 0.5573 |
| 7 | 316 | 2,414 | 2.7639 | 2.1563 | 7.6392 |
| 8 | 145 | 2,585 | 4.2223 | 2.9353 | 17.8276 |
| 9 | 3 | 2,727 | 30.1496 | 6.8134 | 20.0000 |
| 10 | 8 | 2,722 | 18.4459 | 5.8326 | 20.0000 |
| 11 | 560 | 2,170 | 1.9685 | 1.5841 | 3.8750 |
| 13 | 28 | 2,702 | 9.8234 | 4.5799 | 20.0000 |
| 14 | 115 | 2,615 | 4.7686 | 3.1671 | 20.0000 |
| 15 | 13 | 2,717 | 14.4568 | 5.3471 | 20.0000 |
| 16 | 47 | 2,683 | 7.5555 | 4.0619 | 20.0000 |
| 17 | 34 | 2,696 | 8.9047 | 4.3857 | 20.0000 |
| 18 | 302 | 2,428 | 2.8354 | 2.2016 | 8.0397 |
| 21 | 534 | 2,196 | 2.0279 | 1.6317 | 4.1124 |
| 22 | 49 | 2,681 | 7.3969 | 4.0202 | 20.0000 |
| 24 | 218 | 2,512 | 3.3945 | 2.5276 | 11.5229 |
| 25 | 2,096 | 634 | 0.5500 | 0.2643 | 0.3025 |
| 26 | 521 | 2,209 | 2.0591 | 1.6563 | 4.2399 |
| 28 | 240 | 2,490 | 3.2210 | 2.4314 | 10.3750 |
| 29 | 2,711 | 19 | 0.0837 | 0.0070 | 0.0070 |
| 30 | 431 | 2,299 | 2.3096 | 1.8459 | 5.3341 |
| 31 | 60 | 2,670 | 6.6708 | 3.8177 | 20.0000 |

For the rarest type (label *9*), the raw ratio of *909* is reduced to *30.1496* by square-root weighting, *6.8134* by logarithmic weighting, and *20* by capping. This substantially reduces the extreme gradient contribution observed with the aggressive objective.

#### Method 3: Focal Loss

We also implemented binary focal loss,[^5] adapted to the multi-label setting. For every example-label pair, the unreduced BCE loss is first computed. The loss is then multiplied by a focusing factor:

$$
FL(p_t) = (1-p_t)^\gamma BCE(p_t),
$$

where $p_t$ is the predicted probability of the correct binary class and $\gamma \geq 0$ is the focusing parameter. Easy, confidently classified examples receive less weight, allowing training to focus on difficult decisions. When $\gamma=0$, focal loss reduces to ordinary BCE; increasing $\gamma$ suppresses easy examples more strongly.

Our implementation in `paraphrase_detection/focal_loss.py` does not use an additional $\alpha$ class-balancing term, so the experiment isolates the focusing parameter. `bart_detection.py` accepts multiple unique gamma values through repeated `--focal_gamma` arguments and trains a separate model for each value.

## Paraphrase Type Generation (PTG)

_Methodology is to be added._

# Experiments

## Quora Question Pairs (QQP)

### Experimental Setup

All QQP experiments used the provided BERT-based classifier infrastructure.

| Setting | Value |
| --- | --- |
| Task | QQP paraphrase detection |
| Model backbone | BERT, `bert-base-uncased` |
| Training mode | Finetuning |
| Optimizer | Project AdamW implementation |
| Learning rate | $1 \times 10^{-5}$ |
| Batch size | 8 |
| Random seed | 11711 |
| Main metric | Development accuracy |
| Loss | Binary cross-entropy with logits |
| Checkpoint criterion | Highest development accuracy |
| Cluster setting | `--local_files_only`, `--use_gpu` |

The course-reported accuracy of approximately **0.765** is an external reference. The controlled local baseline for our architectural comparisons is the earlier run using the original pair representation $[u,v]$, which reached **0.785** development accuracy.

### Improvement 1: Sentence-Pair Interaction Features

#### Hypothesis

Paraphrase detection depends not only on the individual meanings of both questions, but also on their relation. The original representation $[u, v]$ leaves much of this relation implicit. By adding $\lvert u-v \rvert$ and $u \odot v$, the classifier receives explicit information about dimension-wise difference and dimension-wise overlap between both question embeddings.

We therefore hypothesize that the enriched representation

$$
h_{\mathrm{pair}} = [u, v, \lvert u-v \rvert, u \odot v]
$$

improves paraphrase detection compared to the original pair representation $[u, v]$.

#### Research Question 1

Does adding explicit sentence-pair interaction features improve QQP development accuracy compared to the original BERT-based pair representation $[u, v]$?

### Improvement 2: MLP Classification Head & Dropout

#### Hypothesis

A single linear classifier can only learn a linear decision boundary over the enriched pair representation. Since the representation contains different types of information — the raw embeddings, the absolute-difference features, and the Hadamard-product features — a non-linear classifier may be able to combine these signals more flexibly.

We therefore hypothesize that replacing the linear classification head with an MLP head can improve QQP paraphrase detection when used together with the enriched pair representation.

#### Research Question 2

Does a non-linear MLP classification head improve QQP development accuracy compared to a linear classification head on the same sentence-pair interaction representation?

### Altered Training Configuration

#### Hypothesis

The MLP head may need more training time than the linear classifier because it has more trainable parameters. At the same time, a high dropout value may slow down learning or cause underfitting in the newly added classifier head.

We therefore hypothesize that the same MLP head benefits from weaker dropout and a longer epoch budget, while development-based checkpoint selection prevents selecting a later overfitted model.

#### Research Question 3

Does lowering dropout from `0.3` to `0.1` and increasing the epoch budget to 7, combined with best-checkpoint selection and optional early stopping, improve QQP development accuracy for the MLP head?

## Stanford Sentiment Treebank (SST)

_Experiments are to be added._

## Semantic Textual Similarity (STS)

_Experiments are to be added._

## Paraphrase Type Detection (PTD)

### Data and Split Compliance

| Split | File | Rows | Current status |
| --- | --- | ---: | --- |
| Training source | `data/etpc-paraphrase-train.csv` | 2,730 | Contains the complete original training data; current code removes development IDs in memory before training and class-weight calculation |
| Development | `data/etpc-paraphrase-dev.csv` | 273 | Sampled from the original training data with seed 42; current code excludes matching IDs from the runtime training dataframe |
| Test inputs | `data/etpc-paraphrase-detection-test-student.csv` | Unlabeled | Used only to generate predictions; no test labels are used for training or model selection |

The preliminary results reported below were produced before the runtime overlap filtering was added. They are retained for transparency but do not constitute clean held-out development performance.

### Evaluation Protocol and Experimental Setup

We held the architecture and core training configuration fixed so that the intended independent variable was the loss function:

| Setting | Value |
| --- | --- |
| Model | Course-provided `facebook/bart-large` with a 26-output linear classifier |
| Historical data used for the displayed results | Original 2,730-row training data, including the 273 development rows |
| Current rerun training data | `data/etpc-paraphrase-train.csv` after development IDs are removed in memory |
| Development data | `data/etpc-paraphrase-dev.csv` (273 rows) |
| Epochs | 25 for BCE/Weighted-BCE comparisons and the confirmatory focal runs; 5 for the exploratory focal sweep |
| Batch size | 16 |
| Optimizer | Project `AdamW` implementation |
| Learning rate | $2\times10^{-5}$ |
| Random seed | 11711 |
| Maximum sequence length | 512 tokens |
| Prediction threshold | 0.5 for every label |
| Primary metric | Mean per-label development accuracy |
| Supplementary metric | Mean per-label MCC |
| Checkpoint criterion | Highest development accuracy; an earlier checkpoint wins a tie |

Accuracy remains the primary course metric; MCC and per-label error statistics are supplementary evidence for this imbalanced task.

Before every experiment, the random seed is reset so that compared models start from the same initialization and see the same shuffled training order. After each epoch, the code computes accuracy and MCC, but it saves checkpoints using accuracy only. Each reported comparison row is therefore the MCC of that method's **accuracy-selected** checkpoint, not its best-MCC checkpoint.

### Experiment 1: Aggressive Weighted BCE

We compared unweighted BCE with the aggressive inverse-frequency Weighted BCE for 25 epochs using a batch size of 16. The architecture and core training configuration remained fixed, while the positive term for each label was weighted by $N_c^-/N_c^+$ as described in Method 1. We expected a rare-positive recall/MCC benefit and a possible accuracy/precision cost.

Evaluation followed the shared protocol above: mean per-label development accuracy was primary, mean per-label MCC was supplementary, and the results report each method's accuracy-selected checkpoint.

### Experiment 2: Smoothed Weighted BCE

We trained the three smoothed Weighted BCE variants for 25 epochs with a batch size of 16. Square-root and logarithmic weighting compress the inverse-frequency ratios, while capped weighting uses the predefined $w_{\max}=20$ setting described in Method 2. The unweighted BCE results from Experiment 1 are reused as the common reference; `compare_non_aggressive_weighted` does not rerun that baseline. We expected the variants to retain some minority-label benefit without unstable extreme weights.

Development accuracy and MCC are compared both for the accuracy-selected checkpoints and across selected epochs.

### Experiment 3: Focal Loss

Focal loss replaces fixed class-level weights with the confidence-dependent focusing factor described in Method 3. The 5-epoch exploratory experiment compared 16 focusing parameters, and the 25-epoch confirmatory experiment compared the four selected moderate values $\gamma=0.87$, $1.1$, $1.15$, and $1.25$. We expected focal loss to focus on difficult decisions and potentially improve early convergence and MCC.

The dedicated `focal` mode does not rerun unweighted BCE, so the results reuse the separately obtained deterministic BCE reference. The exploratory values are compared at epoch 5, while the confirmatory table reports fixed epochs 8, 9, and 25 to examine the temporal behavior of the losses.

## Paraphrase Type Generation (PTG)

_Experiments are to be added._

# Results

## Quora Question Pairs (QQP)

### Model Variant Comparison

| Stage | Model variant | Pair representation | Head | Dropout | Epoch budget | Early stopping | Best QQP dev accuracy |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| Course reference/baseline | Default setup with simple task head | $[u, v]$ | Linear | — | — | — | approx. 0.765 |
| Earlier local baseline | Original pair representation | $[u, v]$ | Linear | 0.3 | 3 | No | 0.785 |
| Improvement 1 | Sentence-pair interaction features | $[u, v, \lvert u-v \rvert, u \odot v]$ | Linear | 0.3 | 3 | No | 0.850 |
| Improvement 2 | Same pair features with MLP head | $[u, v, \lvert u-v \rvert, u \odot v]$ | MLP | 0.3 | 3 | No | 0.849 |
| Improvement 2 (altered training) | Same pair features and MLP head with lower dropout and longer training | $[u, v, \lvert u-v \rvert, u \odot v]$ | MLP | 0.1 | 7 | Yes, patience 2 | **0.868** |

The first improvement, sentence-pair interaction features with a linear head, improved development accuracy to **0.850**. This shows that explicitly encoding dimension-wise difference and overlap was beneficial compared to the original $[u, v]$ representation.

The second improvement, replacing the linear head with an MLP head while keeping dropout at `0.3` and training for 3 epochs, reached **0.849**. This did not improve over the simpler linear-head pair-feature model.

The final configuration kept the same pair features and the same MLP head, but reduced dropout to `0.1` and trained for up to 7 epochs with development-based checkpoint selection. This reached the best observed QQP development accuracy of **0.868**.

This comparison does **not** isolate dropout alone because the stronger run also used a longer epoch budget and optional early stopping. It should therefore be interpreted as a combined training-configuration improvement rather than as a complete dropout hyperparameter optimization.

### Best Run: Epoch-Level Results

The strongest model used sentence-pair interaction features, an MLP head, `hidden_dropout_prob = 0.1`, and a maximum of 7 epochs with early stopping enabled.

| Epoch | Train loss | Train accuracy | Dev accuracy | Checkpoint status |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.440 | 0.864 | 0.830 | Saved |
| 2 | 0.305 | 0.927 | 0.857 | Saved |
| 3 | 0.218 | 0.954 | 0.858 | Saved |
| 4 | 0.153 | 0.972 | 0.858 | Saved |
| 5 | 0.110 | 0.986 | 0.865 | Saved |
| 6 | 0.083 | 0.991 | **0.868** | Saved, best checkpoint |
| 7 | 0.066 | 0.993 | 0.866 | Not selected |

The development accuracy peaked at epoch 6. In epoch 7, the training loss continued to decrease from `0.083` to `0.066`, and training accuracy increased from `0.991` to `0.993`, but development accuracy dropped from `0.868` to `0.866`. This indicates mild overfitting after epoch 6. The selected final model is therefore the best development checkpoint rather than the final epoch checkpoint.

### QQP Discussion

#### Research Question 1

**Does adding explicit sentence-pair interaction features improve QQP development accuracy compared to the original BERT-based pair representation $[u, v]$?**

Yes. The pair-feature linear-head model reached **0.850** development accuracy, compared to approximately **0.765** for the course/default reference and **0.785** in our earlier local baseline run with the original $[u, v]$ representation. This supports the hypothesis that explicit difference and overlap features are useful for QQP paraphrase detection.

#### Research Question 2

**Does a non-linear MLP classification head improve QQP development accuracy compared to a linear classification head on the same sentence-pair interaction representation?**

Not by itself. With the same pair representation, dropout `0.3`, and 3 training epochs, the MLP head reached **0.849**, while the linear head reached **0.850**. This suggests that simply adding a more expressive head is not automatically beneficial under the original training configuration.

#### Research Question 3

**Does lowering dropout from `0.3` to `0.1` and increasing the epoch budget to 7, combined with best-checkpoint selection and optional early stopping, improve QQP development accuracy for the MLP head?**

Yes, in our experiment. The MLP model with dropout `0.1` and a 7-epoch budget reached **0.868** development accuracy, improving over both the MLP run with dropout `0.3` and the linear-head pair-feature run. However, because dropout, epoch budget, and early stopping were changed together, this result should be interpreted as a successful combined training configuration, not as a fully isolated proof that `0.1` is the optimal dropout value.

### QQP Limitations

The comparison between the 3-epoch runs and the final 7-epoch run is limited because the shorter runs had not clearly converged. Both the pair-feature linear-head model and the MLP model with dropout `0.3` were still improving at epoch 3. Therefore, the final improvement to **0.868** cannot be attributed only to the lower dropout value or the MLP head. It may also partly result from the longer epoch budget. A stricter comparison would train all model variants with the same maximum epoch budget and the same early-stopping/checkpoint-selection protocol.

The value `hidden_dropout_prob = 0.1` was not selected through a full hyperparameter sweep. We only compared the default value `0.3` with one lower value, `0.1`. Therefore, the final setting should be interpreted as a targeted improvement rather than as an optimized hyperparameter choice.

The final comparison changes multiple factors at once: dropout is reduced, the epoch budget is increased, and early stopping is enabled. This means the improvement cannot be attributed to dropout alone.

All reported QQP experiments use the same random seed, `11711`. We therefore did not measure run-to-run variance across multiple random initializations and data orders.

The development set was used to compare model variants and training configurations. Therefore, the development accuracy should be interpreted as a model-selection metric, not as an independent estimate of final test performance.

We did not perform a complete dropout sweep over values such as `0.01`, `0.05`, `0.15`, or `0.2`. Such a sweep would be necessary to make a stronger claim about the optimal dropout value for this MLP head.

### QQP Conclusion

The results support the usefulness of explicit sentence-pair interaction features for QQP paraphrase detection. The original pair representation $[u, v]$ gives the classifier access to both question embeddings, but leaves the actual comparison mostly implicit. Adding $\lvert u-v \rvert$ and $u \odot v$ makes dimension-wise difference and overlap information directly available to the classifier.

The strongest single architectural improvement was the sentence-pair interaction representation. It improved the development accuracy to **0.850** while still using a linear head. This indicates that the representation itself already provides a much more useful input for the classifier.

The MLP head did not improve performance under the original dropout and epoch settings. The run with dropout `0.3` and 3 epochs reached **0.849**, slightly below the linear-head pair-feature model. This suggests that a more expressive classifier head alone is not sufficient and may require a better-matched training configuration.

The best result was achieved by keeping the MLP head but lowering dropout to `0.1` and training for up to 7 epochs. The training curve shows that the model continued to fit the training data throughout all epochs, while development accuracy peaked at epoch 6. This supports using development-based checkpoint selection.

Early stopping was added mainly as a practical compute-saving mechanism. Since the code already saves the best checkpoint based on development accuracy, early stopping does not fundamentally change which checkpoint is selected if all epochs are completed. Its main purpose is to stop future runs earlier when development accuracy no longer improves, avoiding unnecessary use of HPC (High Performance Cluster) compute budget.

For QQP, we select the model with sentence-pair interaction features, MLP head, `hidden_dropout_prob = 0.1`, and best-checkpoint selection over a 7-epoch budget to be included in our main branch that is to be submitted since it achieved the best development accuracy among our tested QQP variants.

### QQP Hyperparameter Optimization

The value `hidden_dropout_prob = 0.1` was **not** selected by an exhaustive hyperparameter sweep. We compared the original default value `0.3` with one targeted lower value, `0.1`. We did not test additional values such as `0.01`, `0.05`, `0.15`, or `0.2`.

Therefore, this should be interpreted as a targeted training-configuration experiment rather than complete hyperparameter optimization. In addition, the lower-dropout run increased the epoch budget from 3 to 7 and enabled early stopping, so the observed improvement cannot be attributed to dropout alone. A stricter follow-up would compare dropout values under the same epoch budget, early-stopping protocol, and multiple random seeds.

## Stanford Sentiment Treebank (SST)

| Model | Accuracy |
| --- | ---: |
| _To be added_ | — |

## Semantic Textual Similarity (STS)

| Model | Pearson correlation |
| --- | ---: |
| _To be added_ | — |

## Paraphrase Type Detection (PTD)

All values in this **Paraphrase Type Detection (PTD)** section are historical preliminary results because the development examples were still present in the training data when these runs were made. The current code removes that overlap. Metric values are reported to three decimal places, and these preliminary values should not be interpreted as clean held-out performance.

### Model Variant Comparison

| Paraphrase Type Detection (PTD) | Comparison point | Development accuracy | Development MCC |
| --- | --- | ---: | ---: |
| Unweighted BCE | Accuracy-selected checkpoint (25-epoch run) | 1.000 | 0.962 |
| Aggressive Weighted BCE | Accuracy-selected checkpoint (25-epoch run) | 0.979 | 0.896 |
| Square-root Weighted BCE | Accuracy-selected checkpoint (25-epoch run) | 0.999 | 0.956 |
| Logarithmic Weighted BCE | Accuracy-selected checkpoint (25-epoch run) | 0.995 | 0.933 |
| Capped Weighted BCE (cap=20) | Accuracy-selected checkpoint (25-epoch run) | 0.988 | 0.920 |
| Focal $\gamma=0.87$ | Fixed epoch 25 (confirmatory run) | 1.000 | 0.960 |
| Focal $\gamma=1.1$ | Fixed epoch 25 (confirmatory run) | 1.000 | 0.961 |
| Focal $\gamma=1.15$ | Fixed epoch 25 (confirmatory run) | 1.000 | 0.961 |
| Focal $\gamma=1.25$ | Fixed epoch 25 (confirmatory run) | 0.999 | 0.959 |

### Experiment 1 Results: Aggressive Weighted BCE

#### Accuracy-Selected Checkpoints

| Model | Development accuracy | Development MCC |
| --- | ---: | ---: |
| Unweighted BCE | **1.000** | **0.962** |
| Aggressive Weighted BCE | 0.979 | 0.896 |

#### Selected Epochs

| Epoch | Unweighted BCE accuracy | Unweighted BCE MCC | Aggressive Weighted BCE accuracy | Aggressive Weighted BCE MCC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.910 | 0.037 | 0.556 | 0.046 |
| 5 | 0.956 | 0.461 | 0.708 | 0.219 |
| 10 | 0.994 | 0.832 | 0.879 | 0.584 |
| 15 | 0.999 | 0.959 | 0.947 | 0.790 |
| 20 | 1.000 | 0.960 | 0.975 | 0.890 |
| 25 | 0.999 | 0.959 | 0.974 | 0.893 |

Thus, aggressive Weighted BCE did not improve either reported metric. It eventually approached the baseline, but the unweighted objective remained stronger by 0.021 accuracy points and 0.066 MCC points.

#### Aggressive Weighted BCE Discussion

The weights range from 0.007 to 909.000, which is too extreme for naive inverse-frequency reweighting to be stable. For label 9, one positive loss term is multiplied by 909, but this estimate is based on only three positive examples. Labels 10, 15, and 13 are similarly unreliable: 8, 13, and 28 positive examples yield weights of 340.2500, 209.0000, and 96.5000, respectively. The ETPC dataset contains genuinely rare paraphrase types, so estimates and evaluation scores for these labels are inherently noisy.

Weighted BCE also changes the effective decision boundary. If $p$ is the unweighted posterior probability and $q$ is the probability learned under the weighted objective, the optimum satisfies

$$
q = \frac{w p}{w p + (1-p)}.
$$

Using the unchanged prediction rule $q > 0.5$ is therefore equivalent to

$$
p > \frac{1}{1+w}.
$$

For label 9, $w=909$, so an unweighted posterior as small as $1/910 \approx 0.0011$ is sufficient to predict the label as positive. In contrast, label 29 has $w=0.007$, so its posterior must exceed $1/1.007 \approx 0.993$ to be predicted as positive. Thus, raw weighting strongly encourages additional positive predictions for rare labels and strongly suppresses positive predictions for very frequent labels. With a fixed 0.5 threshold, this likely introduces many false positives for rare types; per-label precision and recall would be needed to confirm the error distribution.

For this reason, the weighted objective did not improve on unweighted BCE in our experiment. Its training loss must not be compared numerically with the BCE loss because the positive terms are rescaled; for example, a Weighted BCE loss of 0.0603 is not directly comparable to a BCE loss of 0.0073.

### Experiment 2 Results: Smoothed Weighted BCE

#### Accuracy-Selected Checkpoints

| Model | Development accuracy | Development MCC |
| --- | ---: | ---: |
| Unweighted BCE (reference) | **1.000** | **0.962** |
| Square-Root Weighted BCE | 0.999 | 0.956 |
| Logarithmic Weighted BCE | 0.995 | 0.933 |
| Capped Weighted BCE (cap=20) | 0.988 | 0.920 |

#### Selected Epochs

| Epoch | Baseline accuracy | Square-root accuracy | Logarithmic accuracy | Capped accuracy | Baseline MCC | Square-root MCC | Logarithmic MCC | Capped MCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **0.910** | 0.907 | 0.873 | 0.654 | 0.037 | 0.043 | 0.047 | **0.100** |
| 5 | **0.956** | 0.944 | 0.935 | 0.873 | 0.461 | 0.685 | **0.745** | 0.577 |
| 10 | **0.994** | 0.987 | 0.977 | 0.963 | 0.832 | **0.906** | 0.880 | 0.825 |
| 15 | **0.999** | 0.993 | 0.982 | 0.974 | **0.959** | 0.936 | 0.907 | 0.891 |
| 20 | **1.000** | 0.998 | 0.991 | 0.980 | **0.960** | 0.954 | 0.923 | 0.908 |
| 25 | **0.999** | 0.997 | 0.995 | 0.984 | **0.959** | 0.945 | 0.933 | 0.914 |

Square-root weighting is the strongest smoothed variant, reaching 0.999 development accuracy and 0.956 MCC. It is much closer to the unweighted baseline than the aggressive Weighted BCE run (0.979 accuracy and 0.896 MCC), but it still does not improve on the baseline.

#### Smoothed Weighted BCE Discussion

The results illustrate the expected trade-off of label balancing: compared with unweighted BCE, the smoothed objectives generally sacrifice some accuracy in exchange for higher MCC during the early stages of training. MCC is useful supplementary evidence here because it captures minority-label decisions more informatively than accuracy alone.

During the first 10 epochs, all three smoothing techniques exceed the reused BCE reference in MCC and learn substantially faster than the aggressive objective. With additional training, unweighted BCE catches up and slightly surpasses the smoothed variants at their accuracy-selected checkpoints. The smoothed methods therefore do not improve final development performance over BCE in this preliminary run, although all three outperform aggressive weighting in both reported metrics. These findings must be retested on the corrected non-overlapping split.

### Experiment 3 Results: Focal Loss

#### Exploratory Sweep at Epoch 5

At original precision, $\gamma=1.15$ and $\gamma=1.25$ have the highest development accuracy (0.9577), while $\gamma=0.87$ has the highest MCC (0.5085). Relative to the separately run BCE reference, $\gamma=0.87$ changes accuracy from 0.9560 to 0.9563 and MCC from 0.4610 to 0.5085. Larger values, especially $\gamma=4$, substantially reduce both metrics. The unusually weak $\gamma=0.75$ run is retained rather than discarded; repeated clean-split runs are needed to determine whether it reflects variance or systematic optimization behavior.

| Focal-loss gamma | Development accuracy | Development MCC |
| ---: | ---: | ---: |
| Unweighted BCE reference ($\gamma=0$ equivalence; separate run) | 0.956 | 0.461 |
| 0.25 | 0.951 | 0.420 |
| 0.5 | 0.951 | 0.437 |
| 0.62 | 0.939 | 0.357 |
| 0.75 | 0.911 | 0.038 |
| 0.8 | 0.945 | 0.391 |
| **0.87** | 0.956 | **0.509** |
| 0.9 | 0.950 | 0.452 |
| 0.95 | 0.953 | 0.438 |
| 1 | 0.957 | 0.461 |
| 1.05 | 0.956 | 0.461 |
| **1.1** | **0.958** | 0.467 |
| **1.15** | **0.958** | 0.480 |
| **1.25** | **0.958** | 0.499 |
| 1.5 | 0.954 | 0.449 |
| 2 | 0.946 | 0.413 |
| 4 | 0.923 | 0.183 |

#### Confirmatory Run over 25 Epochs

The raw logs and checkpoints are no longer available. The retained values in `paraphrase_detection/plotter.py` are rounded to three decimals; selected epochs and smaller differences cannot be reconstructed at higher precision. The fixed-epoch values relevant to the temporal claim are:

| Model | Epoch 8 accuracy | Epoch 8 MCC | Epoch 9 accuracy | Epoch 9 MCC | Epoch 25 accuracy | Epoch 25 MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unweighted BCE | 0.987 | 0.732 | 0.989 | 0.744 | 0.999 | 0.959 |
| Focal $\gamma=0.87$ | 0.991 | 0.870 | **0.996** | 0.931 | **1.000** | 0.960 |
| Focal $\gamma=1.1$ | **0.992** | 0.894 | 0.995 | **0.932** | **1.000** | **0.961** |
| Focal $\gamma=1.15$ | **0.992** | **0.900** | 0.993 | 0.917 | **1.000** | **0.961** |
| Focal $\gamma=1.25$ | 0.991 | 0.885 | 0.994 | 0.918 | 0.999 | 0.959 |

At epochs 8 and 9, every focal variant exceeds the BCE reference in both stored metrics. No single gamma consistently dominates: $\gamma=1.15$ has the highest focal MCC at epoch 8, $\gamma=1.1$ has the highest MCC at epoch 9, and $\gamma=0.87$ has the highest epoch-9 accuracy. By epoch 25, the differences have mostly disappeared. Across all 25 stored epochs, BCE and every confirmatory focal run reach a rounded maximum accuracy of 1.000 and a rounded maximum MCC of 0.962. These overlapping-data observations motivate a clean-split convergence study but do not establish generalization.

#### Focal Loss Discussion

Focal loss stays much closer to the unweighted BCE baseline than Weighted BCE because it applies a smooth, confidence-dependent weight to each example-label decision. Weighted BCE assigns a fixed weight to every positive instance of a label, regardless of whether that instance is easy or difficult; for very rare labels, this can strongly alter the optimization trajectory and effective decision boundary. Focal loss instead gradually reduces an example's contribution as the model becomes confident. It is directly connected to the baseline: when $\gamma=0$, it is exactly BCE, while moderate values modify the objective without introducing extreme class-level weights.

The preliminary differences are primarily **temporal**, not evidence of a better final optimum. At a fixed early epoch, focal loss follows a different optimization path and reaches higher scores in the retained curves; with additional training, BCE catches up. Faster convergence would be practically valuable under a fixed compute budget or early stopping. However, because the development examples also occurred in training for these historical runs, the curves may partly measure how quickly each loss memorized repeated examples rather than how quickly it generalized. A clean-split rerun is necessary before attributing the higher early MCC to earlier learning of difficult or minority-label decisions.

Moderate focusing parameters are the most promising candidates for that rerun, while $\gamma=4$ suppresses easy decisions too aggressively in the preliminary sweep. The corrected experiment should predeclare a selection rule and compare epochs-to-target as well as fixed-epoch accuracy and MCC.

### PTD Hyperparameter Optimization

The 16 focal gamma values formed a motivated 5-epoch exploratory sweep of focusing strength. Based on the development accuracy and MCC scores, four moderate values ($\gamma=0.87$, $1.1$, $1.15$, and $1.25$) were selected for the 25-epoch confirmatory run. Moderate values were the most promising, while $\gamma=4$ was too aggressive and substantially reduced both metrics.

For capped Weighted BCE, `cap=20` was the predefined capped-weighting setting; it was not a broad or random parameter search.

Many gamma values were screened and the confirmatory values were selected using the same overlapping development data used for historical reporting. This reuse is a validity limitation, and final claims should use a corrected non-overlapping split with a predeclared selection rule where possible.

### PTD Model Selection

$\gamma=0.87$ is a promising focal-loss candidate because it had the strongest exploratory MCC and the highest retained epoch-9 accuracy. However, no focal setting consistently dominated across epochs or metrics. A final flagship model should be selected only after a clean-split rerun with a predeclared selection criterion.

### PTD Limitations and Threats to Validity

- **Historical train/development overlap:** The displayed results and archived class weights were produced from the unfiltered 2,730-row training data, which still contained the development examples. The current code filters this overlap, so the historical values are not clean held-out estimates and are not expected to be reproduced exactly by current reruns.
- **Single-seed evidence:** All reported values use seed 11711, so run-to-run variance is unknown.
- **Development-set hyperparameter search:** Many gamma values were screened on the same development data used for reporting. Final claims should separate exploratory selection from a predeclared confirmatory comparison where possible.
- **Rare-label uncertainty:** Several labels have fewer than 30 positive training examples. Their weights and per-label metrics are inherently noisy.

## Paraphrase Type Generation (PTG)

| Model | Metric |
| --- | ---: |
| _To be added_ | — |

# Visualizations

## Quora Question Pairs (QQP)

### Train and Development Accuracy

![QQP train/dev accuracy curves](qqp_paraphrase_predict/figures/qqp_train_dev_accuracy_curves.png)

The development accuracy of the strongest run peaked at epoch 6. In epoch 7, training accuracy continued to increase, but development accuracy dropped from `0.868` to `0.866`. This indicates mild overfitting after epoch 6 and supports selecting the best development checkpoint rather than the final epoch checkpoint.

### Training Loss

![QQP training loss curves](qqp_paraphrase_predict/figures/qqp_train_loss_curves.png)

The training curves show that the strongest model continued to fit the training data throughout all epochs, while development accuracy peaked at epoch 6. The pair-feature linear-head model and MLP model with dropout `0.3` were also still improving at the end of their 3-epoch budgets, which limits direct attribution of the final gain to dropout or the MLP head alone.

## Stanford Sentiment Treebank (SST)

_Visualizations are to be added._

## Semantic Textual Similarity (STS)

_Visualizations are to be added._

## Paraphrase Type Detection (PTD)

The PTD plots visualize the same historical preliminary runs with overlapping development data described in the results section.

### Aggressive Weighted BCE

![PTD aggressive Weighted BCE development accuracy](paraphrase_detection/figure/dev_acc.png)

![PTD aggressive Weighted BCE MCC](paraphrase_detection/figure/mcc.png)

These curves show the aggressive objective eventually approaching the baseline, while remaining worse at the reported accuracy-selected checkpoints.

### Smoothed Weighted BCE

![PTD smoothed Weighted BCE development accuracy](paraphrase_detection/figure/dev_acc_soft_weighted.png)

![PTD smoothed Weighted BCE MCC](paraphrase_detection/figure/mcc_soft_weighted.png)

These curves show the smoothed objectives' early MCC advantage and the unweighted BCE baseline catching up and slightly surpassing them with additional training.

### Focal Exploratory Sweep

![PTD focal-loss exploratory sweep](paraphrase_detection/figure/focal_best_performance.png)

The exploratory plot shows the stronger moderate gamma values at epoch 5 and the degradation from the aggressive $\gamma=4$ setting.

### Confirmatory Focal Run

![PTD confirmatory focal development accuracy](paraphrase_detection/figure/dev_acc_focal.png)

![PTD confirmatory focal MCC](paraphrase_detection/figure/mcc_focal.png)

The difference plots show each focal run relative to the separately obtained BCE curve:

![PTD confirmatory focal development-accuracy difference](paraphrase_detection/figure/dev_acc_focal_diff.png)

![PTD confirmatory focal MCC difference](paraphrase_detection/figure/mcc_focal_diff.png)

The confirmatory curves show focal loss ahead at the reported early epochs and the differences mostly disappearing by final convergence at epoch 25.

## Paraphrase Type Generation (PTG)

_Visualizations are to be added._

# Members Contribution

Individual member contributions are _to be added_. This section should identify each member and clearly state their work on implementation, experiments, analysis, documentation, and repository maintenance.

# AI-Usage Card

Artificial Intelligence (AI) aided the development of this project. The project-specific [AI-Usage Card](https://ai-cards.org/) is _to be added here_ before final submission.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html), created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) and [Tolga Ermis](https://github.com/Tollgaermis/) for the 2026 DNLP course at the University of Göttingen.

# References

[^1]: Reimers, N. and Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/). Proceedings of EMNLP-IJCNLP 2019.

[^2]: Conneau, A., Kiela, D., Schwenk, H., Barrault, L., and Bordes, A. (2017). [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://aclanthology.org/D17-1070/). Proceedings of EMNLP 2017.

[^3]: Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html). Journal of Machine Learning Research.

[^4]: Prechelt, L. (1998). [Automatic early stopping using cross validation: quantifying the criteria](https://pubmed.ncbi.nlm.nih.gov/12662814/). Neural Networks.

[^5]: Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

[^6]: Lewis, M. et al. (2020). [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461).

[^7]: PyTorch contributors. [`BCEWithLogitsLoss` documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
