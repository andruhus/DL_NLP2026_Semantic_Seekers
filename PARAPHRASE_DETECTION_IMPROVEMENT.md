# ETPC Paraphrase-Type Detection: Imbalance-Aware Loss Functions

## Task, Baseline, and Contribution

ETPC paraphrase-type detection is formulated as a multi-label classification problem with 26 output labels. Following the course-provided task setup, the baseline fine-tunes `facebook/bart-large` with a 26-output linear classification head and binary cross-entropy (BCE). The use of BART and Hugging Face's tokenizer/model classes is the explicit task-specific exception described by the starter repository: `STRUCTURE.md` specifies `facebook/bart-large` as the starting point, and `setup_gwdg.sh` downloads that checkpoint. This extension retains that provided model and changes only its training objective; it does not introduce another pretrained model or external embedding.

The ETPC labels are strongly imbalanced. In the current 2,730-row training file, only 11,648 of the 70,980 binary label assignments are positive (16.410%). Individual-label positive counts range from 3 to 2,711. Plain BCE can therefore obtain high accuracy by favoring negative decisions while learning little about rare positive labels. We investigate whether inverse-frequency Weighted BCE, smoothed Weighted BCE, or focal loss can improve minority-label behavior and/or reach a useful solution faster than BCE.

## Research Questions and Hypotheses

1. **Final held-out performance:** Does an imbalance-aware objective improve the course's primary ETPC-detection metric, development accuracy, over the unweighted BCE baseline under the same training and checkpoint-selection protocol?
2. **Minority-label behavior:** Does it improve mean per-label MCC and per-label precision/recall, especially for rare paraphrase types, without an unacceptable reduction in accuracy?
3. **Training efficiency:** Under the same fixed epoch budget, does focal loss or smoothed Weighted BCE reach a strong held-out score earlier than ordinary BCE?

| Method | Change from baseline | Hypothesis |
| --- | --- | --- |
| Unweighted BCE | None | Strong aggregate accuracy, but majority-negative bias for rare labels |
| Aggressive Weighted BCE | Positive term for label $c$ multiplied by $N_c^-/N_c^+$ | Better rare-positive recall and MCC, potentially at the cost of accuracy and precision |
| Smoothed Weighted BCE | Compress or cap $N_c^-/N_c^+$ | Retain some minority-label benefit while avoiding unstable extreme weights |
| Focal loss | Down-weight easy decisions by $(1-p_t)^\gamma$ | Focus on difficult decisions without applying fixed class-level weights, improving early convergence and MCC |

A valid improvement claim requires a non-overlapping held-out split and a like-for-like comparison against BCE. Accuracy remains the primary course metric; MCC and per-label error statistics are supplementary evidence for this imbalanced task.

## Data and Split Compliance

| Split | Current file | Rows | Current status |
| --- | --- | ---: | --- |
| Training | `data/etpc-paraphrase-train.csv` | 2,730 | Contains the complete original training data |
| Development | `data/etpc-paraphrase-dev.csv` | 273 | Sampled from the training data with seed 42, but its rows are still present in the training file |
| Test inputs | `data/etpc-paraphrase-detection-test-student.csv` | Unlabeled | Used only to generate predictions; no test labels are used for training or model selection |

## Evaluation Protocol and Experimental Setup

We held the architecture and core training configuration fixed so that the intended independent variable was the loss function:

| Setting | Value |
| --- | --- |
| Model | Course-provided `facebook/bart-large` with a 26-output linear classifier |
| Current training data | `data/etpc-paraphrase-train.csv` (2,730 rows; currently overlaps development data) |
| Current development data | `data/etpc-paraphrase-dev.csv` (273 rows; currently contained in the training data) |
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

Before every experiment, the random seed is reset so that compared models start from the same initialization and see the same shuffled training order. After each epoch, the code computes accuracy and MCC, but it saves checkpoints using accuracy only. Each reported comparison row is therefore the MCC of that method's **accuracy-selected** checkpoint, not its best-MCC checkpoint. In a multi-experiment run, `bart_detection.py` currently generates test predictions from the final experiment in command order rather than selecting the strongest method across experiments.

## Baseline: Unweighted BCE

The baseline concatenates the two sentences with `</s>`, tokenizes to a maximum length of 512, and passes the result through BART-large. The classifier uses the hidden state of the first token to produce one logit for each of the 26 paraphrase types. Every output is treated as an independent binary decision and optimized with `BCEWithLogitsLoss`; at evaluation time, sigmoid probabilities greater than 0.5 are mapped to positive predictions.

## Proposed Loss Functions

### Method 1: Aggressive Weighted BCE

#### Idea
For each paraphrase type $c$, we computed a positive-class weight from the current training file:

$$
w_c = \frac{N_c^-}{N_c^+},
$$

where $N_c^+$ and $N_c^-$ are the numbers of positive and negative training examples for label $c$. The resulting vector is passed to PyTorch's `BCEWithLogitsLoss` as `pos_weight`, yielding

$$
\mathcal{L}_{i,c} = -w_c y_{i,c}\log\sigma(z_{i,c}) - (1-y_{i,c})\log(1-\sigma(z_{i,c})).
$$

This calculation is implemented in `paraphrase_detection/weighted_bce.py` and is called on `train_labels` in `bart_detection.py`;

#### Methodology

##### Description
We compared unweighted BCE with the aggressive inverse-frequency Weighted BCE for 25 epochs using a batch size of 16. The comparison is reproduced with:

```sh
sbatch run_bart_detection.sh 25 compare \
    --compare_bce_only \
    --batch_size 16 \
    --use_gpu
```

This command trains unweighted BCE followed by aggressive Weighted BCE. Although each method reloads its own accuracy-selected checkpoint for reporting, the script currently writes test predictions from the final method in command order (aggressive Weighted BCE), not from the stronger method.

##### Training-Set Class Weights

These preliminary counts and weights come from the overlapping 2,730-row training file and must be regenerated after correcting the split.

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

#### Preliminary Results (overlapping development data)

##### Accuracy-selected checkpoints

| Model | Development accuracy | Development MCC |
| --- | ---: | ---: |
| Unweighted BCE | **1.000** | **0.962** |
| Aggressive Weighted BCE | 0.979 | 0.896 |

##### Epoch display

| Epoch | Unweighted BCE accuracy | Unweighted BCE MCC | Aggressive Weighted BCE accuracy | Aggressive Weighted BCE MCC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.910 | 0.037 | 0.556 | 0.046 |
| 5 | 0.956 | 0.461 | 0.708 | 0.219 |
| 10 | 0.994 | 0.832 | 0.879 | 0.584 |
| 15 | 0.999 | 0.959 | 0.947 | 0.790 |
| 20 | 1.000 | 0.960 | 0.975 | 0.890 |
| 25 | 0.999 | 0.959 | 0.974 | 0.893 |

<img src="paraphrase_detection/figure/dev_acc.png" width="700">

<img src="paraphrase_detection/figure/mcc.png" width="700">


Thus, aggressive Weighted BCE did not improve either reported metric. It eventually approached the baseline, but the unweighted objective remained stronger by 0.021 accuracy points and 0.066 MCC points.

#### Discussion
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

### Method 2: Smoothed Weighted BCE

#### Idea

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

#### Methodology

We trained the three smoothed Weighted BCE variants for 25 epochs with a batch size of 16. The unweighted BCE results from Method 1 are reused as the common reference; `compare_non_aggressive_weighted` does not rerun that baseline. The experiment can be reproduced with:

```sh
sbatch run_bart_detection.sh 25 compare_non_aggressive_weighted \
    --batch_size 16 \
    --weighted_bce_cap 20 \
    --use_gpu
```

This command trains square-root, logarithmic, and capped weighting in that order. The current script consequently writes test predictions from the capped variant, even when another variant has the better development result.

##### Training-Set Class Weights

These preliminary counts and weights come from the overlapping 2,730-row training file and must be regenerated after correcting the split.

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

#### Preliminary Results (overlapping development data)

##### Accuracy-selected checkpoints

| Model | Development accuracy | Development MCC |
| --- | ---: | ---: |
| Unweighted BCE (reference) | **1.0000** | **0.9615** |
| Square-Root Weighted BCE | 0.9994 | 0.9555 |
| Logarithmic Weighted BCE | 0.9946 | 0.9327 |
| Capped Weighted BCE (cap=20) | 0.9875 | 0.9199 |

##### Epoch display

| Epoch | Baseline accuracy | Square-root accuracy | Logarithmic accuracy | Capped accuracy | Baseline MCC | Square-root MCC | Logarithmic MCC | Capped MCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **0.910** | 0.907 | 0.873 | 0.654 | 0.037 | 0.043 | 0.047 | **0.100** |
| 5 | **0.956** | 0.944 | 0.935 | 0.873 | 0.461 | 0.685 | **0.745** | 0.577 |
| 10 | **0.994** | 0.987 | 0.977 | 0.963 | 0.832 | **0.906** | 0.880 | 0.825 |
| 15 | **0.999** | 0.993 | 0.982 | 0.974 | **0.959** | 0.936 | 0.907 | 0.891 |
| 20 | **1.000** | 0.998 | 0.991 | 0.980 | **0.960** | 0.954 | 0.923 | 0.908 |
| 25 | **0.999** | 0.997 | 0.995 | 0.984 | **0.959** | 0.945 | 0.933 | 0.914 |

<img src="paraphrase_detection/figure/dev_acc_soft_weighted.png" width="700">

<img src="paraphrase_detection/figure/mcc_soft_weighted.png" width="700">

Square-root weighting is the strongest smoothed variant, reaching 0.9994 development accuracy and 0.9555 MCC. It is much closer to the unweighted baseline than the aggressive Weighted BCE run (0.9786 accuracy and 0.8959 MCC), but it still does not improve on the baseline.

#### Discussion

The results illustrate the expected trade-off of label balancing: compared with unweighted BCE, the smoothed objectives generally sacrifice some accuracy in exchange for higher MCC during the early stages of training. MCC is useful supplementary evidence here because it captures minority-label decisions more informatively than accuracy alone.

During the first 10 epochs, all three smoothing techniques exceed the reused BCE reference in MCC and learn substantially faster than the aggressive objective. With additional training, unweighted BCE catches up and slightly surpasses the smoothed variants at their accuracy-selected checkpoints. The smoothed methods therefore do not improve final development performance over BCE in this run, although all three outperform aggressive weighting in both reported metrics. These findings must be retested on the corrected non-overlapping split.

### Method 3: Focal Loss

#### Idea
We also implemented binary focal loss, adapted to the multi-label setting. For every example-label pair, the unreduced BCE loss is first computed. The loss is then multiplied by a focusing factor:

$$
FL(p_t) = (1-p_t)^\gamma BCE(p_t),
$$

where $p_t$ is the predicted probability of the correct binary class and $\gamma \geq 0$ is the focusing parameter. Easy, confidently classified examples receive less weight, allowing training to focus on difficult decisions. When $\gamma=0$, focal loss reduces to ordinary BCE; increasing $\gamma$ suppresses easy examples more strongly.

#### Methodology
Our implementation in `paraphrase_detection/focal_loss.py` does not use an additional $\alpha$ class-balancing term, so the experiment isolates the focusing parameter. `bart_detection.py` accepts multiple unique gamma values through repeated `--focal_gamma` arguments and trains a separate model for each value. The dedicated `focal` mode runs only those focal-loss experiments; it does **not** rerun unweighted BCE, so the tables below reuse the separately obtained deterministic BCE reference. The `compare` mode can run unweighted BCE, aggressive Weighted BCE, and requested focal variants, while `compare_non_aggressive_weighted` runs only the square-root, logarithmic, and capped variants.

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

Because experiment order currently controls test prediction, this command would write predictions from $\gamma=1.25$ rather than automatically selecting the best focal variant.

#### Preliminary Results (overlapping development data)

##### Exploratory sweep at epoch 5
<img src="paraphrase_detection/figure/focal_best_performance.png" width="700">

At epoch 5, $\gamma=1.15$ and $\gamma=1.25$ have the highest development accuracy (0.9577), while $\gamma=0.87$ has the highest MCC (0.5085). Relative to the separately run BCE reference, $\gamma=0.87$ changes accuracy from 0.9560 to 0.9563 and MCC from 0.4610 to 0.5085. Larger values, especially $\gamma=4$, substantially reduce both metrics. The unusually weak $\gamma=0.75$ run is retained rather than discarded; repeated clean-split runs are needed to determine whether it reflects variance or systematic optimization behavior.

| Focal-loss gamma | Development accuracy | Development MCC |
| ---: | ---: | ---: |
| 0 (baseline) | 0.9560 | 0.4610 |
| 0.25 | 0.9505 | 0.4201 |
| 0.5 | 0.9507 | 0.4372 |
| 0.62 | 0.9391 | 0.3571 |
| 0.75 | 0.9107 | 0.0380 |
| 0.8 | 0.9445 | 0.3913 |
| **0.87** | 0.9563 | **0.5085** |
| 0.9 | 0.9498 | 0.4516 |
| 0.95 | 0.9528 | 0.4382 |
| 1 | 0.9567 | 0.4606 |
| 1.05 | 0.9560 | 0.4605 |
| **1.1** | 0.9576 | 0.4671 |
| **1.15** | **0.9577** | 0.4800 |
| **1.25** | **0.9577** | 0.4992 |
| 1.5 | 0.9542 | 0.4486 |
| 2 | 0.9462 | 0.4130 |
| 4 | 0.9232 | 0.1830 |

##### Confirmatory run over 25 epochs

<div style="display: flex; gap: 10px;">
  <img src="paraphrase_detection/figure/dev_acc_focal.png" width="500">
  <img src="paraphrase_detection/figure/mcc_focal.png" width="500">
</div>

The difference plots show each focal run relative to the separately obtained BCE curve:

<div style="display: flex; gap: 10px;">
  <img src="paraphrase_detection/figure/dev_acc_focal_diff.png" width="500">
  <img src="paraphrase_detection/figure/mcc_focal_diff.png" width="500">
</div>

The raw logs and checkpoints are no longer available. The retained values in `paraphrase_detection/plotter.py` are rounded to three decimals; selected epochs and smaller differences cannot be reconstructed at higher precision. The fixed-epoch values relevant to the temporal claim are:

| Model | Epoch 8 accuracy | Epoch 8 MCC | Epoch 9 accuracy | Epoch 9 MCC | Epoch 25 accuracy | Epoch 25 MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unweighted BCE | 0.987 | 0.732 | 0.989 | 0.744 | 0.999 | 0.959 |
| Focal $\gamma=0.87$ | 0.991 | 0.870 | **0.996** | 0.931 | **1.000** | 0.960 |
| Focal $\gamma=1.1$ | **0.992** | 0.894 | 0.995 | **0.932** | **1.000** | **0.961** |
| Focal $\gamma=1.15$ | **0.992** | **0.900** | 0.993 | 0.917 | **1.000** | **0.961** |
| Focal $\gamma=1.25$ | 0.991 | 0.885 | 0.994 | 0.918 | 0.999 | 0.959 |

At epochs 8 and 9, every focal variant exceeds the BCE reference in both stored metrics. No single gamma consistently dominates: $\gamma=1.15$ has the highest focal MCC at epoch 8, $\gamma=1.1$ has the highest MCC at epoch 9, and $\gamma=0.87$ has the highest epoch-9 accuracy. By epoch 25, the differences have mostly disappeared. Across all 25 stored epochs, BCE and every confirmatory focal run reach a rounded maximum accuracy of 1.000 and a rounded maximum MCC of 0.962. These overlapping-data observations motivate a clean-split convergence study but do not establish generalization.

#### Discussion

Focal loss stays much closer to the unweighted BCE baseline than Weighted BCE because it applies a smooth, confidence-dependent weight to each example-label decision. Weighted BCE assigns a fixed weight to every positive instance of a label, regardless of whether that instance is easy or difficult; for very rare labels, this can strongly alter the optimization trajectory and effective decision boundary. Focal loss instead gradually reduces an example's contribution as the model becomes confident. It is directly connected to the baseline: when $\gamma=0$, it is exactly BCE, while moderate values modify the objective without introducing extreme class-level weights.

The preliminary differences are primarily **temporal**, not evidence of a better final optimum. At a fixed early epoch, focal loss follows a different optimization path and reaches higher scores in the retained curves; with additional training, BCE catches up. Faster convergence would be practically valuable under a fixed compute budget or early stopping. However, because the current development examples also occur in training, the curves may partly measure how quickly each loss memorizes repeated examples rather than how quickly it generalizes. A clean-split rerun is necessary before attributing the higher early MCC to earlier learning of difficult or minority-label decisions.

Moderate focusing parameters are the most promising candidates for that rerun, while $\gamma=4$ suppresses easy decisions too aggressively in the preliminary sweep. The corrected experiment should predeclare a selection rule and compare epochs-to-target as well as fixed-epoch accuracy and MCC.


## Limitations and Threats to Validity

- **Single-seed evidence:** All reported values use seed 11711, so run-to-run variance is unknown.
- **Development-set hyperparameter search:** Many gamma values were screened on the same development data used for reporting. Final claims should separate exploratory selection from a predeclared confirmatory comparison where possible.
- **Rare-label uncertainty:** Several labels have fewer than 30 positive training examples. Their weights and per-label metrics are inherently noisy.


## References

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
- Lewis, M. et al. (2020). [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461).
- PyTorch contributors. [`BCEWithLogitsLoss` documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
