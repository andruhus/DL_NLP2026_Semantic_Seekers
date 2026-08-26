## Paraphrase Type Detection: Imbalance-Aware Loss Functions

### Motivation
Paraphrase type detection is formulated as a multi-label classification problem with 26 output labels. The ETPC training split is strongly imbalanced: some paraphrase types occur in almost every example, whereas others have only a few positive examples. Across the 2,730 training examples, only 11,648 of the 70,980 binary label assignments are positive (16.410%). At the individual-label level, the number of positive examples ranges from 3 to 2,711.

This imbalance makes plain binary cross-entropy (BCE) potentially misleading. Because most label decisions are negative, a model can obtain high accuracy by favoring the majority class while still failing to identify positive examples for rare paraphrase types. This behavior is visible in the first epoch of our baseline: it reaches a development accuracy of 0.910 but an MCC of only 0.037. We therefore investigated two imbalance-aware alternatives: **Weighted BCE** and **Focal Loss**.

The experiments and their hypotheses were:

| Experiment | Change from baseline | Expectation |
| --- | --- | --- |
| Unweighted BCE | None | Strong overall accuracy, but a bias toward majority decisions for rare labels |
| Weighted BCE | Positive term for label $c$ multiplied by $N_c^-/N_c^+$ | Better recognition of rare positive labels and therefore higher MCC, potentially at the cost of accuracy |
| Focal loss | Easy decisions down-weighted by $(1-p_t)^\gamma$ | Greater focus on difficult labels without getting discontinuous |

### Experimental Setup

We held the model and training configuration fixed so that only the loss function changed:

| Setting | Value |
| --- | --- |
| Model | `facebook/bart-large` with a 26-output linear classifier |
| Training data | ETPC paraphrase detection training split (2,730 examples) |
| Development data | ETPC paraphrase detection development split |
| Epochs | 25 for the BCE comparison; 5 for focal-loss sweeps |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | $2\times10^{-5}$ |
| Random seed | 11711 |
| Maximum sequence length | 512 tokens |
| Prediction threshold | 0.5 |
| Checkpoint criterion | Mean development accuracy across labels |

Before every experiment, the random seed is reset so that the models start from the same initialization and see the same shuffled training order. We report both mean per-label accuracy and mean per-label Matthews correlation coefficient (MCC). Accuracy measures the fraction of correct binary decisions, while MCC is especially informative here because it accounts for all four entries of the binary confusion matrix and is less easily inflated by the majority class.

### Baseline: Unweighted BCE

The baseline uses BART-large with a linear classification head that produces one logit for each of the 26 paraphrase types. The two sentences are concatenated with `</s>`, tokenized to a maximum length of 512, and passed through BART. The hidden state of the first token is used by the classifier. Each output is treated as an independent binary decision and optimized with `BCEWithLogitsLoss`. At evaluation time, sigmoid probabilities greater than 0.5 are mapped to positive predictions.

### Improvement 1: Weighted BCE (naive approach)

#### Idea
For each paraphrase type $c$, we computed a positive-class weight using only the training split:

$$
w_c = \frac{N_c^-}{N_c^+},
$$

where $N_c^+$ and $N_c^-$ are the numbers of positive and negative training examples for label $c$. The resulting vector is passed to PyTorch's `BCEWithLogitsLoss` as `pos_weight`, yielding

$$
\mathcal{L}_{i,c} = -w_c y_{i,c}\log\sigma(z_{i,c})
- (1-y_{i,c})\log(1-\sigma(z_{i,c})).
$$

This calculation is implemented in `paraphrase_detection/weighted_bce.py` and is called on `train_labels` in `bart_detection.py`; no development- or test-set labels are used to calculate the weights.

#### Methodology

##### Description
We compared unweighted BCE with the aggressive inverse-frequency Weighted BCE for 25 epochs using a batch size of 16. The comparison is reproduced with:

```sh
sbatch run_bart_detection.sh 25 compare \
    --compare_bce_only \
    --batch_size 16 \
    --use_gpu
```

##### Training-Set Class Weights

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

#### Results

##### The best checkpoint:

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

### Improvement 2: Weighted BCE (smoothed weights approach)

#### Idea

The raw inverse-frequency ratio used in Improvement 1 can produce extremely large positive weights. To retain its imbalance-aware behavior while reducing the influence of a few rare examples, we define the raw ratio

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

We trained the three smoothed Weighted BCE variants for 25 epochs with a batch size of 16. The unweighted BCE results from Improvement 1 are used as the common reference baseline. The experiment can be reproduced with:

```sh
sbatch run_bart_detection.sh 25 compare_non_aggressive_weighted \
    --batch_size 16 \
    --weighted_bce_cap 20 \
    --use_gpu
```

##### Training-Set Class Weights

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

#### Results

##### The best checkpoint:

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

The results illustrate the expected trade-off of label balancing: compared with unweighted BCE, the smoothed objectives generally sacrifice some accuracy in exchange for a higher MCC, especially during the early stages of training. This is desirable for an imbalanced multi-label task because MCC captures more informative minority-label decisions than accuracy alone.

During the first 10 epochs, all three smoothing techniques provide improvements over benchmark in MCC. They also learn substantially faster than the naive objective. 

With additional training, unweighted BCE catches up and slightly surpasses the smoothed variants on the final selected checkpoint. The smoothed methods therefore do not improve over the unweighted baseline in the final development score, but they all remain clearly better than the naive aggressive approach. 

However all 3 techniques surpass naive aggressive approach in both dev_accuracy as well as MCC

### Improvement 3: Focal Loss

#### Idea
We also implemented binary focal loss, adapted to the multi-label setting. For every example-label pair, the unreduced BCE loss is first computed. The loss is then multiplied by a focusing factor:

$$
\operatorname{FL}(p_t) = (1-p_t)^\gamma\operatorname{BCE}(p_t),
$$

where $p_t$ is the predicted probability of the correct binary class and $\gamma \geq 0$ is the focusing parameter. Easy, confidently classified examples receive less weight, allowing training to focus on difficult decisions. When $\gamma=0$, focal loss reduces to ordinary BCE; increasing $\gamma$ suppresses easy examples more strongly.

#### Methodlogy
Our implementation in `paraphrase_detection/focal_loss.py` does not use an additional $\alpha$ class-balancing term, so the experiment isolates the effect of the focusing parameter. `bart_detection.py` accepts multiple unique gamma values through repeated `--focal_gamma` arguments and trains a separate model for every value. The dedicated `focal` mode runs only the requested focal-loss experiments. The default `compare` mode runs unweighted BCE, aggressive Weighted BCE, and one focal-loss model for every requested gamma value, unless `--compare_bce_only` is supplied. The `compare_weighted` mode compares unweighted BCE with all four Weighted BCE variants, while `compare_non_aggressive_weighted` runs only the square-root, logarithmic, and capped variants.

We evaluated two sets of five focusing parameters with 5 epochs and a batch size of 16. The initial sweep was:

```sh
sbatch run_bart_detection.sh 5 focal --batch_size 16 \
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

We then ran a second sweep around the most promising region:

```sh
sbatch run_bart_detection.sh 25 focal --batch_size 16 --focal_gamma 0.87 --focal_gamma 1.1 --focal_gamma 1.15 --focal_gamma 1.25 --use_gpu
```

#### Results

##### First Round (5 epochs)
<img src="paraphrase_detection/figure/focal_best_performance.png" width="700">

As we can see with the focal loss the accuracy is slightly decreased, while we achieve significant improvement over MCC. The highest development accuracy is shared by $\gamma=1.15$ and $\gamma=1.25$ (0.9577), while the highest MCC is obtained with $\gamma=0.87$ (0.5085). Larger values, especially $\gamma=4$, substantially reduce performance. As for $\gamma=0.75$ we believe it's some sort of outlier. Here is the table with all the results after 5 epoch:

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

##### Second Round (25 epochs)

<div style="display: flex; gap: 10px;">
  <img src="paraphrase_detection/figure/dev_acc_focal.png" width="500">
  <img src="paraphrase_detection/figure/mcc_focal.png" width="500">
</div>

To see the difference better, here's the plot to show the differences:
<div style="display: flex; gap: 10px;">
  <img src="paraphrase_detection/figure/dev_acc_focal_diff.png" width="500">
  <img src="paraphrase_detection/figure/mcc_focal_diff.png" width="500">
</div>

We can see that at epoch *8-9* all the focal losses show improvement over the baseline. In terms of dev_accuracy it's neglegable, while in MCC it's quite significant. We can also outline that $\gamma=0.87$ shows the best temporal outperformance in both metrics

#### Discussion

Focal loss stays much closer to the unweighted BCE baseline than Weighted BCE because it applies a *smooth*, *confidence-dependent* weight to each example-label decision. Weighted BCE assigns a fixed weight to every positive instance of a label, regardless of whether that instance is easy or difficult; for very rare labels, this can strongly alter the optimization trajectory and effective decision boundary. Focal loss instead gradually reduces the contribution of examples as the model becomes confident about them. It is also directly connected to the baseline: when $\gamma=0$, it is exactly BCE, while small or moderate values of $\gamma$ modify the baseline objective without introducing extreme class-level weights. This explains why the focal-loss curves generally follow the BCE curve more closely than the Weighted BCE variants do.

The observed improvements are primarily **temporal** rather than improvements in the final attainable solution. Both objectives train the model to recover the same underlying binary labels, so with sufficient training we expect their classification performance to become similar, even though their loss functions and optimization paths are not identical. The important difference is how quickly they reach a useful solution. Around epochs 8--9, all evaluated focal-loss variants temporarily outperform the BCE baseline. The gain in development accuracy is negligible, but the improvement in MCC is much clearer, indicating that focal loss learns difficult and minority-label decisions earlier instead of merely increasing the already-dominant number of correct negative predictions.

This faster convergence is practically valuable when training time or compute is limited, or when early stopping is used. In our experiments, $\gamma=0.87$ provides the strongest temporal improvement across both metrics, showing that a moderate amount of focusing can accelerate learning without moving too far from the stable BCE objective. 


### References for This Extension

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
- Lewis, M. et al. (2020). [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461).
