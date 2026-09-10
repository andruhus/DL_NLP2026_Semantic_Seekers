# Semantic Seekers

- **Group name:** Semantic Seekers
- **Group code:** **TODO: add group code**
- **Group repository:** **TODO: add group repository URL**
- **Tutor responsible:** **TODO: add tutor responsible**
- **Group team leader:** **TODO: add group team leader**
- **Group members:** **TODO: add group member names**

**ETPC BART Generation: Learning-Rate Scheduler Experiments**

# Setup instructions

## Environment setup

From the repository root, set up the local environment with the existing script and activate the `dnlp` conda environment:

```sh
source setup.sh
conda activate dnlp
```

On GWDG, use the GWDG-specific setup script before activating the same environment:

```sh
source setup_gwdg.sh
conda activate dnlp
```

## Experiment outputs and model IDs

For each experiment, we assign an ID to a model and save the parameters in [`paraphrase_generation/run_5epoch_results.csv`](../paraphrase_generation/run_5epoch_results.csv). We save the information about their BLEU scores during training in [`paraphrase_generation/train_5epoch_results.csv`](../paraphrase_generation/train_5epoch_results.csv).

## Running commands

### Constant Learning Rate

Run the parameter grid recorded below: **13 experiments**, five epochs each. Space-separated option values form a Cartesian product.

```sh
sbatch run_bart_generation.sh 5 constant \
    --batch_size 8 \
    --learning_rate 1e-3 1e-4 2e-4 5e-4 1e-5 5e-5 2e-5 3e-5 4e-5 7e-5 9e-5 1.1e-4 1.25e-4 \
    --min_lr 0 \
    --use_gpu
```

### Step Decay

Run the parameter grid recorded below: **27 experiments**, five epochs each. Space-separated option values form a Cartesian product.

```sh
sbatch run_bart_generation.sh 5 step \
    --batch_size 8 \
    --learning_rate 1e-3 1e-4 1e-5 \
    --min_lr 1e-7 \
    --step_decay_epochs 1 2 3 \
    --step_gamma 0.5 0.2 0.1 \
    --use_gpu
```

### Cosine Decay

Run the parameter grid recorded below: **12 experiments**, five epochs each. Space-separated option values form a Cartesian product.

```sh
sbatch run_bart_generation.sh 5 cosine \
    --batch_size 8 \
    --learning_rate 1e-5 2e-5 5e-5 1e-4 \
    --min_lr 1e-6 2e-6 5e-6 \
    --use_gpu
```

### Linear Decay

Run the parameter grid recorded below: **12 experiments**, five epochs each. Space-separated option values form a Cartesian product.

```sh
sbatch run_bart_generation.sh 5 linear \
    --batch_size 8 \
    --learning_rate 1e-5 2e-5 5e-5 1e-4 \
    --min_lr 1e-6 2e-6 5e-6 \
    --use_gpu
```

### Inverse-Square-Root Decay

Run the parameter grid recorded below: **12 experiments**, five epochs each. Space-separated option values form a Cartesian product.

```sh
sbatch run_bart_generation.sh 5 inverse_sqrt \
    --batch_size 8 \
    --learning_rate 2e-5 5e-5 1e-4 \
    --min_lr 2e-6 5e-6 \
    --warmup_steps 0 2 \
    --use_gpu
```

### Metric-Dependent Decay

Run the parameter grid recorded below: **24 experiments**, five epochs each. Space-separated option values form a Cartesian product.

```sh
sbatch run_bart_generation.sh 5 metric \
    --batch_size 8 \
    --learning_rate 2e-5 9e-5 \
    --min_lr 0 \
    --metric_factor 0.1 0.2 0.5 \
    --metric_patience 0 1 \
    --metric_threshold 3 4 \
    --use_gpu
```

# Methodology

## Task description

ETPC paraphrase generation is formulated as a conditional sequence-to-sequence task. Given `sentence1`, its marked segment location, and the requested paraphrase-type IDs, the model generates `sentence2`. Following the course-provided generation setup, the baseline fine-tunes `facebook/bart-large` with token-level sequence-generation loss and the project's `AdamW` implementation. The repository setup already downloads this checkpoint, and `bart_generation.py` uses the corresponding Hugging Face tokenizer and conditional-generation model. This extension retains the provided model, input representation, objective, optimizer, and decoding procedure; it changes only how the optimizer learning rate evolves during fine-tuning.

### BLEU-Score

To measure the quality of our generation, we use **BLEU-Score**.

For a source input $x$ (`sentence1`), the reference $r$, and the suggestion $h$, we calculate:

$$
B_{\mathrm{ref}} = BLEU(r, h), \qquad
B_{\mathrm{input}} = BLEU(x, h).
$$

$B_{\mathrm{ref}}$ shows us how similar the suggestion is to the reference, and $B_{\mathrm{input}}$ how similar it is to the input.

To balance these metrics, we calculate:

$$
B_{\mathrm{pen}} = \frac{B_{\mathrm{ref}}\bigl(100 - B_{\mathrm{input}}\bigr)}{52}.
$$

SacreBLEU reports this corpus-level score on a $0$--$100$ scale. A larger `reference_bleu` indicates stronger lexical agreement with the target paraphrase.

The factor $1/52$ is a project-specific scaling constant; it changes the magnitude of the score, not the ranking when the other terms are fixed.

In our experiments we'll track the values for $B_{\mathrm{ref}}$ and $B_{\mathrm{pen}}$.

### Data Leakage and the Baseline Result

In Part 1, we made a mistake when we previously reported `penalized_bleu` in `etpc_dev_dataset` to be approximately 39, which was inflated by train–development leakage. All 273 development IDs were also present in the original training CSV, so the model had been optimized on the same examples used for development evaluation. Consequently, that score was not a valid estimate of performance on unseen data.

After removing every training row whose normalized ETPC `id` occurs in the development set, the training split contains 2,457 examples and the development split remains at 273 genuinely held-out examples. Under this corrected protocol, the constant-learning-rate baseline achieves a development `penalized_bleu` of approximately 17. The decrease from 39 to 17 should therefore not be interpreted as a model regression: it is the result of eliminating leakage and measuring generalization on a non-overlapping split. All scheduler comparisons use 17—not the leaked score of 39—as the valid baseline.

As for `reference_bleu`, we've managed to achieve **46.22**, which is close to the expected 47.5.

## Improvement idea

What if we instead used an adaptive learning rate scheduler? This way, we could better converge our loss function and get better results.

### Learning rate scheduler types

| Method                    | Change from baseline                                                                                                                                   | Hypothesis                                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constant learning rate    | None: $\alpha_t = \alpha_0$                                                                                                                            | A simple and competitive baseline, but potentially too aggressive late in training and insufficiently protective at initialization                      |
| Step decay                | Apply $\alpha_t = \max(\alpha_{min}, \alpha_0\gamma^{\lfloor t/s\rfloor})$, where `s` is configured in epochs                                          | Preserve larger exploratory updates early and permit finer later updates, although performance may be sensitive to abrupt drops and the chosen interval |
| Cosine decay              | Smoothly interpolate from $\alpha_0$ to $\alpha_{min}$ over all optimizer updates                                                                      | Combine substantial early progress with increasingly conservative refinement, improving final held-out BLEU without abrupt rate changes                 |
| Linear decay              | Linearly interpolate from $\alpha_0$ to $\alpha_{min}$ over all optimizer updates                                                                      | Provide predictable annealing and stable late optimization, but risk reducing the rate too quickly for the available epoch budget                       |
| Inverse-square-root decay | Optionally warm up linearly to $\alpha_0$, then decay proportionally to $1/\sqrt{t}$                                                                   | Protect pretrained parameters from large early updates while retaining a longer high-learning-rate tail than linear or cosine decay                     |
| Metric-dependent decay    | Multiply the current rate by `metric_factor` after `penalized_bleu` for the `etpc_dev_dataset` fails to improve for more than `metric_patience` epochs | Keep the rate high while held-out performance improves and reduce it only on a plateau, adapting decay timing to observed model behavior                |

### Scope of interest

In these experiments, we won't change the other parameters, such as `batch_size`, `loss_fn`, or `n_epochs`, to stay close to the baseline.

However, we need to mention that different `lr` values might benefit from changing the `batch_size`. Popel and Bojar (2018, Section 4.8) discuss the interaction between learning rate, effective batch size, and the learning-rate schedule in Transformer training. This motivates jointly tuning these parameters in future experiments; our results therefore compare learning rates only under the fixed baseline configuration. (Popel & Bojar, 2018)

# Experiments

## Experimental setup and evaluation

The original ETPC training CSV contains all 273 development examples. Before tokenization, the pipeline normalizes ETPC `id` values and removes these overlapping rows, leaving 2,457 training examples and 273 non-overlapping held-out development examples. Every comparison run uses this same cleaned split and initializes a fresh `facebook/bart-large` model with the same random seed.

| Setting | Fixed experimental value |
| --- | --- |
| Number of runs | 100 different experiments |
| Batch size | `8` |
| Epochs | `5` |
| Approximate runtime | ~12 minutes per run on a GPU |
| Optimizer | The project's `AdamW` implementation |
| Primary metrics | `reference_bleu` ($B_{\mathrm{ref}}$) and `penalized_bleu` ($B_{\mathrm{pen}}$) |

All runs use the corrected held-out development split for evaluation. `input_bleu` is also reported because it is a component of `penalized_bleu`.

## Scheduler experiments

The changes and expectations for each scheduler are summarized in the [Learning rate scheduler types](#learning-rate-scheduler-types) table. From `paraphrase_generation/run_5epoch_results.csv` and `paraphrase_generation/train_5epoch_results.csv`, we generated the following results plots. The complete tables below are detailed/raw results and retain the original four-decimal BLEU reporting.

### Constant Learning Rate

Source: [run_5epoch_results.csv](../paraphrase_generation/run_5epoch_results.csv), filtered to `Constant learning rate`. Sorted by descending `penalized_bleu`. BLEU scores are rounded to four decimal places.

|     ID |       LR | `reference_bleu` | `input_bleu` | `penalized_bleu` |
| -----: | -------: | ---------------: | -----------: | ---------------: |
| **74** |   _9e-5_ |        _42.6154_ |    _73.2415_ |        _21.9293_ |
| **65** |   _1e-4_ |        _42.1565_ |    _73.1281_ |        _21.7851_ |
|     75 |   1.1e-4 |          42.8679 |      74.3334 |          21.1591 |
|     76 |  1.25e-4 |          44.3238 |      75.5372 |          20.8516 |
|     73 |     7e-5 |          44.4819 |      76.9212 |          19.7421 |
| **70** | **2e-5** |      **46.2204** |  **79.9772** |      **17.7973** |
|     69 |     5e-5 |          45.8548 |      80.5357 |          17.1641 |
|     72 |     4e-5 |          45.1790 |      80.6597 |          16.8034 |
|     71 |     3e-5 |          46.5307 |      81.7639 |          16.3180 |
|     68 |     1e-5 |          48.1365 |      88.2866 |          10.8431 |
|     66 |     2e-4 |           0.7036 |       1.0939 |           1.3382 |
|     67 |     5e-4 |           0.0082 |       0.0079 |           0.0158 |
|     64 |     1e-3 |           0.0042 |       0.0040 |           0.0081 |

**Training dynamics.** IDs 74 and 65 are compared with the ID 70 baseline.

![Constant learning-rate selected runs compared with baseline](../paraphrase_generation/figure/constant_training_comparison.png)

#### Observations

Both candidates have pretty similar `lr` and as we can observe we might get good `penalized_bleu` values for the first 4 epochs, but then the loss for the ID 65 starts growing for the 5th epoch, meaning that we took a large `lr`

However, the baseline still possesses a higher `reference_bleu`.

### Step Decay

Source: [run_5epoch_results.csv](../paraphrase_generation/run_5epoch_results.csv), filtered to `Step decay`. Sorted by descending `penalized_bleu`. BLEU scores are rounded to four decimal places.

|     ID |     LR | Min LR | Decay interval (epochs) | Step size (updates) | Gamma | `reference_bleu` | `input_bleu` | `penalized_bleu` |
| -----: | -----: | -----: | ----------------------: | ------------------: | ----: | ---------------: | -----------: | ---------------: |
| **16** | _1e-4_ | _1e-7_ |                     _3_ |               _924_ | _0.5_ |        _42.1565_ |    _73.1281_ |        _21.7851_ |
| **17** | _1e-4_ | _1e-7_ |                     _3_ |               _924_ | _0.2_ |        _42.1565_ |    _73.1281_ |        _21.7851_ |
|     18 | 1e-4 |   1e-7 |                       3 |                 924 |   0.1 |          42.1565 |      73.1281 |          21.7851 |
|     13 |   1e-4 |   1e-7 |                       2 |                 616 |   0.5 |          44.5942 |      76.5706 |          20.0926 |
|     10 | 1e-4 |   1e-7 |                       1 |                 308 |   0.5 |          46.1225 |      79.6345 |          18.0636 |
|     14 | 1e-4 |   1e-7 |                       2 |                 616 |   0.2 |          46.7597 |      81.2418 |          16.8678 |
|     15 | 1e-4 |   1e-7 |                       2 |                 616 |   0.1 |          46.5604 |      82.0296 |          16.0905 |
|     11 | 1e-4 |   1e-7 |                       1 |                 308 |   0.2 |          47.3100 |      84.5849 |          14.0248 |
|     12 | 1e-4 |   1e-7 |                       1 |                 308 |   0.1 |          47.8621 |      86.2523 |          12.6537 |
|     25 | 1e-5 |   1e-7 |                       3 |                 924 |   0.5 |          48.4788 |      87.1102 |          12.0170 |
|     22 | 1e-5 |   1e-7 |                       2 |                 616 |   0.5 |          48.6384 |      90.0102 |           9.3440 |
|     26 | 1e-5 |   1e-7 |                       3 |                 924 |   0.2 |          48.7459 |      90.1498 |           9.2338 |
|     27 | 1e-5 |   1e-7 |                       3 |                 924 |   0.1 |          48.7433 |      91.7289 |           7.7531 |
|     23 | 1e-5 |   1e-7 |                       2 |                 616 |   0.2 |          48.7032 |      93.7172 |           5.8845 |
|     24 | 1e-5 |   1e-7 |                       2 |                 616 |   0.1 |          48.7245 |      94.1514 |           5.4802 |
|     19 | 1e-5 |   1e-7 |                       1 |                 308 |   0.5 |          48.8444 |      94.2596 |           5.3921 |
|     20 | 1e-5 |   1e-7 |                       1 |                 308 |   0.2 |          48.7207 |      96.0419 |           3.7084 |
|     21 | 1e-5 |   1e-7 |                       1 |                 308 |   0.1 |          48.9112 |      96.4586 |           3.3311 |
|      1 | 1e-3 |   1e-7 |                       1 |                 308 |   0.5 |           0.0042 |       0.0040 |           0.0081 |
|      2 | 1e-3 |   1e-7 |                       1 |                 308 |   0.2 |           0.0042 |       0.0040 |           0.0081 |
|      3 | 1e-3 |   1e-7 |                       1 |                 308 |   0.1 |           0.0042 |       0.0040 |           0.0081 |
|      4 | 1e-3 |   1e-7 |                       2 |                 616 |   0.5 |           0.0042 |       0.0040 |           0.0081 |
|      5 | 1e-3 |   1e-7 |                       2 |                 616 |   0.2 |           0.0042 |       0.0040 |           0.0081 |
|      6 | 1e-3 |   1e-7 |                       2 |                 616 |   0.1 |           0.0042 |       0.0040 |           0.0081 |
|      7 | 1e-3 |   1e-7 |                       3 |                 924 |   0.5 |           0.0042 |       0.0040 |           0.0081 |
|      8 | 1e-3 |   1e-7 |                       3 |                 924 |   0.2 |           0.0042 |       0.0040 |           0.0081 |
|      9 | 1e-3 |   1e-7 |                       3 |                 924 |   0.1 |           0.0042 |       0.0040 |           0.0081 |

**Training dynamics.** IDs 16 and 17 are compared with the ID 70 baseline.

![Step-decay selected runs compared with baseline](../paraphrase_generation/figure/step_training_comparison.png)

#### Observations

IDs 16, 17, and 18 report the same top `penalized_bleu` despite different gamma values at the three-epoch decay interval.

### Cosine Decay

Source: [run_5epoch_results.csv](../paraphrase_generation/run_5epoch_results.csv), filtered to `Cosine decay`. Sorted by descending `penalized_bleu` (ties by ascending ID). BLEU scores are rounded to four decimal places.

|     ID |     LR | Min LR | Total updates | `reference_bleu` | `input_bleu` | `penalized_bleu` |
| -----: | -----: | -----: | ------------: | ---------------: | -----------: | ---------------: |
| **46** | _5e-5_ | _1e-6_ |        _1540_ |        _46.1692_ |    _79.0771_ |        _18.5768_ |
| **51** | _1e-4_ | _5e-6_ |        _1540_ |        _45.8081_ |    _79.7121_ |        _17.8721_ |
|     48 |   5e-5 |   5e-6 |          1540 |          46.5291 |      80.5299 |          17.4216 |
|     47 |   5e-5 |   2e-6 |          1540 |          46.6614 |      81.0484 |          17.0059 |
|     49 |   1e-4 |   1e-6 |          1540 |          46.7160 |      83.2764 |          15.0243 |
|     50 |   1e-4 |   2e-6 |          1540 |          46.9698 |      83.5854 |          14.8267 |
|     45 |   2e-5 |   5e-6 |          1540 |          47.9610 |      89.3548 |           9.8184 |
|     42 |   1e-5 |   5e-6 |          1540 |          48.4059 |      89.8842 |           9.4166 |
|     44 |   2e-5 |   2e-6 |          1540 |          48.0509 |      90.8375 |           8.4667 |
|     43 |   2e-5 |   1e-6 |          1540 |          48.6154 |      90.9816 |           8.4314 |
|     41 |   1e-5 |   2e-6 |          1540 |          48.5021 |      91.2907 |           8.1234 |
|     40 |   1e-5 |   1e-6 |          1540 |          48.6739 |      92.6041 |           6.9228 |

**Training dynamics.** IDs 46 and 51 are compared with the ID 70 baseline.

![Cosine-decay selected runs compared with baseline](../paraphrase_generation/figure/cosine_training_comparison.png)

#### Observations

The first models that somewhat outperformed the baseline. However, we can notice that the approximate average learning rate looks to be `2e-05` and the performance resembles it greatly.

### Linear Decay

Source: [run_5epoch_results.csv](../paraphrase_generation/run_5epoch_results.csv), filtered to `Linear decay`. Sorted by descending `penalized_bleu` (ties by ascending ID). BLEU scores are rounded to four decimal places.

|     ID |     LR | Min LR | Total updates | `reference_bleu` | `input_bleu` | `penalized_bleu` |
| -----: | -----: | -----: | ------------: | ---------------: | -----------: | ---------------: |
| **38** | _1e-4_ | _2e-6_ |        _1540_ |        _34.8555_ |    _64.0708_ |        _24.0833_ |
|     39 |   1e-4 |   5e-6 |          1540 |          46.2525 |      79.3179 |          18.3962 |
|     34 |   5e-5 |   1e-6 |          1540 |          46.4726 |      81.7165 |          16.3400 |
|     37 |   1e-4 |   1e-6 |          1540 |          46.3186 |      81.8649 |          16.1537 |
| **33** | _2e-5_ | _5e-6_ |        _1540_ |        _47.6035_ |    _82.7402_ |        _15.8005_ |
|     35 |   5e-5 |   2e-6 |          1540 |          46.6066 |      84.8960 |          13.5374 |
|     32 |   2e-5 |   2e-6 |          1540 |          47.4143 |      85.3911 |          13.3206 |
|     30 |   1e-5 |   5e-6 |          1540 |          48.2923 |      87.5470 |          11.5651 |
|     36 |   5e-5 |   5e-6 |          1540 |          47.6811 |      87.6650 |          11.3105 |
|     31 |   2e-5 |   1e-6 |          1540 |          48.1090 |      88.4786 |          10.6593 |
|     28 |   1e-5 |   1e-6 |          1540 |          48.8003 |      91.3644 |           8.1042 |
|     29 |   1e-5 |   2e-6 |          1540 |          48.4397 |      91.3266 |           8.0796 |

**Training dynamics.** IDs 38 and 33 are compared with the ID 70 baseline.

![Linear-decay selected runs compared with baseline](../paraphrase_generation/figure/linear_training_comparison.png)

#### Observations

1. Model 38 achieved its peak `penalized_bleu` after the 2nd epoch because of the lowest `input_bleu` and despite the abysmal `reference_bleu`. We consider this to be an outlier.
2. Model 33 outperformed the baseline in `reference_bleu`, despite being a little worse in `input_bleu`.

### Inverse-Square-Root Decay

Source: [run_5epoch_results.csv](../paraphrase_generation/run_5epoch_results.csv), filtered to `Inverse square root`. Sorted by descending `penalized_bleu` (ties by ascending ID). BLEU scores are rounded to four decimal places.

|     ID |     LR | Min LR | Total updates | Warmup updates | `reference_bleu` | `input_bleu` | `penalized_bleu` |
| -----: | -----: | -----: | ------------: | -------------: | ---------------: | -----------: | ---------------: |
| **54** | _2e-5_ | _5e-6_ |        _1540_ |            _0_ |        _39.4848_ |    _75.2960_ |        _18.7583_ |
| **63** | _1e-4_ | _5e-6_ |        _1540_ |            _2_ |        _48.4320_ |    _92.5608_ |         _6.9287_ |
|     61 |   1e-4 |   2e-6 |          1540 |              2 |          48.6123 |      92.7174 |           6.8082 |
|     58 |   5e-5 |   5e-6 |          1540 |              0 |          48.4654 |      93.2922 |           6.2518 |
|     62 |   1e-4 |   5e-6 |          1540 |              0 |          48.6915 |      93.4709 |           6.1137 |
|     60 |   1e-4 |   2e-6 |          1540 |              0 |          48.5754 |      94.5559 |           5.0855 |
|     57 |   5e-5 |   2e-6 |          1540 |              2 |          48.6221 |      94.5981 |           5.0510 |
|     59 |   5e-5 |   5e-6 |          1540 |              2 |          48.1402 |      94.6291 |           4.9722 |
|     55 |   2e-5 |   5e-6 |          1540 |              2 |          48.3614 |      95.5488 |           4.1397 |
|     56 |   5e-5 |   2e-6 |          1540 |              0 |          48.6748 |      95.9534 |           3.7878 |
|     52 |   2e-5 |   2e-6 |          1540 |              0 |          48.7384 |      97.0412 |           2.7732 |
|     53 |   2e-5 |   2e-6 |          1540 |              2 |          48.9237 |      97.8553 |           2.0178 |

**Training dynamics.** IDs 54 and 63 are compared with the ID 70 baseline.

![Inverse-square-root selected runs compared with baseline](../paraphrase_generation/figure/inverse_sqrt_training_comparison.png)

#### Observations

Inverse-square-root schedulers decreased extremely fast, which made them analogous to low constant-rate schedulers.

### Metric-Dependent Decay

Source: [run_5epoch_results.csv](../paraphrase_generation/run_5epoch_results.csv), filtered to `Metric dependent`. Sorted by descending `penalized_bleu` (ties by ascending ID). BLEU scores are rounded to four decimal places.

|     ID |     LR | Min LR | Factor (CSV gamma) | Patience | Threshold | `reference_bleu` | `input_bleu` | `penalized_bleu` |
| -----: | -----: | -----: | -----------------: | -------: | --------: | ---------------: | -----------: | ---------------: |
| **91** | _9e-5_ |    _0_ |              _0.1_ |      _1_ |       _3_ |        _42.6154_ |    _73.2415_ |        _21.9293_ |
|     95 |   9e-5 |      0 |                0.2 |        1 |         3 |          42.6154 |      73.2415 |          21.9293 |
|     99 |   9e-5 |      0 |                0.5 |        1 |         3 |          42.6154 |      73.2415 |          21.9293 |
| **97** | _9e-5_ |    _0_ |              _0.5_ |      _0_ |       _3_ |        _44.9549_ |    _76.4521_ |        _20.3576_ |
|     98 |   9e-5 |      0 |                0.5 |        0 |         4 |          44.9549 |      76.4521 |          20.3576 |
|    100 |   9e-5 |      0 |                0.5 |        1 |         4 |          44.3306 |      76.1777 |          20.3088 |
|     96 |   9e-5 |      0 |                0.2 |        1 |         4 |          44.8140 |      77.1246 |          19.7142 |
|     92 |   9e-5 |      0 |                0.1 |        1 |         4 |          45.2885 |      77.6968 |          19.4246 |
|     79 |   2e-5 |      0 |                0.1 |        1 |         3 |          46.2204 |      79.9772 |          17.7973 |
|     80 |   2e-5 |      0 |                0.1 |        1 |         4 |          46.2204 |      79.9772 |          17.7973 |
|     83 |   2e-5 |      0 |                0.2 |        1 |         3 |          46.2204 |      79.9772 |          17.7973 |
|     84 |   2e-5 |      0 |                0.2 |        1 |         4 |          46.2204 |      79.9772 |          17.7973 |
|     87 |   2e-5 |      0 |                0.5 |        1 |         3 |          46.2204 |      79.9772 |          17.7973 |
|     88 |   2e-5 |      0 |                0.5 |        1 |         4 |          46.2204 |      79.9772 |          17.7973 |
|     93 |   9e-5 |      0 |                0.2 |        0 |         3 |          45.6294 |      80.8587 |          16.7963 |
|     94 |   9e-5 |      0 |                0.2 |        0 |         4 |          45.6294 |      80.8587 |          16.7963 |
|     85 |   2e-5 |      0 |                0.5 |        0 |         3 |          46.2492 |      81.6392 |          16.3302 |
|     89 |   9e-5 |      0 |                0.1 |        0 |         3 |          44.8785 |      82.0905 |          15.4568 |
|     90 |   9e-5 |      0 |                0.1 |        0 |         4 |          44.8785 |      82.0905 |          15.4568 |
|     81 |   2e-5 |      0 |                0.2 |        0 |         3 |          47.2718 |      83.8035 |          14.7238 |
|     77 |   2e-5 |      0 |                0.1 |        0 |         3 |          47.9883 |      86.8995 |          12.0898 |
|     86 |   2e-5 |      0 |                0.5 |        0 |         4 |          47.9820 |      88.3039 |          10.7923 |
|     82 |   2e-5 |      0 |                0.2 |        0 |         4 |          48.3951 |      90.1885 |           9.1313 |
|     78 |   2e-5 |      0 |                0.1 |        0 |         4 |          48.6880 |      91.7971 |           7.6805 |

**Training dynamics.** IDs 91 and 97 are compared with the ID 70 baseline.

![Metric-dependent selected runs compared with baseline](../paraphrase_generation/figure/metric_training_comparison.png)

#### Observations

1. Models 95 and 99 have identical results because the threshold and patience did not let the `lr` change significantly.
2. Some models (such as 79 or 80) performed exactly like the baseline because of long patience and an apparently not-so-large threshold.
3. The metrics show the `reference_bleu` vs. `penalized_bleu` trade-off that we currently have.

## Results

The compact Paraphrase Type Generation summary below reports representative selected runs with exactly three decimal places. The rationale column distinguishes the valid baseline, scheduler representatives, and the deliberately selected flagship.

| Paraphrase Type Generation (PTG) selection | `reference_bleu` | `input_bleu` | `penalized_bleu` | Selection rationale |
| --- | ---: | ---: | ---: | --- |
| Baseline constant (ID 70) | 46.220 | 79.977 | 17.797 | Valid leakage-corrected baseline |
| Tuned constant (ID 74) | 42.615 | 73.242 | 21.929 | Highest `penalized_bleu` in the constant-rate grid |
| Step decay (ID 16) | 42.157 | 73.128 | 21.785 | Representative of the tied top step-decay runs |
| Cosine decay (ID 46) | 46.169 | 79.077 | 18.577 | Highest `penalized_bleu` in the cosine-decay grid |
| Linear decay / flagship (ID 33) | 47.604 | 82.740 | 15.801 | Selected flagship rather than outlier ID 38 |
| Inverse-square-root decay (ID 54) | 39.485 | 75.296 | 18.758 | Highest `penalized_bleu` in the inverse-square-root grid |
| Metric-dependent decay (ID 91) | 42.615 | 73.242 | 21.929 | Representative of the tied top metric-dependent runs |

### Flagship model

Basically, there are only 2 models that were closest to outperforming the baseline:

|           ID |       LR |  Min LR | `reference_bleu` | `input_bleu` | `penalized_bleu` |
| -----------: | -------: | ------: | ---------------: | -----------: | ---------------: |
| **Baseline** | **2e-5** | **0.0** |      **46.2204** |  **79.9772** |      **17.7973** |
|       **46** |   _5e-5_ |  _1e-6_ |        _46.1692_ |    _79.0771_ |        _18.5768_ |
|       **33** |   _2e-5_ |  _5e-6_ |        _47.6035_ |    _82.7402_ |        _15.8005_ |

**However, we decided that Model 33 would be our flagship**, because it improves `reference_bleu` significantly while not increasing `input_bleu` that much.

### Hyperparameter Optimization

This is a motivated Cartesian grid of 100 runs across the six scheduler types and their parameters, with the changes and hypotheses recorded in the methodology table. Space-separated option values in each command form a Cartesian product, while non-scheduler settings remain fixed for controlled comparisons.

As discussed in the [Scope of interest](#scope-of-interest), `batch_size` was not jointly tuned because the experiments stay close to the baseline and compare learning rates under the fixed baseline configuration. Different `lr` values might benefit from changing the `batch_size`, so jointly tuning learning rate, effective batch size, and the learning-rate schedule remains future work.

### Overall discussion

The question arises: why did we fail? The obvious reason is that we didn't have enough epochs to schedule meaningfully. Had we had 50 epochs, this could have had a greater impact. However, this doesn't explain the whole picture. For example, why do we get significantly worse `reference_bleu` after improving `penalized_bleu`?

#### Output Exploration

Let's try reading into the outputs. The following example was taken from `etpc_dev_dataset` (entry 9).

**Input:**

> Peterson was arrested near Torrey Pines Golf Course in La Jolla on April 18, the day DNA testing identified the bodies.

**Reference:**

> Peterson, 30, was arrested in La Jolla April 18 after the two bodies were identified through DNA tests.

**Annotations:** [ETPC type definitions](https://github.com/venelink/ETPC/blob/master/Corpus/paraphrase_types.xml)

- `6`: contextual same-polarity substitution — equivalent wording in context.
- `11`: synthetic/analytic substitution — a compact expression versus a more explicit construction.
- `14`: diathesis alternation — changing grammatical voice or argument structure.
- `25`: addition/deletion — adding or removing material.
- `29`: identity — retaining wording unchanged.

**Model baseline:** (added word highlighted)

> Peterson was arrested near Torrey Pines Golf Course in La Jolla on April 18, the **same** day DNA testing identified the bodies.

**Model 33:** (it outperformed the baseline a bit)

> Peterson was arrested near Torrey Pines Golf Course in La Jolla on April 18, the **same** day DNA testing identified the bodies.

We can observe the following:

1. Both outputs closely copy the source: they add only “same” to the input.
2. The reference includes information absent from the source, such as Peterson’s age (“30”). That detail cannot be inferred from the supplied input alone. This explains why data leakage had a profound effect on both `reference_bleu` and `penalized_bleu`.

#### Conflicting metrics

Because the reference contains information not present in the input, the model does not have a better strategy than just copying the input.

We are in the pitfall where increasing `reference_bleu` means basically copying the input. Slightly deviating from the input sentence decreases `reference_bleu` slightly, but doubles or triples the `1- input_bleu` component, thereby increasing `penalized_bleu`.

A related limitation is discussed by Jin et al. (2022), who note that, in text style transfer, “simply copying the input can result in high BLEU scores.” This supports the general concern that BLEU can reward copying, although it does not establish the specific changes in `penalized_bleu` described here. (Jin et al., 2022)

## Visualizations

Scheduler-specific training plots are shown with each scheduler experiment above without duplicating them here.

### Reference/Penalized BLEU vs Loss

Let's investigate how `reference_bleu` and `penalized_bleu` depend on the loss.

Here are the checkpoint values for different epochs. I've excluded some failed experiments with extremely low BLEU scores (~87% of the data left).
![Text 2](../paraphrase_generation/figure/loss_bleu_scatter_87.png)

If we further filter for `reference_bleu <= 46.5` and `penalized_bleu <= 10` (successful checkpoints; ~17% of the data left), we get the following:
![Text 1](../paraphrase_generation/figure/loss_bleu_scatter.png)

We can observe that a decrease in the loss correlates with better `penalized_bleu`, while `reference_bleu` has no correlation, or even a slight negative one.

## Members Contribution

**TODO: Replace the placeholders below with every group member's name and a clear description of their contribution.**

| Group member | Contribution |
| --- | --- |
| **TODO: member name** | **TODO: describe this member's contributions** |

# AI-Usage Card

Artificial Intelligence (AI) aided the restructuring of this report.

**TODO: Replace or supplement this notice with a link to the completed project AI-Usage Card. Use [AI Usage Cards](https://ai-cards.org/) as the template resource; no local card is claimed here.**

# References

Jin, D., Jin, Z., Hu, Z., Vechtomova, O., & Mihalcea, R. (2022). Deep learning for text style transfer: A survey. Computational Linguistics, 48(1), 155–205. https://doi.org/10.1162/coli_a_00426

Popel, M., & Bojar, O. (2018). Training tips for the Transformer model. arXiv:1804.00247. https://arxiv.org/abs/1804.00247
