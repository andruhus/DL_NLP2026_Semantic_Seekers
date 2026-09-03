# ETPC BART Generation: Learning-Rate Scheduler Experiments

## Task, Baseline, and Contribution

ETPC paraphrase generation is formulated as a conditional sequence-to-sequence task. Given `sentence1`, its marked segment location, and the requested paraphrase-type IDs, the model generates `sentence2`. Following the course-provided generation setup, the baseline fine-tunes `facebook/bart-large` with token-level sequence-generation loss and the project's `AdamW` implementation. The repository setup already downloads this checkpoint, and `bart_generation.py` uses the corresponding Hugging Face tokenizer and conditional-generation model. This extension retains the provided model, input representation, objective, optimizer, and decoding procedure; it changes only how the optimizer learning rate evolves during fine-tuning.

A constant learning rate applies the same update scale throughout training, although different stages of BART fine-tuning may benefit from different behavior. Large early updates can disrupt useful pretrained representations, while an unchanged rate late in training can prevent stable refinement around a promising solution. We therefore compare the constant-rate baseline with step, cosine, linear, inverse-square-root, and development-metric-dependent schedules.

The original ETPC training CSV contains all 273 development examples. Before tokenization, the pipeline normalizes ETPC `id` values and removes these overlapping rows, leaving 2,457 training examples and 273 non-overlapping held-out development examples. Every comparison run uses this same cleaned split and initializes a fresh `facebook/bart-large` model with the same random seed.

## Research Questions and Hypotheses

1. **Final held-out performance:** Does a non-constant learning-rate schedule improve checkpoint-selected penalized development BLEU over the constant-rate baseline under the same training budget?
2. **Generation quality and novelty:** Can a schedule improve reference BLEU while controlling input BLEU, rather than increasing the penalized score through only one side of its quality-versus-copying tradeoff?
3. **Training efficiency and stability:** Does warmup, smooth decay, or metric-dependent reduction reach a strong held-out score earlier or avoid damaging updates more effectively than a fixed learning rate?

| Method | Change from baseline | Hypothesis |
| --- | --- | --- |
| Constant learning rate | None: $\alpha_t = \alpha_0$ | A simple and competitive baseline, but potentially too aggressive late in training and insufficiently protective at initialization |
| Step decay | Apply $\alpha_t = \max(\alpha_{min}, \alpha_0\gamma^{\lfloor t/s\rfloor})$, where `s` is configured in epochs | Preserve larger exploratory updates early and permit finer later updates, although performance may be sensitive to abrupt drops and the chosen interval |
| Cosine decay | Smoothly interpolate from $\alpha_0$ to $\alpha_{min}$ over all optimizer updates | Combine substantial early progress with increasingly conservative refinement, improving final held-out BLEU without abrupt rate changes |
| Linear decay | Linearly interpolate from $\alpha_0$ to $\alpha_{min}$ over all optimizer updates | Provide predictable annealing and stable late optimization, but risk reducing the rate too quickly for the available epoch budget |
| Inverse-square-root decay | Optionally warm up linearly to $\alpha_0$, then decay proportionally to $1/\sqrt{t}$ | Protect pretrained parameters from large early updates while retaining a longer high-learning-rate tail than linear or cosine decay |
| Metric-dependent decay | Multiply the current rate by `metric_factor` after penalized development BLEU fails to improve for more than `metric_patience` epochs | Keep the rate high while held-out performance improves and reduce it only on a plateau, adapting decay timing to observed model behavior |

### Data Leakage and the Baseline Result

The previously reported penalized development BLEU of approximately 39 was inflated by train–development leakage. All 273 development IDs were also present in the original training CSV, so the model had been optimized on the same examples used for development evaluation. Consequently, that score was not a valid estimate of performance on unseen data.

After removing every training row whose normalized ETPC `id` occurs in the development set, the training split contains 2,457 examples and the development split remains at 273 genuinely held-out examples. Under this corrected protocol, the constant-learning-rate baseline achieves a penalized development BLEU of approximately 17. The decrease from 39 to 17 should therefore not be interpreted as a model regression: it is the result of eliminating leakage and measuring generalization on a non-overlapping split. All scheduler comparisons use 17—not the leaked score of 39—as the valid baseline.

### Learning schedulers

The plots use the default initial learning rate $\alpha_0=2\times10^{-5}$, minimum learning rate $\alpha_{\min}=0$, five epochs, and batch size 8. With 2,457 cleaned training examples, there are $\lceil2457/8\rceil=308$ optimizer updates per epoch and $T=1540$ updates in total. They can be regenerated with:

```sh
python paraphrase_generation/plotter.py
```

#### Constant Learning Rate

##### Idea

The baseline uses the same learning rate for every optimizer update $t$:

$$
\alpha_t=\alpha_0.
$$

It provides no warmup or decay, making it the control condition for determining whether changing the learning rate over time improves generation.

![Constant learning-rate schedule](paraphrase_generation/figure/constant_learning_rate.png)

##### Methodology

Run the five-epoch constant-rate baseline with the default learning rate and batch size:

```sh
sbatch run_bart_generation.sh 5 constant \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --use_gpu
```

##### Results

##### Discussion

#### Step Decay

##### Idea

Step decay multiplies the learning rate by $\gamma$ after every $s$ optimizer updates, subject to a lower bound:

$$
\alpha_t=\max\left(\alpha_{\min},\alpha_0\gamma^{\left\lfloor t/s\right\rfloor}\right).
$$

For the default experiment, $\gamma=0.5$ and the decay interval is one epoch, so $s=308$. The rate is therefore halved at the end of each epoch.

![Step-decay learning-rate schedule](paraphrase_generation/figure/step_decay.png)

##### Methodology

Run step decay for five epochs, reducing the rate by half after each epoch:

```sh
sbatch run_bart_generation.sh 5 step \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --min_lr 0 \
    --step_decay_epochs 1 \
    --step_gamma 0.5 \
    --use_gpu
```

##### Results

##### Discussion

#### Cosine Decay

##### Idea

Cosine decay changes the learning rate smoothly from $\alpha_0$ to $\alpha_{\min}$ over the complete budget of $T$ optimizer updates:

$$
\alpha_t=\alpha_{\min}+\frac{\alpha_0-\alpha_{\min}}{2}
\left(1+\cos\left(\pi\frac{\min(t,T)}{T}\right)\right).
$$

The gradual early decrease preserves relatively large updates for exploration, while the flatter end of the cosine curve permits conservative refinement near the end of training.

![Cosine-decay learning-rate schedule](paraphrase_generation/figure/cosine_decay.png)

##### Methodology

Run cosine decay over the full five-epoch training budget:

```sh
sbatch run_bart_generation.sh 5 cosine \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --min_lr 0 \
    --use_gpu
```

##### Results

##### Discussion

#### Linear Decay

##### Idea

Linear decay decreases the learning rate by the same amount at every optimizer update until it reaches $\alpha_{\min}$ at update $T$:

$$
\alpha_t=\alpha_{\min}+(\alpha_0-\alpha_{\min})
\left(1-\frac{\min(t,T)}{T}\right).
$$

Unlike step decay, this schedule has no abrupt changes; unlike cosine decay, it assigns a constant rate of decrease throughout training.

![Linear-decay learning-rate schedule](paraphrase_generation/figure/linear_decay.png)

##### Methodology

Run linear decay over the full five-epoch training budget:

```sh
sbatch run_bart_generation.sh 5 linear \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --min_lr 0 \
    --use_gpu
```

##### Results

##### Discussion

#### Inverse-Square-Root Decay

##### Idea

Let $u=t+1$ be the one-based optimizer-update number and $w$ the number of warmup updates. With warmup enabled, the implemented schedule is

$$
\alpha_t=\max\left(
\alpha_{\min},
\alpha_0\min\left(\frac{u}{w},\sqrt{\frac{w}{u}}\right)

\right).
$$

The rate increases linearly during the first $w$ updates, reaches the peak $\alpha_0$ at $u=w$, and then decreases proportionally to $1/\sqrt{u}$. The default $w=100$ corresponds to approximately 0.325 training epochs.

![Inverse-square-root learning-rate schedule](paraphrase_generation/figure/inverse_square_root.png)

##### Methodology

Run inverse-square-root decay with 100 linear-warmup updates:

```sh
sbatch run_bart_generation.sh 5 inverse_sqrt \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --min_lr 0 \
    --warmup_steps 100 \
    --use_gpu
```

##### Results

##### Discussion

#### Metric-Dependent Decay

##### Idea

The metric-dependent scheduler changes the learning rate only after development evaluation. Let $k$ index evaluations, $b_k$ be the number of consecutive evaluations without a sufficient improvement in penalized development BLEU, $p$ be the patience, and $f\in(0,1)$ be the reduction factor. The update is

$$
\alpha_{k+1}=
\begin{cases}
\max(\alpha_{\min},f\alpha_k), & b_k>p,\\
\alpha_k, & b_k\le p.
\end{cases}
$$

An improvement resets the bad-epoch count to zero, and a reduction also starts a new patience window. With the defaults $f=0.5$ and $p=1$, the rate is halved after two consecutive non-improving evaluations. Because this schedule depends on observed development scores, its figure uses an illustrative trajectory: improvement after epochs 1 and 2 followed by two non-improving evaluations, causing the lower rate to be used in epoch 5.

![Metric-dependent learning-rate schedule](paraphrase_generation/figure/metric_dependent.png)

##### Methodology

Run metric-dependent decay with factor 0.5 and patience 1:

```sh
sbatch run_bart_generation.sh 5 metric \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --min_lr 0 \
    --metric_factor 0.5 \
    --metric_patience 1 \
    --use_gpu
```

##### Results

##### Discussion