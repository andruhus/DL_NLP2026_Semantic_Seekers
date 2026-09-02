# ETPC BART Generation: Learning-Rate Scheduler Experiments

## Goal

These experiments compare learning-rate schedules while keeping the ETPC generation model, cleaned train/dev split, optimizer, seed, data order, and checkpoint criterion fixed. The baseline and every scheduler start from `facebook/bart-large` with the same random seed.

The training split is filtered by normalized ETPC `id` before tokenization. This removes the 273 development examples that are also present in the original training CSV, leaving 2,457 training rows and 273 held-out development rows.

## Implemented schedules

| Mode | Learning rate at update `t` | Update frequency |
| --- | --- | --- |
| `constant` | $\alpha_t = \alpha_0$ | Constant baseline |
| `step` | $\alpha_t = \max(\alpha_{min}, \alpha_0\gamma^{\lfloor t/s\rfloor})$ | Every optimizer update; decay interval `s` is configured in epochs |
| `cosine` | Cosine interpolation from $\alpha_0$ to $\alpha_{min}$ | Every optimizer update |
| `linear` | Linear interpolation from $\alpha_0$ to $\alpha_{min}$ | Every optimizer update |
| `inverse_sqrt` | Linear warmup to $\alpha_0$, then $1/\sqrt{t}$ decay | Every optimizer update |
| `metric` | Multiply by `metric_factor` after penalized BLEU fails to improve for more than `metric_patience` epochs | Once after each development evaluation |

The custom `AdamW` receives the scheduler through `lr_sched` and uses `alpha = lr_sched.lr()` for each parameter update. Batch-based schedulers advance immediately after `optimizer.step()`. The metric scheduler advances after penalized development BLEU is available.

## Experiment protocol

- Default initial learning rate: `2e-5`
- Default epochs: `5`
- Default batch size: `8`
- Random seed: `11711`
- Checkpoint criterion: highest penalized development BLEU
- BLEU protocol: displayed reference/input statistics use conventional prediction-to-reference scoring; penalized BLEU preserves the legacy reversed scoring direction
- Tie handling: the earlier checkpoint wins
- `compare` order: constant, step, cosine, linear, inverse square root, metric dependent
- Each run resets the seed and initializes a fresh BART model
- Test predictions are generated with the experiment having the highest checkpoint-selected development BLEU

Every experiment writes a distinct checkpoint under `models/`. The comparison table records the parameterized scheduler name plus reference BLEU, input BLEU, and penalized BLEU. It is written to:

```text
predictions/bart/bart-generation-lr-scheduler-comparison.csv
```

## Running experiments

Run all schedules on SLURM:

```sh
sbatch run_bart_generation.sh 5 compare --use_gpu
```

Run one scheduler:

```sh
sbatch run_bart_generation.sh 10 cosine \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --min_lr 1e-7 \
    --use_gpu
```

Step decay every two epochs with a factor of 0.5:

```sh
sbatch run_bart_generation.sh 10 step \
    --step_decay_epochs 2 \
    --step_gamma 0.5 \
    --use_gpu
```

### Step-decay grid search

Scheduler hyperparameters accept one or more space-separated values. The script runs the Cartesian product of values relevant to the chosen scheduler. For example, this command runs four StepDecay experiments: `(1, 0.3)`, `(1, 0.5)`, `(2, 0.3)`, and `(2, 0.5)`.

```sh
sbatch run_bart_generation.sh 10 step \
    --learning_rate 1e-5 2e-5 \
    --min_lr 0 1e-7 \
    --step_decay_epochs 1 2 \
    --step_gamma 0.3 0.5 \
    --use_gpu
```

This example actually creates 16 runs because learning rate and minimum learning rate are also varied. To vary only the two StepDecay-specific parameters, omit the learning-rate options:

```sh
sbatch run_bart_generation.sh 10 step \
    --step_decay_epochs 1 2 \
    --step_gamma 0.3 0.5 \
    --use_gpu
```

Inverse-square-root decay with 100 warmup updates:

```sh
sbatch run_bart_generation.sh 10 inverse_sqrt \
    --warmup_steps 100 \
    --use_gpu
```

Metric-dependent decay after two tolerated non-improving epochs:

```sh
sbatch run_bart_generation.sh 10 metric \
    --metric_factor 0.5 \
    --metric_patience 2 \
    --min_lr 1e-7 \
    --use_gpu
```

The Python entry point can also be invoked directly:

```sh
python bart_generation.py \
    --epochs 5 \
    --scheduler_mode compare \
    --batch_size 8 \
    --use_gpu
```

## CLI options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--scheduler_mode` | `compare` | One mode or all modes |
| `--epochs` | `5` | Epoch budget per experiment |
| `--batch_size` | `8` | Training/evaluation batch size |
| `--learning_rate` | `2e-5` | One or more initial/peak learning rates |
| `--min_lr` | `0.0` | One or more learning-rate floors |
| `--step_decay_epochs` | `1` | One or more intervals between step decays |
| `--step_gamma` | `0.5` | One or more StepDecay multipliers |
| `--warmup_steps` | `100` | One or more inverse-square-root warmup lengths; use `0` to disable warmup |
| `--metric_factor` | `0.5` | One or more metric-dependent reduction multipliers |
| `--metric_patience` | `1` | One or more tolerated bad-epoch counts before reduction |

## Interpreting results

Use the constant run as the baseline. Compare checkpoint-selected penalized BLEU rather than training loss alone. Because scheduler behavior depends on the epoch budget and number of optimizer updates, keep epochs, batch size, and data fixed in a scheduler comparison. A scheduler that performs best at five epochs may not remain best at a larger budget.
