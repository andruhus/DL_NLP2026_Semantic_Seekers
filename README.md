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

On the GWDG cluster, the corresponding setup script is:

```sh
source setup_gwdg.sh
conda activate dnlp
```

The GWDG setup also downloads `bert-base-uncased` and `facebook/bart-large` before compute jobs are started. This is necessary for runs that use `--local_files_only`. For a local setup, make sure that `bert-base-uncased` is already available in the Hugging Face cache before using this option, or omit `--local_files_only` for the first network-enabled run.

The QQP train, development, and test data are included at:

```text
data/quora-paraphrase-train.csv
data/quora-paraphrase-dev.csv
data/quora-paraphrase-test-student.csv
```

All commands below should be executed from the repository root. Add `--use_gpu` when a CUDA-capable GPU is available; the reported experiments were run with `--use_gpu` on the GWDG cluster.

The model is evaluated after every epoch. Whenever development accuracy improves, the best checkpoint is written to `models/`. After training, the selected checkpoint is loaded and prediction files are written to `predictions/bert/`.

## Reproducing the QQP experiments

The linear and MLP heads were implemented in different source revisions rather than selected through a command-line option. The current `main` branch implements the MLP head, so its runs can be reproduced directly below. Reproducing the linear-head variant requires its recorded revision.

### Improvement 1: Pair-Feature Linear Head Run

This run used the explicit sentence-pair interaction representation with a linear classifier head. It was run at source revision `0329208` (`Modified embedding vectors implemented`). In a clean worktree, check out that revision and run:

```sh
git switch --detach 0329208
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

Result:

```text
Best QQP dev accuracy: 0.850
```

The corresponding archived SLURM script, log, and predictions are in `experiments/qqp_improvement_1_only_sentence_pair_repr/`.

### Improvement 2

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

Result:

```text
Best QQP dev accuracy: 0.849
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

Result:

```text
Best QQP dev accuracy: 0.868
Best epoch: 6
```

The corresponding SLURM script is `slurm_scripts/run_qqp_mlp_head_dropout01_7ep_es.sh`.

The final prediction files from the best run were copied to:

```text
experiments/qqp_mlp_head/dropout01_7ep_es_predictions/quora_mlp_head_dropout01_7ep_es_job15591323_dev.csv
experiments/qqp_mlp_head/dropout01_7ep_es_predictions/quora_mlp_head_dropout01_7ep_es_job15591323_test.csv
```

The corresponding checkpoint was also used to initialize experimental training runs for the STS (Semantic Textual Similarity) task, since both tasks involve sentence-pair comparison. However, this transfer is only expected to be potentially useful for the shared BERT encoder; the QQP-specific classification head is not directly applicable to STS because STS predicts a graded similarity score rather than a binary paraphrase label.

After reproducing an older experiment revision, return to the main branch with `git switch main`.

# Methodology

## QQP Paraphrase Detection

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

# Experiments

## QQP Paraphrase Detection

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

## Results

Only QQP results are populated from the supplied QQP draft. Results for the remaining project tasks will be added when their experiment documentation is available.

### Stanford Sentiment Treebank (SST)

| Model | Accuracy |
| --- | ---: |
| _To be added_ | — |

### Quora Question Pairs (QQP)

#### Model Variant Comparison

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

#### Best Run: Epoch-Level Results

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

### Semantic Textual Similarity (STS)

| Model | Pearson correlation |
| --- | ---: |
| _To be added_ | — |

### Paraphrase Type Detection (PTD)

| Model | Metric |
| --- | ---: |
| _To be added_ | — |

### Paraphrase Type Generation (PTG)

| Model | Metric |
| --- | ---: |
| _To be added_ | — |

### Discussion

#### Research Question 1

**Does adding explicit sentence-pair interaction features improve QQP development accuracy compared to the original BERT-based pair representation $[u, v]$?**

Yes. The pair-feature linear-head model reached **0.850** development accuracy, compared to approximately **0.765** for the course/default reference and **0.785** in our earlier local baseline run with the original $[u, v]$ representation. This supports the hypothesis that explicit difference and overlap features are useful for QQP paraphrase detection.

#### Research Question 2

**Does a non-linear MLP classification head improve QQP development accuracy compared to a linear classification head on the same sentence-pair interaction representation?**

Not by itself. With the same pair representation, dropout `0.3`, and 3 training epochs, the MLP head reached **0.849**, while the linear head reached **0.850**. This suggests that simply adding a more expressive head is not automatically beneficial under the original training configuration.

#### Research Question 3

**Does lowering dropout from `0.3` to `0.1` and increasing the epoch budget to 7, combined with best-checkpoint selection and optional early stopping, improve QQP development accuracy for the MLP head?**

Yes, in our experiment. The MLP model with dropout `0.1` and a 7-epoch budget reached **0.868** development accuracy, improving over both the MLP run with dropout `0.3` and the linear-head pair-feature run. However, because dropout, epoch budget, and early stopping were changed together, this result should be interpreted as a successful combined training configuration, not as a fully isolated proof that `0.1` is the optimal dropout value.

### Limitations

The comparison between the 3-epoch runs and the final 7-epoch run is limited because the shorter runs had not clearly converged. Both the pair-feature linear-head model and the MLP model with dropout `0.3` were still improving at epoch 3. Therefore, the final improvement to **0.868** cannot be attributed only to the lower dropout value or the MLP head. It may also partly result from the longer epoch budget. A stricter comparison would train all model variants with the same maximum epoch budget and the same early-stopping/checkpoint-selection protocol.

The value `hidden_dropout_prob = 0.1` was not selected through a full hyperparameter sweep. We only compared the default value `0.3` with one lower value, `0.1`. Therefore, the final setting should be interpreted as a targeted improvement rather than as an optimized hyperparameter choice.

The final comparison changes multiple factors at once: dropout is reduced, the epoch budget is increased, and early stopping is enabled. This means the improvement cannot be attributed to dropout alone.

All reported QQP experiments use the same random seed, `11711`. We therefore did not measure run-to-run variance across multiple random initializations and data orders.

The development set was used to compare model variants and training configurations. Therefore, the development accuracy should be interpreted as a model-selection metric, not as an independent estimate of final test performance.

We did not perform a complete dropout sweep over values such as `0.01`, `0.05`, `0.15`, or `0.2`. Such a sweep would be necessary to make a stronger claim about the optimal dropout value for this MLP head.

### Conclusion

The results support the usefulness of explicit sentence-pair interaction features for QQP paraphrase detection. The original pair representation $[u, v]$ gives the classifier access to both question embeddings, but leaves the actual comparison mostly implicit. Adding $\lvert u-v \rvert$ and $u \odot v$ makes dimension-wise difference and overlap information directly available to the classifier.

The strongest single architectural improvement was the sentence-pair interaction representation. It improved the development accuracy to **0.850** while still using a linear head. This indicates that the representation itself already provides a much more useful input for the classifier.

The MLP head did not improve performance under the original dropout and epoch settings. The run with dropout `0.3` and 3 epochs reached **0.849**, slightly below the linear-head pair-feature model. This suggests that a more expressive classifier head alone is not sufficient and may require a better-matched training configuration.

The best result was achieved by keeping the MLP head but lowering dropout to `0.1` and training for up to 7 epochs. The training curve shows that the model continued to fit the training data throughout all epochs, while development accuracy peaked at epoch 6. This supports using development-based checkpoint selection.

Early stopping was added mainly as a practical compute-saving mechanism. Since the code already saves the best checkpoint based on development accuracy, early stopping does not fundamentally change which checkpoint is selected if all epochs are completed. Its main purpose is to stop future runs earlier when development accuracy no longer improves, avoiding unnecessary use of HPC (High Performance Cluster) compute budget.

For QQP, we select the model with sentence-pair interaction features, MLP head, `hidden_dropout_prob = 0.1`, and best-checkpoint selection over a 7-epoch budget to be included in our main branch that is to be submitted since it achieved the best development accuracy among our tested QQP variants.

### Hyperparameter Optimization

The value `hidden_dropout_prob = 0.1` was **not** selected by an exhaustive hyperparameter sweep. We compared the original default value `0.3` with one targeted lower value, `0.1`. We did not test additional values such as `0.01`, `0.05`, `0.15`, or `0.2`.

Therefore, this should be interpreted as a targeted training-configuration experiment rather than complete hyperparameter optimization. In addition, the lower-dropout run increased the epoch budget from 3 to 7 and enabled early stopping, so the observed improvement cannot be attributed to dropout alone. A stricter follow-up would compare dropout values under the same epoch budget, early-stopping protocol, and multiple random seeds.

## Visualizations

### QQP Train and Development Accuracy

![QQP train/dev accuracy curves](qqp_paraphrase_predict/figures/qqp_train_dev_accuracy_curves.png)

The development accuracy of the strongest run peaked at epoch 6. In epoch 7, training accuracy continued to increase, but development accuracy dropped from `0.868` to `0.866`. This indicates mild overfitting after epoch 6 and supports selecting the best development checkpoint rather than the final epoch checkpoint.

### QQP Training Loss

![QQP training loss curves](qqp_paraphrase_predict/figures/qqp_train_loss_curves.png)

The training curves show that the strongest model continued to fit the training data throughout all epochs, while development accuracy peaked at epoch 6. The pair-feature linear-head model and MLP model with dropout `0.3` were also still improving at the end of their 3-epoch budgets, which limits direct attribution of the final gain to dropout or the MLP head alone.

## Members Contribution

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
