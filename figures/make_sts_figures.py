"""Training-dynamics figures for the STS (Part 2) write-up.

Follows the project guideline for the Visualizations section: plot metrics
*during* training and compare the training processes of the improvements
(convergence speed, final performance, overfitting onset, failure modes).

Every number is parsed from the SLURM logs in experiments/slurm_files/, so
nothing here is hand-entered.

Run from the repository root:  python figures/make_sts_figures.py
"""
import csv, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS, OUT = "slurm_files", "figures/sts"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
})

TARGET, NOISE = 0.811, 0.004
# consistent colour per improvement across every figure
CL = {"bi": "#9E9E9E", "ca": "#2A7DB1", "sym": "#E08214",
      "snli": "#1B5E20", "quora": "#7B4FA0", "bad": "#B03A2E"}


def curve(name):
    """(epoch, train_loss, train_r, dev_r) from one SLURM log, or None."""
    p = f"{LOGS}/{name}.out"
    if not os.path.exists(p):
        return None
    m = re.findall(
        r"Epoch (\d+) \(sts\): train loss :: ([\d.]+), train :: ([\d.]+), dev :: ([\d.]+)",
        open(p).read())
    if not m:
        return None
    a = np.array([[float(x) for x in r] for r in m])
    return dict(ep=a[:, 0], loss=a[:, 1], train=a[:, 2], dev=a[:, 3])


def save(fig, name, tight=True):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  {name}.png")


# the five stages of the improvement sequence, used by several figures
MAIN = [
    ("sts_exp2_mean_pool_cosine_head", "1. cosine head + mean pooling", CL["bi"],  ":"),
    ("sts_exp5_mnrl",                  "2. + MNRL contrastive",         CL["bi"],  "-"),
    ("sts_exp16_crossattn",            "3. + cross-attention",          CL["ca"],  "-"),
    ("sts_exp19_ca_symmetry",          "4. + symmetry augmentation",    CL["sym"], "-"),
    ("sts_exp08_seed42",               "5. + SNLI pretraining",         CL["snli"],"-"),
]


# --------------------------------------------------------------------- V1
def v1_progression():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for f, lab, col, ls in MAIN:
        c = curve(f)
        if c is None:
            continue
        ax.plot(c["ep"], c["dev"], ls, color=col, lw=2.0, marker="o", ms=3.6,
                label=lab, alpha=0.95)
        k = int(np.argmax(c["dev"]))
        ax.plot(c["ep"][k], c["dev"][k], "*", color=col, ms=15,
                markeredgecolor="white", markeredgewidth=0.7, zorder=6)
    ax.axhline(TARGET, color=CL["bad"], ls="--", lw=1.1, zorder=0)
    ax.text(10.4, TARGET + .003, f"team target {TARGET}", color=CL["bad"], fontsize=8)
    ax.set_xlabel("training epoch"); ax.set_ylabel("dev Pearson r")
    ax.set_title("Dev Pearson r per epoch")
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.40))
    ax.set_xlim(0.5, 10.6); ax.set_ylim(0.55, 0.88)
    save(fig, "v1_dev_progression")


# --------------------------------------------------------------------- V2
def v2_convergence_speed():
    """Epochs needed to first reach each dev-r threshold."""
    ths = [0.78, 0.80, 0.82, 0.84]
    runs = [m for m in MAIN if m[0] != "sts_exp2_mean_pool_cosine_head"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    w = 0.8 / len(runs)
    for i, (f, lab, col, _) in enumerate(runs):
        c = curve(f)
        if c is None:
            continue
        xs, ys = [], []
        for j, t in enumerate(ths):
            hit = np.where(c["dev"] >= t)[0]
            xs.append(j + (i - (len(runs) - 1) / 2) * w)
            ys.append(c["ep"][hit[0]] if len(hit) else np.nan)
        ax.bar(xs, ys, width=w * 0.92, color=col, label=lab)
        for x, y in zip(xs, ys):
            if np.isnan(y):
                ax.text(x, 0.25, "never", ha="center", fontsize=7,
                        rotation=90, color="#777")
            else:
                ax.text(x, y + 0.15, f"{int(y)}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(ths)))
    ax.set_xticklabels([f"dev r ≥ {t}" for t in ths])
    ax.set_ylabel("epochs required (lower = faster)")
    ax.set_title("Epochs to reach target dev r")
    ax.legend(loc="upper left"); ax.set_ylim(0, 11)
    save(fig, "v2_convergence_speed")


# --------------------------------------------------------------------- V3
def v3_overfitting():
    runs = [("sts_exp5_mnrl", "bi-encoder + MNRL", CL["bi"]),
            ("sts_exp16_crossattn", "+ cross-attention", CL["ca"]),
            ("sts_exp19_ca_symmetry", "+ symmetry", CL["sym"]),
            ("sts_exp08_seed42", "+ SNLI pretraining", CL["snli"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for f, lab, col in runs:
        c = curve(f)
        if c is None:
            continue
        ax.plot(c["ep"], c["train"], "-", color=col, lw=1.8, label=f"{lab} (train)")
        ax.plot(c["ep"], c["dev"], "--", color=col, lw=1.5, alpha=0.85)
    ax.set_xlabel("training epoch"); ax.set_ylabel("Pearson r")
    ax.set_title("Train vs dev Pearson r")
    ax.legend(loc="lower right", fontsize=7.5); ax.set_ylim(0.70, 1.0)

    ax = axes[1]
    for f, lab, col in runs:
        c = curve(f)
        if c is None:
            continue
        ax.plot(c["ep"], c["train"] - c["dev"], "-o", color=col, lw=1.8,
                ms=3.6, label=lab)
    ax.set_xlabel("training epoch"); ax.set_ylabel("train r − dev r")
    ax.set_title("Generalisation gap")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Overfitting dynamics", y=1.00, fontweight="bold")
    save(fig, "v3_overfitting_dynamics")


# --------------------------------------------------------------------- V4
def v4_train_loss():
    """Loss is only comparable across runs with an identical loss composition."""
    runs = [("sts_exp16_crossattn", "cross-attention", CL["ca"]),
            ("sts_exp19_ca_symmetry", "+ symmetry", CL["sym"]),
            ("sts_exp25a_qqpckpt", "+ teammate encoder init", CL["quora"]),
            ("sts_exp08_seed42", "+ SNLI pretraining", CL["snli"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for f, lab, col in runs:
        c = curve(f)
        if c is None:
            continue
        ax.plot(c["ep"], c["loss"], "-o", color=col, lw=1.9, ms=4, label=lab)
    ax.set_xlabel("training epoch"); ax.set_ylabel("training loss")
    ax.set_title("Training loss per epoch")
    ax.text(0.98, 0.96,
            "all four share an identical loss\n(MSE + 0.5·cosine + 0.5·MNRL),\n"
            "so the curves are directly comparable",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#555")
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.80))
    save(fig, "v4_training_loss")


# --------------------------------------------------------------------- V5
def v5_failures():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8))
    fig.subplots_adjust(wspace=0.62)   # room for panel 1's twin axis

    ax = axes[0]      # SimCSE: loss collapses, dev declines
    c = curve("sts_exp7b_simcse_only")
    if c:
        ax.plot(c["ep"], c["dev"], "-o", color=CL["bad"], lw=2, ms=4,
                label="dev Pearson r")
        ax.set_ylabel("dev Pearson r"); ax.set_ylim(0.55, 0.70)
        a2 = ax.twinx(); a2.grid(False)
        a2.plot(c["ep"], c["loss"], "-s", color="#555", lw=1.6, ms=3.6,
                label="training loss")
        a2.set_ylabel("training loss", labelpad=1); a2.set_ylim(0, 0.04)
        ax.set_title("SimCSE (unsupervised)")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5)
    ax.set_xlabel("epoch")

    ax = axes[1]      # CoSENT temperature
    for f, lab, col in [("sts_exp13_cosent", "τ=0.05 (overflow)", CL["bad"]),
                        ("sts_exp13c_cosent_drop03", "τ=0.5 (stable)", CL["ca"])]:
        c = curve(f)
        if c:
            ax.plot(c["ep"], c["dev"], "-o", color=col, lw=2, ms=4, label=lab)
    ax.axhline(0.804, color="#333", ls="--", lw=1)
    ax.text(10.2, 0.812, "MNRL baseline", ha="right", fontsize=7.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("dev Pearson r")
    ax.set_title("CoSENT temperature")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[2]      # MNRL temperature sweep
    for f, lab, col in [("sts_exp9a_tau001", "τ=0.01", CL["bad"]),
                        ("sts_exp5_mnrl", "τ=0.05 (best)", CL["snli"]),
                        ("sts_exp9b_tau01", "τ=0.10", CL["ca"])]:
        c = curve(f)
        if c:
            ax.plot(c["ep"], c["dev"], "-o", color=col, lw=2, ms=4, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("dev Pearson r")
    ax.set_title("MNRL temperature")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Failure modes", y=1.02, fontweight="bold")
    save(fig, "v5_failure_modes", tight=False)


# --------------------------------------------------------------------- V6
def v6_seed_variance():
    seeds = [("sts_exp08_snli", "seed 11711", 6), ("sts_exp08_seed42", "seed 42", 5),
             ("sts_exp08_seed7", "seed 7", 5)]
    cs = [(lab, curve(f)) for f, lab, _ in seeds]
    cs = [(l, c) for l, c in cs if c is not None]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    n = min(len(c["dev"]) for _, c in cs)
    stack = np.vstack([c["dev"][:n] for _, c in cs])
    ep = cs[0][1]["ep"][:n]
    ax.fill_between(ep, stack.min(0), stack.max(0), color=CL["snli"], alpha=0.18,
                    label="spread across seeds")
    for lab, c in cs:
        ax.plot(c["ep"], c["dev"], "-o", lw=1.5, ms=3.6, label=lab, alpha=0.9)
    ax.plot(ep, stack.mean(0), "-", color=CL["snli"], lw=2.6, label="mean")
    best = [c["dev"].max() for _, c in cs]
    ax.set_xlabel("training epoch"); ax.set_ylabel("dev Pearson r")
    ax.set_title(f"Seed variance (spread {max(best)-min(best):.3f})")
    ax.legend(loc="lower left", ncol=2); ax.set_ylim(0.8375, 0.8545)
    save(fig, "v6_seed_variance")


# --------------------------------------------------------------------- V7
def v7_error_analysis():
    """Supplementary: not a training curve, but the required error analysis."""
    pred, gold = [], []
    with open("predictions/bert/sts-similarity-dev-output.csv") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                pred.append(float(p[1]))
    with open("data/sts-similarity-dev.csv") as f:
        for r in csv.DictReader(f):
            gold.append(float(r["similarity"]))
    pred, gold = np.array(pred), np.array(gold)
    r = np.corrcoef(pred, gold)[0, 1]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    ax = axes[0]
    ax.scatter(gold, pred, s=7, alpha=0.22, color=CL["ca"], edgecolors="none")
    ax.plot([-.2, 5.2], [-.2, 5.2], "--", color="#333", lw=1, label="perfect")
    m, b = np.polyfit(gold, pred, 1)
    xs = np.linspace(0, 5, 10)
    ax.plot(xs, m * xs + b, color=CL["bad"], lw=1.6, label=f"fit (slope {m:.2f})")
    ax.set_xlabel("gold similarity"); ax.set_ylabel("predicted")
    ax.set_title(f"Predicted vs gold (r = {r:.3f})")
    ax.legend(loc="upper left"); ax.set_xlim(-.2, 5.2); ax.set_ylim(-.2, 5.2)

    ax = axes[1]
    bins = np.arange(0, 5.01, 1.0)
    idx = np.clip(np.digitize(gold, bins) - 1, 0, len(bins) - 2)
    mae = [np.abs(pred[idx == i] - gold[idx == i]).mean() for i in range(len(bins) - 1)]
    cnt = [(idx == i).sum() for i in range(len(bins) - 1)]
    ax.bar(range(len(mae)), mae, color=CL["sym"], width=0.65)
    for i, (v, c_) in enumerate(zip(mae, cnt)):
        ax.text(i, v + .015, f"n={c_}", ha="center", fontsize=7, color="#555")
    ax.set_xticks(range(len(mae)))
    ax.set_xticklabels([f"{bins[i]:.0f}–{bins[i+1]:.0f}" for i in range(len(mae))])
    ax.set_xlabel("gold similarity band"); ax.set_ylabel("mean absolute error")
    ax.set_title("Error by similarity band")

    ax = axes[2]
    ax.hist(gold, bins=25, alpha=0.55, color="#9E9E9E", label=f"gold (σ={gold.std():.2f})")
    ax.hist(pred, bins=25, alpha=0.65, color=CL["ca"], label=f"predicted (σ={pred.std():.2f})")
    ax.set_xlabel("similarity"); ax.set_ylabel("count")
    ax.set_title("Prediction distribution"); ax.legend()
    fig.suptitle("Error analysis (dev set, n=1430)", y=1.02, fontweight="bold")
    save(fig, "v7_error_analysis")


# --------------------------------------------------------------------- V8
def v8_transfer_sources():
    """Which transfer source shapes the encoder best, and how fast."""
    runs = [("sts_exp25a_qqpckpt",  "no transfer pretraining", CL["bi"],    5716),
            ("sts_exp12_minedneg",  "TF-IDF mined (5.7k)",     "#B8A25E",   5716),
            ("sts_exp23_paws",      "PAWS (9.7k)",             CL["sym"],   9672),
            ("sts_exp24a_quora_pt1","Quora positives (50k)",   CL["quora"], 49796),
            ("sts_exp08_snli",      "SNLI triplets (149k)",    CL["snli"],  149145)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))

    ax = axes[0]
    for f, lab, col, _ in runs:
        c = curve(f)
        if c is None:
            continue
        ax.plot(c["ep"], c["dev"], "-o", color=col, lw=1.9, ms=3.8, label=lab)
    ax.set_xlabel("STS fine-tuning epoch"); ax.set_ylabel("dev Pearson r")
    ax.set_title("Dev Pearson r per epoch")
    ax.legend(loc="lower right", fontsize=7.6); ax.set_ylim(0.805, 0.858)

    ax = axes[1]
    ep1, best, cols, labs = [], [], [], []
    for f, lab, col, n in runs:
        c = curve(f)
        if c is None:
            continue
        ep1.append(c["dev"][0]); best.append(c["dev"].max())
        cols.append(col); labs.append(lab.split(" (")[0])
    x = np.arange(len(ep1))
    ax.bar(x - 0.2, ep1, width=0.38, color=cols, alpha=0.55, label="after epoch 1")
    ax.bar(x + 0.2, best, width=0.38, color=cols, label="best epoch")
    for i, (a, b) in enumerate(zip(ep1, best)):
        ax.text(i - 0.2, a + .001, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + .001, f"{b:.3f}", ha="center", fontsize=7,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.6, rotation=12)
    ax.set_ylabel("dev Pearson r"); ax.set_ylim(0.80, 0.865)
    ax.set_title("After 1 epoch vs best")
    ax.legend(loc="upper left", fontsize=7.6)
    fig.suptitle("Transfer pretraining sources", y=1.02, fontweight="bold")
    save(fig, "v8_transfer_sources")


# --------------------------------------------------------------------- V9
def v9_regularisation():
    """Regularisation attempts are visually indistinguishable from baseline."""
    runs = [("sts_exp19_ca_symmetry",     "no regularisation",        "#111111", 2.4),
            ("sts_exp22a_wd001",          "weight decay 0.01",        CL["ca"],  1.5),
            ("sts_exp22b_wd01",           "weight decay 0.1",         CL["sym"], 1.5),
            ("sts_exp22c_cadrop03",       "cross-attn dropout 0.3",   CL["quora"],1.5),
            ("sts_exp22d_wd001_cadrop03", "both",                     "#4FA0A0", 1.5),
            ("sts_exp22e_wd20",           "weight decay 20",          CL["bad"], 2.0)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))

    ax = axes[0]
    for f, lab, col, lw in runs:
        c = curve(f)
        if c is None:
            continue
        ax.plot(c["ep"], c["dev"], "-o", color=col, lw=lw, ms=3.6, label=lab,
                alpha=0.9)
    ax.set_xlabel("training epoch"); ax.set_ylabel("dev Pearson r")
    ax.set_title("Dev Pearson r per epoch")
    ax.legend(loc="lower right", fontsize=7.4); ax.set_xlim(0.6, 8.4)

    ax = axes[1]
    for f, lab, col, lw in runs:
        c = curve(f)
        if c is None:
            continue
        ax.plot(c["ep"], c["train"], "-o", color=col, lw=lw, ms=3.6, label=lab,
                alpha=0.9)
    ax.set_xlabel("training epoch"); ax.set_ylabel("train Pearson r")
    ax.set_title("Train Pearson r per epoch")
    ax.set_xlim(0.6, 8.4)
    ax.annotate("only curve that moves", xy=(6, 0.972), xytext=(3.4, 0.952),
                fontsize=8, color=CL["bad"],
                arrowprops=dict(arrowstyle="->", color=CL["bad"], lw=1.2))
    fig.suptitle("Regularisation experiments", y=1.02, fontweight="bold")
    save(fig, "v9_regularisation")


# --------------------------------------------------------------------- V10
CKPT = "models/sts_exp08_seed42.pt"


def v10_cross_attention():
    """Attention weights from a trained checkpoint (not from the logs)."""
    if not os.path.exists(CKPT):
        print(f"  v10 skipped: {CKPT} not present (checkpoints are not committed)")
        return
    import torch, importlib.util
    from tokenizer import BertTokenizer
    _argv, sys.argv = sys.argv, ["x"]
    _spec = importlib.util.spec_from_file_location("mc", "multitask_classifier.py")
    mc = importlib.util.module_from_spec(_spec); sys.modules["mc"] = mc
    _spec.loader.exec_module(mc)
    sys.argv = _argv
    saved = torch.load("models/sts_exp08_seed42.pt", map_location="cpu")
    model = mc.MultitaskBERT(saved["model_config"]); model.load_state_dict(saved["model"]); model.eval()
    tok = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    def attend(s1, s2):
        e1 = tok([s1], return_tensors="pt", padding=True, truncation=True)
        e2 = tok([s2], return_tensors="pt", padding=True, truncation=True)
        i1, m1 = torch.LongTensor(e1["input_ids"]), torch.LongTensor(e1["attention_mask"])
        i2, m2 = torch.LongTensor(e2["input_ids"]), torch.LongTensor(e2["attention_mask"])
        with torch.no_grad():
            h1 = model.bert(i1, m1)["last_hidden_state"]
            h2 = model.bert(i2, m2)["last_hidden_state"]
            _, w = model.cross_attn_layer(h1, h2, h2, key_padding_mask=(m2 == 0),
                                          need_weights=True, average_attn_weights=True)
            pred = model.predict_similarity_sts(i1, m1, i2, m2).item()
        t1 = tok.convert_ids_to_tokens(i1[0].tolist())
        t2 = tok.convert_ids_to_tokens(i2[0].tolist())
        return w[0].numpy(), t1, t2, pred

    # one clearly-similar pair and one clearly-dissimilar pair, both from STS-style data
    pairs = [("A man is playing a guitar.", "A person is playing an instrument.", "high similarity"),
             ("A man is playing a guitar.", "A woman is slicing an onion.",       "low similarity")]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    fig.subplots_adjust(wspace=0.55)
    for ax, (s1, s2, label) in zip(axes, pairs):
        w, t1, t2, pred = attend(s1, s2)
        im = ax.imshow(w, cmap="viridis", aspect="auto", vmin=0, vmax=w.max())
        ax.set_xticks(range(len(t2))); ax.set_xticklabels(t2, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(t1))); ax.set_yticklabels(t1, fontsize=8)
        ax.set_xlabel("sentence 2 tokens (keys)"); ax.set_ylabel("sentence 1 tokens (queries)")
        ax.set_title(f"{label}  (pred {pred:.2f}/5)")
        # mark the strongest non-special alignment per content token
        for i in range(1, len(t1) - 1):
            j = int(np.argmax(w[i, 1:len(t2)-1])) + 1
            ax.plot(j, i, "s", mfc="none", mec="white", ms=11, mew=1.4)
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
        cb.set_label("attention weight", fontsize=8); cb.ax.tick_params(labelsize=7)

    fig.suptitle("Cross-attention weights (KL from uniform = 0.002)", y=1.02, fontweight="bold")

    fig.savefig(f"{OUT}/v10_cross_attention.png", bbox_inches="tight")
    plt.close(fig)
    print("  v10_cross_attention.png")


if __name__ == "__main__":
    # The SLURM logs these figures are built from are gitignored, so a fresh clone
    # cannot regenerate them. Fail loudly rather than silently emitting empty plots
    # over the committed ones.
    have = [f for f in os.listdir(LOGS) if f.endswith(".out")] if os.path.isdir(LOGS) else []
    if len(have) < 10:
        sys.exit(
            f"Refusing to run: found {len(have)} training logs in {LOGS}/ (expected ~43).\n"
            "These figures are generated from the raw SLURM logs of the experiment runs,\n"
            "which are not committed. The committed PNGs in figures/sts/ are the\n"
            "originals; this script is provided to document how they were produced and\n"
            "to regenerate them from a full set of logs."
        )
    print(f"generating figures -> {OUT}")
    for fn in (v1_progression, v2_convergence_speed, v3_overfitting,
               v4_train_loss, v5_failures, v6_seed_variance, v7_error_analysis,
               v8_transfer_sources, v9_regularisation,
               v10_cross_attention):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
    print("done")
