"""
Generates the paper figures from the data dumped by train_compare.py.

Run AFTER a full training run:
    python train_compare.py      # writes plot_data/plot_data.pkl
    python make_figures.py       # writes figures/*.pdf  (vector, ready for LaTeX)

Figures produced
----------------
  fig_loss_curves.pdf   Cross-entropy and LTN satisfaction, side by side (the
                        neuro-symbolic evidence: the network learns to classify AND
                        to satisfy the physiological axioms).
  fig_perfold_auroc.pdf Paired per-fold AUROC, Backbone vs PGNSF (matches the paired
                        Nadeau-Bengio test: same fold, both models).
  fig_roc.pdf           Mean ROC over folds with +/-1 std band.
  fig_reliability.pdf   Reliability diagram (equal-mass bins) visualizing the ECE gain.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Backbone selects which run's data/figures to use. Pass it as the first CLI argument:
#     python make_figures.py            # default: seecg  -> plot_data_seecg/ , figures/
#     python make_figures.py cnnlstm    #          cnnlstm -> plot_data_cnnlstm/, figures_cnnlstm/
BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "seecg"
PLOT_DATA = os.path.join(f"plot_data_{BACKBONE}", "plot_data.pkl")
OUT_DIR = "figures" if BACKBONE == "seecg" else f"figures_{BACKBONE}"

# Colours: blue = neural backbone (baseline), orange = backbone + neuro-symbolic layer.
C_NN, C_NSA = "#2a78d6", "#eb6834"
_BACKBONE_LABEL = {"seecg": "SE-ECG Net", "cnnlstm": "CNN-BiLSTM"}
_bb = _BACKBONE_LABEL.get(BACKBONE, BACKBONE)
LABEL = {"nn": f"Backbone ({_bb})", "nsa": f"{_bb} + NeSy"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load():
    if not os.path.exists(PLOT_DATA):
        raise SystemExit(f"'{PLOT_DATA}' not found - run train_compare.py first.")
    with open(PLOT_DATA, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- figure 1
def fig_loss_curves(log):
    """Cross-entropy (left) and LTN satisfaction (right), side by side.

    The satisfaction panel is drawn on the full [0, 1] axis with the logical-chance
    line at 0.5, so the honest reading is visible: SatAgg(K) stays HIGH and stable
    (well above chance) throughout training -- the physiological constraints keep the
    predictions consistent with physiology while the cross-entropy is minimized. It
    does NOT climb toward 1; it holds in a narrow high band, which is the expected
    behaviour of a soft regularizer that must trade off against the classification loss.
    """
    ce_nn = np.vstack([f["curve_ce"] for f in log["nn"]])
    ce_nsa = np.vstack([f["curve_ce"] for f in log["nsa"]])
    sat = np.vstack([f["curve_sat"] for f in log["nsa"]])
    epochs = np.arange(1, ce_nn.shape[1] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    for arr, c, key in ((ce_nn, C_NN, "nn"), (ce_nsa, C_NSA, "nsa")):
        mu, sd = arr.mean(0), arr.std(0)
        ax1.plot(epochs, mu, color=c, lw=2, label=LABEL[key])
        ax1.fill_between(epochs, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("(a) Classification loss")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25, lw=0.6)

    mu, sd = np.nanmean(sat, 0), np.nanstd(sat, 0)
    ax2.plot(epochs, mu, color=C_NSA, lw=2, label=r"PGNSF  SatAgg($\mathcal{K}$)")
    ax2.fill_between(epochs, mu - sd, mu + sd, color=C_NSA, alpha=0.25, lw=0)
    ax2.axhline(0.5, color="#999999", ls="--", lw=1)
    ax2.text(epochs[-1], 0.5, "logical chance", color="#777777", fontsize=7,
             va="bottom", ha="right")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel(r"Axiom satisfaction  SatAgg($\mathcal{K}$)")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(b) Physiological consistency")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_loss_curves.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_perfold_auroc(log):
    """Paired per-fold AUROC: one connected pair per fold."""
    a_nn = np.array([f["auroc"] for f in log["nn"]], dtype=float)
    a_nsa = np.array([f["auroc"] for f in log["nsa"]], dtype=float)
    folds = np.arange(1, len(a_nn) + 1)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(0.5, color="#999999", ls="--", lw=1, zorder=1)
    ax.text(len(folds) + 0.35, 0.5, "chance", color="#777777", fontsize=7, va="center")

    for i, (x, p, q) in enumerate(zip(folds, a_nn, a_nsa)):
        ax.plot([x, x], [p, q], color="#c9ccd0", lw=1.6, zorder=2,
                solid_capstyle="round")
    ax.scatter(folds, a_nn, s=42, color=C_NN, zorder=3, label=LABEL["nn"],
               edgecolor="white", linewidth=1.1)
    ax.scatter(folds, a_nsa, s=42, color=C_NSA, zorder=3, label=LABEL["nsa"],
               edgecolor="white", linewidth=1.1)

    wins = int((a_nsa > a_nn).sum())
    ax.set_xticks(folds)
    ax.set_xlabel("Cross-validation fold"); ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-fold AUROC (PGNSF higher in {wins}/{len(folds)} folds)")
    ax.legend(frameon=False, loc="lower right", ncol=1)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_perfold_auroc.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_roc(log):
    """Mean ROC across folds with +/-1 std band."""
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    for key, c in (("nn", C_NN), ("nsa", C_NSA)):
        tprs, aucs = [], []
        for f in log[key]:
            y, p = f["targets"], f["probs"]
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(roc_auc_score(y, p))
        tprs = np.vstack(tprs); tprs[:, 0] = 0.0
        mu, sd = tprs.mean(0), tprs.std(0)
        ax.plot(grid, mu, color=c, lw=2,
                label=f"{LABEL[key]} (AUC {np.mean(aucs):.2f})")
        ax.fill_between(grid, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=c, alpha=0.15, lw=0)

    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Mean ROC over 10 subject-independent folds")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_roc.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_reliability(log, n_bins=10):
    """Reliability diagram with equal-mass bins (matches the ECE definition used)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1, label="Perfect calibration")

    for key, c in (("nn", C_NN), ("nsa", C_NSA)):
        p = np.concatenate([f["probs"] for f in log[key]])
        y = np.concatenate([f["targets"] for f in log[key]]).astype(float)
        order = np.argsort(p)
        p, y = p[order], y[order]
        conf, freq, ece, n = [], [], 0.0, len(p)
        for idx in np.array_split(np.arange(n), n_bins):
            if len(idx) == 0:
                continue
            conf.append(p[idx].mean()); freq.append(y[idx].mean())
            ece += (len(idx) / n) * abs(y[idx].mean() - p[idx].mean())
        ax.plot(conf, freq, "o-", color=c, lw=1.8, ms=5,
                label=f"{LABEL[key]} (ECE {ece:.2f})")

    ax.set_xlabel("Mean predicted P(high load)")
    ax.set_ylabel("Observed frequency of high load")
    ax.set_title("Reliability diagram (equal-mass bins)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_reliability.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    log = load()
    fig_loss_curves(log)
    fig_perfold_auroc(log)
    fig_roc(log)
    fig_reliability(log)
    print(f"Figures written to '{OUT_DIR}/':")
    for n in ("fig_loss_curves", "fig_perfold_auroc", "fig_roc", "fig_reliability"):
        print(f"  - {n}.pdf")
