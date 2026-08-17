"""
Combined cross-architecture figure: AUROC of both backbones, with and without the
neuro-symbolic layer, side by side. This is the paper's key ablation figure -- it shows
at a glance that the physiology-guided symbolic knowledge raises BOTH architectures,
so the gain is not specific to SE-ECG Net.

Reads the two per-fold dumps produced by train_compare.py (one run per backbone):
    plot_data_seecg/plot_data.pkl      (or the legacy plot_data/plot_data.pkl)
    plot_data_cnnlstm/plot_data.pkl

Run AFTER both runs exist:
    python make_combined_figure.py     # -> figures/fig_combined_auroc.pdf
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

OUT_DIR = "figures"
OUT_FILE = "fig_combined_auroc.pdf"

# Same palette as make_figures.py: blue = backbone, orange = + neuro-symbolic.
C_NN, C_NSA = "#2a78d6", "#eb6834"

# (display name, candidate pickle paths in priority order)
BACKBONES = [
    ("SE-ECG Net", ["plot_data_seecg/plot_data.pkl", "plot_data/plot_data.pkl"]),
    ("CNN-BiLSTM", ["plot_data_cnnlstm/plot_data.pkl"]),
]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _fold_aurocs(fold_logs):
    """Per-fold AUROC from a run's per-fold list (uses stored probs/targets)."""
    aucs = []
    for f in fold_logs:
        y, p = np.asarray(f["targets"]), np.asarray(f["probs"])
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, p))
    return np.array(aucs, dtype=float)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    names, nn_mu, nn_sd, nsa_mu, nsa_sd = [], [], [], [], []
    for disp, candidates in BACKBONES:
        path = _first_existing(candidates)
        if path is None:
            print(f"[skip] no pickle for {disp} (looked in {candidates})")
            continue
        log = pickle.load(open(path, "rb"))
        a_nn, a_nsa = _fold_aurocs(log["nn"]), _fold_aurocs(log["nsa"])
        names.append(disp)
        nn_mu.append(a_nn.mean()); nn_sd.append(a_nn.std())
        nsa_mu.append(a_nsa.mean()); nsa_sd.append(a_nsa.std())
        print(f"{disp:12s}: backbone {a_nn.mean():.3f}+/-{a_nn.std():.3f}  "
              f"+NeSy {a_nsa.mean():.3f}+/-{a_nsa.std():.3f}  (Delta {a_nsa.mean()-a_nn.mean():+.3f})")

    if not names:
        raise SystemExit("No run data found. Run train_compare.py for each backbone first.")

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    b1 = ax.bar(x - w / 2, nn_mu, w, yerr=nn_sd, capsize=4, color=C_NN,
                label="Backbone", edgecolor="white", linewidth=0.8,
                error_kw=dict(ecolor="#555", lw=1.1))
    b2 = ax.bar(x + w / 2, nsa_mu, w, yerr=nsa_sd, capsize=4, color=C_NSA,
                label="+ Neuro-Symbolic", edgecolor="white", linewidth=0.8,
                error_kw=dict(ecolor="#555", lw=1.1))

    # Value labels above each bar.
    for bars, mus in ((b1, nn_mu), (b2, nsa_mu)):
        for rect, mu in zip(bars, mus):
            ax.text(rect.get_x() + rect.get_width() / 2, mu + 0.012, f"{mu:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Delta annotation between each pair.
    for xi, (a, b) in enumerate(zip(nn_mu, nsa_mu)):
        ax.annotate(f"+{b - a:.2f}", xy=(xi, max(a, b) + 0.075), ha="center",
                    fontsize=8, color=C_NSA, fontweight="bold")

    ax.axhline(0.5, color="#999999", ls="--", lw=1, zorder=0)
    ax.text(len(names) - 0.5, 0.5, "chance", color="#777777", fontsize=7,
            va="bottom", ha="right")

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.0)
    ax.set_title("Neuro-symbolic gain across two backbones")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, OUT_FILE)
    fig.savefig(out)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
