"""
Factorial ablation table (reviewer-requested): backbone / +HRV branch only /
+LTN axioms only / +full PGNSF, for both backbones, on the SAME folds.

Reads the per-fold dumps written by train_compare.py:
    full PGNSF : plot_data_seecg/ (or legacy plot_data/) and plot_data_cnnlstm/
                 -> 'nn' arm = backbone, 'nsa' arm = HRV+LTN
    HRV only   : plot_data_<backbone>_hrv1_ltn0/   ('nsa' arm)
    LTN only   : plot_data_<backbone>_hrv0_ltn1/   ('nsa' arm)

Run AFTER the ablation trainings:
    python train_compare.py seecg hrv
    python train_compare.py seecg ltn
    python train_compare.py cnnlstm hrv
    python train_compare.py cnnlstm ltn
    python make_ablation_table.py

Missing runs are reported and skipped, so this can be run progressively.
It also checks that the backbone ('nn') results are IDENTICAL across runs,
which the deterministic setup guarantees -- a useful reproducibility check.
"""

import os
import pickle
import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             accuracy_score)

BACKBONES = {
    "SE-ECG Net": {
        "full": ["plot_data_seecg/plot_data.pkl", "plot_data/plot_data.pkl"],
        "hrv":  ["plot_data_seecg_hrv1_ltn0/plot_data.pkl"],
        "ltn":  ["plot_data_seecg_hrv0_ltn1/plot_data.pkl"],
    },
    "CNN-BiLSTM": {
        "full": ["plot_data_cnnlstm/plot_data.pkl"],
        "hrv":  ["plot_data_cnnlstm_hrv1_ltn0/plot_data.pkl"],
        "ltn":  ["plot_data_cnnlstm_hrv0_ltn1/plot_data.pkl"],
    },
}
ROW_LABEL = {"base": "Backbone", "hrv": "+HRV branch only", "ltn": "+LTN axioms only",
             "full": "+HRV+LTN (PGNSF)"}


def ece_eqmass(p, y, n_bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    o = np.argsort(p); p, y = p[o], y[o]
    n, e = len(p), 0.0
    for idx in np.array_split(np.arange(n), min(n_bins, n)):
        if len(idx):
            e += (len(idx) / n) * abs(y[idx].mean() - p[idx].mean())
    return e


def nb_ttest(diffs):
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    k = len(d)
    if k < 2:
        return float("nan"), float("nan")
    m, v = d.mean(), d.var(ddof=1)
    if v == 0:
        return (float("inf") if m else 0.0), (0.0 if m else 1.0)
    t = m / np.sqrt((1.0 / k + 1.0 / (k - 1)) * v)
    return t, 2.0 * stats.t.sf(abs(t), df=k - 1)


def fold_metrics(fold):
    y, p = np.asarray(fold["targets"]), np.asarray(fold["probs"])
    pred = (p >= 0.5).astype(int)
    s = np.asarray(fold["subjects"])
    sp = np.array([p[s == u].mean() for u in np.unique(s)])
    sl = np.array([y[s == u][0] for u in np.unique(s)])
    return {
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "bal_acc": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "acc": accuracy_score(y, pred),
        "ece": ece_eqmass(p, y),
        "s_auroc": roc_auc_score(sl, sp) if len(np.unique(sl)) > 1 else np.nan,
    }


def load_first(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f), p
    return None, None


def main():
    keys = ["auroc", "bal_acc", "f1_macro", "acc", "ece", "s_auroc"]
    for bk_name, sources in BACKBONES.items():
        full_log, full_path = load_first(sources["full"])
        if full_log is None:
            print(f"[{bk_name}] full-PGNSF run not found -- skipping backbone entirely.")
            continue

        arms = {"base": [fold_metrics(f) for f in full_log["nn"]],
                "full": [fold_metrics(f) for f in full_log["nsa"]]}
        base_auroc_ref = np.array([m["auroc"] for m in arms["base"]])

        missing = []
        for arm in ("hrv", "ltn"):
            log, path = load_first(sources[arm])
            if log is None:
                missing.append(arm)
                continue
            arms[arm] = [fold_metrics(f) for f in log["nsa"]]
            # Reproducibility check: the 'nn' arm of an ablation run must match the
            # paper's backbone exactly (deterministic seeds).
            nn_auroc = np.array([fold_metrics(f)["auroc"] for f in log["nn"]])
            if not np.allclose(nn_auroc, base_auroc_ref, atol=1e-6):
                print(f"  [warn] {bk_name}/{arm}: backbone fold AUROCs differ from the "
                      f"reference run -- check seeds/config ({path})")

        print(f"\n{'='*86}\n FACTORIAL ABLATION -- {bk_name}   (source: {full_path})")
        if missing:
            print(f" missing arms (not yet trained): {', '.join(ROW_LABEL[m] for m in missing)}")
        print(f"{'='*86}")
        header = f"{'configuration':<20}" + "".join(f"{k:>11}" for k in keys)
        print(header + f"{'p vs base':>11}")
        for arm in ("base", "hrv", "ltn", "full"):
            if arm not in arms:
                continue
            row = f"{ROW_LABEL[arm]:<20}"
            for k in keys:
                v = np.array([m[k] for m in arms[arm]], float)
                row += f"{np.nanmean(v):11.3f}"
            if arm == "base":
                row += f"{'---':>11}"
            else:
                diffs = [a["auroc"] - b["auroc"] for a, b in zip(arms[arm], arms["base"])]
                _, p = nb_ttest(diffs)
                row += f"{p:11.4f}"
            print(row)
        print(f"{'='*86}")
        print(" (p vs base: Nadeau-Bengio corrected paired t-test on per-fold window AUROC)")


if __name__ == "__main__":
    main()
