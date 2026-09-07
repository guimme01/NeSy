"""
Non-deep baselines on the SAME HRV features and the SAME subject-independent folds
used by train_compare.py (reviewer-requested reference point).

Trains regularized logistic regression and gradient boosting on the ~86 handcrafted
HRV features, with the identical StratifiedGroupKFold(10, shuffle, seed=42) split,
identical leakage-free per-fold median imputation + standardization, and the same
metrics (window AUROC / balanced accuracy / macro-F1 / ECE, subject-level AUROC).

Run:
    python baseline_hrv.py
No GPU or retraining of the deep models is needed; it reads the cached dataset.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             accuracy_score)

CACHE = "processed_real_dataset_88subj_rawhrv_raw1d.pkl"
SEED = 42
N_SPLITS = 10


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


def impute_standardize(hrv, train_idx):
    """Identical to standardize_hrv_per_fold in train_compare.py (train-only stats)."""
    med = np.nanmedian(hrv[train_idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isnan(hrv), med, hrv)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x[train_idx].mean(axis=0)
    sd = x[train_idx].std(axis=0) + 1e-6
    return (x - mu) / sd


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f"cache '{CACHE}' not found - run train_compare.py once first.")
    print(f"Loading cache '{CACHE}' (only HRV/labels/subjects are used)...")
    with open(CACHE, "rb") as f:
        data = pickle.load(f)
    hrv = np.asarray(data["x_hrv"], dtype=np.float64)      # RAW, NaN kept
    y = np.asarray(data["y_labels"])
    groups = np.asarray(data["x_subjects"])
    print(f"windows={len(y)}  features={hrv.shape[1]}  subjects={len(np.unique(groups))}")

    # EXACT same folds as train_compare.py (split depends only on y/groups/seed).
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.zeros(len(y)), y, groups=groups))

    models = {
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "gboost": lambda: HistGradientBoostingClassifier(random_state=SEED,
                                                         class_weight="balanced"),
    }

    results = {name: [] for name in models}
    for fold, (tr, va) in enumerate(splits, 1):
        x = impute_standardize(hrv, tr)
        for name, make in models.items():
            clf = make().fit(x[tr], y[tr])
            p = clf.predict_proba(x[va])[:, 1]
            pred = (p >= 0.5).astype(int)
            # subject-level aggregation (mean window probability per subject)
            s = groups[va]
            sp = np.array([p[s == u].mean() for u in np.unique(s)])
            sl = np.array([y[va][s == u][0] for u in np.unique(s)])
            results[name].append({
                "auroc": roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan,
                "bal_acc": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro", zero_division=0),
                "acc": accuracy_score(y[va], pred),
                "ece": ece_eqmass(p, y[va]),
                "s_auroc": roc_auc_score(sl, sp) if len(np.unique(sl)) > 1 else np.nan,
            })
        print(f"  fold {fold:2d}/{N_SPLITS} done")

    print(f"\n{'='*74}\n HRV-only NON-DEEP baselines (same folds/imputation as the deep models)")
    print(f"{'='*74}")
    print(f"{'model':<8}{'AUROC':>14}{'bal_acc':>12}{'macroF1':>12}{'acc':>12}"
          f"{'ECE':>10}{'subjAUROC':>11}")
    for name, folds in results.items():
        def ms(k):
            v = np.array([f[k] for f in folds], float)
            return np.nanmean(v), np.nanstd(v)
        a, asd = ms("auroc"); b, _ = ms("bal_acc"); f1m, _ = ms("f1_macro")
        ac, _ = ms("acc"); ec, _ = ms("ece"); sa, _ = ms("s_auroc")
        print(f"{name:<8}{a:8.3f}+/-{asd:.2f}{b*100:11.1f}%{f1m*100:11.1f}%"
              f"{ac*100:11.1f}%{ec:10.3f}{sa:11.3f}")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
