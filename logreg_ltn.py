"""
Does the symbolic LTN term help a NON-deep model too?

Paired ablation: torch logistic regression on the ~86 HRV features, trained with
    (A) class-weighted cross-entropy only
    (B) CE + lambda * (1 - SatAgg(K))                 [marker axioms, as PGNSF]
    (C) CE + rest axiom on UNLABELED Training-1 rest windows (forall x in Rest: not HighLoad)
    (D) CE + marker axioms + rest axiom               [full symbolic KB]
on the SAME subject-independent folds, per-fold imputation/standardization and axiom
hyperparameters (lambda=0.5, k=2, w_rest=0.5) used by train_compare.py. Same init seed
per fold for all arms, so the only difference is the symbolic loss terms. Rest windows
are restricted to the fold's TRAIN subjects and standardized with train statistics
(leakage-free), exactly as in the deep pipeline.

Run:  python logreg_ltn.py     (CPU, ~a couple of minutes)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             accuracy_score)

from ltn_axioms import build_axiom_indices, ltn_physio_loss

CACHE = "processed_real_dataset_88subj_rawhrv_raw1d.pkl"
CACHE_REST = "processed_real_dataset_88subj_rawhrv_raw1d_rest.pkl"
SEED = 42
N_SPLITS = 10
EPOCHS = 300          # full-batch; linear model converges quickly
LR = 0.05
WEIGHT_DECAY = 1e-4
LTN_WEIGHT = 0.5      # same lambda as PGNSF
LTN_K = 2.0           # same predicate sharpness as PGNSF
REST_WEIGHT = 0.5     # same rest-axiom weight as the deep pipeline


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


def impute_standardize(hrv, base, train_idx):
    """Same as standardize_hrv_per_fold: shared train-only median + scaler."""
    med = np.nanmedian(hrv[train_idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isnan(hrv), med, hrv)
    b = np.where(np.isnan(base), med, base)
    x = np.nan_to_num(x); b = np.nan_to_num(b)
    mu = x[train_idx].mean(axis=0)
    sd = x[train_idx].std(axis=0) + 1e-6
    return (x - mu) / sd, (b - mu) / sd, med, mu, sd


def apply_stats(hrv, base, med, mu, sd):
    """Impute/standardize additional windows with the fold's TRAIN statistics."""
    x = np.where(np.isnan(hrv), med, hrv)
    b = np.where(np.isnan(base), med, base)
    x = np.nan_to_num(x); b = np.nan_to_num(b)
    return (x - mu) / sd, (b - mu) / sd


def train_eval(x, b, y, groups, tr, va, idxs, use_ltn, seed, rest=None):
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], 2)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    xt = torch.tensor(x[tr], dtype=torch.float32)
    bt = torch.tensor(b[tr], dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.long)
    counts = np.bincount(y[tr], minlength=2).astype(np.float64)
    w = torch.tensor(counts.sum() / (2.0 * np.clip(counts, 1.0, None)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    xr = torch.tensor(rest, dtype=torch.float32) if rest is not None else None

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = model(xt)
        loss = crit(logits, yt)
        if use_ltn:
            p_high = torch.softmax(logits, dim=1)[:, 1]
            loss = loss + LTN_WEIGHT * ltn_physio_loss(p_high, xt, bt, idxs, k=LTN_K)
        if xr is not None:
            # Rest axiom: forall x in Rest: not HighLoad(x), pMeanError(p=2) aggregation.
            p_rest = torch.softmax(model(xr), dim=1)[:, 1]
            sat_rest = 1.0 - torch.sqrt(torch.mean(p_rest ** 2) + 1e-12)
            loss = loss + REST_WEIGHT * (1.0 - sat_rest)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(x[va], dtype=torch.float32)), dim=1)[:, 1].numpy()
    pred = (p >= 0.5).astype(int)
    s = groups[va]
    sp = np.array([p[s == u].mean() for u in np.unique(s)])
    sl = np.array([y[va][s == u][0] for u in np.unique(s)])
    return {
        "auroc": roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan,
        "bal_acc": balanced_accuracy_score(y[va], pred),
        "f1_macro": f1_score(y[va], pred, average="macro", zero_division=0),
        "acc": accuracy_score(y[va], pred),
        "ece": ece_eqmass(p, y[va]),
        "s_auroc": roc_auc_score(sl, sp) if len(np.unique(sl)) > 1 else np.nan,
    }


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f"cache '{CACHE}' not found - run train_compare.py once first.")
    print(f"Loading cache '{CACHE}'...")
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    hrv = np.asarray(d["x_hrv"], float)
    base = np.asarray(d["x_baselines"], float)
    y = np.asarray(d["y_labels"])
    groups = np.asarray(d["x_subjects"])
    idxs = build_axiom_indices(d["hrv_feature_names"])
    print(f"windows={len(y)}  features={hrv.shape[1]}  axiom cols={idxs}")

    rest_hrv = rest_base = rest_subj = None
    if os.path.exists(CACHE_REST):
        with open(CACHE_REST, "rb") as f:
            dr = pickle.load(f)
        rest_hrv = np.asarray(dr["x_rest_hrv"], float)
        rest_base = np.asarray(dr["x_rest_baselines"], float)
        rest_subj = np.asarray(dr["x_rest_subjects"])
        print(f"rest windows (unlabeled Training-1): {len(rest_subj)}")
    else:
        print(f"NOTE: '{CACHE_REST}' not found - rest-axiom arms skipped.")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.zeros(len(y)), y, groups=groups))

    arms = [("plain", False, False), ("ltn", True, False)]
    if rest_hrv is not None:
        arms += [("rest", False, True), ("ltn+rest", True, True)]
    res = {a[0]: [] for a in arms}
    for fold, (tr, va) in enumerate(splits, 1):
        x, b, med, mu, sd = impute_standardize(hrv, base, tr)
        xr = None
        if rest_hrv is not None:
            mask = np.isin(rest_subj, np.unique(groups[tr]))   # train subjects only
            xr, _ = apply_stats(rest_hrv[mask], rest_base[mask], med, mu, sd)
        for arm, use_ltn, use_rest in arms:
            res[arm].append(train_eval(x, b, y, groups, tr, va, idxs, use_ltn,
                                       SEED + fold, rest=xr if use_rest else None))
        line = "  ".join(f"{arm}={res[arm][-1]['auroc']:.3f}" for arm, _, _ in arms)
        print(f"  fold {fold:2d}: AUROC {line}")

    print(f"\n{'='*90}\n Torch logistic regression on HRV features - symbolic ablation")
    print(f" (lambda={LTN_WEIGHT}, k={LTN_K}, w_rest={REST_WEIGHT}; "
          f"same folds/imputation as the deep pipeline)\n{'='*90}")
    header = f"{'metric':<10}" + "".join(f"{a:>16}" for a, _, _ in arms)
    print(header)
    for key in ("auroc", "bal_acc", "f1_macro", "acc", "ece", "s_auroc"):
        row = f"{key:<10}"
        for arm, _, _ in arms:
            v = np.array([f[key] for f in res[arm]], float)
            row += f"{np.nanmean(v):8.3f}+/-{np.nanstd(v):5.3f}"
        print(row)
    print(f"{'-'*90}\n Paired Nadeau-Bengio tests (window AUROC unless noted):")
    ref_pairs = [("ltn", "plain"), ("rest", "plain"), ("ltn+rest", "plain"),
                 ("ltn+rest", "ltn"), ("ltn+rest", "rest")]
    for a, bname in ref_pairs:
        if a not in res or bname not in res:
            continue
        for key, tag in (("auroc", "window"), ("s_auroc", "subject")):
            va_ = np.array([f[key] for f in res[a]], float)
            vb = np.array([f[key] for f in res[bname]], float)
            t, p = nb_ttest(va_ - vb)
            sig = " *" if np.isfinite(p) and p < 0.05 else ""
            wins = int(np.nansum(va_ > vb))
            print(f"   {a:>8} vs {bname:<8} [{tag:>7}]: Delta={np.nanmean(va_-vb):+.4f}"
                  f"  t={t:+.3f}  p={p:.4f}  wins {wins}/10{sig}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
