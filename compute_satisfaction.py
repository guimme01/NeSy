"""
Post-hoc axiom satisfaction: how physiologically consistent are the VALIDATION
predictions of each configuration, measured with the SAME LTN knowledge base used
in training (SatAgg over the three axioms, lambda-independent)?

Works for ANY arm -- including models trained WITHOUT the LTN term (backbone,
HRV-only) -- because satisfaction is just a function of (predicted p_high,
standardized HRV, subject baseline). Splits and per-fold standardization are
reproduced exactly (same seed / procedure as train_compare.py).

Usage:  python compute_satisfaction.py
"""

import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from ltn_axioms import build_axiom_indices, ltn_physio_loss

CACHE = "processed_real_dataset_88subj_rawhrv_raw1d.pkl"
SEED = 42
N_SPLITS = 10
K = 2.0

# (backbone, arm label, pickle candidates, which arm inside the pkl)
SOURCES = [
    ("SE-ECG",  "Backbone",     ["plot_data_seecg/plot_data.pkl", "plot_data/plot_data.pkl"], "nn"),
    ("SE-ECG",  "+HRV only",    ["plot_data_seecg_hrv1_ltn0/plot_data.pkl"],                  "nsa"),
    ("SE-ECG",  "+LTN only",    ["plot_data_seecg_hrv0_ltn1/plot_data.pkl"],                  "nsa"),
    ("SE-ECG",  "+HRV+LTN",     ["plot_data_seecg/plot_data.pkl", "plot_data/plot_data.pkl"], "nsa"),
    ("CNN-LSTM", "Backbone",    ["plot_data_cnnlstm/plot_data.pkl"],                          "nn"),
    ("CNN-LSTM", "+HRV only",   ["plot_data_cnnlstm_hrv1_ltn0/plot_data.pkl"],                "nsa"),
    ("CNN-LSTM", "+LTN only",   ["plot_data_cnnlstm_hrv0_ltn1/plot_data.pkl"],                "nsa"),
    ("CNN-LSTM", "+HRV+LTN",    ["plot_data_cnnlstm/plot_data.pkl"],                          "nsa"),
]


def load_first(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def main():
    print(f"Loading cache '{CACHE}' (HRV/baselines/labels/subjects)...")
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    hrv = np.asarray(d["x_hrv"], float)
    base = np.asarray(d["x_baselines"], float)
    y = np.asarray(d["y_labels"])
    groups = np.asarray(d["x_subjects"])
    idxs = build_axiom_indices(d["hrv_feature_names"])
    # free the big arrays we don't need
    d.clear()

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.zeros(len(y)), y, groups=groups))

    # Precompute standardized val-fold HRV/baseline once per fold (train-only stats).
    fold_val = []
    for tr, va in splits:
        med = np.nanmedian(hrv[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        x = np.where(np.isnan(hrv), med, hrv)
        b = np.where(np.isnan(base), med, base)
        x = np.nan_to_num(x); b = np.nan_to_num(b)
        mu = x[tr].mean(axis=0)
        sd = x[tr].std(axis=0) + 1e-6
        fold_val.append(((x[va] - mu) / sd, (b[va] - mu) / sd))

    print(f"\n{'='*62}\n Post-hoc axiom satisfaction on VALIDATION predictions")
    print(f" SatAgg(K) in [0,1]; higher = more physiologically consistent\n{'='*62}")
    print(f"{'backbone':<10}{'configuration':<14}{'SatAgg mean+/-std over folds':>34}")
    cache_results = {}
    for bk, arm_label, paths, arm in SOURCES:
        log = load_first(paths)
        if log is None:
            print(f"{bk:<10}{arm_label:<14}{'(run not available yet)':>34}")
            continue
        sats = []
        for fold_i, f in enumerate(log[arm]):
            p = torch.tensor(np.asarray(f["probs"]), dtype=torch.float32)
            xv, bv = fold_val[fold_i]
            assert len(p) == len(xv), f"fold {fold_i+1}: probs/val size mismatch"
            with torch.no_grad():
                loss = ltn_physio_loss(p, torch.tensor(xv, dtype=torch.float32),
                                       torch.tensor(bv, dtype=torch.float32), idxs, k=K)
            sats.append(1.0 - float(loss))
        sats = np.array(sats)
        cache_results[(bk, arm_label)] = sats
        print(f"{bk:<10}{arm_label:<14}{np.mean(sats):>22.3f} +/- {np.std(sats):.3f}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
