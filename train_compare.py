"""
Subject-Independent Comparison: plain SE-ECG Net (NN) vs NSA-ECG Net (Neuro-Symbolic).

WHY THIS SCRIPT REPLACES train_nsa_real.py
-------------------------------------------
The previous pipeline reported a "within-subject" 10-fold number produced by a
StratifiedKFold over *pooled* windows. In the Meteier et al. (2023) Exp2 dataset the
cognitive-load label (`label_st`: ST = backward counting = high load, NST = monitoring
= low load) is a BETWEEN-SUBJECTS manipulation -- it is constant for every window of a
given subject. A pooled StratifiedKFold therefore puts windows of the same subject in
both train and test, so the model scores high by fingerprinting the individual, not by
recognising cognitive load. Those numbers are effectively proxy results and are dropped.

This script keeps ONLY the valid protocol: subject-independent GroupKFold, where no
subject appears in both train and test. It then trains BOTH models on the *same* folds
so any difference is attributable to the neuro-symbolic components, and reports a
head-to-head comparison (Accuracy / Precision / Recall / F1 / ROC-AUC, mean +/- std).

DATA CORRECTNESS
----------------
The ECG channel and sampling rate are read from each file's header via
`load_physio_ecg` (no hardcoded skiprows / CH2). Label is the real `label_st`
manipulation. No proxy signals or labels are used.

The user runs training/inference; this script only defines and prepares the pipeline.
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
)
from scipy import stats
from typing import List, Tuple, Optional, Dict

from ecg_preprocessing import (
    load_physio_ecg,
    downsample_ecg,
    denoise_ecg,
    detect_r_peaks,
    segment_and_window,
    ecg_to_spectrogram,
    extract_hrv_features,
)
from se_ecg_net import SEECGNet
from nsa_ecg_net import NSAECGNet
from cnn_bilstm_net import CNNBiLSTMNet
from nsa_cnn_bilstm_net import NSACNNBiLSTMNet
from ltn_axioms import build_axiom_indices, ltn_physio_loss, axiom_weights

# ==============================================================================
# CONFIG  (edit these; you run the training)
# ==============================================================================
BASE_DATASET_DIR = "Exp2/Exp2"
# Which backbone to run the Backbone-vs-Backbone+NSA ablation on. The neuro-symbolic
# layer (HRV branch + LTN axioms) is IDENTICAL for both; only the neural backbone changes,
# which is exactly the point: it shows the symbolic gain is not architecture-specific.
#   "seecg"    -> SE-ECG Net (2D spectrogram CNN, the paper model).
#   "cnnlstm"  -> CNN-BiLSTM (raw 1D ECG waveform).
# Run the script once per backbone (results/figures land in a backbone-specific folder).
BACKBONE = "cnnlstm"                              # "seecg" | "cnnlstm"
SUBJECTS: Optional[List[str]] = None            # None = auto-discover ALL subjects
MAX_SUBJECTS: Optional[int] = None              # None = use ALL subjects (slow preprocessing)
N_SPLITS = 10                                   # StratifiedGroupKFold folds (paper uses 10)
EPOCHS = 8
BATCH_SIZE = 16
LR = 0.001
LR_DROP_EPOCH = 5                               # 0.001 -> 0.0001 after this epoch
SEED = 42

# --- Neuro-Symbolic layer toggles (for the PAPER+NSA arm) ---
USE_HRV_BRANCH = True     # hybrid HRV features (the useful part; HRV is the strong signal)
USE_RULE_ENGINE = False   # physiological baseline-shift rule (on STANDARDIZED HRV)
USE_HANDOVER = False      # causal handover bias: label-INDEPENDENT -> hurts accuracy by design.
                          # It is a safety-conservatism feature, not an accuracy feature.
                          # Keep OFF for the head-to-head; flip ON only to study safety behaviour.

# --- Logic Tensor Network (true neuro-symbolic) constraints, applied to the NSA arm only ---
USE_LTN = True            # add LTN physiological axioms to the NSA training loss
# lambda / k / KB composition, overridable via env for the LTN-improvement sweep:
#   NSA_LTN_WEIGHT: lambda (default 0.5, the paper value)
#   NSA_LTN_K     : predicate sharpness (default 2.0, the paper value)
#   NSA_KB        : 'old' = paper KB (3 forward axioms, unweighted)
#                   'w'   = forward axioms + train-validated weights (silences RMSSD)
#                   'bi'  = forward + converse axiom, unweighted
#                   'biw' = forward + converse + weights (default)
LTN_WEIGHT = float(os.environ.get("NSA_LTN_WEIGHT", "0.5"))
LTN_K = float(os.environ.get("NSA_LTN_K", "2.0"))
KB_MODE = os.environ.get("NSA_KB", "biw")
assert KB_MODE in ("old", "w", "bi", "biw"), f"bad NSA_KB '{KB_MODE}'"

# --- LTN REST axiom (NSA_REST=1): forall x in Rest: NOT HighLoad(x). ---
# Grounded on UNLABELED windows from each training subject's Training-1 resting
# period: protocol knowledge ("at rest, load is low") that gives ST subjects the
# low-load examples the between-subjects CE labels cannot provide. Architecture,
# CE, folds and evaluation are untouched; only the NSA arm's LTN loss sees them.
REST_AXIOM = os.environ.get("NSA_REST", "0") == "1"
REST_WEIGHT = float(os.environ.get("NSA_REST_WEIGHT", str(LTN_WEIGHT)))

# --- Optional CLI overrides (factorial ablation without editing the file) ---
#   python train_compare.py                    -> config above, unchanged
#   python train_compare.py <backbone>         -> seecg | cnnlstm
#   python train_compare.py <backbone> <arm>   -> arm: hrv (HRV branch only)
#                                                      ltn (LTN axioms only)
#                                                      both (full PGNSF)
if len(sys.argv) > 1:
    if sys.argv[1] not in ("seecg", "cnnlstm"):
        raise SystemExit(f"usage: python train_compare.py [seecg|cnnlstm] [hrv|ltn|both]")
    BACKBONE = sys.argv[1]
if len(sys.argv) > 2:
    if sys.argv[2] not in ("hrv", "ltn", "both"):
        raise SystemExit(f"usage: python train_compare.py [seecg|cnnlstm] [hrv|ltn|both]")
    USE_HRV_BRANCH = sys.argv[2] in ("hrv", "both")
    USE_LTN = sys.argv[2] in ("ltn", "both")

# --- Dev mode: NSA_DEV_FOLDS selects a subset of folds (identical splits and
# per-fold seeds as a full run, so fold i reproduces exactly). Accepts a count
# ("3" = first three), a range ("4-6") or a comma list ("2,5,9"). For KB iteration.
def _parse_dev_folds(spec):
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    n = int(spec)
    return list(range(1, n + 1)) if n > 0 else None

DEV_FOLDS = _parse_dev_folds(os.environ.get("NSA_DEV_FOLDS", ""))

# --- Label-scarcity mode: NSA_LABEL_FRACTION=f (0<f<1) keeps the CE labels of only a
# stratified fraction f of TRAINING subjects (seeded per fold); every other training
# window becomes unlabeled (-1, ignored by the cross-entropy). The LTN axioms need no
# labels and still apply to ALL training windows, so the comparison probes whether the
# symbolic physiological knowledge compensates for scarce supervision. Validation
# labels are never touched.
LABEL_FRACTION = float(os.environ.get("NSA_LABEL_FRACTION", "1.0"))

# --- Plot logging ---
# Backbone-specific so the two backbones do not overwrite each other; ablation arms
# (HRV-only / LTN-only) get their own tagged folder so the paper's full-PGNSF
# plot_data is never clobbered by an ablation run. Dev runs get their own suffix too.
PLOT_DIR = f"plot_data_{BACKBONE}"    # per-fold probabilities and per-epoch loss curves
if not (USE_HRV_BRANCH and USE_LTN):
    PLOT_DIR += f"_hrv{int(USE_HRV_BRANCH)}_ltn{int(USE_LTN)}"
if DEV_FOLDS:
    PLOT_DIR += "_dev" + "-".join(str(i) for i in DEV_FOLDS)
if LABEL_FRACTION < 1.0:
    PLOT_DIR += f"_lab{int(round(LABEL_FRACTION * 100))}"
if REST_AXIOM:
    PLOT_DIR += "_rest"
# Optional experiment tag (env NSA_TAG), e.g. "_kb2" for the revised weighted KB,
# so new-KB runs never overwrite the submitted paper's plot_data folders.
PLOT_DIR += os.environ.get("NSA_TAG", "")

np.random.seed(SEED)
torch.manual_seed(SEED)
# Full determinism: without these, cuDNN convolutions are non-deterministic and two runs
# differ by ~0.005-0.02 AUROC, which would desynchronize the tables from the figures.
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. SUBJECT DISCOVERY
# ==============================================================================

def discover_subjects(base_dir: str) -> List[str]:
    """Lists subjects that have BOTH a physio file and a markers file."""
    txt_dir = os.path.join(base_dir, "Raw", "Physio", "Txt")
    subs = []
    for path in sorted(glob.glob(os.path.join(txt_dir, "*.txt"))):
        name = os.path.basename(path)
        if name.endswith("-markers.txt"):
            continue
        code = name[:-4]
        if not re.match(r"^N?ST\d+$", code):
            continue
        if os.path.exists(os.path.join(txt_dir, f"{code}-markers.txt")):
            subs.append(code)
    # Interleave NST / ST so a capped subset stays class-balanced.
    nst = [s for s in subs if s.startswith("NST")]
    st = [s for s in subs if s.startswith("ST")]
    interleaved = [x for pair in zip(nst, st) for x in pair]
    interleaved += nst[len(st):] + st[len(nst):]
    return interleaved


# ==============================================================================
# 2. REAL DATASET PIPELINE: PARSING, PREPROCESSING & CACHING
# ==============================================================================

def parse_markers_file(markers_path: str) -> Dict[str, float]:
    """Parses Training 1 (baseline) and Driving start/end timestamps (seconds)."""
    markers = {}
    with open(markers_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for key, pat in [
        ("t1_start", r"([\d.]+)\s+Training 1 Start"),
        ("t1_end", r"([\d.]+)\s+Training 1 End"),
        ("drv_start", r"([\d.]+)\s+Driving Start"),
        ("drv_end", r"([\d.]+)\s+Driving End"),
    ]:
        m = re.search(pat, content)
        if m:
            markers[key] = float(m.group(1))
    return markers


def load_subject_driving_events(driving_path: str) -> Tuple[List[float], float]:
    """Reads obstacle trigger times (s) and the first driving-log timestamp (s)."""
    df = pd.read_csv(driving_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    start_log_time = float(df["Time"].iloc[0])
    obstacle_rows = df[df["Obstacles"].str.startswith("TriggeredObs", na=False)]
    trigger_times = obstacle_rows["Time"].tolist()
    return trigger_times, start_log_time


def preprocess_real_subject_data(
    subject_id: str,
    base_dir: str,
    target_sr: float = 360.0,
) -> Optional[Dict]:
    """Loads, processes, segments and extracts features for a single subject."""
    physio_txt_dir = os.path.join(base_dir, "Raw", "Physio", "Txt")
    driving_dir = os.path.join(base_dir, "Raw", "Driving")

    physio_path = os.path.join(physio_txt_dir, f"{subject_id}.txt")
    markers_path = os.path.join(physio_txt_dir, f"{subject_id}-markers.txt")
    driving_path = os.path.join(driving_dir, f"{subject_id}.txt")

    if not (os.path.exists(physio_path) and os.path.exists(markers_path)):
        return None

    label = 1 if subject_id.startswith("ST") else 0  # ST=high load, NST=low load

    markers = parse_markers_file(markers_path)
    if "t1_start" not in markers or "drv_start" not in markers:
        return None

    trigger_times, start_log_time = [], 0.0
    if os.path.exists(driving_path):
        try:
            trigger_times, start_log_time = load_subject_driving_events(driving_path)
        except Exception:
            trigger_times, start_log_time = [], 0.0

    # Robust ECG load (channel + header detected from the file, not hardcoded).
    print(f"[{subject_id}] Loading raw ECG signal...")
    raw_time_sec, raw_ecg, orig_sr = load_physio_ecg(physio_path)

    baseline_mask = (raw_time_sec >= markers["t1_start"]) & (raw_time_sec <= markers["t1_end"])
    driving_mask = (raw_time_sec >= markers["drv_start"]) & (raw_time_sec <= markers["drv_end"])
    baseline_ecg = raw_ecg[baseline_mask]
    driving_ecg = raw_ecg[driving_mask]

    # --- Baseline period (for the symbolic baseline-shift rule) ---
    base_down = downsample_ecg(baseline_ecg, orig_sr=orig_sr, target_sr=target_sr)
    base_clean = denoise_ecg(base_down, sampling_rate=target_sr, highpass_cutoff=0.5)
    base_peaks, _ = detect_r_peaks(base_clean, sampling_rate=target_sr)
    if len(base_peaks) < 3:
        template = nk.hrv({"ECG_R_Peaks": np.array([100, 400, 700])}, sampling_rate=360).columns
        baseline_hrv_features = pd.Series(0.0, index=template)
    else:
        baseline_hrv_features = extract_hrv_features(base_clean, fs=target_sr, impute_strategy="mean")

    # --- Rest windows from the Training-1 baseline period (LTN rest axiom) ---
    # Protocol knowledge: during Training-1 the driver is AT REST, so cognitive load is
    # low for EVERY subject, ST included. These windows carry no ST/NST label and are
    # consumed ONLY by the label-free LTN term (forall x in Rest: NOT HighLoad(x)),
    # giving high-load subjects the low-load examples the between-subjects CE lacks.
    rest_windows = np.empty((0, int(60.0 * target_sr)), dtype=np.float32)
    if len(base_peaks) >= 3:
        rest_windows, _ = segment_and_window(
            base_clean, base_peaks, sampling_rate=target_sr,
            window_duration_sec=60.0, r_peak_step=10)
    if len(rest_windows) > 0:
        rest_spectrograms = ecg_to_spectrogram(rest_windows, fs=target_sr,
                                               frame_duration_ms=350.0, norm_type="minmax")
        rest_hrv_df = pd.DataFrame(
            [extract_hrv_features(w, fs=target_sr, impute_strategy="none")
             for w in rest_windows])
    else:
        rest_spectrograms, rest_hrv_df = None, None

    # --- Driving period (the classification target windows) ---
    drv_down = downsample_ecg(driving_ecg, orig_sr=orig_sr, target_sr=target_sr)
    drv_clean = denoise_ecg(drv_down, sampling_rate=target_sr, highpass_cutoff=0.5)
    drv_peaks, _ = detect_r_peaks(drv_clean, sampling_rate=target_sr)
    if len(drv_peaks) < 3:
        print(f"[{subject_id}] Insufficient R-peaks in driving ECG; skipping.")
        return None

    windows, center_peaks = segment_and_window(
        drv_clean, drv_peaks, sampling_rate=target_sr,
        window_duration_sec=60.0, r_peak_step=10,
    )
    if len(windows) == 0:
        return None
    print(f"[{subject_id}] Extracted {len(windows)} windows (label={label}).")

    spectrograms = ecg_to_spectrogram(windows, fs=target_sr, frame_duration_ms=350.0, norm_type="minmax")

    subject_hrv_list, handover_contexts = [], []
    for idx, win in enumerate(windows):
        subject_hrv_list.append(extract_hrv_features(win, fs=target_sr, impute_strategy="none"))

        # Handover-imminent context from the driving log (10s before .. 3s after a trigger).
        center_peak = center_peaks[idx]
        t_log = start_log_time + (center_peak / target_sr)
        is_imminent = 0.0
        for trig in trigger_times:
            if (trig - 10.0) <= t_log <= (trig + 3.0):
                is_imminent = 1.0
                break
        handover_contexts.append(is_imminent)

    return {
        "spectrograms": spectrograms,
        "windows": np.asarray(windows, dtype=np.float32),  # RAW 1D ECG (N, 21600) for CNN-BiLSTM
        "hrv_df": pd.DataFrame(subject_hrv_list),
        "baseline_hrv": baseline_hrv_features,
        "handover": np.array(handover_contexts, dtype=np.float32),
        "labels": np.full(len(windows), label, dtype=np.int64),
        "subject_ids": [subject_id] * len(windows),
        # Rest windows (Training-1): unlabeled, LTN-only. None if the baseline is unusable.
        "rest_windows": np.asarray(rest_windows, dtype=np.float32),
        "rest_spectrograms": rest_spectrograms,
        "rest_hrv_df": rest_hrv_df,
    }


def load_and_cache_dataset(base_dir: str, subjects: List[str], cache_path: str) -> Dict:
    """Processes all subjects, applies dataset-level HRV imputation, caches to disk."""
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from '{cache_path}' "
              f"(delete it if you changed SUBJECTS)...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("\n--- Processing physiological data ---")
    all_specs, all_windows, all_hrv_dfs = [], [], []
    all_baseline_series, all_nwins = [], []
    all_handovers, all_labels, all_subjects = [], [], []
    r_specs, r_raw, r_hrv_dfs, r_base_series, r_counts, r_subjects = [], [], [], [], [], []

    for sub in subjects:
        try:
            res = preprocess_real_subject_data(sub, base_dir)
        except Exception as e:
            print(f"Failed processing subject {sub}: {e}")
            continue
        if res is None:
            continue
        n_wins = len(res["spectrograms"])
        all_specs.append(res["spectrograms"])
        all_windows.append(res["windows"])
        all_hrv_dfs.append(res["hrv_df"])
        # Rest windows (unlabeled, LTN-only); subjects with unusable baselines are skipped.
        if res.get("rest_spectrograms") is not None and len(res["rest_windows"]) > 0:
            r_specs.append(res["rest_spectrograms"])
            r_raw.append(res["rest_windows"])
            r_hrv_dfs.append(res["rest_hrv_df"])
            r_base_series.append(res["baseline_hrv"])
            r_counts.append(len(res["rest_windows"]))
            r_subjects.extend([sub] * len(res["rest_windows"]))
        # Keep the baseline as a Series; align its columns AFTER the window feature
        # set is known (baseline is a 5-min signal -> more HRV cols than 60s windows).
        all_baseline_series.append(res["baseline_hrv"])
        all_nwins.append(n_wins)
        all_handovers.append(res["handover"])
        all_labels.append(res["labels"])
        all_subjects.extend(res["subject_ids"])

    if not all_specs:
        raise ValueError("No subjects processed. Check paths and markers.")

    x_spectrograms = np.concatenate(all_specs, axis=0)
    x_windows = np.concatenate(all_windows, axis=0).astype(np.float32)  # RAW 1D ECG for CNN-BiLSTM
    x_handovers = np.concatenate(all_handovers, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)
    x_subjects = np.array(all_subjects)

    # NO dataset-level imputation here: computing medians over ALL subjects (train+val)
    # before the split leaked validation info. Missing values are kept as NaN and imputed
    # PER-FOLD with TRAIN-only medians in standardize_hrv_per_fold (leakage-free).
    merged_hrv = pd.concat(all_hrv_dfs, axis=0, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    hrv_feature_names = list(merged_hrv.columns)
    x_hrv = merged_hrv.values.astype(np.float32)  # RAW (contains NaN)

    # Align each subject's baseline Series to the WINDOW feature columns (missing -> NaN);
    # kept RAW here and imputed per-fold together with the current-window HRV.
    baseline_rows = []
    for series, n_wins in zip(all_baseline_series, all_nwins):
        row = series.reindex(hrv_feature_names).to_numpy(dtype=np.float32)
        baseline_rows.append(np.tile(row, (n_wins, 1)))
    df_base = pd.DataFrame(np.concatenate(baseline_rows, axis=0), columns=hrv_feature_names)
    df_base = df_base.replace([np.inf, -np.inf], np.nan)
    x_baselines = df_base.values.astype(np.float32)  # RAW (contains NaN)

    # --- Rest windows: align HRV columns and tile the subject baseline (RAW, NaN kept) ---
    if r_specs:
        x_rest_spectrograms = np.concatenate(r_specs, axis=0)
        x_rest_windows = np.concatenate(r_raw, axis=0).astype(np.float32)
        rest_hrv = pd.concat(r_hrv_dfs, axis=0, ignore_index=True)
        rest_hrv = rest_hrv.reindex(columns=hrv_feature_names)
        x_rest_hrv = rest_hrv.replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        x_rest_baselines = np.concatenate(
            [np.tile(s.reindex(hrv_feature_names).to_numpy(dtype=np.float32), (n, 1))
             for s, n in zip(r_base_series, r_counts)], axis=0)
        x_rest_subjects = np.array(r_subjects)
        print(f"Rest windows extracted: {len(x_rest_subjects)} "
              f"from {len(r_counts)} subjects (Training-1 baselines).")
    else:
        x_rest_spectrograms = x_rest_windows = x_rest_hrv = x_rest_baselines = None
        x_rest_subjects = np.array([])

    dataset_dict = {
        "x_spectrograms": x_spectrograms,
        "x_windows": x_windows,
        "x_rest_spectrograms": x_rest_spectrograms,
        "x_rest_windows": x_rest_windows,
        "x_rest_hrv": x_rest_hrv,
        "x_rest_baselines": x_rest_baselines,
        "x_rest_subjects": x_rest_subjects,
        "x_hrv": x_hrv,
        "x_baselines": x_baselines,
        "x_handovers": x_handovers,
        "y_labels": y_labels,
        "x_subjects": x_subjects,
        "hrv_feature_names": hrv_feature_names,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(dataset_dict, f)
    print(f"Cached processed dataset to '{cache_path}'.\n")
    return dataset_dict


# ==============================================================================
# 3. PYTORCH DATASET
# ==============================================================================

class RealECGDataset(Dataset):
    def __init__(self, data: Dict):
        self.spectrograms = torch.tensor(data["x_spectrograms"], dtype=torch.float32)
        # RAW 1D ECG window (kept as numpy to avoid a ~1GB always-resident tensor; the
        # raw is sliced to a tensor per item only when the CNN-BiLSTM backbone reads it).
        self.windows = np.asarray(data["x_windows"], dtype=np.float32)
        self.hrv = torch.tensor(data["x_hrv"], dtype=torch.float32)
        self.baselines = torch.tensor(data["x_baselines"], dtype=torch.float32)
        self.handovers = torch.tensor(data["x_handovers"], dtype=torch.float32)
        self.labels = torch.tensor(data["y_labels"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.spectrograms[idx], torch.from_numpy(self.windows[idx]),
                self.hrv[idx], self.baselines[idx], self.handovers[idx], self.labels[idx])


# ==============================================================================
# 4. MODEL-AGNOSTIC FORWARD / TRAIN / EVAL
# ==============================================================================

# Labels adapt to the active backbone AND ablation arm so logs read correctly.
_BACKBONE_NAME = {"seecg": "SE-ECG", "cnnlstm": "CNN-LSTM"}
_ARM = ("+NSA" if (USE_HRV_BRANCH and USE_LTN)
        else "+HRV" if USE_HRV_BRANCH
        else "+LTN" if USE_LTN
        else "+none")
MODEL_LABEL = {"nn": _BACKBONE_NAME.get(BACKBONE, BACKBONE),
               "nsa": _BACKBONE_NAME.get(BACKBONE, BACKBONE) + _ARM}


def build_model(model_type: str, data: Dict) -> nn.Module:
    """
    'nn'  -> the plain neural backbone selected by BACKBONE.
    'nsa' -> the SAME backbone, wrapped with the additive Neuro-Symbolic layer.

    The backbone is constructed FIRST in both branches, so with the same per-fold seed
    it gets identical initial weights in both models; only the extra NSA parameters differ.
    """
    if BACKBONE == "seecg":
        base = SEECGNet(in_channels=1, num_classes=2, conv_channels=(32, 64, 128))
        nsa_cls = NSAECGNet
    elif BACKBONE == "cnnlstm":
        base = CNNBiLSTMNet(in_channels=1, num_classes=2, conv_channels=(32, 64, 128))
        nsa_cls = NSACNNBiLSTMNet
    else:
        raise ValueError(f"Unknown BACKBONE '{BACKBONE}'")

    if model_type == "nn":
        return base
    elif model_type == "nsa":
        return nsa_cls(
            base_model=base,
            num_hrv_features=data["x_hrv"].shape[1],
            hrv_feature_names=data["hrv_feature_names"],
            handover_bias=1.5,
            use_hrv_branch=USE_HRV_BRANCH,
            use_rule_engine=USE_RULE_ENGINE,
            use_handover=USE_HANDOVER,
        )
    raise ValueError(f"Unknown model_type '{model_type}'")


def forward_logits(model, batch, model_type, device):
    """Unified forward returning logits for either model and either backbone."""
    specs, windows, hrv, baselines, handovers, labels = [t.to(device) for t in batch]
    # The backbone reads the spectrogram (seecg) or the raw 1D ECG window (cnnlstm).
    x_backbone = specs if BACKBONE == "seecg" else windows.unsqueeze(1)  # (B,1,T) for Conv1d
    if model_type == "nn":
        logits = model(x_backbone, return_logits=True)
    else:
        logits = model(x_backbone, x_hrv=hrv, baseline_hrv=baselines,
                       handover_imminent=handovers, return_logits=True)
    return logits, labels


def train_one_epoch(model, loader, criterion, optimizer, device, model_type, ltn_ctx=None):
    """
    Returns (mean_total_loss, accuracy, mean_CE, mean_satisfaction).
    mean_satisfaction is SatAgg(K) in [0,1] (NaN when the LTN term is not applied),
    logged so the CE and the logical-satisfaction curves can be plotted side by side.
    """
    model.train()
    running, correct, total = 0.0, 0, 0
    running_ce, running_sat, n_sat = 0.0, 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        logits, labels = forward_logits(model, batch, model_type, device)
        if (labels >= 0).any():
            ce = criterion(logits, labels)
        else:
            # Batch with no labeled samples (label-scarcity mode): no CE, LTN only.
            ce = logits.sum() * 0.0
        loss = ce
        # Neuro-symbolic term: physiological LTN axioms, NSA arm only (PAPER untouched).
        if ltn_ctx is not None and model_type == "nsa":
            # batch = (specs, windows, hrv, baselines, handovers, labels)
            hrv = batch[2].to(device)
            baseline = batch[3].to(device)
            p_high = torch.softmax(logits, dim=1)[:, 1]
            ltn_term = ltn_physio_loss(p_high, hrv, baseline, ltn_ctx["idxs"],
                                       k=ltn_ctx.get("k", 2.0),
                                       bidirectional=ltn_ctx.get("bidirectional", True),
                                       weights=ltn_ctx.get("weights"))
            loss = loss + ltn_ctx["weight"] * ltn_term

            # REST axiom: forall x in Rest: NOT HighLoad(x), on a random minibatch of
            # UNLABELED Training-1 rest windows (train subjects only). Forall via
            # pMeanError (p=2) on the violation p_high, matching the KB's aggregator.
            rest = ltn_ctx.get("rest")
            if rest is not None and rest["x"].shape[0] > 0:
                ridx = torch.randint(0, rest["x"].shape[0], (labels.size(0),))
                r_logits = model(rest["x"][ridx].to(device),
                                 x_hrv=rest["hrv"][ridx].to(device),
                                 baseline_hrv=rest["base"][ridx].to(device),
                                 return_logits=True)
                p_rest = torch.softmax(r_logits, dim=1)[:, 1]
                sat_rest = 1.0 - torch.sqrt(torch.mean(p_rest ** 2) + 1e-12)
                loss = loss + rest["weight"] * (1.0 - sat_rest)
            running_sat += float(1.0 - ltn_term.detach()) * labels.size(0)  # satisfaction
            n_sat += labels.size(0)
        loss.backward()
        optimizer.step()
        running += loss.item() * labels.size(0)
        running_ce += float(ce.detach()) * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    mean_sat = (running_sat / n_sat) if n_sat > 0 else float("nan")
    return running / total, correct / total, running_ce / total, mean_sat


@torch.no_grad()
def evaluate(model, loader, device, model_type, subject_ids=None):
    model.eval()
    preds, probs, targets = [], [], []
    for batch in loader:
        logits, labels = forward_logits(model, batch, model_type, device)
        p = torch.softmax(logits, dim=1)
        preds.extend(logits.argmax(1).cpu().numpy())
        probs.extend(p[:, 1].cpu().numpy())
        targets.extend(labels.cpu().numpy())
    preds, probs, targets = np.array(preds), np.array(probs), np.array(targets)
    # SUPPLEMENTARY (additive): return the subject ID for each validation window.
    # The val loader uses shuffle=False, so window order matches the passed val indices.
    # The three existing outputs (preds/probs/targets) are unchanged.
    subj = None
    if subject_ids is not None:
        subj = np.asarray(subject_ids)
        assert len(subj) == len(targets), "subject_ids must align 1:1 with val windows"
    return preds, probs, targets, subj


def aggregate_by_subject(labels, probs, subject_ids):
    """
    SUPPLEMENTARY subject-level aggregation. For each subject, average all its window
    probabilities into one subject-level probability, and keep its single ground-truth
    label. Returns (subject_probs, subject_labels). Does not touch window-level metrics.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    subject_ids = np.asarray(subject_ids)
    subj_probs, subj_labels = [], []
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        subj_probs.append(probs[mask].mean())   # mean window probability for this subject
        subj_labels.append(labels[mask][0])      # one label per subject (constant within subject)
    return np.array(subj_probs), np.array(subj_labels)


def expected_calibration_error(probs, targets, n_bins: int = 10) -> float:
    """
    Equal-MASS (quantile) ECE for the positive-class probability: bins hold equal
    numbers of samples, and each bin's |mean predicted P(y=1) - empirical P(y=1)| is
    averaged, weighted by bin size. Lower is better (0 = perfectly calibrated).
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n = len(probs)
    if n == 0 or len(np.unique(targets)) < 2:
        return float("nan")
    order = np.argsort(probs)
    p, y = probs[order], targets[order]
    ece = 0.0
    for idx in np.array_split(np.arange(n), min(n_bins, n)):
        if len(idx) == 0:
            continue
        ece += (len(idx) / n) * abs(y[idx].mean() - p[idx].mean())
    return ece


def corrected_paired_ttest(diffs) -> Tuple[float, float]:
    """
    Nadeau & Bengio (2003) corrected resampled paired t-test on per-fold differences.
    Accounts for train/test overlap across k folds (test frac 1/k, train frac (k-1)/k
    -> variance correction factor 1/k + 1/(k-1)). Returns (t_stat, p_value, two-sided).
    """
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    k = len(d)
    if k < 2:
        return float("nan"), float("nan")
    mean_d, var_d = d.mean(), d.var(ddof=1)
    if var_d == 0:
        return (float("inf") if mean_d != 0 else 0.0), (0.0 if mean_d != 0 else 1.0)
    correction = 1.0 / k + 1.0 / (k - 1)
    t = mean_d / np.sqrt(correction * var_d)
    p = 2.0 * stats.t.sf(abs(t), df=k - 1)
    return t, p


def metrics(targets, preds, probs) -> Dict[str, float]:
    """
    Balance-robust, SYMMETRIC metrics. The previous positive-class-only precision/recall/F1
    (sklearn default pos_label=1) measured only the ST/high-load class and was confounded by
    each fold's class balance and the model's tendency to over-predict positives. We report
    instead: balanced accuracy (mean per-class recall), macro-F1 (both classes averaged),
    and per-class F1 (NST=0, ST=1). AUROC/ECE are threshold-free.
    """
    has_both = len(np.unique(targets)) > 1
    out = {
        "auroc": roc_auc_score(targets, probs) if has_both else float("nan"),
        "ece": expected_calibration_error(probs, targets),
        "bal_acc": balanced_accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "f1_neg": f1_score(targets, preds, pos_label=0, zero_division=0),
        "f1_pos": f1_score(targets, preds, pos_label=1, zero_division=0),
        "acc": accuracy_score(targets, preds),
    }
    return out


# ==============================================================================
# 5. SUBJECT-INDEPENDENT CROSS-VALIDATION (both models, same folds)
# ==============================================================================

def standardize_hrv_per_fold(data: Dict, train_idx: np.ndarray) -> Dict:
    """
    Returns a shallow copy of `data` with HRV (and baseline HRV) IMPUTED and z-scored
    using ONLY the training-fold statistics (no leakage). Steps, all train-only:
      1. Impute missing (NaN) HRV with TRAIN-fold medians (closes the pre-split leak).
      2. Z-score with TRAIN-fold mean/std.
    Current-window HRV and the baseline share the SAME imputation+scaler so the rule
    engine compares like with like. Spectrograms/labels/etc. pass through by reference.
    """
    hrv = data["x_hrv"]
    base = data["x_baselines"]

    # 1. TRAIN-only median imputation (nanmedian ignores missing cells).
    train_median = np.nanmedian(hrv[train_idx], axis=0)
    train_median = np.where(np.isfinite(train_median), train_median, 0.0)  # all-NaN col -> 0
    hrv_imp = np.where(np.isnan(hrv), train_median, hrv)
    base_imp = np.where(np.isnan(base), train_median, base)
    hrv_imp = np.nan_to_num(hrv_imp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    base_imp = np.nan_to_num(base_imp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 2. TRAIN-only standardization.
    mu = hrv_imp[train_idx].mean(axis=0)
    sd = hrv_imp[train_idx].std(axis=0) + 1e-6
    scaled = dict(data)
    scaled["x_hrv"] = ((hrv_imp - mu) / sd).astype(np.float32)
    scaled["x_baselines"] = ((base_imp - mu) / sd).astype(np.float32)
    return scaled


def run_comparison(data: Dict, n_splits: int, epochs: int) -> Dict[str, List[Dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_subjects = len(np.unique(data["x_subjects"]))
    n_splits = max(2, min(n_splits, n_subjects))
    # StratifiedGroupKFold: subject-independent AND class-balanced per fold, so a
    # validation fold never ends up all-NST or all-ST (which gave auroc=nan before).
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(data["x_spectrograms"], data["y_labels"], groups=data["x_subjects"]))
    # Keep the ORIGINAL fold number with each split so a dev subset reproduces the
    # exact per-fold seeds (torch.manual_seed(SEED + fold)) of a full run.
    fold_items = list(enumerate(splits, 1))
    if DEV_FOLDS:
        fold_items = [fold_items[i - 1] for i in DEV_FOLDS]
        print(f"\n*** DEV MODE: running folds {DEV_FOLDS} of {n_splits} ***")

    # LTN context (axiom indices, lambda, k, KB composition), NSA arm only.
    ltn_ctx = None
    if USE_LTN:
        ltn_ctx = {"idxs": build_axiom_indices(data["hrv_feature_names"]),
                   "weight": LTN_WEIGHT, "k": LTN_K,
                   "bidirectional": KB_MODE in ("bi", "biw"),
                   "use_weights": KB_MODE in ("w", "biw")}

    print(f"\n{'='*60}\n Subject-independent {n_splits}-fold StratifiedGroupKFold")
    print(f" Subjects: {n_subjects} | Windows: {len(data['y_labels'])} | Device: {device}")
    print(f" NSA layer: hrv_branch={USE_HRV_BRANCH} rule={USE_RULE_ENGINE} "
          f"handover={USE_HANDOVER} | LTN={USE_LTN} "
          f"(KB={KB_MODE}, lambda={LTN_WEIGHT}, k={LTN_K})\n{'='*60}")

    results = {"nn": [], "nsa": []}
    # SUPPLEMENTARY subject-level results (kept SEPARATE from window-level `results`).
    subject_results = {"nn": [], "nsa": []}
    # Raw per-fold data for the figures (probabilities, labels, loss curves).
    plot_log = {"nn": [], "nsa": []}
    for fold, (train_idx, val_idx) in fold_items:
        train_subj = set(data["x_subjects"][train_idx])
        val_subj = set(data["x_subjects"][val_idx])
        assert train_subj.isdisjoint(val_subj), "LEAKAGE: subject in both train and val!"
        val_labels = data["y_labels"][val_idx]
        print(f"\n--- Fold {fold}/{n_splits} | val subjects: {sorted(val_subj)} "
              f"| val class balance: {np.bincount(val_labels, minlength=2).tolist()} ---")

        # Per-fold HRV standardization (train stats only) -> stable hybrid branch.
        fold_scaled = standardize_hrv_per_fold(data, train_idx)

        # Label-scarcity: keep CE labels for a stratified fraction of TRAIN subjects
        # only; other training windows get label -1 (ignored by the CE, still used by
        # the label-free LTN term). Validation labels are untouched.
        if LABEL_FRACTION < 1.0:
            rng = np.random.RandomState(SEED + fold)
            tr_sub = data["x_subjects"][train_idx]
            tr_lab = data["y_labels"][train_idx]
            labeled = set()
            for cls in (0, 1):
                subs = sorted(set(tr_sub[tr_lab == cls]))
                n_keep = max(1, int(round(LABEL_FRACTION * len(subs))))
                labeled.update(rng.choice(subs, size=n_keep, replace=False))
            unlabeled_train = np.zeros(len(data["y_labels"]), dtype=bool)
            unlabeled_train[train_idx] = np.array([s not in labeled for s in tr_sub])
            y_masked = data["y_labels"].copy()
            y_masked[unlabeled_train] = -1
            fold_scaled = dict(fold_scaled)
            fold_scaled["y_labels"] = y_masked
            print(f"  label-scarcity: {len(labeled)}/{len(set(tr_sub))} train subjects "
                  f"labeled ({int(LABEL_FRACTION*100)}%), "
                  f"{int(unlabeled_train.sum())} unlabeled windows")

        fold_dataset = RealECGDataset(fold_scaled)

        # Empirical validation of the LTN knowledge base on the TRAIN fold only:
        # each axiom is weighted by how much signal its predicate carries here
        # (w = clip(2*(train AUROC - 0.5), 0, 1); a data-refuted axiom is silenced).
        if ltn_ctx is not None and ltn_ctx["use_weights"]:
            # Under label scarcity, validate the KB on the LABELED train windows only
            # (fold_scaled labels equal the full labels when LABEL_FRACTION == 1.0).
            lab_tr = fold_scaled["y_labels"][train_idx]
            sel = lab_tr >= 0
            ltn_ctx["weights"] = axiom_weights(
                fold_scaled["x_hrv"][train_idx][sel],
                fold_scaled["x_baselines"][train_idx][sel],
                lab_tr[sel], ltn_ctx["idxs"], k=ltn_ctx["k"])
            print(f"  LTN axiom weights (train-validated): "
                  f"{ {k: round(v, 2) for k, v in ltn_ctx['weights'].items()} }")
        elif ltn_ctx is not None:
            ltn_ctx["weights"] = None    # unweighted KB ('old' / 'bi' modes)

        # REST axiom: unlabeled Training-1 windows of TRAIN subjects only, standardized
        # with the fold's TRAIN statistics (identical imputation+scaler as the windows).
        if ltn_ctx is not None and REST_AXIOM:
            if data.get("x_rest_hrv") is None:
                raise SystemExit("NSA_REST=1 requires the '_rest' cache "
                                 "(rerun so preprocessing regenerates it).")
            med = np.nanmedian(data["x_hrv"][train_idx], axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            imp = np.nan_to_num(np.where(np.isnan(data["x_hrv"]), med, data["x_hrv"]))
            mu = imp[train_idx].mean(axis=0)
            sd = imp[train_idx].std(axis=0) + 1e-6
            rmask = np.isin(data["x_rest_subjects"], list(train_subj))
            rh = np.nan_to_num(np.where(np.isnan(data["x_rest_hrv"]), med, data["x_rest_hrv"]))
            rb = np.nan_to_num(np.where(np.isnan(data["x_rest_baselines"]), med,
                                        data["x_rest_baselines"]))
            rest_x = (data["x_rest_spectrograms"] if BACKBONE == "seecg"
                      else data["x_rest_windows"][:, None, :])
            ltn_ctx["rest"] = {
                "x": torch.tensor(np.asarray(rest_x)[rmask], dtype=torch.float32),
                "hrv": torch.tensor(((rh - mu) / sd)[rmask], dtype=torch.float32),
                "base": torch.tensor(((rb - mu) / sd)[rmask], dtype=torch.float32),
                "weight": REST_WEIGHT,
            }
            print(f"  REST axiom: {int(rmask.sum())} unlabeled rest windows "
                  f"(train subjects only, weight={REST_WEIGHT})")
        elif ltn_ctx is not None:
            ltn_ctx["rest"] = None
        train_loader = DataLoader(Subset(fold_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(fold_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

        # Class-weighted loss (inverse frequency from the TRAIN fold) to counter the
        # positive-class over-prediction bias. Weights come from LABELED train windows
        # only (all of them when LABEL_FRACTION == 1.0).
        train_labels = fold_scaled["y_labels"][train_idx]
        train_labels = train_labels[train_labels >= 0]
        counts = np.bincount(train_labels, minlength=2).astype(np.float64)
        class_weights = counts.sum() / (2.0 * np.clip(counts, 1.0, None))
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        for model_type in ("nn", "nsa"):
            torch.manual_seed(SEED + fold)  # same init seed per fold for both models
            model = build_model(model_type, data).to(device)
            # ignore_index=-1: unlabeled windows (label-scarcity mode) carry no CE.
            criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[LR_DROP_EPOCH], gamma=0.1)

            curve_ce, curve_sat = [], []
            for _ in range(epochs):
                _, _, ep_ce, ep_sat = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, model_type, ltn_ctx)
                curve_ce.append(ep_ce)
                curve_sat.append(ep_sat)
                scheduler.step()

            preds, probs, targets, subj_ids = evaluate(
                model, val_loader, device, model_type,
                subject_ids=data["x_subjects"][val_idx])
            # --- PRIMARY window-level metrics (UNCHANGED) ---
            m = metrics(targets, preds, probs)
            results[model_type].append(m)
            print(f"  [{MODEL_LABEL[model_type]:>9}] bal_acc={m['bal_acc']*100:5.2f}%  "
                  f"f1_macro={m['f1_macro']*100:5.2f}%  auroc={m['auroc']:.3f}  "
                  f"(f1 NST/ST={m['f1_neg']*100:4.1f}/{m['f1_pos']*100:4.1f})")

            # --- SUPPLEMENTARY subject-level AUROC (separate; does not affect the above) ---
            s_probs, s_labels = aggregate_by_subject(targets, probs, subj_ids)
            s_auroc = (roc_auc_score(s_labels, s_probs)
                       if len(np.unique(s_labels)) > 1 else float("nan"))
            subject_results[model_type].append(s_auroc)

            # --- Dump everything the figures need (does not affect any reported metric) ---
            plot_log[model_type].append({
                "fold": fold,
                "probs": probs, "targets": targets, "subjects": subj_ids,
                "subject_probs": s_probs, "subject_labels": s_labels,
                "curve_ce": np.array(curve_ce, dtype=float),
                "curve_sat": np.array(curve_sat, dtype=float),
                "auroc": m["auroc"], "subject_auroc": s_auroc,
            })

    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(PLOT_DIR, "plot_data.pkl"), "wb") as f:
        pickle.dump(plot_log, f)
    print(f"\nFigure data written to '{os.path.join(PLOT_DIR, 'plot_data.pkl')}'.")

    return results, subject_results


def summarize(results: Dict[str, List[Dict]]):
    # AUROC first (primary, threshold-free); ECE second (calibration, LOWER is better);
    # then balance-robust symmetric metrics. 'acc' (raw) kept last for reference only.
    keys = ["auroc", "ece", "bal_acc", "f1_macro", "f1_neg", "f1_pos", "acc"]
    fraction_keys = {"auroc", "ece", "bal_acc", "f1_macro", "f1_neg", "f1_pos"}

    lbl_nn, lbl_nsa = MODEL_LABEL["nn"], MODEL_LABEL["nsa"]
    print(f"\n{'='*72}\n SUBJECT-INDEPENDENT COMPARISON (mean +/- std over folds) | backbone: {BACKBONE}")
    print(f" AUROC = primary (higher better) | ECE = calibration (LOWER better)\n{'='*72}")
    print(f"{'metric':<8}{lbl_nn:>16}{lbl_nsa:>16}{'Delta(NSA-base)':>18}")

    for key in keys:
        nn_vals = np.array([r[key] for r in results["nn"]])
        nsa_vals = np.array([r[key] for r in results["nsa"]])
        scale = 1.0 if key in fraction_keys else 100.0
        unit = "" if key in fraction_keys else "%"
        nn_s = f"{np.nanmean(nn_vals)*scale:6.2f}+/-{np.nanstd(nn_vals)*scale:5.2f}{unit}"
        nsa_s = f"{np.nanmean(nsa_vals)*scale:6.2f}+/-{np.nanstd(nsa_vals)*scale:5.2f}{unit}"
        delta = (np.nanmean(nsa_vals) - np.nanmean(nn_vals)) * scale
        print(f"{key:<8}{nn_s:>16}{nsa_s:>16}{(f'{delta:+.2f}'+unit):>18}")

    # --- Corrected paired significance test (Nadeau-Bengio) on the key metrics ---
    print(f"\n Paired test {lbl_nsa} vs {lbl_nn} (Nadeau-Bengio corrected, {len(results['nn'])} folds):")
    for key in ["auroc", "bal_acc", "f1_macro", "ece"]:
        diffs = [n[key] - p[key] for n, p in zip(results["nsa"], results["nn"])]
        t, pval = corrected_paired_ttest(diffs)
        sig = "significant" if np.isfinite(pval) and pval < 0.05 else "n.s."
        better = "(NSA better)" if np.nanmean(diffs) > 0 else "(PAPER better)"
        if key == "ece":  # lower is better, so a negative diff favours NSA
            better = "(NSA better)" if np.nanmean(diffs) < 0 else "(PAPER better)"
        print(f"   {key:<6} mean Delta={np.nanmean(diffs):+.4f} {better:<14} "
              f"t={t:+.3f}  p={pval:.4f}  -> {sig}")
    print(f"{'='*72}\n")


def summarize_subject_level(subject_results: Dict[str, List[float]]):
    """
    SUPPLEMENTARY report: subject-level AUROC (window probabilities aggregated per
    subject), addressing within-subject correlation. Uses the SAME fold alignment and
    the SAME Nadeau-Bengio corrected paired test. Reported separately from the primary
    window-level analysis; none of the window-level numbers are affected.
    """
    nn_vals = np.array(subject_results["nn"], dtype=float)
    nsa_vals = np.array(subject_results["nsa"], dtype=float)

    print(f"\n{'='*60}\nSUPPLEMENTARY SUBJECT-LEVEL ANALYSIS")
    print(f"(window probabilities aggregated per subject)\n{'='*60}")
    print(f"{'metric':<16}{'PAPER':>12}{'PAPER+NSA':>12}{'Delta(NSA-Paper)':>18}")
    nn_m, nsa_m = np.nanmean(nn_vals), np.nanmean(nsa_vals)
    print(f"{'subject_AUROC':<16}{nn_m:>12.2f}{nsa_m:>12.2f}{(f'{nsa_m-nn_m:+.2f}'):>18}")

    diffs = [n - p for n, p in zip(nsa_vals, nn_vals)]
    t, pval = corrected_paired_ttest(diffs)
    print(f"\nPaired test (Nadeau-Bengio corrected, {len(nn_vals)} folds):")
    print(f"subject_AUROC mean Delta={np.nanmean(diffs):+.4f}")
    print(f"t={t:+.3f}")
    print(f"p={pval:.4f}  -> {'significant' if np.isfinite(pval) and pval < 0.05 else 'n.s.'}")
    print(f"{'='*60}\n")


# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Report the compute device up front (before the slow preprocessing step).
    if torch.cuda.is_available():
        print(f">>> DEVICE: cuda -> {torch.cuda.get_device_name(0)} "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda})")
    else:
        print(f">>> DEVICE: CPU only (torch {torch.__version__}) -- GPU NOT available!")

    subjects = SUBJECTS if SUBJECTS is not None else discover_subjects(BASE_DATASET_DIR)
    if MAX_SUBJECTS is not None:
        subjects = subjects[:MAX_SUBJECTS]
    print(f"Using {len(subjects)} subjects: {subjects}")

    # Cache keyed by subject count. 'rawhrv' = UN-imputed HRV (NaN kept) for leakage-free
    # per-fold imputation; 'raw1d' = also stores the raw 1D ECG window for the CNN-BiLSTM
    # backbone; '_rest' = additionally stores the unlabeled Training-1 rest windows for
    # the LTN rest axiom (regenerated once when NSA_REST=1; other runs keep the old cache).
    cache_path = f"processed_real_dataset_{len(subjects)}subj_rawhrv_raw1d.pkl"
    if REST_AXIOM:
        cache_path = f"processed_real_dataset_{len(subjects)}subj_rawhrv_raw1d_rest.pkl"
    data = load_and_cache_dataset(BASE_DATASET_DIR, subjects, cache_path)
    results, subject_results = run_comparison(data, n_splits=N_SPLITS, epochs=EPOCHS)
    summarize(results)                          # PRIMARY window-level analysis (unchanged)
    summarize_subject_level(subject_results)    # SUPPLEMENTARY subject-level analysis
