"""
ECG Signal Preprocessing Pipeline for Cognitive Load Classification & Neuro-Symbolic AI (NSA).

This module implements a modular, production-ready preprocessing pipeline for raw ECG signals using
SciPy, NeuroKit2, Pandas, and PyTorch.

Pipeline Stages:
  1. Downsampling: Resample signal from 1000 Hz to 360 Hz.
  2. Denoising:
     - 0.5 Hz high-pass zero-phase Butterworth filter (removes baseline wander).
     - 50 Hz / 60 Hz IIR zero-phase notch filter (removes powerline interference).
  3. R-peak Detection:
     - Detect R-peaks using NeuroKit2's absolute gradient method ('neurokit').
  4. Segmentation & Windowing:
     - Extract 60-second windows centered on R-peaks (30 seconds before, 30 seconds after).
  5. Sliding Window:
     - Shift window by 10 R-peaks for consecutive slices.
  6. Spectrogram Conversion (STFT):
     - Transform 1D ECG windows into 2D dB-scaled normalized spectrograms (350 ms Hamming frames).
  7. HRV Feature Extraction (for Neuro-Symbolic AI):
     - Extract ~86 time-domain, frequency-domain, and non-linear Heart Rate Variability (HRV) metrics
       from 60s windows with NaN imputation handling.
"""

import re
import numpy as np
import scipy.signal
import pandas as pd
import neurokit2 as nk
from typing import List, Tuple, Optional, Union, Dict
import torch
from torch.utils.data import Dataset


def load_physio_ecg(physio_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Robustly loads the ECG channel from a BioPac physio .txt export.

    The header length and channel ordering are NOT constant across the Meteier et al.
    dataset (e.g. NST11.txt has no header at all). Instead of hardcoding
    ``skiprows=11`` / ``CH2``, this function:
      1. Reads the sampling rate from the ``<x> msec/sample`` line (defaults 1000 Hz).
      2. Reads the channel-name lines (EDA / ECG / Respiration ...) to find WHICH
         channel is the ECG, in whatever order the file lists them.
      3. Locates the ``min<TAB>CH1<TAB>...`` data-header line to know where the
         numeric data begins (falling back to a header-less layout when absent).

    Parameters
    ----------
    physio_path : str
        Path to a ``<code>.txt`` BioPac physio export (tab-separated).

    Returns
    -------
    time_sec : np.ndarray
        Elapsed time in SECONDS (the file stores minutes; converted here).
    ecg : np.ndarray
        1D raw ECG signal (float32).
    sampling_rate : float
        Sampling rate in Hz parsed from the header (default 1000.0).
    """
    with open(physio_path, "r", encoding="utf-8", errors="ignore") as f:
        header_lines = [f.readline() for _ in range(40)]

    # --- Sampling rate ---
    sampling_rate = 1000.0
    for ln in header_lines:
        m = re.search(r"([\d.]+)\s*msec/sample", ln)
        if m:
            sampling_rate = 1000.0 / float(m.group(1))
            break

    # --- Locate the "min <TAB> CH1 ..." data-header line ---
    data_header_idx = None
    for i, ln in enumerate(header_lines):
        if ln.strip().lower().startswith("min") and ("ch1" in ln.lower() or "\t" in ln):
            data_header_idx = i
            break

    # --- Find which channel is the ECG, in listing order ---
    ecg_channel_pos = None  # 0-based position among the channels (CH1=0, CH2=1, ...)
    chan_pos = -1
    scan_upto = data_header_idx if data_header_idx is not None else 0
    for ln in header_lines[:scan_upto]:
        s = ln.strip()
        if re.match(r"^(ECG|EDA|Respiration|RESP|SKT|EMG)\b", s, re.IGNORECASE):
            chan_pos += 1
            if s.upper().startswith("ECG"):
                ecg_channel_pos = chan_pos

    if data_header_idx is None:
        # Header-less file (e.g. NST11.txt): assume the dataset's standard
        # column layout  min | EDA | ECG | Respiration  -> ECG is data column 2.
        skiprows = 0
        ecg_col = 2
    else:
        if ecg_channel_pos is None:
            ecg_channel_pos = 1  # standard order EDA, ECG, Resp -> ECG is 2nd channel
        ecg_col = ecg_channel_pos + 1  # +1 for the leading time column
        # Data starts after the header line and the sample-count row beneath it.
        skiprows = data_header_idx + 2

    df = pd.read_csv(
        physio_path,
        sep="\t",
        skiprows=skiprows,
        header=None,
        usecols=[0, ecg_col],
        dtype=np.float32,
        na_values=["", " "],
    ).dropna()

    time_sec = df.iloc[:, 0].to_numpy(dtype=np.float64) * 60.0  # minutes -> seconds
    ecg = df.iloc[:, 1].to_numpy(dtype=np.float32)
    return time_sec, ecg, sampling_rate


def downsample_ecg(
    ecg_signal: np.ndarray,
    orig_sr: float = 1000.0,
    target_sr: float = 360.0
) -> np.ndarray:
    """
    Downsamples an ECG signal from original sampling rate to target sampling rate.

    Parameters
    ----------
    ecg_signal : np.ndarray
        1D array containing raw ECG signal values.
    orig_sr : float, default=1000.0
        Original sampling rate in Hz.
    target_sr : float, default=360.0
        Target downsampled sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Downsampled 1D ECG signal array.
    """
    if orig_sr == target_sr:
        return ecg_signal

    resampled_signal = nk.signal_resample(
        ecg_signal,
        sampling_rate=int(orig_sr),
        desired_sampling_rate=int(target_sr),
        method="interpolation"
    )
    return np.asarray(resampled_signal, dtype=np.float32)


def denoise_ecg(
    ecg_signal: np.ndarray,
    sampling_rate: float = 360.0,
    highpass_cutoff: float = 0.5,
    powerline_freq: float = 50.0,
    butter_order: int = 4,
    notch_q: float = 30.0
) -> np.ndarray:
    """
    Denoises raw/downsampled ECG signal by removing baseline drift and powerline noise.

    Steps:
      - 0.5 Hz High-pass zero-phase Butterworth filter for baseline wander removal.
      - Powerline notch filter (50 Hz or 60 Hz) using zero-phase IIR notch filter.

    Parameters
    ----------
    ecg_signal : np.ndarray
        1D array containing downsampled ECG signal.
    sampling_rate : float, default=360.0
        Sampling frequency of input signal in Hz.
    highpass_cutoff : float, default=0.5
        Cutoff frequency for high-pass baseline drift removal in Hz.
    powerline_freq : float, default=50.0
        Frequency of powerline interference to eliminate in Hz (50.0 or 60.0).
    butter_order : int, default=4
        Order of high-pass Butterworth filter.
    notch_q : float, default=30.0
        Quality factor Q of powerline notch filter.

    Returns
    -------
    np.ndarray
        Denoised 1D ECG signal array.
    """
    # 1. Baseline Drift Removal: Zero-phase High-pass Butterworth filter
    sos = scipy.signal.butter(
        N=butter_order,
        Wn=highpass_cutoff,
        btype="highpass",
        fs=sampling_rate,
        output="sos"
    )
    filtered_signal = scipy.signal.sosfiltfilt(sos, ecg_signal)

    # 2. Powerline Interference Removal: Zero-phase IIR Notch Filter
    if powerline_freq is not None and powerline_freq > 0:
        if powerline_freq < (sampling_rate / 2.0):
            b_notch, a_notch = scipy.signal.iirnotch(
                w0=powerline_freq,
                Q=notch_q,
                fs=sampling_rate
            )
            filtered_signal = scipy.signal.filtfilt(b_notch, a_notch, filtered_signal)

    return np.asarray(filtered_signal, dtype=np.float32)


def detect_r_peaks(
    ecg_cleaned: np.ndarray,
    sampling_rate: float = 360.0,
    method: str = "neurokit"
) -> Tuple[np.ndarray, Dict]:
    """
    Detects R-peaks in a cleaned ECG signal using NeuroKit2's absolute gradient method ('neurokit').

    Parameters
    ----------
    ecg_cleaned : np.ndarray
        Cleaned/denoised ECG signal.
    sampling_rate : float, default=360.0
        Sampling frequency of ECG signal.
    method : str, default="neurokit"
        R-peak detection algorithm in NeuroKit2 ('neurokit' absolute gradient method).

    Returns
    -------
    r_peaks : np.ndarray
        1D array containing sample indices of detected R-peaks.
    info : dict
        NeuroKit2 dictionary output containing peak metadata.
    """
    _, info = nk.ecg_peaks(
        ecg_cleaned=ecg_cleaned,
        sampling_rate=int(sampling_rate),
        method=method,
        correct_artifacts=False
    )
    r_peaks = info["ECG_R_Peaks"]
    return np.asarray(r_peaks, dtype=int), info


def segment_and_window(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float = 360.0,
    window_duration_sec: float = 60.0,
    r_peak_step: int = 10
) -> Tuple[np.ndarray, List[int]]:
    """
    Segments ECG signal into 60-second centered windows around detected R-peaks using a sliding window.

    - Each window is 60 seconds long (-30s before, +30s after each center R-peak).
    - Window centers advance by shifting `r_peak_step` R-peaks (default: 10 R-peaks).

    Parameters
    ----------
    ecg_signal : np.ndarray
        1D array of clean/denoised ECG signal.
    r_peaks : np.ndarray
        1D array of detected R-peak sample indices.
    sampling_rate : float, default=360.0
        Sampling rate in Hz.
    window_duration_sec : float, default=60.0
        Total length of each segment window in seconds.
    r_peak_step : int, default=10
        Number of R-peaks to shift for next slice (sliding window step size).

    Returns
    -------
    segments : np.ndarray
        2D NumPy array of shape (N_windows, window_samples),
        where window_samples = int(window_duration_sec * sampling_rate) = 21600.
    center_peaks : List[int]
        List of sample indices of R-peaks that serve as window centers.
    """
    half_window_sec = window_duration_sec / 2.0
    half_window_samples = int(round(half_window_sec * sampling_rate))
    total_window_samples = half_window_samples * 2  # 30s * 360 * 2 = 21600 samples

    segments = []
    center_peaks = []

    for idx in range(0, len(r_peaks), r_peak_step):
        center_peak = r_peaks[idx]
        start_idx = center_peak - half_window_samples
        end_idx = center_peak + half_window_samples

        if start_idx >= 0 and end_idx <= len(ecg_signal):
            window = ecg_signal[start_idx:end_idx]
            if len(window) == total_window_samples:
                segments.append(window)
                center_peaks.append(center_peak)

    if len(segments) == 0:
        return np.empty((0, total_window_samples), dtype=ecg_signal.dtype), center_peaks

    return np.array(segments, dtype=np.float32), center_peaks


def ecg_to_spectrogram(
    window: Union[np.ndarray, torch.Tensor],
    fs: float = 360.0,
    frame_duration_ms: float = 350.0,
    overlap_ratio: float = 0.5,
    norm_type: str = "minmax",
    eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts a 1D ECG window (or 2D batch of windows) into a 2D/3D dB-scaled normalized spectrogram via STFT.

    Specifications:
      1. STFT: Computed via SciPy or PyTorch STFT.
      2. Frame size: 350 ms Hamming window (350 ms @ 360 Hz = 126 samples).
      3. Log Scaling: Converted to decibels (dB) scale: S_dB = 10 * log10(|STFT|^2 + eps).
      4. Normalization: Min-Max scaled to [0, 1] (or Z-score normalized).
    """
    is_torch = torch.is_tensor(window)
    is_batch = (window.ndim == 2) if not is_torch else (window.dim() == 2)

    n_perseg = int(round((frame_duration_ms / 1000.0) * fs))
    n_overlap = int(round(n_perseg * overlap_ratio))
    hop_length = n_perseg - n_overlap

    if is_torch:
        x_tensor = window.clone().detach() if is_batch else window.clone().detach().unsqueeze(0)
        window_func = torch.hamming_window(n_perseg, device=x_tensor.device)
        
        stft_res = torch.stft(
            x_tensor,
            n_fft=n_perseg,
            hop_length=hop_length,
            win_length=n_perseg,
            window=window_func,
            center=True,
            return_complex=True
        )
        
        power_spec = torch.abs(stft_res) ** 2
        spec_db = 10.0 * torch.log10(power_spec + eps)
        
        if norm_type == "minmax":
            min_val = spec_db.amin(dim=(-2, -1), keepdim=True)
            max_val = spec_db.amax(dim=(-2, -1), keepdim=True)
            spec_norm = (spec_db - min_val) / (max_val - min_val + eps)
        elif norm_type == "zscore":
            mean_val = spec_db.mean(dim=(-2, -1), keepdim=True)
            std_val = spec_db.std(dim=(-2, -1), keepdim=True)
            spec_norm = (spec_db - mean_val) / (std_val + eps)
        else:
            raise ValueError(f"Unknown norm_type: '{norm_type}'")
            
        spec_out = spec_norm.unsqueeze(1)
        if not is_batch:
            spec_out = spec_out.squeeze(0)
            
        return spec_out

    else:
        x_arr = np.asarray(window, dtype=np.float32)
        if not is_batch:
            x_arr = np.expand_dims(x_arr, axis=0)

        batch_specs = []
        for i in range(x_arr.shape[0]):
            _, _, Zxx = scipy.signal.stft(
                x_arr[i],
                fs=fs,
                window="hamming",
                nperseg=n_perseg,
                noverlap=n_overlap,
                boundary="zeros"
            )
            power_spec = np.abs(Zxx) ** 2
            spec_db = 10.0 * np.log10(power_spec + eps)
            
            if norm_type == "minmax":
                min_v = np.min(spec_db)
                max_v = np.max(spec_db)
                spec_norm = (spec_db - min_v) / (max_v - min_v + eps)
            elif norm_type == "zscore":
                mean_v = np.mean(spec_db)
                std_v = np.std(spec_db)
                spec_norm = (spec_db - mean_v) / (std_v + eps)
            else:
                raise ValueError(f"Unknown norm_type: '{norm_type}'")
                
            batch_specs.append(np.expand_dims(spec_norm, axis=0))

        spec_out = np.array(batch_specs, dtype=np.float32)
        if not is_batch:
            spec_out = spec_out[0]
            
        return spec_out


def extract_hrv_features(
    ecg_window: np.ndarray,
    fs: float = 360.0,
    impute_strategy: str = "zero"
) -> pd.Series:
    """
    Extracts explicit Heart Rate Variability (HRV) features from a 60-second ECG window via NeuroKit2.

    Extracts ~86 standard HRV features including:
      - Time-domain: RMSSD, SDNN, pNN50, pNN20, MeanNN, CVNN, etc.
      - Frequency-domain: LF, HF, VLF, LF/HF ratio, LFn, HFn, Normalized Power.
      - Non-linear domain: SD1, SD2, Sample Entropy, DFA alpha, CSI, CVI, etc.

    Handles missing/NaN/Inf values gracefully via imputation strategies.

    Parameters
    ----------
    ecg_window : np.ndarray
        1D array of 60-second clean ECG window signal.
    fs : float, default=360.0
        Sampling frequency in Hz.
    impute_strategy : str, default="zero"
        Strategy to replace NaNs/Infs: "zero" fills NaNs with 0.0; "mean" fills with column means.

    Returns
    -------
    pd.Series
        Pandas Series containing ~86 clean, imputed HRV feature values.
    """
    try:
        # Detect R-peaks within this 60s window
        r_peaks, info = detect_r_peaks(ecg_window, sampling_rate=fs, method="neurokit")

        # NeuroKit2 requires at least 3 R-peaks to compute HRV metrics
        if len(r_peaks) >= 3:
            hrv_df = nk.hrv(info, sampling_rate=int(fs))
        else:
            # Fallback: create empty DataFrame with standard columns if peaks are insufficient
            hrv_df = pd.DataFrame()

    except Exception:
        hrv_df = pd.DataFrame()

    # Replace Infinite values with NaN
    hrv_df = hrv_df.replace([np.inf, -np.inf], np.nan)

    # Convert to 1D Series
    if not hrv_df.empty:
        hrv_series = hrv_df.iloc[0].copy()
    else:
        # Return default series filled with 0.0
        dummy_info = {"ECG_R_Peaks": np.array([100, 450, 800, 1150])}
        template_cols = nk.hrv(dummy_info, sampling_rate=int(fs)).columns
        hrv_series = pd.Series(0.0, index=template_cols)

    # Impute NaN values
    if impute_strategy == "zero":
        hrv_series = hrv_series.fillna(0.0)
    elif impute_strategy == "mean":
        mean_val = hrv_series.mean(skipna=True)
        fill_val = mean_val if not np.isnan(mean_val) else 0.0
        hrv_series = hrv_series.fillna(fill_val)

    return hrv_series.astype(np.float32)


def extract_hrv_features_batch(
    windows: np.ndarray,
    fs: float = 360.0,
    impute_strategy: str = "median"
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts HRV feature matrix across a 2D batch of ECG windows (N_windows, window_samples).

    Applies cross-window median/mean imputation for missing values across the dataset.

    Parameters
    ----------
    windows : np.ndarray
        2D array of shape (N_windows, 21600).
    fs : float, default=360.0
        Sampling frequency in Hz.
    impute_strategy : str, default="median"
        Dataset-level imputation strategy ("median", "mean", or "zero").

    Returns
    -------
    hrv_matrix : np.ndarray
        2D NumPy array of shape (N_windows, N_features).
    feature_names : List[str]
        List of HRV feature names (~86 feature columns).
    """
    batch_series = []
    for i in range(len(windows)):
        s = extract_hrv_features(windows[i], fs=fs, impute_strategy="none")
        batch_series.append(s)

    hrv_df = pd.DataFrame(batch_series)
    hrv_df = hrv_df.replace([np.inf, -np.inf], np.nan)

    # Perform dataset-level imputation
    if impute_strategy == "median":
        hrv_df = hrv_df.fillna(hrv_df.median()).fillna(0.0)
    elif impute_strategy == "mean":
        hrv_df = hrv_df.fillna(hrv_df.mean()).fillna(0.0)
    elif impute_strategy == "zero":
        hrv_df = hrv_df.fillna(0.0)

    feature_names = list(hrv_df.columns)
    hrv_matrix = hrv_df.to_numpy(dtype=np.float32)

    return hrv_matrix, feature_names


if __name__ == "__main__":
    print("Executing ECG Preprocessing + HRV Feature Extraction Test...")

    # Simulate 180 seconds of raw ECG data at 1000 Hz
    duration_sec = 180
    raw_fs = 1000
    ecg_raw = nk.ecg_simulate(duration=duration_sec, sampling_rate=raw_fs, heart_rate=72, noise=0.05)

    # 1. Downsample to 360 Hz & Denoise
    ecg_360 = downsample_ecg(ecg_raw, orig_sr=1000.0, target_sr=360.0)
    ecg_clean = denoise_ecg(ecg_360, sampling_rate=360.0, highpass_cutoff=0.5, powerline_freq=50.0)

    # 2. Detect R-peaks & Window into 60s segments
    r_peaks, _ = detect_r_peaks(ecg_clean, sampling_rate=360.0)
    segments, _ = segment_and_window(ecg_clean, r_peaks, sampling_rate=360.0, window_duration_sec=60.0, r_peak_step=10)

    print(f"Extracted {len(segments)} windows of 60s @ 360Hz")

    if len(segments) > 0:
        # Extract HRV features for a single window
        single_hrv = extract_hrv_features(segments[0], fs=360.0, impute_strategy="zero")
        print(f"\nSingle Window HRV Features Extracted: {len(single_hrv)} features")
        print("Sample Key Features:")
        for key in ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LF", "HRV_HF", "HRV_LFHF"]:
            if key in single_hrv:
                print(f"  - {key}: {single_hrv[key]:.4f}")

        # Extract Batch HRV Features Matrix
        hrv_matrix, feature_names = extract_hrv_features_batch(segments, fs=360.0, impute_strategy="median")
        print(f"\nBatch HRV Matrix shape: {hrv_matrix.shape} ({len(feature_names)} feature columns)")
        print("HRV Extraction Test Successfully Passed!")
