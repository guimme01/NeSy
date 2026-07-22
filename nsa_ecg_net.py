"""
NSA-ECG Net: the EXACT paper model (SE-ECG Net) + a Neuro-Symbolic AI layer.

This wrapper is designed for a clean ablation:
  - `SEECGNet`  = the exact Shi et al. SE-ECG Net (baseline).
  - `NSAECGNet` = the SAME SEECGNet instance, UNCHANGED, with a purely ADDITIVE
    neuro-symbolic layer on top of its logits. The neural backbone is byte-for-byte
    the paper model, so any performance difference is attributable ONLY to the NSA layer.

The neuro-symbolic layer implements the three elements of the integration strategy:
  1. Hybrid feature input : a symbolic branch maps the handcrafted HRV features to a
     2-class logit correction that is ADDED to the paper model's logits.
  2. Physiological rule    : a differentiable baseline-shift rule that lowers the
     High-load confidence when HR (MeanNN) and RMSSD do not move vs. the subject's baseline.
  3. Causal reasoning      : an SAE-L3 handover context term that nudges predictions toward
     the safety-conservative High-load class when a takeover is imminent.

Each element can be toggled off (`use_hrv_branch` / `use_rule_engine` / `use_handover`)
for finer ablations.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from se_ecg_net import SEECGNet


class SymbolicRuleEngine(nn.Module):
    """
    Differentiable physiological rule.

    Rule: "If HR (MeanNN) and RMSSD show no significant shift relative to the subject's
    baseline, lower the model's confidence for the High Cognitive Load class."
    """
    def __init__(
        self,
        mean_nn_idx: int,
        rmssd_idx: int,
        tau_mean_nn: float = 1.0,    # soft threshold for MeanNN shift (STANDARDIZED units)
        tau_rmssd: float = 1.0,      # soft threshold for RMSSD shift (STANDARDIZED units)
        penalty_weight: float = 1.0, # logit penalty on the High-load class
    ):
        super().__init__()
        self.mean_nn_idx = mean_nn_idx
        self.rmssd_idx = rmssd_idx
        self.tau_mean_nn = tau_mean_nn
        self.tau_rmssd = tau_rmssd
        self.penalty_weight = penalty_weight

    def forward(
        self,
        logits: torch.Tensor,
        x_hrv: torch.Tensor,
        baseline_hrv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if baseline_hrv is None:
            return logits

        delta_mean_nn = torch.abs(x_hrv[:, self.mean_nn_idx] - baseline_hrv[:, self.mean_nn_idx])
        delta_rmssd = torch.abs(x_hrv[:, self.rmssd_idx] - baseline_hrv[:, self.rmssd_idx])

        # Soft indicators: ~1.0 when the shift is small/insignificant, ~0.0 when large.
        is_mean_nn_stable = torch.sigmoid(4.0 * (1.0 - delta_mean_nn / self.tau_mean_nn))
        is_rmssd_stable = torch.sigmoid(4.0 * (1.0 - delta_rmssd / self.tau_rmssd))
        no_significant_shift = is_mean_nn_stable * is_rmssd_stable  # soft logical AND

        adjusted = logits.clone()
        adjusted[:, 1] = logits[:, 1] - self.penalty_weight * no_significant_shift
        return adjusted


class NSAECGNet(nn.Module):
    """
    Exact SE-ECG Net (paper model) wrapped with an additive Neuro-Symbolic layer.
    """
    def __init__(
        self,
        base_model: SEECGNet,
        num_hrv_features: int,
        hrv_feature_names: Optional[List[str]] = None,
        dropout_rate: float = 0.3,
        handover_bias: float = 1.5,
        use_hrv_branch: bool = True,
        use_rule_engine: bool = True,
        use_handover: bool = True,
    ):
        super().__init__()
        # The exact paper model, used unchanged.
        self.base = base_model

        self.use_hrv_branch = use_hrv_branch
        self.use_rule_engine = use_rule_engine
        self.use_handover = use_handover
        self.handover_bias = handover_bias

        # 1. Hybrid feature input: symbolic HRV -> additive logit correction.
        self.hrv_branch = nn.Sequential(
            nn.Linear(num_hrv_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 2),
        )

        # 2. Physiological rule engine (locate MeanNN / RMSSD columns).
        if hrv_feature_names is None:
            mean_nn_idx, rmssd_idx = 0, 8
        else:
            mean_nn_idx = hrv_feature_names.index("HRV_MeanNN") if "HRV_MeanNN" in hrv_feature_names else 0
            rmssd_idx = hrv_feature_names.index("HRV_RMSSD") if "HRV_RMSSD" in hrv_feature_names else 8
        self.rule_engine = SymbolicRuleEngine(mean_nn_idx=mean_nn_idx, rmssd_idx=rmssd_idx)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x_spectrogram: torch.Tensor,
        x_hrv: torch.Tensor,
        baseline_hrv: Optional[torch.Tensor] = None,
        handover_imminent: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        # --- EXACT paper model (unchanged) ---
        logits = self.base(x_spectrogram, return_logits=True)

        # --- Neuro-Symbolic layer (purely additive on top) ---
        # 1. Hybrid feature input.
        if self.use_hrv_branch:
            logits = logits + self.hrv_branch(x_hrv)

        # 2. Physiological baseline-shift rule.
        if self.use_rule_engine and baseline_hrv is not None:
            logits = self.rule_engine(logits, x_hrv, baseline_hrv)

        # 3. Causal SAE-L3 handover context.
        if self.use_handover and handover_imminent is not None:
            h = handover_imminent
            if h.dim() == 1:
                h = h.unsqueeze(1)
            adjustment = torch.zeros_like(logits)
            adjustment[:, 1] = self.handover_bias * h.squeeze(1)
            logits = logits + adjustment

        if return_logits:
            return logits
        return self.softmax(logits)


if __name__ == "__main__":
    print("Testing NSA-ECG Net = exact SE-ECG Net + Neuro-Symbolic layer...")

    hrv_names = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD"] + [f"HRV_Feature_{i}" for i in range(83)]

    base = SEECGNet(in_channels=1, num_classes=2, conv_channels=(32, 64, 128))
    model = NSAECGNet(base_model=base, num_hrv_features=86, hrv_feature_names=hrv_names, handover_bias=1.5)
    model.eval()

    specs = torch.randn(3, 1, 64, 344)
    hrv = torch.tensor([
        [800.0, 50.0, 30.0] + [0.0] * 83,   # stable vs baseline
        [720.0, 80.0, 55.0] + [0.0] * 83,   # shifted
        [710.0, 85.0, 60.0] + [0.0] * 83,   # shifted
    ], dtype=torch.float32)
    baseline = torch.tensor([
        [801.0, 49.0, 30.5] + [0.0] * 83,
        [800.0, 50.0, 30.0] + [0.0] * 83,
        [800.0, 50.0, 30.0] + [0.0] * 83,
    ], dtype=torch.float32)
    handover = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    with torch.no_grad():
        # Sanity: with the NSA layer fully disabled, output must equal the base paper model.
        model.use_hrv_branch = model.use_rule_engine = model.use_handover = False
        base_probs = torch.softmax(base(specs, return_logits=True), dim=1)
        nsa_off = model(specs, hrv, baseline, handover)
        assert torch.allclose(base_probs, nsa_off, atol=1e-6), "NSA-off must equal paper model!"

        model.use_hrv_branch = model.use_rule_engine = model.use_handover = True
        probs = model(specs, hrv, baseline, handover)

    print(f"NSA-off == paper model: OK")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"Sample 0 (stable): {probs[0].numpy()}")
    print(f"Sample 2 (shifted + handover): {probs[2].numpy()}")
    print("Model test passed.")
