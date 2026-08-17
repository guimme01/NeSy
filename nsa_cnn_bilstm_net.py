"""
NSA-CNN-BiLSTM Net: the raw-ECG CNN-BiLSTM backbone + the SAME Neuro-Symbolic layer
used for SE-ECG Net.

This is the second arm of the architecture-agnostic ablation. It mirrors nsa_ecg_net.py
exactly, but wraps CNNBiLSTMNet (which reads the RAW 1D ECG window) instead of SEECGNet
(which reads the 2D spectrogram). The neuro-symbolic layer is byte-for-byte the same idea:

  - the backbone produces the base logits (unchanged),
  - an additive HRV MLP branch corrects those logits from the handcrafted HRV features,
  - the physiological LTN axioms are applied to the training loss (in train_compare.py),

so any gain over the plain CNN-BiLSTM backbone is attributable ONLY to the NSA layer,
just as for SE-ECG Net. Keeping a separate class avoids touching the validated SEECGNet
ablation and its self-test.

The rule engine and handover terms are kept for parity with NSAECGNet (default OFF in the
pipeline); the head-to-head uses the HRV branch + LTN axioms.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from cnn_bilstm_net import CNNBiLSTMNet
from nsa_ecg_net import SymbolicRuleEngine


class NSACNNBiLSTMNet(nn.Module):
    """CNN-BiLSTM backbone (raw ECG) wrapped with the additive Neuro-Symbolic layer."""
    def __init__(
        self,
        base_model: CNNBiLSTMNet,
        num_hrv_features: int,
        hrv_feature_names: Optional[List[str]] = None,
        dropout_rate: float = 0.3,
        handover_bias: float = 1.5,
        use_hrv_branch: bool = True,
        use_rule_engine: bool = True,
        use_handover: bool = True,
    ):
        super().__init__()
        # The raw-ECG backbone, used unchanged.
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
        x_raw: torch.Tensor,
        x_hrv: torch.Tensor,
        baseline_hrv: Optional[torch.Tensor] = None,
        handover_imminent: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        # --- Backbone (unchanged): raw 1D ECG window -> logits ---
        logits = self.base(x_raw, return_logits=True)

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
    print("Testing NSA-CNN-BiLSTM = raw-ECG backbone + Neuro-Symbolic layer...")

    hrv_names = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD"] + [f"HRV_Feature_{i}" for i in range(83)]

    base = CNNBiLSTMNet(in_channels=1, num_classes=2)
    model = NSACNNBiLSTMNet(base_model=base, num_hrv_features=86,
                            hrv_feature_names=hrv_names, handover_bias=1.5)
    model.eval()

    raw = torch.randn(3, 1, 21600)
    hrv = torch.randn(3, 86)
    baseline = torch.randn(3, 86)
    handover = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    with torch.no_grad():
        # Sanity: with the NSA layer fully disabled, output must equal the base backbone.
        model.use_hrv_branch = model.use_rule_engine = model.use_handover = False
        base_probs = torch.softmax(base(raw, return_logits=True), dim=1)
        nsa_off = model(raw, hrv, baseline, handover)
        assert torch.allclose(base_probs, nsa_off, atol=1e-6), "NSA-off must equal backbone!"

        model.use_hrv_branch = model.use_rule_engine = model.use_handover = True
        probs = model(raw, hrv, baseline, handover)

    print("NSA-off == backbone: OK")
    print(f"Output probabilities shape: {tuple(probs.shape)}")
    print("Model test passed.")
