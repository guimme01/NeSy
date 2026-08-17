"""
CNN-BiLSTM Net: a SECOND backbone for the neuro-symbolic ablation.

Motivation
----------
SE-ECG Net (se_ecg_net.py) classifies cognitive load from the 2D STFT *spectrogram*
of the ECG window. To show that the benefit of the physiological symbolic knowledge is
NOT specific to that one architecture, we add a structurally different backbone that
reads the RAW 1D ECG waveform of the window:

    raw ECG (B, 1, T)  ->  3x [Conv1d - BN - ReLU - MaxPool]  ->  BiLSTM  ->  FC -> 2 logits

The 1D convolutional stem extracts local beat morphology and, through strided pooling,
downsamples the long 60 s window (T = 21600 samples @ 360 Hz) into a short feature
sequence; the bidirectional LSTM then models the temporal dynamics across the window.
This is a completely different inductive bias from the spectrogram CNN, so replicating
the neuro-symbolic gain here is strong evidence that physiology-guided constraints help
independently of the backbone.

The class mirrors SEECGNet's interface (`forward(x, return_logits=...)`,
`forward_features(x)`) so the SAME NSA wrapper machinery can wrap it unchanged.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock1D(nn.Module):
    """1D convolutional block: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        pool_size: int = 4,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBiLSTMNet(nn.Module):
    """
    CNN-BiLSTM classifier for RAW 1D ECG windows.

    Architecture
    ------------
      - Input : raw ECG window (Batch, 1, T)  [T = 21600 for a 60 s @ 360 Hz window].
      - Stem  : 3 ConvBlock1D layers (1 -> 32 -> 64 -> 128 channels), each with an
                initial stride and MaxPool that together divide the time axis by ~ 4x4x4
                per block, shrinking the sequence to a few dozen steps.
      - BiLSTM: single bidirectional LSTM over the (short) feature sequence.
      - Head  : dropout -> FC -> 2 logits (softmax for probabilities).
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout_rate: float = 0.3,
        stem_stride: int = 2,
    ):
        super().__init__()
        c1, c2, c3 = conv_channels

        # 1D convolutional stem. The first block strides to cut the very long raw window
        # early; all blocks MaxPool by 4. Net downsampling: stem_stride * 4^3 (= 128x for
        # stem_stride=2), so 21600 -> ~168 time steps fed to the LSTM.
        self.conv1 = ConvBlock1D(in_channels, c1, kernel_size=7, stride=stem_stride, pool_size=4)
        self.conv2 = ConvBlock1D(c1, c2, kernel_size=5, stride=1, pool_size=4)
        self.conv3 = ConvBlock1D(c2, c3, kernel_size=3, stride=1, pool_size=4)

        # Bidirectional LSTM over the feature sequence (batch_first).
        self.lstm = nn.LSTM(
            input_size=c3, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
        )
        self._feat_dim = 2 * lstm_hidden  # bidirectional -> forward + backward states

        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self._feat_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    @property
    def feature_dim(self) -> int:
        return self._feat_dim

    def _sequence_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """raw (B, 1, T) or (B, T) -> pooled BiLSTM embedding (B, 2*lstm_hidden)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)            # (B, T) -> (B, 1, T)
        x = self.conv1(x)                 # (B, c1, T/ (stride*4))
        x = self.conv2(x)                 # (B, c2, .../4)
        x = self.conv3(x)                 # (B, c3, .../4)
        x = x.transpose(1, 2)             # (B, L, c3) : time-major sequence for the LSTM
        out, _ = self.lstm(x)             # (B, L, 2*hidden)
        return out.mean(dim=1)            # temporal average pooling -> (B, 2*hidden)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate embedding BEFORE the classifier (for the NSA wrapper)."""
        return self._sequence_embedding(x)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        emb = self._sequence_embedding(x)
        logits = self.classifier(self.dropout(emb))
        if return_logits:
            return logits
        return self.softmax(logits)


if __name__ == "__main__":
    print("Testing CNN-BiLSTM Net on raw ECG windows...")
    model = CNNBiLSTMNet(in_channels=1, num_classes=2)
    model.eval()

    # 60 s window @ 360 Hz = 21600 samples.
    dummy = torch.randn(4, 1, 21600)
    with torch.no_grad():
        probs = model(dummy, return_logits=False)
        logits = model(dummy, return_logits=True)
        feats = model.forward_features(dummy)

    print(f"Input raw ECG shape : {tuple(dummy.shape)}")
    print(f"Probabilities shape : {tuple(probs.shape)}  (sum={probs[0].sum():.4f})")
    print(f"Logits shape        : {tuple(logits.shape)}")
    print(f"Feature embedding   : {tuple(feats.shape)}  (feature_dim={model.feature_dim})")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params    : {total:,}")
