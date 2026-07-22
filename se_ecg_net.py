"""
SE-ECG Net: Deep Learning Architecture for ECG Spectrogram Cognitive Load Classification.

This module implements the 'SE-ECG Net' model in PyTorch for binary classification
of 2D ECG STFT spectrograms (High vs. Low Cognitive Workload).

Key Components:
  1. Convolution Module: 3 Convolutional 2D blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D).
  2. Squeeze-and-Excitation (SE) Block:
     - Squeeze: Global Average Pooling (AdaptiveAvgPool2d) per channel.
     - Excitation: 2 FC layers with ReLU and Sigmoid activation learning channel attention weights.
     - Rescaling: Channel-wise feature map scaling (x * attention_weights).
  3. Classifier: Adaptive Global Pooling -> FC Linear Layer -> Softmax output probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Attention Block.

    Applies channel-wise attention by dynamically reweighting feature maps.
    """
    def __init__(self, channels: int, reduction: int = 16):
        """
        Parameters
        ----------
        channels : int
            Number of input feature channels.
        reduction : int, default=16
            Reduction ratio for the bottleneck excitation FC layer.
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling (b, c, 1, 1) -> (b, c)
        squeezed = self.squeeze(x).view(b, c)
        # Excitation: Learn channel weights (b, c) -> (b, c, 1, 1)
        weights = self.excitation(squeezed).view(b, c, 1, 1)
        # Rescale feature maps
        return x * weights


class ConvBlock2D(nn.Module):
    """
    2D Convolutional Block: Conv2D -> BatchNorm2d -> ReLU -> MaxPool2d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: Tuple[int, int] = (2, 2)
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEECGNet(nn.Module):
    """
    SE-ECG Net Neural Network for ECG Spectrogram Classification.

    Architecture Overview:
      - Input: 2D Spectrogram Tensor of shape (Batch, 1, Freq_Bins, Time_Frames).
      - Conv Module: 3 ConvBlock2D layers (1 -> 32 -> 64 -> 128 channels).
      - SE Module: Squeeze-and-Excitation attention reweighting feature channels.
      - Classifier: Adaptive Average Pooling -> Fully Connected Layer -> Softmax output (2 classes).
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        se_reduction: int = 16,
        dropout_rate: float = 0.3
    ):
        """
        Parameters
        ----------
        in_channels : int, default=1
            Number of input channels (1 for single-channel ECG spectrograms).
        num_classes : int, default=2
            Number of target cognitive load classes (0=Low Load, 1=High Load).
        conv_channels : Tuple[int, int, int], default=(32, 64, 128)
            Number of output filters for the 3 Conv2D blocks.
        se_reduction : int, default=16
            Reduction ratio for the SE block FC layers.
        dropout_rate : float, default=0.3
            Dropout probability before the final classifier FC layer.
        """
        super().__init__()
        c1, c2, c3 = conv_channels

        # 1. Convolution Module: 3 ConvBlock2D layers
        self.conv1 = ConvBlock2D(in_channels, c1, kernel_size=3, padding=1, pool_size=(2, 2))
        self.conv2 = ConvBlock2D(c1, c2, kernel_size=3, padding=1, pool_size=(2, 2))
        self.conv3 = ConvBlock2D(c2, c3, kernel_size=3, padding=1, pool_size=(2, 2))

        # 2. SE-Inception Attention Module
        self.se_block = SEBlock(channels=c3, reduction=se_reduction)

        # 3. Global Pooling & Classification Output Layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(c3, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the penultimate global-pooled CNN embedding (Batch, c3) BEFORE the
        classifier. Used by the NSA wrapper to fuse symbolic features without altering
        the paper model's classifier path.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se_block(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass of SE-ECG Net.

        Parameters
        ----------
        x : torch.Tensor
            Input ECG spectrogram tensor of shape (Batch, 1, Freq_Bins, Time_Frames).
        return_logits : bool, default=False
            If True, returns raw unscaled logits (ideal for nn.CrossEntropyLoss training).
            If False, returns Softmax class probabilities (summing to 1.0 per sample).

        Returns
        -------
        torch.Tensor
            Tensor of shape (Batch, num_classes) containing probabilities or raw logits.
        """
        # 1. Feature Extraction via 3 Conv Blocks
        x = self.conv1(x)  # Shape: (B, c1, F/2, T/2)
        x = self.conv2(x)  # Shape: (B, c2, F/4, T/4)
        x = self.conv3(x)  # Shape: (B, c3, F/8, T/8)

        # 2. Squeeze-and-Excitation Channel Reweighting
        x = self.se_block(x)  # Shape: (B, c3, F/8, T/8)

        # 3. Global Pooling & Classifier
        x = self.global_pool(x)  # Shape: (B, c3, 1, 1)
        x = torch.flatten(x, 1)   # Shape: (B, c3)
        x = self.dropout(x)
        logits = self.classifier(x)  # Shape: (B, num_classes)

        if return_logits:
            return logits

        # Output probabilities via Softmax
        probs = self.softmax(logits)
        return probs


if __name__ == "__main__":
    print("Testing SE-ECG Net Architecture...")

    # Instantiate model
    model = SEECGNet(in_channels=1, num_classes=2, conv_channels=(32, 64, 128))
    model.eval()

    # Create dummy STFT spectrogram batch tensor: (Batch=4, Channels=1, Freq_Bins=64, Time_Frames=344)
    dummy_input = torch.randn(4, 1, 64, 344)

    # Perform forward pass
    with torch.no_grad():
        probs = model(dummy_input, return_logits=False)
        logits = model(dummy_input, return_logits=True)

    print(f"Input Spectrogram batch shape: {dummy_input.shape}")
    print(f"Output Class Probabilities shape: {probs.shape}")
    print(f"Sample Output Probabilities (Sample 0): {probs[0].numpy()}")
    print(f"Probabilities sum check: {probs[0].sum().item():.4f}")
    print(f"Output Logits shape: {logits.shape}")

    # Print model structure summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SE-ECG Net Total Trainable Parameters: {total_params:,}")
