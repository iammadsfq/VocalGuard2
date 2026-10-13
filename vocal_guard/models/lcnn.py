import torch
import torch.nn as nn


class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map (MFM) for Conv2D.
    Splits input channels (2 * C) into two halves and computes elementwise maximum.
    Effectively performs competitive channel feature selection.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        return torch.max(x[:, :c, :, :], x[:, c:, :, :])


class MaxFeatureMap1D(nn.Module):
    """Max-Feature-Map (MFM) for Linear 1D layers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.size(1) // 2
        return torch.max(x[:, :d], x[:, d:])


class ConvMFM2D(nn.Module):
    """Convolution + BatchNorm + MFM activation block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(2 * out_channels)
        self.mfm = MaxFeatureMap2D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class LinearMFM(nn.Module):
    """Linear + MFM activation block."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * out_features)
        self.mfm = MaxFeatureMap1D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.linear(x))


class LCNN(nn.Module):
    """
    Light CNN (LCNN) architecture for Audio Anti-Spoofing.
    Based on Wu et al. (2018) & Lavrentyeva et al. (Interspeech 2017/2019).

    Input:
        x: (Batch, 1, n_lfcc, n_frames) e.g., (B, 1, 60, T)
    Output:
        logits: (Batch, 2) where:
            index 0 -> logit for 'bonafide'
            index 1 -> logit for 'spoof'
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 1, dropout_p: float = 0.5):
        super().__init__()

        # Block 1
        self.layer1 = nn.Sequential(
            ConvMFM2D(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.layer2 = nn.Sequential(
            ConvMFM2D(32, 32, kernel_size=3, stride=1, padding=1),
            ConvMFM2D(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.layer3 = nn.Sequential(
            ConvMFM2D(48, 48, kernel_size=3, stride=1, padding=1),
            ConvMFM2D(48, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4
        self.layer4 = nn.Sequential(
            ConvMFM2D(64, 64, kernel_size=3, stride=1, padding=1),
            ConvMFM2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to support variable audio durations
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 4))
        self.flat_dim = 64 * 3 * 4  # 768

        # Fully connected head
        self.fc1 = LinearMFM(self.flat_dim, 160)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(160, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck embedding vector before classification."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        emb = self.fc1(x)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_features(x)
        emb = self.dropout(emb)
        logits = self.fc2(emb)
        return logits
