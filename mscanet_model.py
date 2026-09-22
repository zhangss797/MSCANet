"""MSCANet model definition for EEG-based Parkinson's disease classification.

This module contains the multi-scale channel attention (MSCA) block and the
complete MSCANet architecture used in the accompanying experiments.
"""

import torch
import torch.nn as nn


class MSCABlock(nn.Module):
    """Multi-scale channel attention block.

    Parallel depthwise temporal convolutions capture complementary temporal
    contexts. A global descriptor is then used to generate scale-specific,
    channel-wise attention weights. Softmax normalization is applied across
    the scale dimension for each latent feature channel.

    Args:
        channels: Number of latent feature channels.
        reduction: Reduction ratio used by the attention bottleneck.
        kernel_sizes: Temporal kernel sizes used by the parallel branches.
        use_residual: Whether to add the input through a residual connection.
    """

    def __init__(
        self,
        channels: int = 16,
        reduction: int = 4,
        kernel_sizes=(7, 25, 51),
        use_residual: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = tuple(kernel_sizes)
        self.use_residual = use_residual

        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=k,
                        padding=k // 2,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.ELU(inplace=True),
                )
                for k in self.kernel_sizes
            ]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        hidden_dim = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden_dim, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, channels * len(self.kernel_sizes), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale temporal extraction and adaptive scale fusion.

        Args:
            x: Input tensor with shape ``[batch, channels, time]``. A singleton
               spatial dimension is also accepted and removed automatically.

        Returns:
            Tensor with shape ``[batch, channels, time]``.
        """
        if x.dim() == 4:
            x = x.squeeze(2)

        residual = x
        path_outputs = [path(x) for path in self.paths]

        # Aggregate multi-scale features to construct a global context vector.
        aggregated = torch.stack(path_outputs, dim=0).sum(dim=0)
        pooled = self.global_pool(aggregated).squeeze(-1)

        # Shape: [batch, scales, channels].
        attention = self.fc2(self.activation(self.fc1(pooled)))
        attention = attention.view(x.size(0), len(self.paths), self.channels)

        # Normalize across temporal scales independently for each channel.
        attention = torch.softmax(attention, dim=1)

        output = torch.zeros_like(path_outputs[0])
        for scale_idx, path_output in enumerate(path_outputs):
            weight = attention[:, scale_idx, :].unsqueeze(-1)
            output = output + path_output * weight

        if self.use_residual:
            output = output + residual

        return output


class EEG_MSCA_Net(nn.Module):
    """MSCANet for binary EEG classification."""

    def __init__(
        self,
        n_channels: int = 57,
        n_classes: int = 2,
        base_channels: int = 16,
        mid_channels: int = 16,
        out_channels: int = 64,
        temporal_kernel: int = 25,
        msca_reduction: int = 4,
        msca_kernel_sizes=(7, 25, 51),
        dropout: float = 0.3,
    ):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                n_channels,
                base_channels,
                kernel_size=temporal_kernel,
                padding=temporal_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.msca = MSCABlock(
            channels=base_channels,
            reduction=msca_reduction,
            kernel_sizes=msca_kernel_sizes,
            use_residual=True,
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(base_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, n_classes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent feature vector before the final classifier."""
        x = self.temporal_conv(x)
        x = self.msca(x)
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.msca(x)
        x = self.feature_extractor(x)
        return self.classifier(x)
