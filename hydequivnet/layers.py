from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import SteerableScaleBasis


class ScaleEquivConv1d(nn.Module):
    """Causal multi-scale convolution using steerable Hermite routing bases."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        n_scales: int = 4,
        n_bases: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_scales = n_scales
        self.stride = stride

        scales = [2.0**i for i in range(n_scales)]
        self.basis_gen = SteerableScaleBasis(
            kernel_size=kernel_size,
            n_bases=n_bases,
            scales=scales,
        )
        self.basis_weights = nn.Parameter(torch.randn(out_channels, in_channels, n_bases))
        nn.init.normal_(self.basis_weights, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Tensor with shape [batch, channels, time].

        Returns:
            Tensor with shape [batch, out_channels, scale, time].
        """
        kernels = self.basis_gen(self.basis_weights)
        n_scales, out_channels, in_channels, kernel_size = kernels.shape
        filters = kernels.reshape(n_scales * out_channels, in_channels, kernel_size)

        x_padded = F.pad(x, (self.kernel_size - 1, 0))
        out = F.conv1d(x_padded, filters, stride=self.stride)
        batch, _, time = out.shape
        return out.reshape(batch, n_scales, out_channels, time).permute(0, 2, 1, 3)

