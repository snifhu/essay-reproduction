from __future__ import annotations

import torch
import torch.nn as nn


class SteerableScaleBasis(nn.Module):
    """Pre-computed scale-indexed Hermite/Gaussian basis bank.

    The zero-order Gaussian term and the first two Hermite-like derivative
    terms provide a compact routing-kernel basis motivated by the analytical
    diffusion-wave/unit-hydrograph response form. This module builds basis
    functions; it does not solve a diffusion-wave PDE.
    """

    def __init__(
        self,
        kernel_size: int = 15,
        n_bases: int = 3,
        scales: list[float] | tuple[float, ...] | None = None,
        time_extent: float = 2.5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_bases = n_bases
        if scales is None:
            scales = [1.0]

        t = torch.linspace(0, time_extent, kernel_size)
        all_bases = []
        for scale in scales:
            t_scaled = t / (float(scale) + 1e-6)
            bases = [
                torch.exp(-(t_scaled**2) / 0.5),
                t_scaled * torch.exp(-(t_scaled**2) / 0.5),
                (t_scaled**2 - 0.5) * torch.exp(-(t_scaled**2) / 0.5),
            ]
            all_bases.append(torch.stack(bases[:n_bases]))

        self.register_buffer("basis_bank", torch.stack(all_bases).float())

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Return kernels with shape [scale, out_channels, in_channels, kernel]."""
        return torch.einsum("oib,sbk->soik", weights, self.basis_bank)

