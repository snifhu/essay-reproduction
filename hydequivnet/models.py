from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ScaleEquivConv1d


class LSTMBaseline(nn.Module):
    """Local basin-wise LSTM baseline using dynamic and static inputs."""

    name = "Baseline_LSTM"

    def __init__(self, input_dim: int, static_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.static_fc = nn.Linear(static_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_dyn)
        dyn_feat = h_n[-1]
        stat_feat = torch.tanh(self.static_fc(x_stat))
        return self.head(torch.cat([dyn_feat, stat_feat], dim=1)).squeeze(1)


class HydEquivNet(nn.Module):
    """HydEquivNet rainfall-runoff model.

    The model combines a scale-equivariant Hermite routing backbone with an
    LSTM residual sequence module and static catchment embeddings.
    """

    name = "HydEquivNet_Causal"

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        n_scales: int = 4,
        kernel_size: int = 15,
        n_bases: int = 3,
        feature_dim: int = 16,
        projected_dim: int = 32,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_scales = n_scales

        self.lift_conv = ScaleEquivConv1d(
            input_dim,
            feature_dim,
            kernel_size=kernel_size,
            n_scales=n_scales,
            n_bases=n_bases,
        )
        self.feature_proj = nn.Linear(feature_dim * n_scales, projected_dim)
        self.lstm = nn.LSTM(input_dim + projected_dim, hidden_dim, batch_first=True)
        self.static_embed = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        x_conv = x_dyn.transpose(1, 2)
        lifted = self.lift_conv(x_conv)
        batch, out_channels, n_scales, time = lifted.shape
        lifted = lifted.permute(0, 3, 1, 2).reshape(batch, time, out_channels * n_scales)
        phys_feats = F.relu(self.feature_proj(lifted))

        lstm_in = torch.cat([x_dyn, phys_feats], dim=2)
        _, (h_n, _) = self.lstm(lstm_in)
        dyn_feat = h_n[-1]
        stat_feat = self.static_embed(x_stat)
        return self.head(torch.cat([dyn_feat, stat_feat], dim=1)).squeeze(1)

