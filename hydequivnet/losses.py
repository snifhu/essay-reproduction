from __future__ import annotations

import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """MSE + NSE-normalized term + optional proportional-response consistency."""

    def __init__(
        self,
        nse_weight: float = 0.2,
        consistency_weight: float = 2.0,
        lambda_min: float = 2.0,
        lambda_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.nse_weight = nse_weight
        self.consistency_weight = consistency_weight
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module | None = None,
        x_input: torch.Tensor | None = None,
        x_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_mse = self.mse(pred, target)
        loss_nse = self.nse_weight * (loss_mse / (torch.var(target) + 1e-6))
        loss_cons = pred.new_tensor(0.0)

        if (
            self.consistency_weight > 0
            and model is not None
            and model.training
            and x_input is not None
            and x_static is not None
        ):
            lam = torch.rand(1, device=pred.device) * (self.lambda_max - self.lambda_min) + self.lambda_min
            x_scaled = x_input.clone()
            x_scaled[:, :, 0] *= lam
            pred_scaled = model(x_scaled, x_static)

            q_base = torch.exp(pred)
            q_scaled = torch.exp(pred_scaled)
            target_cons = q_base * lam
            loss_cons = torch.mean((q_scaled - target_cons) ** 2 / (target_cons.detach() ** 2 + 1e-2))
            loss_cons = self.consistency_weight * loss_cons

        return loss_mse + loss_nse + loss_cons

