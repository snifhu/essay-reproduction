from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import HybridLoss
from .metrics import kge, nse, pearson_r
from .models import HydEquivNet, LSTMBaseline


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, s, y in loader:
            pred = model(x.to(device), s.to(device))
            preds.append(pred.cpu())
            targets.append(y)

    sim = torch.exp(torch.cat(preds)).numpy()
    obs = torch.exp(torch.cat(targets)).numpy()
    return {
        "nse": nse(obs, sim),
        "kge": kge(obs, sim),
        "pearson_r": pearson_r(obs, sim),
    }


def proportional_response_error(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lam: float = 2.0,
) -> float:
    model.eval()
    try:
        x, s, _ = next(iter(loader))
    except StopIteration:
        return float("nan")
    x = x.to(device)
    s = s.to(device)
    with torch.no_grad():
        pred_base = torch.exp(model(x, s))
        x_scaled = x.clone()
        x_scaled[:, :, 0] *= lam
        pred_scaled = torch.exp(model(x_scaled, s))
        target = pred_base * lam
        err = torch.mean(torch.abs(pred_scaled - target) / (target + 1e-6))
    return float(err.item())


def build_model(model_name: str, input_dim: int, static_dim: int, cfg) -> torch.nn.Module:
    if model_name == "lstm":
        return LSTMBaseline(input_dim, static_dim, hidden_dim=cfg.hidden_dim)
    if model_name == "hen":
        return HydEquivNet(
            input_dim,
            static_dim,
            hidden_dim=cfg.hidden_dim,
            n_scales=cfg.n_scales,
            kernel_size=cfg.kernel_size,
            n_bases=cfg.n_bases,
            feature_dim=cfg.feature_dim,
            projected_dim=cfg.projected_dim,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def train_model(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    cfg,
    device: torch.device,
    checkpoint_path: str | Path,
    use_consistency: bool = True,
    epochs: int | None = None,
) -> float:
    epochs = cfg.epochs if epochs is None else epochs
    model.to(device)
    criterion = HybridLoss(
        nse_weight=cfg.nse_weight,
        consistency_weight=cfg.consistency_weight if use_consistency else 0.0,
        lambda_min=cfg.consistency_lambda_min,
        lambda_max=cfg.consistency_lambda_max,
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_nse = -np.inf

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        loop = tqdm(loaders["train"], leave=False, desc=f"epoch {epoch + 1}/{epochs}")
        for x, s, y in loop:
            x = x.to(device)
            s = s.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, s)
            if use_consistency:
                loss = criterion(pred, y, model=model, x_input=x, x_static=s)
            else:
                loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_metrics = evaluate_model(model, loaders["val"], device)
        val_nse = val_metrics["nse"]
        print(
            f"epoch={epoch + 1:03d} "
            f"loss={loss_sum / max(1, len(loaders['train'])):.4f} "
            f"val_nse={val_nse:.4f}"
        )
        if val_nse > best_nse:
            best_nse = val_nse
            torch.save(model.state_dict(), checkpoint_path)

    return float(best_nse)

