from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def extract_base_kernels(model: torch.nn.Module, scale_index: int = 0, input_index: int = 0) -> np.ndarray:
    """Extract learned basis-composed kernels from the HEN lift convolution."""
    if not hasattr(model, "lift_conv"):
        raise AttributeError("Model does not have a lift_conv layer.")
    layer = model.lift_conv
    with torch.no_grad():
        kernels = layer.basis_gen(layer.basis_weights)
        base = kernels[scale_index, :, input_index, :]
    return base.detach().cpu().numpy()


def derive_total_unit_hydrograph(
    model: torch.nn.Module,
    device: torch.device,
    pulse_magnitude: float = 10.0,
    background_rain: float = 0.1,
    max_lag: int = 15,
    seq_len: int = 100,
    area_km2: float | None = None,
    *,
    static_vec: np.ndarray | torch.Tensor | None = None,
    static_mode: str = "basin_static",
) -> dict[str, np.ndarray | float | int | str | None]:
    """Estimate a normalized impulse-derived EUH from a trained model."""
    model.eval()
    input_dim = model.lift_conv.basis_weights.shape[1] if hasattr(model, "lift_conv") else 6
    static_dim = model.static_embed[0].in_features if hasattr(model, "static_embed") else 5

    base_x = torch.zeros(1, seq_len, input_dim, device=device)
    base_x[:, :, 0] = background_rain
    if static_mode == "basin_static":
        if static_vec is None:
            raise ValueError("static_vec is required when static_mode='basin_static'")
        base_s = torch.as_tensor(static_vec, dtype=torch.float32, device=device).view(1, -1)
        if base_s.shape[1] != static_dim:
            raise ValueError(f"Expected static_dim={static_dim}, got {base_s.shape[1]}")
    elif static_mode == "zero_reference":
        if static_vec is not None:
            raise ValueError("Do not pass static_vec when static_mode='zero_reference'")
        base_s = torch.zeros(1, static_dim, device=device)
    else:
        raise ValueError("static_mode must be 'basin_static' or 'zero_reference'")

    batch_x = base_x.repeat(max_lag, 1, 1)
    batch_s = base_s.repeat(max_lag, 1)
    for lag in range(max_lag):
        batch_x[lag, seq_len - 1 - lag, 0] += pulse_magnitude

    with torch.no_grad():
        pred_base = torch.exp(model(base_x, base_s)).item()
        pred_pulse = torch.exp(model(batch_x, batch_s)).detach().cpu().numpy()

    raw = (pred_pulse - pred_base) / pulse_magnitude
    positive = np.maximum(raw, 0.0)
    total = np.sum(positive)
    normalized = positive / total if total > 1e-9 else np.zeros_like(positive)

    if area_km2 is not None:
        response = normalized * (area_km2 * 1000.0 / 86400.0)
    else:
        response = normalized

    peak_lag = int(np.argmax(response))
    peak_value = float(response[peak_lag])
    return {
        "response": response,
        "raw_response": raw,
        "peak_lag": peak_lag,
        "peak_value": peak_value,
        "area_km2": area_km2,
        "static_mode": static_mode,
    }


def save_unit_hydrograph_csv(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    response = np.asarray(result["response"])
    data = np.column_stack([np.arange(len(response)), response])
    np.savetxt(path, data, delimiter=",", header="lag,response", comments="")
