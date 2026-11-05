from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .data import (
    SplitDates,
    create_dataloaders,
    euh_static_vector_from_dict,
    load_euh_static_attribute_dict,
)
from .plotting import plot_kernels, plot_unit_hydrograph
from .train import build_model, evaluate_model, proportional_response_error, train_model
from .unit_hydrograph import derive_total_unit_hydrograph, extract_base_kernels, save_unit_hydrograph_csv
from .utils import ensure_dir


def run_single_basin(basin_id: str, cfg, device: torch.device) -> dict[str, float | str]:
    split_dates = SplitDates(
        train_start=cfg.train_start_date,
        train_end=cfg.train_end_date,
        val_start=cfg.val_start_date,
        val_end=cfg.val_end_date,
        test_start=cfg.test_start_date,
        test_end=cfg.test_end_date,
    )
    loaders = create_dataloaders(
        basin_id,
        cfg.data_dir,
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        split_dates=split_dates,
    )
    input_dim = loaders["train"].dataset.input_dim
    static_dim = loaders["train"].dataset.static_dim
    area_km2 = loaders["train"].dataset.area_km2

    output_dir = ensure_dir(cfg.output_dir)
    checkpoint_dir = ensure_dir(output_dir / cfg.checkpoint_dir)
    figure_dir = ensure_dir(output_dir / "figures")
    uh_dir = ensure_dir(output_dir / "unit_hydrographs")

    result: dict[str, float | str] = {"basin": basin_id}

    if cfg.train_lstm_baseline:
        lstm = build_model("lstm", input_dim, static_dim, cfg).to(device)
        lstm_ckpt = checkpoint_dir / f"{basin_id}_{lstm.name}.pth"
        if cfg.skip_existing and lstm_ckpt.exists():
            lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device))
        else:
            train_model(
                lstm,
                loaders,
                cfg,
                device,
                lstm_ckpt,
                use_consistency=False,
                epochs=cfg.lstm_baseline_epochs,
            )
            lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device))
        lstm_metrics = evaluate_model(lstm, loaders["test"], device)
        result.update({f"lstm_{key}": val for key, val in lstm_metrics.items()})

    hen = build_model("hen", input_dim, static_dim, cfg).to(device)
    hen_ckpt = checkpoint_dir / f"{basin_id}_{hen.name}.pth"
    if cfg.skip_existing and hen_ckpt.exists():
        hen.load_state_dict(torch.load(hen_ckpt, map_location=device))
    else:
        train_model(hen, loaders, cfg, device, hen_ckpt, use_consistency=True, epochs=cfg.epochs)
        hen.load_state_dict(torch.load(hen_ckpt, map_location=device))

    hen_metrics = evaluate_model(hen, loaders["test"], device)
    result.update({f"hen_{key}": val for key, val in hen_metrics.items()})
    for lam in (2.0, 5.0, 10.0):
        result[f"hen_pr_error_lambda_{lam:g}"] = proportional_response_error(
            hen,
            loaders["test"],
            device,
            lam=lam,
        )

    kernels = extract_base_kernels(hen)
    plot_kernels(kernels, figure_dir / f"{basin_id}_kernels.png")

    euh_static_attributes = load_euh_static_attribute_dict(basin_id, cfg.data_dir)
    euh_static_vec = euh_static_vector_from_dict(euh_static_attributes)
    uh = derive_total_unit_hydrograph(
        hen,
        device,
        area_km2=area_km2,
        static_vec=euh_static_vec,
        static_mode="basin_static",
    )
    save_unit_hydrograph_csv(uh, uh_dir / f"{basin_id}_unit_hydrograph.csv")
    plot_unit_hydrograph(
        uh["response"],
        figure_dir / f"{basin_id}_unit_hydrograph.png",
        title=f"Derived Unit Hydrograph ({basin_id})",
    )
    result["peak_lag"] = float(uh["peak_lag"])
    result["peak_value"] = float(uh["peak_value"])
    result["area_km2"] = float(area_km2)
    result["euh_static_mode"] = str(uh["static_mode"])
    result["euh_static_attributes"] = json.dumps(euh_static_attributes, separators=(",", ":"))
    return result


def save_results(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
