from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydequivnet.data import load_basin_data, load_euh_static_attributes
from hydequivnet.models import HydEquivNet
from hydequivnet.unit_hydrograph import derive_total_unit_hydrograph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate basin-specific versus zero-reference EUH diagnostics for 30 basins."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "catchment_dataset")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT.parent / "checkpoint")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=ROOT / "results" / "model_metrics_531_basin_wide.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation_outputs")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--shape-atol", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def configure_reproducible_inference() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select_basins(metrics_file: Path, count: int) -> list[str]:
    metrics = pd.read_csv(metrics_file, dtype={"basin": str})
    if count < 1 or count > len(metrics):
        raise ValueError(f"--count must be between 1 and {len(metrics)}")
    indices = np.linspace(0, len(metrics) - 1, num=count, dtype=int)
    return metrics.iloc[indices]["basin"].astype(str).str.zfill(8).tolist()


def response_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def validate_one(basin_id: str, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    checkpoint = args.checkpoint_dir / f"final_exp_{basin_id}_HydEquivNet_Causal.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    _df, features, _dataset_static, area_km2 = load_basin_data(basin_id, str(args.data_dir))
    basin_static = load_euh_static_attributes(basin_id, str(args.data_dir))
    state_dict = load_state_dict(checkpoint)
    checkpoint_static_dim = int(state_dict["static_embed.0.weight"].shape[1])
    if basin_static.shape != (checkpoint_static_dim,):
        raise AssertionError(
            f"{basin_id}: EUH static shape={basin_static.shape}, checkpoint static_dim={checkpoint_static_dim}"
        )

    model = HydEquivNet(input_dim=len(features), static_dim=checkpoint_static_dim).to(device)
    model.load_state_dict(state_dict, strict=True)

    basin_result = derive_total_unit_hydrograph(
        model,
        device,
        area_km2=area_km2,
        static_vec=basin_static,
    )
    zero_result = derive_total_unit_hydrograph(
        model,
        device,
        area_km2=area_km2,
        static_mode="zero_reference",
    )

    basin_response = np.asarray(basin_result["response"], dtype=float)
    zero_response = np.asarray(zero_result["response"], dtype=float)
    denominator = float(np.sum(np.abs(zero_response)) + 1e-12)
    relative_l1 = float(np.sum(np.abs(basin_response - zero_response)) / denominator)
    delta_peak_lag = int(basin_result["peak_lag"]) - int(zero_result["peak_lag"])
    passed = (
        basin_result["static_mode"] == "basin_static"
        and zero_result["static_mode"] == "zero_reference"
        and delta_peak_lag == 0
        and relative_l1 <= args.shape_atol
    )
    return {
        "basin": basin_id,
        "checkpoint_static_dim": checkpoint_static_dim,
        "euh_static_values": json.dumps(basin_static.tolist()),
        "basin_static_mode": basin_result["static_mode"],
        "zero_static_mode": zero_result["static_mode"],
        "basin_static_peak_lag": int(basin_result["peak_lag"]),
        "zero_reference_peak_lag": int(zero_result["peak_lag"]),
        "delta_peak_lag": delta_peak_lag,
        "response_pearson_r": response_correlation(basin_response, zero_response),
        "response_relative_l1": relative_l1,
        "passed": passed,
        "checkpoint": str(checkpoint.resolve()),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_reproducible_inference()
    device = torch.device(args.device)
    basins = select_basins(args.metrics_file, args.count)

    rows: list[dict[str, object]] = []
    for index, basin_id in enumerate(basins, start=1):
        row = validate_one(basin_id, args, device)
        rows.append(row)
        print(
            f"[{index:02d}/{len(basins):02d}] basin={basin_id} "
            f"peak_delta={row['delta_peak_lag']} "
            f"shape_l1={row['response_relative_l1']:.3e} pass={row['passed']}"
        )

    detail = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "euh_static_validation_30_basins.csv"
    summary_path = args.output_dir / "euh_static_validation_30_summary.json"
    detail.to_csv(detail_path, index=False)

    correlations = pd.to_numeric(detail["response_pearson_r"], errors="coerce")
    summary = {
        "count": int(len(detail)),
        "passed": int(detail["passed"].sum()),
        "failed": int((~detail["passed"]).sum()),
        "same_peak_lag": int((detail["delta_peak_lag"] == 0).sum()),
        "max_abs_delta_peak_lag": int(detail["delta_peak_lag"].abs().max()),
        "median_response_pearson_r": float(correlations.median()),
        "median_response_relative_l1": float(detail["response_relative_l1"].median()),
        "max_response_relative_l1": float(detail["response_relative_l1"].max()),
        "shape_absolute_tolerance": args.shape_atol,
        "all_passed": bool(detail["passed"].all()),
        "detail_csv": str(detail_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
