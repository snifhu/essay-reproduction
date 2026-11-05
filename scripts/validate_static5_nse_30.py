from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydequivnet.config import ExperimentConfig
from hydequivnet.data import CatchmentDataset, STATIC_ATTRIBUTE_DIM, SplitDates
from hydequivnet.train import build_model, evaluate_model


DEFAULT_DATA_DIR = ROOT / "data" / "catchment_dataset"
DEFAULT_CHECKPOINT_DIR = ROOT.parent / "checkpoint"
DEFAULT_METRICS_FILE = ROOT / "results" / "model_metrics_531_basin_wide.csv"
DEFAULT_OUTPUT_DIR = ROOT / "validation_outputs"
CHECKPOINT_SPLIT_DATES = SplitDates(
    train_start="1980-01-01",
    train_end="2003-12-31",
    val_start="2000-01-01",
    val_end="2003-12-31",
    test_start="2004-01-01",
    test_end="2008-12-31",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load 30 released HydEquivNet checkpoints with the explicit "
            "five-dimensional static input and compare recomputed test NSE "
            "against model_metrics_531_basin_wide.csv."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    return parser.parse_args()


def select_evenly_spaced_basins(metrics: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("--count must be positive")
    if count > len(metrics):
        raise ValueError(f"Requested {count} basins, but metrics contain only {len(metrics)}")
    indices = np.linspace(0, len(metrics) - 1, num=count, dtype=int)
    if len(np.unique(indices)) != count:
        raise RuntimeError("Evenly spaced basin selection produced duplicate rows")
    return metrics.iloc[indices].copy()


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def configure_reproducible_inference() -> None:
    """Match the deterministic IEEE-FP32 path used for the released metrics."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    else:  # Compatibility with older PyTorch releases.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def evaluate_one(
    basin_id: str,
    reference_nse: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = args.checkpoint_dir / f"final_exp_{basin_id}_HydEquivNet_Causal.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    train_ds = CatchmentDataset(
        basin_id,
        str(args.data_dir),
        seq_length=365,
        split="train",
        split_dates=CHECKPOINT_SPLIT_DATES,
    )
    test_ds = CatchmentDataset(
        basin_id,
        str(args.data_dir),
        seq_length=365,
        split="test",
        split_dates=CHECKPOINT_SPLIT_DATES,
        scaler=train_ds.scaler_max,
    )
    if int(test_ds.static_dim) != STATIC_ATTRIBUTE_DIM:
        raise AssertionError(
            f"{basin_id}: loader returned static_dim={test_ds.static_dim}, "
            f"expected {STATIC_ATTRIBUTE_DIM}"
        )

    state_dict = load_state_dict(checkpoint)
    checkpoint_static_dim = int(state_dict["static_embed.0.weight"].shape[1])
    if checkpoint_static_dim != STATIC_ATTRIBUTE_DIM:
        raise AssertionError(
            f"{basin_id}: checkpoint expects {checkpoint_static_dim} static inputs, "
            f"expected {STATIC_ATTRIBUTE_DIM}"
        )

    cfg = ExperimentConfig(data_dir=str(args.data_dir))
    model = build_model("hen", test_ds.input_dim, test_ds.static_dim, cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    recomputed_nse = float(evaluate_model(model, loader, device)["nse"])
    abs_difference = abs(recomputed_nse - reference_nse)

    return {
        "basin": basin_id,
        "static_dim_loader": int(test_ds.static_dim),
        "static_dim_checkpoint": checkpoint_static_dim,
        "static_values": json.dumps(test_ds.static_data.tolist()),
        "nse_reference": reference_nse,
        "nse_recomputed": recomputed_nse,
        "nse_reference_3dp": round(reference_nse, 3),
        "nse_recomputed_3dp": round(recomputed_nse, 3),
        "same_at_3dp": round(reference_nse, 3) == round(recomputed_nse, 3),
        "abs_difference": abs_difference,
        "within_tolerance": abs_difference <= args.atol,
        "checkpoint": str(checkpoint.resolve()),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_reproducible_inference()

    metrics = pd.read_csv(args.metrics_file, dtype={"basin": str})
    required_columns = {"basin", "nse_hen"}
    missing = required_columns.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics file is missing columns: {sorted(missing)}")
    metrics["basin"] = metrics["basin"].str.zfill(8)
    selected = select_evenly_spaced_basins(metrics, args.count)

    device = torch.device(args.device)
    print(f"device={device}; basins={len(selected)}; atol={args.atol:g}")
    records: list[dict[str, object]] = []
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        basin_id = str(row.basin).zfill(8)
        record = evaluate_one(basin_id, float(row.nse_hen), args, device)
        records.append(record)
        print(
            f"[{position:02d}/{len(selected):02d}] basin={basin_id} "
            f"static_dim={record['static_dim_loader']} "
            f"NSE(ref={record['nse_reference']:.10f}, "
            f"new={record['nse_recomputed']:.10f}, "
            f"abs_diff={record['abs_difference']:.3e}) "
            f"pass={record['within_tolerance']}"
        )

    output = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "static5_nse_validation_30_basins.csv"
    summary_path = args.output_dir / "static5_nse_validation_30_summary.json"
    output.to_csv(detail_path, index=False)

    summary = {
        "metrics_file": str(args.metrics_file.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "selection": "30 evenly spaced rows across the 531-basin metrics table",
        "checkpoint_scaler_fit_period": "1980-01-01 through 2003-12-31",
        "test_period": "2004-01-01 through 2008-12-31",
        "numeric_mode": "deterministic IEEE FP32 (TF32 disabled)",
        "count": int(len(output)),
        "static_attribute_dim": STATIC_ATTRIBUTE_DIM,
        "absolute_tolerance": args.atol,
        "passed": int(output["within_tolerance"].sum()),
        "failed": int((~output["within_tolerance"]).sum()),
        "same_at_3dp": int(output["same_at_3dp"].sum()),
        "max_abs_difference": float(output["abs_difference"].max()),
        "mean_abs_difference": float(output["abs_difference"].mean()),
        "all_passed": bool(output["within_tolerance"].all()),
        "detail_csv": str(detail_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
