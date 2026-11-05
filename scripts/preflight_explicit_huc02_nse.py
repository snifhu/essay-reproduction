from __future__ import annotations

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
from hydequivnet.data import CatchmentDataset, SplitDates
from hydequivnet.train import build_model, evaluate_model


BASINS = ("01022500", "14185000", "14400000")
DATA_DIR = ROOT / "data" / "catchment_dataset"
CHECKPOINT_DIR = ROOT.parent / "checkpoint"
METRICS_FILE = ROOT / "results" / "model_metrics_531_basin_wide.csv"
OUTPUT_FILE = ROOT / "validation_outputs" / "explicit_huc02_nse_preflight_3_basins.csv"
STATIC_COLUMNS = (
    "BASIN_HUC",
    "Size(km2)",
    "Elevation(m)",
    "Slope(m_km-1)",
    "Frac_Forest(percent)",
)
CHECKPOINT_DATES = SplitDates(train_end="2003-12-31")


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


def load_named_static_table() -> pd.DataFrame:
    files = list(DATA_DIR.glob("**/catchment_attributes.txt"))
    if not files:
        raise FileNotFoundError("catchment_attributes.txt was not found")
    attrs = pd.read_csv(files[0], sep=r"\s+", dtype={"BASIN_ID": str, "BASIN_HUC": str})
    attrs["BASIN_ID"] = attrs["BASIN_ID"].astype(str).str.zfill(8)
    return attrs.set_index("BASIN_ID")


def main() -> None:
    configure_reproducible_inference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attrs = load_named_static_table()
    metrics = pd.read_csv(METRICS_FILE, dtype={"basin": str}).set_index("basin")
    cfg = ExperimentConfig(data_dir=str(DATA_DIR))
    rows: list[dict[str, object]] = []

    for basin_id in BASINS:
        train_ds = CatchmentDataset(
            basin_id,
            str(DATA_DIR),
            split="train",
            split_dates=CHECKPOINT_DATES,
        )
        test_ds = CatchmentDataset(
            basin_id,
            str(DATA_DIR),
            split="test",
            split_dates=CHECKPOINT_DATES,
            scaler=train_ds.scaler_max,
        )
        checkpoint_static = test_ds.static_data.clone()
        named_row = attrs.loc[basin_id, list(STATIC_COLUMNS)]
        named_static = pd.to_numeric(named_row, errors="raise").to_numpy(dtype=np.float32)
        test_ds.static_data = torch.as_tensor(named_static, dtype=torch.float32)

        checkpoint = CHECKPOINT_DIR / f"final_exp_{basin_id}_HydEquivNet_Causal.pth"
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        model = build_model("hen", test_ds.input_dim, test_ds.static_dim, cfg).to(device)
        model.load_state_dict(state, strict=True)
        nse_named = float(
            evaluate_model(model, DataLoader(test_ds, batch_size=64, shuffle=False), device)["nse"]
        )
        nse_reference = float(metrics.loc[basin_id, "nse_hen"])
        rows.append(
            {
                "basin": basin_id,
                "checkpoint_static": checkpoint_static.tolist(),
                "explicit_huc02_static": named_static.tolist(),
                "nse_reference": nse_reference,
                "nse_explicit_huc02": nse_named,
                "abs_difference": abs(nse_named - nse_reference),
            }
        )
        print(rows[-1])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
