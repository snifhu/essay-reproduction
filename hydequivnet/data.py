from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset



PHYSICAL_ATTRIBUTE_COLUMNS = (
    "Size(km2)",
    "Elevation(m)",
    "Slope(m_km-1)",
    "Frac_Forest(percent)",
)
STATIC_INPUT_COLUMNS = ("BASIN_HUC", *PHYSICAL_ATTRIBUTE_COLUMNS)
STATIC_ATTRIBUTE_DIM = len(STATIC_INPUT_COLUMNS)


@dataclass
class SplitDates:
    train_start: str = "1980-01-01"
    train_end: str = "1999-12-31"
    val_start: str = "2000-01-01"
    val_end: str = "2003-12-31"
    test_start: str = "2004-01-01"
    test_end: str = "2008-12-31"


def get_all_basins(data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, "**", "*_observed_discharge.txt")
    files = glob.glob(pattern, recursive=True)
    basins = []
    for file in files:
        basin = os.path.basename(file).split("_")[0]
        if basin.isdigit():
            basins.append(basin)
    return sorted(set(basins))


def load_basin_data(basin_id: str, root_dir: str) -> tuple[pd.DataFrame, list[str], np.ndarray, float]:
    forcing_pattern = os.path.join(root_dir, "**", f"*{basin_id}*_daily_meteorological_inputs.txt")
    forcing_files = glob.glob(forcing_pattern, recursive=True)
    if not forcing_files:
        fallback = os.path.join(root_dir, "**", f"*{basin_id}*_meteorological_inputs_*.txt")
        forcing_files = glob.glob(fallback, recursive=True)
    if not forcing_files:
        raise FileNotFoundError(f"Forcing file not found for basin {basin_id}")
    forcing_file = forcing_files[0]

    with open(forcing_file, "r", encoding="utf-8", errors="ignore") as f:
        header = [f.readline().strip() for _ in range(4)]
    area = float(header[2])
    area_km2 = area / 1e6 if area > 10000 else area

    forcing = pd.read_csv(forcing_file, sep=r"\s+", header=None, skiprows=4)
    col_map = {
        0: "Y",
        1: "M",
        2: "D",
        5: "Prcp",
        6: "Srad",
        7: "Swe",
        8: "Tmax",
        9: "Tmin",
        10: "Vp",
    }
    forcing.rename(columns={k: v for k, v in col_map.items() if k in forcing.columns}, inplace=True)
    forcing["Date"] = pd.to_datetime(
        forcing[["Y", "M", "D"]].rename(columns={"Y": "year", "M": "month", "D": "day"})
    )
    forcing.set_index("Date", inplace=True)

    flow_pattern = os.path.join(root_dir, "**", f"*{basin_id}*_observed_discharge.txt")
    flow_files = glob.glob(flow_pattern, recursive=True)
    if not flow_files:
        raise FileNotFoundError(f"Streamflow file not found for basin {basin_id}")
    flow = pd.read_csv(
        flow_files[0],
        sep=r"\s+",
        header=None,
        names=["ID", "Y", "M", "D", "Q_cfs", "QC"],
    )
    flow["Date"] = pd.to_datetime(
        flow[["Y", "M", "D"]].rename(columns={"Y": "year", "M": "month", "D": "day"})
    )
    flow.set_index("Date", inplace=True)

    feature_candidates = ["Prcp", "Srad", "Tmax", "Tmin", "Vp", "Swe"]
    features = [name for name in feature_candidates if name in forcing.columns]
    df = pd.concat([forcing[features], flow[["Q_cfs"]]], axis=1, join="inner")
    df = df[df["Q_cfs"] >= 0].copy()
    df["Q"] = df["Q_cfs"] * (0.0283168 * 86400 * 1000) / (area_km2 * 1e6)

    static_vec = _load_static_attributes(basin_id, root_dir)
    return df[features + ["Q"]], features, static_vec, area_km2


def _load_static_attributes(basin_id: str, root_dir: str) -> np.ndarray:
    """Load the explicit five-field static-conditioning vector."""
    try:
        static_attributes = load_static_attribute_dict(basin_id, root_dir)
        return static_vector_from_dict(static_attributes)
    except Exception:
        return np.zeros(STATIC_ATTRIBUTE_DIM, dtype=np.float32)


def load_static_attribute_dict(basin_id: str, root_dir: str) -> dict[str, float]:
    """Load the five static-conditioning inputs by explicit field name."""
    zero_attributes = {name: 0.0 for name in STATIC_INPUT_COLUMNS}
    pattern = os.path.join(root_dir, "**", "catchment_attributes.txt")
    static_files = glob.glob(pattern, recursive=True)
    if not static_files:
        return zero_attributes

    # Match the dtype inference and basin lookup used for checkpoint generation.
    attrs = pd.read_csv(static_files[0], sep=r"\s+")
    required = {"BASIN_ID", *STATIC_INPUT_COLUMNS}
    missing = required.difference(attrs.columns)
    if missing:
        return zero_attributes

    basin_str = str(basin_id).zfill(8)
    id_col = None
    for column in attrs.columns:
        if basin_str in attrs[column].astype(str).values:
            id_col = column
            break
    if id_col is None:
        return zero_attributes

    attrs.set_index(id_col, inplace=True)
    row = attrs.loc[int(basin_id)] if int(basin_id) in attrs.index else attrs.loc[basin_str]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    values = pd.to_numeric(row.loc[list(STATIC_INPUT_COLUMNS)], errors="coerce").to_numpy(dtype=np.float32)
    if values.shape != (STATIC_ATTRIBUTE_DIM,):
        return zero_attributes
    if not np.isfinite(values).all():
        values = np.nan_to_num(values).astype(np.float32)
    return {
        name: float(value)
        for name, value in zip(STATIC_INPUT_COLUMNS, values)
    }


def static_vector_from_dict(static_attributes: dict[str, float]) -> np.ndarray:
    """Build ``[HUC02, area, elevation, slope, forest]`` by name."""
    missing = set(STATIC_INPUT_COLUMNS).difference(static_attributes)
    if missing:
        raise ValueError(f"Missing named static inputs: {sorted(missing)}")
    values = np.asarray(
        [static_attributes[name] for name in STATIC_INPUT_COLUMNS],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("Static-input dictionary contains non-finite values")
    return values


def load_euh_static_attribute_dict(basin_id: str, root_dir: str) -> dict[str, float]:
    """Return the same named five-field dictionary for EUH diagnostics."""
    return load_static_attribute_dict(basin_id, root_dir)


def euh_static_vector_from_dict(static_attributes: dict[str, float]) -> np.ndarray:
    """Build the EUH vector using the model's explicit five-field order."""
    return static_vector_from_dict(static_attributes)


def load_euh_static_attributes(basin_id: str, root_dir: str) -> np.ndarray:
    """Load the explicit HUC02-plus-four-attribute EUH vector."""
    static_attributes = load_euh_static_attribute_dict(basin_id, root_dir)
    return euh_static_vector_from_dict(static_attributes)


class CatchmentDataset(Dataset):
    def __init__(
        self,
        basin_id: str,
        data_dir: str,
        seq_length: int = 365,
        split: str = "train",
        split_dates: SplitDates | None = None,
        scaler: np.ndarray | None = None,
        transform_lambda: float = 1.0,
    ) -> None:
        self.seq_length = seq_length
        self.transform_lambda = transform_lambda
        self.split_dates = split_dates or SplitDates()

        df, features, static_vec, self.area_km2 = load_basin_data(basin_id, data_dir)
        self.features = features
        self.input_dim = len(features)
        self.prcp_idx = features.index("Prcp") if "Prcp" in features else 0

        dates = df.index
        train_start = pd.to_datetime(self.split_dates.train_start)
        train_end = pd.to_datetime(self.split_dates.train_end)
        val_start = pd.to_datetime(self.split_dates.val_start)
        val_end = pd.to_datetime(self.split_dates.val_end)
        test_start = pd.to_datetime(self.split_dates.test_start)
        test_end = pd.to_datetime(self.split_dates.test_end)
        if split == "train":
            mask = (dates >= train_start) & (dates <= train_end)
        elif split == "val":
            mask = (dates >= val_start) & (dates <= val_end)
        elif split == "test":
            mask = (dates >= test_start) & (dates <= test_end)
        else:
            raise ValueError(f"Unknown split: {split}")

        split_df = df[mask]
        x_raw = split_df[features].values
        y_raw = split_df["Q"].values
        if len(x_raw) <= seq_length:
            raise ValueError(f"Not enough samples for basin {basin_id} split={split}")

        if split == "train" and scaler is None:
            self.scaler_max = np.max(np.abs(x_raw), axis=0) + 1e-6
        elif scaler is not None:
            self.scaler_max = scaler
        else:
            self.scaler_max = np.max(np.abs(x_raw), axis=0) + 1e-6

        self.x_data = torch.as_tensor(x_raw / self.scaler_max, dtype=torch.float32)
        self.y_data = torch.as_tensor(y_raw, dtype=torch.float32)
        self.static_data = torch.as_tensor(static_vec, dtype=torch.float32)
        self.static_dim = self.static_data.shape[0]

    def __len__(self) -> int:
        return len(self.x_data) - self.seq_length

    def __getitem__(self, idx: int):
        x = self.x_data[idx : idx + self.seq_length].clone()
        y = self.y_data[idx + self.seq_length - 1]
        if self.transform_lambda != 1.0:
            x[:, self.prcp_idx] *= self.transform_lambda
        return x, self.static_data, torch.log(y + 1e-2)


def create_dataloaders(
    basin_id: str,
    data_dir: str,
    batch_size: int = 64,
    seq_length: int = 365,
    split_dates: SplitDates | None = None,
):
    train_ds = CatchmentDataset(
        basin_id,
        data_dir,
        seq_length=seq_length,
        split="train",
        split_dates=split_dates,
    )
    val_ds = CatchmentDataset(
        basin_id,
        data_dir,
        seq_length=seq_length,
        split="val",
        split_dates=split_dates,
        scaler=train_ds.scaler_max,
    )
    test_ds = CatchmentDataset(
        basin_id,
        data_dir,
        seq_length=seq_length,
        split="test",
        split_dates=split_dates,
        scaler=train_ds.scaler_max,
    )
    scaled_ds = CatchmentDataset(
        basin_id,
        data_dir,
        seq_length=seq_length,
        split="test",
        split_dates=split_dates,
        scaler=train_ds.scaler_max,
        transform_lambda=2.0,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "val": val_loader,
        "test": test_loader,
        "test_scaled": DataLoader(scaled_ds, batch_size=batch_size, shuffle=False),
    }
