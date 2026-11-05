from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    data_dir: str
    output_dir: str = "outputs_hen"
    checkpoint_dir: str = "checkpoints"
    results_file: str = "summary_list_run.csv"
    basin_list: str | None = None

    train_start_date: str = "1980-01-01"
    train_end_date: str = "1999-12-31"
    val_start_date: str = "2000-01-01"
    val_end_date: str = "2003-12-31"
    test_start_date: str = "2004-01-01"
    test_end_date: str = "2008-12-31"

    seq_length: int = 365
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    hidden_dim: int = 256
    n_scales: int = 4
    kernel_size: int = 15
    n_bases: int = 3
    feature_dim: int = 16
    projected_dim: int = 32

    nse_weight: float = 0.2
    consistency_weight: float = 2.0
    consistency_lambda_min: float = 2.0
    consistency_lambda_max: float = 5.0

    train_lstm_baseline: bool = False
    lstm_baseline_epochs: int = 60
    skip_existing: bool = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HydEquivNet on catchment basins.")
    parser.add_argument("--data-dir", required=True, help="Root catchment dataset directory.")
    parser.add_argument("--output-dir", default="outputs_hen", help="Output directory.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint subdirectory.")
    parser.add_argument("--basin-list", default=None, help="Text file with one basin ID per line.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-start-date", default="1980-01-01")
    parser.add_argument("--train-end-date", default="1999-12-31")
    parser.add_argument("--val-start-date", default="2000-01-01")
    parser.add_argument("--val-end-date", default="2003-12-31")
    parser.add_argument("--test-start-date", default="2004-01-01")
    parser.add_argument("--test-end-date", default="2008-12-31")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--train-lstm-baseline", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        basin_list=args.basin_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        skip_existing=not args.no_skip_existing,
        train_lstm_baseline=args.train_lstm_baseline,
    )
