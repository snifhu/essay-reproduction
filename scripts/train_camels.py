from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hydequivnet.config import build_arg_parser, config_from_args
from hydequivnet.data import get_all_basins
from hydequivnet.utils import get_device, read_basin_list, set_seed
from hydequivnet.workflows import run_single_basin, save_results


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    if cfg.basin_list:
        basins = read_basin_list(cfg.basin_list)
    else:
        basins = get_all_basins(cfg.data_dir)

    print(f"Basins: {len(basins)}")
    results = []
    for index, basin_id in enumerate(basins, start=1):
        print(f"[{index}/{len(basins)}] basin={basin_id}")
        try:
            result = run_single_basin(basin_id, cfg, device)
            results.append(result)
            save_results(results, Path(cfg.output_dir) / cfg.results_file)
        except Exception as exc:
            print(f"Failed basin {basin_id}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if results:
        save_results(results, Path(cfg.output_dir) / cfg.results_file)
        print(f"Saved: {Path(cfg.output_dir) / cfg.results_file}")


if __name__ == "__main__":
    main()

