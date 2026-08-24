#!/usr/bin/env python
"""Download train-only 15-minute bars for lagged daily tabular aggregates."""

# ruff: noqa: E402 -- vendor NumPy/Numba must be bootstrapped before project imports.

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# AmazingData bundles a NumPy/Numba combination that must be imported before
# the project's pandas stack. Keep that compatibility path inside this CLI.
BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BOOTSTRAP_ROOT / ".env")
amazingdata_package = os.environ.get("AMAZINGDATA_PATH")
if amazingdata_package:
    sys.path.insert(0, amazingdata_package)
sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config import PROJECT_ROOT
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.intraday import (
    AmazingDataIntradayClient,
    RateLimitedTushareIntradayClient,
    download_intraday_train,
    load_train_stock_codes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="断点续传训练期15分钟行情；只用于日级滞后聚合特征"
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "tabular" / "intraday_15m",
    )
    parser.add_argument(
        "--state",
        type=Path,
    )
    parser.add_argument("--limit-codes", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=("amazingdata", "tushare"),
        default="amazingdata",
    )
    parser.add_argument("--minimum-interval-seconds", type=float, default=3601.0)
    args = parser.parse_args()

    split = load_prediction_split_contract()
    tabular = load_tabular_model_contract(split=split)
    codes = load_train_stock_codes(
        args.database,
        start=args.start_date,
        end=args.end_date,
    )
    if not codes:
        raise ValueError("no accepted train-period core-event stock codes found")
    if args.source == "amazingdata":
        required = (
            "AMAZINGDATA_PATH",
            "AMAZINGDATA_USERNAME",
            "AMAZINGDATA_PASSWORD",
            "AMAZINGDATA_HOST",
            "AMAZINGDATA_PORT",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(f"missing AmazingData settings: {missing}")
        client = AmazingDataIntradayClient(
            package_path=Path(os.environ["AMAZINGDATA_PATH"]),
            username=os.environ["AMAZINGDATA_USERNAME"],
            password=os.environ["AMAZINGDATA_PASSWORD"],
            host=os.environ["AMAZINGDATA_HOST"],
            port=int(os.environ["AMAZINGDATA_PORT"]),
            calendar_end=args.end_date,
        )
    else:
        client = RateLimitedTushareIntradayClient(
            minimum_interval_seconds=args.minimum_interval_seconds
        )
    try:
        state_path = args.state or (
            args.output_root / f"state.{args.source}.json"
        )
        state = download_intraday_train(
            codes=codes,
            start=args.start_date,
            end=args.end_date,
            output_root=args.output_root,
            state_path=state_path,
            client=client,
            split=split,
            tabular=tabular,
            limit_codes=max(args.limit_codes, 0),
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    print(
        {
            "selection": state["selection"],
            "completed": len(state.get("completed", {})),
            "failures": len(state.get("failures", {})),
        }
    )


if __name__ == "__main__":
    main()
