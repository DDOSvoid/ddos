#!/usr/bin/env python
"""Backfill train-only Tushare daily valuation and share metrics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.daily_basic import (
    TushareDailyBasicClient,
    download_daily_basic_train,
    load_intraday_train_universe,
)


def main() -> None:
    root = PROJECT_ROOT / "data" / "tabular" / "daily_basic"
    intraday_state = (
        PROJECT_ROOT
        / "data"
        / "tabular"
        / "intraday_15m"
        / "state.amazingdata.json"
    )
    parser = argparse.ArgumentParser(description="Backfill train-only daily_basic data")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--universe-state", type=Path, default=intraday_state)
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument("--state", type=Path, default=root / "state.json")
    parser.add_argument("--minimum-interval-seconds", type=float, default=0.15)
    args = parser.parse_args()
    split = load_prediction_split_contract()
    state = download_daily_basic_train(
        codes=load_intraday_train_universe(args.universe_state),
        start=args.start_date,
        end=args.end_date,
        output_root=args.output_root,
        state_path=args.state,
        client=TushareDailyBasicClient(
            minimum_interval_seconds=args.minimum_interval_seconds
        ),
        split=split,
        tabular=load_tabular_model_contract(split=split),
    )
    print(
        json.dumps(
            {
                "completed": len(state.get("completed", {})),
                "failures": len(state.get("failures", {})),
                "official_test_queried": state["official_test_queried"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
