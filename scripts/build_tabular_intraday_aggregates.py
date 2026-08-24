#!/usr/bin/env python
"""Build causal daily features from downloaded train-period 15-minute bars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.aggregates import build_intraday_aggregate_dataset
from src.prediction.tabular.contract import load_tabular_model_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将完整训练期15分钟交易日聚合为因果日级表格特征"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "tabular" / "intraday_15m",
    )
    parser.add_argument(
        "--raw-state",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "tabular"
            / "intraday_15m"
            / "state.amazingdata.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "tabular" / "intraday_15m",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "tabular"
            / "intraday_15m"
            / "state.aggregates.json"
        ),
    )
    args = parser.parse_args()

    split = load_prediction_split_contract()
    tabular = load_tabular_model_contract(split=split)
    state = build_intraday_aggregate_dataset(
        raw_root=args.raw_root,
        raw_state_path=args.raw_state,
        output_root=args.output_root,
        state_path=args.state,
        tabular=tabular,
        split=split,
    )
    print(
        {
            "completed": len(state.get("completed", {})),
            "failures": len(state.get("failures", {})),
            "official_test_queried": state["official_test_queried"],
        }
    )


if __name__ == "__main__":
    main()
