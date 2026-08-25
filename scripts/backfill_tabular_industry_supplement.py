#!/usr/bin/env python
"""Backfill the train-only legacy Shenwan industry supplement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.industry_supplement import (
    download_industry_supplement,
    load_industry_supplement_contract,
)
from src.prediction.tabular.point_in_time_sources import (
    TusharePointInTimeClient,
    load_train_universe,
)


def main() -> None:
    default_root = PROJECT_ROOT / "data" / "tabular" / "industry_supplement_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=default_root)
    parser.add_argument("--minimum-interval-seconds", type=float, default=0.32)
    args = parser.parse_args()
    contract = load_industry_supplement_contract()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split)
    state = download_industry_supplement(
        codes=load_train_universe(contract.universe_path),
        output_root=args.output_root,
        state_path=args.output_root / "state.json",
        client=TusharePointInTimeClient(
            minimum_interval_seconds=args.minimum_interval_seconds
        ),
        contract=contract,
        split=split,
        development=development,
        tabular=tabular,
    )
    print(
        json.dumps(
            {
                "completed": len(state.get("completed", {})),
                "failures": len(state.get("failures", {})),
                "official_test_queried": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
