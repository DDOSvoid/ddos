#!/usr/bin/env python
"""Backfill raw train-only company/industry and PIT financial sources."""

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
from src.prediction.tabular.point_in_time_sources import (
    TusharePointInTimeClient,
    download_company_industry_sources,
    download_financial_sources,
    load_point_in_time_source_contract,
    load_train_universe,
)


def main() -> None:
    root = PROJECT_ROOT / "data" / "tabular" / "point_in_time_sources"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("company_industry", "fundamentals", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--universe-state",
        type=Path,
        default=(PROJECT_ROOT / "data" / "backfills" / "event_ranking_v1" / "train_universe.json"),
    )
    parser.add_argument("--minimum-interval-seconds", type=float, default=0.32)
    args = parser.parse_args()
    contract = load_point_in_time_source_contract()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split)
    codes = load_train_universe(args.universe_state)
    client = TusharePointInTimeClient(minimum_interval_seconds=args.minimum_interval_seconds)
    result = {}
    if args.group in ("company_industry", "all"):
        state = download_company_industry_sources(
            codes=codes,
            output_root=args.output_root,
            state_path=args.output_root / "company_industry.state.json",
            client=client,
            contract=contract,
            split=split,
            development=development,
            tabular=tabular,
        )
        result["company_industry"] = {
            "completed": len(state.get("completed", {})),
            "failures": len(state.get("failures", {})),
        }
    if args.group in ("fundamentals", "all"):
        state = download_financial_sources(
            codes=codes,
            output_root=args.output_root,
            state_path=args.output_root / "fundamentals.state.json",
            client=client,
            contract=contract,
            split=split,
            development=development,
            tabular=tabular,
        )
        result["fundamentals"] = {
            "completed": len(state.get("completed", {})),
            "failures": len(state.get("failures", {})),
        }
    result["official_test_queried"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
