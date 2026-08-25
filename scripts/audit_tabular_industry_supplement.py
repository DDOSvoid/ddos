#!/usr/bin/env python
"""Audit the train-only legacy Shenwan industry supplement."""

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
    audit_industry_supplement,
    load_industry_supplement_contract,
)
from src.prediction.tabular.point_in_time_sources import save_json_atomic


def main() -> None:
    default_root = PROJECT_ROOT / "data" / "tabular" / "industry_supplement_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--existing-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "tabular" / "point_in_time_sources",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data" / "tabular" / "train_v2" / "features.parquet",
    )
    parser.add_argument("--output", type=Path, default=default_root / "audit.report.json")
    args = parser.parse_args()
    contract = load_industry_supplement_contract()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split)
    report = audit_industry_supplement(
        root=args.root,
        state_path=args.root / "state.json",
        existing_root=args.existing_root,
        samples_path=args.samples,
        contract=contract,
        split=split,
        development=development,
        tabular=tabular,
    )
    save_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
