#!/usr/bin/env python
"""Audit raw point-in-time company/industry and financial archives."""

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
from src.prediction.tabular.point_in_time_audit import audit_point_in_time_sources
from src.prediction.tabular.point_in_time_sources import (
    load_point_in_time_source_contract,
    save_json_atomic,
)


def main() -> None:
    root = PROJECT_ROOT / "data" / "tabular" / "point_in_time_sources"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data" / "tabular" / "train_v2" / "features.parquet",
    )
    parser.add_argument("--output", type=Path, default=root / "audit.report.json")
    args = parser.parse_args()
    contract = load_point_in_time_source_contract()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split)
    report = audit_point_in_time_sources(
        root=args.root,
        company_industry_state_path=args.root / "company_industry.state.json",
        fundamentals_state_path=args.root / "fundamentals.state.json",
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
