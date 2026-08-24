#!/usr/bin/env python
"""Audit all train-only raw 15-minute chunks and their daily aggregates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.audit import audit_intraday_dataset
from src.prediction.tabular.contract import load_tabular_model_contract


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    default_root = PROJECT_ROOT / "data" / "tabular" / "intraday_15m"
    parser = argparse.ArgumentParser(
        description="Audit the complete 2023-2024 train-only intraday dataset"
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--raw-state", type=Path, default=default_root / "state.amazingdata.json"
    )
    parser.add_argument(
        "--aggregate-state", type=Path, default=default_root / "state.aggregates.json"
    )
    parser.add_argument(
        "--report", type=Path, default=default_root / "audit.report.json"
    )
    args = parser.parse_args()

    split = load_prediction_split_contract()
    tabular = load_tabular_model_contract(split=split)
    report = audit_intraday_dataset(
        raw_root=args.root,
        raw_state_path=args.raw_state,
        aggregate_root=args.root,
        aggregate_state_path=args.aggregate_state,
        tabular=tabular,
        split=split,
    )
    _write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
