#!/usr/bin/env python
"""Audit all train-only Tushare daily_basic partitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.daily_basic_audit import audit_daily_basic_dataset


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    root = PROJECT_ROOT / "data" / "tabular" / "daily_basic"
    snapshot_root = PROJECT_ROOT / "data" / "tabular" / "snapshots"
    parser = argparse.ArgumentParser(
        description="Audit the complete 2023-2024 train-only daily_basic dataset"
    )
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--state", type=Path, default=root / "state.json")
    parser.add_argument(
        "--snapshot", type=Path, default=snapshot_root / "ddos.train.sqlite"
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=snapshot_root / "ddos.train.manifest.json",
    )
    parser.add_argument("--report", type=Path, default=root / "audit.report.json")
    args = parser.parse_args()
    split = load_prediction_split_contract()
    report = audit_daily_basic_dataset(
        root=args.root,
        state_path=args.state,
        snapshot_path=args.snapshot,
        snapshot_manifest_path=args.snapshot_manifest,
        tabular=load_tabular_model_contract(split=split),
        split=split,
    )
    _write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
