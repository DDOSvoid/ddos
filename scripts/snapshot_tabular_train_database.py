#!/usr/bin/env python
"""Create a physically train-only SQLite snapshot for tabular development."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.snapshot import build_train_snapshot


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    default_root = PROJECT_ROOT / "data" / "tabular" / "snapshots"
    parser = argparse.ArgumentParser(description="Build train-only tabular SQLite snapshot")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    parser.add_argument(
        "--output", type=Path, default=default_root / "ddos.train.sqlite"
    )
    parser.add_argument(
        "--manifest", type=Path, default=default_root / "ddos.train.manifest.json"
    )
    args = parser.parse_args()
    report = build_train_snapshot(
        source_path=args.source,
        destination_path=args.output,
        split=load_prediction_split_contract(),
    )
    _write_json_atomic(args.manifest, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
