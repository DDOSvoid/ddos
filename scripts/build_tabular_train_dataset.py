#!/usr/bin/env python
"""Build train-only point-in-time tabular features and separate labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.dataset import build_tabular_train_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)


def _write_json_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    tabular_root = PROJECT_ROOT / "data" / "tabular"
    snapshot_root = tabular_root / "snapshots"
    intraday_root = tabular_root / "intraday_15m"
    output_root = tabular_root / "train_v1"
    parser = argparse.ArgumentParser(description="Build causal train-only tabular dataset")
    parser.add_argument(
        "--snapshot", type=Path, default=snapshot_root / "ddos.train.sqlite"
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=snapshot_root / "ddos.train.manifest.json",
    )
    parser.add_argument("--intraday-root", type=Path, default=intraday_root)
    parser.add_argument(
        "--intraday-state", type=Path, default=intraday_root / "state.aggregates.json"
    )
    parser.add_argument(
        "--intraday-audit", type=Path, default=intraday_root / "audit.report.json"
    )
    parser.add_argument(
        "--features", type=Path, default=output_root / "features.parquet"
    )
    parser.add_argument("--labels", type=Path, default=output_root / "labels.parquet")
    parser.add_argument(
        "--manifest", type=Path, default=output_root / "manifest.json"
    )
    args = parser.parse_args()

    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split, development=development)
    features, labels, manifest = build_tabular_train_dataset(
        snapshot_path=args.snapshot,
        snapshot_manifest_path=args.snapshot_manifest,
        intraday_root=args.intraday_root,
        intraday_state_path=args.intraday_state,
        intraday_audit_path=args.intraday_audit,
        tabular=tabular,
        split=split,
        development=development,
    )
    _write_parquet_atomic(features, args.features)
    _write_parquet_atomic(labels, args.labels)
    manifest["artifacts"] = {
        "features": {
            "path": str(args.features.resolve()),
            "sha256": _sha256(args.features),
            "rows": len(features),
        },
        "labels": {
            "path": str(args.labels.resolve()),
            "sha256": _sha256(args.labels),
            "rows": len(labels),
        },
    }
    _write_json_atomic(manifest, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
