#!/usr/bin/env python
"""Build train_v2 from immutable train_v1 plus audited daily_basic features."""

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
from src.prediction.tabular.catalog import load_tabular_feature_catalog
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.dataset_v2 import build_tabular_train_dataset_v2


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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    tabular_root = PROJECT_ROOT / "data" / "tabular"
    base_root = tabular_root / "train_v1"
    daily_basic_root = tabular_root / "daily_basic"
    output_root = tabular_root / "train_v2"
    parser = argparse.ArgumentParser(description="Build causal train-only tabular v2")
    parser.add_argument("--base-features", type=Path, default=base_root / "features.parquet")
    parser.add_argument("--base-labels", type=Path, default=base_root / "labels.parquet")
    parser.add_argument("--base-manifest", type=Path, default=base_root / "manifest.json")
    parser.add_argument("--daily-basic-root", type=Path, default=daily_basic_root)
    parser.add_argument("--daily-basic-state", type=Path, default=daily_basic_root / "state.json")
    parser.add_argument(
        "--daily-basic-audit",
        type=Path,
        default=daily_basic_root / "audit.report.json",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--features", type=Path, default=output_root / "features.parquet")
    parser.add_argument("--labels", type=Path, default=output_root / "labels.parquet")
    parser.add_argument("--manifest", type=Path, default=output_root / "manifest.json")
    args = parser.parse_args()
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    if _sha256(args.base_features) != base_manifest["artifacts"]["features"]["sha256"]:
        raise ValueError("base train_v1 feature hash mismatch")
    if _sha256(args.base_labels) != base_manifest["artifacts"]["labels"]["sha256"]:
        raise ValueError("base train_v1 label hash mismatch")
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split, development=development)
    catalog = load_tabular_feature_catalog(
        *(tuple() if args.catalog is None else (args.catalog,)),
        tabular=tabular,
        split=split,
        development=development,
    )
    features, labels, manifest = build_tabular_train_dataset_v2(
        v1_features=pd.read_parquet(args.base_features),
        v1_labels=pd.read_parquet(args.base_labels),
        v1_manifest=base_manifest,
        daily_basic_root=args.daily_basic_root,
        daily_basic_state_path=args.daily_basic_state,
        daily_basic_audit_path=args.daily_basic_audit,
        catalog=catalog,
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
