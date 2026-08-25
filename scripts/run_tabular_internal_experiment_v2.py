#!/usr/bin/env python
"""Run calibrated train-only tabular-v2 OOF experiments and review gates."""

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
from src.prediction.tabular.experiment_v2 import (
    load_tabular_experiment_v2_config,
    review_internal_experiment_v2,
    run_internal_experiment_v2,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    data_root = PROJECT_ROOT / "data" / "tabular" / "train_v2"
    output_root = PROJECT_ROOT / "models" / "tabular" / "research_v2"
    parser = argparse.ArgumentParser(description="Run train-only tabular-v2 OOF")
    parser.add_argument("--features", type=Path, default=data_root / "features.parquet")
    parser.add_argument("--labels", type=Path, default=data_root / "labels.parquet")
    parser.add_argument("--dataset-manifest", type=Path, default=data_root / "manifest.json")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--oof", type=Path, default=output_root / "oof_predictions.parquet")
    parser.add_argument("--report", type=Path, default=output_root / "report.json")
    parser.add_argument("--review", type=Path, default=output_root / "selection_review.json")
    args = parser.parse_args()
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if _sha256(args.features) != manifest["artifacts"]["features"]["sha256"]:
        raise ValueError("train_v2 feature artifact hash mismatch")
    if _sha256(args.labels) != manifest["artifacts"]["labels"]["sha256"]:
        raise ValueError("train_v2 label artifact hash mismatch")
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split, development=development)
    catalog = load_tabular_feature_catalog(
        *(tuple() if args.catalog is None else (args.catalog,)),
        tabular=tabular,
        split=split,
        development=development,
    )
    config = (
        load_tabular_experiment_v2_config()
        if args.config is None
        else load_tabular_experiment_v2_config(args.config)
    )
    oof, report = run_internal_experiment_v2(
        features=pd.read_parquet(args.features),
        labels=pd.read_parquet(args.labels),
        dataset_manifest=manifest,
        development=development,
        catalog=catalog,
        config=config,
    )
    _atomic_parquet(oof, args.oof)
    report["artifacts"] = {
        "oof_predictions": {
            "path": str(args.oof.resolve()),
            "sha256": _sha256(args.oof),
            "rows": len(oof),
        }
    }
    _atomic_json(report, args.report)
    review = review_internal_experiment_v2(report)
    review["report_sha256"] = _sha256(args.report)
    _atomic_json(review, args.review)
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
