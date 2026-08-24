#!/usr/bin/env python
"""Run train-only CatBoost risk ablation without intraday features."""

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
from src.prediction.tabular.experiment import (
    load_tabular_experiment_config,
    run_intraday_risk_ablation,
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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    data_root = PROJECT_ROOT / "data" / "tabular" / "train_v1"
    model_root = PROJECT_ROOT / "models" / "tabular" / "research_v1"
    parser = argparse.ArgumentParser(description="Ablate train-only intraday risk features")
    parser.add_argument("--features", type=Path, default=data_root / "features.parquet")
    parser.add_argument("--labels", type=Path, default=data_root / "labels.parquet")
    parser.add_argument("--dataset-manifest", type=Path, default=data_root / "manifest.json")
    parser.add_argument("--full-report", type=Path, default=model_root / "report.json")
    parser.add_argument(
        "--oof", type=Path, default=model_root / "intraday_risk_ablation_oof.parquet"
    )
    parser.add_argument(
        "--report", type=Path, default=model_root / "intraday_risk_ablation.json"
    )
    args = parser.parse_args()
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if _sha256(args.features) != manifest["artifacts"]["features"]["sha256"]:
        raise ValueError("feature artifact hash mismatch")
    if _sha256(args.labels) != manifest["artifacts"]["labels"]["sha256"]:
        raise ValueError("label artifact hash mismatch")
    split = load_prediction_split_contract()
    oof, report = run_intraday_risk_ablation(
        features=pd.read_parquet(args.features),
        labels=pd.read_parquet(args.labels),
        dataset_manifest=manifest,
        development=load_causal_development_contract(split=split),
        config=load_tabular_experiment_config(),
        full_report=json.loads(args.full_report.read_text(encoding="utf-8")),
    )
    _atomic_parquet(oof, args.oof)
    report["artifacts"] = {
        "oof": {
            "path": str(args.oof.resolve()),
            "sha256": _sha256(args.oof),
            "rows": len(oof),
        }
    }
    _atomic_json(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
