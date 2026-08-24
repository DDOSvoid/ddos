"""Run the train-only daily-v1 logistic OOF diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.artifacts import (
    _file_sha256,
    _write_csv_atomic,
    _write_json_atomic,
)
from src.prediction.timeseries.baselines import run_logistic_baseline_oof
from src.prediction.timeseries.contract import (
    TimeseriesModelContract,
    load_timeseries_model_contract,
)
from src.prediction.timeseries.dataset import assemble_train_frame


def _require_under(path: Path, boundary: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(boundary.resolve()):
        raise ValueError(f"{label} must remain under {boundary}")
    return resolved


def run_train_baseline(
    artifact_directory: Path,
    *,
    contract: TimeseriesModelContract,
    force: bool = False,
) -> dict[str, object]:
    artifact_root = _require_under(
        artifact_directory,
        contract.data_directory,
        label="train artifact directory",
    )
    manifest_path = artifact_root / "manifests" / "train_daily_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["contract"] != contract.contract:
        raise ValueError("train artifact and model contracts do not match")
    if manifest["contract_sha256"] != contract.source_sha256:
        raise ValueError("time-series contract changed after train artifact build")
    for role in ("official_test", "quarantine", "forward_validation"):
        if manifest[f"{role}_read"]:
            raise PermissionError(f"sealed role unexpectedly read by train artifact: {role}")
    feature_path = artifact_root / manifest["features_file"]
    label_path = artifact_root / manifest["labels_file"]
    if _file_sha256(feature_path) != manifest["features_sha256"]:
        raise ValueError("train feature artifact hash changed")
    if _file_sha256(label_path) != manifest["labels_sha256"]:
        raise ValueError("train label artifact hash changed")

    features = pd.read_csv(
        feature_path,
        parse_dates=["published_date", "prediction_as_of", "last_bar_available_at"],
    )
    features["published_date"] = features["published_date"].dt.date
    features["last_bar_trade_date"] = pd.to_datetime(
        features["last_bar_trade_date"], errors="raise"
    ).dt.date
    labels = pd.read_csv(label_path, parse_dates=["outcome_available_at"])
    frame = assemble_train_frame(
        features,
        labels,
        feature_columns=manifest["feature_columns"],
        contract=contract,
    )
    development = load_causal_development_contract()
    result = run_logistic_baseline_oof(
        frame,
        feature_columns=manifest["feature_columns"],
        contract=contract,
        development=development,
    )
    oof = result.oof_predictions.merge(
        frame.loc[:, ["sample_id", "excess_return"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    oof_path = artifact_root / "oof" / "train_daily_v1_logistic_oof.csv"
    report_path = artifact_root / "reports" / "train_daily_v1_logistic_oof.json"
    existing = [path for path in (oof_path, report_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"baseline output already exists: {existing[0]}")
    _write_csv_atomic(oof, oof_path)
    report = {
        "experiment": "causal-timeseries-daily-v1-logistic-baseline",
        "status": "train_only_diagnostic_not_release_eligible",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "train_manifest_file": manifest_path.relative_to(artifact_root).as_posix(),
        "train_manifest_sha256": _file_sha256(manifest_path),
        "train_features_sha256": manifest["features_sha256"],
        "train_labels_sha256": manifest["labels_sha256"],
        "oof_file": oof_path.relative_to(artifact_root).as_posix(),
        "oof_sha256": _file_sha256(oof_path),
        "limitations": [
            "bullish threshold remains the preregistered diagnostic 0.5",
            "transaction-cost confidence interval is not yet evaluated",
            "incremental comparison awaits table-model OOF predictions",
            "small TCN remains blocked until a causal baseline shows stable increment",
        ],
        **result.report,
    }
    _write_json_atomic(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run_train_baseline(
        args.artifact_dir,
        contract=load_timeseries_model_contract(),
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
