"""Build train-only time-series artifacts from audited 15-minute daily aggregates."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.timeseries.artifacts import (
    _file_sha256,
    _write_csv_atomic,
    _write_json_atomic,
)
from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.intraday_contract import (
    IntradayAggregateTimeseriesContract,
    load_intraday_aggregate_timeseries_contract,
)
from src.prediction.timeseries.intraday_features import (
    build_intraday_rolling_panel,
    intraday_rolling_feature_names,
)


def _load_and_validate_source_snapshot(
    contract: IntradayAggregateTimeseriesContract,
) -> tuple[pd.DataFrame, dict, dict]:
    audit = json.loads(contract.source_audit_path.read_text(encoding="utf-8"))
    state = json.loads(contract.aggregate_state_path.read_text(encoding="utf-8"))
    if audit.get("contract") != "tabular-intraday-audit-v1":
        raise ValueError("unexpected intraday source audit contract")
    if audit.get("passed") is not True or audit.get("errors"):
        raise ValueError("intraday source audit did not pass")
    if audit.get("official_test_queried") is not False:
        raise PermissionError("intraday source audit does not prove test isolation")
    if audit.get("train_range") != {
        "start": contract.training_start.isoformat(),
        "end": contract.training_end.isoformat(),
    }:
        raise ValueError("intraday source audit range differs from train partition")
    if audit.get("feature_schema_sha256") != contract.source_feature_schema_sha256:
        raise ValueError("intraday source audit schema differs from contract")
    if state.get("official_test_queried") is not False:
        raise PermissionError("intraday aggregate state does not prove test isolation")
    selection = state.get("selection", {})
    if selection.get("dataset_role") != "train":
        raise PermissionError("intraday aggregate state is not train-only")
    if selection.get("aggregation_version") != contract.source_aggregation_version:
        raise ValueError("intraday aggregate state version differs")
    if selection.get("feature_schema_sha256") != contract.source_feature_schema_sha256:
        raise ValueError("intraday aggregate state schema differs")

    completed = state.get("completed", {})
    all_state_codes = {str(key).split(":", 1)[0] for key in completed}
    if len(all_state_codes) != int(audit["stats"]["stocks"]):
        raise ValueError("intraday state stock universe differs from audit")
    frames = []
    verified_files = 0
    aggregated_codes = set()
    aggregate_root = contract.intraday_source_directory
    for key, item in sorted(completed.items()):
        status = item.get("status")
        if status == "empty_source":
            continue
        if status != "aggregated":
            raise ValueError(f"unexpected intraday aggregate status for {key}: {status}")
        source_path = aggregate_root / item["path"]
        if _file_sha256(source_path) != item["sha256"]:
            raise ValueError(f"intraday aggregate hash mismatch: {source_path}")
        frames.append(pd.read_parquet(source_path))
        verified_files += 1
        aggregated_codes.add(str(key).split(":", 1)[0])
    if not frames:
        raise ValueError("no audited intraday aggregate files")
    source = pd.concat(frames, ignore_index=True)
    if len(source) != int(audit["stats"]["aggregate_rows"]):
        raise ValueError("intraday aggregate row count differs from audit")
    if verified_files != int(audit["stats"]["downloaded_chunks"]):
        raise ValueError("verified intraday file count differs from audit")
    if source["stock_code"].astype(str).nunique() != len(aggregated_codes):
        raise ValueError("intraday aggregate stock count differs from source state")
    source.attrs["verified_files"] = verified_files
    return source, audit, state


def build_intraday_train_artifacts(
    output_directory: Path,
    *,
    contract: IntradayAggregateTimeseriesContract,
    parent_artifact_directory: Path,
    force: bool = False,
) -> dict[str, object]:
    output = output_directory.resolve()
    if not output.is_relative_to(contract.data_directory):
        raise ValueError("intraday time-series output must remain under data/timeseries")
    parent_root = parent_artifact_directory.resolve()
    if not parent_root.is_relative_to(contract.data_directory):
        raise ValueError("parent daily artifacts must remain under data/timeseries")

    feature_path = output / "features" / "train_intraday_aggregates_v1_features.csv"
    label_path = output / "labels" / "train_intraday_aggregates_v1_labels.csv"
    manifest_path = output / "manifests" / "train_intraday_aggregates_v1_manifest.json"
    existing = [path for path in (feature_path, label_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"intraday train artifact already exists: {existing[0]}")

    parent_contract = load_timeseries_model_contract()
    if parent_contract.source_sha256 != contract.parent_contract_sha256:
        raise ValueError("parent daily contract hash changed")
    parent_manifest_path = (
        parent_root / "manifests" / "train_daily_v1_manifest.json"
    )
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest["contract_sha256"] != contract.parent_contract_sha256:
        raise ValueError("parent daily artifact and intraday contract differ")
    if parent_manifest.get("official_test_read") is not False:
        raise PermissionError("parent daily artifact does not prove test isolation")
    parent_feature_path = parent_root / parent_manifest["features_file"]
    parent_label_path = parent_root / parent_manifest["labels_file"]
    if _file_sha256(parent_feature_path) != parent_manifest["features_sha256"]:
        raise ValueError("parent daily feature artifact hash changed")
    if _file_sha256(parent_label_path) != parent_manifest["labels_sha256"]:
        raise ValueError("parent daily label artifact hash changed")
    parent_features = pd.read_csv(
        parent_feature_path,
        parse_dates=["published_date", "prediction_as_of", "last_bar_available_at"],
    )
    parent_features["published_date"] = parent_features["published_date"].dt.date
    parent_features["last_bar_trade_date"] = pd.to_datetime(
        parent_features["last_bar_trade_date"], errors="raise"
    ).dt.date
    parent_labels = pd.read_csv(
        parent_label_path,
        parse_dates=["outcome_available_at"],
    )

    source, audit, state = _load_and_validate_source_snapshot(contract)
    panels = {
        str(stock_code): build_intraday_rolling_panel(group, contract=contract)
        for stock_code, group in source.groupby("stock_code", sort=True)
    }
    feature_names = intraday_rolling_feature_names(contract)
    exclusions: defaultdict[str, int] = defaultdict(int)
    records = []
    for row in parent_features.itertuples(index=False):
        stock_code = str(row.stock_code)
        panel = panels.get(stock_code)
        if panel is None:
            exclusions["missing_intraday_stock"] += 1
            continue
        panel_dates = panel["observation_date"].tolist()
        endpoint_index = bisect.bisect_left(panel_dates, row.published_date) - 1
        if endpoint_index < 0:
            exclusions["no_prepublication_intraday_day"] += 1
            continue
        endpoint = panel.iloc[endpoint_index]
        if int(endpoint["complete_intraday_days"]) < contract.minimum_complete_intraday_days:
            exclusions["insufficient_complete_intraday_days"] += 1
            continue
        available_at = pd.Timestamp(endpoint["intraday_features_available_at_utc"])
        prediction_as_of = pd.Timestamp(row.prediction_as_of)
        if available_at > prediction_as_of:
            raise ValueError("future intraday aggregate reached a feature row")
        if endpoint["observation_date"] >= row.published_date:
            raise ValueError("publication-day intraday aggregate reached a feature row")
        feature_values = {
            name: endpoint[name]
            for name in feature_names
            if name != "intraday_staleness_calendar_days"
        }
        feature_values["intraday_staleness_calendar_days"] = (
            row.published_date - endpoint["observation_date"]
        ).days
        records.append(
            {
                "sample_id": row.sample_id,
                "stock_code": stock_code,
                "published_date": row.published_date,
                "prediction_as_of": row.prediction_as_of,
                "last_bar_trade_date": endpoint["observation_date"],
                "last_bar_available_at": available_at,
                "dataset_role": "train",
                **feature_values,
            }
        )
    feature_frame = pd.DataFrame(records).sort_values(
        ["published_date", "stock_code", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    included_ids = set(feature_frame["sample_id"])
    label_frame = parent_labels.loc[
        parent_labels["sample_id"].isin(included_ids)
    ].copy()
    label_frame = label_frame.sort_values(
        ["horizon_sessions", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(feature_frame) != len(label_frame):
        raise ValueError("intraday feature and parent label counts differ")
    if set(feature_frame["sample_id"]) != set(label_frame["sample_id"]):
        raise ValueError("intraday feature and parent label identities differ")

    _write_csv_atomic(feature_frame, feature_path)
    _write_csv_atomic(label_frame, label_path)
    manifest = {
        "artifact": "causal-timeseries-intraday-aggregates-v1-train",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "parent_contract": contract.parent_contract,
        "parent_contract_sha256": contract.parent_contract_sha256,
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "publication_day_intraday_used": False,
        "input_parent_samples": len(parent_features),
        "included_samples": len(feature_frame),
        "excluded_samples": len(parent_features) - len(feature_frame),
        "exclusions": dict(sorted(exclusions.items())),
        "feature_columns": list(feature_names),
        "features_file": feature_path.relative_to(output).as_posix(),
        "features_sha256": _file_sha256(feature_path),
        "labels_file": label_path.relative_to(output).as_posix(),
        "labels_sha256": _file_sha256(label_path),
        "parent_manifest_sha256": _file_sha256(parent_manifest_path),
        "parent_features_sha256": parent_manifest["features_sha256"],
        "parent_labels_sha256": parent_manifest["labels_sha256"],
        "source_audit_file": str(contract.source_audit_path),
        "source_audit_sha256": _file_sha256(contract.source_audit_path),
        "source_aggregate_state_file": str(contract.aggregate_state_path),
        "source_aggregate_state_sha256": _file_sha256(contract.aggregate_state_path),
        "source_verified_aggregate_files": source.attrs["verified_files"],
        "source_aggregate_rows": len(source),
        "source_stocks": int(source["stock_code"].astype(str).nunique()),
        "source_audit_contract": audit["contract"],
        "source_state_contract": state["contract"],
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "timeseries" / "intraday_aggregates_v1"
        ),
    )
    parser.add_argument(
        "--parent-artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_intraday_train_artifacts(
        args.output_dir,
        contract=load_intraday_aggregate_timeseries_contract(),
        parent_artifact_directory=args.parent_artifact_dir,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
