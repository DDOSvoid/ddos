"""Build tabular train_v2 by extending immutable train_v1 with audited inputs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.catalog import V2_DATASET_CONTRACT, TabularFeatureCatalog
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.daily_basic_audit import DAILY_BASIC_AUDIT_CONTRACT
from src.prediction.tabular.dataset import DATASET_CONTRACT as V1_DATASET_CONTRACT
from src.prediction.tabular.features import (
    require_train_only,
    validate_and_select_features,
)

DAILY_BASIC_FEATURE_COLUMNS = (
    "daily_basic_turnover_rate",
    "daily_basic_free_float_turnover_rate",
    "daily_basic_volume_ratio",
    "daily_basic_pe",
    "daily_basic_pe_ttm",
    "daily_basic_pb",
    "daily_basic_ps",
    "daily_basic_ps_ttm",
    "daily_basic_dividend_yield",
    "daily_basic_dividend_yield_ttm",
    "daily_basic_log_total_market_value",
    "daily_basic_log_circulating_market_value",
    "daily_basic_float_share_ratio",
    "daily_basic_free_share_ratio",
    "daily_basic_circulating_market_value_ratio",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def build_daily_basic_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create documented v2 valuation/liquidity inputs from audited source rows."""
    working = raw.copy()
    numeric = (
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    )
    for column in numeric:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["daily_basic_turnover_rate"] = working["turnover_rate"] / 100.0
    working["daily_basic_free_float_turnover_rate"] = (
        working["turnover_rate_f"] / 100.0
    )
    working["daily_basic_volume_ratio"] = working["volume_ratio"]
    for source, target in (
        ("pe", "daily_basic_pe"),
        ("pe_ttm", "daily_basic_pe_ttm"),
        ("pb", "daily_basic_pb"),
        ("ps", "daily_basic_ps"),
        ("ps_ttm", "daily_basic_ps_ttm"),
    ):
        working[target] = working[source]
    working["daily_basic_dividend_yield"] = working["dv_ratio"] / 100.0
    working["daily_basic_dividend_yield_ttm"] = working["dv_ttm"] / 100.0
    working["daily_basic_log_total_market_value"] = np.log1p(working["total_mv"])
    working["daily_basic_log_circulating_market_value"] = np.log1p(
        working["circ_mv"]
    )
    working["daily_basic_float_share_ratio"] = (
        working["float_share"] / working["total_share"]
    )
    working["daily_basic_free_share_ratio"] = (
        working["free_share"] / working["total_share"]
    )
    working["daily_basic_circulating_market_value_ratio"] = (
        working["circ_mv"] / working["total_mv"]
    )
    working["daily_basic_observation_date"] = pd.to_datetime(
        working["trade_date"], errors="raise"
    ).dt.date
    working["daily_basic_features_available_at_utc"] = pd.to_datetime(
        working["available_at_utc"], errors="raise", utc=True
    )
    return working.loc[
        :,
        [
            "ts_code",
            "daily_basic_observation_date",
            "daily_basic_features_available_at_utc",
            *DAILY_BASIC_FEATURE_COLUMNS,
        ],
    ].rename(columns={"ts_code": "stock_code"})


def merge_strictly_prior_daily_basic(
    samples: pd.DataFrame, daily_basic: pd.DataFrame
) -> pd.DataFrame:
    """Attach only daily_basic rows strictly before the publication date."""
    working = samples.copy()
    working["__row_order_v2"] = np.arange(len(working))
    working["__published_timestamp_v2"] = pd.to_datetime(
        working["published_date"], errors="raise"
    ).astype("datetime64[ns]")
    pieces = []
    for stock_code, group in working.groupby("stock_code", sort=False):
        left = group.sort_values("__published_timestamp_v2", kind="mergesort")
        right = daily_basic.loc[daily_basic["stock_code"] == stock_code].copy()
        if right.empty:
            merged = left.copy()
            merged["daily_basic_observation_date"] = pd.NaT
            merged["daily_basic_features_available_at_utc"] = pd.NaT
            for name in DAILY_BASIC_FEATURE_COLUMNS:
                merged[name] = np.nan
        else:
            right["__daily_basic_timestamp_v2"] = pd.to_datetime(
                right["daily_basic_observation_date"], errors="raise"
            ).astype("datetime64[ns]")
            right = right.sort_values("__daily_basic_timestamp_v2", kind="mergesort")
            merged = pd.merge_asof(
                left,
                right.drop(columns="stock_code"),
                left_on="__published_timestamp_v2",
                right_on="__daily_basic_timestamp_v2",
                direction="backward",
                allow_exact_matches=False,
            ).drop(columns="__daily_basic_timestamp_v2")
        pieces.append(merged)
    combined = pd.concat(pieces, ignore_index=True).sort_values(
        "__row_order_v2", kind="mergesort"
    )
    return combined.drop(columns=["__row_order_v2", "__published_timestamp_v2"])


def _load_audited_daily_basic(
    *,
    root: Path,
    state_path: Path,
    audit_path: Path,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
) -> tuple[pd.DataFrame, dict]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("contract") != DAILY_BASIC_AUDIT_CONTRACT or not audit.get("passed"):
        raise PermissionError("daily_basic audit is absent or failed")
    if audit.get("official_test_queried") is not False:
        raise PermissionError("daily_basic audit does not prove test isolation")
    if audit.get("tabular_contract_sha256") != tabular.source_sha256:
        raise ValueError("daily_basic audit tabular hash mismatch")
    if audit.get("split_source_sha256") != split.source_sha256:
        raise ValueError("daily_basic audit current split hash mismatch")
    if audit.get("state_sha256") != _sha256_file(state_path):
        raise ValueError("daily_basic state changed after audit")
    if state.get("official_test_queried") is not False or state.get("failures"):
        raise PermissionError("daily_basic state is not complete and train-only")
    frames = []
    resolved_root = root.resolve()
    for key, item in sorted(state.get("completed", {}).items()):
        if item.get("status") == "empty":
            continue
        if item.get("status") != "downloaded":
            raise ValueError(f"unexpected daily_basic status: {key}")
        path = (resolved_root / item["path"]).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"daily_basic path escapes root: {key}") from exc
        if _sha256_file(path) != item.get("sha256"):
            raise ValueError(f"daily_basic hash mismatch: {key}")
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("no audited daily_basic rows are available")
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate stock-date in daily_basic source")
    return build_daily_basic_features(combined), audit


def build_tabular_train_dataset_v2(
    *,
    v1_features: pd.DataFrame,
    v1_labels: pd.DataFrame,
    v1_manifest: dict,
    daily_basic_root: Path,
    daily_basic_state_path: Path,
    daily_basic_audit_path: Path,
    catalog: TabularFeatureCatalog,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build and validate a train-only v2 wide table without mutating v1."""
    if v1_manifest.get("contract") != V1_DATASET_CONTRACT:
        raise ValueError("unexpected base train_v1 dataset contract")
    if v1_manifest.get("official_test_queried") is not False:
        raise PermissionError("base train_v1 manifest does not prove test isolation")
    if v1_manifest.get("dataset_role") != "train":
        raise PermissionError("base tabular dataset is not train-only")
    if v1_manifest.get("contracts", {}).get("tabular") != tabular.source_sha256:
        raise ValueError("base dataset tabular contract hash mismatch")
    base_columns = tuple(v1_manifest.get("feature_columns", []))
    expected_columns = (*base_columns, *DAILY_BASIC_FEATURE_COLUMNS)
    if catalog.feature_names != expected_columns:
        raise ValueError("v2 catalog does not equal base plus daily_basic schema")
    require_train_only(v1_features)
    require_train_only(v1_labels)
    if set(v1_features["sample_id"]) != set(v1_labels["sample_id"]):
        raise ValueError("base feature and label sample identities differ")
    daily_basic, daily_basic_audit = _load_audited_daily_basic(
        root=daily_basic_root,
        state_path=daily_basic_state_path,
        audit_path=daily_basic_audit_path,
        tabular=tabular,
        split=split,
    )
    features = merge_strictly_prior_daily_basic(v1_features, daily_basic)
    features["daily_basic_available"] = features[
        "daily_basic_observation_date"
    ].notna()
    features["daily_basic_missing_reason"] = np.where(
        features["daily_basic_available"], "", "no_strictly_prior_daily_basic_row"
    )
    selected = validate_and_select_features(
        features,
        specs=catalog.feature_specs,
        tabular=tabular,
        development=development,
    )
    if selected.columns.tolist() != list(catalog.feature_names):
        raise ValueError("validated v2 feature schema order changed")
    if features["sample_id"].duplicated().any() or v1_labels[
        "sample_id"
    ].duplicated().any():
        raise ValueError("duplicate sample identity in tabular v2")
    feature_columns = list(catalog.feature_names)
    metadata_columns = [
        name
        for name in features.columns
        if name not in feature_columns and not name.startswith("__")
    ]
    features = features.loc[:, [*metadata_columns, *feature_columns]]
    labels = v1_labels.copy()
    schema_sha256 = _stable_sha256(*feature_columns)
    manifest = {
        "contract": V2_DATASET_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "train_range": {
            "start": split.train.start.isoformat(),
            "end": split.train.end.isoformat(),
        },
        "rows": len(features),
        "rows_by_horizon": v1_manifest["rows_by_horizon"],
        "feature_columns": feature_columns,
        "categorical_feature_columns": v1_manifest[
            "categorical_feature_columns"
        ],
        "feature_groups": {
            name: list(values) for name, values in catalog.feature_groups.items()
        },
        "feature_catalog": [
            {
                "name": item.name,
                "group": item.group,
                "source": item.source,
                "available_at_column": item.available_at_column,
                "observation_date_column": item.observation_date_column,
                "unit": item.unit,
                "transformation": item.transformation,
                "missing_policy": item.missing_policy,
            }
            for item in catalog.features
        ],
        "feature_schema_sha256": schema_sha256,
        "label_columns": v1_manifest["label_columns"],
        "primary_risk": {
            "target": catalog.primary_risk_target,
            "horizons_sessions": list(catalog.primary_risk_horizons),
            "output": catalog.primary_risk_output,
        },
        "unified_output_schema": catalog.unified_output_schema,
        "unified_output_fields": list(catalog.unified_output_fields),
        "blocked_feature_groups": dict(catalog.blocked_feature_groups),
        "contracts": {
            "catalog": catalog.source_sha256,
            "tabular": tabular.source_sha256,
            "split": split.source_sha256,
            "development": development.source_sha256,
            "base_v1_split": v1_manifest["contracts"]["split"],
            "base_v1_development": v1_manifest["contracts"]["development"],
        },
        "sources": {
            "base_v1_feature_schema_sha256": v1_manifest[
                "feature_schema_sha256"
            ],
            "base_v1_feature_artifact_sha256": v1_manifest["artifacts"][
                "features"
            ]["sha256"],
            "base_v1_label_artifact_sha256": v1_manifest["artifacts"]["labels"][
                "sha256"
            ],
            "daily_basic_state_sha256": daily_basic_audit["state_sha256"],
            "daily_basic_audit_sha256": _sha256_file(daily_basic_audit_path),
            "daily_basic_snapshot_sha256": daily_basic_audit["snapshot_sha256"],
        },
        "coverage": {
            "daily_basic_available_rows": int(features["daily_basic_available"].sum()),
            "daily_basic_missing_rows": int((~features["daily_basic_available"]).sum()),
            "daily_basic_available_fraction": float(
                features["daily_basic_available"].mean()
            ),
        },
        "missing_fraction": {
            name: float(features[name].isna().mean()) for name in feature_columns
        },
    }
    return features, labels, manifest
