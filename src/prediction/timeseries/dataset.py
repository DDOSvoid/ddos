"""Physical feature/label separation for time-series training rows."""

from __future__ import annotations

import pandas as pd

from src.prediction.timeseries.contract import TimeseriesModelContract
from src.prediction.timeseries.walk_forward import require_timeseries_train_only

FEATURE_METADATA_COLUMNS = frozenset(
    {
        "sample_id",
        "stock_code",
        "published_date",
        "prediction_as_of",
        "last_bar_trade_date",
        "last_bar_available_at",
        "dataset_role",
    }
)
LABEL_COLUMNS = frozenset(
    {
        "sample_id",
        "horizon_sessions",
        "target",
        "excess_return",
        "outcome_available_at",
        "dataset_role",
    }
)


def assemble_train_frame(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str],
    contract: TimeseriesModelContract,
) -> pd.DataFrame:
    """Join separately stored train artifacts after schema and role checks."""
    feature_names = tuple(feature_columns)
    if not feature_names:
        raise ValueError("at least one time-series feature is required")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("duplicate time-series feature names")
    missing_features = [name for name in feature_names if name not in features]
    if missing_features:
        raise ValueError(f"missing time-series feature columns: {missing_features}")
    forbidden = [name for name in feature_names if not contract.feature_name_allowed(name)]
    if forbidden:
        raise ValueError(f"target or future time-series features rejected: {forbidden}")
    required_feature_metadata = FEATURE_METADATA_COLUMNS - set(features.columns)
    if required_feature_metadata:
        raise ValueError(
            "feature artifact missing temporal metadata: "
            f"{sorted(required_feature_metadata)}"
        )
    required_labels = LABEL_COLUMNS - set(labels.columns)
    if required_labels:
        raise ValueError(f"label artifact missing columns: {sorted(required_labels)}")
    unexpected_labels = set(labels.columns) - LABEL_COLUMNS
    if unexpected_labels:
        raise ValueError(f"label artifact contains unexpected columns: {sorted(unexpected_labels)}")
    require_timeseries_train_only(features)
    require_timeseries_train_only(labels)
    if features["sample_id"].duplicated().any():
        raise ValueError("feature artifact contains duplicate sample_id")
    if labels["sample_id"].duplicated().any():
        raise ValueError("label artifact contains duplicate sample_id")

    feature_payload = features.loc[
        :, [*FEATURE_METADATA_COLUMNS, *feature_names]
    ].copy()
    label_payload = labels.drop(columns="dataset_role").copy()
    merged = feature_payload.merge(
        label_payload,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(labels):
        raise ValueError("feature and label sample_id sets do not match")
    return merged.sort_values(
        ["published_date", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
