"""Availability-aware feature validation for causal tabular models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.tabular.contract import TabularModelContract

LAGGED_MARKET_SOURCES = frozenset(
    {"market_daily", "market_intraday_aggregate"}
)


@dataclass(frozen=True)
class FeatureSpec:
    """One model input and the columns proving when its value existed."""

    name: str
    source: str
    available_at_column: str
    observation_date_column: str | None = None


def _aware_timestamp(value: object, *, field: str) -> pd.Timestamp:
    if pd.isna(value):
        raise ValueError(f"{field} contains a missing timestamp")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def validate_and_select_features(
    frame: pd.DataFrame,
    *,
    specs: list[FeatureSpec] | tuple[FeatureSpec, ...],
    tabular: TabularModelContract,
    development: CausalDevelopmentContract,
    published_date_column: str = "published_date",
    prediction_as_of_column: str = "prediction_as_of",
) -> pd.DataFrame:
    """Reject leaked inputs and return only validated model feature columns."""
    if not specs:
        raise ValueError("at least one tabular feature is required")
    feature_names = [spec.name for spec in specs]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("duplicate tabular feature names")
    for required in (published_date_column, prediction_as_of_column):
        if required not in frame.columns:
            raise ValueError(f"missing required temporal column: {required}")

    for spec in specs:
        if not tabular.feature_name_allowed(spec.name):
            raise ValueError(f"target or post-outcome feature rejected: {spec.name}")
        if spec.name not in frame.columns:
            raise ValueError(f"missing feature column: {spec.name}")
        if spec.available_at_column not in frame.columns:
            raise ValueError(
                f"missing availability metadata for {spec.name}: "
                f"{spec.available_at_column}"
            )
        if spec.source in LAGGED_MARKET_SOURCES and not spec.observation_date_column:
            raise ValueError(
                f"lagged market feature requires an observation date: {spec.name}"
            )
        if spec.observation_date_column and spec.observation_date_column not in frame:
            raise ValueError(
                f"missing observation-date column for {spec.name}: "
                f"{spec.observation_date_column}"
            )

    for row_index, row in frame.iterrows():
        prediction_as_of = _aware_timestamp(
            row[prediction_as_of_column],
            field=f"{prediction_as_of_column}[{row_index}]",
        )
        published_date = pd.Timestamp(row[published_date_column]).date()
        for spec in specs:
            # Missing feature values carry no information and therefore have no
            # availability timestamp to prove. A present value must always have
            # complete point-in-time metadata.
            if pd.isna(row[spec.name]):
                continue
            available_at = _aware_timestamp(
                row[spec.available_at_column],
                field=f"{spec.available_at_column}[{row_index}]",
            )
            development.require_available(
                feature_name=spec.name,
                available_at=available_at.to_pydatetime(),
                prediction_as_of=prediction_as_of.to_pydatetime(),
            )
            if spec.source in LAGGED_MARKET_SOURCES:
                observation_date = pd.Timestamp(
                    row[spec.observation_date_column]
                ).date()
                if not development.historical_daily_bar_allowed(
                    trade_date=observation_date,
                    announcement_published_date=published_date,
                ):
                    raise ValueError(
                        f"same-day or future market bar rejected for {spec.name}: "
                        f"{observation_date} versus publication {published_date}"
                    )
    return frame.loc[:, feature_names].copy()


def require_train_only(
    frame: pd.DataFrame,
    *,
    role_column: str = "dataset_role",
) -> None:
    """Prevent test, quarantine, or forward rows from fitting/model selection."""
    if role_column not in frame.columns:
        raise ValueError(f"missing dataset role column: {role_column}")
    roles = {str(value) for value in frame[role_column].dropna().unique()}
    if roles != {"train"}:
        raise PermissionError(
            f"tabular fitting/model selection accepts train only; found {sorted(roles)}"
        )
