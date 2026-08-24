"""Chronological expanding-window frames for time-series development."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.prediction.development_contract import CausalDevelopmentContract


@dataclass(frozen=True)
class TimeseriesWalkForwardFrames:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame


def require_timeseries_train_only(
    frame: pd.DataFrame,
    *,
    role_column: str = "dataset_role",
) -> None:
    if role_column not in frame.columns:
        raise ValueError(f"missing dataset role column: {role_column}")
    roles = {str(value) for value in frame[role_column].dropna().unique()}
    if roles != {"train"}:
        raise PermissionError(
            "time-series fitting/model selection accepts train only; "
            f"found {sorted(roles)}"
        )


def _aware_datetime(value: object, *, field: str) -> pd.Timestamp:
    if pd.isna(value):
        raise ValueError(f"{field} contains a missing timestamp")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp


def iter_timeseries_expanding_frames(
    frame: pd.DataFrame,
    *,
    development: CausalDevelopmentContract,
    published_date_column: str = "published_date",
    outcome_available_at_column: str = "outcome_available_at",
    role_column: str = "dataset_role",
):
    """Yield train-only chronological folds and exclude immature outcomes."""
    require_timeseries_train_only(frame, role_column=role_column)
    for required in (published_date_column, outcome_available_at_column):
        if required not in frame.columns:
            raise ValueError(f"missing walk-forward temporal column: {required}")

    ordered = frame.copy()
    ordered["__published_date"] = pd.to_datetime(
        ordered[published_date_column], errors="raise"
    ).dt.date
    ordered = ordered.sort_values("__published_date", kind="mergesort")
    for fold in development.folds:
        roles = []
        for row_index, row in ordered.iterrows():
            outcome_available_at = _aware_datetime(
                row[outcome_available_at_column],
                field=f"{outcome_available_at_column}[{row_index}]",
            )
            roles.append(
                development.fold_role(
                    fold_name=fold.name,
                    published_date=row["__published_date"],
                    outcome_available_at=outcome_available_at.to_pydatetime(),
                )
            )
        role_series = pd.Series(roles, index=ordered.index)
        train = ordered.loc[role_series == "fold_train"].drop(
            columns="__published_date"
        )
        validation = ordered.loc[role_series == "fold_validation"].drop(
            columns="__published_date"
        )
        if train.empty or validation.empty:
            raise ValueError(f"{fold.name} has an empty train or validation frame")
        yield TimeseriesWalkForwardFrames(
            name=fold.name,
            train=train.copy(),
            validation=validation.copy(),
        )
