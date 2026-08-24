"""Backward-only rolling features from complete 15-minute trading days."""

from __future__ import annotations

import pandas as pd

from src.prediction.tabular.aggregates import (
    AGGREGATION_VERSION,
    INTRADAY_FEATURE_COLUMNS,
)
from src.prediction.timeseries.intraday_contract import (
    IntradayAggregateTimeseriesContract,
)

REQUIRED_AGGREGATE_METADATA = frozenset(
    {
        "stock_code",
        "observation_date",
        "intraday_features_available_at_utc",
        "frequency",
        "price_adjustment",
        "aggregation_version",
        "feature_schema_sha256",
        *INTRADAY_FEATURE_COLUMNS,
    }
)


def intraday_rolling_feature_names(
    contract: IntradayAggregateTimeseriesContract,
) -> tuple[str, ...]:
    names = []
    for source_name in INTRADAY_FEATURE_COLUMNS:
        names.append(f"{source_name}_last_1d")
        for lookback in contract.lookbacks_complete_days:
            if lookback == 1:
                continue
            names.append(f"{source_name}_mean_{lookback}d")
            names.append(f"{source_name}_std_{lookback}d")
    names.append("intraday_staleness_calendar_days")
    return tuple(names)


def build_intraday_rolling_panel(
    frame: pd.DataFrame,
    *,
    contract: IntradayAggregateTimeseriesContract,
) -> pd.DataFrame:
    """Create causal rolling summaries for one stock's complete intraday days."""
    missing = REQUIRED_AGGREGATE_METADATA - set(frame.columns)
    if missing:
        raise ValueError(f"intraday aggregate frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("intraday aggregate frame is empty")
    working = frame.copy()
    codes = set(working["stock_code"].astype(str).unique())
    if len(codes) != 1:
        raise ValueError("intraday rolling panel accepts exactly one stock")
    if set(working["frequency"].astype(str).unique()) != {"15min_to_daily"}:
        raise ValueError("intraday rolling panel requires complete-day aggregates")
    if set(working["price_adjustment"].astype(str).unique()) != {
        "raw_within_day_only"
    }:
        raise ValueError("intraday cross-day raw price usage is forbidden")
    if set(working["aggregation_version"].astype(str).unique()) != {
        AGGREGATION_VERSION
    }:
        raise ValueError("intraday aggregation version changed")
    if set(working["feature_schema_sha256"].astype(str).unique()) != {
        contract.source_feature_schema_sha256
    }:
        raise ValueError("intraday source feature schema changed")
    working["observation_date"] = pd.to_datetime(
        working["observation_date"], errors="raise"
    ).dt.date
    working["intraday_features_available_at_utc"] = pd.to_datetime(
        working["intraday_features_available_at_utc"],
        errors="raise",
        utc=True,
    )
    working = working.sort_values("observation_date", kind="mergesort").reset_index(
        drop=True
    )
    if working["observation_date"].duplicated().any():
        raise ValueError("intraday aggregate frame contains duplicate stock-days")
    for name in INTRADAY_FEATURE_COLUMNS:
        working[name] = pd.to_numeric(working[name], errors="raise")
        if working[name].isna().any():
            raise ValueError(f"intraday aggregate contains missing {name}")
        working[f"{name}_last_1d"] = working[name]
        for lookback in contract.lookbacks_complete_days:
            if lookback == 1:
                continue
            rolling = working[name].rolling(window=lookback, min_periods=1)
            working[f"{name}_mean_{lookback}d"] = rolling.mean()
            working[f"{name}_std_{lookback}d"] = working[name].rolling(
                window=lookback,
                min_periods=2,
            ).std(ddof=1)
    working["complete_intraday_days"] = range(1, len(working) + 1)
    return working.loc[
        :,
        [
            "observation_date",
            "intraday_features_available_at_utc",
            "complete_intraday_days",
            *(
                name
                for name in intraday_rolling_feature_names(contract)
                if name != "intraday_staleness_calendar_days"
            ),
        ],
    ].copy()
