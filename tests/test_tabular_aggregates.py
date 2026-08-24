"""Causal 15-minute to daily aggregation tests."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.aggregates import (
    EXPECTED_BAR_LABELS,
    INTRADAY_FEATURE_COLUMNS,
    aggregate_intraday_days,
    intraday_feature_specs,
)
from src.prediction.tabular.contract import load_tabular_model_contract


def _complete_day() -> pd.DataFrame:
    labels = [f"2024-04-24 {value}:00" for value in EXPECTED_BAR_LABELS]
    trade_time = pd.to_datetime(labels)
    prices = np.linspace(10.0, 10.3, len(labels))
    return pd.DataFrame(
        {
            "ts_code": ["300750.SZ"] * len(labels),
            "trade_time": trade_time,
            "trade_date": [date(2024, 4, 24)] * len(labels),
            "open": prices,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices + 0.02,
            "vol": np.arange(1, len(labels) + 1) * 100,
            "amount": np.arange(1, len(labels) + 1) * 1000.0,
            "available_at_utc": (
                trade_time.tz_localize("Asia/Shanghai")
                + pd.Timedelta(minutes=15)
            ).tz_convert("UTC"),
            "source": ["amazingdata_query_kline"] * len(labels),
            "frequency": ["15min"] * len(labels),
            "bar_timestamp_semantics": ["bar_start"] * len(labels),
            "price_adjustment": ["raw"] * len(labels),
        }
    )


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return aggregate_intraday_days(
        frame,
        source_file_sha256="a" * 64,
        tabular=load_tabular_model_contract(),
        split=load_prediction_split_contract(),
    )


def test_complete_day_produces_only_daily_lagged_features():
    aggregate = _aggregate(_complete_day())
    assert len(aggregate) == 1
    assert aggregate.iloc[0]["bar_count"] == 16
    assert str(aggregate.iloc[0]["intraday_features_available_at_utc"]) == (
        "2024-04-24 07:00:00+00:00"
    )
    assert aggregate.iloc[0]["intraday_open_to_close_return"] == pytest.approx(
        10.32 / 10.0 - 1.0
    )
    assert aggregate.iloc[0]["intraday_last_30m_amount_share"] == pytest.approx(
        (15 + 16) / sum(range(1, 17))
    )
    assert set(INTRADAY_FEATURE_COLUMNS).issubset(aggregate.columns)
    assert not any("future" in column for column in aggregate.columns)


def test_incomplete_day_is_rejected_instead_of_silently_aggregated():
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        _aggregate(_complete_day().iloc[:-1].copy())


def test_bar_start_availability_must_include_full_15_minutes():
    frame = _complete_day()
    frame["available_at_utc"] = frame["available_at_utc"] - pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="availability semantics changed"):
        _aggregate(frame)


def test_all_intraday_features_share_causal_availability_metadata():
    specs = intraday_feature_specs()
    assert tuple(spec.name for spec in specs) == INTRADAY_FEATURE_COLUMNS
    assert {spec.source for spec in specs} == {"market_intraday_aggregate"}
    assert {spec.available_at_column for spec in specs} == {
        "intraday_features_available_at_utc"
    }
    assert {spec.observation_date_column for spec in specs} == {"observation_date"}
