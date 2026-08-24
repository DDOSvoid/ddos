"""Causal rolling tests for complete 15-minute trading-day aggregates."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.prediction.tabular.aggregates import (
    AGGREGATION_VERSION,
    INTRADAY_FEATURE_COLUMNS,
    intraday_feature_schema_sha256,
)
from src.prediction.timeseries.intraday_contract import (
    load_intraday_aggregate_timeseries_contract,
)
from src.prediction.timeseries.intraday_features import (
    build_intraday_rolling_panel,
    intraday_rolling_feature_names,
)


def _aggregates(days: pd.DatetimeIndex, *, offset: float = 0.0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "stock_code": ["000001.SZ"] * len(days),
            "observation_date": days.date,
            "intraday_features_available_at_utc": pd.to_datetime(days.date).tz_localize(
                "Asia/Shanghai"
            )
            + pd.Timedelta(hours=15),
            "frequency": ["15min_to_daily"] * len(days),
            "price_adjustment": ["raw_within_day_only"] * len(days),
            "aggregation_version": [AGGREGATION_VERSION] * len(days),
            "feature_schema_sha256": [intraday_feature_schema_sha256()] * len(days),
        }
    )
    base = np.arange(len(days), dtype=float) + offset
    for index, name in enumerate(INTRADAY_FEATURE_COLUMNS, 1):
        frame[name] = base / (100.0 + index)
    frame["intraday_features_available_at_utc"] = frame[
        "intraday_features_available_at_utc"
    ].dt.tz_convert("UTC")
    return frame


def test_intraday_panel_uses_complete_days_and_backward_rolling_only():
    contract = load_intraday_aggregate_timeseries_contract()
    days = pd.bdate_range("2024-01-01", periods=25)
    panel = build_intraday_rolling_panel(_aggregates(days), contract=contract)
    assert len(panel) == 25
    assert panel["complete_intraday_days"].iloc[-1] == 25
    assert set(intraday_rolling_feature_names(contract)) - {
        "intraday_staleness_calendar_days"
    } <= set(panel.columns)
    source = _aggregates(days)["intraday_amount_hhi"]
    assert panel["intraday_amount_hhi_mean_20d"].iloc[-1] == source.tail(20).mean()


def test_future_intraday_days_do_not_change_earlier_rolling_features():
    contract = load_intraday_aggregate_timeseries_contract()
    past_days = pd.bdate_range("2024-01-01", periods=25)
    future_days = pd.bdate_range(past_days[-1] + pd.Timedelta(days=1), periods=5)
    past = _aggregates(past_days)
    baseline = build_intraday_rolling_panel(past, contract=contract)
    extended = build_intraday_rolling_panel(
        pd.concat([past, _aggregates(future_days, offset=1_000_000)], ignore_index=True),
        contract=contract,
    )
    pdt.assert_frame_equal(baseline, extended.iloc[: len(baseline)].reset_index(drop=True))
