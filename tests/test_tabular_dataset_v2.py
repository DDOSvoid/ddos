"""Point-in-time tabular train_v2 feature tests."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from src.prediction.tabular.dataset_v2 import (
    DAILY_BASIC_FEATURE_COLUMNS,
    build_daily_basic_features,
    merge_strictly_prior_daily_basic,
)


def _raw_daily_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["300750.SZ", "300750.SZ"],
            "trade_date": [date(2024, 1, 19), date(2024, 1, 20)],
            "turnover_rate": [2.0, 3.0],
            "turnover_rate_f": [4.0, 5.0],
            "volume_ratio": [1.2, 1.3],
            "pe": [10.0, 11.0],
            "pe_ttm": [9.0, 10.0],
            "pb": [2.0, 2.1],
            "ps": [3.0, 3.1],
            "ps_ttm": [2.9, 3.0],
            "dv_ratio": [1.0, 1.1],
            "dv_ttm": [0.8, 0.9],
            "total_share": [100.0, 100.0],
            "float_share": [80.0, 80.0],
            "free_share": [60.0, 60.0],
            "total_mv": [1000.0, 1100.0],
            "circ_mv": [800.0, 880.0],
            "available_at_utc": [
                datetime(2024, 1, 19, 10, tzinfo=UTC),
                datetime(2024, 1, 20, 10, tzinfo=UTC),
            ],
        }
    )


def test_daily_basic_v2_transformations_are_explicit_and_finite():
    features = build_daily_basic_features(_raw_daily_basic())
    assert set(DAILY_BASIC_FEATURE_COLUMNS).issubset(features.columns)
    assert features.iloc[0]["daily_basic_turnover_rate"] == 0.02
    assert features.iloc[0]["daily_basic_float_share_ratio"] == 0.8
    assert features.iloc[0]["daily_basic_free_share_ratio"] == 0.6
    assert features.iloc[0]["daily_basic_circulating_market_value_ratio"] == 0.8
    assert features.iloc[0]["daily_basic_log_total_market_value"] == np.log1p(1000.0)


def test_daily_basic_v2_merge_rejects_publication_day_row():
    daily_basic = build_daily_basic_features(_raw_daily_basic())
    samples = pd.DataFrame(
        {
            "sample_id": ["sample"],
            "stock_code": ["300750.SZ"],
            "published_date": [date(2024, 1, 20)],
        }
    )
    merged = merge_strictly_prior_daily_basic(samples, daily_basic)
    assert merged.iloc[0]["daily_basic_observation_date"] == date(2024, 1, 19)
    assert merged.iloc[0]["daily_basic_pe"] == 10.0
