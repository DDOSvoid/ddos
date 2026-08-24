"""Point-in-time tabular train-dataset tests."""

from datetime import date

import pandas as pd

from src.prediction.tabular.dataset import (
    DAILY_MARKET_FEATURE_COLUMNS,
    _merge_strictly_prior_market,
    build_daily_market_features,
)


def _price_rows() -> pd.DataFrame:
    rows = []
    for code, instrument_type in (("300750.SZ", "stock"), ("000300.SH", "index")):
        for index, trade_date in enumerate(pd.date_range("2024-01-01", periods=25, freq="D")):
            close = 100.0 + index
            rows.append(
                {
                    "instrument_code": code,
                    "instrument_type": instrument_type,
                    "trade_date": trade_date.date(),
                    "open_price": close - 0.5,
                    "high_price": close + 1.0,
                    "low_price": close - 1.0,
                    "close_price": close,
                    "pre_close": close - 1.0,
                    "pct_change": 100.0 * (close / (close - 1.0) - 1.0),
                    "volume": 1000.0 + index,
                    "amount": 10000.0 + index,
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def test_market_merge_is_strictly_before_publication_and_normalizes_datetime_units():
    market = build_daily_market_features(_price_rows())
    samples = pd.DataFrame(
        {
            "stock_code": ["300750.SZ"],
            "published_date": [date(2024, 1, 20)],
            "published_timestamp": pd.Series(
                ["2024-01-20"], dtype="datetime64[s]"
            ),
            "__row_order": [0],
        }
    )
    merged = _merge_strictly_prior_market(samples, market)
    assert merged.iloc[0]["market_observation_date"] == date(2024, 1, 19)
    assert set(DAILY_MARKET_FEATURE_COLUMNS).issubset(merged.columns)
