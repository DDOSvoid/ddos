"""Train-only daily-basic download tests."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.prediction.tabular.daily_basic import (
    DAILY_BASIC_FIELDS,
    normalize_daily_basic_frame,
)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{name: [1.0] for name in DAILY_BASIC_FIELDS[2:]},
            "ts_code": ["300750.SZ"],
            "trade_date": ["20240424"],
            "close": [190.0],
            "total_share": [240000.0],
            "total_mv": [45600000.0],
            "circ_mv": [40000000.0],
        }
    )


def test_daily_basic_normalization_records_conservative_availability_and_units():
    normalized = normalize_daily_basic_frame(
        _raw(),
        stock_code="300750.SZ",
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
        fetched_at=datetime(2026, 8, 22, tzinfo=UTC),
    )
    assert str(normalized.iloc[0]["available_at_utc"]) == "2024-04-24 10:00:00+00:00"
    assert normalized.iloc[0]["unit_market_value"] == "CNY_10K"
    assert normalized.iloc[0]["unit_share"] == "SHARES_10K"


def test_daily_basic_rejects_rows_outside_train_or_for_another_stock():
    outside = _raw()
    outside.loc[0, "trade_date"] = "20250102"
    with pytest.raises(ValueError, match="leaves the train partition"):
        normalize_daily_basic_frame(
            outside,
            stock_code="300750.SZ",
            start=date(2023, 1, 1),
            end=date(2024, 12, 31),
            fetched_at=datetime(2026, 8, 22, tzinfo=UTC),
        )
    other = _raw()
    other.loc[0, "ts_code"] = "000001.SZ"
    with pytest.raises(ValueError, match="unexpected stock code"):
        normalize_daily_basic_frame(
            other,
            stock_code="300750.SZ",
            start=date(2023, 1, 1),
            end=date(2024, 12, 31),
            fetched_at=datetime(2026, 8, 22, tzinfo=UTC),
        )
