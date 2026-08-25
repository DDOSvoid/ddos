from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.lstm_contract import load_lstm_timeseries_contract
from src.prediction.timeseries.sequence_artifacts import build_direct_sequence_bundle
from src.prediction.timeseries.sequence_identity import canonical_company_day_id

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _bars(days: pd.DatetimeIndex, *, scale: float) -> pd.DataFrame:
    index = np.arange(len(days), dtype=float)
    close = scale + index * 0.2
    return pd.DataFrame(
        {
            "trade_date": days.date,
            "open_price": close * 0.998,
            "high_price": close * 1.01,
            "low_price": close * 0.99,
            "close_price": close,
            "pre_close": np.r_[close[0] / 1.001, close[:-1]],
            "volume": 1_000_000 + index * 1_000,
            "amount": 10_000_000 + index * 10_000,
        }
    )


def _contracts():
    development = load_causal_development_contract()
    parent = load_timeseries_model_contract(development=development)
    contract = load_lstm_timeseries_contract(
        parent=parent,
        development=development,
    )
    return development, parent, contract


def test_direct_sequence_bundle_is_label_free_and_uses_shared_prediction_time():
    published = date(2024, 4, 25)
    days = pd.bdate_range("2024-01-01", "2024-05-03")
    company_days = pd.DataFrame(
        {
            "company_day_id": [canonical_company_day_id("000001.SZ", published)],
            "stock_code": ["000001.SZ"],
            "published_date": [published],
            "dataset_role": ["train"],
        }
    )
    development, parent, contract = _contracts()
    bundle = build_direct_sequence_bundle(
        company_days,
        stock_bars_by_code={"000001.SZ": _bars(days, scale=10)},
        benchmark_bars=_bars(days, scale=3_000),
        parent_contract=parent,
        contract=contract,
        development=development,
    )

    assert bundle.sequences.shape == (1, 60, 10)
    assert bundle.time_masks.shape == (1, 60)
    assert bundle.time_masks.all()
    assert bundle.metadata.loc[0, "sequence_available"]
    assert bundle.metadata.loc[0, "prediction_as_of"] == datetime(
        2024, 4, 26, 8, 30, tzinfo=SHANGHAI
    )
    assert bundle.metadata.loc[0, "last_bar_trade_date"] < published
    assert "target" not in bundle.metadata
    assert "future_excess_return" not in bundle.metadata


def test_missing_stock_sequence_preserves_canonical_identity():
    published = date(2024, 4, 25)
    company_day_id = canonical_company_day_id("000001.SZ", published)
    company_days = pd.DataFrame(
        {
            "company_day_id": [company_day_id],
            "stock_code": ["000001.SZ"],
            "published_date": [published],
            "dataset_role": ["train"],
        }
    )
    days = pd.bdate_range("2024-01-01", "2024-05-03")
    development, parent, contract = _contracts()
    bundle = build_direct_sequence_bundle(
        company_days,
        stock_bars_by_code={},
        benchmark_bars=_bars(days, scale=3_000),
        parent_contract=parent,
        contract=contract,
        development=development,
    )

    assert bundle.sequences.shape == (0, 60, 10)
    assert bundle.metadata.loc[0, "company_day_id"] == company_day_id
    assert not bundle.metadata.loc[0, "sequence_available"]
    assert bundle.metadata.loc[0, "unavailable_reason"] == "missing_stock_bars"


def test_outcome_column_is_rejected_before_sequence_construction():
    published = date(2024, 4, 25)
    company_days = pd.DataFrame(
        {
            "company_day_id": [canonical_company_day_id("000001.SZ", published)],
            "stock_code": ["000001.SZ"],
            "published_date": [published],
            "dataset_role": ["train"],
            "future_excess_return": [0.10],
        }
    )
    days = pd.bdate_range("2024-01-01", "2024-05-03")
    development, parent, contract = _contracts()

    with pytest.raises(ValueError, match="outcome columns rejected"):
        build_direct_sequence_bundle(
            company_days,
            stock_bars_by_code={"000001.SZ": _bars(days, scale=10)},
            benchmark_bars=_bars(days, scale=3_000),
            parent_contract=parent,
            contract=contract,
            development=development,
        )
