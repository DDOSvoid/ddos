from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.prediction.timeseries.lstm_contract import load_lstm_timeseries_contract
from src.prediction.timeseries.sequence_identity import (
    canonical_company_day_id,
    canonical_sample_id,
)
from src.prediction.timeseries.sequence_labels import build_train_sequence_labels


def _bars(days: list[date], closes: list[float]) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "trade_date": days,
            "open_price": close - 0.5,
            "high_price": close + 1.0,
            "low_price": close - 1.0,
            "close_price": close,
            "pre_close": np.r_[close[0] - 1.0, close[:-1]],
            "volume": 1_000.0,
            "amount": 10_000.0,
        }
    )


def test_train_labels_are_recomputed_from_strictly_future_train_bars():
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
    days = [date(2024, 4, 24), date(2024, 4, 26), date(2024, 4, 29), date(2024, 4, 30), date(2024, 5, 6), date(2024, 5, 7)]
    stock = _bars(days, [10, 11, 12, 13, 14, 15])
    benchmark = _bars(days, [100, 101, 102, 103, 104, 105])
    contract = load_lstm_timeseries_contract()
    labels = build_train_sequence_labels(
        company_days,
        stock_bars_by_code={"000001.SZ": stock},
        benchmark_bars=benchmark,
        contract=contract,
    )

    assert labels["horizon_sessions"].tolist() == [1, 3, 5]
    first = labels.iloc[0]
    expected = (11 / 10.5 - 1) - (101 / 100.5 - 1)
    assert first["entry_date"] == date(2024, 4, 26)
    assert first["future_excess_return"] == pytest.approx(expected)
    assert first["sample_id"] == canonical_sample_id(company_day_id, 1)
    assert first["dataset_role"] == "train"
    assert first["outcome_available_at"].tzinfo is not None
