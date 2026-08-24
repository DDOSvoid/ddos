"""Causal sequence construction and future-perturbation tests."""

from datetime import date

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.features import (
    build_daily_feature_panel,
    build_daily_sequence,
    summarize_daily_sequence,
    summary_feature_names,
)


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


def _sequence(stock: pd.DataFrame, benchmark: pd.DataFrame):
    development = load_causal_development_contract()
    contract = load_timeseries_model_contract(development=development)
    return build_daily_sequence(
        sample_id="sample-1",
        stock_code="000001.SZ",
        benchmark_code="000300.SH",
        published_date=date(2024, 4, 25),
        stock_bars=stock,
        benchmark_bars=benchmark,
        contract=contract,
        development=development,
    )


def test_daily_sequence_is_target_free_and_strictly_before_publication():
    days = pd.bdate_range("2024-01-01", "2024-05-10")
    sequence = _sequence(_bars(days, scale=10), _bars(days, scale=3_000))
    assert len(sequence.frame) == 60
    assert max(sequence.frame["trade_date"]) < sequence.published_date
    assert sequence.frame["bar_available_at"].max() <= sequence.prediction_as_of
    assert "target" not in sequence.frame
    assert "future_high" not in sequence.frame

    contract = load_timeseries_model_contract()
    summary = summarize_daily_sequence(sequence, contract=contract)
    assert summary["last_bar_trade_date"] < sequence.published_date
    assert "stock_momentum_20d" in summary
    assert "target" not in summary


def test_future_price_perturbation_does_not_change_historical_sequence_or_features():
    past_days = pd.bdate_range("2024-01-01", "2024-04-24")
    future_days = pd.bdate_range("2024-04-25", "2024-05-10")
    stock_past = _bars(past_days, scale=10)
    benchmark_past = _bars(past_days, scale=3_000)
    baseline = _sequence(stock_past, benchmark_past)

    stock_future = _bars(future_days, scale=1_000_000)
    benchmark_future = _bars(future_days, scale=9_000_000)
    perturbed = _sequence(
        pd.concat([stock_past, stock_future], ignore_index=True),
        pd.concat([benchmark_past, benchmark_future], ignore_index=True),
    )
    pdt.assert_frame_equal(baseline.frame, perturbed.frame)
    contract = load_timeseries_model_contract()
    assert summarize_daily_sequence(
        baseline, contract=contract
    ) == summarize_daily_sequence(perturbed, contract=contract)


def test_backward_only_feature_panel_matches_single_sequence_summary():
    days = pd.bdate_range("2024-01-01", "2024-05-10")
    stock = _bars(days, scale=10)
    benchmark = _bars(days, scale=3_000)
    sequence = _sequence(stock, benchmark)
    contract = load_timeseries_model_contract()
    summary = summarize_daily_sequence(sequence, contract=contract)
    panel = build_daily_feature_panel(
        stock_bars=stock,
        benchmark_bars=benchmark,
        contract=contract,
    )
    endpoint = panel.loc[panel["trade_date"] < sequence.published_date].iloc[-1]
    for name in summary_feature_names(contract):
        assert endpoint[name] == pytest.approx(summary[name], nan_ok=True)
