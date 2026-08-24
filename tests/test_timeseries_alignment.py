"""Announcement decision and bar-availability alignment tests."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.alignment import (
    daily_bar_available_at,
    daily_v1_prediction_as_of,
    first_authoritative_preopen_decision,
    require_daily_v1_bar_usable,
    require_timestamped_bar_available,
)
from src.prediction.timeseries.contract import load_timeseries_model_contract

SHANGHAI = ZoneInfo("Asia/Shanghai")


def test_daily_v1_rejects_publication_day_close_even_if_input_contains_it():
    development = load_causal_development_contract()
    contract = load_timeseries_model_contract(development=development)
    published = date(2024, 4, 25)
    as_of = daily_v1_prediction_as_of(published, development=development)
    same_day_available = daily_bar_available_at(
        published,
        contract=contract,
        development=development,
    )
    with pytest.raises(ValueError, match="same-day or future"):
        require_daily_v1_bar_usable(
            trade_date=published,
            published_date=published,
            prediction_as_of=as_of,
            bar_available_at=same_day_available,
            development=development,
        )


def test_after_close_authoritative_publication_moves_to_next_trading_day():
    development = load_causal_development_contract()
    published_at = datetime(2024, 4, 26, 18, 0, tzinfo=SHANGHAI)  # Friday
    decision = first_authoritative_preopen_decision(
        published_at,
        trading_days=[date(2024, 4, 26), date(2024, 4, 29)],
        publication_time_authoritative=True,
        development=development,
    )
    assert decision == datetime(2024, 4, 29, 8, 30, tzinfo=SHANGHAI)

    # Friday's completed close is known by Monday's decision; Monday's close is not.
    require_timestamped_bar_available(
        bar_available_at=datetime(2024, 4, 26, 15, 5, tzinfo=SHANGHAI),
        prediction_as_of=decision,
    )
    with pytest.raises(ValueError, match="future bar rejected"):
        require_timestamped_bar_available(
            bar_available_at=datetime(2024, 4, 29, 15, 5, tzinfo=SHANGHAI),
            prediction_as_of=decision,
        )


def test_intraday_timing_is_blocked_without_authoritative_timestamp():
    development = load_causal_development_contract()
    with pytest.raises(PermissionError, match="authoritative"):
        first_authoritative_preopen_decision(
            datetime(2024, 4, 26, 18, 0, tzinfo=SHANGHAI),
            trading_days=[date(2024, 4, 29)],
            publication_time_authoritative=False,
            development=development,
        )
