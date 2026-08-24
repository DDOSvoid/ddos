"""Daily-price cache tests."""

from datetime import UTC, datetime

import pandas as pd

from scripts.backfill_market_prices import upsert_bars
from src.database.models import DailyPrice


def _frame(close: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260820",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": close,
                "pre_close": 9.9,
                "pct_chg": 2.0,
                "vol": 1000.0,
                "amount": 10000.0,
            }
        ]
    )


def test_daily_price_upsert_is_idempotent(db_session):
    fetched_at = datetime(2026, 8, 20, 9, tzinfo=UTC)
    assert upsert_bars(
        db_session,
        _frame(10.1),
        instrument_type="stock",
        fetched_at=fetched_at,
    ) == 1
    db_session.commit()
    assert upsert_bars(
        db_session,
        _frame(10.2),
        instrument_type="stock",
        fetched_at=fetched_at,
    ) == 1
    db_session.commit()
    rows = db_session.query(DailyPrice).all()
    assert len(rows) == 1
    assert rows[0].close_price == 10.2
