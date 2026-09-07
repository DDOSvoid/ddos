import pandas as pd

from scripts.audit_electrical_pilot_readiness import check_daily_frame


def frame():
    return pd.DataFrame({
        "ts_code": ["000001.SZ"], "trade_date": ["2023-04-04"],
        "available_at_utc": ["2023-04-04T10:00:00Z"],
    })


def test_valid_daily_availability():
    assert check_daily_frame(frame(), "000001.SZ") == []


def test_future_date_and_early_availability_rejected():
    value = frame()
    value.loc[0, "trade_date"] = "2026-01-01"
    assert "invalid/out-of-train trade dates" in check_daily_frame(value, "000001.SZ")
    value = frame()
    value.loc[0, "available_at_utc"] = "2023-04-04T01:00:00Z"
    assert "daily availability is not 18:00 Shanghai" in check_daily_frame(value, "000001.SZ")


def test_duplicate_and_wrong_stock_rejected():
    value = pd.concat([frame(), frame()])
    errors = check_daily_frame(value, "000002.SZ")
    assert "duplicate stock dates" in errors
    assert "stock identity mismatch" in errors
