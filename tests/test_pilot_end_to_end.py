import sqlite3

import pandas as pd
import pytest

from scripts.validate_pilot_end_to_end import build_samples


def fixture_db(path):
    with sqlite3.connect(path) as con:
        con.executescript("""
        CREATE TABLE companies(id INTEGER,stock_code TEXT);
        CREATE TABLE announcements(id INTEGER,company_id INTEGER,announcement_id TEXT,
            published_date TEXT,title TEXT);
        CREATE TABLE classifications(announcement_id INTEGER,major_category TEXT,
            needs_review INTEGER,relevance TEXT);
        CREATE TABLE daily_prices(instrument_code TEXT,trade_date TEXT,open_price REAL,
            high_price REAL,low_price REAL,close_price REAL,volume REAL,amount REAL);
        INSERT INTO companies VALUES(1,'000001.SZ');
        INSERT INTO announcements VALUES(1,1,'A1','2023-03-04','earnings'),
            (2,1,'A2','2023-03-05','buyback'),(3,1,'SEALED','2026-03-05','sealed');
        INSERT INTO classifications VALUES(1,'A',0,'core_event'),
            (2,'C',0,'core_event'),(3,'A',0,'core_event');
        """)
        for code in ("000001.SZ", "000300.SH"):
            for i, day in enumerate(pd.bdate_range("2023-01-01", "2023-03-20")):
                close = 100 + i / 10
                con.execute(
                    "INSERT INTO daily_prices VALUES(?,?,?,?,?,?,?,?)",
                    (code, str(day.date()), 100, 110, 90, close, 1000, 10000),
                )
    return {
        "announcement_ids": ["A1", "A2", "SEALED"],
        "dataset_role": "train",
        "market_targets_queried": False,
    }


def test_features_do_not_change_when_future_returns_change(tmp_path):
    path = tmp_path / "db.sqlite"
    plan = fixture_db(path)
    f1, l1, _ = build_samples(path, plan)
    with sqlite3.connect(path) as con:
        con.execute(
            "UPDATE daily_prices SET close_price=close_price*2 "
            "WHERE instrument_code='000001.SZ' AND trade_date >= '2023-03-06'"
        )
    f2, l2, _ = build_samples(path, plan)
    pd.testing.assert_frame_equal(f1, f2)
    assert not l1.future_excess_return.equals(l2.future_excess_return)


def test_weekend_dates_merge_and_sealed_metadata_excluded(tmp_path):
    path = tmp_path / "db.sqlite"
    plan = fixture_db(path)
    features, labels, resolved = build_samples(path, plan)
    assert resolved == 2
    assert len(features) == 3
    assert features.company_day_id.nunique() == 1
    assert features.tab_0.eq(2).all()
    assert labels.outcome_date.max() < "2025-01-01"
    assert (features.data_as_of < features.prediction_as_of).all()


def test_reject_outcome_selected_plan(tmp_path):
    path = tmp_path / "db.sqlite"
    plan = fixture_db(path)
    plan["market_targets_queried"] = True
    with pytest.raises(PermissionError):
        build_samples(path, plan)
