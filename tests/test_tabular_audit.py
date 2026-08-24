"""End-to-end audit tests for train-only intraday aggregates."""

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.aggregates import (
    AGGREGATION_VERSION,
    EXPECTED_BAR_LABELS,
    aggregate_intraday_days,
    intraday_feature_schema_sha256,
)
from src.prediction.tabular.audit import audit_intraday_dataset
from src.prediction.tabular.contract import load_tabular_model_contract


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_day() -> pd.DataFrame:
    trade_time = pd.to_datetime(
        [f"2024-04-24 {label}:00" for label in EXPECTED_BAR_LABELS]
    )
    prices = np.linspace(10.0, 10.3, len(trade_time))
    return pd.DataFrame(
        {
            "ts_code": "300750.SZ",
            "trade_time": trade_time,
            "trade_date": date(2024, 4, 24),
            "open": prices,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices + 0.02,
            "vol": np.arange(1, 17) * 100,
            "amount": np.arange(1, 17) * 1000.0,
            "available_at_utc": (
                trade_time.tz_localize("Asia/Shanghai")
                + pd.Timedelta(minutes=15)
            ).tz_convert("UTC"),
            "source": "amazingdata_query_kline",
            "frequency": "15min",
            "bar_timestamp_semantics": "bar_start",
            "price_adjustment": "raw",
        }
    )


def _dataset(tmp_path):
    split = load_prediction_split_contract()
    tabular = load_tabular_model_contract(split=split)
    root = tmp_path / "intraday"
    raw_relative = "raw/amazingdata_query_kline/300750.SZ/2024.parquet"
    raw_path = root / raw_relative
    raw_path.parent.mkdir(parents=True)
    raw_frame = _raw_day()
    raw_frame.to_parquet(raw_path, index=False)
    raw_sha256 = _sha256(raw_path)

    aggregate_relative = (
        "aggregates/amazingdata_query_kline/300750.SZ/2024.parquet"
    )
    aggregate_path = root / aggregate_relative
    aggregate_path.parent.mkdir(parents=True)
    aggregate = aggregate_intraday_days(
        raw_frame,
        source_file_sha256=raw_sha256,
        tabular=tabular,
        split=split,
    )
    aggregate.to_parquet(aggregate_path, index=False)
    aggregate_sha256 = _sha256(aggregate_path)

    codes_sha256 = hashlib.sha256(b"300750.SZ").hexdigest()
    key_2023 = "300750.SZ:2023-01-01:2023-12-31"
    key_2024 = "300750.SZ:2024-01-01:2024-12-31"
    raw_state = {
        "tabular_contract_sha256": tabular.source_sha256,
        "split_source_sha256": split.source_sha256,
        "official_test_queried": False,
        "selection": {
            "dataset_role": "train",
            "start": "2023-01-01",
            "end": "2024-12-31",
            "frequency": "15min",
            "usage": "lagged_daily_aggregates_only",
            "source": "amazingdata_query_kline",
            "bar_timestamp_semantics": "bar_start",
            "codes": 1,
            "codes_sha256": codes_sha256,
        },
        "completed": {
            key_2023: {"status": "empty", "rows": 0, "path": None},
            key_2024: {
                "status": "downloaded",
                "rows": len(raw_frame),
                "path": raw_relative,
                "sha256": raw_sha256,
            },
        },
        "failures": {},
    }
    aggregate_state = {
        "tabular_contract_sha256": tabular.source_sha256,
        "split_source_sha256": split.source_sha256,
        "official_test_queried": False,
        "selection": {
            "aggregation_version": AGGREGATION_VERSION,
            "feature_schema_sha256": intraday_feature_schema_sha256(),
            "raw_codes_sha256": codes_sha256,
        },
        "completed": {
            key_2023: {
                "status": "empty_source",
                "rows": 0,
                "path": None,
                "source_file_sha256": None,
            },
            key_2024: {
                "status": "aggregated",
                "rows": len(aggregate),
                "path": aggregate_relative,
                "sha256": aggregate_sha256,
                "source_file_sha256": raw_sha256,
            },
        },
        "failures": {},
    }
    raw_state_path = root / "state.amazingdata.json"
    aggregate_state_path = root / "state.aggregates.json"
    raw_state_path.write_text(json.dumps(raw_state), encoding="utf-8")
    aggregate_state_path.write_text(json.dumps(aggregate_state), encoding="utf-8")
    return root, raw_state_path, aggregate_state_path, aggregate_path


def _audit(root, raw_state_path, aggregate_state_path):
    split = load_prediction_split_contract()
    return audit_intraday_dataset(
        raw_root=root,
        raw_state_path=raw_state_path,
        aggregate_root=root,
        aggregate_state_path=aggregate_state_path,
        tabular=load_tabular_model_contract(split=split),
        split=split,
    )


def test_audit_recomputes_and_accepts_matching_complete_dataset(tmp_path):
    root, raw_state, aggregate_state, _ = _dataset(tmp_path)
    report = _audit(root, raw_state, aggregate_state)
    assert report["passed"] is True
    assert report["stats"] == {
        "stocks": 1,
        "expected_chunks": 2,
        "downloaded_chunks": 1,
        "empty_chunks": 1,
        "raw_rows": 16,
        "aggregate_rows": 1,
    }


def test_audit_rejects_feature_file_that_no_longer_matches_raw_bars(tmp_path):
    root, raw_state, aggregate_state_path, aggregate_path = _dataset(tmp_path)
    changed = pd.read_parquet(aggregate_path)
    changed.loc[0, "intraday_realized_volatility"] += 1.0
    changed.to_parquet(aggregate_path, index=False)
    state = json.loads(aggregate_state_path.read_text(encoding="utf-8"))
    key = "300750.SZ:2024-01-01:2024-12-31"
    state["completed"][key]["sha256"] = _sha256(aggregate_path)
    aggregate_state_path.write_text(json.dumps(state), encoding="utf-8")

    report = _audit(root, raw_state, aggregate_state_path)
    assert report["passed"] is False
    assert any(
        key in error and "AssertionError" in error for error in report["errors"]
    )
