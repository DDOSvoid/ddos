from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from src.prediction.tabular.point_in_time_sources import (
    PIT_CONFIG_CONTRACT,
    _normalize_financial,
    _normalize_industry,
    load_point_in_time_source_contract,
)


def test_point_in_time_source_contract_is_train_only() -> None:
    contract = load_point_in_time_source_contract()
    assert contract.dataset_role == "train"
    assert contract.financial_end == date(2024, 12, 31)
    assert set(contract.financial_fields) == {"income", "balancesheet", "cashflow"}
    assert PIT_CONFIG_CONTRACT == "tabular-point-in-time-sources-v1"


def test_financial_normalization_uses_actual_disclosure_end_of_day() -> None:
    fields = (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "revenue",
        "update_flag",
    )
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20230420",
                "f_ann_date": "20230421",
                "end_date": "20230331",
                "report_type": "1",
                "comp_type": "1",
                "end_type": "1",
                "revenue": 10.0,
                "update_flag": "1",
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20230820",
                "f_ann_date": None,
                "end_date": "20230630",
                "report_type": "1",
                "comp_type": "1",
                "end_type": "2",
                "revenue": 20.0,
                "update_flag": "1",
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20241231",
                "f_ann_date": "20250102",
                "end_date": "20240930",
                "report_type": "1",
                "comp_type": "1",
                "end_type": "3",
                "revenue": 30.0,
                "update_flag": "1",
            },
        ]
    )
    result = _normalize_financial(
        raw,
        stock_code="000001.SZ",
        endpoint="income",
        fields=fields,
        start=date(2020, 1, 1),
        end=date(2024, 12, 31),
        fetched_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert result["actual_disclosure_date"].tolist() == [
        date(2023, 4, 21),
        date(2023, 8, 20),
        date(2025, 1, 2),
    ]
    assert result["availability_date_source"].tolist() == [
        "f_ann_date",
        "ann_date",
        "f_ann_date",
    ]
    assert result["available_at_utc"].dt.tz is not None
    assert result["dataset_role"].eq("train").all()


def test_industry_normalization_preserves_overlapping_raw_intervals() -> None:
    fields = (
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "ts_code",
        "name",
        "in_date",
        "out_date",
        "is_new",
    )
    raw = pd.DataFrame(
        [
            {
                "l1_code": "801730.SI",
                "l1_name": "电力设备",
                "l2_code": "801737.SI",
                "l2_name": "电池",
                "l3_code": "857372.SI",
                "l3_name": "电池化学品",
                "ts_code": "000009.SZ",
                "name": "中国宝安",
                "in_date": "20000104",
                "out_date": None,
                "is_new": "Y",
            }
        ]
    )
    result = _normalize_industry(
        raw,
        stock_code="000009.SZ",
        fields=fields,
        query_is_new="Y",
        fetched_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert result.loc[0, "in_date"] == date(2000, 1, 4)
    assert pd.isna(result.loc[0, "out_date"])
    assert result.loc[0, "query_is_new"] == "Y"
    assert result.loc[0, "source"] == "tushare_index_member_all"
