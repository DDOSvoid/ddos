from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from src.prediction.tabular.industry_supplement import (
    INDUSTRY_SUPPLEMENT_CONFIG_CONTRACT,
    _active_identities,
    _normalize_membership,
    _normalize_membership_all,
    load_industry_supplement_contract,
)


def test_industry_supplement_contract_is_train_only() -> None:
    contract = load_industry_supplement_contract()
    assert INDUSTRY_SUPPLEMENT_CONFIG_CONTRACT == "tabular-industry-supplement-v1"
    assert contract.classification_sources == ("SW2014", "SW2021")
    assert contract.classification_levels == ("L1", "L2", "L3")
    assert contract.membership_all_fields[0] == "l1_code"


def test_legacy_membership_normalization_preserves_effective_interval() -> None:
    fields = ("index_code", "con_code", "in_date", "out_date", "is_new")
    raw = pd.DataFrame(
        [
            {
                "index_code": "801050.SI",
                "con_code": "300390.SZ",
                "in_date": "20220801",
                "out_date": "20230918",
                "is_new": "N",
            }
        ]
    )
    result = _normalize_membership(
        raw,
        stock_code="300390.SZ",
        fields=fields,
        fetched_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert result.loc[0, "in_date"] == date(2022, 8, 1)
    assert result.loc[0, "out_date"] == date(2023, 9, 18)
    assert result.loc[0, "query_ts_code"] == "300390.SZ"
    assert _active_identities(result, date(2023, 1, 3), "index_code") == {
        "801050.SI"
    }
    assert not _active_identities(result, date(2023, 9, 18), "index_code")


def test_index_member_all_normalization_maps_l1_interval() -> None:
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
                "ts_code": "688005.SH",
                "name": "容百科技",
                "in_date": "20230925",
                "out_date": None,
                "is_new": "Y",
            }
        ]
    )
    result = _normalize_membership_all(
        raw,
        stock_code="688005.SH",
        fields=fields,
        fetched_at=datetime(2026, 8, 25, tzinfo=UTC),
    )
    assert result.loc[0, "index_code"] == "801730.SI"
    assert result.loc[0, "con_code"] == "688005.SH"
    assert _active_identities(result, date(2024, 1, 2), "index_code") == {
        "801730.SI"
    }
