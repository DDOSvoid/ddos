from __future__ import annotations

from datetime import date

import pandas as pd

from src.prediction.tabular.point_in_time_audit import (
    _financial_sample_coverage,
    _industry_sample_coverage,
)


def test_industry_coverage_exposes_overlap_and_missing_dates() -> None:
    memberships = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "in_date": date(2020, 1, 1),
                "out_date": date(2023, 6, 1),
                "l1_code": "old",
                "l2_code": "old2",
                "l3_code": "old3",
            },
            {
                "ts_code": "000001.SZ",
                "in_date": date(2023, 1, 1),
                "out_date": None,
                "l1_code": "new",
                "l2_code": "new2",
                "l3_code": "new3",
            },
        ]
    )
    samples = pd.DataFrame(
        {
            "stock_code": ["000001.SZ"] * 3,
            "published_date": ["2019-01-01", "2023-03-01", "2023-07-01"],
        }
    )
    result = _industry_sample_coverage(memberships, samples)
    assert result["missing_company_days"] == 1
    assert result["ambiguous_company_days"] == 1
    assert result["unambiguous_company_days"] == 1


def test_financial_coverage_is_strictly_before_publication() -> None:
    samples = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ"],
            "published_date": ["2023-04-20", "2023-04-21"],
        }
    )
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "actual_disclosure_date": [date(2023, 4, 20)],
        }
    )
    result = _financial_sample_coverage(
        {"income": frame, "balancesheet": frame, "cashflow": frame}, samples
    )
    assert result["all_three_available_company_days"] == 1
    assert result["all_three_missing_company_days"] == 1
