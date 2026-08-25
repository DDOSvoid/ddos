from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.alignment import (
    next_trading_preopen_prediction_as_of,
)
from src.prediction.timeseries.sequence_identity import (
    canonical_company_day_id,
    canonical_identities,
    canonical_sample_id,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def test_canonical_ids_match_existing_tabular_v1_identity():
    company_day_id, sample_id = canonical_identities(
        "000009.SZ",
        date(2023, 3, 1),
        1,
    )

    assert company_day_id == (
        "e99ec3d3c9bae93a75041f160a895a47c712122d8e88881f9aa723af39d7f213"
    )
    assert sample_id == (
        "96bb55dd84457aabba14dd5a0f075dc5f4d8b5bb99a583407c7d8fccd2e3301b"
    )


def test_next_trading_preopen_is_strictly_after_publication_date():
    development = load_causal_development_contract()
    prediction = next_trading_preopen_prediction_as_of(
        date(2024, 4, 26),
        trading_days=[date(2024, 4, 26), date(2024, 4, 29), date(2024, 4, 30)],
        development=development,
    )

    assert prediction == datetime(2024, 4, 29, 8, 30, tzinfo=SHANGHAI)


def test_next_trading_preopen_requires_a_later_calendar_session():
    development = load_causal_development_contract()

    with pytest.raises(ValueError, match="strictly after"):
        next_trading_preopen_prediction_as_of(
            date(2024, 4, 26),
            trading_days=[date(2024, 4, 25), date(2024, 4, 26)],
            development=development,
        )


def test_canonical_sample_id_rejects_unknown_horizon():
    company_day_id = canonical_company_day_id("000009.SZ", date(2023, 3, 1))

    with pytest.raises(ValueError, match="unsupported horizon"):
        canonical_sample_id(company_day_id, 10)
