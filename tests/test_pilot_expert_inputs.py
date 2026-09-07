import pandas as pd

from scripts.prepare_pilot_expert_inputs import eligible_at_publication, training_disclosure_mask
from scripts.train_pilot_specialists import financial_asof


def test_prelisting_is_not_a_historical_candidate():
    assert not eligible_at_publication("2023-01-01", "2023-02-01")
    assert not eligible_at_publication("2023-01-01", None)
    assert eligible_at_publication("2023-02-01", "2023-02-01")


def test_financial_view_excludes_future_disclosures():
    frame = pd.DataFrame(
        dict(
            actual_disclosure_date=["2024-12-31", "2025-01-01", "2026-01-01"],
            available_at_utc=[
                "2024-12-31T15:59:59Z",
                "2025-01-01T15:59:59Z",
                "2026-01-01T15:59:59Z",
            ],
        )
    )
    assert training_disclosure_mask(frame).tolist() == [True, False, False]


def test_later_revision_cannot_change_prior_financial_features():
    frame = pd.DataFrame(
        dict(
            available_at_utc=pd.to_datetime(["2023-04-01", "2023-06-01"], utc=True),
            report_type=["1", "1"],
            end_date=["2022-12-31", "2022-12-31"],
            update_flag=["0", "1"],
            revenue=[10.0, 100.0],
        )
    )
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    assert financial_asof(frame, asof).revenue == 10
    frame.loc[1, "revenue"] = 1000000
    assert financial_asof(frame, asof).revenue == 10
    assert financial_asof(frame, pd.Timestamp("2023-03-01", tz="UTC")) is None
