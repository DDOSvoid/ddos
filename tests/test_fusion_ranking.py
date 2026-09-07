from datetime import date

import pandas as pd

from src.prediction.fusion.ranking import (
    evaluate_daily_ranking,
    rank_company_day_predictions,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "eligible_entry_date": [date(2024, 1, 3)] * 3,
            "horizon_sessions": [1] * 3,
            "expected_excess_return": [0.03, 0.02, 0.01],
            "risk_scale": [0.01, 0.04, 0.01],
            "fusion_available": [True, True, False],
            "future_excess_return": [0.04, -0.01, 0.02],
        }
    )


def test_rank_is_risk_and_cost_aware_and_excludes_unavailable_rows():
    ranked = rank_company_day_predictions(
        _predictions(), risk_aversion=0.5, round_trip_cost_bps=20
    )
    first = ranked.loc[ranked["stock_code"] == "000001.SZ"].iloc[0]
    unavailable = ranked.loc[ranked["stock_code"] == "000003.SZ"].iloc[0]
    assert first["daily_rank"] == 1
    assert abs(first["rank_score"] - 0.023) < 1e-12
    assert pd.isna(unavailable["daily_rank"])
    assert pd.isna(unavailable["rank_score"])


def test_ranking_evaluation_reports_cross_sectional_top_k_metrics():
    ranked = rank_company_day_predictions(
        _predictions(), risk_aversion=0.5, round_trip_cost_bps=20
    )
    report = evaluate_daily_ranking(ranked, top_k=(1,), round_trip_cost_bps=20)
    horizon = report["horizons"][0]
    assert horizon["eligible_rows"] == 2
    assert horizon["top_k"]["1"]["mean_excess_return"] == 0.04
    assert horizon["top_k"]["1"]["cost_adjusted_mean_excess_return"] == 0.038
