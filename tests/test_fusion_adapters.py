from datetime import UTC, date, datetime

import pandas as pd

from src.prediction.fusion.adapters import (
    adapt_tabular_v2_oof,
    adapt_timeseries_lstm_oof,
)


def _identity() -> dict[str, list]:
    return {
        "sample_id": ["sample-1"],
        "company_day_id": ["company-day-1"],
        "stock_code": ["000001.SZ"],
        "published_date": [date(2024, 1, 2)],
        "prediction_as_of": [datetime(2024, 1, 3, 0, 30, tzinfo=UTC)],
        "horizon_sessions": [1],
    }


def test_tabular_adapter_selects_one_candidate_and_adds_entry_date():
    frame = pd.DataFrame(
        {
            **_identity(),
            "candidate": ["catboost"],
            "feature_set": ["full_v2"],
            "predicted_excess_return": [0.02],
            "calibrated_bullish_probability": [0.6],
            "predicted_absolute_excess_return": [0.03],
            "component_available": [True],
            "refusal_reason": [""],
            "model_version": ["tabular-v2"],
            "data_as_of": [datetime(2024, 1, 3, 0, 30, tzinfo=UTC)],
            "evidence_sha256": ["a" * 64],
        }
    )
    output = adapt_tabular_v2_oof(
        frame, candidate="catboost", feature_set="full_v2"
    )
    assert output.loc[0, "eligible_entry_date"] == date(2024, 1, 3)
    assert output.loc[0, "expert_signal"] == 0.02
    assert output.loc[0, "risk_scale"] == 0.03


def test_timeseries_adapter_preserves_abstention_and_diagnostic_heads():
    frame = pd.DataFrame(
        {
            **_identity(),
            "model_version": ["lstm-v1"],
            "bullish_probability": [0.55],
            "expected_excess_return": [0.01],
            "risk_scale": [0.02],
            "sequence_available": [True],
            "unavailable_reason": [None],
            "data_as_of": [datetime(2024, 1, 2, 8, tzinfo=UTC)],
            "evidence_sha256": ["b" * 64],
        }
    )
    output = adapt_timeseries_lstm_oof(frame, model_version="lstm-v1")
    assert bool(output.loc[0, "available"]) is True
    assert output.loc[0, "bullish_probability"] == 0.55
    assert output.loc[0, "expert_version"] == "lstm-v1"
