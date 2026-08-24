"""Fixed tabular experiment contract and metric tests."""

import pandas as pd
import pytest

from src.prediction.tabular.experiment import (
    load_tabular_experiment_config,
    prediction_metrics,
    review_internal_experiment,
    wilson_lower,
)


def test_experiment_contract_is_train_only_and_has_fixed_candidates():
    config = load_tabular_experiment_config()
    assert config.dataset_role == "train"
    assert set(config.candidates) == {"linear", "lightgbm", "catboost"}
    assert set(config.return_horizons) == {1, 3, 5}
    assert set(config.risk_horizons) == {3, 5}


def test_prediction_metrics_include_direction_return_risk_and_cost_gate():
    frame = pd.DataFrame(
        {
            "future_excess_return": [0.02, -0.01, 0.03, -0.02],
            "predicted_excess_return": [0.01, -0.01, 0.02, -0.01],
            "baseline_excess_return": [0.0, 0.0, 0.0, 0.0],
            "bullish_probability": [0.8, 0.2, 0.7, 0.6],
            "baseline_bullish_probability": [0.5, 0.5, 0.5, 0.5],
            "future_realized_volatility": [0.03, 0.02, 0.04, 0.03],
            "predicted_realized_volatility": [0.02, 0.02, 0.03, 0.04],
            "baseline_realized_volatility": [0.03, 0.03, 0.03, 0.03],
        }
    )
    metrics = prediction_metrics(frame, threshold=0.5, round_trip_cost_bps=20)
    assert metrics["bullish_signals"] == 3
    assert metrics["bullish_hits"] == 2
    assert metrics["return_mae"] == pytest.approx(0.0075)
    assert metrics["risk_mae"] == pytest.approx(0.0075)
    assert metrics["baseline_brier_score"] == pytest.approx(0.25)
    assert 0 < metrics["bullish_wilson_lower"] < metrics["bullish_hit_rate"]


def test_wilson_lower_returns_nan_without_signals():
    assert pd.isna(wilson_lower(0, 0))


def test_gate_review_keeps_failed_return_candidate_research_only():
    aggregate = {
        "candidate": "catboost",
        "horizon_sessions": 3,
        "bullish_signals": 500,
        "bullish_wilson_lower": 0.49,
        "cost_adjusted_mean_excess_return_ci_lower": -0.001,
        "brier_score": 0.26,
        "baseline_brier_score": 0.25,
        "return_mae": 0.03,
        "baseline_return_mae": 0.02,
        "risk_mae": 0.02,
        "baseline_risk_mae": 0.03,
        "risk_rmse": 0.03,
        "baseline_risk_rmse": 0.04,
    }
    folds = [
        {
            "candidate": "catboost",
            "horizon_sessions": 3,
            "bullish_hit_rate": value,
            "risk_mae": 0.02,
            "baseline_risk_mae": 0.03,
        }
        for value in (0.49, 0.51, 0.50)
    ]
    review = review_internal_experiment(
        {
            "official_test_queried": False,
            "fold_metrics": folds,
            "aggregate_metrics": [aggregate],
        }
    )
    assert review["decision"] == "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE"
    assert review["official_test_action"] == "DO_NOT_READ"
    assert review["risk_reviews"][0]["research_pass"] is True
