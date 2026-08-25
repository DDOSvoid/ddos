"""Calibrated tabular-v2 experiment contract and gate tests."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.tabular.experiment_v2 import (
    apply_platt_calibrator,
    fit_platt_calibrator,
    load_tabular_experiment_v2_config,
    prediction_metrics_v2,
    review_internal_experiment_v2,
)


def test_v2_experiment_contract_uses_all_horizon_primary_risk():
    config = load_tabular_experiment_v2_config()
    assert config.dataset_role == "train"
    assert config.risk_target == "future_absolute_excess_return"
    assert set(config.risk_horizons) == {1, 3, 5}
    assert {item.name for item in config.feature_sets} == {
        "full_v2",
        "without_valuation_liquidity",
    }


def test_platt_calibrator_returns_probabilities_without_using_validation_labels():
    raw = np.array([0.1, 0.2, 0.8, 0.9])
    actual = np.array([0, 0, 1, 1])
    calibrator = fit_platt_calibrator(raw, actual, epsilon=1e-6, seed=7)
    calibrated = apply_platt_calibrator(calibrator, raw, epsilon=1e-6)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert calibrated[0] < calibrated[-1]


def test_v2_metrics_cover_calibrated_direction_return_and_primary_risk():
    frame = pd.DataFrame(
        {
            "future_excess_return": [0.02, -0.01, 0.03, -0.02],
            "future_absolute_excess_return": [0.02, 0.01, 0.03, 0.02],
            "predicted_excess_return": [0.01, -0.01, 0.02, -0.01],
            "predicted_absolute_excess_return": [0.02, 0.02, 0.03, 0.01],
            "baseline_excess_return": [0.0] * 4,
            "baseline_absolute_excess_return": [0.02] * 4,
            "raw_bullish_probability": [0.8, 0.2, 0.7, 0.6],
            "calibrated_bullish_probability": [0.7, 0.3, 0.65, 0.55],
            "baseline_bullish_probability": [0.5] * 4,
        }
    )
    metrics = prediction_metrics_v2(
        frame, threshold=0.5, round_trip_cost_bps=20
    )
    assert metrics["bullish_signals"] == 3
    assert metrics["return_mae"] == pytest.approx(0.0075)
    assert metrics["risk_mae"] == pytest.approx(0.005)
    assert "raw_brier_score" in metrics


def test_v2_review_requires_every_direction_return_risk_and_calibration_gate():
    aggregate = {
        "feature_set": "full_v2",
        "candidate": "catboost",
        "horizon_sessions": 1,
        "bullish_signals": 500,
        "bullish_wilson_lower": 0.49,
        "cost_adjusted_mean_excess_return_ci_lower": -0.001,
        "brier_score": 0.24,
        "raw_brier_score": 0.25,
        "baseline_brier_score": 0.25,
        "return_mae": 0.03,
        "baseline_return_mae": 0.02,
        "risk_mae": 0.01,
        "baseline_risk_mae": 0.02,
    }
    folds = [
        {
            "feature_set": "full_v2",
            "candidate": "catboost",
            "horizon_sessions": 1,
            "bullish_hit_rate": value,
            "risk_mae": 0.01,
            "baseline_risk_mae": 0.02,
        }
        for value in (0.49, 0.51, 0.50)
    ]
    review = review_internal_experiment_v2(
        {
            "official_test_queried": False,
            "aggregate_metrics": [aggregate],
            "fold_metrics": folds,
        }
    )
    assert review["decision"] == "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE"
    assert review["official_test_action"] == "DO_NOT_READ"
