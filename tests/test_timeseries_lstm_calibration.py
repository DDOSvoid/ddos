from __future__ import annotations

import numpy as np

from src.prediction.timeseries.calibrate_lstm import (
    fit_interval_expansion,
    fit_temperature_scaling,
)


def test_temperature_scaling_is_fit_only_from_supplied_calibration_rows():
    logits = np.array([4.0, 4.0, -4.0, -4.0])
    targets = np.array([1.0, 0.0, 0.0, 1.0])
    calibration = fit_temperature_scaling(
        logits,
        targets,
        bounds=(0.05, 10.0),
        identity_fallback_if_brier_worsens=True,
    )

    assert calibration.temperature > 1.0
    assert calibration.calibrated_nll < calibration.raw_nll
    assert calibration.calibrated_brier <= calibration.raw_brier
    probabilities = calibration.predict_probability(np.array([1.0, -1.0]))
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_split_conformal_interval_expands_to_finite_sample_target():
    targets = np.linspace(-0.05, 0.05, 20)
    lower = np.full(20, -0.01)
    upper = np.full(20, 0.01)
    calibration = fit_interval_expansion(
        lower,
        upper,
        targets,
        target_coverage=0.8,
    )
    calibrated_lower, calibrated_upper = calibration.transform(lower, upper)
    coverage = np.mean(
        (targets >= calibrated_lower) & (targets <= calibrated_upper)
    )

    assert calibration.correction > 0.0
    assert coverage >= 0.8
    assert calibration.calibrated_coverage >= calibration.raw_coverage
