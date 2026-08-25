"""Fold-local probability and interval calibration for LSTM research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def _validate_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _binary_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    fitted_temperature: float
    identity_fallback_used: bool
    raw_nll: float
    calibrated_nll: float
    raw_brier: float
    calibrated_brier: float

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        values = _validate_vector(logits, name="temperature logits")
        return values / self.temperature

    def predict_probability(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(self.transform_logits(logits))

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "fitted_temperature": self.fitted_temperature,
            "identity_fallback_used": self.identity_fallback_used,
            "raw_nll": self.raw_nll,
            "calibrated_nll": self.calibrated_nll,
            "raw_brier": self.raw_brier,
            "calibrated_brier": self.calibrated_brier,
        }


def fit_temperature_scaling(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    bounds: tuple[float, float],
    identity_fallback_if_brier_worsens: bool,
) -> TemperatureCalibration:
    values = _validate_vector(logits, name="calibration logits")
    labels = _validate_vector(targets, name="calibration targets")
    if values.shape != labels.shape or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("temperature calibration requires aligned binary targets")
    lower, upper = bounds
    if not 0 < lower < upper:
        raise ValueError("invalid temperature bounds")
    result = minimize_scalar(
        lambda log_temperature: _binary_nll(
            values / np.exp(log_temperature), labels
        ),
        bounds=(np.log(lower), np.log(upper)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    fitted = float(np.clip(np.exp(result.x), lower, upper))
    raw_probability = _sigmoid(values)
    fitted_probability = _sigmoid(values / fitted)
    raw_brier = float(np.mean(np.square(raw_probability - labels)))
    fitted_brier = float(np.mean(np.square(fitted_probability - labels)))
    fallback = bool(
        identity_fallback_if_brier_worsens and fitted_brier > raw_brier + 1e-15
    )
    temperature = 1.0 if fallback else fitted
    probability = raw_probability if fallback else fitted_probability
    return TemperatureCalibration(
        temperature=temperature,
        fitted_temperature=fitted,
        identity_fallback_used=fallback,
        raw_nll=_binary_nll(values, labels),
        calibrated_nll=_binary_nll(values / temperature, labels),
        raw_brier=raw_brier,
        calibrated_brier=float(np.mean(np.square(probability - labels))),
    )


@dataclass(frozen=True)
class IntervalCalibration:
    correction: float
    target_coverage: float
    calibration_samples: int
    finite_sample_rank: int
    raw_coverage: float
    calibrated_coverage: float

    def transform(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_values = _validate_vector(lower, name="interval lower")
        upper_values = _validate_vector(upper, name="interval upper")
        if lower_values.shape != upper_values.shape:
            raise ValueError("interval bounds must align")
        if (lower_values > upper_values).any():
            raise ValueError("interval lower exceeds upper")
        return lower_values - self.correction, upper_values + self.correction

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "split_conformal_interval_expansion",
            "correction": self.correction,
            "target_coverage": self.target_coverage,
            "calibration_samples": self.calibration_samples,
            "finite_sample_rank": self.finite_sample_rank,
            "raw_coverage": self.raw_coverage,
            "calibrated_coverage": self.calibrated_coverage,
        }


def fit_interval_expansion(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    *,
    target_coverage: float,
) -> IntervalCalibration:
    lower_values = _validate_vector(lower, name="calibration interval lower")
    upper_values = _validate_vector(upper, name="calibration interval upper")
    actual = _validate_vector(targets, name="calibration interval targets")
    if not (lower_values.shape == upper_values.shape == actual.shape):
        raise ValueError("interval calibration arrays must align")
    if (lower_values > upper_values).any():
        raise ValueError("raw interval lower exceeds upper")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("interval target coverage must be in (0, 1)")
    scores = np.maximum(lower_values - actual, actual - upper_values)
    sample_count = len(scores)
    rank = min(
        int(np.ceil((sample_count + 1) * target_coverage)),
        sample_count,
    )
    correction = max(0.0, float(np.sort(scores)[rank - 1]))
    raw_coverage = float(
        np.mean((actual >= lower_values) & (actual <= upper_values))
    )
    calibrated_lower = lower_values - correction
    calibrated_upper = upper_values + correction
    calibrated_coverage = float(
        np.mean((actual >= calibrated_lower) & (actual <= calibrated_upper))
    )
    return IntervalCalibration(
        correction=correction,
        target_coverage=target_coverage,
        calibration_samples=sample_count,
        finite_sample_rank=rank,
        raw_coverage=raw_coverage,
        calibrated_coverage=calibrated_coverage,
    )
