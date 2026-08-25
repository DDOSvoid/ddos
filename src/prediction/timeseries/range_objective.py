"""Causal range-based target and acceptance helpers for LSTM v2 research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReturnTargetRange:
    lower: float
    upper: float
    lower_quantile: float
    upper_quantile: float
    fit_samples: int
    fit_scope: str = "outer_fold_train_only"

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("range membership values must be finite")
        return (array >= self.lower) & (array <= self.upper)

    def as_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "fit_samples": self.fit_samples,
            "fit_scope": self.fit_scope,
        }


def fit_return_target_range(
    returns: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
    minimum_samples: int = 1,
) -> ReturnTargetRange:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < minimum_samples:
        raise ValueError("return range needs enough one-dimensional training values")
    if not np.isfinite(values).all():
        raise ValueError("return range training values must be finite")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("return range quantiles must be ordered in (0, 1)")
    lower = float(np.quantile(values, lower_quantile, method="linear"))
    upper = float(np.quantile(values, upper_quantile, method="linear"))
    if lower > upper:
        raise ValueError("return range lower exceeds upper")
    return ReturnTargetRange(
        lower=lower,
        upper=upper,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        fit_samples=len(values),
    )


def value_inclusive_range(
    value: float,
    bounds: tuple[float, float],
    *,
    name: str = "value",
) -> bool:
    lower, upper = (float(item) for item in bounds)
    if not np.isfinite((lower, upper, value)).all() or lower > upper:
        raise ValueError(f"invalid inclusive range for {name}")
    return lower <= float(value) <= upper
