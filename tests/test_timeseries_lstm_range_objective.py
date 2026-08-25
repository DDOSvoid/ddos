from __future__ import annotations

import numpy as np
import pytest

from src.prediction.timeseries.range_objective import (
    fit_return_target_range,
    value_inclusive_range,
)


def test_return_target_range_is_fit_from_supplied_fold_train_values():
    result = fit_return_target_range(
        np.arange(100, dtype=float),
        lower_quantile=0.10,
        upper_quantile=0.90,
        minimum_samples=100,
    )

    assert result.lower == pytest.approx(9.9)
    assert result.upper == pytest.approx(89.1)
    assert result.fit_scope == "outer_fold_train_only"
    assert result.contains(np.array([9.9, 50.0, 89.1])).tolist() == [True, True, True]
    assert result.contains(np.array([0.0, 99.0])).tolist() == [False, False]


def test_inclusive_metric_range_replaces_single_point_gate():
    assert value_inclusive_range(0.85, (0.80, 0.95), name="coverage")
    assert not value_inclusive_range(0.70, (0.80, 0.95), name="coverage")


def test_return_target_range_rejects_non_finite_or_bad_quantiles():
    with pytest.raises(ValueError):
        fit_return_target_range(
            np.array([0.0, np.nan]),
            lower_quantile=0.10,
            upper_quantile=0.90,
        )
    with pytest.raises(ValueError):
        fit_return_target_range(
            np.arange(10, dtype=float),
            lower_quantile=0.90,
            upper_quantile=0.10,
        )
