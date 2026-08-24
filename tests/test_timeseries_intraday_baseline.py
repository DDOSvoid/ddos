"""Decision-gate tests for the matched intraday aggregate baseline."""

import pytest

from src.prediction.timeseries.train_intraday_baseline import (
    _brier_increment_comparison,
)


def _horizon_report(pooled_brier: float, fold_briers: tuple[float, ...]) -> dict:
    return {
        "pooled_metrics": {"brier_score": pooled_brier},
        "folds": [
            {"fold": f"fold_{index}", "brier_score": value}
            for index, value in enumerate(fold_briers, 1)
        ],
    }


def test_brier_increment_requires_pooled_and_two_fold_improvements():
    daily = _horizon_report(0.25, (0.24, 0.25, 0.26))
    combined = _horizon_report(0.24, (0.23, 0.24, 0.27))
    daily_brier, combined_brier, fold_improvements, stable = (
        _brier_increment_comparison(daily, combined)
    )
    assert daily_brier == 0.25
    assert combined_brier == 0.24
    assert fold_improvements == pytest.approx(
        {"fold_1": 0.01, "fold_2": 0.01, "fold_3": -0.01}
    )
    assert stable


def test_brier_increment_rejects_pooled_gain_without_fold_stability():
    daily = _horizon_report(0.25, (0.24, 0.25, 0.26))
    combined = _horizon_report(0.24, (0.23, 0.26, 0.27))
    assert not _brier_increment_comparison(daily, combined)[-1]


def test_brier_increment_rejects_different_fold_identities():
    daily = _horizon_report(0.25, (0.24, 0.25, 0.26))
    combined = _horizon_report(0.24, (0.23, 0.24))
    with pytest.raises(ValueError, match="fold identities differ"):
        _brier_increment_comparison(daily, combined)
