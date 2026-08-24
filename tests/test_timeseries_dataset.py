"""Physical time-series feature and label separation tests."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.dataset import assemble_train_frame


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1"],
            "stock_code": ["000001.SZ"],
            "published_date": [date(2024, 4, 25)],
            "prediction_as_of": [datetime(2024, 4, 25, 0, 30, tzinfo=UTC)],
            "last_bar_trade_date": [date(2024, 4, 24)],
            "last_bar_available_at": [datetime(2024, 4, 24, 7, 5, tzinfo=UTC)],
            "dataset_role": ["train"],
            "stock_momentum_20d": [0.1],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1"],
            "horizon_sessions": [1],
            "target": [1],
            "excess_return": [0.02],
            "outcome_available_at": [datetime(2024, 4, 26, 7, 0, tzinfo=UTC)],
            "dataset_role": ["train"],
        }
    )


def test_separate_feature_and_label_artifacts_join_after_validation():
    joined = assemble_train_frame(
        _features(),
        _labels(),
        feature_columns=["stock_momentum_20d"],
        contract=load_timeseries_model_contract(),
    )
    assert joined.loc[0, "target"] == 1
    assert joined.loc[0, "stock_momentum_20d"] == pytest.approx(0.1)


def test_future_outcome_cannot_be_declared_as_a_feature():
    features = _features()
    features["future_high"] = 12.0
    with pytest.raises(ValueError, match="future time-series features rejected"):
        assemble_train_frame(
            features,
            _labels(),
            feature_columns=["future_high"],
            contract=load_timeseries_model_contract(),
        )


def test_sealed_roles_cannot_be_assembled_for_model_selection():
    features = _features()
    labels = _labels()
    features["dataset_role"] = "test"
    labels["dataset_role"] = "test"
    with pytest.raises(PermissionError, match="accepts train only"):
        assemble_train_frame(
            features,
            labels,
            feature_columns=["stock_momentum_20d"],
            contract=load_timeseries_model_contract(),
        )
