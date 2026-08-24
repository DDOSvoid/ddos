"""Point-in-time tabular feature validation tests."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.features import (
    FeatureSpec,
    require_train_only,
    validate_and_select_features,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "published_date": [date(2024, 4, 25)],
            "prediction_as_of": [datetime(2024, 4, 25, 0, 30, tzinfo=UTC)],
            "momentum_20d": [0.12],
            "momentum_20d_available_at": [
                datetime(2024, 4, 24, 7, 0, tzinfo=UTC)
            ],
            "market_observation_date": [date(2024, 4, 24)],
            "dataset_role": ["train"],
        }
    )


def _market_spec(name: str = "momentum_20d") -> FeatureSpec:
    return FeatureSpec(
        name=name,
        source="market_daily",
        available_at_column=f"{name}_available_at",
        observation_date_column="market_observation_date",
    )


def test_valid_lagged_market_feature_is_selected():
    selected = validate_and_select_features(
        _valid_frame(),
        specs=[_market_spec()],
        tabular=load_tabular_model_contract(),
        development=load_causal_development_contract(),
    )
    assert selected.columns.tolist() == ["momentum_20d"]
    assert selected.iloc[0, 0] == pytest.approx(0.12)


def test_valid_lagged_intraday_aggregate_uses_same_strict_date_rule():
    spec = _market_spec()
    spec = FeatureSpec(
        name=spec.name,
        source="market_intraday_aggregate",
        available_at_column=spec.available_at_column,
        observation_date_column=spec.observation_date_column,
    )
    selected = validate_and_select_features(
        _valid_frame(),
        specs=[spec],
        tabular=load_tabular_model_contract(),
        development=load_causal_development_contract(),
    )
    assert selected.columns.tolist() == ["momentum_20d"]


def test_missing_optional_feature_does_not_require_fake_availability_timestamp():
    frame = _valid_frame()
    frame.loc[0, "momentum_20d"] = None
    frame.loc[0, "momentum_20d_available_at"] = None
    frame.loc[0, "market_observation_date"] = None
    selected = validate_and_select_features(
        frame,
        specs=[_market_spec()],
        tabular=load_tabular_model_contract(),
        development=load_causal_development_contract(),
    )
    assert pd.isna(selected.iloc[0, 0])


def test_present_feature_still_requires_availability_timestamp():
    frame = _valid_frame()
    frame.loc[0, "momentum_20d_available_at"] = None
    with pytest.raises(ValueError, match="missing timestamp"):
        validate_and_select_features(
            frame,
            specs=[_market_spec()],
            tabular=load_tabular_model_contract(),
            development=load_causal_development_contract(),
        )


def test_future_return_is_rejected_even_if_named_as_a_feature():
    frame = _valid_frame().rename(
        columns={
            "momentum_20d": "future_excess_return",
            "momentum_20d_available_at": "future_excess_return_available_at",
        }
    )
    with pytest.raises(ValueError, match="target or post-outcome"):
        validate_and_select_features(
            frame,
            specs=[_market_spec("future_excess_return")],
            tabular=load_tabular_model_contract(),
            development=load_causal_development_contract(),
        )


def test_same_day_market_bar_and_late_fundamental_are_rejected():
    frame = _valid_frame()
    frame.loc[0, "market_observation_date"] = date(2024, 4, 25)
    with pytest.raises(ValueError, match="same-day or future market bar"):
        validate_and_select_features(
            frame,
            specs=[_market_spec()],
            tabular=load_tabular_model_contract(),
            development=load_causal_development_contract(),
        )

    frame = _valid_frame()
    frame.loc[0, "momentum_20d_available_at"] = datetime(
        2024, 4, 25, 7, 0, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="future feature rejected"):
        validate_and_select_features(
            frame,
            specs=[_market_spec()],
            tabular=load_tabular_model_contract(),
            development=load_causal_development_contract(),
        )


@pytest.mark.parametrize("role", ["test", "quarantine", "forward_validation"])
def test_non_train_roles_cannot_fit_or_select_models(role):
    frame = pd.DataFrame({"dataset_role": ["train", role]})
    with pytest.raises(PermissionError, match="accepts train only"):
        require_train_only(frame)
