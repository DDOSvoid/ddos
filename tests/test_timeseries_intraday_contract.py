"""Strict contract tests for lagged 15-minute complete-day features."""

from datetime import date

from src.config import PROJECT_ROOT
from src.prediction.timeseries.intraday_contract import (
    load_intraday_aggregate_timeseries_contract,
)


def test_intraday_aggregate_contract_keeps_outputs_in_timeseries_boundary():
    contract = load_intraday_aggregate_timeseries_contract()
    assert contract.source_directory == (
        PROJECT_ROOT / "src" / "prediction" / "timeseries"
    ).resolve()
    assert contract.data_directory == (PROJECT_ROOT / "data" / "timeseries").resolve()
    assert contract.model_directory == (
        PROJECT_ROOT / "models" / "timeseries"
    ).resolve()
    assert contract.intraday_source_directory == (
        PROJECT_ROOT / "data" / "tabular" / "intraday_15m"
    ).resolve()


def test_intraday_aggregate_contract_is_train_only_and_fixed_diagnostic():
    contract = load_intraday_aggregate_timeseries_contract()
    assert contract.training_start == date(2023, 1, 1)
    assert contract.training_end == date(2024, 12, 31)
    assert contract.official_test_start == date(2025, 1, 1)
    assert contract.official_test_end == date(2025, 8, 19)
    assert contract.logistic_c_values == (0.1,)
    assert contract.logistic_class_weights == ("none",)
    assert set(contract.feature_set_candidates) == {
        "daily_only_matched",
        "intraday_only",
        "daily_plus_intraday",
    }


def test_intraday_contract_forbids_future_and_outcome_features():
    contract = load_intraday_aggregate_timeseries_contract()
    assert not contract.feature_name_allowed("future_high")
    assert not contract.feature_name_allowed("excess_return")
    assert not contract.feature_name_allowed("post_announcement_close")
    assert contract.feature_name_allowed("intraday_amount_hhi_mean_20d")
