"""Strict time-series model contract tests."""

from datetime import date

from src.config import PROJECT_ROOT
from src.prediction.timeseries.contract import load_timeseries_model_contract


def test_timeseries_contract_has_independent_daily_v1_boundaries():
    contract = load_timeseries_model_contract()
    assert contract.source_directory == (
        PROJECT_ROOT / "src" / "prediction" / "timeseries"
    ).resolve()
    assert contract.data_directory == (PROJECT_ROOT / "data" / "timeseries").resolve()
    assert contract.model_directory == (
        PROJECT_ROOT / "models" / "timeseries"
    ).resolve()
    assert contract.feature_frequency == "daily"
    assert contract.lookbacks_sessions == (5, 20, 60)
    assert contract.logistic_c_values == (0.1, 1.0, 10.0)
    assert set(contract.logistic_class_weights) == {"none", "balanced"}
    assert contract.intraday_candidate_frequency == "15min"
    assert contract.intraday_prediction_status.startswith("blocked_")


def test_timeseries_contract_matches_all_temporal_boundaries():
    contract = load_timeseries_model_contract()
    assert contract.training_start == date(2023, 1, 1)
    assert contract.training_end == date(2024, 12, 31)
    assert contract.official_test_start == date(2025, 1, 1)
    assert contract.official_test_end == date(2025, 8, 19)
    assert contract.quarantine_start == date(2025, 8, 20)
    assert contract.quarantine_end == date(2025, 12, 31)
    assert contract.forward_validation_start == date(2026, 1, 1)
    assert contract.official_test_read_limit == 1


def test_future_outcomes_and_post_announcement_names_are_forbidden():
    contract = load_timeseries_model_contract()
    for name in (
        "target",
        "actual_direction",
        "excess_return",
        "future_high",
        "future_low",
        "future_realized_volatility",
        "post_announcement_close",
    ):
        assert not contract.feature_name_allowed(name)
    assert contract.feature_name_allowed("stock_momentum_20d")
    assert contract.feature_name_allowed("stock_volatility_20d")
