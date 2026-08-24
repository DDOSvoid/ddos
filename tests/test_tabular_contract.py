"""Strict tabular model contract tests."""

from datetime import date

from src.config import PROJECT_ROOT
from src.prediction.tabular.contract import load_tabular_model_contract


def test_tabular_contract_has_independent_boundaries_and_models():
    contract = load_tabular_model_contract()
    assert contract.source_directory == (
        PROJECT_ROOT / "src" / "prediction" / "tabular"
    ).resolve()
    assert contract.data_directory == (PROJECT_ROOT / "data" / "tabular").resolve()
    assert contract.model_directory == (PROJECT_ROOT / "models" / "tabular").resolve()
    assert set(contract.model_families) == {"lightgbm", "catboost"}
    assert contract.feature_frequency == "daily"
    assert contract.raw_intraday_frequency == "15min"
    assert contract.raw_intraday_usage == "lagged_daily_aggregates_only"


def test_tabular_contract_matches_all_temporal_boundaries():
    contract = load_tabular_model_contract()
    assert contract.training_start == date(2023, 1, 1)
    assert contract.training_end == date(2024, 12, 31)
    assert contract.official_test_start == date(2025, 1, 1)
    assert contract.official_test_end == date(2025, 8, 19)
    assert contract.quarantine_start == date(2025, 8, 20)
    assert contract.quarantine_end == date(2025, 12, 31)
    assert contract.forward_validation_start == date(2026, 1, 1)
    assert contract.official_test_read_limit == 1


def test_target_and_post_announcement_names_are_never_features():
    contract = load_tabular_model_contract()
    assert not contract.feature_name_allowed("excess_return")
    assert not contract.feature_name_allowed("future_realized_volatility")
    assert not contract.feature_name_allowed("post_announcement_close")
    assert contract.feature_name_allowed("momentum_20d")
    assert contract.feature_name_allowed("forward_pe_consensus")
