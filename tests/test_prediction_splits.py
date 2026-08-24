"""Strict prediction split-contract tests."""

from datetime import UTC, date, datetime

from src.prediction.splits import load_prediction_split_contract


def test_split_roles_are_disjoint_and_2026_is_validation_only():
    contract = load_prediction_split_contract()
    assert contract.role_for(date(2024, 12, 31)) == "train"
    assert contract.role_for(date(2025, 1, 1)) == "test"
    assert contract.role_for(date(2025, 8, 20)) == "quarantine"
    assert contract.role_for(date(2026, 1, 1)) == "forward_validation"
    assert contract.test.end.year == 2025


def test_outcome_after_partition_cutoff_is_excluded():
    contract = load_prediction_split_contract()
    assert contract.role_for(
        date(2024, 12, 31),
        datetime(2025, 1, 2, tzinfo=UTC),
    ) == "train_outcome_after_cutoff"
    assert contract.role_for(
        date(2025, 8, 19),
        datetime(2025, 8, 21, tzinfo=UTC),
    ) == "test_outcome_after_cutoff"


def test_2026_outcome_never_changes_validation_role():
    contract = load_prediction_split_contract()
    assert contract.role_for(
        date(2026, 3, 1),
        datetime(2026, 3, 10, tzinfo=UTC),
    ) == "forward_validation"
